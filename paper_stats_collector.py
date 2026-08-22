"""Train the four maps and collect the paper's map/response panel data.

Completed maps are checkpointed for safe resumption. Displacement and sparse-
connectivity readouts are derived live from those maps rather than collected.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.ndimage import rotate as scipy_rotate
from sklearn.decomposition import PCA
from tqdm import tqdm

from helpers.map_plotting import _make_sine_grating
from helpers.wiring_efficiency_utils import (
    create_dataloader,
    gaussian_local_permutation,
    get_typical_dist_fourier,
    radial_mean_angle_distance,
)
from neuralsheet import NeuralSheet
from stats_collector import (
    DEFAULT_BASE_SEED,
    STRICT_DETERMINISM,
    _configure_reproducibility,
    _derive_seed,
    _fixed_train_and_evaluation_data,
    _seed_everything,
)


COLLECTOR_VERSION = 1
BASE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE / "paper_results" / "paper_panel_data.pt"
DEFAULT_CHECKPOINT_DIR = BASE / "paper_results" / "map_checkpoints"
DEFAULT_STRINGER_DIR = BASE / "data" / "stringer"
STRINGER_RECORDING = "static_sin_rand_TX36_2019_10_21_1.npy"
STRINGER_FIGSHARE_ARTICLE = 8279387
STRINGER_FIGSHARE_VERSION = 3


@dataclass(frozen=True)
class MapSpec:
    """One L4 map needed by the paper panel collector."""

    name: str
    sheet_size: int
    r_long: float
    microcolumnar: bool

    @property
    def r_sre(self) -> float:
        return self.r_long / (8.1 if self.microcolumnar else 3.0)

    @property
    def n_lat_continuous(self) -> float:
        return math.pi * self.r_sre**2

    @property
    def reference_ratio(self) -> float:
        return self.sheet_size**2 / self.n_lat_continuous


def paper_map_specs() -> tuple[MapSpec, ...]:
    """Return the four maps used by the paper panel analysis.

    The 90/18 pair has the same dimensionless geometry as the 60/12 pair.
    """

    return (
        MapSpec("macro60", 60, 12.0, False),
        MapSpec("micro60", 60, 12.0, True),
        MapSpec("macro90", 90, 18.0, False),
        MapSpec("micro90", 90, 18.0, True),
    )


@dataclass
class PaperStatsConfig:
    input_stimuli: str = str(BASE / "input_stimuli")
    output_file: str = str(DEFAULT_OUTPUT)
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)
    stringer_dir: str = str(DEFAULT_STRINGER_DIR)
    local_stringer_source: str | None = None
    device: str = "cuda"
    base_seed: int = DEFAULT_BASE_SEED
    strict_determinism: bool = STRICT_DETERMINISM
    crop_size: int = 24
    r_rf: float = 7.0
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 2
    brightness_threshold: float = 0.15
    lr_initial: float = 1e-3
    lr_floor: float = 1e-4
    lr_beta: float = 1.0 - 5e-5
    hebbian_lr_scale: float = 100.0
    natural_response_samples: int = 256
    grating_samples: int = 3_831
    orientation_count: int = 18
    orientation_phases: int = 8
    orientation_spatial_frequency: float = 0.05
    orientation_scaler: float = 0.3
    displacement_sigma_panel: float = 1.3
    analysis_crop: int = 45
    connectivity_fraction: float = 0.25
    connectivity_threshold_multiple: float = 2.0
    connectivity_units: int | None = None
    stringer_neurons: int = 6_000
    stringer_orientation_bins: int = 24
    stringer_min_trials_per_bin: int = 15
    umap_pca_components: int = 50
    umap_neighbors: int = 15
    umap_min_dist: float = 0.5
    # Testing controls. None means the complete declared experiment.
    max_train_batches: int | None = None
    map_names: tuple[str, ...] | None = None


_CHECKPOINT_TENSORS = (
    "afferent_weights",
    "lateral_correlations",
    "lateral_correlations_exc",
    "mean_activations",
    "thresholds",
    "mean_fr",
    "mean_lat",
    "lat_gain",
    "mean_aff",
    "aff_gain",
    "mix",
    "avg_hist",
    "old_style_mean_fr",
    "old_style_mean_aff",
)


def _training_signature(spec: MapSpec, config: PaperStatsConfig) -> dict[str, Any]:
    """Configuration fields that can change a trained checkpoint."""

    return {
        "collector_version": COLLECTOR_VERSION,
        "spec": asdict(spec),
        "input_stimuli": str(Path(config.input_stimuli).resolve()),
        "base_seed": int(config.base_seed),
        "strict_determinism": bool(config.strict_determinism),
        "crop_size": int(config.crop_size),
        "r_rf": float(config.r_rf),
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "epochs": int(config.epochs),
        "brightness_threshold": float(config.brightness_threshold),
        "lr_initial": float(config.lr_initial),
        "lr_floor": float(config.lr_floor),
        "lr_beta": float(config.lr_beta),
        "hebbian_lr_scale": float(config.hebbian_lr_scale),
        "max_train_batches": config.max_train_batches,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def _cleanup_model() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _trim_unused_l3_storage(model: NeuralSheet) -> NeuralSheet:
    """Drop L2/3-only tensors from an explicitly L4-only experiment.

    NeuralSheet historically allocates both layers in its constructor.  None
    of the deleted tensors is read by ``forward(..., layer_3=False)`` or by
    the corresponding Hebbian step.  Shrinking ``slicing_var`` likewise
    removes only interaction channels that the L4-only path never indexes.
    """

    keep_l4_noise_placeholder = model.noise_l3
    for name in tuple(vars(model)):
        if name.endswith("_l3") or "_l4_l3" in name:
            if name != "noise_l3":
                delattr(model, name)
    model.noise_l3 = keep_l4_noise_placeholder
    channels = 3 if model.microcolumnar else 2
    old = model.slicing_var
    model.slicing_var = torch.zeros(
        (model.sheet_size**2, channels, old.shape[-2], old.shape[-1]),
        device=model.device,
        dtype=old.dtype,
    )
    del old
    return model


def _make_model(spec: MapSpec, config: PaperStatsConfig, seed: int) -> NeuralSheet:
    _seed_everything(seed)
    model = NeuralSheet(
        config.crop_size,
        spec.sheet_size,
        config.r_rf,
        R_long=spec.r_long,
        device=config.device,
        microcolumnar=spec.microcolumnar,
    )
    return _trim_unused_l3_storage(model)


def _checkpoint_path(config: PaperStatsConfig, spec: MapSpec) -> Path:
    return Path(config.checkpoint_dir) / f"{spec.name}.pt"


def _save_model_checkpoint(
    model: NeuralSheet,
    spec: MapSpec,
    config: PaperStatsConfig,
    training_seed: int,
    accepted_presentations: int,
) -> Path:
    state = {}
    for name in _CHECKPOINT_TENSORS:
        if name == "lateral_correlations_exc" and not spec.microcolumnar:
            continue
        value = getattr(model, name)
        state[name] = value.detach().cpu()
    payload = {
        "collector_version": COLLECTOR_VERSION,
        "complete": True,
        "spec": asdict(spec),
        "training_signature": _training_signature(spec, config),
        "training": {
            "epochs": config.epochs,
            "accepted_presentations": accepted_presentations,
            "training_seed": int(training_seed),
            "max_train_batches": config.max_train_batches,
            "completed_at_utc": _utc_now(),
        },
        "state": state,
    }
    path = _checkpoint_path(config, spec)
    _atomic_torch_save(payload, path)
    return path


def _checkpoint_matches(path: Path, spec: MapSpec, config: PaperStatsConfig) -> bool:
    if not path.is_file():
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(
        checkpoint.get("complete")
        and checkpoint.get("collector_version") == COLLECTOR_VERSION
        and checkpoint.get("spec") == asdict(spec)
        and checkpoint.get("training_signature") == _training_signature(spec, config)
    )


def _load_model(spec: MapSpec, config: PaperStatsConfig) -> NeuralSheet:
    path = _checkpoint_path(config, spec)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_seed = _derive_seed(config.base_seed, "remaining", spec.name, "model")
    model = _make_model(spec, config, model_seed)
    for name, value in checkpoint["state"].items():
        setattr(model, name, value.to(config.device))
    return model


def _train_map(spec: MapSpec, config: PaperStatsConfig, dataset: Any) -> Path:
    checkpoint = _checkpoint_path(config, spec)
    if _checkpoint_matches(checkpoint, spec, config):
        print(f"[map] reusing complete checkpoint {checkpoint}")
        return checkpoint

    model_seed = _derive_seed(config.base_seed, "remaining", spec.name, "model")
    training_seed = _derive_seed(config.base_seed, "remaining", spec.name, "training")
    loader_seed = _derive_seed(config.base_seed, "remaining", spec.name, "loader")
    model = _make_model(spec, config, model_seed)
    loader = create_dataloader(
        config.input_stimuli,
        config.crop_size,
        config.batch_size,
        config.num_workers,
        seed=loader_seed,
        dataset=dataset,
    )
    _seed_everything(training_seed)
    lr = config.lr_initial
    accepted = 0
    print(
        f"[map] training {spec.name}: {spec.sheet_size}x{spec.sheet_size}, "
        f"R_long={spec.r_long:.6g}, micro={spec.microcolumnar}"
    )
    for epoch in range(config.epochs):
        progress = tqdm(loader, desc=f"{spec.name} epoch {epoch + 1}/{config.epochs}")
        for batch_index, batch in enumerate(progress):
            if config.max_train_batches is not None and batch_index >= config.max_train_batches:
                break
            batch = batch.to(config.device, non_blocking=True)
            valid = batch[batch[:, 0:1].mean(dim=(1, 2, 3)) > config.brightness_threshold]
            for image in valid:
                image = image[0:1][None].flip(1)
                lr = max(lr * config.lr_beta, config.lr_floor)
                model.hebbian_lr = lr * config.hebbian_lr_scale
                model.homeo_lr = lr
                model(image, adaptation=True, layer_3=False)
                model.hebbian_step(layer_3=False)
                accepted += 1
            progress.set_postfix(
                accepted=accepted,
                lr=f"{lr:.2e}",
                activity=f"{float(model.mean_activations.mean()):.3f}",
            )
    path = _save_model_checkpoint(model, spec, config, training_seed, accepted)
    print(f"[map] saved {path}")
    del model
    _cleanup_model()
    return path


def _orientation_tuning(model: NeuralSheet, config: PaperStatsConfig) -> dict[str, Any]:
    """Estimate orientation preference from settled responses to gratings."""

    device = torch.device(config.device)
    thetas = torch.linspace(0, math.pi, config.orientation_count + 1, device=device)[:-1]
    phases = torch.linspace(0, 2 * math.pi, config.orientation_phases + 1, device=device)[:-1]
    responses = []
    with torch.no_grad():
        for theta in tqdm(thetas.tolist(), desc="orientation tuning", leave=False):
            phase_sum = None
            for phase in phases.tolist():
                stimulus = _make_sine_grating(
                    W=config.crop_size,
                    theta_rad=float(theta),
                    spatial_freq_cyc_per_px=config.orientation_spatial_frequency,
                    phase_rad=float(phase),
                    device=device,
                    dtype=torch.float32,
                )
                stimulus = (stimulus + 1.0) * config.orientation_scaler
                model(stimulus[None, None], adaptation=False, layer_3=False)
                response = torch.relu(model.current_response[0, 0])
                phase_sum = response if phase_sum is None else phase_sum + response
            responses.append(phase_sum / len(phases))
    response_stack = torch.stack(responses)
    weights = torch.exp(2j * thetas.to(torch.complex64))[:, None, None]
    vector = (response_stack.to(torch.complex64) * weights).sum(0)
    orientation = torch.remainder(0.5 * torch.angle(vector), math.pi)
    osi = vector.abs() / response_stack.sum(0).clamp_min(1e-11)
    # The established ring estimator constructs its matching masks on CPU.
    orientation_cpu = orientation.cpu()
    period, spectrum, ring = get_typical_dist_fourier(orientation_cpu, mask=1)
    histogram = torch.histc(orientation_cpu, bins=18, min=0, max=math.pi)
    histogram /= histogram.sum().clamp_min(1e-11)
    return {
        "orientation_rad": orientation_cpu,
        "orientation_osi": osi.float().cpu(),
        "orientation_responses": response_stack.float().cpu(),
        "orientation_angles_rad": thetas.cpu(),
        "orientation_histogram": histogram.cpu(),
        "fourier_period_pixels": float(torch.as_tensor(period).cpu()),
        "fourier_spectrum": spectrum.float().cpu(),
        "fourier_ring_profile": ring.float().cpu(),
        "estimator": "settled L4 responses, phase mean, axial vector sum",
    }


def _retinotopic_centres(model: NeuralSheet, afferent: torch.Tensor) -> torch.Tensor:
    """Return the RF-centre fishnet in cortical lattice coordinates.

    This is the bundle-producing equivalent of ``plot_absolute_phases``:
    compute the centre of mass within each local afferent patch and add that
    displacement to the corresponding cortical lattice position.
    """

    mass = afferent.sum(dim=1)
    coordinates = torch.arange(
        model.rf_size, device=afferent.device, dtype=afferent.dtype
    ) - model.rf_size // 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    local_grid = torch.stack((yy, xx))
    local_centres = (mass[:, None] * local_grid[None]).sum((2, 3))
    local_centres /= mass.sum((1, 2))[:, None].clamp_min(1e-11)

    cortical = torch.arange(
        model.sheet_size, device=afferent.device, dtype=afferent.dtype
    )
    cy, cx = torch.meshgrid(cortical, cortical, indexing="ij")
    cortical_centres = torch.stack((cy, cx), dim=-1).reshape(-1, 2)
    return cortical_centres + local_centres


def _radial_average(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    side = field.shape[-1]
    yy, xx = np.indices((side, side))
    radius = np.rint(np.hypot(yy - side // 2, xx - side // 2)).astype(int)
    limit = side // 2
    means = np.zeros(limit, dtype=np.float64)
    for index in range(limit):
        selected = radius == index
        means[index] = field[selected].mean() if selected.any() else np.nan
    return np.arange(limit), means


def _evaluate_images(
    model: NeuralSheet,
    inputs: torch.Tensor,
    count: int,
    config: PaperStatsConfig,
) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for image in tqdm(inputs[:count], desc="natural responses", leave=False):
            model(image[None].to(config.device), adaptation=False, layer_3=False)
            chunks.append(model.current_response.detach().cpu())
    return torch.cat(chunks)


def _map_summary(
    model: NeuralSheet,
    orientation: dict[str, Any],
    evaluation_inputs: torch.Tensor,
    config: PaperStatsConfig,
) -> dict[str, Any]:
    with torch.no_grad():
        afferent = model.get_aff_weights()
        centres = _retinotopic_centres(model, afferent)
        centre_index = (model.sheet_size // 2) * model.sheet_size + model.sheet_size // 2
        row = centre_index // model.sheet_size
        col = centre_index % model.sheet_size
        neighbour_indices = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbour_indices.append((row + dy) * model.sheet_size + col + dx)
        rf_examples = afferent[neighbour_indices].detach().cpu()
        model.update_interactions(layer_3=False)
        inhibitory_field = model.inh[centre_index, 0].detach().cpu()
        short_excitation = model.s_exc[centre_index, 0].detach().cpu()
        cde_field = (
            model.l_exc[centre_index, 0].detach().cpu()
            if model.microcolumnar
            else torch.zeros_like(inhibitory_field)
        )

    responses = _evaluate_images(
        model, evaluation_inputs, config.natural_response_samples, config
    )
    window_1d = torch.hann_window(
        model.sheet_size,
        periodic=False,
        dtype=responses.dtype,
    )
    response_window = torch.outer(window_1d, window_1d)
    centred_responses = responses[:, 0] - responses[:, 0].mean(
        dim=(-2, -1), keepdim=True
    )
    windowed_responses = centred_responses * response_window
    mean_amplitude = torch.fft.fftshift(
        torch.fft.fft2(windowed_responses).abs(), dim=(-2, -1)
    ).mean(0)
    radial_frequency_px, radial_amplitude = _radial_average(mean_amplitude.numpy())
    return {
        **orientation,
        "retinotopic_centres_lattice": centres.reshape(
            model.sheet_size, model.sheet_size, 2
        ).cpu(),
        "afferent_rf_examples": rf_examples,
        "example_unit_index": centre_index,
        "example_inhibitory_field": inhibitory_field,
        "example_sre_field": short_excitation,
        "example_cde_field": cde_field,
        "natural_inputs": evaluation_inputs[: min(12, len(evaluation_inputs))].cpu(),
        "natural_responses": responses[:12].cpu(),
        "mean_response_fourier_amplitude": mean_amplitude.cpu(),
        "response_fourier_window": "separable Hann",
        "response_fourier_radius_px": torch.as_tensor(radial_frequency_px),
        "response_fourier_radial_amplitude": torch.as_tensor(radial_amplitude),
    }


def _normalized_axial_autocorrelation(orientation: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    z = torch.exp(2j * orientation.cpu())
    z -= z.mean()
    autocorrelation = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.fft2(z).abs().square()).real
    )
    autocorrelation /= autocorrelation.max().clamp_min(1e-11)
    return _radial_average(autocorrelation.numpy())


def _centre_crop(value: torch.Tensor, side: int) -> torch.Tensor:
    start = (value.shape[-1] - side) // 2
    return value[..., start : start + side, start : start + side]


def _displacement_record(
    original: torch.Tensor,
    sigma: float,
    config: PaperStatsConfig,
) -> dict[str, Any]:
    """Apply one deterministic displacement and collect its spatial readouts."""

    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    displaced, permutation, mean_displacement = gaussian_local_permutation(
        original.clone(),
        sigma,
        seed=_derive_seed(
            config.base_seed,
            "remaining",
            "displacement",
            sigma,
        ),
    )
    crop = _centre_crop(displaced, config.analysis_crop)
    radius, autocorrelation = _normalized_axial_autocorrelation(crop)
    angle_difference = radial_mean_angle_distance(crop.numpy())
    return {
        "sigma": sigma,
        "realised_mean_displacement": float(mean_displacement),
        "orientation_crop_rad": crop,
        "permutation_new_to_old": permutation.cpu(),
        "autocorrelation_radius_lattice": torch.as_tensor(radius),
        "autocorrelation": torch.as_tensor(autocorrelation),
        "orientation_difference_deg": torch.as_tensor(angle_difference),
    }


def _connectivity_targets(side: int, crop: int, maximum: int | None, seed: int) -> torch.Tensor:
    start = (side - crop) // 2
    rows, cols = torch.meshgrid(
        torch.arange(start, start + crop),
        torch.arange(start, start + crop),
        indexing="ij",
    )
    targets = (rows * side + cols).flatten()
    if maximum is not None and maximum < len(targets):
        generator = torch.Generator().manual_seed(seed)
        targets = targets[torch.randperm(len(targets), generator=generator)[:maximum]]
    return targets


def _sample_sparse_binary_fields(
    eligible: torch.Tensor, fraction: float, seed: int
) -> torch.Tensor:
    """Retain ``fraction`` of the supra-threshold sites in each field."""

    fields, side, _ = eligible.shape
    output = torch.zeros_like(eligible, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    flat = eligible.reshape(fields, -1)
    output_flat = output.reshape(fields, -1)
    for index in range(fields):
        locations = torch.nonzero(flat[index], as_tuple=False).flatten()
        keep = int(round(fraction * len(locations)))
        if keep:
            choice = torch.randperm(len(locations), generator=generator)[:keep]
            output_flat[index, locations[choice]] = 1.0
    return output


def _aligned_fourier_profiles(
    fields: torch.Tensor, pixel_size_mm: float
) -> dict[str, Any]:
    side = fields.shape[-1]
    gaussian_y, gaussian_x = np.indices((side, side))
    sigma = side / 2
    gaussian = np.exp(
        -((gaussian_y - side // 2) ** 2 + (gaussian_x - side // 2) ** 2)
        / (2 * sigma**2)
    )
    profiles = []
    aligned_examples = []
    for field_index, field in enumerate(fields.numpy()):
        masked = field * gaussian
        coordinates = np.argwhere(masked > 0)
        if len(coordinates) < 10:
            continue
        centered = coordinates - np.asarray([side // 2, side // 2])
        covariance = np.cov(centered.T)
        values, vectors = np.linalg.eigh(covariance)
        principal = vectors[:, np.argmax(values)]
        angle = -math.degrees(math.atan2(principal[0], principal[1]))
        aligned = scipy_rotate(masked, angle, reshape=False, order=0, mode="constant")
        amplitude = np.abs(np.fft.fftshift(np.fft.fft2(aligned)))
        dc = amplitude[side // 2, side // 2]
        if dc <= 0:
            continue
        profiles.append(amplitude[side // 2] / dc)
        if len(aligned_examples) < 3:
            aligned_examples.append(aligned)
    frequencies = np.fft.fftshift(np.fft.fftfreq(side, d=pixel_size_mm))
    if profiles:
        profile_array = np.asarray(profiles)
        mean_profile = profile_array.mean(0)
        std_profile = profile_array.std(0)
    else:
        mean_profile = np.zeros(side, dtype=np.float64)
        std_profile = np.zeros(side, dtype=np.float64)
    return {
        "frequency_cycles_per_mm": torch.as_tensor(frequencies),
        "mean_normalized_amplitude": torch.as_tensor(mean_profile),
        "std_normalized_amplitude": torch.as_tensor(std_profile),
        "profile_count": len(profiles),
        "aligned_examples": torch.as_tensor(np.asarray(aligned_examples)),
    }


def _connectivity_condition_from_weights(
    lateral_correlations: torch.Tensor,
    side: int,
    permutation_new_to_old: torch.Tensor,
    config: PaperStatsConfig,
    condition_name: str,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    targets_new = _connectivity_targets(
        side,
        config.analysis_crop,
        config.connectivity_units,
        _derive_seed(config.base_seed, "remaining", condition_name, "targets"),
    )
    permutation = permutation_new_to_old.to(torch.long)
    targets_old = permutation[targets_new]
    weight_device = lateral_correlations.device
    source_order = permutation.to(weight_device)
    threshold = (
        float(lateral_correlations.float().mean())
        * config.connectivity_threshold_multiple
    )
    chunks = []
    chunk_size = 128
    starts: Iterable[int] = range(0, len(targets_old), chunk_size)
    if show_progress:
        starts = tqdm(
            starts,
            desc=f"connectivity {condition_name}",
            leave=False,
        )
    with torch.no_grad():
        for start in starts:
            old = targets_old[start : start + chunk_size].to(weight_device)
            fields = lateral_correlations[old, 0].flatten(1)[:, source_order]
            chunks.append((fields > threshold).cpu().reshape(-1, side, side))
    eligible = torch.cat(chunks)
    sparse = _sample_sparse_binary_fields(
        eligible,
        config.connectivity_fraction,
        _derive_seed(config.base_seed, "remaining", condition_name, "sparse"),
    )
    profiles = _aligned_fourier_profiles(sparse, pixel_size_mm=1.0 / config.analysis_crop)
    representative = len(sparse) // 2
    return {
        "target_positions_new": targets_new,
        "target_units_old": targets_old,
        "global_weight_threshold": threshold,
        "threshold_multiple": config.connectivity_threshold_multiple,
        "subtractive_threshold_in_global_means": (
            config.connectivity_threshold_multiple
        ),
        "sample_fraction_of_eligible_connections": (
            config.connectivity_fraction
        ),
        "sampling_basis": "supra-threshold connections",
        "eligible_examples": eligible[:3],
        "sparse_examples": sparse[
            [0, representative, len(sparse) - 1]
        ],
        "coordinate_transform": (
            "W_new[new_target,new_source] = "
            "W_old[perm[new_target],perm[new_source]]"
        ),
        **profiles,
    }


def configurable_displacement_analysis(
    bundle: dict[str, Any],
    *,
    sigma: float,
) -> dict[str, dict[str, Any]]:
    """Apply any non-negative displacement sigma to the saved orientation maps."""

    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    stored_config = bundle["metadata"]["config"]
    config = PaperStatsConfig(
        base_seed=int(stored_config["base_seed"]),
        analysis_crop=int(stored_config["analysis_crop"]),
    )
    orientations = bundle["orientation_maps_90"]
    return {
        map_name: _displacement_record(
            orientations[map_name]["orientation_rad"].cpu(),
            sigma,
            config,
        )
        for map_name in ("macro90", "micro90")
    }


def configurable_connectivity_analysis(
    bundle: dict[str, Any],
    *,
    subtractive_threshold: float = 2.0,
    connection_fraction: float = 0.25,
    sigma: float = 1.3,
    checkpoint_dir: str | Path | None = None,
    displacement_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recompute connectivity display data from trained map checkpoints.

    This lightweight post-training entry point exists so the paper notebook
    can expose the three visual-analysis choices in its plotting call.  The
    threshold is expressed in units of the global mean learned weight: a
    value of 2 subtracts ``2 * mean(weight)`` and retains positive entries.
    ``connection_fraction`` is the fraction of those remaining entries that
    is sampled. ``sigma`` is applied live and need not have been collected in
    the original displacement sweep.
    """

    subtractive_threshold = float(subtractive_threshold)
    connection_fraction = float(connection_fraction)
    sigma = float(sigma)
    if subtractive_threshold < 0:
        raise ValueError("subtractive_threshold must be non-negative")
    if not 0 < connection_fraction <= 1:
        raise ValueError("connection_fraction must satisfy 0 < value <= 1")

    metadata = bundle["metadata"]
    stored_config = metadata["config"]
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if displacement_records is None:
        displacement_records = configurable_displacement_analysis(
            bundle,
            sigma=sigma,
        )

    root = Path(
        checkpoint_dir
        if checkpoint_dir is not None
        else stored_config["checkpoint_dir"]
    )
    config = PaperStatsConfig(
        checkpoint_dir=str(root),
        device="cpu",
        base_seed=int(stored_config["base_seed"]),
        analysis_crop=int(stored_config["analysis_crop"]),
        connectivity_units=stored_config.get("connectivity_units"),
        connectivity_fraction=connection_fraction,
        connectivity_threshold_multiple=subtractive_threshold,
        displacement_sigma_panel=sigma,
    )
    specs = {
        name: MapSpec(**values)
        for name, values in metadata["map_specs"].items()
    }
    result: dict[str, Any] = {}
    for map_name in ("macro90", "micro90"):
        checkpoint_path = root / f"{map_name}.pt"
        compact_path = (
            BASE / "paper_results" / "connectivity_weights" / f"{map_name}.pt"
        )
        checkpoint = None
        if checkpoint_path.is_file():
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            weights = checkpoint["state"]["lateral_correlations"]
        elif compact_path.is_file():
            checkpoint = torch.load(
                compact_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            weights = checkpoint["quantized_weights"]
        else:
            raise FileNotFoundError(
                f"Missing {checkpoint_path} and compact fallback {compact_path}"
            )
        side = specs[map_name].sheet_size
        ideal_permutation = torch.arange(side**2)
        moved_record = displacement_records[map_name]
        result[map_name] = {
            "ideal": _connectivity_condition_from_weights(
                weights,
                side,
                ideal_permutation,
                config,
                f"{map_name}_ideal_sigma_{sigma:g}",
                show_progress=False,
            ),
            "displaced": _connectivity_condition_from_weights(
                weights,
                side,
                moved_record["permutation_new_to_old"],
                config,
                f"{map_name}_displaced_sigma_{sigma:g}",
                show_progress=False,
            ),
            "displacement_sigma": sigma,
            "realised_mean_displacement": moved_record[
                "realised_mean_displacement"
            ],
        }
        del weights, checkpoint
        _cleanup_model()
    return result


def _static_gratings(
    count: int, side: int, seed: int, cycles: float = 4.0
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2 * math.pi, count)
    phases = rng.uniform(0.0, 2 * math.pi, count)
    coords = np.linspace(-0.5, 0.5, side, endpoint=False, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    images = np.empty((count, side, side), dtype=np.float32)
    for index, (angle, phase) in enumerate(zip(angles, phases)):
        carrier = xx * math.cos(angle) + yy * math.sin(angle)
        images[index] = np.cos(2 * math.pi * cycles * carrier + phase)
    model_inputs = torch.from_numpy((images + 1.0) * 0.3)[:, None]
    return model_inputs, images.reshape(count, -1), angles


def _umap_embedding(matrix: np.ndarray, config: PaperStatsConfig, seed: int) -> np.ndarray:
    import umap

    components = min(config.umap_pca_components, matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(matrix)
    return umap.UMAP(
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        metric="euclidean",
        n_components=3,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(reduced)


def _model_grating_responses(
    model: NeuralSheet, inputs: torch.Tensor, config: PaperStatsConfig
) -> np.ndarray:
    responses = []
    with torch.no_grad():
        for image in tqdm(inputs, desc="grating population responses", leave=False):
            model(image[None].to(config.device), adaptation=False, layer_3=False)
            responses.append(model.current_response[0, 0].cpu())
    return torch.stack(responses).flatten(1).numpy()


def _figshare_download(destination: Path) -> None:
    api = (
        f"https://api.figshare.com/v2/articles/{STRINGER_FIGSHARE_ARTICLE}"
        f"/versions/{STRINGER_FIGSHARE_VERSION}"
    )
    with urllib.request.urlopen(api) as response:
        metadata = json.load(response)
    matches = [item for item in metadata["files"] if item["name"] == STRINGER_RECORDING]
    if not matches:
        raise FileNotFoundError(f"{STRINGER_RECORDING} is absent from {api}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"[Stringer] downloading {STRINGER_RECORDING} from Figshare")
    urllib.request.urlretrieve(matches[0]["download_url"], temporary)
    os.replace(temporary, destination)


def ensure_stringer_recording(config: PaperStatsConfig, download: bool = False) -> Path:
    destination = Path(config.stringer_dir) / STRINGER_RECORDING
    if destination.is_file():
        return destination
    if config.local_stringer_source:
        source = Path(config.local_stringer_source)
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Stringer] copying local recording from {source}")
            shutil.copy2(source, destination)
            return destination
    if download:
        _figshare_download(destination)
        return destination
    raise FileNotFoundError(
        f"Missing {destination}. Pass local_stringer_source, or call with download_stringer=True."
    )


def _stringer_embedding(path: Path, config: PaperStatsConfig) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True).item()
    responses = np.asarray(data["sresp"], dtype=np.float32)
    angles = np.asarray(data["istim"], dtype=np.float64)
    if responses.shape[1] != len(angles):
        raise ValueError("Stringer sresp and istim trial axes disagree")
    edges = np.linspace(0, 2 * math.pi, config.stringer_orientation_bins + 1)
    bin_id = np.clip(
        np.digitize(angles, edges) - 1,
        0,
        config.stringer_orientation_bins - 1,
    )
    counts = np.bincount(bin_id, minlength=config.stringer_orientation_bins)
    valid = counts[bin_id] >= config.stringer_min_trials_per_bin

    # The population-average orientation-residual score is accumulated in
    # neuron batches so the 18,802 x 3,831 residual matrix need not coexist
    # with the raw recording.
    arousal_sum = np.zeros(len(angles), dtype=np.float64)
    for start in tqdm(range(0, responses.shape[0], 1024), desc="Stringer trial selection"):
        block = responses[start : start + 1024].T
        block = (block - block.mean(0, keepdims=True)) / np.maximum(
            block.std(0, keepdims=True), 1e-6
        )
        residual = np.zeros_like(block)
        for orientation_bin in range(config.stringer_orientation_bins):
            selected = bin_id == orientation_bin
            if selected.sum() >= config.stringer_min_trials_per_bin:
                residual[selected] = block[selected] - block[selected].mean(
                    0, keepdims=True
                )
        arousal_sum += residual.sum(1)
    arousal = arousal_sum / responses.shape[0]
    high = valid & (arousal >= np.median(arousal[valid]))

    rng = np.random.default_rng(
        _derive_seed(config.base_seed, "remaining", "stringer", "neurons")
    )
    neuron_count = min(config.stringer_neurons, responses.shape[0])
    neuron_ids = np.sort(rng.choice(responses.shape[0], neuron_count, replace=False))
    matrix = responses[neuron_ids].T
    matrix = (matrix - matrix.mean(0, keepdims=True)) / np.maximum(
        matrix.std(0, keepdims=True), 1e-6
    )
    selected_matrix = matrix[high]
    selected_angles = angles[high]
    seed = _derive_seed(config.base_seed, "remaining", "stringer", "umap")
    embedding = _umap_embedding(selected_matrix, config, seed)
    return {
        "embedding": torch.as_tensor(embedding),
        "orientation_rad": torch.as_tensor(selected_angles),
        "selected_trial_indices": torch.as_tensor(np.flatnonzero(high)),
        "selected_neuron_indices": torch.as_tensor(neuron_ids),
        "valid_trial_count": int(valid.sum()),
        "selected_trial_count": int(high.sum()),
        "recorded_neuron_count": int(responses.shape[0]),
        "source_file": path.name,
        "selection": (
            "z-score each neuron across trials; subtract each neuron's "
            "24-bin orientation mean; retain trials at or above the median "
            "population-average residual"
        ),
    }


def _population_embeddings(
    specs: dict[str, MapSpec],
    config: PaperStatsConfig,
    stringer_path: Path | None,
) -> dict[str, Any]:
    grating_seed = _derive_seed(config.base_seed, "remaining", "gratings")
    inputs, images, angles = _static_gratings(
        config.grating_samples, config.crop_size, grating_seed
    )
    stimulus_seed = _derive_seed(config.base_seed, "remaining", "stimulus_umap")
    result: dict[str, Any] = {
        "stimulus": {
            "embedding": torch.as_tensor(_umap_embedding(images, config, stimulus_seed)),
            "orientation_rad": torch.as_tensor(angles),
        },
        "models": {},
        "umap_parameters": {
            "n_neighbors": config.umap_neighbors,
            "min_dist": config.umap_min_dist,
            "n_components": 3,
            "pca_components": config.umap_pca_components,
            "metric": "euclidean",
            "independently_fitted": True,
        },
    }
    for map_name in ("macro60", "micro60"):
        model = _load_model(specs[map_name], config)
        responses = _model_grating_responses(model, inputs, config)
        seed = _derive_seed(config.base_seed, "remaining", map_name, "umap")
        result["models"][map_name] = {
            "embedding": torch.as_tensor(_umap_embedding(responses, config, seed)),
            "orientation_rad": torch.as_tensor(angles),
        }
        del model
        _cleanup_model()
    if stringer_path is not None:
        result["stringer"] = _stringer_embedding(stringer_path, config)
    else:
        result["stringer"] = None
    return result


def _bundle_metadata(config: PaperStatsConfig, specs: Iterable[MapSpec]) -> dict[str, Any]:
    source_files = (
        BASE / "paper_stats_collector.py",
        BASE / "neuralsheet.py",
        BASE / "helpers" / "wiring_efficiency_utils.py",
    )
    hashes = {}
    for path in source_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(BASE))] = digest
    return {
        "collector_version": COLLECTOR_VERSION,
        "created_at_utc": _utc_now(),
        "config": asdict(config),
        "map_specs": {spec.name: asdict(spec) for spec in specs},
        "source_sha256": hashes,
        "geometry_notes": {
            "macro_reference_ratio_target": 72,
            "micro_reference_ratio_target": 523,
            "tree_shrew_squirrel": (
                "90x90 maps at R_long=18; tuning read from central 45x45; "
                "connectivity source fields retain the full 90x90 span"
            ),
        },
    }


def collect_paper_panel_data(
    config: PaperStatsConfig | None = None,
    *,
    download_stringer: bool = False,
) -> Path:
    """Train the paper maps and save their orientation and response data.

    Parameters
    ----------
    config:
        Complete run configuration.  Defaults reproduce the declared main
        panel dataset.  ``max_train_batches`` and ``map_names`` are intended
        only for smoke tests.
    download_stringer:
        If true and no local recording is supplied, obtain the named public
        recording from Figshare.  Model analyses still run if the Stringer
        file is deliberately omitted; only that embedding is absent.
    """

    config = config or PaperStatsConfig()
    _configure_reproducibility(config.strict_determinism)
    _seed_everything(config.base_seed)
    output = Path(config.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_specs = paper_map_specs()
    requested = set(config.map_names or [spec.name for spec in all_specs])
    specs = tuple(spec for spec in all_specs if spec.name in requested)
    if requested != {spec.name for spec in specs}:
        missing = sorted(requested - {spec.name for spec in specs})
        raise ValueError(f"Unknown map names: {missing}")

    dataset, evaluation_inputs, evaluation_metadata = _fixed_train_and_evaluation_data(
        config.input_stimuli,
        config.crop_size,
        config.batch_size,
        config.num_workers,
        config.base_seed,
        evaluation_samples=config.natural_response_samples,
    )
    for spec in specs:
        _train_map(spec, config, dataset)

    required = {spec.name for spec in all_specs}
    if requested != required:
        print("[smoke] map subset requested; training test completed without panel analyses")
        smoke = {
            "metadata": _bundle_metadata(config, specs),
            "evaluation": evaluation_metadata,
            "smoke_test": True,
            "completed_at_utc": _utc_now(),
        }
        _atomic_torch_save(smoke, output)
        return output

    spec_by_name = {spec.name: spec for spec in all_specs}
    bundle: dict[str, Any] = {
        "metadata": _bundle_metadata(config, all_specs),
        "evaluation": evaluation_metadata,
    }
    orientations: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for map_name in ("macro60", "micro60", "macro90", "micro90"):
        model = _load_model(spec_by_name[map_name], config)
        orientations[map_name] = _orientation_tuning(model, config)
        if map_name in ("macro60", "micro60"):
            summaries[map_name] = _map_summary(
                model, orientations[map_name], evaluation_inputs, config
            )
        del model
        _cleanup_model()
    bundle["map_summaries"] = summaries
    bundle["orientation_maps_90"] = orientations
    _atomic_torch_save(bundle, output)

    stringer_path = None
    try:
        stringer_path = ensure_stringer_recording(config, download=download_stringer)
    except FileNotFoundError as error:
        print(f"[Stringer] skipped: {error}")
    bundle["population_embeddings"] = _population_embeddings(
        spec_by_name, config, stringer_path
    )
    bundle["completed_at_utc"] = _utc_now()
    _atomic_torch_save(bundle, output)
    print(f"[done] complete paper-panel bundle: {output}")
    return output


__all__ = [
    "MapSpec",
    "PaperStatsConfig",
    "collect_paper_panel_data",
    "configurable_connectivity_analysis",
    "configurable_displacement_analysis",
    "ensure_stringer_recording",
    "paper_map_specs",
]

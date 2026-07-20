"""Data collection for the GitHub NeuralSheet self-organisation demo.

The long-running entry point is :func:`collect_microdomain_demo`.  It writes one
small, self-contained shard per learning snapshot plus a final reloadable L4
checkpoint.  Rendering is intentionally kept separate from collection so GIFs
can be restyled without rerunning the simulation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from neuralsheet import NeuralSheet
from helpers.map_plotting import detect_orientation_map_from_aff_weights
from helpers.wiring_efficiency_utils import (
    RandomCropDataset,
    get_typical_dist_fourier,
    init_nn,
    nn_loss,
)


SCHEMA_VERSION = 1


@dataclass
class MicrodomainDemoConfig:
    """All simulation, sampling, and storage controls for the demo run."""

    output_dir: str = "data_l4/github_demo_microdomain"
    root_dir: str = "./input_stimuli"
    device: str = "cuda"
    seed: int = 31

    # Requested simulation geometry.
    crop_size: int = 50
    sheet_size: int = 100
    r_rf: int = 7
    r_long: float = 12.0
    microcolumnar: bool = True
    # Fractional epoch count; values above one run multiple shuffled passes.
    train_fraction: float = 2.0

    # Current NeuralSheet defaults, written explicitly into the archive.
    homeo_target: float = 0.04
    act_target: float = 0.3
    aff_baseline: float = 0.3
    lat_dom: float = 0.34
    loc_b: float = 0.4
    iterations: int = 30
    model_lr: float = 1e-3
    hebbian_lr_ratio: float = 100.0

    # Existing notebook training schedule.
    batch_size: int = 32
    num_workers: int = 4
    min_input_mean: float = 0.15
    lr_initial: float = 1e-3
    lr_floor: float = 3e-4
    lr_decay: float = 1.0 - 5e-5

    # Snapshot/evaluation controls.
    n_snapshots: int = 100
    n_eval_stimuli: int = 128
    n_robustness_stimuli: int = 100
    n_reconstruction_examples: int = 6
    noise_gamma: float = 0.06
    noise_beta: float = 0.8
    pca_components: int = 100
    orientation_bins: int = 36
    n_afferent_samples: int = 64
    n_lateral_samples: int = 16

    # Float16 keeps the complete 100-frame archive near laptop/Git-LFS scale.
    storage_dtype: str = "float16"
    store_clean_states: bool = True
    store_noisy_states: bool = True
    overwrite: bool = False

    def validate(self) -> None:
        if self.train_fraction <= 0:
            raise ValueError("train_fraction must be positive.")
        if self.n_snapshots < 2:
            raise ValueError("n_snapshots must include at least initial and final frames.")
        if self.n_eval_stimuli < self.pca_components + 1:
            raise ValueError(
                "n_eval_stimuli must be at least pca_components + 1 so all requested "
                "PCA directions have non-zero rank support."
            )
        if not (1 <= self.n_robustness_stimuli <= self.n_eval_stimuli):
            raise ValueError("n_robustness_stimuli must be between 1 and n_eval_stimuli.")
        if not (1 <= self.n_reconstruction_examples <= self.n_eval_stimuli):
            raise ValueError("n_reconstruction_examples must be between 1 and n_eval_stimuli.")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype must be 'float16' or 'float32'.")
        if not self.microcolumnar:
            raise ValueError("This collector is specifically for the microdomain architecture.")


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    temporary.replace(path)


def _storage_dtype(config: MicrodomainDemoConfig) -> torch.dtype:
    return torch.float16 if config.storage_dtype == "float16" else torch.float32


def _model_kwargs(config: MicrodomainDemoConfig) -> dict[str, Any]:
    return {
        "input_size": config.crop_size,
        "sheet_size": config.sheet_size,
        "R_rf": config.r_rf,
        "R_long": config.r_long,
        "homeo_target": config.homeo_target,
        "act_target": config.act_target,
        "aff_baseline": config.aff_baseline,
        "lat_dom": config.lat_dom,
        "loc_b": config.loc_b,
        "iterations": config.iterations,
        "lr": config.model_lr,
        "hebbian_lr_ratio": config.hebbian_lr_ratio,
        "microcolumnar": config.microcolumnar,
        "device": config.device,
    }


def _prepare_output_directory(config: MicrodomainDemoConfig) -> tuple[Path, Path]:
    output_dir = Path(config.output_dir)
    frame_dir = output_dir / "frames"
    existing_frames = list(frame_dir.glob("frame_*.pt")) if frame_dir.exists() else []
    if existing_frames and not config.overwrite:
        raise FileExistsError(
            f"{frame_dir} already contains snapshot files. Choose a new output_dir or "
            "set overwrite=True after checking that replacement is intended."
        )
    if existing_frames and config.overwrite:
        for path in existing_frames:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, frame_dir


def _fixed_evaluation_stimuli(
    dataset: RandomCropDataset,
    config: MicrodomainDemoConfig,
) -> tuple[torch.Tensor, list[int]]:
    """Choose deterministic, valid crops without perturbing the training RNG."""

    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    try:
        dataset.images.sort()
        rng = np.random.default_rng(config.seed + 101)
        candidate_indices = rng.permutation(len(dataset)).tolist()
        stimuli: list[torch.Tensor] = []
        source_indices: list[int] = []
        for index in candidate_indices:
            random.seed(config.seed * 1_000_003 + int(index))
            torch.manual_seed(config.seed * 1_000_033 + int(index))
            image = dataset[int(index)][0:1]
            if float(image.mean()) <= config.min_input_mean:
                continue
            stimuli.append(image)
            source_indices.append(int(index))
            if len(stimuli) == config.n_eval_stimuli:
                break
        if len(stimuli) != config.n_eval_stimuli:
            raise RuntimeError(
                f"Only found {len(stimuli)} valid fixed crops; requested "
                f"{config.n_eval_stimuli}."
            )
        return torch.stack(stimuli), source_indices
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        np.random.set_state(numpy_state)


def _make_tracked_smiley(
    reference_stimuli: torch.Tensor,
    config: MicrodomainDemoConfig,
) -> torch.Tensor:
    """Build the fixed synthetic face reconstructed at every learning snapshot."""

    size = config.crop_size
    coordinates = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    smoothness = 0.075

    def gaussian_stroke(distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * (distance / smoothness).square())

    def soft_ellipse(cx: float, cy: float, rx: float, ry: float) -> torch.Tensor:
        normalized_radius = torch.sqrt(
            ((xx - cx) / rx).square() + ((yy - cy) / ry).square()
        )
        distance = (normalized_radius - 1) * min(rx, ry)
        edge = ((smoothness - distance) / (2 * smoothness)).clamp(0, 1)
        return edge.square() * (3 - 2 * edge)

    radius = torch.sqrt(xx.square() + yy.square())
    mouth_curve = 0.34 - 0.85 * xx.square()
    mouth_distance = torch.sqrt(
        (yy - mouth_curve).square() + (xx.abs() - 0.42).clamp_min(0).square()
    )
    raw = torch.stack(
        [
            gaussian_stroke((radius - 0.88).abs()),
            soft_ellipse(-0.30, -0.25, 0.165, 0.125),
            soft_ellipse(0.30, -0.25, 0.165, 0.125),
            gaussian_stroke(mouth_distance),
        ]
    ).amax(dim=0)

    reconstruction_references = reference_stimuli[: config.n_reconstruction_examples]
    zero_fractions = (reconstruction_references == 0).flatten(1).float().mean(1)
    target_zero_fraction = float(zero_fractions.max())
    target_mean = float(reconstruction_references.float().mean())
    threshold = torch.quantile(raw.flatten(), target_zero_fraction)
    scaled = (raw - threshold).clamp_min(0)
    scaled /= scaled.max().clamp_min(1e-12)
    low, high = 0.02, 20.0
    for _ in range(60):
        gamma = 0.5 * (low + high)
        if float(scaled.pow(gamma).mean()) > target_mean:
            low = gamma
        else:
            high = gamma
    face = scaled.pow(0.5 * (low + high))
    face[scaled == 0] = 0
    return face.unsqueeze(0)


@contextmanager
def _preserved_torch_rng(seed: int, device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def _cosine_per_sample(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return numerator / denominator.clamp_min(1e-11)


def _canonicalize_component_signs(components: torch.Tensor) -> torch.Tensor:
    flat = components.flatten(1)
    anchor_index = flat.abs().argmax(dim=1)
    anchor = flat.gather(1, anchor_index[:, None]).squeeze(1)
    signs = torch.where(anchor < 0, -torch.ones_like(anchor), torch.ones_like(anchor))
    return components * signs[:, None, None]


def _activity_pca(
    responses: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Exact PCA of fixed-stimulus L4 states, returned as spatial components."""

    matrix = responses.float().flatten(1)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, all_singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    all_variance = all_singular_values.square() / max(1, centered.shape[0] - 1)
    total_variance = centered.square().sum() / max(1, centered.shape[0] - 1)
    all_ratio = all_variance / total_variance.clamp_min(1e-11)
    effective_dim_95 = int((all_ratio.cumsum(dim=0) < 0.95).sum().item() + 1)

    n_components = min(n_components, vh.shape[0], centered.shape[0] - 1)
    singular_values = all_singular_values[:n_components]
    components = vh[:n_components].reshape(
        n_components,
        responses.shape[-2],
        responses.shape[-1],
    )
    components = _canonicalize_component_signs(components)
    explained_variance = singular_values.square() / max(1, centered.shape[0] - 1)
    explained_ratio = explained_variance / total_variance.clamp_min(1e-11)
    return components, explained_variance, explained_ratio, effective_dim_95


def _retinotopic_centres(model: NeuralSheet, afferent_weights: torch.Tensor) -> torch.Tensor:
    mass = afferent_weights.sum(dim=1)
    grid = model.rf_grids
    centre_normalized = (
        mass[..., None] * grid
    ).sum(dim=(1, 2)) / mass.sum(dim=(1, 2), keepdim=False)[:, None].clamp_min(1e-11)
    return (centre_normalized + 1.0) * (model.input_size - 1) / 2.0


def _orientation_products(
    model: NeuralSheet,
    config: MicrodomainDemoConfig,
) -> dict[str, torch.Tensor | float]:
    with torch.no_grad():
        afferent = model.get_aff_weights()
        tuning = detect_orientation_map_from_aff_weights(
            afferent,
            num_orientations=18,
            num_phases=8,
            return_degrees=False,
        )
        orientation = tuning["pref"].reshape(config.sheet_size, config.sheet_size).float().cpu()
        osi = tuning["osi"].reshape(config.sheet_size, config.sheet_size).float().cpu()
        period, spectrum, ring = get_typical_dist_fourier(orientation, mask=1)
        histogram = torch.histc(
            orientation,
            bins=config.orientation_bins,
            min=0.0,
            max=math.pi,
        )
        histogram /= histogram.sum().clamp_min(1e-11)
    return {
        "orientation_rad": orientation,
        "orientation_osi": osi,
        "orientation_histogram": histogram,
        "fourier_spectrum": spectrum.float().cpu(),
        "fourier_ring_profile": ring.float().cpu(),
        "fourier_period_pixels": float(torch.as_tensor(period).cpu()),
    }


def _evaluate_fixed_stimuli(
    model: NeuralSheet,
    decoder: dict[str, Any],
    stimuli: torch.Tensor,
    reconstruction_baseline: torch.Tensor,
    tracked_stimuli: dict[str, torch.Tensor],
    config: MicrodomainDemoConfig,
    snapshot_index: int,
) -> dict[str, Any]:
    device = torch.device(config.device)
    clean_responses: list[torch.Tensor] = []
    noisy_responses: list[torch.Tensor] = []

    decoder["model"].eval()
    with torch.no_grad(), _preserved_torch_rng(config.seed + 10_000, device):
        for stimulus_index, stimulus_cpu in enumerate(stimuli):
            stimulus = stimulus_cpu.unsqueeze(0).to(device, non_blocking=True)
            model(stimulus, adaptation=False, noise_gamma=0.0, layer_3=False)
            clean_responses.append(model.current_response.detach().clone())

            if stimulus_index < config.n_robustness_stimuli:
                # The same seed per stimulus at every snapshot isolates learning effects.
                noise_seed = config.seed * 1_000_003 + stimulus_index
                torch.manual_seed(noise_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(noise_seed)
                model(
                    stimulus,
                    adaptation=False,
                    noise_gamma=config.noise_gamma,
                    noise_beta=config.noise_beta,
                    layer_3=False,
                )
                noisy_responses.append(model.current_response.detach().clone())

        clean = torch.cat(clean_responses, dim=0)
        noisy = torch.cat(noisy_responses, dim=0)
        robust_clean = clean[: config.n_robustness_stimuli]
        stability = _cosine_per_sample(robust_clean, noisy)

        n_reconstruction = config.n_reconstruction_examples
        reconstructions = decoder["activ"](
            decoder["model"](clean[:n_reconstruction])
        )
        reconstruction_targets = stimuli[:n_reconstruction].to(device)
        reconstruction_cosine = _cosine_per_sample(reconstruction_targets, reconstructions)
        baseline = reconstruction_baseline[:n_reconstruction].to(device)
        reconstruction_relative = (
            reconstruction_cosine - baseline
        ) / (1.0 - baseline).clamp_min(1e-6)

        pca_components, pca_variance, pca_ratio, pca_dim_95 = _activity_pca(
            clean,
            config.pca_components,
        )

        tracked_l4_activities: dict[str, torch.Tensor] = {}
        tracked_reconstructions: dict[str, torch.Tensor] = {}
        tracked_reconstruction_cosine: dict[str, float] = {}
        for name, tracked_cpu in tracked_stimuli.items():
            tracked_input = tracked_cpu.unsqueeze(0).to(device, non_blocking=True)
            model(tracked_input, adaptation=False, noise_gamma=0.0, layer_3=False)
            tracked_activity = model.current_response.detach().clone()
            tracked_reconstruction = decoder["activ"](
                decoder["model"](tracked_activity)
            )
            tracked_l4_activities[name] = tracked_activity
            tracked_reconstructions[name] = tracked_reconstruction
            tracked_reconstruction_cosine[name] = float(
                _cosine_per_sample(tracked_input, tracked_reconstruction)[0].cpu()
            )

    decoder["model"].train()
    storage_dtype = _storage_dtype(config)
    result: dict[str, Any] = {
        "stability_per_stimulus": stability.float().cpu(),
        "stability_mean": float(stability.mean().cpu()),
        "stability_std": float(stability.std().cpu()),
        "reconstructions": reconstructions.to(storage_dtype).cpu(),
        "reconstruction_cosine": reconstruction_cosine.float().cpu(),
        "reconstruction_relative": reconstruction_relative.float().cpu(),
        "reconstruction_cosine_mean": float(reconstruction_cosine.mean().cpu()),
        "reconstruction_relative_mean": float(reconstruction_relative.mean().cpu()),
        "pca_components": pca_components.to(storage_dtype).cpu(),
        "pca_explained_variance": pca_variance.float().cpu(),
        "pca_explained_variance_ratio": pca_ratio.float().cpu(),
        "pca_effective_dim_95": pca_dim_95,
        "evaluation_snapshot_index": snapshot_index,
        "tracked_l4_activities": {
            name: value.to(storage_dtype).cpu()
            for name, value in tracked_l4_activities.items()
        },
        "tracked_reconstructions": {
            name: value.to(storage_dtype).cpu()
            for name, value in tracked_reconstructions.items()
        },
        "tracked_reconstruction_cosine": tracked_reconstruction_cosine,
    }
    if config.store_clean_states:
        result["clean_settled_states"] = robust_clean.to(storage_dtype).cpu()
    if config.store_noisy_states:
        result["noisy_settled_states"] = noisy.to(storage_dtype).cpu()
    return result


def _capture_snapshot(
    model: NeuralSheet,
    decoder: dict[str, Any],
    stimuli: torch.Tensor,
    reconstruction_baseline: torch.Tensor,
    tracked_stimuli: dict[str, torch.Tensor],
    config: MicrodomainDemoConfig,
    frame_dir: Path,
    snapshot_index: int,
    target_seen: int,
    seen: int,
    accepted: int,
    learning_rate: float,
    afferent_indices: torch.Tensor,
    lateral_indices: torch.Tensor,
) -> tuple[Path, dict[str, Any]]:
    storage_dtype = _storage_dtype(config)
    orientation_products = _orientation_products(model, config)
    evaluation = _evaluate_fixed_stimuli(
        model,
        decoder,
        stimuli,
        reconstruction_baseline,
        tracked_stimuli,
        config,
        snapshot_index,
    )

    with torch.no_grad():
        model.update_interactions(layer_3=False)
        afferent = model.get_aff_weights()
        retinotopy = _retinotopic_centres(model, afferent)
        frame = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_index": snapshot_index,
            "target_seen": int(target_seen),
            "seen": int(seen),
            "accepted": int(accepted),
            "training_fraction": float(seen / max(1, target_seen)),
            "learning_rate": float(learning_rate),
            "noise_gamma": float(config.noise_gamma),
            "noise_beta": float(config.noise_beta),
            **orientation_products,
            "retinotopy_xy_pixels": retinotopy.to(storage_dtype).cpu(),
            "sampled_afferent_weights": afferent[afferent_indices].to(storage_dtype).cpu(),
            "sampled_lateral_exc_correlations": model.lateral_correlations_exc[
                lateral_indices, 0
            ].to(storage_dtype).cpu(),
            "sampled_lateral_exc_effective": model.l_exc[
                lateral_indices, 0
            ].to(storage_dtype).cpu(),
            "mean_activation": float(model.mean_activations.mean().cpu()),
            "mean_activation_cv": float(
                (model.mean_activations.std() / model.mean_activations.mean().clamp_min(1e-11)).cpu()
            ),
            "aff_gain": float(model.aff_gain.mean().cpu()),
            "lat_gain": float(model.lat_gain.mean().cpu()),
            **evaluation,
        }

    frame_path = frame_dir / f"frame_{snapshot_index:03d}.pt"
    torch.save(frame, frame_path)
    summary = {
        "snapshot_index": snapshot_index,
        "frame": str(frame_path),
        "target_seen": int(target_seen),
        "seen": int(seen),
        "accepted": int(accepted),
        "learning_rate": float(learning_rate),
        "stability_mean": frame["stability_mean"],
        "reconstruction_cosine_mean": frame["reconstruction_cosine_mean"],
        "reconstruction_relative_mean": frame["reconstruction_relative_mean"],
        "tracked_reconstruction_cosine": frame["tracked_reconstruction_cosine"],
        "pca_effective_dim_95": frame["pca_effective_dim_95"],
        "fourier_period_pixels": frame["fourier_period_pixels"],
    }
    return frame_path, summary


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


L4_CHECKPOINT_TENSORS = (
    "afferent_weights",
    "lateral_correlations",
    "lateral_correlations_exc",
    "current_response",
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
    "l4_inh_threshold",
    "rw_l4_inh",
)


def save_l4_demo_checkpoint(
    path: str | Path,
    model: NeuralSheet,
    decoder: dict[str, Any],
    config: MicrodomainDemoConfig,
    progress: dict[str, Any],
) -> Path:
    """Save the complete learned L4 state without duplicating unused L2/3 tensors."""

    path = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_class": "neuralsheet.NeuralSheet",
        "model_kwargs": _model_kwargs(config),
        "l4_state": {
            name: getattr(model, name).detach().cpu()
            for name in L4_CHECKPOINT_TENSORS
        },
        "training_state": {
            "homeo_lr": float(model.homeo_lr),
            "hebbian_lr": float(model.hebbian_lr),
            **_jsonable(progress),
        },
        "decoder": {
            "input_size": config.sheet_size,
            "output_size": config.crop_size,
            "out_channels": 1,
            # nn_template.Network keeps layers in a plain dict, so its inherited
            # state_dict() is empty. Persist every layer explicitly.
            "layer_state_dicts": {
                int(index): _to_cpu_tree(layer.state_dict())
                for index, layer in decoder["model"].layers.items()
            },
            "optimizer_state_dict": _to_cpu_tree(decoder["optim"].state_dict()),
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": asdict(config),
    }
    torch.save(payload, path)
    return path


def load_l4_demo_checkpoint(
    path: str | Path,
    device: str = "cuda",
) -> tuple[NeuralSheet, dict[str, Any], dict[str, Any]]:
    """Reconstruct a NeuralSheet and decoder from :func:`save_l4_demo_checkpoint`."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    kwargs = dict(payload["model_kwargs"])
    kwargs["device"] = device
    model = NeuralSheet(**kwargs).to(device)
    with torch.no_grad():
        for name, tensor in payload["l4_state"].items():
            destination = getattr(model, name)
            restored = tensor.to(device=device, dtype=destination.dtype)
            if destination.shape == restored.shape:
                destination.copy_(restored)
            else:
                # Some adaptive scalars (notably mix) begin as [1,1,1,1]
                # tensors but are reassigned to scalar tensors during forward.
                setattr(model, name, restored.clone())
    model.homeo_lr = float(payload["training_state"]["homeo_lr"])
    model.hebbian_lr = float(payload["training_state"]["hebbian_lr"])
    model.update_interactions(layer_3=False)

    decoder_info = payload["decoder"]
    decoder = init_nn(
        decoder_info["input_size"],
        decoder_info["output_size"],
        decoder_info["out_channels"],
        device=device,
    )
    for index, state_dict in decoder_info["layer_state_dicts"].items():
        decoder["model"].layers[int(index)].load_state_dict(state_dict)
    decoder["optim"].load_state_dict(decoder_info["optimizer_state_dict"])
    return model, decoder, payload


def estimate_demo_archive_gib(config: MicrodomainDemoConfig) -> dict[str, float]:
    """Conservative payload estimate before starting the long run."""

    config.validate()
    bytes_per_value = 2 if config.storage_dtype == "float16" else 4
    n = config.n_snapshots
    sheet_pixels = config.sheet_size**2
    rf_pixels = (round(config.r_rf * 2) + 1) ** 2
    frame_bytes = 0
    frame_bytes += config.pca_components * sheet_pixels * bytes_per_value
    frame_bytes += 2 * config.n_lateral_samples * sheet_pixels * bytes_per_value
    frame_bytes += config.n_afferent_samples * 2 * rf_pixels * bytes_per_value
    if config.store_clean_states:
        frame_bytes += config.n_robustness_stimuli * sheet_pixels * bytes_per_value
    if config.store_noisy_states:
        frame_bytes += config.n_robustness_stimuli * sheet_pixels * bytes_per_value
    # One fixed synthetic face: its L4 activity and decoder reconstruction per frame.
    frame_bytes += (sheet_pixels + config.crop_size**2) * bytes_per_value
    frame_bytes += config.sheet_size**2 * (4 + 4 + bytes_per_value * 2)
    checkpoint_bytes = (
        2 * config.sheet_size**4 * 4
        + config.sheet_size**2 * 2 * rf_pixels * 4
    )
    gib = 1024**3
    return {
        "per_frame_gib": frame_bytes / gib,
        "all_frames_gib": frame_bytes * n / gib,
        "final_checkpoint_gib": checkpoint_bytes / gib,
        "estimated_total_gib": (frame_bytes * n + checkpoint_bytes) / gib,
    }


def collect_microdomain_demo(config: MicrodomainDemoConfig) -> dict[str, Any]:
    """Train the requested L4 sheet and collect approximately 100 demo frames."""

    config.validate()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_dir, frame_dir = _prepare_output_directory(config)
    device = torch.device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    dataset = RandomCropDataset(config.root_dir, config.crop_size)
    dataset.images.sort()
    stimuli, stimulus_source_indices = _fixed_evaluation_stimuli(dataset, config)
    tracked_stimuli = {
        "smiley_face": _make_tracked_smiley(stimuli, config),
    }
    representative_path = output_dir / "representative_inputs.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": stimuli.to(_storage_dtype(config)),
            "source_dataset_indices": stimulus_source_indices,
            "reconstruction_indices": list(range(config.n_reconstruction_examples)),
            "robustness_indices": list(range(config.n_robustness_stimuli)),
            "tracked_inputs": {
                name: value.to(_storage_dtype(config))
                for name, value in tracked_stimuli.items()
            },
        },
        representative_path,
    )

    mean_image = stimuli.mean(dim=0, keepdim=True)
    reconstruction_baseline = _cosine_per_sample(
        stimuli,
        mean_image.expand_as(stimuli),
    ).cpu()

    model = NeuralSheet(**_model_kwargs(config)).to(device)
    decoder = init_nn(config.sheet_size, config.crop_size, out_channels=1, device=config.device)
    model.train()
    decoder["model"].train()

    sample_rng = np.random.default_rng(config.seed + 202)
    afferent_indices = torch.as_tensor(
        sample_rng.choice(
            config.sheet_size**2,
            size=config.n_afferent_samples,
            replace=False,
        ),
        device=device,
        dtype=torch.long,
    )
    lateral_indices = torch.as_tensor(
        sample_rng.choice(
            config.sheet_size**2,
            size=config.n_lateral_samples,
            replace=False,
        ),
        device=device,
        dtype=torch.long,
    )

    target_seen = int(math.ceil(len(dataset) * config.train_fraction))
    schedule = np.rint(
        np.linspace(0, target_seen, config.n_snapshots)
    ).astype(int)
    if len(np.unique(schedule)) != config.n_snapshots:
        raise ValueError("Dataset/fraction is too small to place all snapshots uniquely.")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed + 303)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        worker_init_fn=_seed_worker,
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "config": asdict(config),
        "model_kwargs": _model_kwargs(config),
        "dataset_size": len(dataset),
        "target_seen": target_seen,
        "snapshot_schedule_seen": schedule.tolist(),
        "representative_inputs": str(representative_path),
        "reconstruction_baseline_cosine": reconstruction_baseline.tolist(),
        "afferent_sample_indices": afferent_indices.cpu().tolist(),
        "lateral_sample_indices": lateral_indices.cpu().tolist(),
        "frames": [],
        "archive_estimate_gib": estimate_demo_archive_gib(config),
        "measurement_notes": {
            "orientation": "Afferent ON/OFF grating-bank preference, radians modulo pi.",
            "retinotopy": "Afferent mass centroid in input-pixel coordinates for every L4 unit.",
            "pca": "Exact PCA of the same fixed clean evaluation stimuli at every frame.",
            "reconstruction": "Current concurrently-trained decoder on fixed inputs; relative score uses the fixed mean-image cosine baseline.",
            "tracked_reconstruction": "The fixed synthetic smiley face, its settled L4 activity, decoder reconstruction, and cosine score are saved at every snapshot.",
            "robustness": "Cosine similarity of clean/noisy settled L4 states; each stimulus receives the same recurrent-noise realization at every frame.",
            "lateral": "Both raw learned L4 excitatory correlations and their current cutoff-normalized effective kernels for fixed sampled source cells.",
        },
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)

    seen = 0
    accepted = 0
    learning_rate = config.lr_initial
    next_snapshot = 0

    def capture_due_snapshots() -> None:
        nonlocal next_snapshot
        while next_snapshot < config.n_snapshots and seen >= int(schedule[next_snapshot]):
            _, summary = _capture_snapshot(
                model=model,
                decoder=decoder,
                stimuli=stimuli,
                reconstruction_baseline=reconstruction_baseline,
                tracked_stimuli=tracked_stimuli,
                config=config,
                frame_dir=frame_dir,
                snapshot_index=next_snapshot,
                target_seen=target_seen,
                seen=seen,
                accepted=accepted,
                learning_rate=learning_rate,
                afferent_indices=afferent_indices,
                lateral_indices=lateral_indices,
            )
            manifest["frames"].append(summary)
            manifest["last_completed_snapshot"] = next_snapshot
            _write_json_atomic(output_dir / "manifest.json", manifest)
            next_snapshot += 1

    capture_due_snapshots()
    progress = tqdm(total=target_seen, desc="microdomain demo training", unit="images")
    def repeated_training_batches():
        """Yield freshly shuffled dataloader passes until the target is reached."""

        while True:
            yield from dataloader

    for batch in repeated_training_batches():
        if seen >= target_seen:
            break
        remaining = target_seen - seen
        batch = batch[:remaining]
        seen += int(batch.shape[0])
        batch = batch.to(device, non_blocking=True)

        batch_responses: list[torch.Tensor] = []
        batch_inputs: list[torch.Tensor] = []
        for image in batch:
            image = image[0:1].unsqueeze(0)
            if float(image.mean()) <= config.min_input_mean:
                continue

            learning_rate = max(learning_rate * config.lr_decay, config.lr_floor)
            model.hebbian_lr = learning_rate * config.hebbian_lr_ratio
            model.homeo_lr = learning_rate
            model(image, adaptation=True, noise_gamma=0.0, layer_3=False)
            model.hebbian_step(layer_3=False)
            batch_responses.append(model.current_response.detach().clone())
            batch_inputs.append(model.current_input.detach().clone())
            accepted += 1

        if batch_responses:
            responses = torch.cat(batch_responses, dim=0)
            targets = torch.cat(batch_inputs, dim=0)
            reconstructions = decoder["activ"](decoder["model"](responses))
            loss, _ = nn_loss(decoder, targets, reconstructions)
            decoder["optim"].zero_grad(set_to_none=True)
            loss.backward()
            decoder["optim"].step()

        progress.update(int(batch.shape[0]))
        progress.set_postfix(
            accepted=accepted,
            lr=f"{learning_rate:.5f}",
            snapshots=f"{next_snapshot}/{config.n_snapshots}",
        )
        capture_due_snapshots()
    progress.close()

    if seen != target_seen:
        raise RuntimeError(f"Training stopped after {seen} images; expected {target_seen}.")
    capture_due_snapshots()
    if next_snapshot != config.n_snapshots:
        raise RuntimeError(
            f"Collected {next_snapshot} snapshots; expected {config.n_snapshots}."
        )

    final_checkpoint = save_l4_demo_checkpoint(
        output_dir / "final_l4_checkpoint.pt",
        model,
        decoder,
        config,
        progress={"seen": seen, "accepted": accepted, "learning_rate": learning_rate},
    )
    summary_path = output_dir / "summary.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "frames": manifest["frames"],
            "snapshot_schedule_seen": schedule,
            "reconstruction_baseline_cosine": reconstruction_baseline,
            "config": asdict(config),
        },
        summary_path,
    )
    manifest.update(
        {
            "status": "complete",
            "seen": seen,
            "accepted": accepted,
            "final_checkpoint": str(final_checkpoint),
            "summary": str(summary_path),
        }
    )
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return {
        "model": model,
        "decoder": decoder,
        "manifest": manifest,
        "manifest_path": output_dir / "manifest.json",
        "summary_path": summary_path,
        "final_checkpoint_path": final_checkpoint,
    }


def plot_demo_snapshot(
    output_dir: str | Path,
    snapshot_index: int = -1,
    reconstruction_example: int = 0,
    lateral_example: int = 0,
) -> plt.Figure:
    """Compact collection sanity-check; final publication/GIF styling comes later."""

    output_dir = Path(output_dir)
    frame_paths = sorted((output_dir / "frames").glob("frame_*.pt"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame files found under {output_dir / 'frames'}.")
    frame_path = frame_paths[snapshot_index]
    frame = torch.load(frame_path, map_location="cpu", weights_only=False)
    representatives = torch.load(
        output_dir / "representative_inputs.pt",
        map_location="cpu",
        weights_only=False,
    )

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes[0, 0].imshow(frame["orientation_rad"], cmap="hsv", vmin=0, vmax=math.pi)
    axes[0, 0].set_title("L4 orientation")
    axes[0, 1].imshow(torch.log1p(frame["fourier_spectrum"]), cmap="magma")
    axes[0, 1].set_title("Fourier ring")
    axes[0, 2].plot(frame["pca_explained_variance_ratio"].cumsum(0))
    axes[0, 2].axhline(0.95, color="0.5", linestyle="--")
    axes[0, 2].set_title(f"PCA (95% dim={frame['pca_effective_dim_95']})")
    axes[0, 2].set_ylim(0, 1.02)
    axes[0, 3].imshow(frame["sampled_lateral_exc_effective"][lateral_example], cmap="magma")
    axes[0, 3].set_title("Sampled long-range excitation")

    target = representatives["inputs"][reconstruction_example, 0]
    reconstruction = frame["reconstructions"][reconstruction_example, 0]
    axes[1, 0].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Fixed input")
    axes[1, 1].imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(
        f"Reconstruction (r={frame['reconstruction_cosine'][reconstruction_example]:.2f})"
    )
    if "clean_settled_states" in frame:
        axes[1, 2].imshow(frame["clean_settled_states"][reconstruction_example, 0], cmap="gray")
        axes[1, 2].set_title("Clean settled L4 state")
    if "noisy_settled_states" in frame:
        axes[1, 3].imshow(frame["noisy_settled_states"][reconstruction_example, 0], cmap="gray")
        axes[1, 3].set_title(
            f"Noise={frame.get('noise_gamma', 'configured')}; stability={frame['stability_per_stimulus'][reconstruction_example]:.2f}"
        )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        if axis.images:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        f"Snapshot {frame['snapshot_index']:03d} · {frame['seen']:,} images seen",
        fontsize=15,
    )
    return fig

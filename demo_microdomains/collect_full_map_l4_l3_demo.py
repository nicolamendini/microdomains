"""Collect a full-map L4 + L2/3 training archive for a future demo.

This is deliberately a normal Python script, not a notebook.  It is safe to
run under ``nohup`` or a terminal multiplexer: snapshots are written
incrementally, the manifest is updated after every snapshot, and status files
are maintained under ``run_status/``.  Importing this module never starts a
run.

Example detached launch (run this only when the current GPU job is finished)::

    nohup python -u demo_microdomains/collect_full_map_l4_l3_demo.py \
      > run_status/full_map_l4_l3.log 2>&1 &

The default experiment is the requested 100x100 topological/full map with
both L4 and L2/3 enabled, ``microcolumnar=False``, and two training epochs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralsheet import NeuralSheet
from demo_microdomains.helpers.demo_collection import (
    _activity_pca,
    _cosine_per_sample,
    _fixed_evaluation_stimuli,
    _orientation_products,
    _retinotopic_centres,
    _seed_worker,
    _to_cpu_tree,
    _write_json_atomic,
)
from demo_microdomains.helpers.microdomain_demo import make_synthetic_inputs
from helpers.wiring_efficiency_utils import RandomCropDataset


SCHEMA_VERSION = 1
BASE_DIR = Path(__file__).resolve().parents[1]
STATUS_DIR = BASE_DIR / "run_status"
RUN_LABEL = "full_map_l4_l3_demo"


class SpatialReconstructionDecoder(nn.Module):
    """Small retinotopy-aware decoder that does not dominate GPU memory."""

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = int(output_size)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        reconstruction = self.layers(activity)
        return F.interpolate(
            reconstruction,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )


def _init_decoder(output_size: int, device: str) -> dict[str, Any]:
    model = SpatialReconstructionDecoder(output_size).to(device)
    return {
        "model": model,
        "optim": torch.optim.Adam(model.parameters(), lr=1e-3),
        "activ": torch.sigmoid,
    }


@dataclass
class FullMapL4L3Config:
    """Simulation, evaluation, and archive controls for the detached run."""

    output_dir: str = "data_l3/full_map_l4_l3_demo"
    root_dir: str = "./input_stimuli"
    device: str = "cuda"
    seed: int = 43

    # Full/topological map: L4 and L2/3 are both active; minicolumns are off.
    crop_size: int = 80
    sheet_size: int = 100
    r_rf: int = 7
    r_long: float = 12.0
    microcolumnar: bool = False
    train_epochs: float = 2.0

    homeo_target: float = 0.04
    act_target: float = 0.3
    aff_baseline: float = 0.3
    lat_dom: float = 0.5
    lat_dom_l3: float = 0.7
    loc_b: float = 0.4
    iterations: int = 30
    model_lr: float = 1e-3
    hebbian_lr_ratio: float = 100.0

    batch_size: int = 32
    num_workers: int = 4
    min_input_mean: float = 0.15
    lr_initial: float = 1e-3
    lr_floor: float = 3e-4
    lr_decay: float = 1.0 - 5e-5

    n_snapshots: int = 100
    n_eval_stimuli: int = 64
    n_activity_stimuli: int = 12
    n_reconstruction_examples: int = 6
    pca_components: int = 32
    orientation_bins: int = 36
    n_afferent_samples: int = 48
    n_connection_samples: int = 8
    storage_dtype: str = "float16"
    overwrite: bool = False

    def validate(self) -> None:
        if self.microcolumnar:
            raise ValueError("This collector is fixed to microcolumnar=False.")
        if self.train_epochs <= 0:
            raise ValueError("train_epochs must be positive.")
        if self.n_snapshots < 2:
            raise ValueError("n_snapshots must include initial and final snapshots.")
        if self.n_eval_stimuli < self.pca_components + 1:
            raise ValueError("n_eval_stimuli must be at least pca_components + 1.")
        if not (1 <= self.n_activity_stimuli <= self.n_eval_stimuli):
            raise ValueError("n_activity_stimuli must be within the evaluation set.")
        if not (1 <= self.n_reconstruction_examples <= self.n_eval_stimuli):
            raise ValueError("n_reconstruction_examples must be within the evaluation set.")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype must be float16 or float32.")


def _dtype(config: FullMapL4L3Config) -> torch.dtype:
    return torch.float16 if config.storage_dtype == "float16" else torch.float32


def _model_kwargs(config: FullMapL4L3Config) -> dict[str, Any]:
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
        "lat_dom_l3": config.lat_dom_l3,
        "iterations": config.iterations,
        "lr": config.model_lr,
        "hebbian_lr_ratio": config.hebbian_lr_ratio,
        "microcolumnar": False,
        "device": config.device,
    }


def _prepare_output(config: FullMapL4L3Config) -> tuple[Path, Path]:
    output = Path(config.output_dir)
    if not output.is_absolute():
        output = BASE_DIR / output
    frames = output / "frames"
    existing = sorted(frames.glob("frame_*.pt")) if frames.exists() else []
    archive_files = [
        output / "manifest.json",
        output / "representative_inputs.pt",
        output / "summary.pt",
        output / "final_l4_l3_checkpoint.pt",
    ]
    occupied = existing or [path for path in archive_files if path.exists()]
    if occupied and not config.overwrite:
        raise FileExistsError(
            f"{output} already contains collector output; choose another "
            "--output-dir or explicitly pass --overwrite after checking it."
        )
    if existing:
        for path in existing:
            path.unlink()
    if config.overwrite:
        for path in archive_files:
            if path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)
    return output, frames


def _status(name: str, value: Any) -> None:
    STATUS_DIR.mkdir(exist_ok=True)
    path = STATUS_DIR / f"{RUN_LABEL}.{name}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(str(value) + "\n")
    temporary.replace(path)


def _decoder_state(decoder: dict[str, Any]) -> dict[str, Any]:
    return {
        "class": "SpatialReconstructionDecoder",
        "model_state_dict": _to_cpu_tree(decoder["model"].state_dict()),
        "optimizer_state_dict": _to_cpu_tree(decoder["optim"].state_dict()),
    }


MODEL_STATE_TENSORS = (
    # L4 learned and adaptive state.
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
    # L4 -> L2/3 and L2/3 learned and adaptive state.
    "afferent_weights_l3",
    "lateral_correlations_l4_l3",
    "lateral_correlations_exc_l4_l3",
    "lateral_correlations_l3",
    "lateral_correlations_exc_l3",
    "current_response_l3",
    "mean_activations_l3",
    "thresholds_l3",
    "mean_fr_l3",
    "mean_lat_l3",
    "lat_gain_l3",
    "mean_aff_l3",
    "aff_gain_l3",
    "mix_l3",
    "avg_hist_l3",
    "old_style_mean_fr_l3",
    "old_style_mean_aff_l3",
    "global_exc_threshold_l3",
    "global_inh_threshold_l3",
    "rw_global_exc_l3",
    "rw_global_inh_l3",
    "rw_global_net_l3",
    "delta_mag",
)


def save_checkpoint(
    path: Path,
    model: NeuralSheet,
    decoder_l4: dict[str, Any],
    decoder_l3: dict[str, Any],
    config: FullMapL4L3Config,
    progress: dict[str, Any],
) -> Path:
    """Save every learned/adaptive tensor needed to continue the two-layer run."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_class": "neuralsheet.NeuralSheet",
        "model_kwargs": _model_kwargs(config),
        "model_state": {
            name: getattr(model, name).detach().cpu() for name in MODEL_STATE_TENSORS
        },
        "training_state": {
            "homeo_lr": float(model.homeo_lr),
            "hebbian_lr": float(model.hebbian_lr),
            **progress,
        },
        "decoders": {
            "l4": _decoder_state(decoder_l4),
            "l3": _decoder_state(decoder_l3),
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


def _load_decoder(
    state: dict[str, Any], config: FullMapL4L3Config, device: str
) -> dict[str, Any]:
    decoder = _init_decoder(config.crop_size, device)
    decoder["model"].load_state_dict(state["model_state_dict"])
    decoder["optim"].load_state_dict(state["optimizer_state_dict"])
    return decoder


def load_checkpoint(
    path: str | Path, device: str = "cuda"
) -> tuple[NeuralSheet, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reload the final model and both reconstruction decoders for a later demo."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = FullMapL4L3Config(**payload["config"])
    kwargs = dict(payload["model_kwargs"])
    kwargs["device"] = device
    model = NeuralSheet(**kwargs).to(device)
    with torch.no_grad():
        for name, tensor in payload["model_state"].items():
            destination = getattr(model, name)
            restored = tensor.to(device=device, dtype=destination.dtype)
            if destination.shape == restored.shape:
                destination.copy_(restored)
            else:
                setattr(model, name, restored.clone())
    model.homeo_lr = float(payload["training_state"]["homeo_lr"])
    model.hebbian_lr = float(payload["training_state"]["hebbian_lr"])
    # These masks are intentionally not archived because this run uses no sparsity.
    model.global_exc_sparsity_l3 = 1
    model.global_inh_sparsity_l3 = 1
    model.loc_lat_sparsity = 1
    model.update_interactions(layer_3=True)
    decoder_l4 = _load_decoder(payload["decoders"]["l4"], config, device)
    decoder_l3 = _load_decoder(payload["decoders"]["l3"], config, device)
    return model, decoder_l4, decoder_l3, payload


def estimate_archive_gib(config: FullMapL4L3Config) -> dict[str, float]:
    """Give a conservative size estimate without constructing the large model."""

    config.validate()
    b = 2 if config.storage_dtype == "float16" else 4
    s2 = config.sheet_size**2
    rf2 = (round(config.r_rf * 2) + 1) ** 2
    rf_l3 = int(config.r_long * 2 / 1.8)
    rf_l3 += 1 - rf_l3 % 2
    frame = 2 * config.pca_components * s2 * b
    frame += 2 * config.n_activity_stimuli * s2 * b
    frame += 2 * config.n_reconstruction_examples * config.crop_size**2 * b
    frame += config.n_afferent_samples * (2 * rf2 + rf_l3**2) * b
    frame += 6 * config.n_connection_samples * s2 * b
    frame += 2 * config.iterations * s2 * b
    # Four synthetic probes, each with L4/L2/3 activity and reconstruction.
    frame += 4 * (2 * s2 + 2 * config.crop_size**2) * b
    # Six dense learned connection fields dominate the float32 checkpoint.
    checkpoint = 6 * s2 * s2 * 4 + s2 * (2 * rf2 + rf_l3**2) * 4
    gib = 1024**3
    return {
        "per_frame_gib": frame / gib,
        "all_frames_gib": frame * config.n_snapshots / gib,
        "final_checkpoint_gib": checkpoint / gib,
        "estimated_total_gib": (frame * config.n_snapshots + checkpoint) / gib,
    }


def _evaluate(
    model: NeuralSheet,
    decoder_l4: dict[str, Any],
    decoder_l3: dict[str, Any],
    stimuli: torch.Tensor,
    tracked_inputs: dict[str, torch.Tensor],
    config: FullMapL4L3Config,
) -> dict[str, Any]:
    device = torch.device(config.device)
    l4_responses: list[torch.Tensor] = []
    l3_responses: list[torch.Tensor] = []
    trajectory_l4 = trajectory_l3 = None

    decoder_l4["model"].eval()
    decoder_l3["model"].eval()
    with torch.no_grad():
        for index, stimulus_cpu in enumerate(stimuli):
            stimulus = stimulus_cpu.unsqueeze(0).to(device, non_blocking=True)
            model(
                stimulus,
                adaptation=False,
                noise_gamma=0.0,
                sparsity=0,
                loc_sparsity=0,
                layer_3=True,
                track_response=index == 0,
            )
            l4_responses.append(model.current_response.detach().clone())
            l3_responses.append(model.current_response_l3.detach().clone())
            if index == 0:
                trajectory_l4 = model.response_tracker.detach().clone()
                trajectory_l3 = model.response_tracker_l3.detach().clone()

        l4 = torch.cat(l4_responses)
        l3 = torch.cat(l3_responses)
        n_reco = config.n_reconstruction_examples
        target = stimuli[:n_reco].to(device)
        reco_l4 = decoder_l4["activ"](decoder_l4["model"](l4[:n_reco]))
        reco_l3 = decoder_l3["activ"](decoder_l3["model"](l3[:n_reco]))
        cosine_l4 = _cosine_per_sample(target, reco_l4)
        cosine_l3 = _cosine_per_sample(target, reco_l3)
        pca_l4 = _activity_pca(l4, config.pca_components)
        pca_l3 = _activity_pca(l3, config.pca_components)

        tracked_l4: dict[str, torch.Tensor] = {}
        tracked_l3: dict[str, torch.Tensor] = {}
        tracked_reco_l4: dict[str, torch.Tensor] = {}
        tracked_reco_l3: dict[str, torch.Tensor] = {}
        tracked_cosine_l4: dict[str, float] = {}
        tracked_cosine_l3: dict[str, float] = {}
        for name, tracked_cpu in tracked_inputs.items():
            tracked = tracked_cpu[None, None].to(device, non_blocking=True)
            model(
                tracked,
                adaptation=False,
                noise_gamma=0.0,
                sparsity=0,
                loc_sparsity=0,
                layer_3=True,
            )
            activity_l4 = model.current_response.detach().clone()
            activity_l3 = model.current_response_l3.detach().clone()
            reconstruction_l4 = decoder_l4["activ"](decoder_l4["model"](activity_l4))
            reconstruction_l3 = decoder_l3["activ"](decoder_l3["model"](activity_l3))
            tracked_l4[name] = activity_l4
            tracked_l3[name] = activity_l3
            tracked_reco_l4[name] = reconstruction_l4
            tracked_reco_l3[name] = reconstruction_l3
            tracked_cosine_l4[name] = float(
                _cosine_per_sample(tracked, reconstruction_l4)[0].cpu()
            )
            tracked_cosine_l3[name] = float(
                _cosine_per_sample(tracked, reconstruction_l3)[0].cpu()
            )

    decoder_l4["model"].train()
    decoder_l3["model"].train()
    storage = _dtype(config)
    return {
        "activities_l4": l4[: config.n_activity_stimuli].to(storage).cpu(),
        "activities_l3": l3[: config.n_activity_stimuli].to(storage).cpu(),
        "reconstructions_l4": reco_l4.to(storage).cpu(),
        "reconstructions_l3": reco_l3.to(storage).cpu(),
        "reconstruction_cosine_l4": cosine_l4.float().cpu(),
        "reconstruction_cosine_l3": cosine_l3.float().cpu(),
        "reconstruction_cosine_mean_l4": float(cosine_l4.mean().cpu()),
        "reconstruction_cosine_mean_l3": float(cosine_l3.mean().cpu()),
        "pca_components_l4": pca_l4[0].to(storage).cpu(),
        "pca_explained_variance_l4": pca_l4[1].float().cpu(),
        "pca_explained_variance_ratio_l4": pca_l4[2].float().cpu(),
        "pca_effective_dim_95_l4": pca_l4[3],
        "pca_components_l3": pca_l3[0].to(storage).cpu(),
        "pca_explained_variance_l3": pca_l3[1].float().cpu(),
        "pca_explained_variance_ratio_l3": pca_l3[2].float().cpu(),
        "pca_effective_dim_95_l3": pca_l3[3],
        "settling_trajectory_l4": trajectory_l4.to(storage).cpu(),
        "settling_trajectory_l3": trajectory_l3.to(storage).cpu(),
        "tracked_activities_l4": {
            name: value.to(storage).cpu() for name, value in tracked_l4.items()
        },
        "tracked_activities_l3": {
            name: value.to(storage).cpu() for name, value in tracked_l3.items()
        },
        "tracked_reconstructions_l4": {
            name: value.to(storage).cpu() for name, value in tracked_reco_l4.items()
        },
        "tracked_reconstructions_l3": {
            name: value.to(storage).cpu() for name, value in tracked_reco_l3.items()
        },
        "tracked_reconstruction_cosine_l4": tracked_cosine_l4,
        "tracked_reconstruction_cosine_l3": tracked_cosine_l3,
    }


def _capture_snapshot(
    model: NeuralSheet,
    decoder_l4: dict[str, Any],
    decoder_l3: dict[str, Any],
    stimuli: torch.Tensor,
    tracked_inputs: dict[str, torch.Tensor],
    config: FullMapL4L3Config,
    frame_dir: Path,
    snapshot_index: int,
    target_seen: int,
    seen: int,
    accepted: int,
    learning_rate: float,
    afferent_indices: torch.Tensor,
    connection_indices: torch.Tensor,
) -> dict[str, Any]:
    evaluation = _evaluate(
        model, decoder_l4, decoder_l3, stimuli, tracked_inputs, config
    )
    orientation = _orientation_products(model, config)
    storage = _dtype(config)
    with torch.no_grad():
        model.global_exc_sparsity_l3 = 1
        model.global_inh_sparsity_l3 = 1
        model.loc_lat_sparsity = 1
        model.update_interactions(layer_3=True)
        afferent_l4 = model.get_aff_weights()
        afferent_l3 = model.get_aff_weights_l3()
        frame = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_index": snapshot_index,
            "target_seen": int(target_seen),
            "seen": int(seen),
            "accepted": int(accepted),
            "training_progress": float(seen / max(1, target_seen)),
            "learning_rate": float(learning_rate),
            **orientation,
            "retinotopy_l4_xy_pixels": _retinotopic_centres(
                model, afferent_l4
            ).to(storage).cpu(),
            "sampled_afferent_weights_l4": afferent_l4[afferent_indices].to(storage).cpu(),
            "sampled_afferent_weights_l4_l3": afferent_l3[afferent_indices].to(storage).cpu(),
            "sampled_l4_inhibition_raw": model.lateral_correlations[
                connection_indices, 0
            ].to(storage).cpu(),
            "sampled_l4_inhibition_effective": model.inh[
                connection_indices, 0
            ].to(storage).cpu(),
            "sampled_l4_l3_raw": model.lateral_correlations_l4_l3[
                connection_indices, 0
            ].to(storage).cpu(),
            "sampled_l4_l3_effective": model.inh_l4_l3[
                connection_indices, 0
            ].to(storage).cpu(),
            "sampled_l3_global_exc_effective": model.global_exc_l3[
                connection_indices, 0
            ].to(storage).cpu(),
            "sampled_l3_global_inh_effective": model.global_inh_l3[
                connection_indices, 0
            ].to(storage).cpu(),
            "mean_activation_l4": float(model.mean_activations.mean().cpu()),
            "mean_activation_l3": float(model.mean_activations_l3.mean().cpu()),
            "aff_gain_l4": float(model.aff_gain.mean().cpu()),
            "lat_gain_l4": float(model.lat_gain.mean().cpu()),
            "aff_gain_l3": float(model.aff_gain_l3.mean().cpu()),
            "lat_gain_l3": float(model.lat_gain_l3.mean().cpu()),
            **evaluation,
        }
    path = frame_dir / f"frame_{snapshot_index:03d}.pt"
    torch.save(frame, path)
    return {
        "snapshot_index": snapshot_index,
        "frame": str(path),
        "seen": int(seen),
        "accepted": int(accepted),
        "learning_rate": float(learning_rate),
        "reconstruction_cosine_mean_l4": frame["reconstruction_cosine_mean_l4"],
        "reconstruction_cosine_mean_l3": frame["reconstruction_cosine_mean_l3"],
        "pca_effective_dim_95_l4": frame["pca_effective_dim_95_l4"],
        "pca_effective_dim_95_l3": frame["pca_effective_dim_95_l3"],
        "fourier_period_pixels": frame["fourier_period_pixels"],
    }


def collect(config: FullMapL4L3Config) -> dict[str, Any]:
    """Run two-layer learning and incrementally collect the future-demo archive."""

    config.validate()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    output, frame_dir = _prepare_output(config)
    device = torch.device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    root_dir = Path(config.root_dir)
    if not root_dir.is_absolute():
        root_dir = BASE_DIR / root_dir
    dataset = RandomCropDataset(str(root_dir), config.crop_size)
    dataset.images.sort()
    stimuli, source_indices = _fixed_evaluation_stimuli(dataset, config)
    synthetic = make_synthetic_inputs(
        stimuli[: config.n_reconstruction_examples], fourier_cutoff=0.12
    )
    tracked_inputs = {
        "smiley_normal": synthetic["normal"][0],
        "neuron_normal": synthetic["normal"][1],
        "smiley_fourier": synthetic["fourier"][0],
        "neuron_fourier": synthetic["fourier"][1],
    }
    representative_path = output / "representative_inputs.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": stimuli.to(_dtype(config)),
            "source_dataset_indices": source_indices,
            "activity_indices": list(range(config.n_activity_stimuli)),
            "reconstruction_indices": list(range(config.n_reconstruction_examples)),
            "tracked_inputs": {
                name: value.to(_dtype(config)) for name, value in tracked_inputs.items()
            },
            "tracked_input_metadata": {
                "fourier_cutoff": synthetic["fourier_cutoff"],
                "target_zero_fraction": synthetic["target_zero_fraction"],
                "target_mean": synthetic["target_mean"],
            },
        },
        representative_path,
    )

    model = NeuralSheet(**_model_kwargs(config)).to(device)
    decoder_l4 = _init_decoder(config.crop_size, config.device)
    decoder_l3 = _init_decoder(config.crop_size, config.device)
    model.train()
    decoder_l4["model"].train()
    decoder_l3["model"].train()

    sample_rng = np.random.default_rng(config.seed + 202)
    afferent_indices = torch.as_tensor(
        sample_rng.choice(
            config.sheet_size**2, config.n_afferent_samples, replace=False
        ),
        device=device,
        dtype=torch.long,
    )
    connection_indices = torch.as_tensor(
        sample_rng.choice(
            config.sheet_size**2, config.n_connection_samples, replace=False
        ),
        device=device,
        dtype=torch.long,
    )

    target_seen = int(math.ceil(len(dataset) * config.train_epochs))
    schedule = np.rint(np.linspace(0, target_seen, config.n_snapshots)).astype(int)
    if len(np.unique(schedule)) != config.n_snapshots:
        raise ValueError("Too few training samples to place every snapshot uniquely.")
    generator = torch.Generator().manual_seed(config.seed + 303)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
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
        "archive_estimate_gib": estimate_archive_gib(config),
        "afferent_sample_indices": afferent_indices.cpu().tolist(),
        "connection_sample_indices": connection_indices.cpu().tolist(),
        "frames": [],
        "measurement_notes": {
            "layers": "Every forward/training step enables layer_3=True; both L4 and L2/3 are learned.",
            "reconstruction": "Independent L4 and L2/3 decoders are trained concurrently and evaluated on identical fixed crops.",
            "activity": "Fixed-stimulus L4/L2/3 activity, PCA maps, and one full settling trajectory are stored per snapshot.",
            "synthetic_probes": "Normal and Fourier-limited smiley/neuron inputs, both-layer activities, reconstructions, and cosine scores are stored at every snapshot.",
            "connectivity": "Fixed source cells track L4 inhibition, L4-to-L2/3 drive, and L2/3 global excitation/inhibition.",
            "architecture": "Full/topological map with microcolumnar=False and no imposed connection sparsity.",
        },
    }
    manifest_path = output / "manifest.json"
    _write_json_atomic(manifest_path, manifest)

    seen = accepted = next_snapshot = 0
    learning_rate = config.lr_initial

    def capture_due() -> None:
        nonlocal next_snapshot
        while next_snapshot < config.n_snapshots and seen >= int(schedule[next_snapshot]):
            summary = _capture_snapshot(
                model,
                decoder_l4,
                decoder_l3,
                stimuli,
                tracked_inputs,
                config,
                frame_dir,
                next_snapshot,
                target_seen,
                seen,
                accepted,
                learning_rate,
                afferent_indices,
                connection_indices,
            )
            manifest["frames"].append(summary)
            manifest["last_completed_snapshot"] = next_snapshot
            _write_json_atomic(manifest_path, manifest)
            _status("progress.json", json.dumps(summary, indent=2))
            next_snapshot += 1

    capture_due()
    progress = tqdm(total=target_seen, desc="full-map L4 + L2/3", unit="images")

    def repeated_batches():
        while True:
            yield from dataloader

    for batch in repeated_batches():
        if seen >= target_seen:
            break
        batch = batch[: target_seen - seen]
        seen += int(batch.shape[0])
        batch = batch.to(device, non_blocking=True)
        l4_batch: list[torch.Tensor] = []
        l3_batch: list[torch.Tensor] = []
        input_batch: list[torch.Tensor] = []
        for image in batch:
            image = image[0:1].unsqueeze(0)
            if float(image.mean()) <= config.min_input_mean:
                continue
            learning_rate = max(learning_rate * config.lr_decay, config.lr_floor)
            model.homeo_lr = learning_rate
            model.hebbian_lr = learning_rate * config.hebbian_lr_ratio
            model(
                image,
                adaptation=True,
                noise_gamma=0.0,
                sparsity=0,
                loc_sparsity=0,
                layer_3=True,
            )
            model.hebbian_step(layer_3=True)
            l4_batch.append(model.current_response.detach().clone())
            l3_batch.append(model.current_response_l3.detach().clone())
            input_batch.append(model.current_input.detach().clone())
            accepted += 1

        if input_batch:
            targets = torch.cat(input_batch)
            for decoder, responses in (
                (decoder_l4, torch.cat(l4_batch)),
                (decoder_l3, torch.cat(l3_batch)),
            ):
                reconstruction = decoder["activ"](decoder["model"](responses))
                loss = (targets - reconstruction).square().mean()
                decoder["optim"].zero_grad(set_to_none=True)
                loss.backward()
                decoder["optim"].step()

        progress.update(int(batch.shape[0]))
        progress.set_postfix(
            accepted=accepted,
            lr=f"{learning_rate:.5f}",
            snapshots=f"{next_snapshot}/{config.n_snapshots}",
        )
        capture_due()
    progress.close()

    if seen != target_seen or next_snapshot != config.n_snapshots:
        raise RuntimeError(
            f"Incomplete run: seen={seen}/{target_seen}, snapshots="
            f"{next_snapshot}/{config.n_snapshots}."
        )
    checkpoint = save_checkpoint(
        output / "final_l4_l3_checkpoint.pt",
        model,
        decoder_l4,
        decoder_l3,
        config,
        {"seen": seen, "accepted": accepted, "learning_rate": learning_rate},
    )
    summary_path = output / "summary.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "frames": manifest["frames"],
            "snapshot_schedule_seen": schedule,
            "config": asdict(config),
        },
        summary_path,
    )
    manifest.update(
        status="complete",
        seen=seen,
        accepted=accepted,
        final_checkpoint=str(checkpoint),
        summary=str(summary_path),
    )
    _write_json_atomic(manifest_path, manifest)
    return {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "checkpoint_path": checkpoint,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=FullMapL4L3Config.output_dir)
    parser.add_argument("--root-dir", default=FullMapL4L3Config.root_dir)
    parser.add_argument("--device", default=FullMapL4L3Config.device)
    parser.add_argument("--epochs", type=float, default=FullMapL4L3Config.train_epochs)
    parser.add_argument("--snapshots", type=int, default=FullMapL4L3Config.n_snapshots)
    parser.add_argument("--seed", type=int, default=FullMapL4L3Config.seed)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="print the expected archive size and exit without allocating the model",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = FullMapL4L3Config(
        output_dir=args.output_dir,
        root_dir=args.root_dir,
        device=args.device,
        train_epochs=args.epochs,
        n_snapshots=args.snapshots,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    if args.estimate_only:
        print(json.dumps(estimate_archive_gib(config), indent=2))
        return 0

    _status("pid", os.getpid())
    _status("started", time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    try:
        result = collect(config)
    except Exception:
        _status("error", traceback.format_exc())
        _status("exit", 1)
        raise
    _status("done", time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    _status("exit", 0)
    print(f"Complete: {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

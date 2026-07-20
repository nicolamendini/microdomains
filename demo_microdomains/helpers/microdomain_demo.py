"""Presentation and analysis helpers for the completed microdomain archive.

The expensive simulation remains in :mod:`demo_microdomains.helpers.demo_collection`.  This
module contains the plotting, animation, input-statistics, synthetic-stimulus,
and scatter-permutation code used by the public GitHub demo notebook.  Keeping
it here makes that notebook a readable sequence of scientific results instead
of a second implementation of the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
import torch.nn.functional as F

from .demo_collection import (
    MicrodomainDemoConfig,
    collect_microdomain_demo,
    estimate_demo_archive_gib,
    load_l4_demo_checkpoint,
)
from .cellular_orientation_displacement import (
    MapAnalysis,
    plot_fixed_sigma_summary,
    plot_sparse_displacement_links,
    results_table as cellular_results_table,
    run_fixed_sigma_analysis,
)
from helpers.wiring_efficiency_utils import (
    gaussian_local_permutation,
    get_typical_dist_fourier,
)


DEMO_FONT_SIZE = 16
DEMO_SUPTITLE_SIZE = 18
# README assets share one visual height.  Two-row summaries get extra vertical
# room without becoming twice as tall as the single-row animations.
README_ROW_HEIGHT = 5.0
README_TWO_ROW_HEIGHT = 1.5 * README_ROW_HEIGHT
README_PANEL_WIDTH = 5.0
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIRECTORY = REPOSITORY_ROOT / "demo_microdomains"
DEMO_ASSET_DIRECTORY = DEMO_DIRECTORY / "demo_assets" / "microdomain"
DEMO_UMAP_DATA = DEMO_DIRECTORY / "data" / "umap" / "four_panel_umap_embeddings.npz"


@dataclass
class MicrodomainArchive:
    """Lazy reader for the incremental frame archive."""

    root: Path
    manifest: dict[str, Any]
    representative: dict[str, Any]
    frame_paths: list[Path]

    @classmethod
    def open(cls, root: str | Path | None = None) -> "MicrodomainArchive":
        if root is None:
            root = REPOSITORY_ROOT / "data_l4" / "github_demo_microdomain"
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing microdomain manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        frame_paths = sorted((root / "frames").glob("frame_*.pt"))
        if not frame_paths:
            raise FileNotFoundError(f"No frame shards found in {root / 'frames'}")
        representative = torch.load(
            root / "representative_inputs.pt",
            map_location="cpu",
            weights_only=False,
        )
        return cls(root, manifest, representative, frame_paths)

    def frame(self, index: int) -> dict[str, Any]:
        return torch.load(
            self.frame_paths[index], map_location="cpu", weights_only=False
        )

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)

    @property
    def dataset_size(self) -> int:
        return int(self.manifest["dataset_size"])

    def epoch_at(self, frame: dict[str, Any]) -> float:
        return float(frame["seen"]) / max(1, self.dataset_size)

    def sampled_frame_indices(self, n: int = 25) -> list[int]:
        n = min(max(2, int(n)), self.n_frames)
        return np.unique(np.rint(np.linspace(0, self.n_frames - 1, n)).astype(int)).tolist()


def load_archive(root: str | Path | None = None) -> MicrodomainArchive:
    """Open and validate the completed archive used throughout the notebook."""

    archive = MicrodomainArchive.open(root)
    if archive.manifest.get("status") != "complete":
        raise RuntimeError(
            f"Archive status is {archive.manifest.get('status')!r}, not 'complete'."
        )
    return archive


def train_or_load_demo(config: MicrodomainDemoConfig) -> MicrodomainArchive:
    """Run collection only when a complete archive is not already available."""

    manifest_path = Path(config.output_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            return load_archive(config.output_dir)
    collect_microdomain_demo(config)
    return load_archive(config.output_dir)


def github_demo_config(
    output_dir: str | Path | None = None,
    device: str = "cuda",
    overwrite: bool = False,
) -> MicrodomainDemoConfig:
    """Canonical two-epoch configuration shared by training and presentation."""

    if output_dir is None:
        output_dir = REPOSITORY_ROOT / "data_l4" / "github_demo_microdomain"

    return MicrodomainDemoConfig(
        output_dir=str(output_dir),
        root_dir=str(REPOSITORY_ROOT / "input_stimuli"),
        device=device,
        seed=31,
        crop_size=80,
        sheet_size=100,
        r_rf=7,
        r_long=9,
        microcolumnar=True,
        train_fraction=2.0,
        n_snapshots=100,
        n_eval_stimuli=128,
        n_robustness_stimuli=100,
        n_reconstruction_examples=6,
        noise_gamma=0.06,
        noise_beta=0.8,
        pca_components=100,
        n_afferent_samples=64,
        n_lateral_samples=16,
        orientation_bins=36,
        storage_dtype="float16",
        store_clean_states=True,
        store_noisy_states=True,
        overwrite=overwrite,
    )


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)


def _apply_demo_typography(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    """Use one readable type scale across every public-demo panel."""

    for axis in axes:
        axis.title.set_fontsize(DEMO_FONT_SIZE)
        axis.xaxis.label.set_fontsize(DEMO_FONT_SIZE)
        axis.yaxis.label.set_fontsize(DEMO_FONT_SIZE)
        axis.tick_params(labelsize=DEMO_FONT_SIZE - 2)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(DEMO_SUPTITLE_SIZE)


def _white_zero_cmap(name: str = "magma") -> ListedColormap:
    """Return a perceptually uniform map whose exact-zero bin is white."""

    colours = plt.get_cmap(name)(np.linspace(0, 1, 256))
    # Reserve a narrow visible bin for zero without disturbing the remainder
    # of the perceptually uniform sequential scale.
    colours[:3] = (1, 1, 1, 1)
    result = ListedColormap(colours, name=f"white_zero_{name}")
    result.set_bad("white")
    return result


def _add_colorbar(
    fig: plt.Figure,
    axes: plt.Axes | Sequence[plt.Axes],
    cmap: Any,
    norm: Normalize,
    label: str,
) -> Any:
    """Add the common compact vertical colour scale used by demo panels."""

    colourbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="vertical",
        fraction=0.047,
        pad=0.025,
    )
    colourbar.set_label(label)
    return colourbar


def _radial_power(images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    images = images.float()
    spectra = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1)).abs().square()
    spectrum = spectra.mean(dim=0)
    height, width = spectrum.shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    radius = torch.sqrt((yy - height // 2).float().square() + (xx - width // 2).float().square())
    bins = radius.floor().long()
    profile = torch.zeros(int(bins.max()) + 1)
    counts = torch.zeros_like(profile)
    profile.scatter_add_(0, bins.flatten(), spectrum.flatten())
    counts.scatter_add_(0, bins.flatten(), torch.ones_like(spectrum).flatten())
    profile /= counts.clamp_min(1)
    frequency = torch.arange(len(profile)) / min(height, width)
    return _numpy(frequency), _numpy(profile)


def input_effective_dimension(inputs: torch.Tensor) -> int:
    """Number of centered input PCs required for 95% variance."""

    matrix = inputs.float().flatten(1).clone()
    matrix -= matrix.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    ratio = singular.square() / singular.square().sum().clamp_min(1e-12)
    return int((ratio.cumsum(0) < 0.95).sum().item() + 1)


def plot_lgn_inputs_and_statistics(
    archive: MicrodomainArchive,
    input_indices: Sequence[int] = (0, 1, 2, 3),
) -> plt.Figure:
    """Natural-image LGN drive plus sparsity, intensity, and frequency summaries."""

    inputs = archive.representative["inputs"].float()[:, 0]
    chosen = inputs[list(input_indices)]
    zero_fraction = (inputs == 0).flatten(1).float().mean(1)
    means = inputs.flatten(1).mean(1)
    frequency, radial = _radial_power(inputs)
    lgn_dim = input_effective_dimension(inputs)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(4 * README_PANEL_WIDTH, README_TWO_ROW_HEIGHT),
        constrained_layout=True,
    )
    for column, image in enumerate(chosen):
        axes[0, column].imshow(image, cmap="gray", vmin=0, vmax=1)
        dataset_index = archive.representative["source_dataset_indices"][input_indices[column]]
        axes[0, column].set_title(
            f"Crop {input_indices[column]} · source {dataset_index}"
        )
        _image_axis(axes[0, column])

    axes[1, 0].hist(_numpy(inputs.flatten()), bins=50, color="black")
    axes[1, 0].set_xlabel("pixel activity")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].set_title("Input activity")

    axes[1, 1].hist(_numpy(zero_fraction), bins=20, color="black")
    axes[1, 1].axvline(float(zero_fraction.mean()), color="0.55", linestyle="--")
    axes[1, 1].set_xlabel("fraction exactly zero")
    axes[1, 1].set_title(f"Sparsity · mean={zero_fraction.mean():.2f}")

    axes[1, 2].scatter(_numpy(means), _numpy(zero_fraction), s=18, color="black", alpha=0.65)
    axes[1, 2].set_xlabel("crop mean")
    axes[1, 2].set_ylabel("fraction exactly zero")
    axes[1, 2].set_title("Intensity vs sparsity")

    axes[1, 3].plot(frequency[1:], radial[1:] / max(radial[1:].max(), 1e-12), color="black")
    axes[1, 3].set_xlabel("frequency (cycles/pixel)")
    axes[1, 3].set_ylabel("normalised power")
    axes[1, 3].set_yscale("log")
    axes[1, 3].set_title(f"LGN spectrum · $d_{{95}}$={lgn_dim}")
    for axis in axes[1]:
        _style_axis(axis)
    fig.suptitle("Fixed natural-image inputs to the model")
    _apply_demo_typography(fig, axes.flat)
    return fig


def run_macaque_displacement_demo(
    smoothing_sigma_um: float = 100.0,
    scoring_border_um: float = 0.0,
    max_displacement_um: float | None = 350.0,
    data_root: str | Path | None = None,
    n_maps: int = 3,
    grid_size: int = 256,
) -> list[MapAnalysis]:
    """Run the fixed-bandwidth Chen et al. macaque V1 comparison."""

    if data_root is None:
        data_root = (
            DEMO_DIRECTORY
            / "data"
            / "cellular_orientation_displacement"
            / "macaque_cfs"
        )
    return run_fixed_sigma_analysis(
        data_root,
        sigma_um=smoothing_sigma_um,
        scoring_border_um=scoring_border_um,
        max_displacement_um=max_displacement_um,
        n_maps=n_maps,
        grid_size=grid_size,
    )


def plot_macaque_displacement_summary(
    results: Sequence[MapAnalysis],
    figure_path: str | Path | None = None,
) -> plt.Figure:
    """Two-row demo layout: cellular scatters above smoothed maps."""

    return plot_fixed_sigma_summary(results, figure_path=figure_path)


def plot_macaque_displacement_links(
    results: Sequence[MapAnalysis],
    n_links: int = 20,
    seed: int = 7,
    map_alpha: float = 0.28,
    figure_path: str | Path | None = None,
) -> plt.Figure:
    """Sparse examples linking somata to exact smooth-map predictions."""

    return plot_sparse_displacement_links(
        results,
        n_links=n_links,
        seed=seed,
        map_alpha=map_alpha,
        figure_path=figure_path,
    )


def macaque_displacement_table(results: Sequence[MapAnalysis]):
    """Return the per-field displacement summary as a notebook-friendly table."""

    import pandas as pd

    table = pd.DataFrame(cellular_results_table(results))
    return table.rename(
        columns={
            "fov": "dataset",
            "n_tuned_cells": "total tuned cells",
            "n_exact_contours_found": "included cells",
            "smoothing_sigma_um": "Gaussian σ (µm)",
            "mean_displacement_um": "mean displacement (µm)",
        }
    )


def _draw_fishnet(
    axis: plt.Axes,
    retinotopy: torch.Tensor,
    size: int,
    linewidth: float = 0.38,
    zoom_fraction: float = 0.5,
) -> None:
    points = _numpy(retinotopy).reshape(size, size, 2)
    axis.add_collection(
        LineCollection(points, colors="black", linewidths=linewidth, alpha=0.8)
    )
    axis.add_collection(
        LineCollection(
            points.transpose(1, 0, 2),
            colors="black",
            linewidths=linewidth,
            alpha=0.8,
        )
    )
    if not 0 < zoom_fraction <= 1:
        raise ValueError("zoom_fraction must be in (0, 1].")
    x_min, x_max = float(points[..., 0].min()), float(points[..., 0].max())
    y_min, y_max = float(points[..., 1].min()), float(points[..., 1].max())
    x_mid, y_mid = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    x_half = 0.5 * zoom_fraction * (x_max - x_min)
    y_half = 0.5 * zoom_fraction * (y_max - y_min)
    axis.set_xlim(x_mid - x_half, x_mid + x_half)
    axis.set_ylim(y_mid + y_half, y_mid - y_half)
    axis.set_aspect("equal")
    _image_axis(axis)


def animate_map_learning(
    archive: MicrodomainArchive,
    n_animation_frames: int = 25,
    interval: int = 350,
) -> animation.FuncAnimation:
    """Orientation, horizontal retinotopy, zoomed fishnet, and Fourier ring."""

    indices = archive.sampled_frame_indices(n_animation_frames)
    size = int(archive.manifest["config"]["sheet_size"])
    input_size = int(archive.manifest["config"]["crop_size"])
    fourier_vmax = max(
        float(torch.log1p(archive.frame(index)["fourier_spectrum"].float()).max())
        for index in indices
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(4 * README_PANEL_WIDTH, README_ROW_HEIGHT),
        constrained_layout=True,
    )
    orientation_norm = Normalize(0, math.pi)
    retinotopy_norm = Normalize(0, input_size - 1)
    fourier_norm = Normalize(0, max(fourier_vmax, 1e-12))
    orientation_colourbar = _add_colorbar(
        fig, axes[0], plt.get_cmap("hsv"), orientation_norm, "orientation preference (rad)"
    )
    orientation_colourbar.set_ticks((0, math.pi / 2, math.pi), labels=("0", "π/2", "π"))
    _add_colorbar(
        fig, axes[1], plt.get_cmap("hsv"), retinotopy_norm, "retinotopic x-position (pixels)"
    )
    _add_colorbar(
        fig, axes[3], plt.get_cmap("Greys"), fourier_norm, "log(1 + Fourier power)"
    )

    def update(position: int):
        frame = archive.frame(indices[position])
        for axis in axes:
            axis.clear()
        axes[0].imshow(frame["orientation_rad"], cmap="hsv", norm=orientation_norm)
        axes[0].set_title("Learned orientation preference")
        _image_axis(axes[0])

        retinotopy = frame["retinotopy_xy_pixels"].float()
        axes[1].imshow(
            retinotopy[:, 0].reshape(size, size), cmap="hsv", norm=retinotopy_norm
        )
        axes[1].set_title("Horizontal retinotopy")
        _image_axis(axes[1])

        _draw_fishnet(axes[2], retinotopy, size)
        axes[2].set_title("Zoomed retinotopic fishnet")

        axes[3].imshow(
            torch.log1p(frame["fourier_spectrum"].float()),
            cmap="Greys",
            norm=fourier_norm,
        )
        axes[3].set_title(
            f"Orientation Fourier power · period={frame['fourier_period_pixels']:.1f} px"
        )
        _image_axis(axes[3])
        fig.suptitle(
            f"Map formation · epoch {archive.epoch_at(frame):.2f} · snapshot {indices[position]:02d}"
        )
        _apply_demo_typography(fig, fig.axes)
        return []

    result = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=interval, repeat=True, blit=False
    )
    result._draw_was_started = True
    plt.close(fig)
    return result


def _tile_mosaic(images: torch.Tensor, rows: int = 5, columns: int = 5) -> np.ndarray:
    images = images.detach().float().cpu()
    height, width = images.shape[-2:]
    mosaic = torch.full((rows * height, columns * width), float("nan"))
    for index, image in enumerate(images[: rows * columns]):
        row, column = divmod(index, columns)
        mosaic[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return _numpy(mosaic)


def _centred_square_mosaic(images: torch.Tensor, canvas_tiles: int = 5) -> np.ndarray:
    """Centre the largest complete square of samples on a fixed tile canvas."""

    images = images.detach().float().cpu()
    grid = int(math.ceil(math.sqrt(len(images))))
    height, width = images.shape[-2:]
    canvas = torch.full(
        (canvas_tiles * height, canvas_tiles * width), float("nan")
    )
    offset_y = (canvas.shape[0] - grid * height) // 2
    offset_x = (canvas.shape[1] - grid * width) // 2
    for index, image in enumerate(images[: grid * grid]):
        row, column = divmod(index, grid)
        top = offset_y + row * height
        left = offset_x + column * width
        canvas[top : top + height, left : left + width] = image
    return _numpy(canvas)


def _normalise_receptive_fields(fields: torch.Tensor) -> torch.Tensor:
    """Match NeuralSheet-style RF display with independent zero-centred contrast."""

    fields = fields.float()
    fields = fields - fields.mean(dim=(-2, -1), keepdim=True)
    scale = fields.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return fields / scale


def _normalise_nonnegative_fields(fields: torch.Tensor) -> torch.Tensor:
    """Scale each non-negative field to [0, 1] while retaining exact zeros."""

    fields = fields.float().clamp_min(0)
    scale = fields.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return fields / scale


def _border_safe_connection_crops(
    fields: torch.Tensor,
    source_indices: Sequence[int],
    crop_size: int = 21,
) -> tuple[torch.Tensor, list[int]]:
    """Select a square set of fields whose centred crops never touch padding."""

    if crop_size % 2 != 1:
        raise ValueError("crop_size must be odd.")
    fields = fields.float()
    sheet_size = int(fields.shape[-1])
    radius = crop_size // 2
    crops = []
    selected_sources: list[int] = []
    for field, source in zip(fields, source_indices):
        row, column = divmod(int(source), sheet_size)
        if not (
            radius <= row < sheet_size - radius
            and radius <= column < sheet_size - radius
        ):
            continue
        crops.append(
            field[
                row - radius : row + radius + 1,
                column - radius : column + radius + 1,
            ]
        )
        selected_sources.append(int(source))
    grid_size = int(math.floor(math.sqrt(len(crops))))
    if grid_size < 1:
        raise RuntimeError("No border-safe archived cross-domain fields are available.")
    keep = grid_size**2
    return torch.stack(crops[:keep]), selected_sources[:keep]


def animate_weight_learning(
    archive: MicrodomainArchive,
    n_animation_frames: int = 25,
    interval: int = 350,
    lateral_crop_size: int = 21,
) -> animation.FuncAnimation:
    """Fixed afferent fields and complete cross-domain excitation crops."""

    indices = archive.sampled_frame_indices(n_animation_frames)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2 * README_PANEL_WIDTH, README_ROW_HEIGHT),
        constrained_layout=True,
    )
    field_cmap = _white_zero_cmap("magma")
    field_norm = Normalize(0, 1)
    _add_colorbar(
        fig, list(axes), field_cmap, field_norm, "normalised connection weight"
    )

    def update(position: int):
        frame = archive.frame(indices[position])
        afferent = _normalise_nonnegative_fields(
            frame["sampled_afferent_weights"].float()[:, 0]
        )
        cross_domain, selected_sources = _border_safe_connection_crops(
            frame["sampled_lateral_exc_effective"],
            archive.manifest["lateral_sample_indices"],
            crop_size=lateral_crop_size,
        )
        cross_domain = _normalise_nonnegative_fields(cross_domain)
        cross_domain_grid = int(math.sqrt(len(cross_domain)))
        for axis in axes:
            axis.clear()
        axes[0].imshow(
            _tile_mosaic(afferent, 5, 5),
            cmap=field_cmap,
            norm=field_norm,
        )
        axes[0].set_title("25 fixed sampled afferent fields")
        axes[1].imshow(
            _tile_mosaic(
                cross_domain,
                rows=cross_domain_grid,
                columns=cross_domain_grid,
            ),
            cmap=field_cmap,
            norm=field_norm,
        )
        axes[1].set_title(
            f"{len(selected_sources)} cross-domain excitation (CDE) fields"
        )
        for axis in axes:
            _image_axis(axis)
        fig.suptitle(f"Weight learning · epoch {archive.epoch_at(frame):.2f}")
        _apply_demo_typography(fig, fig.axes)
        return []

    result = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=interval, repeat=True, blit=False
    )
    result._draw_was_started = True
    plt.close(fig)
    return result


def make_synthetic_inputs(
    reference_inputs: torch.Tensor,
    feature_smoothness: float = 0.075,
    fourier_cutoff: float = 0.12,
) -> dict[str, Any]:
    """Create the sparse smiley/neuron stimuli formerly defined in the notebook."""

    reference_inputs = reference_inputs.float()
    size = int(reference_inputs.shape[-1])
    target_zero_fraction = float(
        (reference_inputs == 0).flatten(1).float().mean(1).max()
    )
    target_mean = float(reference_inputs.mean())
    coordinates = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def gaussian_stroke(distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * (distance / feature_smoothness).square())

    def soft_ellipse(cx: float, cy: float, rx: float, ry: float) -> torch.Tensor:
        normalized = torch.sqrt(((xx - cx) / rx).square() + ((yy - cy) / ry).square())
        distance = (normalized - 1) * min(rx, ry)
        edge = ((feature_smoothness - distance) / (2 * feature_smoothness)).clamp(0, 1)
        return edge.square() * (3 - 2 * edge)

    def segment_distance(ax: float, ay: float, bx: float, by: float) -> torch.Tensor:
        vx, vy = bx - ax, by - ay
        position = (((xx - ax) * vx + (yy - ay) * vy) / (vx * vx + vy * vy)).clamp(0, 1)
        return torch.sqrt((xx - ax - position * vx).square() + (yy - ay - position * vy).square())

    def match(raw: torch.Tensor) -> torch.Tensor:
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
        matched = scaled.pow(0.5 * (low + high))
        matched[scaled == 0] = 0
        return matched

    radius = torch.sqrt(xx.square() + yy.square())
    mouth_curve = 0.34 - 0.85 * xx.square()
    mouth_distance = torch.sqrt(
        (yy - mouth_curve).square() + (xx.abs() - 0.42).clamp_min(0).square()
    )
    smiley = torch.stack(
        (
            gaussian_stroke((radius - 0.88).abs()),
            soft_ellipse(-0.30, -0.25, 0.165, 0.125),
            soft_ellipse(0.30, -0.25, 0.165, 0.125),
            gaussian_stroke(mouth_distance),
        )
    ).amax(0)

    angle = -0.60
    cosine, sine = math.cos(angle), math.sin(angle)

    def rotate(point_x: float, point_y: float) -> tuple[float, float]:
        return cosine * point_x - sine * point_y, sine * point_x + cosine * point_y

    def rotated_segment(ax: float, ay: float, bx: float, by: float) -> torch.Tensor:
        ax, ay = rotate(ax, ay)
        bx, by = rotate(bx, by)
        return segment_distance(ax, ay, bx, by)

    soma_x, soma_y = rotate(-0.24, 0.0)
    neuron = torch.stack(
        (
            soft_ellipse(soma_x, soma_y, 0.20, 0.20),
            gaussian_stroke(rotated_segment(-0.04, 0.00, 0.55, 0.00)),
            gaussian_stroke(rotated_segment(0.55, 0.00, 0.90, -0.34)),
            gaussian_stroke(rotated_segment(0.55, 0.00, 0.90, 0.34)),
            gaussian_stroke(rotated_segment(-0.40, -0.08, -0.82, -0.48)),
            gaussian_stroke(rotated_segment(-0.40, 0.08, -0.82, 0.48)),
            gaussian_stroke(rotated_segment(-0.24, -0.18, -0.18, -0.80)),
        )
    ).amax(0)
    originals = torch.stack((match(smiley), match(neuron)))

    frequency = torch.fft.fftfreq(size)
    fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
    ring = torch.sqrt(fx.square() + fy.square())
    transition_start = 0.8 * fourier_cutoff
    transition = ((ring - transition_start) / max(fourier_cutoff - transition_start, 1e-12)).clamp(0, 1)
    mask = 0.5 * (1 + torch.cos(torch.pi * transition))
    mask[ring >= fourier_cutoff] = 0
    filtered = torch.fft.ifft2(torch.fft.fft2(originals) * mask).real
    filtered -= filtered.amin(dim=(-2, -1), keepdim=True)
    filtered /= filtered.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return {
        "names": ("Smiley face", "Simple neuron"),
        "normal": originals,
        "fourier": filtered,
        "fourier_cutoff": fourier_cutoff,
        "target_zero_fraction": target_zero_fraction,
        "target_mean": target_mean,
    }


def evaluate_synthetic_final(
    archive: MicrodomainArchive,
    device: str = "cuda",
    activity_scale: float = 1.0,
    fourier_cutoff: float = 0.12,
    cache: bool = True,
) -> dict[str, Any]:
    """Evaluate face/neuron once with the final model and optionally cache CPU arrays."""

    cache_path = archive.root / "synthetic_final.pt"
    if cache and cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    reference_indices = archive.representative["reconstruction_indices"]
    synthetic = make_synthetic_inputs(
        archive.representative["inputs"][reference_indices],
        fourier_cutoff=fourier_cutoff,
    )
    model, decoder, _ = load_l4_demo_checkpoint(
        archive.root / "final_l4_checkpoint.pt", device=device
    )
    decoder["model"].eval()
    inputs = torch.cat((synthetic["normal"], synthetic["fourier"]))
    activities = []
    trajectories = []
    with torch.no_grad():
        for image in inputs:
            model(
                image[None, None].to(device),
                adaptation=False,
                noise_gamma=0,
                layer_3=False,
                track_response=True,
            )
            activities.append(model.current_response.detach().cpu() * activity_scale)
            trajectories.append(model.response_tracker.detach().cpu() * activity_scale)
        activities = torch.cat(activities)
        reconstructions = decoder["activ"](
            decoder["model"](activities.to(device))
        ).detach().cpu()
    scores = F.cosine_similarity(inputs.flatten(1), reconstructions[:, 0].flatten(1), dim=1)
    result = {
        **synthetic,
        "activities": activities,
        "trajectories": torch.stack(trajectories),
        "reconstructions": reconstructions,
        "cosine": scores,
        "activity_scale": activity_scale,
    }
    if cache:
        torch.save(result, cache_path)
    return result


def animate_synthetic_learning(
    archive: MicrodomainArchive,
    synthetic_final: dict[str, Any] | None = None,
    n_animation_frames: int = 25,
    interval: int = 350,
) -> animation.FuncAnimation:
    """Genuinely tracked face input, activity, and reconstruction through learning."""

    indices = archive.sampled_frame_indices(n_animation_frames)
    # ``synthetic_final`` is retained only for compatibility with the first
    # public-demo draft; the untracked neuron row has intentionally been removed.
    del synthetic_final
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(3 * README_PANEL_WIDTH, README_ROW_HEIGHT),
        constrained_layout=True,
    )
    face_input = archive.representative["tracked_inputs"]["smiley_face"][0].float()
    grey_cmap = plt.get_cmap("Greys")
    image_norm = Normalize(0, 1)
    activity_vmax = max(
        float(
            archive.frame(index)["tracked_l4_activities"]["smiley_face"][0, 0]
            .float()
            .quantile(0.995)
        )
        for index in indices
    )
    activity_norm = Normalize(0, max(activity_vmax, 1e-12))
    _add_colorbar(
        fig,
        [axes[0], axes[2]],
        grey_cmap,
        image_norm,
        "LGN input / reconstruction activity",
    )
    _add_colorbar(fig, axes[1], grey_cmap, activity_norm, "L4 activity")

    def update(position: int):
        frame = archive.frame(indices[position])
        for axis in axes:
            axis.clear()
        face_activity = frame["tracked_l4_activities"]["smiley_face"][0, 0].float()
        face_reco = frame["tracked_reconstructions"]["smiley_face"][0, 0].float()
        face_score = frame["tracked_reconstruction_cosine"]["smiley_face"]
        axes[0].imshow(face_input, cmap=grey_cmap, norm=image_norm)
        axes[1].imshow(face_activity, cmap=grey_cmap, norm=activity_norm)
        axes[2].imshow(face_reco, cmap=grey_cmap, norm=image_norm)
        axes[0].set_title("Face input")
        axes[1].set_title(f"L4 activity · mean={face_activity.mean():.3f}")
        axes[2].set_title(f"Reconstruction · cosine={face_score:.3f}")
        for axis in axes:
            _image_axis(axis)
        fig.suptitle(f"Synthetic probes · learning epoch {archive.epoch_at(frame):.2f}")
        _apply_demo_typography(fig, fig.axes)
        return []

    result = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=interval, repeat=True, blit=False
    )
    result._draw_was_started = True
    plt.close(fig)
    return result


def _dimensionality_series(
    archive: MicrodomainArchive,
) -> tuple[np.ndarray, np.ndarray, int]:
    summaries = archive.manifest["frames"]
    epochs = np.asarray([item["seen"] / archive.dataset_size for item in summaries])
    v1_dim = np.asarray([item["pca_effective_dim_95"] for item in summaries])
    lgn_dim = input_effective_dimension(archive.representative["inputs"])
    return epochs, v1_dim, lgn_dim


def animate_dimensionality_learning(
    archive: MicrodomainArchive,
    n_animation_frames: int = 25,
    interval: int = 350,
    component_indices: Sequence[int] = (0, 1, 2),
) -> animation.FuncAnimation:
    """Three spatial PCA components followed by the evolving dimensionality."""

    if len(component_indices) != 3:
        raise ValueError("component_indices must contain exactly three components.")
    indices = archive.sampled_frame_indices(n_animation_frames)
    epochs, v1_dim, lgn_dim = _dimensionality_series(archive)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(4 * README_PANEL_WIDTH, README_ROW_HEIGHT),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1, 1, 1, 1.2), "wspace": 0.16},
    )
    component_cmap = plt.get_cmap("viridis")
    component_norm = Normalize(0, 1)
    _add_colorbar(
        fig,
        list(axes[:3]),
        component_cmap,
        component_norm,
        "normalised absolute PCA loading",
    )

    def update(position: int):
        frame_index = indices[position]
        frame = archive.frame(frame_index)
        for axis in axes:
            axis.clear()

        components = frame["pca_components"][list(component_indices)].float().abs()
        components /= components.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        ratios = frame["pca_explained_variance_ratio"][list(component_indices)].float()
        for column, (component_index, component) in enumerate(
            zip(component_indices, components)
        ):
            axes[column].imshow(component, cmap=component_cmap, norm=component_norm)
            axes[column].set_title(
                f"PC {component_index + 1} · variance={ratios[column]:.3f}"
            )
            _image_axis(axes[column])

        axis = axes[3]
        axis.plot(
            epochs[: frame_index + 1],
            v1_dim[: frame_index + 1],
            color="black",
            linewidth=2.5,
        )
        axis.axhline(lgn_dim, color="0.45", linestyle="--")
        axis.scatter(epochs[frame_index], v1_dim[frame_index], color="black", zorder=3)
        ratio = v1_dim[frame_index] / max(1, lgn_dim)
        axis.text(
            0.97,
            0.82,
            f"V1/LGN={ratio:.2f}\nV1 $d_{{95}}$={v1_dim[frame_index]}\nLGN $d_{{95}}$={lgn_dim}",
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=DEMO_FONT_SIZE,
        )
        axis.set_xlabel("training epoch")
        axis.set_ylabel("components for 95% variance")
        axis.set_xlim(float(epochs.min()), float(epochs.max()))
        padding = max(2.0, 0.05 * float(max(v1_dim.max(), lgn_dim)))
        axis.set_ylim(
            float(min(v1_dim.min(), lgn_dim) - padding),
            float(max(v1_dim.max(), lgn_dim) + padding),
        )
        axis.set_title("Relative representational dimensionality")
        _style_axis(axis)
        fig.suptitle(f"PCA geometry · epoch {archive.epoch_at(frame):.2f}")
        _apply_demo_typography(fig, fig.axes)
        return []

    result = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=interval, repeat=True, blit=False
    )
    result._draw_was_started = True
    plt.close(fig)
    return result


def animate_robustness_learning(
    archive: MicrodomainArchive,
    n_animation_frames: int = 25,
    interval: int = 350,
    activity_example: int = 0,
    activity_window_size: int = 20,
) -> animation.FuncAnimation:
    """Input, zoomed clean/noisy responses, and population stability."""

    indices = archive.sampled_frame_indices(n_animation_frames)
    if activity_window_size < 1:
        raise ValueError("activity_window_size must be positive.")
    summaries = archive.manifest["frames"]
    epochs = np.asarray([item["seen"] / archive.dataset_size for item in summaries])
    stability = np.asarray([item["stability_mean"] for item in summaries])
    input_index = archive.representative["robustness_indices"][activity_example]
    robustness_input = archive.representative["inputs"][input_index, 0].float()
    final_clean = archive.frame(indices[-1])["clean_settled_states"][activity_example, 0].float()
    height, width = final_clean.shape
    if activity_window_size > min(height, width):
        raise ValueError("activity_window_size exceeds the activity sheet.")
    window_scores = F.conv2d(
        final_clean[None, None],
        torch.ones(1, 1, activity_window_size, activity_window_size),
    )[0, 0]
    top = int(window_scores.argmax() // window_scores.shape[1])
    left = int(window_scores.argmax() % window_scores.shape[1])
    row_slice = slice(top, top + activity_window_size)
    column_slice = slice(left, left + activity_window_size)
    activity_vmax = max(
        float(
            torch.stack(
                (
                    archive.frame(index)["clean_settled_states"][activity_example, 0][row_slice, column_slice],
                    archive.frame(index)["noisy_settled_states"][activity_example, 0][row_slice, column_slice],
                )
            )
            .float()
            .quantile(0.995)
        )
        for index in indices
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(4 * README_PANEL_WIDTH, README_ROW_HEIGHT),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1, 1, 1, 1.2), "wspace": 0.2},
    )
    grey_cmap = plt.get_cmap("Greys")
    input_norm = Normalize(0, 1)
    activity_norm = Normalize(0, max(activity_vmax, 1e-12))
    _add_colorbar(fig, axes[0], grey_cmap, input_norm, "LGN input activity")
    _add_colorbar(
        fig, list(axes[1:3]), grey_cmap, activity_norm, "L4 activity"
    )

    def update(position: int):
        frame_index = indices[position]
        frame = archive.frame(frame_index)
        for axis in axes:
            axis.clear()
        clean = frame["clean_settled_states"][activity_example, 0][row_slice, column_slice].float()
        noisy = frame["noisy_settled_states"][activity_example, 0][row_slice, column_slice].float()
        axes[0].imshow(robustness_input, cmap=grey_cmap, norm=input_norm)
        axes[1].imshow(clean, cmap=grey_cmap, norm=activity_norm)
        axes[2].imshow(noisy, cmap=grey_cmap, norm=activity_norm)
        axes[0].set_title("Fixed LGN input")
        axes[1].set_title(f"Clean activity · {activity_window_size}×{activity_window_size} zoom")
        axes[2].set_title(
            f"Noise intensity={frame['noise_gamma']:.2f} · {activity_window_size}×{activity_window_size} zoom"
        )
        for axis in axes[:3]:
            _image_axis(axis)

        axes[3].plot(
            epochs[: frame_index + 1],
            stability[: frame_index + 1],
            color="black",
            linewidth=2.5,
        )
        axes[3].scatter(
            epochs[frame_index], stability[frame_index], color="black", zorder=3
        )
        axes[3].set_xlim(float(epochs.min()), float(epochs.max()))
        axes[3].set_ylim(0, 1)
        axes[3].set_xlabel("training epoch")
        axes[3].set_ylabel("clean/noisy cosine similarity")
        axes[3].set_title(f"Population stability={frame['stability_mean']:.3f}")
        _style_axis(axes[3])
        fig.suptitle(f"Noise robustness · epoch {archive.epoch_at(frame):.2f}")
        _apply_demo_typography(fig, fig.axes)
        return []

    result = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=interval, repeat=True, blit=False
    )
    result._draw_was_started = True
    plt.close(fig)
    return result


@lru_cache(maxsize=32)
def gaussian_std_for_mean_displacement(
    size: int,
    target_displacement: float = 2.0,
    seed: int = 7,
    calibration_steps: int = 12,
) -> tuple[float, float]:
    """Calibrate Gaussian swap width to a requested mean cell displacement."""

    target_displacement = float(target_displacement)
    if target_displacement < 0:
        raise ValueError("target_displacement must be non-negative.")
    if target_displacement == 0:
        return 0.0, 0.0
    template = torch.zeros(int(size), int(size))

    def displacement_at(std: float) -> float:
        return gaussian_local_permutation(template, std, seed=seed)[2]

    low, high = 0.0, max(0.25, target_displacement)
    high_displacement = displacement_at(high)
    while high_displacement < target_displacement and high < 4 * size:
        high *= 2
        high_displacement = displacement_at(high)
    if high_displacement < target_displacement:
        raise ValueError(
            f"Could not reach mean displacement {target_displacement:g} for size {size}."
        )
    for _ in range(calibration_steps):
        midpoint = 0.5 * (low + high)
        if displacement_at(midpoint) < target_displacement:
            low = midpoint
        else:
            high = midpoint
    std = 0.5 * (low + high)
    return std, displacement_at(std)


def animate_scattered_map_learning(
    archive: MicrodomainArchive,
    mean_displacement: float = 2.0,
    scatter_seed: int = 7,
    n_animation_frames: int = 25,
    interval: int = 350,
) -> animation.FuncAnimation:
    """Apply one permutation to orientation and horizontal retinotopy."""

    indices = archive.sampled_frame_indices(n_animation_frames)
    # The seed fixes one anatomical permutation, reused for every learning frame.
    template = archive.frame(0)["orientation_rad"].float()
    scatter_std, calibrated_displacement = gaussian_std_for_mean_displacement(
        int(template.shape[-1]),
        target_displacement=float(mean_displacement),
        seed=scatter_seed,
    )
    _, permutation, displacement = gaussian_local_permutation(
        template, scatter_std, seed=scatter_seed
    )
    size = int(template.shape[-1])
    input_size = int(archive.manifest["config"]["crop_size"])
    scattered_spectra = []
    for index in indices:
        orientation = archive.frame(index)["orientation_rad"].float()
        scattered = orientation.flatten()[permutation].reshape_as(orientation)
        _, spectrum, _ = get_typical_dist_fourier(scattered, mask=1)
        scattered_spectra.append(torch.log1p(spectrum.float()))
    fourier_vmax = max(float(spectrum.max()) for spectrum in scattered_spectra)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(4 * README_PANEL_WIDTH, README_ROW_HEIGHT),
        constrained_layout=True,
    )
    orientation_norm = Normalize(0, math.pi)
    retinotopy_norm = Normalize(0, input_size - 1)
    fourier_norm = Normalize(0, max(fourier_vmax, 1e-12))
    orientation_colourbar = _add_colorbar(
        fig, axes[0], plt.get_cmap("hsv"), orientation_norm, "orientation preference (rad)"
    )
    orientation_colourbar.set_ticks((0, math.pi / 2, math.pi), labels=("0", "π/2", "π"))
    _add_colorbar(
        fig, axes[1], plt.get_cmap("hsv"), retinotopy_norm, "retinotopic x-position (pixels)"
    )
    _add_colorbar(
        fig, axes[3], plt.get_cmap("Greys"), fourier_norm, "log(1 + Fourier power)"
    )

    def update(position: int):
        frame = archive.frame(indices[position])
        for axis in axes:
            axis.clear()
        orientation = frame["orientation_rad"].float()
        scattered = orientation.flatten()[permutation].reshape_as(orientation)
        _, spectrum, _ = get_typical_dist_fourier(scattered, mask=1)
        retinotopy = frame["retinotopy_xy_pixels"].float()
        scattered_retinotopy = retinotopy[permutation]

        axes[0].imshow(scattered, cmap="hsv", norm=orientation_norm)
        axes[0].set_title(
            f"Scattered orientation · displacement={displacement:.1f} cells"
        )
        axes[1].imshow(
            scattered_retinotopy[:, 0].reshape(size, size),
            cmap="hsv",
            norm=retinotopy_norm,
        )
        axes[1].set_title("Horizontal retinotopy after scatter")
        _draw_fishnet(axes[2], scattered_retinotopy, size, linewidth=0.12)
        axes[2].set_title("Zoomed retinotopic fishnet after scatter")
        axes[3].imshow(
            torch.log1p(spectrum.float()), cmap="Greys", norm=fourier_norm
        )
        axes[3].set_title("Fourier power after scatter")
        _image_axis(axes[0])
        _image_axis(axes[1])
        _image_axis(axes[3])
        fig.suptitle(
            f"Gaussian local permutation · target={mean_displacement:g}, "
            f"calibrated={calibrated_displacement:.2f} cells · "
            f"epoch {archive.epoch_at(frame):.2f}"
        )
        _apply_demo_typography(fig, fig.axes)
        return []

    result = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=interval, repeat=True, blit=False
    )
    result._draw_was_started = True
    plt.close(fig)
    return result


def load_rotating_umap_data(
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load the compact four-panel UMAP cache copied into the demo bundle."""

    path = DEMO_UMAP_DATA if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing rotating UMAP cache: {path}")
    cache = np.load(path, allow_pickle=False)
    panels = []
    index = 0
    while f"embedding_{index}" in cache:
        embedding = np.asarray(cache[f"embedding_{index}"], dtype=np.float32)
        angles = np.asarray(cache[f"angles_{index}"], dtype=np.float32)
        if embedding.ndim != 2 or embedding.shape[1] != 3:
            raise ValueError(f"embedding_{index} must have shape [samples, 3]")
        if len(embedding) != len(angles):
            raise ValueError(f"embedding_{index} and angles_{index} disagree")
        panels.append(
            {
                "title": str(cache[f"title_{index}"]),
                "embedding": embedding,
                "angles": angles,
            }
        )
        index += 1
    if len(panels) != 4:
        raise ValueError(f"Expected four UMAP panels, found {len(panels)}")
    return panels


def animate_rotating_umap_grid(
    panels: Sequence[dict[str, Any]] | None = None,
    n_animation_frames: int = 48,
    interval: int = 120,
    elevation_deg: float = 35.264,
    initial_azimuth_deg: float = 45.0,
) -> animation.FuncAnimation:
    """Rotate static-grating, model, and Stringer-recording-1 UMAPs about z."""

    panels = load_rotating_umap_data() if panels is None else list(panels)
    if len(panels) != 4:
        raise ValueError("panels must contain exactly four embeddings")
    if n_animation_frames < 2:
        raise ValueError("n_animation_frames must be at least 2")
    display_titles = (
        "Static gratings",
        "Topographic simulation",
        "Salt-and-pepper simulation",
        "Stringer recording 1",
    )
    figure = plt.figure(
        figsize=(4 * README_PANEL_WIDTH, README_ROW_HEIGHT),
        constrained_layout=True,
    )
    axes = [
        figure.add_subplot(1, 4, index + 1, projection="3d")
        for index in range(4)
    ]
    colormap = plt.get_cmap("hsv")
    normalization = Normalize(0, 180)
    for axis, panel, title in zip(axes, panels, display_titles):
        embedding = np.asarray(panel["embedding"], dtype=float)
        angles = np.asarray(panel["angles"], dtype=float) % 180
        axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=angles,
            cmap=colormap,
            norm=normalization,
            s=4.0,
            alpha=0.76,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        lower = embedding.min(axis=0)
        upper = embedding.max(axis=0)
        padding = 0.05 * np.maximum(upper - lower, 1e-12)
        axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        axis.set_zlim(lower[2] - padding[2], upper[2] + padding[2])
        # Plotly's source cell uses ``aspectmode='data'``: preserve the
        # relative scale of the three fitted UMAP coordinates instead of
        # stretching every axis to an equal visual length.
        axis.set_box_aspect(np.maximum(upper - lower, 1e-12))
        axis.set_axis_off()
        axis.set_title(title, pad=4)
    colourbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=colormap),
        ax=axes,
        orientation="vertical",
        ticks=(0, 45, 90, 135, 180),
        fraction=0.025,
        pad=0.01,
        shrink=0.72,
    )
    colourbar.set_label("grating orientation (degrees; axial)")
    figure.suptitle("Rotating 3D population-response UMAPs")
    _apply_demo_typography(figure, figure.axes)
    azimuths = initial_azimuth_deg + np.linspace(
        0, 360, n_animation_frames, endpoint=False
    )

    def update(position: int):
        for axis in axes:
            axis.view_init(elev=elevation_deg, azim=float(azimuths[position]))
        return []

    result = animation.FuncAnimation(
        figure,
        update,
        frames=n_animation_frames,
        interval=interval,
        repeat=True,
        blit=False,
    )
    result._draw_was_started = True
    plt.close(figure)
    return result


def animation_html(
    result: animation.FuncAnimation,
    embed_limit_mb: float = 100.0,
):
    """Convert a complete animation to an inline object with no global state."""

    from IPython.display import HTML

    with plt.rc_context({"animation.embed_limit": embed_limit_mb}):
        return HTML(result.to_jshtml())


def save_animation(
    result: animation.FuncAnimation,
    path: str | Path,
    fps: float = 3.0,
    dpi: int = 80,
) -> Path:
    """Save a compact GIF that remains visible in GitHub's notebook renderer."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    return path


def render_github_assets(
    archive: MicrodomainArchive,
    output_dir: str | Path | None = None,
    n_animation_frames: int = 20,
    fps: float = 3.0,
    dpi: int = 65,
    scatter_displacement: float = 2.0,
) -> dict[str, Path]:
    """Render every tracked asset used by the merged GitHub notebook."""

    output_dir = DEMO_ASSET_DIRECTORY if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "lgn_inputs.png"
    input_figure = plot_lgn_inputs_and_statistics(archive)
    input_figure.savefig(input_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(input_figure)

    cellular_results = run_macaque_displacement_demo(
        smoothing_sigma_um=100.0,
        scoring_border_um=0.0,
        max_displacement_um=350.0,
    )
    cellular_summary_path = output_dir / "macaque_displacement_summary.png"
    cellular_summary = plot_macaque_displacement_summary(
        cellular_results,
        figure_path=cellular_summary_path,
    )
    plt.close(cellular_summary)
    cellular_links_path = output_dir / "macaque_displacement_links.png"
    cellular_links = plot_macaque_displacement_links(
        cellular_results,
        figure_path=cellular_links_path,
    )
    plt.close(cellular_links)

    animations = {
        "map_learning": animate_map_learning(archive, n_animation_frames),
        "weight_learning": animate_weight_learning(archive, n_animation_frames),
        "synthetic_learning": animate_synthetic_learning(archive, n_animation_frames=n_animation_frames),
        "dimensionality": animate_dimensionality_learning(
            archive, n_animation_frames
        ),
        "robustness": animate_robustness_learning(archive, n_animation_frames),
        "scattered_learning": animate_scattered_map_learning(
            archive,
            mean_displacement=scatter_displacement,
            n_animation_frames=n_animation_frames,
        ),
        "rotating_umap": animate_rotating_umap_grid(
            n_animation_frames=max(48, n_animation_frames),
        ),
    }
    paths: dict[str, Path] = {
        "lgn_inputs": input_path,
        "macaque_displacement_summary": cellular_summary_path,
        "macaque_displacement_links": cellular_links_path,
    }
    for name, result in animations.items():
        paths[name] = save_animation(
            result,
            output_dir / f"{name}.gif",
            fps=6.0 if name == "rotating_umap" else fps,
            dpi=dpi,
        )
    return paths


def ensure_github_assets(
    archive: MicrodomainArchive,
    output_dir: str | Path | None = None,
    force: bool = False,
    **render_kwargs: Any,
) -> dict[str, Path]:
    """Return shipped assets, rendering them only when missing or forced."""

    output_dir = DEMO_ASSET_DIRECTORY if output_dir is None else Path(output_dir)
    names = (
        "lgn_inputs.png",
        "map_learning.gif",
        "weight_learning.gif",
        "synthetic_learning.gif",
        "dimensionality.gif",
        "robustness.gif",
        "scattered_learning.gif",
        "macaque_displacement_summary.png",
        "macaque_displacement_links.png",
        "rotating_umap.gif",
    )
    expected = {name.rsplit(".", 1)[0]: output_dir / name for name in names}
    if force or not all(path.exists() for path in expected.values()):
        return render_github_assets(
            archive, output_dir=output_dir, **render_kwargs
        )
    return expected


def final_reconstruction_summary(archive: MicrodomainArchive) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot all fixed final reconstructions and report the best representative."""

    frame = archive.frame(-1)
    indices = archive.representative["reconstruction_indices"]
    inputs = archive.representative["inputs"][indices, 0].float()
    reconstructions = frame["reconstructions"][:, 0].float()
    cosine = frame["reconstruction_cosine"].float()
    relative = frame["reconstruction_relative"].float()
    best = int(cosine.argmax())
    fig, axes = plt.subplots(
        2,
        len(indices),
        figsize=(3.0 * len(indices), README_TWO_ROW_HEIGHT),
        constrained_layout=True,
    )
    for column in range(len(indices)):
        axes[0, column].imshow(inputs[column], cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(reconstructions[column], cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"Input {column}")
        axes[1, column].set_title(f"cos={cosine[column]:.3f}\nrelative={relative[column]:.3f}")
        for axis in axes[:, column]:
            _image_axis(axis)
            if column == best:
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("tab:green")
                    spine.set_linewidth(2.5)
    summary = {
        "best_example": best,
        "representative_index": int(indices[best]),
        "dataset_index": int(
            archive.representative["source_dataset_indices"][indices[best]]
        ),
        "cosine": float(cosine[best]),
        "baseline_relative": float(relative[best]),
    }
    fig.suptitle("Final fixed-input reconstructions", fontsize=15)
    return fig, summary

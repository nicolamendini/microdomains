"""Plot the saved paper data and derive lightweight spatial readouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .map_plotting import (
    plot_orientation_histogram,
    plot_orientation_periodicity_profile,
    plot_patchiness_profile,
    plot_precomputed_umap_with_angles_3d,
    plot_receptive_field_mosaic,
    plot_retinotopy_grid,
    plot_sparse_tensor,
    plot_wiring_efficiency_six_panel_summary,
)


BASE = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = BASE / "paper_results" / "paper_panel_data.pt"
DEFAULT_MANIFEST = BASE / "paper_results" / "paper_run.json"


def plot_core_sweep_summary(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    show: bool = True,
):
    """Plot the six fidelity, complexity, and robustness sweep panels."""

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run run_all_paper_experiments.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    outputs = manifest["core_outputs"]
    def data_path(name: str) -> Path:
        path = Path(outputs[name])
        return path if path.is_absolute() else BASE / path

    accdim = torch.load(data_path("accdim"), map_location="cpu", weights_only=False)
    null_accuracy = float(accdim["null_accuracy_mean"])
    input_dimensionality = int(accdim["input_pca_dimensionality"])
    fidelity = (
        torch.as_tensor(accdim["heldout_accuracy"], dtype=torch.float32)
        - null_accuracy
    ) / (1.0 - null_accuracy)
    dimensionality = (
        torch.as_tensor(accdim["se_pca_tracker"], dtype=torch.float32)
        / input_dimensionality
    )
    neighborhood_size = (
        torch.as_tensor(accdim["trialvar"], dtype=torch.float32) / 3.0
    ).square() * math.pi
    return plot_wiring_efficiency_six_panel_summary(
        neighborhood_size,
        fidelity,
        dimensionality,
        data_path("noise_topo"),
        data_path("noise_sp"),
        acc_baseline=null_accuracy,
        dim_baseline=input_dimensionality,
        robustness_threshold=0.95,
        omit_first_k=1,
        rep_spread="2std",
        figure_paths=(None,) * 6,
        stability_path=None,
        show=show,
    )


def load_paper_bundle(
    path: str | Path = DEFAULT_BUNDLE, *, require_complete: bool = False
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run run_all_paper_experiments.py first."
        )
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "map_summaries",
        "orientation_maps_90",
    }
    if require_complete:
        required.add("population_embeddings")
    missing = sorted(required - set(bundle))
    if missing:
        raise RuntimeError(f"The bundle is incomplete; missing sections: {missing}")
    return bundle


def _orientation_image(axis, value, title):
    image = axis.imshow(value, cmap="hsv", vmin=0, vmax=math.pi)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def _windowed_response_fourier(record):
    """Return an edge-apodized response spectrum for old and new bundles."""

    if record.get("response_fourier_window") == "separable Hann":
        return torch.as_tensor(record["mean_response_fourier_amplitude"])
    responses = torch.as_tensor(record["natural_responses"])[:, 0]
    responses = responses - responses.mean(dim=(-2, -1), keepdim=True)
    window = torch.hann_window(
        responses.shape[-1], periodic=False, dtype=responses.dtype
    )
    responses = responses * torch.outer(window, window)
    return torch.fft.fftshift(
        torch.fft.fft2(responses).abs(), dim=(-2, -1)
    ).mean(0)


def plot_map_organisation(bundle: dict[str, Any] | None = None):
    bundle = bundle or load_paper_bundle()
    summaries = bundle["map_summaries"]
    fig, axes = plt.subplots(2, 8, figsize=(21, 6.2), constrained_layout=True)
    column_titles = (
        "orientation map",
        "orientation coverage",
        "orientation Fourier",
        "neighbouring RFs",
        "retinotopy",
        "effective lateral field",
        "settled response",
        "response Fourier",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)

    for row, name in enumerate(("macro60", "micro60")):
        record = summaries[name]
        _orientation_image(
            axes[row, 0], record["orientation_rad"], ""
        )
        plot_orientation_histogram(
            record["orientation_rad"],
            ax=axes[row, 1],
            bins=18,
            font_size=10,
            show=False,
        )
        axes[row, 2].imshow(
            torch.log1p(record["fourier_spectrum"]), cmap="Greys"
        )
        axes[row, 2].set_axis_off()
        plot_receptive_field_mosaic(
            record["afferent_rf_examples"], ax=axes[row, 3], show=False
        )
        plot_retinotopy_grid(
            record["retinotopic_centres_lattice"]
            if "retinotopic_centres_lattice" in record
            else record["retinotopic_centres_input_px"],
            ax=axes[row, 4],
            max_lines=15,
            show=False,
        )
        lateral = (
            record["example_sre_field"]
            + record["example_cde_field"]
            - record["example_inhibitory_field"]
        )
        lateral_limit = float(torch.as_tensor(lateral).abs().max())
        axes[row, 5].imshow(
            lateral,
            cmap="coolwarm",
            vmin=-lateral_limit,
            vmax=lateral_limit,
        )
        axes[row, 5].set_axis_off()
        axes[row, 6].imshow(record["natural_responses"][0, 0], cmap="Greys")
        axes[row, 6].set_axis_off()
        axes[row, 7].imshow(
            torch.log1p(_windowed_response_fourier(record)), cmap="Greys"
        )
        axes[row, 7].set_axis_off()
        axes[row, 0].text(
            -0.16,
            0.5,
            "macro-GCAL" if name == "macro60" else "micro-GCAL",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
        )
        axes[row, 2].text(
            0.03,
            0.03,
            rf"$\Lambda={record['fourier_period_pixels']:.1f}$",
            transform=axes[row, 2].transAxes,
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    return fig


def plot_displacement_and_connectivity(
    bundle: dict[str, Any] | None = None,
    *,
    subtractive_threshold: float | None = None,
    connection_fraction: float | None = None,
    sigma: float | None = None,
    show: bool = False,
):
    """Plot displacement and connectivity with optional live resampling.

    The displacement and connectivity readouts are always derived live from
    the saved orientation maps and 90x90 checkpoints; they are not stored in
    the collection bundle and this function never retrains a map. Any
    non-negative ``sigma`` is valid.
    ``subtractive_threshold`` is in global-mean weight units, and
    ``connection_fraction`` is the fraction retained after thresholding.
    """

    bundle = bundle or load_paper_bundle()
    stored_config = bundle["metadata"]["config"]
    subtractive_threshold = (
        float(subtractive_threshold)
        if subtractive_threshold is not None
        else float(stored_config["connectivity_threshold_multiple"])
    )
    connection_fraction = (
        float(connection_fraction)
        if connection_fraction is not None
        else float(stored_config["connectivity_fraction"])
    )
    sigma = (
        float(sigma)
        if sigma is not None
        else float(stored_config["displacement_sigma_panel"])
    )
    from paper_stats_collector import (
        configurable_connectivity_analysis,
        configurable_displacement_analysis,
    )

    ideal_displacement = configurable_displacement_analysis(bundle, sigma=0.0)
    live_displacement = configurable_displacement_analysis(bundle, sigma=sigma)
    connectivity = configurable_connectivity_analysis(
        bundle,
        subtractive_threshold=subtractive_threshold,
        connection_fraction=connection_fraction,
        sigma=sigma,
        displacement_records=live_displacement,
    )
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    for column, name in enumerate(("macro90", "micro90")):
        ideal = ideal_displacement[name]
        moved = live_displacement[name]
        _orientation_image(
            axes[0, 2 * column], ideal["orientation_crop_rad"], f"{name}: ideal 45×45"
        )
        _orientation_image(
            axes[0, 2 * column + 1],
            moved["orientation_crop_rad"],
            f"displaced; mean={moved['realised_mean_displacement']:.2f}",
        )
        plot_orientation_periodicity_profile(
            ideal["autocorrelation_radius_lattice"],
            ideal["autocorrelation"],
            ax=axes[1, 2 * column],
            x_label="lattice distance",
            font_size=11,
            label="ideal",
            color="black",
            show=False,
        )
        plot_orientation_periodicity_profile(
            moved["autocorrelation_radius_lattice"],
            moved["autocorrelation"],
            ax=axes[1, 2 * column],
            x_label="lattice distance",
            font_size=11,
            label="displaced",
            color="0.55",
            line_style="--",
            show=False,
        )
        axes[1, 2 * column].set_title(f"{name}: axial autocorrelation")
        axes[1, 2 * column + 1].plot(
            ideal["orientation_difference_deg"], label="ideal"
        )
        axes[1, 2 * column + 1].plot(
            moved["orientation_difference_deg"], label="displaced"
        )
        axes[1, 2 * column + 1].set(
            title=f"{name}: pairwise tuning difference",
            xlabel="lattice distance",
            ylabel="difference (deg)",
        )

        for row, condition in zip((2, 3), ("ideal", "displaced")):
            record = connectivity[name][condition]
            representative = len(record["target_positions_new"]) // 2
            plot_sparse_tensor(
                torch.as_tensor(record["sparse_examples"][1]),
                int(record["target_positions_new"][representative]),
                ax=axes[row, 2 * column],
                show=False,
                marker_size=8,
                edge_linewidth=0.7,
            )
            if "sample_fraction_of_eligible_connections" in record:
                retained_fraction = record[
                    "sample_fraction_of_eligible_connections"
                ]
            else:
                retained_fraction = record["sample_fraction_of_full_field"]
            fraction_percent = 100 * float(retained_fraction)
            axes[row, 2 * column].set_title(
                f"{name} {condition}: {fraction_percent:g}% after threshold"
            )
            plot_patchiness_profile(
                record["frequency_cycles_per_mm"],
                record["mean_normalized_amplitude"],
                record["std_normalized_amplitude"],
                record["profile_count"],
                ax=axes[row, 2 * column + 1],
                max_freq_cyc_per_mm=10,
                band="sem",
                font_size=11,
                linewidth=2,
                show=False,
            )
            axes[row, 2 * column + 1].set_title(
                f"{condition}: aligned Fourier amplitude"
            )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_population_embeddings(bundle: dict[str, Any] | None = None):
    bundle = bundle or load_paper_bundle()
    if "population_embeddings" not in bundle:
        raise RuntimeError(
            "Population embeddings are missing; rerun run_all_paper_experiments.py."
        )
    data = bundle["population_embeddings"]
    panels = [
        (data["stimulus"], "grating stimuli"),
        (data["models"]["macro60"], "macro-GCAL responses"),
        (data["models"]["micro60"], "micro-GCAL responses"),
    ]
    if data.get("stringer") is not None:
        panels.append((data["stringer"], "mouse V1 responses (Stringer)"))
    fig = plt.figure(figsize=(4.2 * len(panels), 4))
    for index, (record, title) in enumerate(panels, start=1):
        axis = fig.add_subplot(1, len(panels), index, projection="3d")
        plot_precomputed_umap_with_angles_3d(
            record["embedding"],
            np.mod(record["orientation_rad"], math.pi),
            title=title,
            point_size=3,
            ax=axis,
            show=False,
        )
    fig.tight_layout()
    return fig


def plot_all_panels(
    bundle_path: str | Path = DEFAULT_BUNDLE,
    *,
    show: bool = True,
):
    """Display the stored paper panels in manuscript order."""

    bundle = load_paper_bundle(bundle_path, require_complete=True)
    core = plot_core_sweep_summary(show=False)
    figures = [
        core["figure"],
        plot_map_organisation(bundle),
        plot_displacement_and_connectivity(bundle),
        plot_population_embeddings(bundle),
    ]
    if show:
        plt.show()
    return figures


__all__ = [
    "load_paper_bundle",
    "plot_all_panels",
    "plot_core_sweep_summary",
    "plot_displacement_and_connectivity",
    "plot_map_organisation",
    "plot_population_embeddings",
]

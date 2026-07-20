"""Cellular orientation-map smoothing and exact-contour displacement analysis.

The public Chen et al. macaque V1 release contains one ROI mask and one preferred
orientation index per cell for four 850 x 850 micrometre fields.  This module keeps
the notebook thin and implements three safeguards that matter for the requested
measurement:

* axial orientations are smoothed as ``exp(2j * theta)``;
* the smoothing scale can be selected by leave-one-neuron-out prediction or by
  matching an externally specified orientation wavelength;
* each scored neuron's own contribution is analytically removed from the map.

The reported distance is the Euclidean distance from the soma to the nearest
linearly interpolated zero contour whose axial phase exactly matches that neuron's
orientation (modulo 180 degrees).  It is not a nearest-grid-cell approximation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import contourpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist


ZENODO_RECORD_URL = (
    "https://zenodo.org/api/records/20053907/files/Data_repository.zip/content"
)
ZENODO_ARCHIVE_MD5 = "0ae4ce1cf680a4cd8169d1a77394a1e1"
FOV_SIZE_UM = 850.0
IMAGE_SIZE_PX = 512
ALL_FOVS = ("MA_1", "MA_2", "MB_1", "MB_2")
REQUIRED_FILES = (
    "CCtotal.mat",
    "G4_PeakOriListTotal_base.mat",
    "targetcell_base_1.mat",
    "Y1_AnovaListTotal_base.mat",
)


@dataclass(frozen=True)
class CellularMap:
    """One tuned-cell point cloud from one physical imaging field."""

    name: str
    xy_um: np.ndarray
    orientation_deg: np.ndarray
    source_cell_indices: np.ndarray


@dataclass(frozen=True)
class MapAnalysis:
    """Computed smooth field and leave-one-out displacement values."""

    cellular_map: CellularMap
    sigma_um: float
    sigma_candidates_um: np.ndarray
    cv_mean_axial_similarity: np.ndarray
    cv_sem_axial_similarity: np.ndarray
    grid_x_um: np.ndarray
    grid_y_um: np.ndarray
    smoothed_orientation_deg: np.ndarray
    displacement_um: np.ndarray
    matched_xy_um: np.ndarray
    scoring_border_um: float = 0.0
    max_displacement_um: float | None = None
    target_wavelength_um: float | None = None
    achieved_rms_gradient_rad_per_um: float | None = None

    @property
    def mean_displacement_um(self) -> float:
        return float(np.nanmean(self.displacement_um))

    @property
    def n_finite_displacements(self) -> int:
        return int(np.isfinite(self.displacement_um).sum())


def _required_paths(data_root: Path) -> list[Path]:
    return [data_root / fov / filename for fov in ALL_FOVS for filename in REQUIRED_FILES]


def ensure_macaque_cfs_data(data_root: str | Path) -> Path:
    """Download and selectively extract the public archive when files are absent."""

    data_root = Path(data_root)
    required = _required_paths(data_root)
    if all(path.exists() for path in required):
        return data_root

    data_root.mkdir(parents=True, exist_ok=True)
    download_dir = data_root / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "Data_repository.zip"

    if not archive_path.exists() or _md5(archive_path) != ZENODO_ARCHIVE_MD5:
        urllib.request.urlretrieve(ZENODO_RECORD_URL, archive_path)
    if _md5(archive_path) != ZENODO_ARCHIVE_MD5:
        raise RuntimeError("Downloaded Zenodo archive failed its published MD5 check")

    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for fov in ALL_FOVS:
            (data_root / fov).mkdir(parents=True, exist_ok=True)
            for filename in REQUIRED_FILES:
                member = f"Data_repository/{fov}/{filename}"
                if member not in members:
                    raise FileNotFoundError(f"Archive does not contain {member}")
                destination = data_root / fov / filename
                if not destination.exists():
                    with archive.open(member) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset extraction is incomplete: {missing}")
    return data_root


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - used only to verify the repository checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _roi_centroids_um(cc_total: Iterable[np.ndarray]) -> np.ndarray:
    """Convert MATLAB 1-based column-major ROI indices to x/y centroids."""

    centroids_px = []
    for roi_indices in cc_total:
        flat = np.asarray(roi_indices, dtype=np.int64).reshape(-1) - 1
        if flat.size == 0 or flat.min() < 0 or flat.max() >= IMAGE_SIZE_PX**2:
            raise ValueError("CCtotal contains an empty or out-of-range ROI")
        y_px = flat % IMAGE_SIZE_PX
        x_px = flat // IMAGE_SIZE_PX
        centroids_px.append((x_px.mean(), y_px.mean()))
    return np.asarray(centroids_px, dtype=float) * (FOV_SIZE_UM / IMAGE_SIZE_PX)


def load_macaque_cfs_fov(data_root: str | Path, fov: str) -> CellularMap:
    """Load the authors' p<0.01 tuned-cell set without re-binning orientations."""

    if fov not in ALL_FOVS:
        raise ValueError(f"Unknown FOV {fov!r}; choose one of {ALL_FOVS}")
    folder = Path(data_root) / fov

    cc_total = loadmat(folder / "CCtotal.mat", simplify_cells=True)["CCtotal"]
    xy_all = _roi_centroids_um(cc_total)
    peak_index = np.asarray(
        loadmat(
            folder / "G4_PeakOriListTotal_base.mat", simplify_cells=True
        )["G4_PeakOriListTotal_base"]
    ).reshape(-1)
    p_anova = np.asarray(
        loadmat(folder / "Y1_AnovaListTotal_base.mat", simplify_cells=True)[
            "Y1_AnovaListTotal_base"
        ]
    ).reshape(-1)
    tuned_indices = np.asarray(
        loadmat(folder / "targetcell_base_1.mat", simplify_cells=True)[
            "targetcell_base_1"
        ]
    ).reshape(-1).astype(np.int64) - 1

    expected = np.flatnonzero(p_anova < 0.01)
    if not np.array_equal(np.sort(tuned_indices), expected):
        raise ValueError(f"{fov}: targetcell list does not match the p<0.01 release filter")
    if not (len(xy_all) == len(peak_index) == len(p_anova)):
        raise ValueError(f"{fov}: cell arrays have inconsistent lengths")
    if np.any((peak_index < 1) | (peak_index > 12)):
        raise ValueError(f"{fov}: preferred-orientation indices must lie in 1..12")

    # Twelve tested axial orientations at 15-degree spacing.  This preserves all
    # released angular resolution; the analysis does not collapse them into bins.
    orientation_deg = (peak_index[tuned_indices].astype(float) - 1.0) * 15.0
    return CellularMap(
        name=fov,
        xy_um=xy_all[tuned_indices],
        orientation_deg=orientation_deg,
        source_cell_indices=tuned_indices,
    )


def rank_fovs_by_tuned_cells(data_root: str | Path) -> list[CellularMap]:
    """Return all four maps in descending tuned-cell-count order."""

    maps = [load_macaque_cfs_fov(data_root, fov) for fov in ALL_FOVS]
    return sorted(maps, key=lambda item: len(item.xy_um), reverse=True)


def select_smoothing_sigma(
    cellular_map: CellularMap,
    candidates_um: Sequence[float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Use LOO circular prediction and the one-SE rule to choose a smooth scale."""

    candidates = np.asarray(candidates_um, dtype=float)
    if candidates.ndim != 1 or candidates.size == 0 or np.any(candidates <= 0):
        raise ValueError("candidates_um must be a non-empty sequence of positive values")

    xy = cellular_map.xy_um
    axial = np.exp(2j * np.deg2rad(cellular_map.orientation_deg))
    distance_sq = cdist(xy, xy, metric="sqeuclidean")
    score_by_sigma = np.empty((len(xy), len(candidates)), dtype=float)

    for column, sigma in enumerate(candidates):
        weights = np.exp(-distance_sq / (2.0 * sigma**2))
        np.fill_diagonal(weights, 0.0)
        prediction = weights @ axial
        prediction /= np.maximum(np.abs(prediction), np.finfo(float).eps)
        score_by_sigma[:, column] = np.real(np.conj(axial) * prediction)

    means = score_by_sigma.mean(axis=0)
    sems = score_by_sigma.std(axis=0, ddof=1) / np.sqrt(len(xy))
    best = int(np.argmax(means))
    eligible = np.flatnonzero(means >= means[best] - sems[best])
    chosen = int(eligible[-1])  # smoothest candidate statistically tied with the best
    return float(candidates[chosen]), means, sems


def _complex_kernel_sum(
    cellular_map: CellularMap,
    sigma_um: float,
    grid_x_um: np.ndarray,
    grid_y_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(grid_x_um, grid_y_um)
    numerator = np.zeros_like(xx, dtype=np.complex128)
    denominator = np.zeros_like(xx, dtype=float)
    axial = np.exp(2j * np.deg2rad(cellular_map.orientation_deg))

    for (x_um, y_um), mark in zip(cellular_map.xy_um, axial):
        weight = np.exp(-((xx - x_um) ** 2 + (yy - y_um) ** 2) / (2.0 * sigma_um**2))
        numerator += weight * mark
        denominator += weight
    return numerator, denominator


def _rms_orientation_gradient(
    numerator: np.ndarray,
    denominator: np.ndarray,
    grid_x_um: np.ndarray,
    grid_y_um: np.ndarray,
) -> float:
    """Density/coherence-weighted RMS gradient of the axial orientation field.

    For ``z = exp(2j * theta)``, half the phase gradient is the orientation
    gradient.  A cosine-like orientation cycle of wavelength Lambda therefore
    has target gradient ``pi / Lambda`` radians per micrometre.
    """

    eps = np.finfo(float).eps
    unit_axial = numerator / np.maximum(np.abs(numerator), eps)
    spacing_x = float(grid_x_um[1] - grid_x_um[0])
    spacing_y = float(grid_y_um[1] - grid_y_um[0])
    derivative_y, derivative_x = np.gradient(unit_axial, spacing_y, spacing_x)
    gradient_x = 0.5 * np.imag(np.conj(unit_axial) * derivative_x)
    gradient_y = 0.5 * np.imag(np.conj(unit_axial) * derivative_y)
    coherence = np.abs(numerator) / np.maximum(denominator, eps)
    support = denominator / np.maximum(float(denominator.max()), eps)
    weights = coherence**2 * support
    gradient_sq = gradient_x**2 + gradient_y**2
    return float(np.sqrt(np.sum(weights * gradient_sq) / np.sum(weights)))


def _analysis_at_fixed_sigma(
    cellular_map: CellularMap,
    sigma_um: float,
    grid_size: int,
    *,
    target_wavelength_um: float | None = None,
    scoring_border_um: float = 0.0,
    max_displacement_um: float | None = None,
) -> MapAnalysis:
    """Build and score a map at an already determined bandwidth."""

    if grid_size < 64:
        raise ValueError("grid_size must be at least 64 for contour-distance accuracy")
    grid_x_um = np.linspace(0.0, FOV_SIZE_UM, grid_size)
    grid_y_um = np.linspace(0.0, FOV_SIZE_UM, grid_size)
    numerator, denominator = _complex_kernel_sum(
        cellular_map, sigma_um, grid_x_um, grid_y_um
    )
    smooth_axial = numerator / np.maximum(denominator, np.finfo(float).eps)
    smooth_deg = np.mod(0.5 * np.rad2deg(np.angle(smooth_axial)), 180.0)
    displacement, matched_xy_um = _leave_one_out_exact_displacements(
        cellular_map, sigma_um, grid_x_um, grid_y_um, numerator
    )
    if scoring_border_um < 0 or scoring_border_um >= FOV_SIZE_UM / 2:
        raise ValueError("scoring_border_um must lie in [0, FOV_SIZE_UM / 2)")
    if scoring_border_um > 0:
        xy = cellular_map.xy_um
        central = (
            (xy[:, 0] >= scoring_border_um)
            & (xy[:, 0] <= FOV_SIZE_UM - scoring_border_um)
            & (xy[:, 1] >= scoring_border_um)
            & (xy[:, 1] <= FOV_SIZE_UM - scoring_border_um)
        )
        displacement[~central] = np.nan
        matched_xy_um[~central] = np.nan
    if max_displacement_um is not None:
        if max_displacement_um <= 0:
            raise ValueError("max_displacement_um must be positive when provided")
        excessive = displacement > max_displacement_um
        displacement[excessive] = np.nan
        matched_xy_um[excessive] = np.nan
    achieved_gradient = (
        _rms_orientation_gradient(numerator, denominator, grid_x_um, grid_y_um)
        if target_wavelength_um is not None
        else None
    )
    return MapAnalysis(
        cellular_map=cellular_map,
        sigma_um=float(sigma_um),
        sigma_candidates_um=np.asarray([], dtype=float),
        cv_mean_axial_similarity=np.asarray([], dtype=float),
        cv_sem_axial_similarity=np.asarray([], dtype=float),
        grid_x_um=grid_x_um,
        grid_y_um=grid_y_um,
        smoothed_orientation_deg=smooth_deg,
        displacement_um=displacement,
        matched_xy_um=matched_xy_um,
        scoring_border_um=float(scoring_border_um),
        max_displacement_um=max_displacement_um,
        target_wavelength_um=target_wavelength_um,
        achieved_rms_gradient_rad_per_um=achieved_gradient,
    )


def select_sigma_for_wavelength(
    cellular_map: CellularMap,
    wavelength_um: float,
    sigma_search_um: Sequence[float] = tuple(np.geomspace(8.0, 300.0, 36)),
    grid_size: int = 256,
) -> float:
    """Infer the bandwidth whose map gradient matches a specified wavelength."""

    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive")
    candidates = np.asarray(sigma_search_um, dtype=float)
    if (
        candidates.ndim != 1
        or candidates.size < 2
        or np.any(candidates <= 0)
        or np.any(np.diff(candidates) <= 0)
    ):
        raise ValueError("sigma_search_um must be a strictly increasing positive sequence")
    grid_x_um = np.linspace(0.0, FOV_SIZE_UM, grid_size)
    grid_y_um = np.linspace(0.0, FOV_SIZE_UM, grid_size)
    gradients = []
    for sigma_um in candidates:
        numerator, denominator = _complex_kernel_sum(
            cellular_map, float(sigma_um), grid_x_um, grid_y_um
        )
        gradients.append(
            _rms_orientation_gradient(numerator, denominator, grid_x_um, grid_y_um)
        )
    gradients = np.asarray(gradients)
    target = np.pi / wavelength_um
    crossing = np.flatnonzero((gradients[:-1] - target) * (gradients[1:] - target) <= 0)
    if not len(crossing):
        raise ValueError(
            f"target gradient {target:g} is outside the searched range "
            f"[{gradients.min():g}, {gradients.max():g}]"
        )
    index = int(crossing[0])
    gradient_pair = gradients[index : index + 2]
    sigma_pair = candidates[index : index + 2]
    log_sigma = np.interp(
        np.log(target), np.log(gradient_pair[::-1]), np.log(sigma_pair[::-1])
    )
    return float(np.exp(log_sigma))


def _point_to_segments_nearest(
    point: np.ndarray,
    lines: list[np.ndarray],
    real: np.ndarray,
    grid_x_um: np.ndarray,
    grid_y_um: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Distance and nearest point on the correct exact-orientation contour."""

    best_sq = np.inf
    best_point = np.full(2, np.nan, dtype=float)
    dx = float(grid_x_um[1] - grid_x_um[0])
    dy = float(grid_y_um[1] - grid_y_um[0])
    for line in lines:
        if len(line) < 2:
            continue
        start = line[:-1]
        end = line[1:]
        midpoint = 0.5 * (start + end)
        ix = np.clip(np.rint((midpoint[:, 0] - grid_x_um[0]) / dx).astype(int), 0, len(grid_x_um) - 1)
        iy = np.clip(np.rint((midpoint[:, 1] - grid_y_um[0]) / dy).astype(int), 0, len(grid_y_um) - 1)
        correct_branch = real[iy, ix] > 0.0
        if not np.any(correct_branch):
            continue
        start = start[correct_branch]
        end = end[correct_branch]
        segment = end - start
        segment_sq = np.einsum("ij,ij->i", segment, segment)
        valid = segment_sq > 0.0
        if not np.any(valid):
            continue
        start = start[valid]
        segment = segment[valid]
        segment_sq = segment_sq[valid]
        projection = np.einsum("ij,ij->i", point[None, :] - start, segment) / segment_sq
        projection = np.clip(projection, 0.0, 1.0)
        nearest = start + projection[:, None] * segment
        distance_sq = np.einsum("ij,ij->i", nearest - point[None, :], nearest - point[None, :])
        if distance_sq.size:
            nearest_index = int(np.argmin(distance_sq))
            if float(distance_sq[nearest_index]) < best_sq:
                best_sq = float(distance_sq[nearest_index])
                best_point = nearest[nearest_index].copy()
    distance = float(np.sqrt(best_sq)) if np.isfinite(best_sq) else np.nan
    return distance, best_point


def _point_to_segments_min_distance(
    point: np.ndarray,
    lines: list[np.ndarray],
    real: np.ndarray,
    grid_x_um: np.ndarray,
    grid_y_um: np.ndarray,
) -> float:
    """Compatibility wrapper returning only the nearest-contour distance."""

    distance, _ = _point_to_segments_nearest(
        point, lines, real, grid_x_um, grid_y_um
    )
    return distance


def _leave_one_out_exact_displacements(
    cellular_map: CellularMap,
    sigma_um: float,
    grid_x_um: np.ndarray,
    grid_y_um: np.ndarray,
    full_numerator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure and locate each soma's nearest exact iso-orientation contour."""

    xx, yy = np.meshgrid(grid_x_um, grid_y_um)
    axial = np.exp(2j * np.deg2rad(cellular_map.orientation_deg))
    distances = np.full(len(cellular_map.xy_um), np.nan, dtype=float)
    matched_xy_um = np.full((len(cellular_map.xy_um), 2), np.nan, dtype=float)

    for index, ((x_um, y_um), mark) in enumerate(zip(cellular_map.xy_um, axial)):
        self_weight = np.exp(
            -((xx - x_um) ** 2 + (yy - y_um) ** 2) / (2.0 * sigma_um**2)
        )
        loo_numerator = full_numerator - self_weight * mark
        rotated = loo_numerator * np.conj(mark)
        imaginary = np.imag(rotated)
        generator = contourpy.contour_generator(
            x=grid_x_um,
            y=grid_y_um,
            z=imaginary,
            name="serial",
            corner_mask=True,
        )
        lines = generator.lines(0.0)
        distances[index], matched_xy_um[index] = _point_to_segments_nearest(
            np.asarray((x_um, y_um)), lines, np.real(rotated), grid_x_um, grid_y_um
        )
    return distances, matched_xy_um


def analyze_cellular_map(
    cellular_map: CellularMap,
    sigma_candidates_um: Sequence[float] = (15, 20, 25, 30, 40, 50, 65, 80, 100, 125, 160, 200),
    grid_size: int = 256,
) -> MapAnalysis:
    """Fit the smooth axial field and compute LOO exact-contour distances."""

    sigma, cv_mean, cv_sem = select_smoothing_sigma(cellular_map, sigma_candidates_um)
    fixed = _analysis_at_fixed_sigma(cellular_map, sigma, grid_size)
    return MapAnalysis(
        cellular_map=fixed.cellular_map,
        sigma_um=fixed.sigma_um,
        sigma_candidates_um=np.asarray(sigma_candidates_um, dtype=float),
        cv_mean_axial_similarity=cv_mean,
        cv_sem_axial_similarity=cv_sem,
        grid_x_um=fixed.grid_x_um,
        grid_y_um=fixed.grid_y_um,
        smoothed_orientation_deg=fixed.smoothed_orientation_deg,
        displacement_um=fixed.displacement_um,
        matched_xy_um=fixed.matched_xy_um,
    )


def run_macaque_cfs_analysis(
    data_root: str | Path,
    n_maps: int = 3,
    sigma_candidates_um: Sequence[float] = (15, 20, 25, 30, 40, 50, 65, 80, 100, 125, 160, 200),
    grid_size: int = 256,
) -> list[MapAnalysis]:
    """Analyze the one-to-three densest released cellular maps."""

    if n_maps not in (1, 2, 3):
        raise ValueError("n_maps must be 1, 2, or 3")
    root = ensure_macaque_cfs_data(data_root)
    maps = rank_fovs_by_tuned_cells(root)[:n_maps]
    return [
        analyze_cellular_map(
            cellular_map,
            sigma_candidates_um=sigma_candidates_um,
            grid_size=grid_size,
        )
        for cellular_map in maps
    ]


def run_fixed_sigma_analysis(
    data_root: str | Path,
    sigma_um: float = 100.0,
    scoring_border_um: float = 0.0,
    max_displacement_um: float | None = 350.0,
    n_maps: int = 3,
    grid_size: int = 256,
) -> list[MapAnalysis]:
    """Analyze the densest fields with one shared, user-specified bandwidth."""

    if sigma_um <= 0:
        raise ValueError("sigma_um must be positive")
    if n_maps not in (1, 2, 3):
        raise ValueError("n_maps must be 1, 2, or 3")
    root = ensure_macaque_cfs_data(data_root)
    maps = rank_fovs_by_tuned_cells(root)[:n_maps]
    return [
        _analysis_at_fixed_sigma(
            cellular_map,
            float(sigma_um),
            grid_size,
            scoring_border_um=float(scoring_border_um),
            max_displacement_um=max_displacement_um,
        )
        for cellular_map in maps
    ]


def run_wavelength_anchored_analysis(
    data_root: str | Path,
    wavelengths_um: Sequence[float] = (600.0, 700.0, 800.0),
    n_maps: int = 3,
    grid_size: int = 256,
    sigma_search_um: Sequence[float] = tuple(np.geomspace(8.0, 300.0, 36)),
) -> dict[float, list[MapAnalysis]]:
    """Analyze dense maps after anchoring their RMS gradient to each wavelength."""

    if n_maps not in (1, 2, 3):
        raise ValueError("n_maps must be 1, 2, or 3")
    wavelengths = np.asarray(wavelengths_um, dtype=float)
    if wavelengths.ndim != 1 or not len(wavelengths) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths_um must be a non-empty positive sequence")
    root = ensure_macaque_cfs_data(data_root)
    maps = rank_fovs_by_tuned_cells(root)[:n_maps]
    output: dict[float, list[MapAnalysis]] = {}
    for wavelength_um in wavelengths:
        analyses = []
        for cellular_map in maps:
            sigma_um = select_sigma_for_wavelength(
                cellular_map,
                float(wavelength_um),
                sigma_search_um=sigma_search_um,
                grid_size=grid_size,
            )
            analyses.append(
                _analysis_at_fixed_sigma(
                    cellular_map,
                    sigma_um,
                    grid_size,
                    target_wavelength_um=float(wavelength_um),
                )
            )
        output[float(wavelength_um)] = analyses
    return output


def wavelength_sensitivity_table(
    results_by_wavelength: dict[float, Sequence[MapAnalysis]],
) -> list[dict[str, float | int | str]]:
    """Tabulate the wavelength sweep, including achieved gradient matching."""

    rows = []
    for wavelength_um, results in results_by_wavelength.items():
        for result in results:
            rows.append(
                {
                    "fov": result.cellular_map.name,
                    "n_tuned_cells": len(result.cellular_map.xy_um),
                    "target_wavelength_um": wavelength_um,
                    "smoothing_sigma_um": result.sigma_um,
                    "achieved_rms_gradient_rad_per_um": (
                        result.achieved_rms_gradient_rad_per_um
                    ),
                    "mean_displacement_um": result.mean_displacement_um,
                }
            )
    return rows


def results_table(results: Sequence[MapAnalysis]) -> list[dict[str, float | int | str]]:
    """Small serializable summary; individual distances remain in the NPZ output."""

    rows = []
    for result in results:
        row: dict[str, float | int | str] = {
            "fov": result.cellular_map.name,
            "n_tuned_cells": len(result.cellular_map.xy_um),
            "n_exact_contours_found": result.n_finite_displacements,
            "smoothing_sigma_um": result.sigma_um,
            "mean_displacement_um": result.mean_displacement_um,
        }
        if result.target_wavelength_um is not None:
            row["target_wavelength_um"] = result.target_wavelength_um
            row["achieved_rms_gradient_rad_per_um"] = float(
                result.achieved_rms_gradient_rad_per_um
            )
        rows.append(row)
    return rows


def plot_cellular_map_analysis(
    results: Sequence[MapAnalysis],
    figure_path: str | Path | None = None,
    dpi: int = 200,
) -> plt.Figure:
    """Render only the requested raw scatter and smoothed map for each FOV."""

    if not results:
        raise ValueError("results is empty")
    figure, axes = plt.subplots(
        len(results),
        2,
        figsize=(2 * README_PANEL_WIDTH, README_ROW_HEIGHT * len(results)),
        squeeze=False,
        constrained_layout=True,
    )
    colormap = plt.get_cmap("hsv")
    normalization = plt.Normalize(0.0, 180.0)
    last_image = None

    for row, result in enumerate(results):
        cellular_map = result.cellular_map
        raw_axis, smooth_axis = axes[row]
        raw_axis.scatter(
            cellular_map.xy_um[:, 0],
            cellular_map.xy_um[:, 1],
            c=cellular_map.orientation_deg,
            cmap=colormap,
            norm=normalization,
            s=12,
            linewidths=0,
        )
        last_image = smooth_axis.imshow(
            result.smoothed_orientation_deg,
            origin="lower",
            extent=(0.0, FOV_SIZE_UM, 0.0, FOV_SIZE_UM),
            cmap=colormap,
            norm=normalization,
            interpolation="nearest",
            rasterized=True,
        )
        raw_axis.set_title(
            f"{cellular_map.name}: cellular scatter (n={len(cellular_map.xy_um)})",
            fontsize=11,
        )
        anchor = (
            f", wavelength={result.target_wavelength_um:g} um"
            if result.target_wavelength_um is not None
            else ""
        )
        smooth_axis.set_title(
            f"{cellular_map.name}: circular smooth map (sigma={result.sigma_um:.1f} um{anchor})\n"
            f"mean displacement = {result.mean_displacement_um:.1f} um\n"
            "(nearest exact-orientation contour)",
            fontsize=10.5,
        )
        for axis in (raw_axis, smooth_axis):
            axis.set_xlim(0.0, FOV_SIZE_UM)
            axis.set_ylim(0.0, FOV_SIZE_UM)
            axis.set_aspect("equal")
            axis.set_xlabel("x (um)")
            axis.set_ylabel("y (um)")

    if last_image is not None:
        colorbar = figure.colorbar(last_image, ax=axes, ticks=(0, 45, 90, 135, 180), shrink=0.75)
        colorbar.set_label("preferred orientation (deg; axial)")
    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    return figure


def plot_fixed_sigma_summary(
    results: Sequence[MapAnalysis],
    figure_path: str | Path | None = None,
    dpi: int = 120,
    font_size: float = 14.0,
) -> plt.Figure:
    """Plot cellular scatters above their common-bandwidth smooth maps."""

    if not results:
        raise ValueError("results is empty")
    n_columns = len(results)
    figure = plt.figure(
        figsize=(README_PANEL_WIDTH * n_columns, 9.2),
        constrained_layout=False,
    )
    grid = figure.add_gridspec(
        2,
        n_columns + 1,
        width_ratios=(*([1.0] * n_columns), 0.045),
        left=0.055,
        right=0.94,
        bottom=0.075,
        top=0.91,
        wspace=0.10,
        hspace=0.20,
    )
    axes = np.asarray(
        [
            [figure.add_subplot(grid[row, column]) for column in range(n_columns)]
            for row in range(2)
        ]
    )
    colorbar_axis = figure.add_subplot(grid[:, -1])
    colormap = plt.get_cmap("hsv")
    normalization = plt.Normalize(0.0, 180.0)
    for column, result in enumerate(results):
        cellular_map = result.cellular_map
        scatter_axis, smooth_axis = axes[:, column]
        scatter_axis.scatter(
            cellular_map.xy_um[:, 0],
            cellular_map.xy_um[:, 1],
            c=cellular_map.orientation_deg,
            cmap=colormap,
            norm=normalization,
            s=14,
            linewidths=0,
        )
        smooth_axis.imshow(
            result.smoothed_orientation_deg,
            origin="lower",
            extent=(0.0, FOV_SIZE_UM, 0.0, FOV_SIZE_UM),
            cmap=colormap,
            norm=normalization,
            interpolation="bilinear",
            rasterized=True,
        )
        scatter_axis.set_title(
            f"{cellular_map.name} · measured cells (n={len(cellular_map.xy_um)})"
        )
        smooth_axis.set_title(
            f"σ={result.sigma_um:g} µm smooth · mean d={result.mean_displacement_um:.1f} µm"
        )
        for axis in (scatter_axis, smooth_axis):
            axis.set_xlim(0.0, FOV_SIZE_UM)
            axis.set_ylim(0.0, FOV_SIZE_UM)
            axis.set_aspect("equal")
            axis.tick_params(labelsize=font_size - 2)
            axis.title.set_fontsize(font_size)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=colormap),
        cax=colorbar_axis,
        ticks=(0, 45, 90, 135, 180),
    )
    colorbar.set_label("preferred orientation (degrees; axial)", fontsize=font_size)
    colorbar.ax.tick_params(labelsize=font_size - 2)
    figure.supxlabel("cortical x-position (µm)", fontsize=font_size, y=0.015)
    figure.supylabel("cortical y-position (µm)", fontsize=font_size, x=0.012)
    figure.suptitle(
        "Macaque V1 cellular orientation displacement",
        fontsize=font_size + 4,
        y=0.985,
    )
    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=dpi, facecolor="white")
    return figure


def plot_sparse_displacement_links(
    results: Sequence[MapAnalysis],
    n_links: int = 20,
    seed: int = 7,
    map_alpha: float = 0.28,
    point_alpha: float = 0.78,
    figure_path: str | Path | None = None,
    dpi: int = 120,
    font_size: float = 14.0,
) -> plt.Figure:
    """Show sparse soma-to-matched-contour links over faint smooth maps."""

    if not results:
        raise ValueError("results is empty")
    if n_links < 1:
        raise ValueError("n_links must be positive")
    if not 0 <= map_alpha <= 1 or not 0 <= point_alpha <= 1:
        raise ValueError("map_alpha and point_alpha must lie in [0, 1]")
    n_columns = len(results)
    figure = plt.figure(
        figsize=(README_PANEL_WIDTH * n_columns, README_ROW_HEIGHT),
        constrained_layout=False,
    )
    grid = figure.add_gridspec(
        1,
        n_columns + 1,
        width_ratios=(*([1.0] * n_columns), 0.045),
        left=0.055,
        right=0.94,
        bottom=0.15,
        top=0.78,
        wspace=0.10,
    )
    axes = np.asarray(
        [figure.add_subplot(grid[0, column]) for column in range(n_columns)]
    )
    colorbar_axis = figure.add_subplot(grid[0, -1])
    colormap = plt.get_cmap("hsv")
    normalization = plt.Normalize(0.0, 180.0)
    generator = np.random.default_rng(seed)
    for column, (axis, result) in enumerate(zip(axes, results)):
        cellular_map = result.cellular_map
        axis.imshow(
            result.smoothed_orientation_deg,
            origin="lower",
            extent=(0.0, FOV_SIZE_UM, 0.0, FOV_SIZE_UM),
            cmap=colormap,
            norm=normalization,
            interpolation="bilinear",
            alpha=map_alpha,
            rasterized=True,
        )
        finite = np.flatnonzero(
            np.isfinite(result.displacement_um)
            & np.isfinite(result.matched_xy_um).all(axis=1)
        )
        chosen = np.sort(
            generator.choice(finite, size=min(n_links, len(finite)), replace=False)
        )
        starts = cellular_map.xy_um[chosen]
        ends = result.matched_xy_um[chosen]
        for start, end in zip(starts, ends):
            axis.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                color="black",
                linewidth=1.25,
                linestyle="-",
                alpha=0.9,
                zorder=2,
            )
        axis.scatter(
            starts[:, 0],
            starts[:, 1],
            c=cellular_map.orientation_deg[chosen],
            cmap=colormap,
            norm=normalization,
            s=30,
            alpha=point_alpha,
            linewidths=0,
            zorder=3,
        )
        axis.scatter(
            ends[:, 0],
            ends[:, 1],
            color="black",
            marker="x",
            s=22,
            linewidths=1.2,
            zorder=4,
        )
        axis.set_title(
            f"{cellular_map.name} · {len(chosen)} examples · "
            f"mean d={result.mean_displacement_um:.1f} µm"
        )
        axis.set_xlim(0.0, FOV_SIZE_UM)
        axis.set_ylim(0.0, FOV_SIZE_UM)
        axis.set_aspect("equal")
        axis.tick_params(labelsize=font_size - 2)
        axis.title.set_fontsize(font_size)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=colormap),
        cax=colorbar_axis,
        ticks=(0, 45, 90, 135, 180),
    )
    colorbar.set_label("preferred orientation (degrees; axial)", fontsize=font_size)
    colorbar.ax.tick_params(labelsize=font_size - 2)
    figure.supxlabel("cortical x-position (µm)", fontsize=font_size, y=0.035)
    figure.supylabel("cortical y-position (µm)", fontsize=font_size, x=0.012)
    figure.suptitle(
        "Example soma-to-map displacement correspondences",
        fontsize=font_size + 4,
        y=0.98,
    )
    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=dpi, facecolor="white")
    return figure


def save_analysis_results(
    results: Sequence[MapAnalysis],
    output_dir: str | Path,
    prefix: str = "macaque_cfs",
) -> tuple[Path, Path]:
    """Save the mean summary and all per-neuron distances for audit/reuse."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{prefix}_displacement_summary.csv"
    distance_path = output_dir / f"{prefix}_displacements.npz"
    rows = results_table(results)

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    arrays: dict[str, np.ndarray] = {}
    metadata = []
    for result in results:
        name = result.cellular_map.name
        arrays[f"{name}_xy_um"] = result.cellular_map.xy_um
        arrays[f"{name}_orientation_deg"] = result.cellular_map.orientation_deg
        arrays[f"{name}_displacement_um"] = result.displacement_um
        arrays[f"{name}_matched_xy_um"] = result.matched_xy_um
        arrays[f"{name}_smoothed_orientation_deg"] = result.smoothed_orientation_deg
        metadata.append(
            {
                "fov": name,
                "sigma_um": result.sigma_um,
                "grid_size": len(result.grid_x_um),
                "mean_displacement_um": result.mean_displacement_um,
                "scoring_border_um": result.scoring_border_um,
                "max_displacement_um": result.max_displacement_um,
                "leave_one_out_scoring": True,
                "orientation_period_deg": 180,
                "target_wavelength_um": result.target_wavelength_um,
                "achieved_rms_gradient_rad_per_um": (
                    result.achieved_rms_gradient_rad_per_um
                ),
            }
        )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(distance_path, **arrays)
    return summary_path, distance_path
# Match the README-facing plots in ``microdomain_demo.py`` without importing
# that public facade back into this lower-level analysis module.
README_ROW_HEIGHT = 5.0
README_TWO_ROW_HEIGHT = 1.5 * README_ROW_HEIGHT
README_PANEL_WIDTH = 5.0

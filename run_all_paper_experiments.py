"""Reproduce the numerical data and generated figures for the paper.

The runner executes the replicated fidelity/complexity/robustness sweeps,
then trains the four maps used by the spatial and population-response panels.
Map checkpoints are reused automatically after an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE = Path(__file__).resolve().parent
DEFAULT_STRINGER_SOURCE = (
    BASE.parent
    / "project-microdomains"
    / "data"
    / "stringer"
    / "static_sin_rand_TX36_2019_10_21_1.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-stimuli", type=Path, default=BASE / "input_stimuli")
    parser.add_argument("--base-seed", type=int, default=20260728)
    parser.add_argument("--n-reps", type=int, default=3)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--output-file", type=Path, default=BASE / "paper_results" / "paper_panel_data.pt")
    parser.add_argument("--checkpoint-dir", type=Path, default=BASE / "paper_results" / "map_checkpoints")
    parser.add_argument("--stringer-source", type=Path)
    parser.add_argument(
        "--download-stringer",
        action="store_true",
        help="Download public Stringer recording 1 (TX36) from Figshare if needed.",
    )
    parser.add_argument("--no-strict-determinism", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train the two main geometries for one batch and stop before panel analyses.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _ensure_preimport_environment(args: argparse.Namespace) -> None:
    seed = str(args.base_seed)
    required = {
        "PYTHONHASHSEED": seed,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        ),
        # UMAP uses Numba internally.  A single workqueue worker avoids
        # contention with the PyTorch/CUDA thread pools used earlier.
        "NUMBA_NUM_THREADS": "1",
        "NUMBA_THREADING_LAYER": "workqueue",
    }
    if all(os.environ.get(key) == value for key, value in required.items()):
        return
    environment = dict(os.environ)
    environment.update(required)
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)


def _run_core(args: argparse.Namespace) -> dict[str, str]:
    tag = time.strftime("%Y%m%d_%H%M%S") + "_paper"
    command = [
        sys.executable,
        str(BASE / "run_accdim_and_noise_sweeps.py"),
        "--base-seed",
        str(args.base_seed),
        "--n-reps",
        str(args.n_reps),
        "--trials",
        str(args.trials),
        "--output-tag",
        tag,
        "--input-stimuli",
        str(args.input_stimuli),
    ]
    if args.no_strict_determinism:
        command.append("--no-strict-determinism")
    if args.check_only:
        command.append("--check-only")
    subprocess.run(command, cwd=BASE, check=True)
    return {
        "accdim": str(BASE / "data_l4" / f"data_accdim_{tag}.pt"),
        "noise_sp": str(BASE / "data_l4" / f"data_noise_sp_{tag}.pt"),
        "noise_topo": str(BASE / "data_l4" / f"data_noise_topo_{tag}.pt"),
        "figure_directory": str(BASE / "figures" / f"accdim_and_noise_{tag}"),
    }


def _run_panels(args: argparse.Namespace) -> Path | None:
    from paper_stats_collector import (
        PaperStatsConfig,
        collect_paper_panel_data,
    )

    source = args.stringer_source
    if source is None and DEFAULT_STRINGER_SOURCE.is_file():
        source = DEFAULT_STRINGER_SOURCE
    config_kwargs = {
        "input_stimuli": str(args.input_stimuli.resolve()),
        "output_file": str(args.output_file.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "local_stringer_source": str(source.resolve()) if source else None,
        "base_seed": args.base_seed,
        "strict_determinism": not args.no_strict_determinism,
    }
    if args.smoke_test:
        smoke_root = BASE / "paper_results" / "smoke_test"
        config_kwargs.update(
            output_file=str(smoke_root / "smoke_bundle.pt"),
            checkpoint_dir=str(smoke_root / "checkpoints"),
            natural_response_samples=64,
            epochs=1,
            max_train_batches=1,
            map_names=("macro60", "micro60"),
            num_workers=0,
        )
    if args.check_only:
        config = PaperStatsConfig(**config_kwargs)
        for key, value in vars(config).items():
            print(f"{key}: {value}")
        print("paper-panel configuration validation passed")
        return None
    return collect_paper_panel_data(
        PaperStatsConfig(**config_kwargs),
        download_stringer=args.download_stringer,
    )


def main() -> None:
    args = parse_args()
    _ensure_preimport_environment(args)
    os.chdir(BASE)
    if not args.input_stimuli.is_dir():
        raise FileNotFoundError(f"Missing input-stimulus directory: {args.input_stimuli}")
    core_outputs = None if args.smoke_test else _run_core(args)
    panel_data = _run_panels(args)
    if core_outputs is not None and panel_data is not None and not args.check_only:
        manifest = {
            "core_outputs": core_outputs,
            "panel_data": str(panel_data),
        }
        manifest_path = BASE / "paper_results" / "paper_run.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[done] paper run manifest: {manifest_path}")


if __name__ == "__main__":
    main()

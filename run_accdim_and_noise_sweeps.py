"""Run the current L4 accuracy/dimensionality and noise collectors in order.

The scientific implementation remains in ``stats_collector.py`` so this
entry point cannot silently diverge from the current ``NeuralSheet``.  Its
job is limited to reproducible configuration, unique output names, phase
status files, and failure handling.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone


BASE = Path(__file__).resolve().parent
STATUS_DIR = BASE / "run_status"
DATA_DIR = BASE / "data_l4"
DEFAULT_BASE_SEED = 20260728
DEFAULT_N_REPS = 3
DEFAULT_TRIALS = 12


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--run-label")
    parser.add_argument("--output-tag")
    parser.add_argument(
        "--input-stimuli",
        type=Path,
        help=(
            "Image directory or ZIP archive. By default, use input_stimuli/ "
            "or extract input_stimuli.zip beside this script."
        ),
    )
    parser.add_argument(
        "--no-strict-determinism",
        action="store_true",
        help="Disable deterministic PyTorch/CUDA algorithms.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration/imports without launching a sweep.",
    )
    return parser.parse_args()


def ensure_preimport_environment(args, run_label, output_tag):
    """Set determinism variables before stats_collector imports PyTorch."""
    desired = str(args.base_seed)
    if os.environ.get("PYTHONHASHSEED") == desired:
        return

    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = desired
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment["NEURALSHEET_COMBINED_RUN_LABEL"] = run_label
    environment["NEURALSHEET_COMBINED_OUTPUT_TAG"] = output_tag
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(run_label, suffix, value):
    STATUS_DIR.mkdir(exist_ok=True)
    (STATUS_DIR / f"{run_label}.{suffix}").write_text(str(value))


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def prepare_input_stimuli(requested_path):
    """Resolve an image directory, extracting the adjacent ZIP when needed."""
    candidate = requested_path.expanduser().resolve() if requested_path else None
    if candidate is None:
        directory = BASE / "input_stimuli"
        archive = BASE / "input_stimuli.zip"
    elif candidate.is_dir():
        return candidate
    elif candidate.suffix.lower() == ".zip":
        directory = BASE / "input_stimuli"
        archive = candidate
    else:
        raise FileNotFoundError(
            f"--input-stimuli must be an existing directory or ZIP: {candidate}"
        )

    if directory.is_dir():
        return directory
    if not archive.is_file():
        raise FileNotFoundError(
            "No stimuli found. Put input_stimuli/ or input_stimuli.zip beside "
            "this script, or pass --input-stimuli PATH."
        )

    print(f"Extracting {archive.name} (this needs about 11 GB of free space)...")
    destination = BASE.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"Unsafe path in stimuli ZIP: {member.filename}")
        bundle.extractall(destination)
    if not directory.is_dir():
        raise RuntimeError(
            f"{archive} did not contain the expected input_stimuli/ directory"
        )
    return directory


def phase(run_label, name, function):
    write_status(run_label, f"{name}.started", timestamp())
    function()
    write_status(run_label, f"{name}.done", timestamp())


def generate_figures(outputs, figure_dir):
    """Reproduce the six fitted sweep panels headlessly."""
    import matplotlib

    matplotlib.use("Agg")
    import torch
    from helpers import map_plotting

    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "fidelity": figure_dir / "accuracy.svg",
        "dimensionality": figure_dir / "dimensionality.svg",
        "fidelity_dimensionality": figure_dir / "accdimratio.svg",
        "topological_robustness": figure_dir / "topological_robustness.svg",
        "stability": figure_dir / "stability.svg",
        "robustness": figure_dir / "robustness.svg",
        "micro_fidelity_dimensionality": figure_dir / "efficiency.svg",
    }
    rep_spread = "2std"

    accdim = torch.load(outputs["accdim"], map_location="cpu", weights_only=False)
    acc_baseline = float(accdim["null_accuracy_mean"])
    fidelity = (
        torch.as_tensor(accdim["heldout_accuracy"]).float()
        - acc_baseline
    ) / (1 - acc_baseline)
    neighborhood_x = (
        torch.as_tensor(accdim["trialvar"]).float() / 3
    ) ** 2 * torch.pi
    dim_baseline = int(accdim["input_pca_dimensionality"])
    dimensionality = (
        torch.as_tensor(accdim["se_pca_tracker"]).float() / dim_baseline
    )

    result = map_plotting.plot_wiring_efficiency_six_panel_summary(
        neighborhood_x,
        fidelity,
        dimensionality,
        outputs["noise_topo"],
        outputs["noise_sp"],
        acc_baseline=acc_baseline,
        dim_baseline=dim_baseline,
        robustness_threshold=0.95,
        omit_first_k=1,
        legend_labels=("Δ=1", "Δ=4", "Δ=9"),
        rep_spread=rep_spread,
        omit_second_from_dimensionality_fit=True,
        figure_paths=(
            str(figure_paths["fidelity"]),
            str(figure_paths["dimensionality"]),
            str(figure_paths["fidelity_dimensionality"]),
            str(figure_paths["topological_robustness"]),
            str(figure_paths["robustness"]),
            str(figure_paths["micro_fidelity_dimensionality"]),
        ),
        stability_path=str(figure_paths["stability"]),
        show=False,
    )
    import matplotlib.pyplot as plt
    plt.close(result["figure"])

    missing = [str(path) for path in figure_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Expected figure files were not created: {missing}")
    return figure_paths


def main():
    args = parse_args()
    if args.n_reps < 1:
        raise ValueError("--n-reps must be at least 1")
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")

    generated_tag = time.strftime("%Y%m%d_%H%M%S") + (
        f"_{args.n_reps}reps_{args.trials}trials_current_params"
    )
    output_tag = (
        args.output_tag
        or os.environ.get("NEURALSHEET_COMBINED_OUTPUT_TAG")
        or generated_tag
    )
    run_label = (
        args.run_label
        or os.environ.get("NEURALSHEET_COMBINED_RUN_LABEL")
        or f"accdim_and_noise_{output_tag}"
    )
    ensure_preimport_environment(args, run_label, output_tag)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.chdir(BASE)

    input_stimuli = prepare_input_stimuli(args.input_stimuli)
    os.environ["NEURALSHEET_INPUT_STIMULI"] = str(input_stimuli)

    import torch
    import stats_collector

    strict = not args.no_strict_determinism
    outputs = {
        "accdim": DATA_DIR / f"data_accdim_{output_tag}.pt",
        "noise_sp": DATA_DIR / f"data_noise_sp_{output_tag}.pt",
        "noise_topo": DATA_DIR / f"data_noise_topo_{output_tag}.pt",
    }
    figure_dir = BASE / "figures" / f"accdim_and_noise_{output_tag}"
    if os.environ.get("NEURALSHEET_STANDALONE_SOURCE"):
        sources = [Path(os.environ["NEURALSHEET_STANDALONE_SOURCE"]).resolve()]
    else:
        sources = [
            BASE / "neuralsheet.py",
            BASE / "stats_collector.py",
            BASE / "nn_template.py",
            BASE / "helpers" / "wiring_efficiency_utils.py",
            BASE / "helpers" / "map_plotting.py",
            Path(__file__).resolve(),
        ]
    config = {
        "run_label": run_label,
        "output_tag": output_tag,
        "runner_pid": os.getpid(),
        "working_directory": str(BASE),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": args.base_seed,
        "n_reps": args.n_reps,
        "trials": args.trials,
        "input_stimuli": str(input_stimuli),
        "strict_determinism": strict,
        "phases": [
            "accuracy_dimensionality",
            "noise_sweeps_sp_then_topo",
            "six_sweep_panels",
        ],
        "outputs": {name: str(path) for name, path in outputs.items()},
        "figure_directory": str(figure_dir),
        "source_sha256": {str(path.relative_to(BASE)): sha256(path) for path in sources},
        "environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }

    required = [input_stimuli, *sources]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current collector configuration")
    collisions = [str(path) for path in outputs.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {collisions}")

    if args.check_only:
        print(json.dumps(config, indent=2, sort_keys=True))
        print("check-only validation passed; no sweep was launched")
        return

    DATA_DIR.mkdir(exist_ok=True)
    write_status(run_label, "pid", os.getpid())
    write_status(run_label, "started", timestamp())
    write_status(run_label, "launch_config.json", json.dumps(config, indent=2, sort_keys=True))

    try:
        phase(
            run_label,
            "accdim",
            lambda: stats_collector.collect_stats(
                n_reps=args.n_reps,
                trials=args.trials,
                output_file=str(outputs["accdim"]),
                base_seed=args.base_seed,
                strict_determinism=strict,
            ),
        )
        phase(
            run_label,
            "noise",
            lambda: stats_collector.run_noise_sweeps(
                n_reps=args.n_reps,
                trials=args.trials,
                output_files=(str(outputs["noise_sp"]), str(outputs["noise_topo"])),
                base_seed=args.base_seed,
                strict_determinism=strict,
            ),
        )
        phase(
            run_label,
            "figures",
            lambda: generate_figures(outputs, figure_dir),
        )
    except Exception:
        write_status(run_label, "error", traceback.format_exc())
        write_status(run_label, "exit", 1)
        raise
    else:
        write_status(run_label, "done", timestamp())
        write_status(run_label, "exit", 0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

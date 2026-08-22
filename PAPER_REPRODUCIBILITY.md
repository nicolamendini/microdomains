# Reproducing the paper data

## Run

Install the dependencies in `requirements-paper.txt`, provide the preprocessed
natural images as `input_stimuli/`, and run:

```bash
python run_all_paper_experiments.py --download-stringer
```

The Stringer mouse recording is optional. Omit `--download-stringer` to
produce every model panel without the empirical mouse UMAP. To preprocess a
source image directory with the repository implementation, run:

```bash
python process.py SOURCE_IMAGE_DIRECTORY input_stimuli
```

The full experiment requires CUDA and is long-running. A short installation
and model-path test is available as:

```bash
python run_all_paper_experiments.py --smoke-test
```

After collection, open `paper_panels.ipynb`. Its displacement/connectivity
cell lets you select the subtractive weight threshold, retained connection
fraction, and any non-negative displacement sigma. These inexpensive spatial
readouts are computed live from the saved trained maps.

The repository also includes the compact numerical panel bundle and linearly
quantized connectivity fields needed to open the notebook without retraining.
Full-precision checkpoints and the natural-image training set are omitted
because of their size; a new run regenerates them.

## What is generated

- Three-repeat fidelity/complexity and robustness sweeps, including the
  fitted reference-point and robustness figures.
- Four trained maps: 60x60 macro- and micro-GCAL maps for organisation and
  response panels, plus 90x90 maps for the central-45x45 displacement analysis
  and full-span connectivity fields.
- Orientation tuning, receptive fields, retinotopy, response Fourier spectra,
  grating-response UMAPs, and the optional Stringer UMAP.

One fixed, separately seeded set of 10,000 post-training transformations is
reused for reconstruction fidelity, LGN/V1 PCA dimensionality, and every
clean-versus-noisy robustness comparison. Three constant-input null decoders
provide the reconstruction baseline. Seeds, source hashes, environment
details, and the input-data fingerprint are stored with the numerical output.

## Main files

- `run_all_paper_experiments.py` — single entry point.
- `stats_collector.py` — replicated fidelity, PCA, null-decoder, and noise data.
- `paper_stats_collector.py` — four trained maps and panel data.
- `neuralsheet.py` — cortical model and plasticity.
- `process.py` — image preprocessing.
- `helpers/wiring_efficiency_utils.py` — datasets and shared metrics.
- `helpers/map_plotting.py` — sweep fitting and plotting.
- `helpers/paper_panel_plotting.py` — spatial and population panel plotting.
- `paper_panels.ipynb` — display notebook.

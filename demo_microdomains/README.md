# Microdomain self-organisation demo

Open `github_self_organisation_demo.ipynb` and run the cells in order. The
notebook contains controls and calls only; its plotting, analysis, animation,
and training backend is packaged in `helpers/microdomain_demo.py` here.

The folder is also the portable, GitHub-rendered presentation bundle: the
notebook Markdown resolves all shipped figures from `demo_assets/microdomain/`,
so the results remain visible without downloading the multi-gigabyte training
archive. Re-running the training cells additionally requires the parent model
repository's `neuralsheet.py`, shared `helpers/`, natural-image corpus, and a
CUDA-capable PyTorch environment.

The folder contains all presentation assets and compact external-analysis data:

- `demo_assets/microdomain/`: the model-architecture figure and GitHub-visible
  PNG/GIF results;
- `data/cellular_orientation_displacement/`: baseline orientation-map files
  from the public [Chen et al. macaque V1 dataset](https://doi.org/10.5281/zenodo.20053907)
  and the derived displacement summaries;
- `data/umap/`: copied topographic/salt-and-pepper response tensors and the
  compact four-panel UMAP cache, using high-arousal trials from
  [Stringer recording 1](https://doi.org/10.25378/janelia.8279387.v3) only.

The multi-gigabyte trained snapshot archive and natural-image corpus remain in
the repository-level `data_l4/` and `input_stimuli/` directories rather than
being duplicated. The shared helper resolves those paths from its own location,
so the notebook works whether Jupyter starts in this folder or at repository
root.

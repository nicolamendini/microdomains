# Self-organisation of functional cortical maps without macroscopic spatial patterning — Preprint

Preprint by Nicola Mendini and Stuart P. Wilson.

This study uses a self-organising model of cortical map development to show how fine-scale, coupled functional domains can preserve coding properties and robust dynamics while becoming difficult to detect as a macroscopic spatial pattern. The results offer a model for how structured cortical self-organisation can appear salt-and-pepper at the cortical surface.

[Read the preprint](./self_organisation_without_macroscopic_patterning_preprint.pdf)

## Cortical microdomain self-organisation demo

The [complete demo notebook](./demo_microdomains/github_self_organisation_demo.ipynb)
follows a 100 × 100 salt-and-pepper cortical sheet through two epochs of
natural-image learning. It connects the model's local circuit plasticity to
orientation-map formation, reconstruction, dimensionality, robustness, wiring
efficiency, macaque V1 measurements, and response geometry. The notebook is
kept deliberately short; its reusable collection and plotting code lives in
the accompanying [`demo_microdomains`](./demo_microdomains/) folder.

### Natural-image drive

The fixed held-out LGN crops provide a common reference for input intensity,
exact-zero sparsity, spatial power, and effective dimensionality.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/lgn_inputs.png" width="100%" alt="Natural-image LGN inputs and summary statistics">
</p>

### Emergence of orientation and retinotopic structure

Orientation preference, horizontal retinotopy, a central retinotopic fishnet,
and orientation Fourier power are tracked together throughout learning.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/map_learning.gif" width="100%" alt="Animation of orientation-map and retinotopy formation">
</p>

The same snapshots expose the development of afferent receptive fields and
cross-domain excitation.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/weight_learning.gif" width="70%" alt="Animation of afferent and lateral plasticity">
</p>

### Reconstruction, dimensionality, and robustness

A fixed synthetic face is evaluated—not interpolated—at every archived
learning snapshot, showing the corresponding L4 response and decoder output.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/synthetic_learning.gif" width="85%" alt="Tracked synthetic-face activity and reconstruction">
</p>

The leading spatial principal components and the V1-to-LGN dimensionality
ratio evolve alongside the learned representation.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/dimensionality.gif" width="100%" alt="Animation of PCA geometry and effective dimensionality">
</p>

Robustness is measured with a fixed input and a matched noise realization at
every snapshot.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/robustness.gif" width="100%" alt="Animation of response robustness to noise">
</p>

### Wiring efficiency and cellular-scale map structure

A seeded local permutation tests how small anatomical displacements can hide
macroscopic orientation order while retaining local functional structure.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/scattered_learning.gif" width="100%" alt="Animation of the locally scattered orientation map">
</p>

The corresponding biological analysis compares cellular orientation
measurements in three densely sampled macaque V1 fields with circularly
smoothed maps and quantifies each cell's nearest exact-orientation match.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/macaque_displacement_summary.png" width="100%" alt="Macaque V1 cellular measurements and smoothed orientation maps">
</p>

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/macaque_displacement_links.png" width="100%" alt="Example macaque soma-to-map orientation correspondences">
</p>

### Response geometry

The final comparison rotates matched three-dimensional UMAP embeddings of raw
gratings, topographic-model responses, salt-and-pepper-model responses, and one
experimental recording.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/rotating_umap.gif" width="100%" alt="Rotating four-panel UMAP comparison">
</p>

The large trained snapshot archive and natural-image corpus are intentionally
not duplicated in this repository. The folder includes the presentation
assets, compact external-analysis data, notebook, and all demo-specific source
code needed to inspect how the results were produced.

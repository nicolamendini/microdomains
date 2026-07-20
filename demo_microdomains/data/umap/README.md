# Four-panel rotating UMAP data

`four_panel_umap_embeddings.npz` is the compact plotting cache used by the
GitHub demo. It contains four 3D UMAP coordinates and their orientation labels:

1. 4,000 generated static gratings (2,000 reproducibly retained for display);
2. 2,000 topographic-model grating responses;
3. 2,000 salt-and-pepper-model grating responses;
4. high-arousal responses from public Stringer random-phase recording 1 only
   (1,916 trials fitted and displayed).

The fit matches the source `project-microdomains/analysis.ipynb` controls:
50 PCA components, UMAP `n_neighbors=50`, `min_dist=0.2`, three output
components, random state 0, and a 180-degree orientation colour period. Static
gratings are 30x30 pixels with a 12-pixel wavelength. All inferred cortical
depths are included.

`topo_simu_codes.pt` and `sp_simu_codes.pt` are copied source response files
from `Desktop/project-microdomains`. The several-gigabyte raw Stringer files
are not duplicated; the recording-1 embedding needed by this demo is stored in the
compact NPZ cache above.

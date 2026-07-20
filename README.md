# Self-organisation of functional cortical maps without macroscopic spatial patterning — Preprint

Preprint by Nicola Mendini and Stuart P. Wilson.

This study uses a self-organising model of cortical map development to show how fine-scale, coupled functional domains can preserve coding properties and robust dynamics while becoming difficult to detect as a macroscopic spatial pattern. The results offer a model for how structured cortical self-organisation can appear salt-and-pepper at the cortical surface.

[Read the preprint](./self_organisation_without_macroscopic_patterning_preprint.pdf)

## Cortical microdomain self-organisation demo 🧠

**Can a seemingly random salt-and-pepper cortex be the product of
self-organisation?** The [complete demo notebook](./demo_microdomains/github_self_organisation_demo.ipynb)
starts from a slightly more playful version of the question: what if the map
is not missing, but hiding? 🧂 It follows a 100 × 100 V1 sheet through two
epochs of natural-image learning. The model builds an orderly fabric of tiny,
interconnected domains; modest neuronal displacement then makes that structure
look random without destroying what the network learned.

The feature we follow is **orientation preference**: which edge angle makes a
V1 neuron respond most strongly. In a smooth orientation map, nearby neurons
prefer similar angles; in a salt-and-pepper map, different preferences appear
intermixed at cellular scale. The puzzle is whether that apparently untidy
arrangement can still grow from orderly learning rules.

**Self-organisation** means that nobody supplies the finished map—not a
supervisor, a label, or a built-in cortical blueprint. Each neuron responds to
its own input, interacts with other neurons, and adjusts its connections using
local activity. When those small steps are repeated many times, population-level
structure can emerge by itself. This demo asks whether the same kind of process
can produce structure that later *looks* random.

Reusable collection and plotting code lives
in the accompanying [`demo_microdomains`](./demo_microdomains/) folder.

### 1. Meet micro-GCAL: local competition, distant cooperation 🤝

Before the learning begins, let's introduce the model itself! Read it from
the bottom up: a visual stimulus is converted into sparse, contrast-normalised
activity in the **LGN**, which projects to a recurrent sheet representing
**V1 layer 4**.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/micro_gcal_architecture.png" width="39%" alt="Micro-GCAL architecture with LGN input, a recurrent V1 sheet, short-range excitation, longer-range inhibition, and cross-domain excitation">
</p>

Each cortical neuron combines four inputs:

- **Afferent input** from a small local LGN patch. Plastic afferent weights
  become the neuron's visual receptive field—the region and pattern in the
  visual input that drive it.
- **Short-range excitation (SRE)** from its nearest cortical neighbours. It
  lets nearby co-active neurons reinforce one another and settle as a local
  patch.
- **Longer-range inhibition (LRI)** from a wider surround. It creates
  competition, separates neighbouring patches, and prevents activity from
  spreading across the whole sheet.
- **Cross-domain excitation (CDE)** at the widest scale. CDE is not a uniform
  excitatory halo: it is learned selectively between neurons that are
  repeatedly strongly co-active, allowing separated but functionally related
  patches to cooperate.

Their spatial ordering is **SRE < LRI < CDE**: local cooperation, broader
competition, then selective cooperation again at the longest scale. The first
two interactions keep individual domains small; CDE connects them with learned
bridges and stabilises their recurrent responses. The result is **many tiny,
coupled domains**—not one large smooth domain, and not a collection of
independent random neurons. 🏝️

For every stimulus, activity is updated recurrently until it settles. Hebbian
plasticity then strengthens co-active afferent and recurrent connections,
while adaptive thresholds and gain control keep activity sparse and balance
feedforward with recurrent drive. Repeating this settle–learn cycle jointly
shapes receptive fields, connectivity, and the cortical map.

### 2. Give the cortex something to look at 👀

Natural-image patches pass through an LGN-like contrast filter and gain
control. V1 receives sparse edges and textures, with no orientation labels and
no hidden answer sheet. It has to work out the useful structure for itself.
The LGN is the relay between retina and cortex; here its simplified job is to
emphasise local light–dark boundaries and normalise their contrast.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/lgn_inputs.png" width="100%" alt="Natural-image LGN inputs and summary statistics">
</p>

### 3. Let the neurons negotiate 💡

Each input starts a brief recurrent conversation: excite, inhibit, settle,
learn, repeat. Tiny orientation domains gradually appear. The Fourier ring
reveals their preferred spacing, while the retinotopic fishnet bends locally
without losing the global plot.

Here is the visual key. In the orientation map, colour is preferred edge angle,
so same-coloured neighbours form a domain. A ring in Fourier space means that
similar features repeat at a characteristic distance in every direction. The
retinotopy panels instead ask *where in the image does each cortical location
look?* A globally ordered but locally bent grid means neighbouring regions of
visual space are still represented nearby, despite fine-scale distortions.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/map_learning.gif" width="100%" alt="Animation of orientation-map and retinotopy formation">
</p>

At the same time, afferent receptive fields become selective and cross-domain
excitation learns which separated patches should cooperate. The domains are
small, but they are already exchanging phone numbers.

Each small tile is one neuron's incoming connection pattern. Bright, structured
afferent tiles indicate selective visual receptive fields; bright patches in
the CDE tiles show which more distant cortical partners have acquired strong
excitatory links.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/weight_learning.gif" width="70%" alt="Animation of afferent and lateral plasticity">
</p>

### 4. Can it remember a face? 🙂

A fixed synthetic face makes reconstruction progress easy to see. A decoder
tries to rebuild the input using only the V1 population activity. The final
curve averages reconstruction similarity over the full held-out set: higher
cosine similarity means the cortical code preserves more of the input.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/synthetic_learning.gif" width="85%" alt="Tracked synthetic-face activity and reconstruction">
</p>

PCA then reveals that ten thousand neurons do not need ten thousand independent
opinions. The V1 code uses fewer effective dimensions than its LGN input: a
compact population representation, rather than a lossy shrug.

In plain terms, PCA counts how many independent patterns are needed to describe
most of the population's variation. If many neurons change together, many
individual responses can be summarised by fewer shared patterns. The plot uses
the number required to explain 95% of the variance.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/dimensionality.gif" width="100%" alt="Animation of PCA geometry and effective dimensionality">
</p>

Next we give the recurrent dynamics a noisy day ⚡. A matched perturbation at
every snapshot shows that selective interaction helps the population return
to nearly the same answer.

The clean and noisy activity panels show the same cortical region responding
to the same input. Their cosine similarity is 1 when the two population codes
point in exactly the same direction, and falls as noise changes the response.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/robustness.gif" width="100%" alt="Animation of response robustness to noise">
</p>

### 5. Shake the seating plan 🌀

**Experiment 1 — a controlled shuffle.** After learning is complete, we move
each model neuron with a one-to-one, seeded Gaussian permutation whose mean
displacement is two lattice locations. The receptive fields and orientation
preferences do not change; only their cortical addresses do. This modest
shuffle preserves short-range clustering—consistent with cellular-scale V1
measurements ([Ringach et al., 2016](https://doi.org/10.1038/ncomms12270))—but
erases the global Fourier signature. The orderly map has gone undercover as
salt-and-pepper cortex.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/scattered_learning.gif" width="100%" alt="Animation of the locally scattered orientation map">
</p>

**Experiment 2 — an estimate from real cortex.** We cannot rewind cortical
development, but dense two-photon recordings from superficial V1 in two awake,
fixating macaques offer a clue. The source experiment sampled two 850 × 850 µm
fields per animal with gratings at 12 axial orientations, spaced by 15°. We
chose these data because the unusually dense spatial sampling and many tested
orientations give a much less discretised view of the map than the smaller
orientation sets often used in physiology. We retain all 12 orientations and
the significantly tuned cells rather than coarsening them into bins. See
[Chen et al. (2026)](https://doi.org/10.7554/eLife.107518) and the
[source dataset](https://doi.org/10.5281/zenodo.20053907).

For each of the three densest fields, we smooth axial orientation preferences
in complex form, using a 100 µm spatial scale and leave-one-out prediction at
each soma. We then measure the shortest exact distance to the contour on which
that smooth map predicts the neuron's preferred orientation. Points beyond
350 µm remain visible in the scatter but are excluded from the displayed means
and correspondence links. This is a model-based displacement proxy—not a
literal measurement of neurons migrating during development.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/macaque_displacement_summary.png" width="90%" alt="Macaque V1 cellular measurements and smoothed orientation maps">
</p>

The first figure shows the population-level estimate: measured cellular maps,
their smooth inferred counterparts, and the resulting displacement
distributions. The second makes the geometry tangible for 20 fixed example
neurons per field. Each coloured dot is a soma, its black × is the closest
same-orientation point on the inferred map, and the connecting segment is the
estimated displacement. These links visualise the calculation

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/macaque_displacement_links.png" width="90%" alt="Example macaque soma-to-map orientation correspondences">
</p>

### 6. Leave the sheet and find the hidden shape ✨

Displacement hides structure on the cortical sheet, but it does not scramble
the learned responses. Rotating UMAPs of gratings, topographic-model activity,
salt-and-pepper-model activity, and high-arousal mouse V1 data bring the order
back into view as smooth, folded response geometries. The map may disappear
from cortical space while its shape survives in the code.

Each dot is the activity of an entire population for one stimulus or trial;
UMAP places dots nearby when those high-dimensional activity patterns are
similar. Colour marks grating orientation. Smooth colour progressions around
the folded shapes therefore reveal an ordered representation in neural
activity—even when the neurons no longer form an obvious map on the sheet.
From left to right, the panels provide a stimulus baseline, the topographic
simulation, the salt-and-pepper simulation, and recorded mouse V1 activity.

The mouse comparison uses the 1,916 high-arousal trials from recording 1 of the
[Stringer et al. public dataset](https://doi.org/10.25378/janelia.8279387.v3)
([paper](https://doi.org/10.1038/s41586-019-1346-5)). Colours are fixed to each
sample before rotation, so the animation changes the viewpoint—not the labels.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/rotating_umap.gif" width="100%" alt="Rotating four-panel UMAP comparison">
</p>

### Take-home idea 🏡

Salt-and-pepper need not mean structureless. It may mean **beautifully
organised, then very lightly shuffled**—with selective connectivity, robust
dynamics, and an orderly population representation still hiding underneath.

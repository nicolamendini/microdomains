# Self-organisation of functional cortical maps without macroscopic spatial patterning — Preprint

Preprint by Nicola Mendini and Stuart P. Wilson.

This study uses a self-organising model of cortical map development to show how fine-scale, coupled functional domains can preserve coding properties and robust dynamics while becoming difficult to detect as a macroscopic spatial pattern. The results offer a model for how structured cortical self-organisation can appear salt-and-pepper at the cortical surface.

[Read the preprint](./self_organisation_without_macroscopic_patterning_preprint.pdf)

## Cortical microdomain self-organisation demo

**Can a seemingly random salt-and-pepper cortex be the product of
self-organisation?** The [complete demo notebook](./demo_microdomains/github_self_organisation_demo.ipynb)
starts from a slightly more playful version of the question: what if the map
is not missing, but hiding? 🧂 It follows a 100 × 100 V1 sheet through two
epochs of natural-image learning. The model builds an orderly fabric of tiny,
interconnected domains; modest neuronal displacement then makes that structure
look random without destroying what the network learned.

The notebook develops this argument one step at a time. Every section starts
with a short narrative and places the corresponding technical explanation in
a collapsible block, so it can be read as either a visual story or a
reproducible modelling workflow. Reusable collection and plotting code lives
in the accompanying [`demo_microdomains`](./demo_microdomains/) folder.

### 1. Meet micro-GCAL: local competition, distant cooperation

Before the learning begins, Figure 4 introduces the model itself. Read it from
the bottom up: a visual stimulus is converted into sparse, contrast-normalised
activity in the **LGN**, which projects to a recurrent sheet representing
**V1 layer 4**. This is the only cortical layer simulated in this demo, so the
rest of the text simply calls it **V1** or **cortex**.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/micro_gcal_architecture.png" width="65%" alt="Micro-GCAL architecture with LGN input, a recurrent V1 sheet, short-range excitation, longer-range inhibition, and cross-domain excitation">
</p>

Each cortical neuron combines four inputs:

- **Afferent input** from a small local LGN patch. Plastic afferent weights
  become the neuron's visual receptive field.
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

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/lgn_inputs.png" width="100%" alt="Natural-image LGN inputs and summary statistics">
</p>

### 3. Let the neurons negotiate

Each input starts a brief recurrent conversation: excite, inhibit, settle,
learn, repeat. Tiny orientation domains gradually appear. The Fourier ring
reveals their preferred spacing, while the retinotopic fishnet bends locally
without losing the global plot.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/map_learning.gif" width="100%" alt="Animation of orientation-map and retinotopy formation">
</p>

At the same time, afferent receptive fields become selective and cross-domain
excitation learns which separated patches should cooperate. The domains are
small, but they are already exchanging phone numbers.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/weight_learning.gif" width="70%" alt="Animation of afferent and lateral plasticity">
</p>

### 4. Can it remember a face? 🙂

A fixed synthetic face makes reconstruction progress easy to see. The new
final panel keeps our cheerful volunteer honest by tracking average fidelity
over the full held-out evaluation set, rather than reporting the face alone.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/synthetic_learning.gif" width="85%" alt="Tracked synthetic-face activity and reconstruction">
</p>

PCA then reveals that ten thousand neurons do not need ten thousand independent
opinions. The V1 code uses fewer effective dimensions than its LGN input: a
compact population representation, rather than a lossy shrug.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/dimensionality.gif" width="100%" alt="Animation of PCA geometry and effective dimensionality">
</p>

Next we give the recurrent dynamics a noisy day ⚡. A matched perturbation at
every snapshot shows that selective interaction helps the population return
to nearly the same answer.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/robustness.gif" width="100%" alt="Animation of response robustness to noise">
</p>

### 5. Shake the seating plan 🌀

A mean displacement of only two model locations preserves short-range
clustering but erases the fine global periodicity. Nothing about the learned
responses has changed; only the neurons' chairs have moved. The orderly map
has gone undercover as salt-and-pepper cortex.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/scattered_learning.gif" width="100%" alt="Animation of the locally scattered orientation map">
</p>

How plausible is that amount of chair-moving? We cannot rewind cortical
development, but dense macaque V1 recordings offer a clue. We compare each
measured soma with the nearest location where a smooth underlying map predicts
the same orientation.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/macaque_displacement_summary.png" width="100%" alt="Macaque V1 cellular measurements and smoothed orientation maps">
</p>

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/macaque_displacement_links.png" width="100%" alt="Example macaque soma-to-map orientation correspondences">
</p>

### 6. Leave the sheet and find the hidden shape ✨

Displacement hides structure on the cortical sheet, but it does not scramble
the learned responses. Rotating UMAPs of gratings, topographic-model activity,
salt-and-pepper-model activity, and high-arousal mouse V1 data bring the order
back into view as smooth, folded response geometries. The map may disappear
from cortical space while its shape survives in the code.

<p align="center">
  <img src="./demo_microdomains/demo_assets/microdomain/rotating_umap.gif" width="100%" alt="Rotating four-panel UMAP comparison">
</p>

### Take-home idea

Salt-and-pepper need not mean structureless. It may mean **beautifully
organised, then very lightly shuffled**—with selective connectivity, robust
dynamics, and an orderly population representation still hiding underneath.

## 1. Cone Algebra in z-Space (Operational Semantics of Commitment)

In continuous TAM, a *cone* is not stored explicitly in trajectory space but is **induced** by a latent commitment \( z \) together with a situation \( s_0 \). Cone algebra refers to the set of **operations on commitments** that produce predictable transformations of the induced affordance cone. These operations live in **z-space**, not trajectory space, and correspond to agent-level moves such as *tightening*, *widening*, *hedging*, or *extending horizon*. Formally, we treat the cone as a functional \( \mathcal{C}(s_0, z) \subset \mathcal{T} \), and cone algebra consists of operators \( \mathcal{O} : z \mapsto z' \) such that \( \mathcal{C}(s_0, z') \) has a known inclusion or deformation relationship to \( \mathcal{C}(s_0, z) \). For example, decreasing predicted log-variance corresponds to cone contraction, increasing p_stop shifts mass earlier in time, and linear transforms of z can correspond to coarse-to-fine abstraction. Importantly, these are **symbolic operations without symbols**: algebraic structure arises from geometric regularities learned by the tube network, not from explicit enumeration.

---

## 2. Memory over Regions of z without Discretization (Latent Commitment Recall)

Rather than storing discrete ports, the agent can maintain **memory over regions of z-space** by learning a continuous density or metric structure over past commitments and their outcomes. Each episode produces a tuple \( (z, s_0, \text{bind success}, \text{cone stats}) \), which can be written to a replay memory. Memory access is then **query-based**, not index-based: given a new situation \( s_0 \), the actor proposes a candidate z, and auxiliary mechanisms (e.g. kNN in z-space, kernel density estimates, or a learned critic over z) bias sampling toward regions with favorable tradeoffs between tightness and reliability. Crucially, nothing is discretized: regions emerge as **high-density basins** in latent space, not as labeled categories. This supports gradual drift, partial reuse, and smooth forgetting — properties impossible with hard port IDs.

---

## 3. Emergent Concept Reuse without Port IDs (Attractors as Concepts)

In this framing, *concepts* are not tokens but **attractors in z-space**: regions repeatedly induced across different situations because they support stable, reliable cones under environmental variation. When similar commitments are repeatedly sampled and succeed, gradient descent naturally sharpens the mapping \( q(z \mid s_0) \) so that many situations funnel into overlapping z-regions. This produces reuse without naming: the same geometric commitment is instantiated again and again because it works. Over time, these attractors can acquire structure (e.g. lower KL cost, lower dual pressure, stable horizons), making them “concept-like.” If desired, discrete symbols can be *read out* later by clustering or probing these regions, but this is an **epiphenomenon**, not a control mechanism. The agent reasons by *re-entering* these regions, not by selecting labels.

---

## 4. Category Theory & Fiber Bundles (The Clean Mathematical Backbone)

TAM with continuous ports is naturally described as a **fiber bundle**. The base space is the space of situations \( \mathcal{S} \); the fiber over each situation is the space of latent commitments \( \mathcal{Z}_{s} \); and the total space maps to trajectory distributions \( \mathcal{T} \). A port is not an object but a **section**: a rule that assigns a commitment \( z \) to each situation. Binding corresponds to choosing a section locally and holding it fixed over time. The world acts as a functor that maps sections (commitments) to realized trajectories, and binding failure corresponds to a violation of expected morphisms. Cone algebra corresponds to endomorphisms on fibers; memory corresponds to measures over fibers; and emergent concepts correspond to stable submanifolds. This framing explains why discreteness is unnecessary: the architecture is about **structure-preserving mappings**, not symbolic selection.

# TA Properties

Derived properties and interpretations of Trajectory-Affordance.

## Representational Capacity

The state space $\mathcal{X}$ contains exactly those states that could arise from some inference:

$$
\mathcal{X} = \bigcup_{p \in \mathcal{P}, x \in \mathcal{X}, \vec{c} \in \mathcal{C}^*} \{ \tau[i] \mid \tau = \mathsf{Infer}_p(x, \vec{c}), \, 0 \le i < |\tau| \}
$$

The trajectory space $\mathcal{T}(\mathcal{X})$ is the cumulative affordance across all ports, states, and contexts:

$$
\mathcal{T}(\mathcal{X}) = \bigcup_{p \in \mathcal{P}, x \in \mathcal{X}, \vec{c} \in \mathcal{C}^*} \Phi_p(x, \vec{c})
$$

What is representable is bounded by what the ports can collectively infer and afford.

## Agency

Agency correlates inversely with cone width. A narrow cone commits to specific outcomes while a wide cone accepts many outcomes.

$$
\Phi_p(x, \vec{c}) = \mathcal{T}(x) \implies \text{agency} = 0
$$

When $\mathcal{X}$ (and by extension $\mathcal{T}(x)$) are finite, agency admits a simple cardinality ratio:

$$
\text{agency}(p, x, \vec{c}) = 1 - \frac{|\Phi_p(x, \vec{c})|}{|\mathcal{T}(x)|}
$$

In continuous spaces, analogous measures can be derived from relative volumes or measure-theoretic properties of $\Phi_p$

## Epistemic Failure

The most useful practical learning signal in TAM is not just task failure, but
epistemic failure under a committed local frame.

After binding a port, the actor has effectively committed to a local view of
trajectory space. When the realized trajectory falls outside the relevant cone,
the failure is informative in two ways:

- the selected local frame was too broad, too narrow, or simply wrong
- the atlas may need refinement through widening, narrowing, splitting, or
  proliferating charts

This makes contradiction more than a loss term. It becomes the signal that a
chosen way of understanding the current situation was not locally valid.

## Atlas Reading

One useful interpretation of TAM is as atlas learning over latent trajectory
space.

- situations act as base points
- ports expose local charts or projections into reachable future structure
- cones are admissible regions inside those local charts
- agency corresponds to the ability to make specific and accurate claims within
  those local coordinates

Under this reading, representational capacity is not just the number of states
the system can encode. It is also the richness and quality of the atlas it can
maintain over its own trajectory manifold.

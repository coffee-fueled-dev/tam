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

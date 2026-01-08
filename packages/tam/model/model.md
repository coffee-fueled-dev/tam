# The Trajectory-Affordance Model

The core cycle:

1. $\mathsf{A}$ is in a situation
2. $\mathsf{A}$ selects a port from those currently afforded
3. $\mathsf{A}$ binds the port
4. $\mathsf{W}$ responds with a context episode
5. $\mathsf{A}$ interprets the episode and evaluates whether it falls within expectations
6. A new situation arises; the cycle repeats

Ports define affordance cones: trajectories $\mathsf{A}$ is willing to accept for a given mode of interaction. Binding succeeds when the outcome lands in the cone; it fails otherwise.

---

### Actor $\mathsf{A}$

The actor selects modes of interaction and updates its state based on the world's response.

### World $\mathsf{W}$

The world is an external source of context.

All information $\mathsf{A}$ receives from $\mathsf{W}$ is expressed as values from the context domain $\mathcal{C}$.

When $\mathsf{A}$ binds a port, $\mathsf{W}$ responds with a sequence of context called an episode:

$$
e_{n \to n+1}
=
(c_{n,0}, c_{n,1}, \dots, c_{n,k}, c_{n+1})
$$

### State Space $\mathcal{X}$

The set of distinct states $\mathsf{A}$ is capable of representing.

### Trajectories $\mathcal{T}(\mathcal{X})$

Causal chains of states, represented as ordered tuples:

$$
\mathcal{T}(\mathcal{X})
\subseteq
\bigcup_{k \ge 0}
\{ (x_0, \dots, x_k) \mid x_i \in \mathcal{X} \}
$$

Trajectories from state $x_n$ are those in $\mathcal{T}(\mathcal{X})$ with $x_n$ as initial state:

$$
\mathcal{T}(x_n)
=
\{ \tau \in \mathcal{T}(\mathcal{X}) \mid \tau[0] = x_n \}
$$

### Ports $\mathcal{P}$

A port $p \in \mathcal{P}$ is a polysemous mode of interaction with $W$

Each port consists of an inference map that interprets context sequences as trajectories:

$$
\mathsf{Infer}_p : \mathcal{X} \times \mathcal{C}^* \to \mathcal{T}(\mathcal{X})
$$

As well as an affordance predicate on trajectories:

$$
\chi_p :
\mathcal{T}(\mathcal{X})
\times \mathcal{X}
\times \mathcal{C}^*
\to
\{\mathsf{true}, \mathsf{false}\}
$$

Given an internal state $x_n$ and a context sequence $\vec{c}_n \in \mathcal{C}^*$, the predicate determines an affordance cone:

$$
\Phi_p(x_n, \vec{c}_n)
=
\{ \tau \in \mathcal{T}(x_n)
\mid
\chi_p(\tau, x_n, \vec{c}_n)
\}
$$

Two ports $p$ and $q$ are equivalent when their inference maps and affordance cones coincide for all states and context sequences:

$$
p \equiv q
\iff
\forall x \in \mathcal{X}, \vec{c} \in \mathcal{C}^* :
\mathsf{Infer}_p(x, \vec{c}) = \mathsf{Infer}_q(x, \vec{c})
\land
\Phi_p(x, \vec{c}) = \Phi_q(x, \vec{c})
$$

### Situations

A situation is an indexed instance of internal state:

$$
s_n = (n, x_n)
$$

The index $n$ marks the step in the causal chain. Two situations $s_n$ and $s_m$ are distinct even if $x_n = x_m$.

In each situation $s_n$, there is a prior context $\vec{c}^{\,\text{prior}}_n \in \mathcal{C}^*$
which is some subsequence of all context received up to that point.

Each situation requires binding an afforded port, and each binding produces the next situation. A port is afforded in $s_n$ when its affordance cone is non-empty:

$$
\mathsf{Ports}(s_n)
=
\{ p \in \mathcal{P} \mid \Phi_p(x_n, \vec{c}^{\,\text{prior}}_n) \neq \emptyset \}
$$

---

### Binding

Binding is a transition from situation $s_n$ to situation $s_{n+1}$ via a port $p_n \in \mathsf{Ports}(s_n)$.

Upon binding, the world produces a context episode $e_{n \to n+1} \in \mathcal{C}^*$.

There is now a post context $\vec{c}^{\,\text{post}}_n \in \mathcal{C}^*$: some subsequence of all context received up to and including the episode.

The port interprets the episode as a trajectory:

$$
\hat{\tau}_n = \mathsf{Infer}_{p_n}(x_n, e_{n \to n+1})
\in
\mathcal{T}(x_n)
$$

The next situation arises:

$$
s_{n+1} = \hat{\tau}_n[\mathrm{end}]
$$

Binding succeeds when the inferred trajectory remains within the affordance cone:

$$
\hat{\tau}_n \in \Phi_{p_n}(x_n, \vec{c}^{\,\text{post}}_n)
$$

Binding fails when the inferred trajectory exits the cone:

$$
\hat{\tau}_n \notin \Phi_{p_n}(x_n, \vec{c}^{\,\text{post}}_n)
$$

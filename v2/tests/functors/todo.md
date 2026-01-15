The **minimal probe generator** whose only job is to create a _shared reference context_ that:

- excites the same **basin structure** (topology) across actors,
- but is **agnostic to each actor’s gauge** (rotation / permutation / sign),
- and is **small** (minimal number of probes),
- while still being **sufficient to fix gauge** when you train functors on probe-consistency.

---

## 0) What are probes, concretely?

A probe is just a state \(s\) (or observation) that you feed into each actor’s inference:

\[
z^\*(s) = \arg\min_z E(s,z)
\]

Then you look at which **basin** it lands in (or the direction of \(z^\*\) if continuous).

A good probe set makes different basins reliably fire in a **consistent** way.

---

## 1) The minimal probe set: “boundary + symmetry breaking”

If your environment has **K basins** (e.g., K goals), the minimal probe set you want is:

### **K “boundary probes”** + **1 “tie-break probe”**

So: **K+1 probes** total (that’s close to minimal in practice).

Why:

- Boundary probes cause the agent to “decide” between basins (high sensitivity → reveals labeling)
- One extra tie-break breaks reflection/rotation ambiguities that boundary probes alone might not resolve.

---

## 2) Minimal probe generator (environment-agnostic interface)

We’ll assume your environment can do at least one of these:

- **Option A (best):** sample an observation given a latent “goal index” / mode, even if you don’t label it.
- **Option B:** exposes the goal vectors \(g_k\) (CMG does).
- **Option C:** you can reset and read state \(x\), and you can set goals.

CMG is Option B/C, so we can do a very clean generator.

### Probe types

#### Probe type 1: Pairwise midpoints between goals (boundary probes)

For each basin \(k\), choose a neighbor \(j\) and create a midpoint:

\[
x\_{k} = \frac{g_k + g_j}{2}
\]

These are “decision boundary” probes.

Pick neighbor \(j\) as the _nearest goal_ to \(g_k\). That ensures the midpoint lies on the actual competing boundary.

That gives you K probes (one per basin).

#### Probe type 2: “asymmetric” probe (tie-break)

Pick a random direction \(u\) and slightly bias one coordinate:

\[
x\_{\text{tie}} = \bar g + \epsilon u,\quad \bar g = \frac{1}{K}\sum_k g_k
\]

This probe breaks global sign/permutation symmetries because it is not invariant under the same transformations as the midpoint set.

---

## 3) The actual generator (drop-in code)

```python
import numpy as np

def minimal_probe_set_from_goals(goals: np.ndarray, eps: float = 0.1, seed: int = 0):
    """
    goals: (K, d) goal vectors in state space.
    returns probes: (K+1, d)
    """
    rng = np.random.default_rng(seed)
    K, d = goals.shape

    # 1) boundary probes: midpoint to nearest neighbor for each goal
    probes = []
    for k in range(K):
        # nearest neighbor j != k
        diffs = goals - goals[k]
        dists = np.linalg.norm(diffs, axis=1)
        dists[k] = np.inf
        j = int(np.argmin(dists))
        probes.append(0.5 * (goals[k] + goals[j]))

    probes = np.stack(probes, axis=0)  # (K, d)

    # 2) tie-break probe: centroid + small random offset
    g_bar = goals.mean(axis=0)
    u = rng.normal(size=d)
    u = u / (np.linalg.norm(u) + 1e-8)
    tie = g_bar + eps * u

    probes = np.concatenate([probes, tie[None, :]], axis=0)
    return probes
```

**This is the minimal “good” probe generator for CMG-like forked-goal topologies.**

---

## 4) How to use probes to fix gauge (training signal)

You don’t need to “train on the probe environment.” You just add a _consistency regularizer_:

For each probe \(s_i\):

1. Get each actor’s best commitment \(z_A^\*(s_i), z_B^\*(s_i), z_C^\*(s_i)\)
2. Map them through functors, compare predicted vs direct:

Example (A→C consistency via B):

\[
\|F*{AC}(z_A^\*) - F*{BC}(F\_{AB}(z_A^\*))\|^2
\]

And **also** enforce “probe agreement”:

- compute which basin each actor chose on probes (by clustering z\* on the fly or nearest basin prototype)
- encourage the functors to map probe-induced basin IDs consistently

This is the asymmetry you were missing.

---

## 5) What “minimal” means here (stopping rule)

Start with K probes only (midpoints). Train. Measure:

- composition similarity (your metric)
- basin agreement accuracy on probes

If composition still flips sign/permutation, add the tie-break probe.

If still unstable, add **one more tie-break** (so K+2 total).

In practice, for K=3 on the unit circle, **4 probes is often enough** (3 midpoints + 1 tie-break).

---

## 6) Why this works (intuitively)

- Midpoints are where “choice happens,” so they expose the basin structure sharply.
- But midpoints alone are symmetric: rotations/permutations can explain them equally well.
- The tie-break is a single “landmark” that prevents those symmetries from being equally valid.

So you get gauge fixing with almost no extra structure.

---

## 7) If you want it even more minimal (K probes only)

If you _must_ try with exactly K probes, make them **cyclic midpoints** (a directed cycle) instead of nearest-neighbor midpoints:

\[
x*k = \frac{g_k + g*{(k+1)\bmod K}}{2}
\]

This adds _orientation information_ and sometimes removes the need for the tie-break probe.

But it’s a bit more assumption-y.

# TAM: Architecture and Implementation Plan

This plan is grounded in `formulation/model.md`, `formulation/properties.md`,
`formulation/architecture.md`, and the two existing implementations (`atlas/`,
`v4/`). It answers four questions:

1. how world signal becomes a situation
2. how a bind outcome decides between narrowing, widening, splitting, and
   proliferation
3. what mathematical foundations to build on
4. how to phase the implementation against the existing code

---

## 0. The one structural correction

The two codebases each hold half of the formulation:

- `v4` gets binding right: a `PortFiber` is a claim in **latent transition
  space**, frozen at bind time, and contradiction is measured against the
  frozen claim.
- `atlas` gets persistence right: charts are long-lived objects with retrieval
  keys, calibration buffers, lineage, and spawn-on-miss.

But `atlas/projector.py` measures contradiction as the distance of the
**situation latent** from the chart center. That is a *fit* score ("am I in
this chart's territory?"), not an affordance claim ("what will happen if I
bind?"). The formulation requires the cone to live in trajectory space
\(\mathcal{T}(x_n)\), not in situation space.

So the unifying move is:

> A chart owns two geometric objects: a **support region** in situation space
> (where the chart applies) and a **claim model** in transition space (what
> trajectories it affords from inside that support). The cone is the claim
> model conditioned on the current situation. Fit-residuals drive retrieval and
> spawning; claim-residuals drive cone width and splitting. They must never be
> conflated, because they answer to different failures.

Everything below follows from keeping these two residuals separate.

---

## 1. Signal → Situation (`Infer`)

### 1.1 What a situation *is*

The formulation defines a situation as \((n, x_n)\) plus prior context, but
does not say what makes a good \(x_n\). The principled answer: **a situation
latent is a sufficient statistic of history for the distribution over afforded
short-horizon trajectories.** Two histories that afford the same futures are
the same situation; two histories that afford different futures must be
encoded apart, no matter how similar the raw observations look.

This is the predictive-state / bisimulation criterion, and it makes `Infer`
non-arbitrary: the encoder is *defined* by what it must preserve (afforded
futures) and what it may discard (everything else). It is also exactly the
property retrieval needs — charts indexed by "what this situation affords"
rather than "what this situation looks like".

### 1.2 Pipeline

```
context atoms ──► sequence encoder ──► x_n (geometry latent)
 (episodes,          (GRU / small  ──► q_n (retrieval key)
  observations)       transformer)
```

- **Input**: the prior-context window as a sequence of typed atoms
  (`v4.core.ContextAtom` is the right shape; `atlas` currently takes a single
  flat vector and should adopt the sequence form).
- **Two heads, kept separate** (as in `atlas/encoder.py`):
  - `x_n` feeds chart-local geometry; trained for predictive sufficiency.
  - `q_n` feeds retrieval; trained with a metric loss so that distance in key
    space approximates a **bisimulation metric**: \(d(s, s') \approx\)
    divergence between transition distributions afforded at \(s\) and \(s'\).
- **Training signals** (all from the replay of binding records, no labels):
  1. next-transition prediction: \(x_n\) must linearly predict the realized
     transition delta within the bound chart's basis — this is the
     sufficiency pressure;
  2. contrastive separation: situations whose realized transitions were
     explained by different charts are pushed apart in key space; situations
     resolved by the same chart are pulled together (generalizes
     `atlas/training.py`, replacing regime labels — which are world-side
     ground truth we won't have — with chart assignments, which are
     self-generated);
  3. slowness/smoothness regularizer within episodes.

### 1.3 Encoder drift vs. chart stability

Charts store geometry in the encoder's coordinate system, so gradient updates
to the encoder silently invalidate the atlas. Handle this explicitly:

- the encoder is **versioned**; charts record the encoder version they were
  built under;
- charts keep a small buffer of raw support exemplars (already present as
  `support_examples`; store tensors, not strings);
- after an encoder update, re-encode exemplars and re-fit each chart's
  retrieval key, center, and basis. Charts whose exemplars no longer cluster
  are flagged for split/retire;
- encoder updates run on the slow timescale only (see §3.4), never inside the
  bind loop.

---

## 2. The bind outcome decision: narrow, widen, split, or proliferate

This is the heart of the system. The decision is driven by a **decomposition
of the realized residual**, not by a single scalar.

### 2.1 The claim, precisely

At bind time the chosen chart freezes a claim over the transition delta
\(d = x_{n+1} - x_n\) (later: over k-step trajectory segments, §5):

- anchor \(a\): expected delta,
- basis \(B\) (latent_dim × r): the directions the chart claims motion occurs
  along,
- coordinate region \(K\): bounds on the along-basis coordinates (the cone's
  extent),
- transverse radius \(\rho\): tolerated off-basis deviation.

The realized residual splits orthogonally:

\[
r = d - a,\qquad
r_\parallel = BB^{+} r \;\;(\text{on-model}),\qquad
r_\perp = r - r_\parallel \;\;(\text{off-model}).
\]

\(r_\parallel\) asks "was the *extent* right?"; \(r_\perp\) asks "was the
*kind of motion* right?". These have different remedies, which is precisely
why a single contradiction scalar (current `atlas` and `v4` both reduce to
one number) cannot drive structure learning.

### 2.2 Decision table

Each chart maintains conformal calibration buffers (extending
`ChartCalibration`) **separately for \(\|r_\parallel\|\), \(\|r_\perp\|\), and
the situation-fit residual**. Let \(p_\parallel, p_\perp\) be the conformal
p-values of the realized residuals, and let `alt_fit` be the best
counterfactual explanation: project the realized transition through the other
top-k retrieved charts and take the best p-value among them.

| \(p_\perp\) | \(p_\parallel\) | alt_fit | reading | action |
|---|---|---|---|---|
| high | high, margin consistently large | — | cone wider than reality requires | **narrow** \(K, \rho\) toward the empirical \(1{-}\alpha\) residual quantile |
| high | low | — | right kind of motion, wrong extent | **widen** \(K\) (and re-center \(a\)) |
| low | — | good | another chart explains it | **misretrieval**: train retrieval (hard negative), leave geometry alone |
| low | — | poor | no frame fits | **proliferate**: spawn child chart seeded from the realized transition |
| recurring bimodal residuals | — | — | one chart covering incompatible regimes | **split** along the separating direction |

Key properties of this rule:

- **Narrowing is earned, never event-driven.** A single in-cone success says
  nothing; narrowing triggers only when the rolling \(1{-}\alpha\) quantile of
  residuals sits well inside the current bounds. This is what makes cone width
  a calibrated uncertainty estimate rather than a mood.
- **Widening is asymmetric and fast.** A coverage violation with small
  \(r_\perp\) is evidence the cone is too tight *now*; widen immediately
  (multiplicatively), then let narrowing re-earn precision. Fast-widen /
  slow-narrow mirrors how conformal methods track distribution shift.
- **Proliferation requires double failure**: the bound chart failed *and* no
  retrieved sibling explains the outcome. This kills the dominant failure mode
  of spawn-on-miss systems — chart explosion from noise — because noise
  produces unexplained residuals that are *also* unexplained by the spawned
  child, whereas a genuinely new regime produces a child that immediately
  starts accumulating support.
- **Splitting is a statistic, not an event.** Track per-chart residual
  bimodality (2-component mixture vs. 1-component, BIC or a simple dip test
  over the calibration buffer). Sustained bimodality = the chart's territory
  contains two incompatible affordance regimes; split it, inherit lineage.

### 2.3 Blame assignment across the stack

A contradiction can be caused by world stochasticity, wrong cone size, wrong
chart, or bad encoding. The architecture distinguishes them by *where the
anomaly shows up*:

| symptom | culprit | timescale |
|---|---|---|
| stationary, unimodal residuals at the radius floor | aleatoric noise | none — this *is* the floor |
| coverage drift in one chart | cone size / center | fast (per-bind calibration) |
| \(r_\perp\) failures explained by siblings | retrieval | medium (metric learning on hard negatives) |
| correlated coverage failures across many charts | encoder | slow (gradient retraining + chart re-keying) |

The last row is the crucial one: per-chart **test martingales** over conformal
p-values (§3.3) give a running, anytime-valid alarm. One martingale exploding
= that chart is wrong. Many exploding together = the coordinate system itself
is wrong, and only then is encoder retraining justified.

---

## 3. Mathematical foundations

The geometric instinct is right, and it can be made literal. Four bodies of
theory each contribute one load-bearing piece.

### 3.1 Differential geometry: atlas, charts, and gluing

Treat the space of afforded short-horizon trajectories as a (stratified)
manifold. Charts are exactly what the name says: local coordinate maps. What
the current code is missing is the part of the definition that gives an atlas
its power — **transition maps**. When two charts' supports overlap, the
coordinate change between them is learnable from situations both claim. This
buys:

- a *consistency check*: incompatible transition maps on an overlap mean at
  least one chart is wrong (a contradiction signal that needs no world
  feedback at all);
- a *merge criterion*: two charts whose transition map is near-identity and
  whose claims agree are the same chart (`merge` currently has no principled
  trigger in either codebase);
- the formal substrate for the convergence claims in §6.

The port/fiber language in `v4` is already implicitly a fiber-bundle picture:
situation space is the base, transition space the fiber, a chart trivializes
the bundle locally, and a port is a *section with tolerance* — anchor plus
allowed deviation. This is worth keeping as the organizing picture; it cleanly
separates "where am I" (base) from "what do I claim" (fiber), which is the
same separation as §0.

### 3.2 Cones, literally: direction–magnitude parameterization

The word "cone" is currently implemented as an axis-aligned box. Make it a
cone. Parameterize a claimed transition as direction × magnitude:

\[
d = m \cdot u,\quad u \in S^{k-1},\ m \ge 0
\]

and a claim as (anchor direction \(u_0\), angular radius \(\theta\), magnitude
band \([m_-, m_+]\)) within the chart basis, plus transverse radius \(\rho\).
This buys three things:

- scale invariance: "same kind of motion, more of it" stays in-cone, matching
  the affordance intuition;
- a clean, dimension-comparable **agency measure** (§3.5): solid-angle
  fraction × magnitude fraction;
- the narrow/widen operations become one-parameter each (\(\theta\), band
  width) instead of per-axis radius surgery.

### 3.3 Conformal prediction, adaptive conformal inference, and test martingales

This is the cleanest available answer to "a model that predicts its own
uncertainty":

> **The affordance cone is a conformal prediction set at level \(1-\alpha\)
> over the chart's local transition distribution.**

- *Coverage guarantee*: under exchangeability within a chart, the realized
  trajectory lands in the cone with probability \(\ge 1-\alpha\),
  distribution-free. The model's uncertainty claim is *calibrated by
  construction*, not by hope. `ChartCalibration` already computes empirical
  quantiles and p-values — it is a conformal calibrator in embryo; §2.2
  extends it to decomposed residuals.
- *Target failure rate*: a perfectly calibrated agent fails at rate exactly
  \(\alpha\). Epistemic health is measurable as |empirical coverage −
  \((1-\alpha)\)| per chart.
- *Realtime adaptation*: Gibbs & Candès' **adaptive conformal inference**
  updates the effective \(\alpha\) online under distribution shift — this is
  the principled version of the fast-widen rule in §2.2.
- *Self-testing*: a **test martingale** over each chart's conformal p-values
  (e.g. a betting martingale) grows without bound iff the chart is
  miscalibrated. This is "epistemic failure as signal" in its sharpest form:
  an anytime-valid accumulating evidence process, per chart, that triggers
  structural operations when it crosses a threshold. This replaces ad-hoc
  EMA thresholds (`contradiction_ema`, `stability_prior`) with quantities that
  have guarantees.

### 3.4 Nonparametric mixtures: a prior over proliferation

Spawn-on-miss with a fixed threshold (current `atlas`) has no answer to "how
many charts should exist?". Model chart assignment as a **Dirichlet-process
mixture** in key space: the probability that a situation belongs to a new
chart is the CRP posterior, which depends on concentration, existing chart
masses, and fit likelihoods. Spawning becomes a posterior decision rather than
a threshold crossing, and the same machinery prices merges (two components
whose posterior predictive distributions are indistinguishable). The
`retrieval_strength` / `stability_prior` heuristics in `store.py` become
component masses and likelihoods.

### 3.5 Agency in bits

`properties.md` defines agency as \(1 - |\Phi|/|\mathcal{T}(x)|\). The
measure-theoretic version is sharper as a log:

\[
\text{agency}(p, x, \vec c) = -\log_2 \frac{\nu(\Phi_p(x,\vec c))}{\nu(\mathcal{T}(x))}
\quad\text{bits}
\]

with \(\nu\) the reference measure on the chart's fiber (uniform on the
direction sphere × magnitude range, under §3.2). Bits of commitment are:

- additive across independent claims,
- comparable across charts of different dimension,
- and they connect the two halves of the system: **calibration fixes the
  failure rate at \(\alpha\); learning then maximizes bits-per-bind at that
  fixed rate.** Agency growth is the system earning narrower cones without
  losing coverage. This single scalar — mean bits per bind at fixed coverage
  — is the headline metric for the whole project.

The relationship to free energy / active inference is worth stating honestly:
cone width plays the role of inverse precision and contradiction the role of
prediction error, but TAM differs in that commitment is *discrete and frozen*
(a bound port, not a running gradient) and failure is *frame-level* (the
chosen local view was invalid), not just a loss term. That difference is what
makes agency measurable here.

---

## 4. Self-predicted uncertainty and realtime learning

Three learning loops at three timescales, only the first of which is in the
bind path:

| loop | trigger | updates | cost |
|---|---|---|---|
| **fast** (every bind) | bind outcome | calibration buffers, cone bounds (narrow/widen), chart stats, martingale state | closed-form, microseconds |
| **structural** (event-driven) | martingale alarm, bimodality, CRP posterior, overlap inconsistency | spawn / split / merge / retire; retrieval hard-negatives | small fits over exemplar buffers |
| **slow** (background) | accumulated replay | encoder + retrieval-key gradients, then chart re-keying (§1.3) | minutes, off the bind path |

Aleatoric vs. epistemic uncertainty falls out of the structure:

- **aleatoric** = the residual floor of a converged, calibrated chart. The
  radius floor (`radius_floor`) is its estimate; narrowing converges to it
  and stops. Irreducible; correct response is acceptance.
- **epistemic** = miscoverage (martingale growth), retrieval ambiguity
  (`ambiguity_score`), or absence of any fitting chart (novelty). Reducible;
  correct response is the structural loop.

The system "predicts its own uncertainty in a given situation" concretely: at
bind time it emits the cone (a calibrated prediction set), its agency in bits
(how much it is claiming), and its martingale state (how much it currently
trusts the frame it chose).

---

## 5. Trajectories, not just transitions

Both codebases currently claim over one-step deltas. The formulation is about
trajectories. Extension path, in order:

1. **k-step tubes**: a claim becomes a sequence of fibers along a predicted
   path, or equivalently a tube in \(\mathcal{X}^k\); contradiction = first
   exit time + exit magnitude. Episodes already arrive as sequences
   (`ContextEpisode`); the interpreter (`v4/inference.py`) currently encodes
   only `episode.last()` — extend it to encode the path.
2. **trajectory embeddings**: encode whole segments into a trajectory latent
   and claim regions there; cheaper than tubes for long horizons, and the
   natural home for "literal space of afforded trajectories".
3. Cone-in-trajectory-space is then exactly §3.2 applied to the trajectory
   latent: afforded trajectory = direction in trajectory-embedding space ×
   extent.

---

## 6. Grounding, invariants, and convergence

The theory to test: *reality grounds models through contradiction pressure,
and invariant-rich worlds force convergence.*

Mechanically, in this architecture: a world with conservation laws or hard
constraints produces realized transitions confined to low-dimensional sets.
Charts that claim off-constraint motion get contradicted at rates their
martingales detect; charts aligned with the constraint manifold survive and
narrow. **The atlas converges toward the tangent structure of the world's
constraint manifold not because it is rewarded, but because everything else
is pruned.** Invariants are the fixed points of contradiction pressure.

This is testable:

- **Physics benchmark worlds** (extending `atlas/benchmark/worlds.py`):
  pendulum, bouncing ball, two-regime contact systems. Ground truth regime
  boundaries (contact events, energy shells) are known; measure whether chart
  supports align with physical regimes and chart bases align with constraint
  tangent spaces (subspace angle metrics).
- **Two-agent convergence**: train two atlases with different seeds,
  architectures, and observation encodings on the same world. Measure
  correspondence: mutual information between chart assignments, and
  cross-agent coverage — *does a trajectory realized under agent A's bind land
  in agent B's cone for the matched situation?* High cross-coverage on
  invariant-rich worlds and low on noise worlds is the "base reality" claim,
  made measurable.
- **Evolutionary compatibility** (hypothesis-level, last): two agents are
  compatible when their atlases glue — their charts admit consistent
  transition maps over shared situations (§3.1 machinery, applied across
  agents). Compatibility is then a graded, measurable relation: the fraction
  of shared situation space on which their transition maps compose
  consistently. Incompatible agents are those whose local frames cannot be
  reconciled even where they co-occur.

---

## 7. Implementation phases

Build on `atlas/` as the trunk; pull `v4`'s bind-time semantics into it.

**Phase 1 — transition claims in charts** *(corrects §0)*
Add a claim model (anchor, basis, coordinate region, transverse radius — i.e.
`v4.PortFiber` + bounds) to `ChartRecord`. `PortView` becomes a frozen
`ClaimedRegion` at bind time. The runtime step splits into `bind()` and
`observe_outcome()`; a `BindingOutcome` record (as in `v4`) is stored to a
replay buffer. Situation-fit residual and claim residual become separate
fields end-to-end.

**Phase 2 — decomposed residuals and the decision table** *(§2)*
Per-chart calibration buffers for \(r_\parallel\), \(r_\perp\), fit. Implement
narrow (quantile-earned), widen (fast, asymmetric), counterfactual sibling
projection, double-failure spawn, bimodality split. Replace
`stability_prior`-style heuristics with conformal p-values + one test
martingale per chart. Validate on existing benchmark: false-spawn rate and
chart churn should drop; add coverage-error and bits-per-bind metrics to
`atlas/benchmark/metrics.py`.

**Phase 3 — literal cones and agency** *(§3.2, §3.5)*
Direction–magnitude claim parameterization; agency-in-bits per bind; headline
metric = mean bits at fixed coverage. Report per chart and per run.

**Phase 4 — sequence encoder and learned retrieval** *(§1)*
Context-window sequence encoder; training on replay with prediction +
chart-contrastive + smoothness losses; encoder versioning and chart re-keying
from exemplar buffers. This phase is where `atlas/training.py`'s dependence on
ground-truth regime labels is removed.

**Phase 5 — trajectories** *(§5)*
k-step tube claims, path-aware episode interpreter, then trajectory
embeddings.

**Phase 6 — grounding experiments** *(§6)*
Physics worlds, constraint-alignment metrics, two-agent convergence and
cross-coverage. This is where the theoretical claims get their evidence.

**Phase 7 — agent integration**
Only after the substrate is validated: the turn-level loop from
`architecture.md`, with the LLM selecting among legal ports and TAM enforcing
commitment. Deliberately last — every prior phase is testable in synthetic
worlds without an agent in the loop.

### Risks

- **Encoder drift invalidating charts** — mitigated by versioning, exemplar
  re-keying, slow-timescale-only updates (§1.3). The hardest open problem;
  if it proves unstable, freeze the encoder per "era" and re-found the atlas
  on era boundaries.
- **Chart explosion** — double-failure spawn + CRP prior + merge-by-gluing;
  watch `chart_churn` and `same_regime_duplicate_chart_rate` (already in
  benchmark metrics).
- **Stochasticity masquerading as epistemic failure** — martingales
  distinguish persistent miscalibration from noise by construction; the
  radius floor absorbs the aleatoric component.
- **Cold start** — new charts open with wide cones (agency ≈ 0) and earn
  narrowing; \(\alpha\) can be annealed down as the atlas matures.

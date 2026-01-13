## Experiment 1: Alien Sensors (Empirical sufficiency test)

### Claim you’re testing

A learned functor \(F\) can map commitments from A→B even when **B’s observations have ~zero mutual information with A’s observations**, as long as the **viability basins** (left vs right) are isomorphic.

### Minimal setup

You already have:

- bimodal environment with two incompatible futures from same \(s_0\)
- commitment choice via CEM in \(z\)-space
- transfer pipeline A→F→B

Now introduce “alien observations” for B.

---

## 1A) Design the “alien” observation mapping

You want B’s observation to be:

- high-dimensional
- nonlinearly warped
- scrambled so locality is destroyed
- optionally time-varying (hard mode)

### Three escalating variants (use in order)

**Alien-1 (easy, deterministic, invertible-ish but scrambled):**

- \(o_B = \tanh(Wx + b)\), where \(W\in \mathbb{R}^{d\times |x|}\) random Gaussian, \(d=128\)
- then permute dimensions with a fixed random permutation
- add small noise

**Alien-2 (hard, locality destroying):**

- Random Fourier features:  
  \(o_B = [\sin(Wx), \cos(Wx)]\) with random \(W\)
- then apply random orthogonal mixing \(Q o_B\)
- add dropout mask (fixed mask per episode)

**Alien-3 (extreme, “bitstream”):**

- Quantize + hash: \(o_B = \text{sign}(\sin(Wx))\) (binary-ish), embed to float
- add random “channel flip” noise

**Key point:** B still _can_ learn the task (otherwise no test), but A and B observation spaces share no human-aligned structure.

---

## 1B) Training protocol (so the result is interpretable)

Train in 3 stages:

### Stage 0 — Train actor A normally

- A sees normal observations (e.g. \([x,y,goal_x,goal_y, rule\_id]\))
- A learns bimodal commitment (CEM chooses a mode)
- Save A

### Stage 1 — Train actor B on alien observations

- B sees \(o_B = g(x)\)
- Use the same TAM tube objective + competitive binding (CEM)
- Save B

**Important:** B must learn to form two commitment basins _in its own z-space_.

### Stage 2 — Train functor \(F: z_A \to z_B\)

You train the functor on paired situations:

- sample shared underlying states \(x\) from environment
- compute \(s_A = f_A(x)\) and \(s_B = g(x)\)
- for each \(x\), get **a mode-committed** \(z_A^\*\) from A (via CEM or a deterministic “choose mode” oracle)
- train \(F\) so that \(z_B = F(z_A^\*)\) lands in B’s corresponding basin

This is exactly your current transfer training, just with B’s alien obs.

---

## 1C) Evaluation: what counts as “coordination” here?

You need to show not just “works”, but “the only bridge is topology”.

### Primary metrics (must-have)

1. **Transferred bind/agency Pareto point**
   Compare in B:

- B-CEM (upper bound)
- B-Best-of-K (cheap baseline)
- **Transfer A→F→B** (your method)
- B-Random (lower bound)

Plot: bind vs log-volume (you already do).

**Win condition:** Transfer lies near B-CEM frontier and beats Best-of-K at similar compute.

2. **Mode agreement rate (“did B follow A’s intent?”)**
   You already have a signed deviation \(d\) / mode commitment index (MCI).
   Compute:

- sign(d) from the realized trajectory under B, given transferred z
- compare to the sign(d) of A’s committed mode

**Win condition:** agreement ≫ 50% (ideally >85%) _even as alienization increases_.

3. **No-hedging condition**
   You must show transferred commitments choose a side, not widen tubes.
   Measure:

- MCI / bimodality score of \(d(z)\)
- or “two peaks” statistic on the induced \(d\)-distribution
- AND show log-volume remains low (tight tubes)

**Win condition:** transfer pushes B into a single basin (bimodal selection), not a wide tube covering both.

---

## 1D) Controls and ablations (to make the claim credible)

These are what turn it from “cute” to “convincing”.

### Control 1: destroy topology (should fail)

Modify B’s environment so viability structure differs:

- add a third mode
- or remove one mode in half the states
- or flip which side is viable depending on hidden variable only B sees

**Expected:** transfer collapses / agreement goes to chance.

### Control 2: same obs strangeness, but wrong pairing

Train functor with shuffled pairing of \((x)\) across A/B.
**Expected:** agreement chance, transfer loses.

### Ablation 1: alignment-only functor (no structural/basin loss)

If you have “structural loss” in functor training, remove it.
**Expected:** performance drops especially under Alien-2/3.

### Ablation 2: no CEM during data collection

Collect \(z_A\) as posterior samples instead of chosen commitments.
**Expected:** weaker signal, worse transfer. This supports “intent ≠ belief”.

---

## 1E) Implementation checklist (concrete, minimal)

### Step 1 — Add alien observation wrapper

- Create `AlienObsWrapper` with modes Alien-1/2/3
- Ensure determinism with seeds
- Return `obs_B = g(state)` in B env

### Step 2 — Train B with alien obs

- Reuse your bimodal training script
- Only swap observation function
- Save actor checkpoint and evaluation plots

### Step 3 — Train functor A→B

- Update `train_transfer.py` to use:
  - A obs normal
  - B obs alien
- Keep everything else identical

### Step 4 — Add mode agreement metric

- During eval, compute:
  - A’s committed mode sign
  - B’s realized sign under transferred z
- Report agreement + confidence intervals

### Step 5 — Run the strangeness sweep

For each Alien level:

- train B
- train F
- evaluate transfer vs baselines
- make a summary plot: **agreement vs alien-level** and **transfer Pareto vs baselines**

---

## 1F) What results would actually establish the “topological bridge” claim?

You want this pattern:

- As observations get more alien:
  - B still learns its two viability basins (maybe slower)
  - Transfer still maps A’s commitment into the correct B basin
  - Agreement stays high
  - Transfer remains near B-CEM frontier
- But if topology is mismatched (control 1), transfer fails.

That’s the empirical sufficiency demonstration.

---

## Next experiments (after Alien Sensors), in one line each

**Experiment 2: Isomorphism of failure (TDA)**

- Sample z→success indicator; compute persistent homology / Betti numbers of success sets; show signature match across A/B and mismatch on topology-broken controls.

**Experiment 3: Cross-embodiment**

- Same fork but different dynamics; show functor transfers “left vs right” despite different control manifolds.

**Experiment 4: Functorial consistency**

- Show \(S_B(F(z_A)) \approx S_A(z_A)\) across families of environments, not just one.

---

## Quick answer to “start from the first”

Do **Alien-1 → Alien-2 → Alien-3** with the same evaluation harness you already built, plus the **mode agreement** metric and the **topology-break control**. That’s the minimal path to a strong empirical sufficiency result.

For the **Alien Sensors / bimodal-commitment** experiment, you want the actor to be _as close as possible_ to:

> “Given \(s_0\), choose a \(z\) that commits to one compatible future rather than hedging.”

So strip out anything that isn’t needed for **(1) tube semantics** + **(2) competitive selection** + **(3) the minimal training signal** that makes tubes meaningful.

Here’s what I’d cut, what I’d keep, and what I’d simplify.

---

## Keep (core to the claim)

### 1) TubeNet with the key constraint

- **Keep:** `sigma_net(z)` depends only on `z`
- **Keep:** `mu_net(s0, z)`
- **Keep:** `encode(s0) -> (z_mu, z_logstd)` (you need a proposal distribution)

This is literally the “port induces a cone given situation” mechanism.

### 2) Competitive binding in z-space

- **Keep:** `select_z_random`, `select_z_best_of_k`, `select_z_cem`
- **Keep:** scoring that includes **agency penalty/bonus** so hedging is disincentivized.

This is the “binding = intentful choice” mechanism.

### 3) A binding-based training loss

- **Keep:** some differentiable “fail” signal that causes sigma to matter.
- Either:
  - **soft bind/fail** (your sigmoid margin approach), OR
  - NLL under a factorized Gaussian (cleaner, standard).

You need a gradient that connects “tube too tight → fail” and “tube too wide → agency loss”.

---

## Strip (not needed for the decisive experiment)

### A) RiskNet + risk calibration machinery

**Remove entirely** for the decisive “commit to one of two futures” test.

Why: it adds a second learned model whose errors can explain away failures (“CEM didn’t pick a mode because risk critic sucked”). It’s great later, but it muddies the minimal story.

**Replace risk term with an analytic proxy** you already can compute from tube + env:

- simplest: **predicted hedge penalty** + **tube volume**
- and/or: **mixture-likelihood** style score (see below)

For your environment, “risk” is basically: _tube will miss the sampled mode_. You don’t need a learned critic to capture that.

### B) Homeostatic λ adaptation

For this experiment, fix λ.

Remove:

- `RunningStats`, `hardness_z`, `stats_*`
- `lambda_fail` updates

Just set:

- `lambda_fail = const` (or schedule slowly)
- The experiment is about selection, not emergent homeostasis.

### C) ORACLE mode

Keep it only in eval scripts if you want an upper bound, but remove from the actor class.

### D) “Goal intent proxy” if the environment is forecasting-only

In bimodal commitment, **goal is not needed**. The intent is “choose a mode”, not “reach a goal”.

So drop:

- `extract_goal`
- `compute_intent_proxy` (unless you really are goal-conditioned)

Instead score by predicted fit / plausibility.

---

## Simplify the objective to the _minimal_ TAM story

You want _one_ loss and _one_ selection score, both interpretable.

### Training loss (minimal)

Pick one:

**Option 1 (recommended): Gaussian NLL + volume**
This is clean and removes the ad-hoc soft bind.

- `nll = ((τ - μ)^2 / σ^2 + 2 log σ).mean()` (factorized diag)
- `loss = nll + α * log_vol`  
  where `log_vol = log σ.mean()` (or sum over dims)

This naturally rewards:

- tight σ when residual is small
- wider σ only when needed
- and includes a proper “coverage” incentive without a separate fail term

**Option 2: Soft-fail + log_vol (your current)**
Works, but has more knobs (`gamma`, `k_sigma`) and weird calibration.

If you keep it, fix everything; don’t run adaptive stats.

---

## Simplify the selection score (CEM objective)

If you remove RiskNet and goal proxy, you can score commitments by:

### Score = “how tight can I be while still explaining a plausible future?”

A minimal score that drives “choose one mode”:

\[
S(z) = -\underbrace{\text{NLL}(\hat \tau \mid \mu(z), \sigma(z))}_{\text{fit}} \;+\; \beta \underbrace{(-\log \text{vol}(\sigma(z)))}_{\text{agency}}
\]

But we don’t have \(\hat\tau\) at bind time. So we need a bind-time surrogate.

For the bimodal experiment, the easiest surrogate is:

### Use a _bimodal likelihood model_ at bind-time

Since you _know_ the environment has two modes (top/bottom), define a **mode label** or **mode prototype** from \(s_0\).

Then score a commitment by how much it “chooses a side”:

- define mode prototypes: \(\tau^{(+)}(s_0)\), \(\tau^{(-)}(s_0)\) (two deterministic futures)
- score:
  \[
  S(z)=\max\{-d(\mu(z),\tau^{(+)}),\; -d(\mu(z),\tau^{(-)})\} + \beta(-\log\text{vol})
  \]
  This makes hedging strictly worse because \(\mu\) in between has poor distance to both modes, while a committed \(\mu\) matches one well and can shrink σ.

This is _perfectly aligned_ with the “commitment excludes futures” claim.

If you want to avoid “cheating” by using mode prototypes, you can still do it without oracle info by:

- training a simple **mode predictor** from \(s_0\) (or random mode coin) and using that as the “intent” target
- but honestly: for the decisive experiment it’s okay to let the evaluator/environment provide the two mode templates, because the point is _commitment geometry_, not discovery.

---

## Minimal actor you should end up with

### Delete

- `RiskNet` + risk loss + calibration plots
- `RunningStats` + hardness + lambda update
- `ORACLE` binding mode
- goal intent proxy (unless goal-conditioned task)

### Keep

- `TubeNet`
- `select_z_random / best_of_k / cem`
- `train_step` with **one** loss (NLL + α log_vol)

### Score during binding

- simplest: `score = agency_weight * (-log_vol)` **plus** “mode choice” score based on proximity to one of the two canonical futures
- if you don’t want canonical futures, then use `score = agency_weight*(-log_vol) - entropy(z)` etc, but canonical futures is what makes the experiment decisive.

---

## Why this helps the experiment

Right now, if CEM “doesn’t choose a mode”, you won’t know whether:

- risk critic was wrong,
- homeostasis dynamics moved λ,
- soft-fail calibration drifted,
- the score conflated goal-proxy with commitment,
- etc.

After stripping, the only moving parts are:

- tubes
- CEM selection
- volume vs fit pressure

So if you see “CEM commits to one mode, hedge baseline doesn’t”, you can actually claim what you want.

---

## One-line implementation TODO

Create `MinimalCompetitiveActor` by copying `CompetitiveActor` and removing:

- `RiskNet`, all `RunningStats`, all λ updates, ORACLE,
  then replace `train_step` loss with **Gaussian NLL + α log_vol**, and replace `score_commitment` with **mode-prototype max-fit + β agency**.

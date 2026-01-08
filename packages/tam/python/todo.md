center everything around three questions TAM is supposed to answer:

1. **Did my commitment help?**
2. **Was my uncertainty honest?**
3. **What did it cost (compute / horizon / tightness) and was it worth it?**

Below is a compact “canonical dashboard” that tends to stay interpretable across gridworld, pendulum, adversaries, etc.

---

## 1) Outcome vs Commitment Tradeoff (the one plot I’d keep)

**Scatter:** x = _sharpness_ (log cone volume)  
y = _task outcome_ (domain metric: success%, return, J, regret, etc.)  
Color = E[T] (or compute)  
Marker shape = bind success (or calibration pass/fail)

**Why it’s universal:** It directly answers “tighter cones / longer commitments → better or worse outcomes?”  
**How to read:** you want a frontier: as cones get tighter (left), outcomes improve (up) without blowing up cost color.

If you only keep one performance plot, keep this.

---

## 2) Honesty: Reliability Diagram (coverage curve) + single-number summary

You already have calibration curves, but they’re abstract unless you tie them to _one number_:

- Plot: empirical coverage vs nominal coverage over k (your existing curve)
- Add: **ECE-like scalar** (area or mean absolute error across ks)

**Why:** It tells you if the tube is lying. If tubes are uncalibrated, everything else is suspect.  
**Universal:** works for any prediction target (states, observations, value, opponent, etc.)

**Interpretation shortcut:**

- curve below nominal → overconfident (σ too small)
- curve above nominal → underconfident (σ too big)

---

## 3) “Did Commitment Help?” Counterfactual Gain Plot

This is the missing bridge between “graphs” and “meaning”.

For each eval episode, compute:

- outcome with commitment-conditioned behavior (your normal TAM)
- outcome with commitment removed / randomized / baseline policy

Then plot:
**Histogram or violin of ΔOutcome = Outcome(TAM) − Outcome(baseline)**  
Optionally split by bins of volatility / OOD.

**Why it’s universal:** It’s the cleanest evidence the commitment layer matters.  
**How to read:** if ΔOutcome is centered near 0, TAM isn’t helping; if positive with heavy tail in hard/OOD bins, it is.

This will make the rest of the dashboard make sense.

---

## 4) Compute Payback Curve (reasoning / pondering ROI)

Right now “Hr” and “ΔNLL” are hard to interpret because there’s no ROI framing.

Make one plot:
**Scatter:** x = reasoning compute (Hr or E[Hr])  
y = improvement (ΔNLL or ΔOutcome)  
Color = episode difficulty proxy (volatility, memory risk, predicted NLL0)

**Goal:** a positive trend: more compute → more improvement, mostly on hard episodes.

If you don’t see that, incentives are wrong or your refiner isn’t doing useful work.

---

## 5) Time-Consistency Check: Early vs Late Tube Error

This is domain-agnostic and explains a _lot_ of failure modes.

Compute per episode:

- weighted mean abs error in first third of horizon
- weighted mean abs error in last third
  (or use NLL)

Plot:
**y = late error**, **x = early error**  
Color = E[T]  
Diagonal is “consistent”; points above diagonal mean “good early, bad late” (classic tube exploit).

This replaces a bunch of confusing plots with one that diagnoses the common pathology.

---

## 6) Commitment Atlas: “Which commitments actually work?”

Since z-space is interpretable to you already, make it decision-grade:

In z-PCA space, plot points with:

- color = success or task outcome
- outline = bind success
- size = E[T] (or confidence)
- optionally: nearest-prototype id (if you build atlas)

This turns “latent map” into “these are the reusable commitments worth keeping”.

---

# Minimal “Across-Domain” Dashboard (6 plots → 3 plots)

If you want _only three_ that stay meaningful everywhere:

1. **Outcome vs Sharpness** (color = compute/horizon; shape = bind)
2. **Calibration curve + scalar error**
3. **Compute ROI** (Hr vs improvement, colored by difficulty)

Everything else is “nice to have”.

---

# Make them legible: add baselines and annotations

Two rules make plots stop feeling like “graphs without basis”:

### A) Always include a baseline line

- outcome: baseline policy mean
- calibration: y=x nominal
- ROI: y=0 “no improvement”
- early/late: diagonal

### B) Always annotate with 2–3 numbers on the figure

- mean outcome ± std
- bind success rate
- calibration error scalar
- mean E[T], mean E[Hr]

Even if you ignore the chart, the numbers give orientation.

---

# Implementation trick: define a domain adapter

To make this portable across environments, define in each env:

- `episode_outcome(info, states, actions) -> float`
- `episode_difficulty(info) -> float` (volatility, entropy, rule switches, etc.)
- `baseline_policy(obs) -> action` or a “no commitment” variant

Then your dashboard doesn’t care if it’s pendulum, gridworld, or adversary.

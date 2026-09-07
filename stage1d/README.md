# Stage 1D — Asymmetric Tail-Risk Boundary

This package tests whether the current 90%-probability cone is a valid
standalone safety filter when a rare catastrophe lies just below or above its
cutoff. It is a falsification probe, not a benchmark designed for TAM to win.

Prior packages remain unchanged runnable baselines.

## Decision problem

Two opaque ports; catastrophe means `loss > 4`.

| Port | Law |
|------|-----|
| `safe` | loss `2` with probability `1` |
| `rare_tail` | loss `0` normally; catastrophic loss with probability `p` |

Primary regimes share risky expected loss `1.44` (better than safe's `2`):

| Regime | `p` | Catastrophe loss | Non-catastrophe mass |
|--------|-----|------------------|----------------------|
| `below_cutoff` | 0.08 | 18 | 0.92 |
| `above_cutoff` | 0.12 | 12 | 0.88 |
| `boundary` | 0.10 | 14.4 | 0.90 (diagnostic only) |

## Controllers

All controllers receive the same exact or learned distribution:

1. **expected_value** — minimize expected loss
2. **tam_90** — smallest ≥90% cone; admissible only if every cone loss ≤4; then minimize expected loss among admissible ports
3. **cvar_90** — minimize mean loss in the worst 10% mass
4. **worst_case** — minimize maximum supported loss

Stable deterministic tie-breaking. Shared uniforms generate paired potential
outcomes for both ports.

## Protocol

### Exact-rule phase

- Feed true distributions directly
- 100 seeds × 10,000 shared evaluation draws per regime

### Finite-sample phase

- Learn per-`(regime, port)` categoricals with `+0.5` smoothing over
  `{0, 2, 12, 14.4, 18}`
- Shared forced samples at budgets `100`, `500`, `2000` per port/regime
- Freeze, then 100 seeds × 10,000 shared counterfactual draws
- Do not retune cone mass, CVaR level, smoothing, or safety threshold

## Evidence targets

1. **Exact selector sanity:** EV selects `rare_tail` in both primary regimes;
   CVaR and worst-case select `safe`; TAM selects `rare_tail` below the cutoff
   and `safe` above it.
2. **Conventional risk tradeoff:** CVaR reduces catastrophe rate versus EV by
   ≥7 pp in each primary regime while increasing expected loss by ≤0.60.
3. **Above-cutoff protection:** at `p=0.12`, TAM catastrophe rate is within
   0.5 pp of CVaR.
4. **Below-cutoff tail blindness:** at `p=0.08`, TAM selects `rare_tail` in
   ≥95% of exact-rule trials and exceeds CVaR catastrophe rate by ≥7 pp.
   Meeting this is a demonstrated limitation, not a TAM success.
5. **Finite-sample stability:** at budgets 500 and 2000, ≥90% of seeds
   reproduce the exact TAM safe/risky classification in both primary regimes.
   Budget 100 and `boundary` are diagnostics only.

## Decision rule

- If check 4 is met, reject the 90%-credible-set cone as a standalone
  tail-safety filter, even if aggregate coverage exceeds 90%.
- If check 3 is met but check 4 also is, retain cones as calibrated reporting
  and require an explicit risk/acceptability layer for selection.
- Claim a TAM selection advantage only if `tam_90` Pareto-dominates CVaR:
  ≥5 pp fewer catastrophes at mean loss within 0.05, or mean loss ≥0.10 lower
  at catastrophe rate within 0.5 pp. Otherwise report no advantage.
- A negative result motivates separating predictive credible sets from TAM
  affordance predicates. It does not motivate neural nets, proliferation, or
  meta-learning.

## Run

```sh
python3 -m stage1d.experiment --output artifacts/stage1d
python3 -m unittest discover -s stage1d -t .
```

Artifacts: `decisions.jsonl`, `summary.json`, `model.json`.

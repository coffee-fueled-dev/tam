# An evidence-driven path beyond minimal TAM

This is a research guide, not an architectural roadmap. Complexity is earned by
an observed limitation under a predeclared experiment. Each stage preserves the
TAM cycle described in `formulation/model.md`: an actor in a situation selects
and binds a port, the external world returns context, the actor infers a
trajectory, evaluates it against the commitment made at binding time, and uses
experience to refine future commitments.

## What changes, and what does not

Core commitments that survive architectural changes:

- The world supplies observations but no transition table, regime label, or
  simulator gradient.
- Ports remain modes of interaction with an inference rule and an affordance
  cone. A binding is judged against the saved pre-action cone.
- A port's inference map remains fixed: learning changes what that port expects,
  not how an already-bound episode is interpreted. A materially different
  interpretation must be introduced and evaluated explicitly rather than
  silently changing the meaning of prior commitments.
- Binding reliability and goal achievement remain separate measurements.
- Learning uses experienced episodes, and refinement is evaluated by reliable
  specificity and useful behavior together.
- Claims about agency, convergence, or geometry remain hypotheses unless their
  assumptions and measurements apply to the implementation.

Stage 0 deliberately uses integer observations, one-step trajectories, fixed
ports, unconditional bounded counts, fixed subtraction as inference, and a fixed
selector and update rule. It cannot represent contextual effects, partial
observability, delayed consequences, continuous inputs, new ports, or learned
refinement strategies.

The refinement articles distinguish widening, narrowing, proliferation, and
no-op as possible updates, and study grounding as a fixed point of a monotone
operator on an affordance lattice. Stage 0's bounded-window update can shift a
cone in either direction and is not claimed to be such an operator. Therefore
its empirical stabilization is not evidence of lattice-theoretic grounding.

The minimal experiments can support or contradict only three hypotheses:
experience sharpens reliable expectations, those expectations support control,
and bounded persistent learning adapts after change. Neural generalization,
long-term retention, transfer, port proliferation, temporal belief, and
learning-to-learn remain untested regardless of the Stage 0 result.

The longer-term goal requires a stateful model that learns through its own
experience. Necessary capabilities may include contextual prediction,
generalization, transient belief, durable memory, calibrated uncertainty,
action specialization, and temporally extended commitments. They should not be
treated as one claim or introduced as one system.

Before each experiment, fix the environment, interaction budget, metrics, and
numerical decision thresholds. The stages below intentionally do not invent
thresholds for environments that have not yet been built.

## Stage 0 — Validate the minimal loop

1. **Demand:** Determine whether the basic commitment loop works before adding
   representation capacity.
2. **Hypothesis:** Experience yields narrower reliable cones, useful action, and
   recovery after an unannounced reversal.
3. **Smallest addition:** None beyond the bounded per-port learner in this
   package.
4. **Environment:** The declared stationary, noisy, and hidden-reversal 1D
   worlds.
5. **Baselines and ablations:** Frozen, amnesic, full-domain cones, and constant
   stay.
6. **Measures:** Pre-update calibration and Brier score, coverage, cone width,
   target success, first-hit interactions, and reversal blocks; also runtime and
   stored samples.
7. **Decision:** Keep Stage 0 if the predeclared criteria pass. If they fail,
   first separate implementation defects, inadequate exploration, unsuitable
   forgetting, and insufficient representation. Revise the responsible fixed
   mechanism; remove mechanisms that add no measured benefit.
8. **Not established:** Superiority to conventional control, generalization,
   durable retention, or the need for neural networks.

## Stage 1 — Context and higher-dimensional worlds

1. **Demand:** Unconditional port counts fail when outcomes depend on observable
   location or context.
2. **Hypothesis:** Conditioning outcome predictions on observed state improves
   calibration and transfers useful structure between situations.
3. **Smallest addition:** First add a small 2D observation and statistical
   contextual predictor. Add obstacles and context-dependent dynamics in
   separate experiments.
4. **Environment:** A 2D world with unknown action effects, followed by a world
   where the same port has different effects in observable contexts.
5. **Baselines and ablations:** Stage 0 counts, a context lookup table, and the
   contextual predictor with context features removed.
6. **Measures:** Calibration, coverage at matched specificity, target success,
   interactions to competence, unseen-context performance, memory, and runtime
   as state and action dimensions grow.
7. **Decision:** Keep conditioning only if gains survive matched experience and
   compute. Revise features if it memorizes cells without transfer; remove it if
   unconditional or tabular baselines match it.
8. **Not established:** Higher dimensions alone do not establish a need for
   deep learning or learned observation representations.

## Stage 2 — Neural representations and generalization

1. **Demand:** Continuous inputs, nonlinear contextual effects, or sparse
   observations expose measured limits in the strongest statistical predictor.
2. **Hypothesis:** A small neural outcome model generalizes calibrated
   commitments to unseen states or action parameters.
3. **Smallest addition:** Replace only the outcome predictor. Keep episode
   interpretation, commitment evaluation, selector, and observed-transition
   training fixed. Learn observation representations only in a separate
   experiment where raw inputs are demonstrably inadequate.
4. **Environment:** A continuous or combinatorial contextual world with held-out
   states and action parameters.
5. **Baselines and ablations:** Strongest Stage 1 predictor, parameter-matched
   shallow models, and neural models without learned representations.
6. **Measures:** Held-out Brier score and calibration, coverage-specificity
   curves, control success, sample efficiency, interference, wall time, memory,
   and training compute under equal interaction budgets.
7. **Decision:** Keep the neural model only for held-out benefit that survives
   compute accounting. Revise uncertainty estimation if narrow cones become
   unreliable; remove it if gains are only training fit.
8. **Not established:** A successful predictor does not establish temporal
   reasoning, durable memory, autonomous representation discovery, or
   learning-to-learn.

## Stage 3 — Temporal state and durable experience

1. **Demand:** Current observations become insufficient, consequences are
   delayed, or A → B → A experiments reveal costly forgetting.
2. **Hypothesis:** Explicit history or a compact belief state resolves partial
   observability, while durable memory reduces reacquisition without preventing
   adaptation.
3. **Smallest addition:** Add fixed action/outcome history features first, then
   a small recurrent belief state if those fail. Add replay or episodic
   retrieval only after measured forgetting justifies it. Keep transient belief
   separate from durable learned capability.
4. **Environment:** A partially observable world with aliased observations,
   delayed effects, checkpoints, and A → B → A regime sequences.
5. **Baselines and ablations:** Bounded Stage 0 history, frame stacks, recurrent
   state reset at boundaries, durable memory removed, and matched-capacity
   feed-forward predictors.
6. **Measures:** History-dependent calibration, delayed-task success,
   adaptation and reacquisition curves, A-state retention, checkpoint
   continuity, storage, sequence length, and compute.
7. **Decision:** Keep the least stateful mechanism that resolves aliasing or
   retention. Revise memory when stale evidence blocks adaptation; remove replay
   or recurrence if simpler history performs equally.
8. **Not established:** Temporal success does not imply causal inference,
   lifelong retention, or transfer to unrelated dynamics.

## Stage 4 — Port proliferation

1. **Demand:** One underlying action has distinct contextual consequences that a
   single cone represents poorly.
2. **Hypothesis:** Evidence-driven specialized ports improve specificity and
   control without sacrificing coverage or becoming unbounded memorization.
3. **Smallest addition:** A fixed rule creates a port that wraps an existing
   action with an observable applicability condition and its own commitment.
   Add explicit limits plus merge and prune rules; give no privileged world
   access.
4. **Environment:** Contexts in which a shared action has separable consequence
   modes and where specialization must generalize beyond one visited state.
5. **Baselines and ablations:** Contextual prediction with no explicit new
   ports, fixed hand-authored partitions, no merge/prune rule, and no
   specialization.
6. **Measures:** Specificity at matched coverage, control success, sample
   efficiency, transfer, port count and churn, memory, and decision cost.
7. **Decision:** Keep proliferation only if it beats contextual prediction at a
   bounded port count. Revise split/merge evidence when ports oscillate or
   duplicate; remove proliferation if it merely indexes observations.
8. **Not established:** Useful specialization does not establish autonomous
   concept formation, hierarchy, or a universal action vocabulary.

## Stage 5 — Learning-to-learn

1. **Demand:** Fixed refinement strategies show repeatable, complementary
   failures across dynamics or change schedules.
2. **Hypothesis:** A learner choosing update strength, forgetting rate, or
   specialization adapts better on held-out environment families.
3. **Smallest addition:** Learn a choice among concrete refinement operations;
   retain the same saved-commitment evaluation. Do not optimize immediate
   binding success alone, because widening every cone can game it.
4. **Environment:** Separate meta-training and evaluation families with varied
   dynamics and unannounced change schedules.
5. **Baselines and ablations:** Tuned fixed rules, simple adaptive heuristics,
   equal-compute search, and each refinement choice removed in turn.
6. **Measures:** Subsequent calibration and control, adaptation and retention,
   held-out performance, meta-training interactions, wall time, memory, and
   inference cost.
7. **Decision:** Keep meta-learning only when held-out benefit remains after all
   meta-training experience and compute are counted. Revise the objective if it
   rewards broad cones; remove it if tuned heuristics match it.
8. **Not established:** Held-out gains do not imply open-ended self-improvement,
   convergence, or safe autonomous objective selection.

## Stage 6 — Longer trajectories and richer ports

1. **Demand:** One-step control fails on delayed or temporally structured tasks.
2. **Hypothesis:** Multi-step commitments improve planning while retaining
   testable expectations throughout execution.
3. **Smallest addition:** Test fixed action sequences before learned options,
   macros, or continuous port optimization. Preserve the original commitment
   and define completion, interruption, and failure semantics.
4. **Environment:** Tasks where success requires a sequence and locally useful
   actions can cause delayed failure.
5. **Baselines and ablations:** One-step receding-horizon control, fixed
   sequences, sequence length ablations, and planning without uncertainty.
6. **Measures:** Planning benefit, trajectory coverage, compounding prediction
   error, interruption outcomes, search cost, compute, and sample efficiency.
7. **Decision:** Keep the shortest useful temporal abstraction. Revise
   commitments when their meaning drifts during execution; remove learned
   options if fixed sequences or replanning match them. Revisit tokenizers or
   graph structures only when recurring sequences or retrieval demands provide
   direct evidence.
8. **Not established:** Longer commitments do not establish hierarchical
   agency, compositionality, or correct long-horizon world models.

The progression can branch. Higher dimensions, neural prediction, and port
proliferation need not occur in this order; the observed limitation determines
the next experiment.

## Experiment record

Copy this record before implementing a research experiment:

```text
Question and observed limitation:
Hypothesis:
Mechanism added (one major mechanism):
Unchanged TAM commitments:
Environment and hidden information:
Interaction budget and randomization:
Baselines:
Ablations:
Predeclared metrics and numerical thresholds:
Compute and memory accounting:
Expected failure signatures:
Result:
Decision — keep, revise, or remove:
What the result rules out:
What the result does not establish:
Code, configuration, seeds, and artifact locations:
```

Maintain these rules across stages:

- Add one major mechanism at a time.
- Keep the simpler implementation as a runnable baseline.
- Separate gains from more experience, more compute, and architecture.
- Measure reliable specificity and useful behavior together.
- Treat adaptation, retention, generalization, and learning-to-learn as
  distinct claims.
- Preserve branches in the progression rather than forcing a fixed order.
- Preserve negative results and state what they rule out.
- Treat convergence and geometric claims as hypotheses requiring applicable
  assumptions, not implementation guarantees.

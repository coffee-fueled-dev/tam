 # TAM Architecture Notes

 This document summarizes the practical architecture that emerged from the
 original formulation.

 The short version:

 - TAM is not the whole agent
 - TAM is a structured substrate for situation inference, local frame selection,
   and epistemic failure
 - an LLM or other planner can sit on top of TAM, but TAM supplies the
   commitment structure that the planner must act through

 ## Core Idea

 TAM learns an atlas over latent trajectory space.

 - the world emits context
 - an inference pipeline maps that context into a latent situation
 - TAM retrieves locally valid charts from an atlas
 - the agent must bind one port, which commits it to a local view
 - the world responds
 - TAM measures contradiction between the committed claim and the realized
   outcome
 - contradiction drives refinement, specialization, and eventually chart
   proliferation

 ## Main Objects

 - `WorldContext`: the structured observations returned by the world on each turn
 - `Situation`: the current latent state plus prior context
 - `Infer`: the mapping from world context into a TAM-readable situation and a
   candidate set of legal ports
 - `Atlas`: the persistent chart library over latent trajectory space
 - `Chart`: a local projection or coordinate view into trajectory space
 - `Port`: the bindable handle an agent uses to commit to a chart-relative mode
   of interaction
 - `Cone`: the subset of trajectories accepted by that chart in the current
   situation
 - `Contradiction`: the measured mismatch between the committed cone and the
   realized trajectory

 ## Layered View

 ```mermaid
 flowchart TD
     World[World]
     Context[WorldContext]
     Infer[InferPipeline]
     Situation[SituationLatent]
     Atlas[AtlasStore]
     Candidates[LegalPorts]
     Agent[AgentOrLLM]
     Bind[PortBinding]
     Episode[ContextEpisode]
     Outcome[RealizedTrajectory]
     Update[ContradictionAndAtlasUpdate]

     World --> Context
     Context --> Infer
     Infer --> Situation
     Situation --> Atlas
     Atlas --> Candidates
     Candidates --> Agent
     Agent --> Bind
     Bind --> World
     World --> Episode
     Episode --> Outcome
     Situation --> Outcome
     Bind --> Update
     Outcome --> Update
     Update --> Atlas
 ```

 ## Why `Infer` Is Central

 The original formulation already required an interpretation step, but in the
 architecture this becomes explicit.

 `Infer` has to do three jobs:

 1. map raw world context into a latent situation
 2. produce a retrieval key or local neighborhood in situation space
 3. surface the ports that are currently legal or plausible

 Without `Infer`, the set of available ports is arbitrary and the agent has no
 grounded way to commit to a trajectory view.

 ## Turn-Level Loop

 ```mermaid
 flowchart LR
     Prev[PriorContext]
     Obs[Observation]
     InferTurn[Infer]
     Ports[CandidatePorts]
     Commit[CommitToPort]
     Act[ToolChoiceOrAction]
     Resp[WorldResponse]
     Eval[EvaluateContradiction]
     Learn[RefineOrProliferate]

     Prev --> InferTurn
     Obs --> InferTurn
     InferTurn --> Ports
     Ports --> Commit
     Commit --> Act
     Act --> Resp
     Resp --> Eval
     Commit --> Eval
     Eval --> Learn
 ```

 Each turn should force commitment:

 - TAM context is injected into the turn
 - the agent selects a port before acting
 - the chosen port influences the action or tool call
 - the realized outcome is evaluated against that committed port

 If port choice does not change what happens next, the architecture collapses
 into commentary rather than binding.

 ## Atlas Interpretation

 In the atlas reading:

 - trajectory space is the manifold of reachable short-horizon latent
   evolutions
 - charts are local projections into that manifold
 - cones are admissible regions within those local projections
 - ports are the bindable interface through which an agent commits to one chart

 This is closer to local geometric modeling than to a library of hand-authored
 skills.

The most useful chart-first reading is:

- the atlas is a family of lower-dimensional projections into trajectory space
- a situation conditions a local subset of that atlas
- the conditioned atlas surfaces the charts whose support overlaps the current
  situation
- ports are bindable fibers derived from those active charts
- choosing a port commits the agent to a chart-relative admissible region of
  trajectory space, even though the agent does not explicitly represent that
  full global region at commitment time

This means charts are more primary than ports. The atlas organizes local views
of trajectory structure, while ports are the operational handle through which an
agent binds one of those views and stands behind the region it exposes.

 ## Relationship To LLM Agents

 An LLM can sit above TAM, but TAM should remain the operational substrate.

 The practical split is:

 - TAM handles situation inference, chart retrieval, contradiction, and
   persistent local structure
 - the LLM handles broad planning, language, and action selection within the
   currently bound frame

 This makes TAM useful as a "system 2" interface:

 - it forces a local framing choice
 - it tracks when that framing fails
 - it can refine the atlas from epistemic failure over time

 ## Persistent Runtime

 The learned geometry is naturally an ML problem, but a persistent atlas also
 implies a systems layer.

 A practical implementation will usually combine:

 - a PyTorch-style learned core for encoders, retrieval keys, and local geometry
 - a persistent chart store for chart identity, retrieval vectors, lineage, and
   support statistics
 - an agent-facing API that injects TAM context into each turn

 ## Learning Signals

 TAM should not depend on the agent choosing to consult it.

 It should learn from every turn:

 - situation before action
 - chosen port
 - realized world response
 - contradiction
 - task progress

 This allows TAM to act as:

 - an always-on observer and learner
 - an always-present framing layer
 - a persistent source of structured feedback for the agent

 ## Structural Updates

 The atlas itself may evolve through explicit structural operations:

 - widen or narrow a cone
 - split a chart when one local model covers incompatible outcomes
 - proliferate a new chart when no existing chart fits
 - merge or retire charts when they become redundant

 These operations should be informed by latent statistics, but they are best
 treated as explicit atlas updates rather than hidden side effects inside one
 large neural network.

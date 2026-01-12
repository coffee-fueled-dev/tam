The **Trajectory-Affordance Model (TAM)** is a teleological framework that treats intelligence as the capacity to negotiate commitments with the future. Unlike traditional Reinforcement Learning, which maximizes a scalar reward, TAM defines an agent's "agency" by its ability to select and maintain specific, low-entropy "tubes" in trajectory space.

### 1. The Core Ontology

The model operates on a cycle of **situations** and **bindings**:

- **Situation ():** An indexed instance of internal state and prior context that marks a specific point in a causal chain.
- **Ports ():** Modes of interaction that define **Affordance Cones**—the set of all future trajectories an agent is willing to accept for that mode.
- **Binding:** The transition from one situation to the next when the world’s response falls within the agent's chosen cone.

### 2. Operationalizing Agency

In TAM, agency is not a vague philosophical concept but a measurable geometric property:

- **Agency as Compression:** Agency is inversely correlated with the width of the affordance cone. A "smart" agent exerts agency by committing to a narrow, specific future (minimal tube volume) while ensuring that the "bind" remains reliable.
- **The Zero-Agency Baseline:** An agent that accepts any possible future has an agency of zero.

### 3. The Deliberation Engine (Competitive Binding)

The implementation uses a **Cross-Entropy Method (CEM)** to simulate "thinking" before acting. The agent evaluates candidate commitments () based on a tri-part scoring function:

1. **Intent Proxy:** How accurately the commitment predicts the desired outcome.
2. **Agency:** The tightness of the predicted trajectory tube (rewarding precision).
3. **Risk Pred (Reliability):** A learned critic's estimate of whether the world will actually stay within that tube.

### 4. Key Empirical Insights

- **Rejection of Hedging:** In bimodal environments where two different futures are possible, a standard model might "hedge" by predicting a vague middle ground. TAM forces a **mode commitment**, where the agent chooses one specific path (e.g., "Left" or "Right") and rejects the high-entropy middle.
- **Homeostasis of Control:** The agent constantly seeks an equilibrium where it is as precise as possible without being so rigid that it constantly "breaks the bind" with reality.

### Summary for Researchers

TAM can be described as a **geometric theory of intent**. It shifts the focus of AI from "what action is best" to "what future am I willing to accept?" This makes it particularly relevant to **AI Safety** (ensuring predictable, steerable behavior) and **Mechanistic Interpretability** (mapping internal latent modes to explicit future commitments).

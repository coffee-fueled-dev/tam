/**
 * Fixed and learned policies for port selection and refinement.
 * These implement the strategy interfaces.
 */

import type {
  PortSelectionPolicy,
  RefinementPolicy,
  PortSelectionContext,
  RefinementContext,
  Vec,
} from "./types";
import  { Actor } from "./actor";

/**
 * Fixed curiosity-driven port selection policy.
 *
 * With probability explorationRate:
 * - Favor ports with mid-range agency (testing boundaries)
 * - Or ports with few samples (epistemic uncertainty)
 *
 * Otherwise exploit (select highest agency * binding rate).
 * This prevents selecting overly narrow ports that fail frequently.
 */
export class FixedPortSelectionPolicy implements PortSelectionPolicy {
  constructor(private explorationRate: number = 0.1) {}

  selectPort(context: PortSelectionContext): number {
    const { concentrations, portSamples, bindingRates } = context;

    if (concentrations.length === 0) return -1;

    const explore = Math.random() < this.explorationRate;

    // Convert concentrations to agencies
    const agencies = concentrations.map(c => c / (1 + c));

    if (explore) {
      // Curiosity-driven: favor mid-range agency or low samples
      const curiosityScores = agencies.map((agency, idx) => {
        const agencyCuriosity = 1 - Math.abs(agency - 0.5) * 2; // Peaks at 0.5
        const sampleBonus = 1 / (portSamples[idx]! + 1); // Bonus for unexplored
        return { idx, score: agencyCuriosity + sampleBonus };
      });

      const best = curiosityScores.reduce((a, b) => (b.score > a.score ? b : a));
      return best.idx;
    } else {
      // Exploit: select highest (agency * bindingRate)
      // This balances tight cones with actual binding success
      // Prevents selecting overly narrow ports that fail frequently
      let bestIdx = 0;
      let bestScore = agencies[0]! * (bindingRates?.[0] ?? 1.0);
      for (let i = 1; i < agencies.length; i++) {
        const score = agencies[i]! * (bindingRates?.[i] ?? 1.0);
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }
      return bestIdx;
    }
  }
}

/**
 * Fixed refinement policy (DEPRECATED in unified geometric architecture).
 *
 * NOTE: In the new dual-head CausalNet architecture, concentration is derived
 * geometrically from prediction uncertainty. This policy is no longer used
 * for learning, but remains for backward compatibility with the policy interface.
 *
 * Previously used BINARY binding predicate to couple learning:
 * - SUCCESS (inside cone): allow narrowing based on margin
 * - FAILURE (outside cone): force widening
 */
export class FixedRefinementPolicy implements RefinementPolicy {
  private narrowThreshold: number = 0.5;

  decideRefinement(context: RefinementContext): { concentrate: number; disperse: number } {
    const { bindingStrength, normalizedDistance } = context;

    // Binary binding check (restores v1 coupling)
    const bound = normalizedDistance < 1.0;

    if (bound) {
      // BINDING SUCCESS - trajectory is inside cone
      const margin = 1 - normalizedDistance;

      if (margin > this.narrowThreshold) {
        // Well inside - safe to narrow (increase agency)
        // Use soft strength as pressure weight for graduated learning
        const pressure = (bindingStrength - 0.5) * 2; // Maps [0.5, 1.0] -> [0, 1]
        return { concentrate: Math.max(0, pressure), disperse: 0 };
      } else {
        // Near boundary - don't over-tighten
        return { concentrate: 0, disperse: 0 };
      }
    } else {
      // BINDING FAILURE - trajectory is outside cone
      // Must widen to accommodate the trajectory
      // Use soft strength as pressure weight
      const pressure = (0.5 - bindingStrength) * 2; // Maps [0, 0.5] -> [1, 0]
      return { concentrate: 0, disperse: Math.max(0, pressure) };
    }
  }
}

/**
 * Proportional refinement policy.
 *
 * Adjusts concentration proportionally to the geometric violation magnitude:
 * - Outside cone: widen proportionally to (normalizedDistance - 1.0)
 * - Inside cone: narrow proportionally to (1.0 - normalizedDistance), conservatively
 *
 * This acts as a proportional controller, using normalizedDistance as error signal.
 * More principled than fixed step sizes - responds to HOW FAR the trajectory is from ideal.
 */
export class ProportionalRefinementPolicy implements RefinementPolicy {
  constructor(
    private narrowGain: number = 0.3,  // Conservative narrowing (avoid over-tightening)
    private widenGain: number = 1.0,   // Aggressive widening (must accommodate trajectory)
    private minMargin: number = 0.2    // Don't narrow if margin is small (safety buffer)
  ) {}

  decideRefinement(context: RefinementContext): { concentrate: number; disperse: number } {
    const { normalizedDistance } = context;

    // Binary binding check
    const bound = normalizedDistance < 1.0;

    if (bound) {
      // INSIDE CONE: Can narrow if margin is sufficient
      const margin = 1.0 - normalizedDistance;

      if (margin > this.minMargin) {
        // Narrow proportionally to margin (but conservatively)
        // More margin = more room to tighten
        const pressure = margin * this.narrowGain;
        return { concentrate: Math.min(1.0, pressure), disperse: 0 };
      } else {
        // Too close to boundary - don't risk over-tightening
        return { concentrate: 0, disperse: 0 };
      }
    } else {
      // OUTSIDE CONE: Must widen proportionally to violation
      const violation = normalizedDistance - 1.0;
      const pressure = violation * this.widenGain;
      return { concentrate: 0, disperse: Math.min(1.0, pressure) };
    }
  }
}

/**
 * Learned refinement policy using TAM.
 *
 * Uses a TAM Actor internally to learn optimal refinement decisions based on:
 * - State and port geometry
 * - Binding outcomes
 * - Overlap with nearby ports
 *
 * Multi-objective reward:
 * - Maximize agency (tight cones)
 * - Maintain binding success
 * - Minimize overlap with other ports
 * - Improve prediction accuracy
 */
export class LearnedRefinementPolicy implements RefinementPolicy {
  private policyActor: Actor<RefinementPolicyInput>;
  private rewardWeights: RewardWeights;
  private pendingObservation: {
    input: RefinementPolicyInput;
    decision: { concentrate: number; disperse: number };
  } | null = null;

  constructor(
    embeddingDim: number,
    rewardWeights?: Partial<RewardWeights>
  ) {
    this.rewardWeights = {
      agencyWeight: rewardWeights?.agencyWeight ?? 1.0,
      bindingWeight: rewardWeights?.bindingWeight ?? 2.0,
      overlapPenalty: rewardWeights?.overlapPenalty ?? 1.0,
      errorPenalty: rewardWeights?.errorPenalty ?? 0.5,
    };

    // Create policy actor with small networks (this is meta-learning)
    this.policyActor = new Actor<RefinementPolicyInput>(
      embedRefinementPolicyInput,
      {
        causal: { hiddenSizes: [16, 16], learningRate: 0.001 },
        proliferation: { enabled: false }, // Single port for now
      }
    );
  }

  decideRefinement(context: RefinementContext): { concentrate: number; disperse: number } {
    // If we have a pending observation, complete it with outcome
    if (this.pendingObservation) {
      this.observeOutcome(context);
    }

    // Build input for policy
    const input: RefinementPolicyInput = {
      stateEmb: context.stateEmb,
      portEmb: context.portEmb,
      concentration: context.concentration,
      normalizedDistance: context.normalizedDistance,
      bindingStrength: context.bindingStrength,
      predictionError: context.predictionError,
      nearbyPortDistances: context.nearbyPortDistances ?? [],
    };

    // Predict refinement decision
    const prediction = this.policyActor.predict(input);

    // Interpret prediction delta as (concentrate, disperse)
    const concentrate = Math.max(0, Math.min(1, prediction.delta[0] ?? 0));
    const disperse = Math.max(0, Math.min(1, prediction.delta[1] ?? 0));

    const decision = { concentrate, disperse };

    // Store for next outcome observation
    this.pendingObservation = { input, decision };

    return decision;
  }

  /**
   * Observe the outcome of the previous refinement decision.
   * Called at the next refinement step with updated context.
   */
  private observeOutcome(nextContext: RefinementContext) {
    if (!this.pendingObservation) return;

    const { input, decision } = this.pendingObservation;

    // Compute multi-objective reward
    const reward = this.computeReward(input, decision, nextContext);

    // Create "after" state: the decision led to this outcome
    const afterInput: RefinementPolicyInput = {
      ...input,
      concentration: nextContext.concentration,
      normalizedDistance: nextContext.normalizedDistance,
      bindingStrength: nextContext.bindingStrength,
      predictionError: nextContext.predictionError,
    };

    // Observe the transition: decision → outcome
    // The "delta" is the change in (concentrate, disperse) we would predict
    // But actually we want to reinforce decisions that led to good outcomes
    // So we'll use reward as implicit supervision
    this.policyActor.observe({
      before: { state: input },
      after: { state: afterInput },
    });

    this.pendingObservation = null;
  }

  /**
   * Compute multi-objective reward for a refinement decision.
   */
  private computeReward(
    input: RefinementPolicyInput,
    decision: { concentrate: number; disperse: number },
    outcome: RefinementContext
  ): number {
    // 1. Agency reward: higher concentration = higher reward
    const agency = outcome.concentration / (1 + outcome.concentration);
    const agencyReward = agency * this.rewardWeights.agencyWeight;

    // 2. Binding reward: successful binding = positive, failure = negative
    const bindingSuccess = outcome.normalizedDistance < 1.0 ? 1.0 : 0.0;
    const bindingReward = bindingSuccess * this.rewardWeights.bindingWeight;

    // 3. Overlap penalty: penalize ports that are close in embedding space
    // Small distance in embedding space = ports respond to similar states = overlap
    let overlapPenalty = 0;
    const nearbyDistances = outcome.nearbyPortDistances ?? [];
    for (const distance of nearbyDistances) {
      // Penalize inversely to distance: smaller distance = higher penalty
      // Use exponential decay so only very close ports are penalized heavily
      const overlap = Math.exp(-distance * 2.0); // Scale factor of 2.0 for sensitivity
      overlapPenalty += overlap * this.rewardWeights.overlapPenalty;
    }

    // 4. Error penalty: penalize high prediction errors
    const errorPenalty = outcome.predictionError * this.rewardWeights.errorPenalty;

    return agencyReward + bindingReward - overlapPenalty - errorPenalty;
  }

  async flush() {
    await this.policyActor.flush();
  }

  dispose() {
    this.policyActor.dispose();
  }
}

/**
 * Input to the learned refinement policy
 */
interface RefinementPolicyInput {
  stateEmb: Vec;
  portEmb: Vec;
  concentration: number;
  normalizedDistance: number;
  bindingStrength: number;
  predictionError: number;
  nearbyPortDistances: number[];  // Distances to k-nearest ports in embedding space
}

/**
 * Reward weights for multi-objective learning
 */
interface RewardWeights {
  agencyWeight: number;      // Reward for high agency (tight cones)
  bindingWeight: number;     // Reward for binding success
  overlapPenalty: number;    // Penalty for overlapping with nearby ports
  errorPenalty: number;      // Penalty for prediction errors
}

/**
 * Embed refinement policy input into fixed-dimensional vector
 */
function embedRefinementPolicyInput(input: RefinementPolicyInput): Vec {
  // Fixed-size embedding with padding for nearby port distances
  const maxNearbyPorts = 8;
  const nearbyPadded = input.nearbyPortDistances
    .slice(0, maxNearbyPorts)
    .concat(Array(Math.max(0, maxNearbyPorts - input.nearbyPortDistances.length)).fill(0));

  return [
    ...input.stateEmb,
    ...input.portEmb,
    input.concentration,
    input.normalizedDistance,
    input.bindingStrength,
    input.predictionError,
    ...nearbyPadded,
  ];
}

// ============================================================================
// TAM-Based Learned Policies
// ============================================================================

/**
 * Unified query type for TAM-based meta-policies.
 * Allows a single Actor to handle domain, port selection, and refinement queries.
 */
export interface UnifiedQuery {
  queryType: 'domain' | 'portSelection' | 'refinement';
  domainState: Vec;
  metaContext?: {
    portEmbs?: Vec[];
    concentrations?: number[];
    portSamples?: number[];
    bindingStrength?: number;
    normalizedDistance?: number;
  };
}

/**
 * Embed unified query into fixed-dimensional vector.
 * Uses query type encoding + domain state + flattened meta context.
 */
export function embedUnifiedQuery(query: UnifiedQuery): Vec {
  // One-hot encode query type
  const typeEncoding =
    query.queryType === 'domain'
      ? [1, 0, 0]
      : query.queryType === 'portSelection'
      ? [0, 1, 0]
      : [0, 0, 1];

  // Flatten meta context (pad to consistent size for each field)
  const maxPorts = 32; // Max number of ports to embed
  const portDim = query.domainState.length; // Assume ports have same dim as domain state

  const portEmbs = query.metaContext?.portEmbs ?? [];
  const flatPortEmbs = portEmbs
    .slice(0, maxPorts)
    .flatMap((p) => p)
    .concat(Array((maxPorts - portEmbs.length) * portDim).fill(0));

  const concentrations = (query.metaContext?.concentrations ?? [])
    .slice(0, maxPorts)
    .concat(Array(maxPorts - (query.metaContext?.concentrations?.length ?? 0)).fill(0));

  const portSamples = (query.metaContext?.portSamples ?? [])
    .slice(0, maxPorts)
    .concat(Array(maxPorts - (query.metaContext?.portSamples?.length ?? 0)).fill(0));

  const bindingStrength = query.metaContext?.bindingStrength ?? 0;
  const normalizedDistance = query.metaContext?.normalizedDistance ?? 0;

  return [
    ...typeEncoding,
    ...query.domainState,
    ...flatPortEmbs,
    ...concentrations,
    ...portSamples,
    bindingStrength,
    normalizedDistance,
  ];
}

/**
 * TAM-based port selection policy.
 * Uses a learned Actor to predict which port to select.
 */
export class TAMPortSelectionPolicy implements PortSelectionPolicy {
  constructor(private actor: Actor<UnifiedQuery>) {}

  selectPort(context: PortSelectionContext): number {
    const query: UnifiedQuery = {
      queryType: 'portSelection',
      domainState: context.stateEmb,
      metaContext: {
        portEmbs: context.portEmbs,
        concentrations: context.concentrations,
        portSamples: context.portSamples,
      },
    };

    // Predict using TAM
    const prediction = this.actor.predict(query);

    // Interpret prediction: find port index with highest predicted value
    // Prediction delta is expected to be a vector where each component
    // corresponds to a port's score
    const portScores = prediction.delta.slice(0, context.portEmbs.length);
    let bestIdx = 0;
    let bestScore = portScores[0] ?? -Infinity;

    for (let i = 1; i < portScores.length; i++) {
      if (portScores[i]! > bestScore) {
        bestScore = portScores[i]!;
        bestIdx = i;
      }
    }

    return bestIdx;
  }
}

/**
 * TAM-based refinement policy.
 * Uses a learned Actor to predict refinement pressures.
 */
export class TAMRefinementPolicy implements RefinementPolicy {
  constructor(private actor: Actor<UnifiedQuery>) {}

  decideRefinement(context: RefinementContext): { concentrate: number; disperse: number } {
    const query: UnifiedQuery = {
      queryType: 'refinement',
      domainState: context.stateEmb,
      metaContext: {
        concentrations: [context.concentration],
        bindingStrength: context.bindingStrength,
        normalizedDistance: context.normalizedDistance,
      },
    };

    // Predict using TAM
    const prediction = this.actor.predict(query);

    // Interpret prediction: first component = concentrate, second = disperse
    const concentrate = Math.max(0, Math.min(1, prediction.delta[0] ?? 0));
    const disperse = Math.max(0, Math.min(1, prediction.delta[1] ?? 0));

    return { concentrate, disperse };
  }
}

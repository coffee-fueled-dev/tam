/**
 * Sokoban agent using TAM for commitment-based learning.
 *
 * The agent learns:
 * 1. Affordance cones: what outcomes to expect per action
 * 2. Value function: how valuable situations are
 *
 * And selects actions by maximizing value over its learned cones.
 * No search algorithm - the agent learns through experience.
 */

import {
  TAMAgent,
  type Encoders,
  type Situation,
  type Transition,
  type Vec,
  TabularValue,
} from "tam";
import { type GameState, type Direction, DIRECTIONS, posKey } from "./types";
import {
  getMoveContext,
  isWon,
  move as executeMove,
  cloneState,
} from "./engine";

// ============================================================================
// Sokoban-specific situation types
// ============================================================================

interface SokobanState {
  player: { x: number; y: number };
  boxPositions: string[];
  goalPositions: string[];
  boxesOnGoals: number;
}

interface SokobanContext {
  wallAhead: boolean;
  boxAhead: boolean;
  wallBehindBox: boolean;
  boxBehindBox: boolean;
}

export type SokobanSituation = Situation<SokobanState, SokobanContext>;

// ============================================================================
// Encoders
// ============================================================================

const sokobanEncoders: Encoders<SokobanState, SokobanContext> = {
  /**
   * Embed situation as a feature vector.
   * Includes both context (local) and state (global progress) features.
   */
  embedSituation(sit: SokobanSituation): Vec {
    const { player, boxesOnGoals, goalPositions } = sit.state;
    const ctx = sit.context;

    // Progress ratio (how close to winning)
    const progress =
      goalPositions.length > 0 ? boxesOnGoals / goalPositions.length : 0;

    return [
      // Normalized player position
      player.x / 10,
      player.y / 10,
      // Context features
      ctx.wallAhead ? 1 : 0,
      ctx.boxAhead ? 1 : 0,
      ctx.wallBehindBox ? 1 : 0,
      ctx.boxBehindBox ? 1 : 0,
      // Progress toward goal
      progress,
    ];
  },

  /**
   * Compute delta between situations (player displacement).
   */
  delta(before: SokobanSituation, after: SokobanSituation): Vec {
    return [
      after.state.player.x - before.state.player.x,
      after.state.player.y - before.state.player.y,
    ];
  },
};

// ============================================================================
// Helpers
// ============================================================================

/**
 * Convert GameState to SokobanSituation for a given action context.
 */
export function gameStateToSituation(
  state: GameState,
  action?: Direction
): SokobanSituation {
  const ctx = action
    ? getMoveContext(state, action)
    : {
        wallAhead: false,
        boxAhead: false,
        wallBehindBox: false,
        boxBehindBox: false,
      };

  const goalPositions = Array.from(state.goals);
  const boxPositions = Array.from(state.boxes);
  const boxesOnGoals = boxPositions.filter((b) => state.goals.has(b)).length;

  return {
    state: {
      player: { ...state.player },
      boxPositions,
      goalPositions,
      boxesOnGoals,
    },
    context: ctx,
  };
}

/**
 * Compute potential: negative sum of distances from boxes to nearest goals.
 * Higher potential = closer to solution.
 */
function computePotential(state: GameState): number {
  let totalDist = 0;
  for (const boxKey of state.boxes) {
    const [bx, by] = boxKey.split(",").map(Number);
    let minDist = Infinity;
    for (const goalKey of state.goals) {
      const [gx, gy] = goalKey.split(",").map(Number);
      const dist = Math.abs(bx! - gx!) + Math.abs(by! - gy!);
      minDist = Math.min(minDist, dist);
    }
    totalDist += minDist;
  }
  // Negative because lower distance = higher potential
  return -totalDist;
}

/**
 * Compute reward for a transition using potential-based shaping.
 *
 * R = r + γΦ(s') - Φ(s)
 *
 * This provides a dense reward signal that guides toward the goal.
 */
function computeReward(
  beforeState: GameState,
  afterState: GameState,
  delta: Vec
): number {
  // Win bonus
  if (isWon(afterState)) {
    return 10;
  }

  // Potential-based shaping
  const potentialBefore = computePotential(beforeState);
  const potentialAfter = computePotential(afterState);
  const shapingReward = 0.95 * potentialAfter - potentialBefore;

  // Boxes on goals
  const beforeBoxesOnGoals = Array.from(beforeState.boxes).filter((b) =>
    beforeState.goals.has(b)
  ).length;
  const afterBoxesOnGoals = Array.from(afterState.boxes).filter((b) =>
    afterState.goals.has(b)
  ).length;
  const boxProgress = afterBoxesOnGoals - beforeBoxesOnGoals;

  // Combined reward
  let reward = 0;

  // Big bonus for getting box on goal
  if (boxProgress > 0) {
    reward += 2 * boxProgress;
  } else if (boxProgress < 0) {
    reward -= 1 * Math.abs(boxProgress);
  }

  // Shaping reward (scaled)
  reward += 0.1 * shapingReward;

  // Small penalty for each step (efficiency)
  const moved = delta[0] !== 0 || delta[1] !== 0;
  reward += moved ? -0.01 : -0.1; // Bigger penalty for not moving

  return reward;
}

/**
 * How delta affects the embedding.
 */
function deltaToNextPhi(phi: Vec, delta: Vec): Vec {
  const newPhi = [...phi];
  // Update player position features
  newPhi[0] = (newPhi[0] ?? 0) + (delta[0] ?? 0) / 10;
  newPhi[1] = (newPhi[1] ?? 0) + (delta[1] ?? 0) / 10;
  return newPhi;
}

// ============================================================================
// Sokoban Agent
// ============================================================================

/**
 * Create a TAM agent configured for Sokoban.
 */
export function createSokobanAgent(): TAMAgent<SokobanState, SokobanContext> {
  return new TAMAgent({
    encoders: sokobanEncoders,
    config: {
      actions: ["up", "down", "left", "right"],
      gamma: 0.95,
      valueLr: 0.2,
      aggregator: "mean",
      // Agency bonus: prefer actions we understand
      agencyBonus: 0.05,
      // Curiosity: intrinsic reward for learning (agency gain)
      curiosityCoef: 0.3,
      // Exploration: UCB-style bonus for unvisited state-actions
      explorationCoef: 0.5,
      portConfig: {
        maxComponents: 6,
        reservoirSize: 20,
      },
    },
    deltaToNextPhi,
    rewardFn: (sit, delta) => {
      // Estimate reward from delta (used for planning before execution)
      // Movement is slightly positive, staying still is negative
      const moved = delta[0] !== 0 || delta[1] !== 0;
      // Bonus for progress (crude estimate from embedding)
      const progressBonus = sit.state.boxesOnGoals * 0.5;
      return (moved ? 0.01 : -0.05) + progressBonus * 0.1;
    },
    // Use tabular value for discrete Sokoban states
    valueFunction: new TabularValue(0.2, 0),
  });
}

// ============================================================================
// Training Loop
// ============================================================================

export interface TrainingStats {
  episodes: number;
  wins: number;
  totalSteps: number;
  avgStepsPerEpisode: number;
  avgReward: number;
}

/**
 * Train the agent through episodes of play.
 */
export function trainAgent(
  agent: TAMAgent<SokobanState, SokobanContext>,
  initialState: GameState,
  episodes: number = 100,
  maxSteps: number = 100,
  epsilon: number = 0.3
): TrainingStats {
  let wins = 0;
  let totalSteps = 0;
  let totalReward = 0;

  for (let ep = 0; ep < episodes; ep++) {
    let state = cloneState(initialState);
    let steps = 0;

    // Keep high exploration for first half, then decay
    const progress = ep / episodes;
    const currentEpsilon =
      progress < 0.5
        ? epsilon // Full exploration first half
        : epsilon * (1 - (progress - 0.5) * 2); // Decay in second half

    // Track trajectory for potential replay
    const trajectory: Array<{
      state: GameState;
      action: Direction;
      reward: number;
      transition: Transition<SokobanState, SokobanContext>;
    }> = [];

    while (!isWon(state) && steps < maxSteps) {
      // Get proper context for each action we consider
      const beforeSit = gameStateToSituation(state, "up");

      // Select action with exploration
      const action = agent.selectWithExploration(
        beforeSit,
        currentEpsilon
      ) as Direction;

      // Update context for chosen action
      const beforeSitWithAction = gameStateToSituation(state, action);

      // Execute action
      const result = executeMove(state, action);
      const afterSit = gameStateToSituation(result.newState, action);

      // Compute reward
      const delta = sokobanEncoders.delta(beforeSitWithAction, afterSit);
      const reward = computeReward(state, result.newState, delta);
      totalReward += reward;

      // Create transition
      const transition: Transition<SokobanState, SokobanContext> = {
        action,
        before: beforeSitWithAction,
        after: afterSit,
      };

      // Store for potential replay
      trajectory.push({ state: cloneState(state), action, reward, transition });

      // Learn online
      if (isWon(result.newState)) {
        agent.observeTerminal(transition, reward);
      } else {
        agent.observe(transition, reward);
      }

      state = result.newState;
      steps++;
    }

    // If we won, replay the trajectory backward to propagate value
    if (isWon(state)) {
      wins++;

      // Backward replay: re-observe with updated value estimates
      // This helps propagate the win reward backward through the trajectory
      for (let i = trajectory.length - 1; i >= 0; i--) {
        const { transition, reward } = trajectory[i]!;
        if (i === trajectory.length - 1) {
          agent.observeTerminal(transition, reward);
        } else {
          agent.observe(transition, reward);
        }
      }
    }

    totalSteps += steps;
  }

  return {
    episodes,
    wins,
    totalSteps,
    avgStepsPerEpisode: totalSteps / episodes,
    avgReward: totalReward / Math.max(1, totalSteps),
  };
}

// ============================================================================
// Solving (using learned agent)
// ============================================================================

/**
 * Attempt to solve a level using the learned agent.
 * Uses exploration to avoid getting stuck, and learns from the attempt.
 */
export function solve(
  agent: TAMAgent<SokobanState, SokobanContext>,
  initialState: GameState,
  maxSteps: number = 200
): { solved: boolean; moves: Direction[]; finalState: GameState } {
  let state = cloneState(initialState);
  const moves: Direction[] = [];
  const visitedStates = new Set<string>();
  let stuckCount = 0;

  while (!isWon(state) && moves.length < maxSteps) {
    // Track visited states to detect loops
    const stateKey = `${posKey(state.player)}|${Array.from(state.boxes)
      .sort()
      .join(",")}`;

    if (visitedStates.has(stateKey)) {
      stuckCount++;
    } else {
      stuckCount = 0;
      visitedStates.add(stateKey);
    }

    const beforeSit = gameStateToSituation(state, "up");

    // Use exploration if stuck, otherwise mostly greedy
    const epsilon = stuckCount > 3 ? 0.5 : 0.1;
    const action = agent.selectWithExploration(beforeSit, epsilon) as Direction;

    const result = executeMove(state, action);
    const afterSit = gameStateToSituation(result.newState, action);

    // Compute reward
    const delta = sokobanEncoders.delta(beforeSit, afterSit);
    const reward = computeReward(state, result.newState, delta);

    // Learn from this step
    const transition: Transition<SokobanState, SokobanContext> = {
      action,
      before: beforeSit,
      after: afterSit,
    };

    if (isWon(result.newState)) {
      agent.observeTerminal(transition, reward);
    } else {
      agent.observe(transition, reward);
    }

    moves.push(action);
    state = result.newState;

    // If completely stuck (no valid moves change state), break
    if (stuckCount > 20) {
      break;
    }
  }

  return {
    solved: isWon(state),
    moves,
    finalState: state,
  };
}

/**
 * Get action values for display in UI.
 */
export function getActionValues(
  agent: TAMAgent<SokobanState, SokobanContext>,
  state: GameState
): Array<{ action: string; value: number; agency: number }> {
  const sit = gameStateToSituation(state, "up");
  return agent.getActionValues(sit);
}

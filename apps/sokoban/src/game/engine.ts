/**
 * Sokoban game engine
 */

import {
  type GameState,
  type Direction,
  type Position,
  type MoveResult,
  type TransitionData,
  DIRECTIONS,
  posKey,
  keyToPos,
} from "./types";

/**
 * Create initial game state from a level string
 * Legend: # = wall, . = goal, @ = player, + = player on goal,
 *         $ = box, * = box on goal, (space) = empty
 */
export function parseLevel(levelString: string): GameState {
  const lines = levelString.trim().split("\n");
  const height = lines.length;
  const width = Math.max(...lines.map((l) => l.length));

  const walls = new Set<string>();
  const goals = new Set<string>();
  const boxes = new Set<string>();
  let player: Position = { x: 0, y: 0 };

  for (let y = 0; y < lines.length; y++) {
    const line = lines[y]!;
    for (let x = 0; x < line.length; x++) {
      const char = line[x];
      const key = posKey({ x, y });

      switch (char) {
        case "#":
          walls.add(key);
          break;
        case ".":
          goals.add(key);
          break;
        case "@":
          player = { x, y };
          break;
        case "+":
          player = { x, y };
          goals.add(key);
          break;
        case "$":
          boxes.add(key);
          break;
        case "*":
          boxes.add(key);
          goals.add(key);
          break;
      }
    }
  }

  return { width, height, walls, goals, boxes, player };
}

/**
 * Clone game state (deep copy of Sets)
 */
export function cloneState(state: GameState): GameState {
  return {
    ...state,
    walls: new Set(state.walls),
    goals: new Set(state.goals),
    boxes: new Set(state.boxes),
    player: { ...state.player },
  };
}

/**
 * Check if the game is won (all boxes on goals)
 */
export function isWon(state: GameState): boolean {
  if (state.boxes.size !== state.goals.size) return false;
  for (const box of state.boxes) {
    if (!state.goals.has(box)) return false;
  }
  return true;
}

/**
 * Get what's at a position
 */
export function getCell(
  state: GameState,
  pos: Position
): "wall" | "box" | "empty" {
  const key = posKey(pos);
  if (state.walls.has(key)) return "wall";
  if (state.boxes.has(key)) return "box";
  return "empty";
}

/**
 * Get context for a potential move (used by TA model)
 */
export function getMoveContext(
  state: GameState,
  direction: Direction
): TransitionData["context"] {
  const [dx, dy] = DIRECTIONS[direction];
  const ahead: Position = { x: state.player.x + dx, y: state.player.y + dy };
  const twoAhead: Position = {
    x: state.player.x + 2 * dx,
    y: state.player.y + 2 * dy,
  };

  const aheadKey = posKey(ahead);
  const twoAheadKey = posKey(twoAhead);

  const wallAhead = state.walls.has(aheadKey);
  const boxAhead = state.boxes.has(aheadKey);
  const wallBehindBox = boxAhead && state.walls.has(twoAheadKey);
  const boxBehindBox = boxAhead && state.boxes.has(twoAheadKey);

  return { wallAhead, boxAhead, wallBehindBox, boxBehindBox };
}

/**
 * Execute a move and return the result
 */
export function move(state: GameState, direction: Direction): MoveResult {
  const [dx, dy] = DIRECTIONS[direction];
  const newPlayerPos: Position = {
    x: state.player.x + dx,
    y: state.player.y + dy,
  };
  const newPlayerKey = posKey(newPlayerPos);

  // Check wall collision
  if (state.walls.has(newPlayerKey)) {
    return { success: false, newState: state, pushed: false, blocked: true };
  }

  // Check box push
  if (state.boxes.has(newPlayerKey)) {
    const newBoxPos: Position = {
      x: newPlayerPos.x + dx,
      y: newPlayerPos.y + dy,
    };
    const newBoxKey = posKey(newBoxPos);

    // Can't push into wall or another box
    if (state.walls.has(newBoxKey) || state.boxes.has(newBoxKey)) {
      return { success: false, newState: state, pushed: false, blocked: true };
    }

    // Push the box
    const newState = cloneState(state);
    newState.boxes.delete(newPlayerKey);
    newState.boxes.add(newBoxKey);
    newState.player = newPlayerPos;

    return { success: true, newState, pushed: true, blocked: false };
  }

  // Simple move
  const newState = cloneState(state);
  newState.player = newPlayerPos;

  return { success: true, newState, pushed: false, blocked: false };
}

/**
 * Generate a transition record for the TA model
 */
export function recordTransition(
  before: GameState,
  after: GameState,
  action: Direction
): TransitionData {
  return {
    action,
    beforePlayer: before.player,
    afterPlayer: after.player,
    beforeBoxes: Array.from(before.boxes).map(keyToPos),
    afterBoxes: Array.from(after.boxes).map(keyToPos),
    context: getMoveContext(before, action),
  };
}

/**
 * Convert state to a simple grid for display
 */
export function stateToGrid(state: GameState): string[][] {
  const grid: string[][] = [];

  for (let y = 0; y < state.height; y++) {
    const row: string[] = [];
    for (let x = 0; x < state.width; x++) {
      const key = posKey({ x, y });
      const isGoal = state.goals.has(key);
      const isBox = state.boxes.has(key);
      const isPlayer = state.player.x === x && state.player.y === y;
      const isWall = state.walls.has(key);

      if (isWall) row.push("#");
      else if (isPlayer && isGoal) row.push("+");
      else if (isPlayer) row.push("@");
      else if (isBox && isGoal) row.push("*");
      else if (isBox) row.push("$");
      else if (isGoal) row.push(".");
      else row.push(" ");
    }
    grid.push(row);
  }

  return grid;
}

// ============================================================================
// Sample levels
// ============================================================================

export const LEVELS = {
  // Trivial: one box, one goal, straight push
  trivial: `
#####
#@$.#
#####
`.trim(),

  // Simple: one box, need to navigate
  simple: `
######
#    #
# @$ #
#  . #
######
`.trim(),

  // Two boxes
  twoBox: `
########
#      #
# @$$  #
#  ..  #
########
`.trim(),

  // Classic small level
  classic1: `
  ####
###  ####
#     $ #
# #  #$ #
# . .#@ #
#########
`.trim(),

  // Slightly harder
  classic2: `
########
#  @   #
# $$ $ #
# .#.  #
#  .   #
########
`.trim(),
};

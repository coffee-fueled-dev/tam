/**
 * Core types for Sokoban game
 */

export type Cell =
  | "empty"
  | "wall"
  | "goal"
  | "player"
  | "box"
  | "player_on_goal"
  | "box_on_goal";

export type Direction = "up" | "down" | "left" | "right";

export const DIRECTIONS: Record<Direction, [number, number]> = {
  up: [0, -1],
  down: [0, 1],
  left: [-1, 0],
  right: [1, 0],
};

export interface Position {
  x: number;
  y: number;
}

export interface GameState {
  width: number;
  height: number;
  walls: Set<string>; // "x,y" format
  goals: Set<string>;
  boxes: Set<string>;
  player: Position;
}

export interface MoveResult {
  success: boolean;
  newState: GameState;
  pushed: boolean; // Did we push a box?
  blocked: boolean; // Was the move blocked?
}

// For TA model
export interface TransitionData {
  action: Direction;
  beforePlayer: Position;
  afterPlayer: Position;
  beforeBoxes: Position[];
  afterBoxes: Position[];
  context: {
    wallAhead: boolean;
    boxAhead: boolean;
    wallBehindBox: boolean;
    boxBehindBox: boolean;
  };
}

export function posKey(p: Position): string {
  return `${p.x},${p.y}`;
}

export function keyToPos(key: string): Position {
  const [x, y] = key.split(",").map(Number);
  return { x: x!, y: y! };
}

/**
 * Core types for 2D physics simulation
 */

export interface Vec2 {
  x: number;
  y: number;
}

export interface Ball {
  id: string;
  pos: Vec2;
  vel: Vec2;
  mass: number;
  radius: number;
  color: string;
}

export interface Wall {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Zone {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  friction: number; // 0 = ice, 1 = normal, 2+ = mud
  color: string;
}

export interface World {
  width: number;
  height: number;
  balls: Ball[];
  walls: Wall[];
  zones: Zone[];
  gravity: Vec2;
  defaultFriction: number;
}

export interface Force {
  x: number;
  y: number;
}

// For TAM learning
export interface PhysicsState {
  ballPos: Vec2;
  ballVel: Vec2;
  nearWall: boolean;
  wallDirection: Vec2 | null; // Direction to nearest wall
  friction: number;
  mass: number;
}

export interface PhysicsContext {
  nearWall: boolean;
  friction: number;
  mass: number;
}

// Utility functions
export function vec2(x: number, y: number): Vec2 {
  return { x, y };
}

export function addVec(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function subVec(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function scaleVec(v: Vec2, s: number): Vec2 {
  return { x: v.x * s, y: v.y * s };
}

export function lenVec(v: Vec2): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

export function normVec(v: Vec2): Vec2 {
  const len = lenVec(v);
  if (len === 0) return { x: 0, y: 0 };
  return { x: v.x / len, y: v.y / len };
}

export function dotVec(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}

export function reflectVec(v: Vec2, normal: Vec2): Vec2 {
  const d = 2 * dotVec(v, normal);
  return { x: v.x - d * normal.x, y: v.y - d * normal.y };
}

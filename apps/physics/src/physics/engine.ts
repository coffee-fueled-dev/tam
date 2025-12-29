/**
 * Simple 2D physics engine
 */

import {
  type World,
  type Ball,
  type Force,
  type Vec2,
  type PhysicsState,
  type PhysicsContext,
  vec2,
  addVec,
  scaleVec,
  subVec,
  lenVec,
  normVec,
  dotVec,
  reflectVec,
} from "./types";

/**
 * Create a default world with walls around the edges
 */
export function createWorld(
  width: number,
  height: number,
  options?: { gravity?: boolean }
): World {
  return {
    width,
    height,
    balls: [],
    walls: [],
    zones: [
      // Ice zone (low friction)
      {
        id: "ice",
        x: 50,
        y: 50,
        width: 150,
        height: 150,
        friction: 0.02,
        color: "rgba(100, 200, 255, 0.3)",
      },
      // Mud zone (high friction)
      {
        id: "mud",
        x: width - 200,
        y: height - 200,
        width: 150,
        height: 150,
        friction: 0.95,
        color: "rgba(139, 90, 43, 0.3)",
      },
    ],
    // Gravity pulls downward (positive y) - strong enough to see acceleration
    gravity: options?.gravity ? vec2(0, 0.8) : vec2(0, 0),
    defaultFriction: 0.01, // Low friction for bouncier physics
  };
}

/**
 * Add a ball to the world
 */
export function addBall(
  world: World,
  x: number,
  y: number,
  mass: number = 1,
  color: string = "#f6ad55"
): Ball {
  const ball: Ball = {
    id: `ball-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    pos: vec2(x, y),
    vel: vec2(0, 0),
    mass,
    radius: 15 + mass * 5,
    color,
  };
  world.balls.push(ball);
  return ball;
}

/**
 * Apply a force to a ball
 */
export function applyForce(ball: Ball, force: Force): void {
  // F = ma, so a = F/m
  const accel = scaleVec(force, 1 / ball.mass);
  ball.vel = addVec(ball.vel, accel);
}

/**
 * Get friction at a position
 */
export function getFrictionAt(world: World, pos: Vec2): number {
  for (const zone of world.zones) {
    if (
      pos.x >= zone.x &&
      pos.x <= zone.x + zone.width &&
      pos.y >= zone.y &&
      pos.y <= zone.y + zone.height
    ) {
      return zone.friction;
    }
  }
  return world.defaultFriction;
}

/**
 * Check if near a wall and get the wall normal
 */
export function getNearestWallInfo(
  world: World,
  pos: Vec2,
  radius: number
): { near: boolean; normal: Vec2 | null; distance: number } {
  const threshold = radius + 20; // Detection distance

  let nearestDist = Infinity;
  let nearestNormal: Vec2 | null = null;

  // Check edges
  // Left wall
  if (pos.x < threshold) {
    const dist = pos.x;
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestNormal = vec2(1, 0);
    }
  }
  // Right wall
  if (pos.x > world.width - threshold) {
    const dist = world.width - pos.x;
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestNormal = vec2(-1, 0);
    }
  }
  // Top wall
  if (pos.y < threshold) {
    const dist = pos.y;
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestNormal = vec2(0, 1);
    }
  }
  // Bottom wall
  if (pos.y > world.height - threshold) {
    const dist = world.height - pos.y;
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestNormal = vec2(0, -1);
    }
  }

  return {
    near: nearestDist < threshold,
    normal: nearestNormal,
    distance: nearestDist,
  };
}

/**
 * Step the physics simulation
 */
export function stepWorld(world: World, dt: number = 1 / 60): void {
  for (const ball of world.balls) {
    // Apply gravity
    ball.vel = addVec(ball.vel, scaleVec(world.gravity, dt));

    // Apply friction
    const friction = getFrictionAt(world, ball.pos);
    ball.vel = scaleVec(ball.vel, 1 - friction * dt);

    // Update position
    ball.pos = addVec(ball.pos, scaleVec(ball.vel, dt * 60));

    // Collision with walls (edges)
    const restitution = 0.8; // Bounce factor

    if (ball.pos.x - ball.radius < 0) {
      ball.pos.x = ball.radius;
      ball.vel.x = -ball.vel.x * restitution;
    }
    if (ball.pos.x + ball.radius > world.width) {
      ball.pos.x = world.width - ball.radius;
      ball.vel.x = -ball.vel.x * restitution;
    }
    if (ball.pos.y - ball.radius < 0) {
      ball.pos.y = ball.radius;
      ball.vel.y = -ball.vel.y * restitution;
    }
    if (ball.pos.y + ball.radius > world.height) {
      ball.pos.y = world.height - ball.radius;
      ball.vel.y = -ball.vel.y * restitution;
    }

    // Clamp velocity to prevent explosion
    const maxVel = 20;
    const velLen = lenVec(ball.vel);
    if (velLen > maxVel) {
      ball.vel = scaleVec(normVec(ball.vel), maxVel);
    }
  }
}

/**
 * Get physics state for TAM learning
 */
export function getPhysicsState(world: World, ball: Ball): PhysicsState {
  const wallInfo = getNearestWallInfo(world, ball.pos, ball.radius);
  const friction = getFrictionAt(world, ball.pos);

  return {
    ballPos: { ...ball.pos },
    ballVel: { ...ball.vel },
    nearWall: wallInfo.near,
    wallDirection: wallInfo.normal,
    friction,
    mass: ball.mass,
  };
}

/**
 * Get physics context for TAM (features that affect dynamics)
 */
export function getPhysicsContext(state: PhysicsState): PhysicsContext {
  return {
    nearWall: state.nearWall,
    friction: state.friction,
    mass: state.mass,
  };
}

/**
 * Compute delta between two physics states
 */
export function computeDelta(before: PhysicsState, after: PhysicsState): Vec2[] {
  return [
    subVec(after.ballPos, before.ballPos),
    subVec(after.ballVel, before.ballVel),
  ];
}

/**
 * Clone world state
 */
export function cloneWorld(world: World): World {
  return {
    ...world,
    balls: world.balls.map((b) => ({
      ...b,
      pos: { ...b.pos },
      vel: { ...b.vel },
    })),
    zones: [...world.zones],
    walls: [...world.walls],
    gravity: { ...world.gravity },
  };
}


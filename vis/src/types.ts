export interface Obstacle {
  position: [number, number, number];
  radius: number;
}

export interface Frame {
  step: number;
  episode: number;
  mu_t: number[][]; // (T, state_dim) planned tube trajectory (relative)
  sigma_t: number[][]; // (T, state_dim) tube radii (per-dimension)
  actual_path: number[][]; // (T, state_dim) actual path taken
  current_pos: number[]; // (state_dim,) current position
  active_goals: number[][]; // List of active goal positions, each (state_dim,)
  energy: number | null;
  max_energy: number | null;
  goal_reached: boolean;
  agency?: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  loss?: number | null;
}

export interface Bounds {
  min: number[];
  max: number[];
}

export interface Metadata {
  obstacles: Array<[number, number, number, number]>; // [x, y, z, radius]
  bounds: Bounds;
  state_dim: number;
  total_steps: number;
  compression: "none" | "gzip" | "delta";
  created_at?: string;
  world_seed?: number;
  config?: Record<string, any>;
}

export interface GoalStat {
  moves_taken: number;
  agency: {
    mean: number;
    std: number;
  };
}

export interface TrainingMetrics {
  loss_history: number[];
  goal_stats: GoalStat[];
}

export interface VisualizationData {
  frames: Frame[];
  metadata: Metadata;
  trainingMetrics?: TrainingMetrics;
  reachedGoalsByFrame: Record<number, number[][]>;
}

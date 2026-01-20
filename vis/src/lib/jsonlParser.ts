import type { Frame, Metadata, TrainingMetrics, VisualizationData } from "../types";

/**
 * Decode delta-encoded frames by merging with previous frame.
 */
function decodeDeltaFrames(lines: string[]): Frame[] {
  const frames: Frame[] = [];
  let previousFrame: Frame | null = null;

  for (const line of lines) {
    const deltaFrame = JSON.parse(line) as Partial<Frame>;
    let fullFrame: Frame;

    if (previousFrame === null) {
      // First frame: use as-is
      fullFrame = { ...deltaFrame } as Frame;
    } else {
      // Merge with previous frame
      fullFrame = { ...previousFrame };
      for (const [key, value] of Object.entries(deltaFrame)) {
        if (value !== undefined) {
          (fullFrame as any)[key] = value;
        }
      }
    }

    frames.push(fullFrame);
    previousFrame = fullFrame;
  }

  return frames;
}

/**
 * Parse JSONL file content into frames array.
 */
export function parseJSONL(
  content: string,
  compression: "none" | "gzip" | "delta" = "none"
): Frame[] {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (compression === "delta") {
    return decodeDeltaFrames(lines);
  } else {
    return lines.map((line, idx) => {
      try {
        return JSON.parse(line) as Frame;
      } catch (e) {
        console.error(`Error parsing line ${idx}:`, e, line.substring(0, 100));
        throw e;
      }
    });
  }
}

/**
 * Build a map of reached goals per frame (cumulative).
 * Tracks goals that have been reached (removed from active_goals).
 */
export function buildReachedGoalsByFrame(frames: Frame[]): Record<number, number[][]> {
  const reachedGoalsByFrame: Record<number, number[][]> = {};
  const cumulativeReachedGoals: number[][] = [];
  const seenGoals = new Set<string>(); // Track goals we've seen

    frames.forEach((frame, idx) => {
      // Track goals that were active in previous frame but are no longer active
      // This indicates they were reached
      if (idx > 0) {
        const prevFrame = frames[idx - 1];
        if (prevFrame) {
          const prevActiveGoals = prevFrame.active_goals || [];
          const currActiveGoals = frame.active_goals || [];
          
          // Find goals that were in previous frame but not in current frame
          prevActiveGoals.forEach((prevGoal) => {
            const goalKey = prevGoal.join(",");
            const stillActive = currActiveGoals.some(
              (currGoal) => currGoal.join(",") === goalKey
            );
            
            if (!stillActive && !seenGoals.has(goalKey)) {
              // Goal was reached (removed from active list)
              seenGoals.add(goalKey);
              cumulativeReachedGoals.push([...prevGoal]);
            }
          });
        }
      }
      
      // Store cumulative reached goals for this frame
      reachedGoalsByFrame[idx] = [...cumulativeReachedGoals];
    });

  return reachedGoalsByFrame;
}

/**
 * Parse environment data JSONL and build frames from sequential steps.
 */
function parseEnvironmentData(content: string): Array<{
  agent_position: number[];
  active_goals: number[][];
  obstacles?: number[][];
  energy?: number;
  max_energy?: number;
  move_number?: number;
}> {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  
  return lines.map((line) => JSON.parse(line));
}

/**
 * Parse training stats JSONL.
 */
function parseTrainingStats(content: string): Array<{
  binding_loss: number;
  agency_cost: number;
  knot_components?: any;
  mu_t?: number[][];  // Selected tube trajectory (relative to start)
  sigma_t?: number[][];  // Selected sigma (per-dimension precision)
}> {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  
  return lines.map((line) => JSON.parse(line));
}

/**
 * Build frames from environment data and training stats.
 * Frames are applied sequentially, with tube/path data overlaid from training_stats.
 */
function buildFramesFromEnvironmentData(
  envData: ReturnType<typeof parseEnvironmentData>,
  trainingStats: ReturnType<typeof parseTrainingStats>,
  metadata: Metadata
): Frame[] {
  const frames: Frame[] = [];
  
  // Group environment steps by move_number
  const stepsByMove: Record<number, typeof envData> = {};
  envData.forEach((step, lineIdx) => {
    // Line number = step number (0-indexed)
    const moveNum = step.move_number ?? Math.floor(lineIdx / 30); // Fallback: estimate from line number
    if (!stepsByMove[moveNum]) {
      stepsByMove[moveNum] = [];
    }
    stepsByMove[moveNum].push(step);
  });
  
  // Build frames: one per environment step, applied sequentially
  let frameIdx = 0;
  const moveNumbers = Object.keys(stepsByMove)
    .map(Number)
    .sort((a, b) => a - b);
  
  for (let moveIdx = 0; moveIdx < moveNumbers.length; moveIdx++) {
    const moveNum = moveNumbers[moveIdx];
    const steps = stepsByMove[moveNum];
    
    // Get training stats for this move (line number = move number, 0-indexed)
    // moveNum is 1-indexed (from environment_data), so use moveNum - 1 for 0-indexed array
    // Fallback to moveIdx if moveNum is out of bounds
    const statsIdx = moveNum > 0 ? moveNum - 1 : moveIdx;
    const moveStats = trainingStats[statsIdx] || null;
    
    // Build actual_path from sequential agent positions
    const actualPath: number[][] = steps.map((s) => [...s.agent_position]);
    
    // Get mu_t and sigma_t from training_stats if available
    let mu_t: number[][] = [];
    let sigma_t: number[][] = [];
    
    if (moveStats?.mu_t && Array.isArray(moveStats.mu_t)) {
      mu_t = moveStats.mu_t.map((pt) => [...pt]);
    } else if (steps.length > 1) {
      // Fallback: compute mu_t as relative displacements from start
      const startPos = steps[0].agent_position;
      for (let i = 1; i < steps.length; i++) {
        const relPos = steps[i].agent_position.map(
          (val, idx) => val - startPos[idx]
        );
        mu_t.push(relPos);
      }
    }
    
    if (moveStats?.sigma_t && Array.isArray(moveStats.sigma_t)) {
      sigma_t = moveStats.sigma_t.map((s) => [...s]);
    }
    
    // Create a frame for each step in this move
    // Apply frames sequentially, overlay FULL tube/path data from training_stats for entire move
    // All frames in the same move show the complete tube and path
    const moveStartPos = steps[0].agent_position;
    
    // Keep mu_t relative (don't convert to absolute - scene will add move_start_pos)
    
    for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
      const step = steps[stepIdx];
      
      const frame: Frame = {
        step: frameIdx,
        episode: moveNum,
        current_pos: [...step.agent_position],
        active_goals: step.active_goals.map((g) => [...g]),
        move_start_pos: [...moveStartPos],  // Store move start position for tube rendering
        // Overlay FULL tube/path data from training_stats for entire move
        // mu_t is relative to move_start_pos (scene will add move_start_pos when rendering)
        mu_t: mu_t,  // Full planned trajectory for entire move (relative to move_start_pos)
        sigma_t: sigma_t,  // Full sigma for entire move
        actual_path: actualPath,  // Full actual path for entire move (absolute coords)
        energy: step.energy,
        max_energy: step.max_energy,
        loss: moveStats?.binding_loss,
      };
      
      frames.push(frame);
      frameIdx++;
    }
  }
  
  return frames;
}

/**
 * Parse visualization data from files.
 */
export async function parseVisualizationData(
  jsonlContent: string,
  metadataContent: string | null,
  trainingMetricsContent: string | null,
  trainingConfigContent: string | null = null,
  environmentDataContent: string | null = null
): Promise<VisualizationData> {
  // Parse metadata
  let metadata: Metadata = metadataContent
    ? JSON.parse(metadataContent)
    : {
        obstacles: [],
        bounds: { min: [-2, -2, -2], max: [12, 12, 12] },
        state_dim: 3,
        total_steps: 0,
        reached_goals: [],
        compression: "none",
      };

  // Extract obstacles and other config from training config file if available
  if (trainingConfigContent) {
    try {
      const trainingConfig = JSON.parse(trainingConfigContent);
      const envConfig = trainingConfig.environment_config || {};
      
      // Extract obstacles if present
      if (envConfig.obstacles && Array.isArray(envConfig.obstacles)) {
        // Convert obstacles format: [x, y, z, radius] -> ensure 3D format
        metadata.obstacles = envConfig.obstacles.map((obs: any) => {
          if (Array.isArray(obs) && obs.length >= 3) {
            // Already in [x, y, z, radius] format
            return obs;
          } else if (Array.isArray(obs) && obs.length === 2) {
            // [x, y] format, add z=0 and radius=0.5
            return [obs[0], obs[1], 0, 0.5];
          }
          return obs;
        });
      }
      
      // Extract bounds if present
      if (envConfig.bounds) {
        metadata.bounds = envConfig.bounds;
      }
      
      // Extract state_dim if present
      if (envConfig.state_dim !== undefined) {
        metadata.state_dim = envConfig.state_dim;
      }
      
      // Store full config for reference
      metadata.config = trainingConfig;
    } catch (e) {
      console.warn("Failed to parse training config:", e);
    }
  }

  // Parse frames: use environment_data if available, otherwise use visualization_data
  let frames: Frame[];
  if (environmentDataContent) {
    // Build frames from environment_data.jsonl
    const envData = parseEnvironmentData(environmentDataContent);
    // trainingMetricsContent should be training_stats.jsonl when environment_data is present
    let trainingStats: ReturnType<typeof parseTrainingStats> = [];
    if (trainingMetricsContent) {
      try {
        // Try parsing as training_stats JSONL (one object per line)
        trainingStats = parseTrainingStats(trainingMetricsContent);
      } catch (e) {
        console.warn("Failed to parse training_stats, using empty array:", e);
        trainingStats = [];
      }
    }
    frames = buildFramesFromEnvironmentData(envData, trainingStats, metadata);
  } else {
    // Fall back to existing visualization_data.jsonl parsing
    frames = parseJSONL(jsonlContent, metadata.compression);
  }

  // Build reached goals map
  const reachedGoalsByFrame = buildReachedGoalsByFrame(frames);

  // Parse training metrics if provided (only goal_stats now, loss is per-frame)
  let trainingMetrics: TrainingMetrics | undefined;
  if (trainingMetricsContent) {
    try {
      const parsed = JSON.parse(trainingMetricsContent);
      trainingMetrics = {
        loss_history: [], // Loss is now per-frame, not aggregated
        goal_stats: parsed.goal_stats || [],
      };
    } catch (e) {
      console.warn("Failed to parse training metrics:", e);
    }
  }

  return {
    frames,
    metadata,
    trainingMetrics,
    reachedGoalsByFrame,
  };
}

/**
 * Read file as text using FileReader API.
 */
export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result && typeof e.target.result === "string") {
        resolve(e.target.result);
      } else {
        reject(new Error("Failed to read file as text"));
      }
    };
    reader.onerror = () => reject(new Error("File read error"));
    reader.readAsText(file);
  });
}

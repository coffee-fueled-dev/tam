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
 * Parse visualization data from files.
 */
export async function parseVisualizationData(
  jsonlContent: string,
  metadataContent: string | null,
  trainingMetricsContent: string | null
): Promise<VisualizationData> {
  // Parse metadata
  const metadata: Metadata = metadataContent
    ? JSON.parse(metadataContent)
    : {
        obstacles: [],
        bounds: { min: [-2, -2, -2], max: [12, 12, 12] },
        state_dim: 3,
        total_steps: 0,
        reached_goals: [],
        compression: "none",
      };

  // Parse frames
  const frames = parseJSONL(jsonlContent, metadata.compression);

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

import type { Frame, Metadata } from "@/types";
import { useEffect, useRef } from "react";

interface Scene2DProps {
  frames: Frame[];
  metadata: Metadata;
  currentFrameIndex: number;
  maxHistory: number;
  reachedGoalsByFrame: Record<number, number[][]>;
}

// Colors matching the original HTML visualization
const COLORS = {
  trajectory: "#6B7280",
  tube_high_agency: "#2563EB",
  tube_low_agency: "#CBD5E1",
  obstacle: "#EF4444",
  goal: "#F59E0B",
  goal_reached: "#FBBF24",
  start: "#1F2937",
  current_pos: "#3B82F6", // Blue
  boundary: "#9CA3AF",
};

export function Scene2D({
  frames,
  metadata,
  currentFrameIndex,
  maxHistory,
  reachedGoalsByFrame,
}: Scene2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Get bounds
    const bounds = metadata.bounds;
    const minX = bounds.min[0] ?? -5;
    const minY = bounds.min[1] ?? -5;
    const maxX = bounds.max[0] ?? 5;
    const maxY = bounds.max[1] ?? 5;

    // Calculate scale and offset to fit scene in canvas
    const sceneWidth = maxX - minX;
    const sceneHeight = maxY - minY;
    const padding = Math.max(sceneWidth, sceneHeight) * 0.1;
    const scaleX = (canvas.width - padding * 2) / sceneWidth;
    const scaleY = (canvas.height - padding * 2) / sceneHeight;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (canvas.width - sceneWidth * scale) / 2 - minX * scale;
    const offsetY = (canvas.height - sceneHeight * scale) / 2 - minY * scale;

    const toScreenX = (x: number) => x * scale + offsetX;
    const toScreenY = (y: number) => y * scale + offsetY;

    // Helper function to draw ellipse (for anisotropic affordance cones)
    const drawEllipse = (
      ctx: CanvasRenderingContext2D,
      centerX: number,
      centerY: number,
      radiusX: number,
      radiusY: number,
      fillStyle: string,
      strokeStyle: string,
      fillAlpha: number,
      strokeAlpha: number
    ) => {
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);

      // Fill
      ctx.fillStyle = fillStyle;
      ctx.globalAlpha = fillAlpha;
      ctx.fill();

      // Stroke
      ctx.strokeStyle = strokeStyle;
      ctx.globalAlpha = strokeAlpha;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.restore();
    };

    const render = () => {
      // Clear canvas
      ctx.fillStyle = "#f5f5f5";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw boundaries
      ctx.strokeStyle = COLORS.boundary;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.4;
      ctx.strokeRect(
        toScreenX(minX),
        toScreenY(minY),
        sceneWidth * scale,
        sceneHeight * scale
      );
      ctx.globalAlpha = 1.0;

      // Draw obstacles
      metadata.obstacles.forEach((obs) => {
        const [x, y, z, radius] = obs;
        ctx.fillStyle = COLORS.obstacle;
        ctx.globalAlpha = 0.25;
        ctx.beginPath();
        ctx.arc(toScreenX(x), toScreenY(y), radius * scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1.0;
      });

      if (
        currentFrameIndex < 0 ||
        currentFrameIndex >= frames.length
      ) {
        return;
      }

      const frame = frames[currentFrameIndex];

      // Render history: show last N moves
      const historyStart = Math.max(0, currentFrameIndex - maxHistory + 1);

      // Track which moves we've rendered tubes for (render once per move)
      const renderedMoves = new Set<number>();

      for (let i = historyStart; i <= currentFrameIndex; i++) {
        const histFrame = frames[i];
        if (!histFrame) continue;

        const isCurrent = i === currentFrameIndex;
        const opacity = isCurrent ? 0.85 : 0.5;
        const moveNum = histFrame.episode;

        // Render planned next step (mu_t) once per move
        // For single-step generation, mu_t is (1, state_dim) - single next step
        // mu_t is relative to move_start_pos, so add move_start_pos to convert to absolute
        if (histFrame.mu_t && Array.isArray(histFrame.mu_t) && histFrame.mu_t.length > 0 && !renderedMoves.has(moveNum)) {
          renderedMoves.add(moveNum);

          // Get move start position (fallback to current_pos if not available)
          const moveStartPos = histFrame.move_start_pos || histFrame.current_pos || [0, 0];

          // For single-step generation, mu_t has one point (the next step)
          const nextStep = histFrame.mu_t[0];
          if (nextStep && Array.isArray(nextStep)) {
            // Convert relative next step to absolute
            const nextStepAbs = [
              (nextStep[0] ?? 0) + (moveStartPos[0] ?? 0),
              (nextStep[1] ?? 0) + (moveStartPos[1] ?? 0)
            ];

            // Draw line from start to next step
            ctx.strokeStyle = COLORS.trajectory;
            ctx.lineWidth = 2;
            ctx.globalAlpha = opacity;
            ctx.beginPath();
            ctx.moveTo(toScreenX(moveStartPos[0] ?? 0), toScreenY(moveStartPos[1] ?? 0));
            ctx.lineTo(toScreenX(nextStepAbs[0] ?? 0), toScreenY(nextStepAbs[1] ?? 0));
            ctx.stroke();

            // Draw anisotropic affordance cone as ellipse at next step
            // sigma_t is (1, state_dim) - per-dimension precision [sigma_x, sigma_y]
            if (histFrame.sigma_t && Array.isArray(histFrame.sigma_t) && histFrame.sigma_t.length > 0) {
              const sigma = histFrame.sigma_t[0];

              // Extract per-dimension sigma values for anisotropic affordance
              let sigmaX = 0.1;
              let sigmaY = 0.1;

              if (Array.isArray(sigma)) {
                // sigma is [sigma_x, sigma_y, ...] - anisotropic per-dimension precision
                sigmaX = sigma[0] ?? 0.1;
                sigmaY = sigma[1] ?? 0.1;
              } else if (typeof sigma === 'number') {
                // Fallback: isotropic if single value (shouldn't happen with anisotropic Actor)
                sigmaX = sigmaY = sigma;
              }

              // Draw ellipse at next step position (projected affordance cone)
              const centerX = toScreenX(nextStepAbs[0] ?? 0);
              const centerY = toScreenY(nextStepAbs[1] ?? 0);
              const radiusX = Math.max(sigmaX * scale, 2);
              const radiusY = Math.max(sigmaY * scale, 2);

              drawEllipse(
                ctx,
                centerX,
                centerY,
                radiusX,
                radiusY,
                COLORS.trajectory,
                COLORS.trajectory,
                isCurrent ? 0.25 : 0.15,
                isCurrent ? 0.4 : 0.25
              );
            }
          }
        }

        // Render actual path
        if (
          histFrame.actual_path &&
          Array.isArray(histFrame.actual_path) &&
          histFrame.actual_path.length > 0
        ) {
          ctx.strokeStyle = COLORS.trajectory;
          ctx.lineWidth = 3;
          ctx.globalAlpha = isCurrent ? 0.9 : 0.6;
          ctx.beginPath();
          histFrame.actual_path.forEach((pt, idx) => {
            const x = pt[0] ?? 0;
            const y = pt[1] ?? 0;
            if (idx === 0) {
              ctx.moveTo(toScreenX(x), toScreenY(y));
            } else {
              ctx.lineTo(toScreenX(x), toScreenY(y));
            }
          });
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }
      }

      // Draw current position marker
      const pos = frame?.current_pos;
      if (pos && Array.isArray(pos) && pos.length >= 2) {
        const x = pos[0] ?? 0;
        const y = pos[1] ?? 0;
        ctx.fillStyle = COLORS.current_pos;
        ctx.beginPath();
        ctx.arc(toScreenX(x), toScreenY(y), 8, 0, Math.PI * 2);
        ctx.fill();
        // Outer ring for visibility
        ctx.strokeStyle = COLORS.current_pos;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(toScreenX(x), toScreenY(y), 12, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Draw active goals
      const activeGoals = frame?.active_goals || [];
      activeGoals.forEach((goalPos) => {
        if (goalPos && Array.isArray(goalPos) && goalPos.length >= 2) {
          const x = goalPos[0] ?? 0;
          const y = goalPos[1] ?? 0;
          ctx.fillStyle = COLORS.goal;
          ctx.beginPath();
          ctx.arc(toScreenX(x), toScreenY(y), 8, 0, Math.PI * 2);
          ctx.fill();
          // Outer ring
          ctx.strokeStyle = COLORS.goal;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(toScreenX(x), toScreenY(y), 12, 0, Math.PI * 2);
          ctx.stroke();
        }
      });

      // Draw reached goals
      const reachedGoals = reachedGoalsByFrame[currentFrameIndex] || [];
      reachedGoals.forEach((goalPos) => {
        if (goalPos && Array.isArray(goalPos) && goalPos.length >= 2) {
          const x = goalPos[0] ?? 0;
          const y = goalPos[1] ?? 0;
          // Filled circle
          ctx.fillStyle = COLORS.goal_reached;
          ctx.globalAlpha = 0.3;
          ctx.beginPath();
          ctx.arc(toScreenX(x), toScreenY(y), 6, 0, Math.PI * 2);
          ctx.fill();
          // Wireframe circle
          ctx.strokeStyle = COLORS.goal_reached;
          ctx.globalAlpha = 0.8;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(toScreenX(x), toScreenY(y), 6, 0, Math.PI * 2);
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }
      });
    };

    render();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [frames, metadata, currentFrameIndex, maxHistory, reachedGoalsByFrame]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full"
      style={{ pointerEvents: "auto" }}
    />
  );
}

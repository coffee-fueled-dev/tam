import {
  useState,
  useCallback,
  useEffect,
  useRef,
  type MouseEvent,
} from "react";
import "./index.css";
import {
  type World,
  type PhysicsState,
  createWorld,
  addBall,
  applyForce,
  stepWorld,
  getPhysicsState,
  cloneWorld,
} from "./physics";
import { PhysicsAgent, forceToAction, actionToForce } from "./physics/agent";

// ============================================================================
// Training Scenario Types
// ============================================================================

interface Scenario {
  id: string;
  name: string;
  direction: "up" | "down" | "left" | "right";
  color: string;
  world: World;
  errors: number[]; // prediction errors per epoch
  lastPrediction: { x: number; y: number } | null; // Last 1-second prediction
}

interface TrainingState {
  scenarios: Scenario[];
  epoch: number;
  isRunning: boolean;
  targetEpochs: number; // 0 = infinite
  horizonSeconds: number; // How long each scenario runs
  convergenceHistory: Array<{
    epoch: number;
    avgError: number;
    minError: number;
    maxError: number;
    checkpointErrors: number[]; // Error at each 1-second checkpoint
  }>;
}

const FPS = 60;
const PREDICTION_WINDOW = 60; // 1 second ahead
const FORCE_MAGNITUDE = 8;

function createScenario(
  id: string,
  name: string,
  direction: "up" | "down" | "left" | "right",
  color: string
): Scenario {
  const w = createWorld(200, 150, { gravity: true });
  // Start ball in center
  const ball = addBall(w, 100, 75, 1.0, color);
  ball.vel = { x: 0, y: 0 };
  return {
    id,
    name,
    direction,
    color,
    world: w,
    errors: [],
    lastPrediction: null,
  };
}

function resetScenario(scenario: Scenario): void {
  const ball = scenario.world.balls[0];
  if (ball) {
    ball.pos = { x: 100, y: 75 };
    ball.vel = { x: 0, y: 0 };
    // Apply initial force based on direction
    const force = actionToForce(scenario.direction, FORCE_MAGNITUDE);
    applyForce(ball, force);
  }
}

// Mini renderer for scenario canvas
function renderScenarioWorld(
  ctx: CanvasRenderingContext2D,
  scenario: Scenario,
  prediction: { x: number; y: number } | null
): void {
  const { world } = scenario;
  const { width, height } = world;

  // Clear
  ctx.fillStyle = "#0e0e14";
  ctx.fillRect(0, 0, width, height);

  // Draw zones
  for (const zone of world.zones) {
    ctx.fillStyle = zone.color;
    ctx.fillRect(zone.x, zone.y, zone.width, zone.height);
  }

  // Draw ball
  const ball = world.balls[0];
  if (ball) {
    // Ball
    ctx.fillStyle = scenario.color;
    ctx.beginPath();
    ctx.arc(ball.pos.x, ball.pos.y, ball.radius, 0, Math.PI * 2);
    ctx.fill();

    // Prediction ghost
    if (prediction) {
      ctx.strokeStyle = "rgba(0, 212, 255, 0.6)";
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(prediction.x, prediction.y, ball.radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Border
  ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, width, height);
}

// Convergence graph renderer
function renderConvergenceGraph(
  ctx: CanvasRenderingContext2D,
  history: TrainingState["convergenceHistory"],
  width: number,
  height: number
): void {
  ctx.fillStyle = "#0e0e14";
  ctx.fillRect(0, 0, width, height);

  if (history.length < 2) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
    ctx.font = "12px Azeret Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Run epochs to see convergence", width / 2, height / 2);
    return;
  }

  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const graphW = width - padding.left - padding.right;
  const graphH = height - padding.top - padding.bottom;

  // Find max error for scaling
  const maxError = Math.max(...history.map((h) => h.maxError), 50);

  // Draw axes
  ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
  ctx.font = "10px Azeret Mono, monospace";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (graphH * i) / 4;
    const val = maxError * (1 - i / 4);
    ctx.fillText(val.toFixed(0), padding.left - 5, y + 3);
  }

  // X-axis label
  ctx.textAlign = "center";
  ctx.fillText("Epoch", width / 2, height - 5);

  // Draw error band (min-max)
  ctx.fillStyle = "rgba(0, 212, 255, 0.1)";
  ctx.beginPath();
  history.forEach((h, i) => {
    const x = padding.left + (i / (history.length - 1)) * graphW;
    const yMin = padding.top + (1 - h.minError / maxError) * graphH;
    if (i === 0) ctx.moveTo(x, yMin);
    else ctx.lineTo(x, yMin);
  });
  for (let i = history.length - 1; i >= 0; i--) {
    const h = history[i]!;
    const x = padding.left + (i / (history.length - 1)) * graphW;
    const yMax = padding.top + (1 - h.maxError / maxError) * graphH;
    ctx.lineTo(x, yMax);
  }
  ctx.closePath();
  ctx.fill();

  // Draw average line
  ctx.strokeStyle = "#00d4ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((h, i) => {
    const x = padding.left + (i / (history.length - 1)) * graphW;
    const y = padding.top + (1 - h.avgError / maxError) * graphH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Current value label
  const last = history[history.length - 1]!;
  ctx.fillStyle = "#00d4ff";
  ctx.font = "11px Azeret Mono, monospace";
  ctx.textAlign = "left";
  ctx.fillText(
    `Avg: ${last.avgError.toFixed(1)}px`,
    padding.left + 5,
    padding.top + 15
  );
}

// ============================================================================
// Canvas Renderer
// ============================================================================

interface RenderState {
  world: World;
  selectedBall: string | null;
  autonomous: boolean;
  trajectoryPrediction: {
    predictedPos: { x: number; y: number };
    startPos: { x: number; y: number };
  } | null;
}

function renderWorld(ctx: CanvasRenderingContext2D, state: RenderState): void {
  const { world, selectedBall } = state;
  const { width, height } = world;

  // Clear
  ctx.fillStyle = "#0e0e14";
  ctx.fillRect(0, 0, width, height);

  // Draw grid (subtle)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
  ctx.lineWidth = 1;
  const gridSize = 40;
  for (let x = 0; x <= width; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Draw zones
  for (const zone of world.zones) {
    ctx.fillStyle = zone.color;
    ctx.fillRect(zone.x, zone.y, zone.width, zone.height);

    // Label
    ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
    ctx.font = "11px Azeret Mono, monospace";
    ctx.textAlign = "center";
    const label = zone.friction < 0.1 ? "ICE" : "MUD";
    ctx.fillText(label, zone.x + zone.width / 2, zone.y + zone.height / 2 + 4);
  }

  // Draw balls
  for (const b of world.balls) {
    const isSelected = b.id === selectedBall;

    // Shadow
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.beginPath();
    ctx.ellipse(
      b.pos.x + 3,
      b.pos.y + 3,
      b.radius,
      b.radius * 0.6,
      0,
      0,
      Math.PI * 2
    );
    ctx.fill();

    // Ball gradient
    const ballGrad = ctx.createRadialGradient(
      b.pos.x - b.radius * 0.3,
      b.pos.y - b.radius * 0.3,
      0,
      b.pos.x,
      b.pos.y,
      b.radius
    );
    ballGrad.addColorStop(0, "#ffe066");
    ballGrad.addColorStop(0.6, b.color);
    ballGrad.addColorStop(1, "#b87a00");

    ctx.fillStyle = ballGrad;
    ctx.beginPath();
    ctx.arc(b.pos.x, b.pos.y, b.radius, 0, Math.PI * 2);
    ctx.fill();

    // Selection ring
    if (isSelected) {
      ctx.strokeStyle = "#00d4ff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(b.pos.x, b.pos.y, b.radius + 4, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Velocity indicator
    if (Math.abs(b.vel.x) > 0.5 || Math.abs(b.vel.y) > 0.5) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(b.pos.x, b.pos.y);
      ctx.lineTo(b.pos.x + b.vel.x * 5, b.pos.y + b.vel.y * 5);
      ctx.stroke();
    }
  }

  // Draw trajectory prediction target (where agent thinks ball will be in 1 second)
  const { trajectoryPrediction: trajPred } = state;
  const selectedBallObj = world.balls.find((b) => b.id === selectedBall);
  if (trajPred && state.autonomous && selectedBallObj) {
    const { predictedPos, startPos } = trajPred;
    const radius = selectedBallObj.radius;

    // Draw dashed line from start to predicted end
    ctx.strokeStyle = "rgba(0, 212, 255, 0.4)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    ctx.moveTo(startPos.x, startPos.y);
    ctx.lineTo(predictedPos.x, predictedPos.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw predicted ball position (hollow circle matching ball size)
    ctx.strokeStyle = "rgba(0, 212, 255, 0.8)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(predictedPos.x, predictedPos.y, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Small crosshair at center
    const crossSize = 6;
    ctx.beginPath();
    ctx.moveTo(predictedPos.x - crossSize, predictedPos.y);
    ctx.lineTo(predictedPos.x + crossSize, predictedPos.y);
    ctx.moveTo(predictedPos.x, predictedPos.y - crossSize);
    ctx.lineTo(predictedPos.x, predictedPos.y + crossSize);
    ctx.stroke();

    // Label
    ctx.fillStyle = "rgba(0, 212, 255, 0.8)";
    ctx.font = "10px Azeret Mono, monospace";
    ctx.fillText(
      "predicted (1s)",
      predictedPos.x + radius + 5,
      predictedPos.y + 3
    );
  }

  // Draw border
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, width - 2, height - 2);
}

// ============================================================================
// Training View Component
// ============================================================================

function TrainingView({ agent }: { agent: PhysicsAgent }) {
  const scenarioRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const graphRef = useRef<HTMLCanvasElement>(null);

  const [training, setTraining] = useState<TrainingState>(() => ({
    scenarios: [
      createScenario("up", "↑ Up", "up", "#68d391"),
      createScenario("down", "↓ Down", "down", "#fc8181"),
      createScenario("left", "← Left", "left", "#4fd1c5"),
      createScenario("right", "→ Right", "right", "#f6ad55"),
    ],
    epoch: 0,
    isRunning: false,
    targetEpochs: 1000,
    horizonSeconds: 10, // 10 seconds per scenario
    convergenceHistory: [],
  }));

  const [stats, setStats] = useState(() => agent.getStats());

  // Helper to run a single epoch with proper 1-second checkpoint predictions
  const executeEpoch = useCallback(
    (prev: TrainingState): TrainingState => {
      const totalFrames = prev.horizonSeconds * FPS;

      const newScenarios = prev.scenarios.map((s) => ({
        ...s,
        world: cloneWorld(s.world),
        errors: [...s.errors],
      }));

      // Reset all scenarios to starting position
      newScenarios.forEach(resetScenario);

      // Track checkpoint states for each scenario
      // We observe CUMULATIVE 60-frame deltas, not per-frame deltas
      const checkpointStates: Map<string, PhysicsState> = new Map();
      const pendingPredictions: Map<
        string,
        { predictedPos: { x: number; y: number }; checkpointFrame: number }
      > = new Map();

      const checkpointErrors: number[] = [];

      // Run all scenarios for totalFrames
      for (let frame = 0; frame < totalFrames; frame++) {
        for (const scenario of newScenarios) {
          const ball = scenario.world.balls[0];
          if (!ball) continue;

          // At each checkpoint start, record state and make prediction
          if (
            frame % PREDICTION_WINDOW === 0 &&
            frame < totalFrames - PREDICTION_WINDOW
          ) {
            const startState = getPhysicsState(scenario.world, ball);
            checkpointStates.set(scenario.id, startState);

            // Make prediction: agent predicts the 60-frame cumulative delta
            agent.predictBefore(startState, "physics");
            const pred = agent.getLastPrediction();

            if (pred) {
              // Now prediction IS the 60-frame delta (no scaling needed!)
              const predictedPos = {
                x: pred.pos.x,
                y: pred.pos.y,
              };

              pendingPredictions.set(scenario.id, {
                predictedPos,
                checkpointFrame: frame + PREDICTION_WINDOW,
              });

              scenario.lastPrediction = predictedPos;
            }
          }

          // Step physics
          stepWorld(scenario.world);

          // At checkpoint end, observe the CUMULATIVE delta and evaluate prediction
          if ((frame + 1) % PREDICTION_WINDOW === 0) {
            const startState = checkpointStates.get(scenario.id);
            if (startState) {
              const endState = getPhysicsState(scenario.world, ball);

              // Observe the 60-frame cumulative transition
              // This is what the agent learns from!
              agent.observe(startState, endState, "physics");

              // Evaluate prediction accuracy
              const pending = pendingPredictions.get(scenario.id);
              if (pending) {
                const error = Math.sqrt(
                  (ball.pos.x - pending.predictedPos.x) ** 2 +
                    (ball.pos.y - pending.predictedPos.y) ** 2
                );
                checkpointErrors.push(error);
                scenario.errors.push(error);
                pendingPredictions.delete(scenario.id);
              }

              checkpointStates.delete(scenario.id);
            }
          }
        }
      }

      const avgError =
        checkpointErrors.length > 0
          ? checkpointErrors.reduce((a, b) => a + b, 0) /
            checkpointErrors.length
          : 0;
      const minError =
        checkpointErrors.length > 0 ? Math.min(...checkpointErrors) : 0;
      const maxError =
        checkpointErrors.length > 0 ? Math.max(...checkpointErrors) : 0;

      return {
        scenarios: newScenarios,
        epoch: prev.epoch + 1,
        isRunning: prev.isRunning,
        targetEpochs: prev.targetEpochs,
        horizonSeconds: prev.horizonSeconds,
        convergenceHistory: [
          ...prev.convergenceHistory,
          {
            epoch: prev.epoch + 1,
            avgError,
            minError,
            maxError,
            checkpointErrors,
          },
        ],
      };
    },
    [agent]
  );

  // Run a single epoch manually
  const runEpoch = useCallback(() => {
    setTraining((prev) => executeEpoch(prev));
    setStats(agent.getStats());
  }, [executeEpoch, agent]);

  // Auto-run epochs
  useEffect(() => {
    if (!training.isRunning) return;

    const timeout = setTimeout(() => {
      setTraining((prev) => {
        // Stop if we've reached target (0 = infinite)
        if (prev.targetEpochs > 0 && prev.epoch >= prev.targetEpochs) {
          return { ...prev, isRunning: false };
        }
        return executeEpoch(prev);
      });

      setStats(agent.getStats());
    }, 10);

    return () => clearTimeout(timeout);
  }, [training.isRunning, training.epoch, executeEpoch, agent]);

  // Render scenarios
  useEffect(() => {
    training.scenarios.forEach((scenario, i) => {
      const canvas = scenarioRefs.current[i];
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Use the scenario's stored 1-second prediction
      renderScenarioWorld(ctx, scenario, scenario.lastPrediction);
    });
  }, [training.scenarios]);

  // Render convergence graph
  useEffect(() => {
    const canvas = graphRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    renderConvergenceGraph(ctx, training.convergenceHistory, 400, 200);
  }, [training.convergenceHistory]);

  const agencyPct = (stats.totalAgency * 100).toFixed(0);
  const lastError =
    training.convergenceHistory[training.convergenceHistory.length - 1];

  return (
    <div className="training-view">
      <div className="training-header">
        <h2>Training Regime</h2>
        <div className="training-stats">
          <span className="stat-pill">
            Epoch: <strong>{training.epoch}</strong>
          </span>
          <span className="stat-pill">
            Checkpoints/epoch: <strong>{training.horizonSeconds * 4}</strong>
          </span>
          <span className="stat-pill">
            Agency: <strong className="cyan">{agencyPct}%</strong>
          </span>
          <span className="stat-pill">
            Modes: <strong className="lime">{stats.components}</strong>
          </span>
          {lastError && (
            <span className="stat-pill">
              1s Prediction Error:{" "}
              <strong
                className={
                  lastError.avgError < 20
                    ? "lime"
                    : lastError.avgError < 50
                    ? "amber"
                    : "magenta"
                }
              >
                {lastError.avgError.toFixed(1)}px
              </strong>
            </span>
          )}
        </div>
      </div>

      <div className="scenarios-grid">
        {training.scenarios.map((scenario, i) => (
          <div key={scenario.id} className="scenario-card">
            <div className="scenario-header" style={{ color: scenario.color }}>
              {scenario.name}
            </div>
            <canvas
              ref={(el) => {
                scenarioRefs.current[i] = el;
              }}
              width={200}
              height={150}
            />
            {scenario.errors.length > 0 && (
              <div className="scenario-error">
                Last: {scenario.errors[scenario.errors.length - 1]?.toFixed(1)}
                px
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="convergence-section">
        <h3>Convergence</h3>
        <canvas ref={graphRef} width={400} height={200} />
      </div>

      <div className="training-controls">
        <div className="epoch-target">
          <label>Epochs:</label>
          <input
            type="number"
            value={training.targetEpochs}
            onChange={(e) =>
              setTraining((t) => ({
                ...t,
                targetEpochs: Math.max(0, parseInt(e.target.value) || 0),
              }))
            }
            min={0}
            step={100}
          />
          <span className="hint">(0 = ∞)</span>
        </div>
        <div className="epoch-target">
          <label>Horizon:</label>
          <input
            type="number"
            value={training.horizonSeconds}
            onChange={(e) =>
              setTraining((t) => ({
                ...t,
                horizonSeconds: Math.max(1, parseInt(e.target.value) || 10),
              }))
            }
            min={1}
            max={60}
          />
          <span className="hint">sec</span>
        </div>
        <button className="action-btn train" onClick={runEpoch}>
          +1 Epoch
        </button>
        <button
          className={`action-btn ${training.isRunning ? "stop" : "auto"}`}
          onClick={() =>
            setTraining((t) => ({ ...t, isRunning: !t.isRunning }))
          }
        >
          {training.isRunning
            ? "◼ Stop"
            : `▶ Run ${
                training.targetEpochs > 0
                  ? training.targetEpochs - training.epoch
                  : "∞"
              }`}
        </button>
        <button
          className="action-btn reset"
          onClick={() =>
            setTraining({
              scenarios: [
                createScenario("up", "↑ Up", "up", "#68d391"),
                createScenario("down", "↓ Down", "down", "#fc8181"),
                createScenario("left", "← Left", "left", "#4fd1c5"),
                createScenario("right", "→ Right", "right", "#f6ad55"),
              ],
              epoch: 0,
              isRunning: false,
              targetEpochs: training.targetEpochs,
              horizonSeconds: training.horizonSeconds,
              convergenceHistory: [],
            })
          }
        >
          Reset
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// App Component
// ============================================================================

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  // View mode: playground or training
  const [viewMode, setViewMode] = useState<"playground" | "training">(
    "playground"
  );

  // Autonomous mode with gravity
  const [autonomous, setAutonomous] = useState(true);

  const [world, setWorld] = useState<World>(() => {
    const w = createWorld(600, 450, { gravity: true });
    // Start ball in upper area with some horizontal velocity
    const ball = addBall(w, 150, 100, 1.2, "#f6ad55");
    ball.vel = { x: 3, y: 0 };
    return w;
  });

  // Shared agent between playground and training
  const [agent] = useState(() => new PhysicsAgent());
  const [selectedBall, setSelectedBall] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(true); // Auto-start in autonomous
  const [stats, setStats] = useState(() => agent.getStats());

  // Trajectory prediction: predict where ball will be in N frames
  const PREDICTION_HORIZON = 60; // Predict 1 second ahead (60 frames)
  const [trajectoryPrediction, setTrajectoryPrediction] = useState<{
    predictedPos: { x: number; y: number };
    startPos: { x: number; y: number };
    startFrame: number;
  } | null>(null);

  const [trajectoryResult, setTrajectoryResult] = useState<{
    predictedPos: { x: number; y: number };
    actualPos: { x: number; y: number };
    error: number;
  } | null>(null);

  // Select first ball by default
  useEffect(() => {
    if (!selectedBall && world.balls.length > 0) {
      setSelectedBall(world.balls[0]!.id);
    }
  }, [world.balls, selectedBall]);

  // Physics loop
  useEffect(() => {
    if (!isRunning) return;

    let prevWorld = cloneWorld(world);
    let frameCount = 0;
    // Track checkpoint states for 60-frame cumulative observation
    const checkpointStatesRef: Map<
      string,
      { state: PhysicsState; frame: number }
    > = new Map();

    const tick = () => {
      frameCount++;

      setWorld((w) => {
        const next = cloneWorld(w);
        stepWorld(next);

        for (const ball of next.balls) {
          const prevBall = prevWorld.balls.find((b) => b.id === ball.id);
          if (prevBall) {
            const currentState = getPhysicsState(next, ball);

            if (autonomous) {
              // At checkpoint starts, record state for later observation
              if (frameCount % PREDICTION_HORIZON === 1) {
                const beforeState = getPhysicsState(prevWorld, prevBall);
                checkpointStatesRef.set(ball.id, {
                  state: beforeState,
                  frame: frameCount,
                });
              }

              // At checkpoint ends, observe the 60-frame cumulative delta
              if (frameCount % PREDICTION_HORIZON === 0) {
                const checkpoint = checkpointStatesRef.get(ball.id);
                if (checkpoint) {
                  // Observe 60-frame cumulative transition
                  agent.observe(checkpoint.state, currentState, "physics");
                  checkpointStatesRef.delete(ball.id);
                }

                // Also make prediction and evaluate for selected ball
                if (ball.id === selectedBall) {
                  setTrajectoryPrediction((prev) => {
                    if (prev) {
                      const error = Math.sqrt(
                        (ball.pos.x - prev.predictedPos.x) ** 2 +
                          (ball.pos.y - prev.predictedPos.y) ** 2
                      );
                      setTrajectoryResult({
                        predictedPos: prev.predictedPos,
                        actualPos: { x: ball.pos.x, y: ball.pos.y },
                        error,
                      });
                    }

                    // Make new prediction for next 60 frames
                    agent.predictBefore(currentState, "physics");
                    const pred = agent.getLastPrediction();
                    if (pred) {
                      return {
                        predictedPos: { x: pred.pos.x, y: pred.pos.y },
                        startPos: { x: ball.pos.x, y: ball.pos.y },
                        startFrame: frameCount,
                      };
                    }
                    return null;
                  });
                }
              }
            } else {
              // Manual mode: infer action from velocity change and observe per-frame
              const beforeState = getPhysicsState(prevWorld, prevBall);
              const afterState = currentState;
              const action = forceToAction({
                x: ball.vel.x - prevBall.vel.x,
                y: ball.vel.y - prevBall.vel.y,
              });
              if (action !== "none") {
                agent.observe(beforeState, afterState, action);
              }
            }
          }
        }

        prevWorld = cloneWorld(next);
        return next;
      });

      animRef.current = requestAnimationFrame(tick);
    };

    animRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(animRef.current);
  }, [isRunning, agent, autonomous, selectedBall]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    renderWorld(ctx, {
      world,
      selectedBall,
      autonomous,
      trajectoryPrediction,
    });
  }, [world, selectedBall, autonomous, trajectoryPrediction]);

  // Update stats periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setStats(agent.getStats());
    }, 500);
    return () => clearInterval(interval);
  }, [agent]);

  // Apply force to selected ball
  const pushBall = useCallback(
    (action: string) => {
      const ball = world.balls.find((b) => b.id === selectedBall);
      if (!ball) return;

      const beforeState = getPhysicsState(world, ball);

      // Record prediction BEFORE the step
      agent.predictBefore(beforeState, action);

      const force = actionToForce(action, 8);

      setWorld((w) => {
        const next = cloneWorld(w);
        const b = next.balls.find((x) => x.id === selectedBall);
        if (b) {
          applyForce(b, force);
          stepWorld(next);
        }
        return next;
      });

      // Observe after force and compare to prediction
      setTimeout(() => {
        setWorld((w) => {
          const b = w.balls.find((x) => x.id === selectedBall);
          if (b) {
            const afterState = getPhysicsState(w, b);
            agent.observe(beforeState, afterState, action);
            setStats(agent.getStats());
          }
          return w;
        });
      }, 0);
    },
    [world, selectedBall, agent]
  );

  // Train with exploration that covers diverse contexts
  const train = useCallback(
    (steps: number) => {
      let w = cloneWorld(world);
      const actions = ["up", "down", "left", "right"];

      // Phase 1: Explore different regions by teleporting ball
      const regions = [
        { x: 125, y: 125 }, // Ice zone
        { x: w.width - 125, y: w.height - 125 }, // Mud zone
        { x: w.width / 2, y: w.height / 2 }, // Center
        { x: 30, y: w.height / 2 }, // Near left wall
        { x: w.width - 30, y: w.height / 2 }, // Near right wall
        { x: w.width / 2, y: 30 }, // Near top wall
        { x: w.width / 2, y: w.height - 30 }, // Near bottom wall
      ];

      for (let i = 0; i < steps; i++) {
        for (const ball of w.balls) {
          // Every 7 steps, teleport to a new region for diversity
          if (i % 7 === 0) {
            const region = regions[i % regions.length]!;
            ball.pos = { x: region.x, y: region.y };
            ball.vel = { x: 0, y: 0 };
          }

          const action = actions[Math.floor(Math.random() * actions.length)]!;
          const beforeState = getPhysicsState(w, ball);

          // Record prediction BEFORE step (for accuracy tracking)
          agent.predictBefore(beforeState, action);

          const force = actionToForce(action, 5 + Math.random() * 10);
          applyForce(ball, force);
          stepWorld(w);

          const afterState = getPhysicsState(w, ball);
          agent.observe(beforeState, afterState, action);
        }
      }

      setWorld(w);
      setStats(agent.getStats());
    },
    [world, agent]
  );

  // Add a new ball
  const addNewBall = useCallback(() => {
    setWorld((w) => {
      const next = cloneWorld(w);
      const x = 100 + Math.random() * (w.width - 200);
      const y = 100 + Math.random() * (w.height - 200);
      const mass = 0.8 + Math.random() * 1.5;
      const colors = ["#f6ad55", "#fc8181", "#68d391", "#4fd1c5", "#b794f4"];
      const color = colors[Math.floor(Math.random() * colors.length)]!;
      const ball = addBall(next, x, y, mass, color);
      setSelectedBall(ball.id);
      return next;
    });
  }, []);

  // Reset
  const reset = useCallback(() => {
    const w = createWorld(600, 450, { gravity: autonomous });
    const ball = addBall(
      w,
      autonomous ? 150 : 300,
      autonomous ? 100 : 225,
      1.2,
      "#f6ad55"
    );
    if (autonomous) {
      ball.vel = { x: 3, y: 0 }; // Initial horizontal velocity
    }
    setWorld(w);
    setSelectedBall(w.balls[0]?.id ?? null);
    setTrajectoryPrediction(null);
    setTrajectoryResult(null);
  }, [autonomous]);

  // Canvas click to select ball
  const handleCanvasClick = useCallback(
    (e: MouseEvent<HTMLCanvasElement>) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Find clicked ball
      for (const ball of world.balls) {
        const dx = ball.pos.x - x;
        const dy = ball.pos.y - y;
        if (Math.sqrt(dx * dx + dy * dy) < ball.radius + 5) {
          setSelectedBall(ball.id);
          return;
        }
      }
    },
    [world.balls]
  );

  // Keyboard controls
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      switch (e.key) {
        case "ArrowUp":
        case "w":
          pushBall("up");
          break;
        case "ArrowDown":
        case "s":
          pushBall("down");
          break;
        case "ArrowLeft":
        case "a":
          pushBall("left");
          break;
        case "ArrowRight":
        case "d":
          pushBall("right");
          break;
        case " ":
          e.preventDefault();
          setIsRunning((r) => !r);
          break;
      }
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [pushBall]);

  const agencyPct = (stats.totalAgency * 100).toFixed(0);

  return (
    <div className="app">
      <header>
        <h1>
          <span className="ψ">Ψ</span> Physics Playground
        </h1>
        <div className="view-toggle">
          <button
            className={viewMode === "playground" ? "active" : ""}
            onClick={() => setViewMode("playground")}
          >
            Playground
          </button>
          <button
            className={viewMode === "training" ? "active" : ""}
            onClick={() => setViewMode("training")}
          >
            Training
          </button>
        </div>
      </header>

      {viewMode === "training" ? (
        <TrainingView agent={agent} />
      ) : (
        <>
          <span className="subtitle">
            {autonomous
              ? "learning gravity: orange = actual, cyan = predicted"
              : "manual mode: push the ball to teach dynamics"}
          </span>

          <div className="canvas-wrap">
            <canvas
              ref={canvasRef}
              width={600}
              height={450}
              onClick={handleCanvasClick}
              style={{ cursor: "pointer" }}
            />
          </div>

          <aside className="sidebar">
            <div className="panel">
              <h3>Universal Port</h3>
              <div className="stat-grid">
                <div className="stat">
                  <div className="label">Agency</div>
                  <div
                    className={`value ${
                      Number(agencyPct) > 50 ? "cyan" : "magenta"
                    }`}
                  >
                    {agencyPct}%
                  </div>
                </div>
                <div className="stat">
                  <div className="label">Observations</div>
                  <div className="value amber">{stats.observations}</div>
                </div>
                <div className="stat">
                  <div className="label">Modes</div>
                  <div className="value lime">{stats.components}</div>
                </div>
                <div className="stat">
                  <div className="label">Avg σ²</div>
                  <div className="value">
                    {stats.modes.length > 0
                      ? (
                          stats.modes.reduce((s, m) => s + m.avgVar, 0) /
                          stats.modes.length
                        ).toFixed(1)
                      : "—"}
                  </div>
                </div>
              </div>
              {/* Mode details */}
              {stats.modes.length > 0 && (
                <div className="mode-list">
                  {stats.modes.map((m) => (
                    <div key={m.index} className="mode-item">
                      <span className="mode-idx">#{m.index}</span>
                      <span
                        className="mode-agency"
                        style={{
                          color:
                            m.agency > 0.5
                              ? "var(--accent-cyan)"
                              : "var(--accent-magenta)",
                        }}
                      >
                        {(m.agency * 100).toFixed(0)}%
                      </span>
                      <span className="mode-seen">n={m.seen}</span>
                      <span className="mode-var">σ²={m.avgVar.toFixed(1)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Trajectory Prediction: where will ball be in 1 second? */}
            {autonomous && trajectoryResult && (
              <div className="panel debug-panel">
                <h3>1-Second Prediction</h3>
                <div className="debug-row">
                  <span className="debug-label">Predicted:</span>
                  <span
                    className="debug-value"
                    style={{ color: "var(--accent-cyan)" }}
                  >
                    ({trajectoryResult.predictedPos.x.toFixed(0)},{" "}
                    {trajectoryResult.predictedPos.y.toFixed(0)})
                  </span>
                </div>
                <div className="debug-row">
                  <span className="debug-label">Actual:</span>
                  <span
                    className="debug-value"
                    style={{ color: "var(--accent-amber)" }}
                  >
                    ({trajectoryResult.actualPos.x.toFixed(0)},{" "}
                    {trajectoryResult.actualPos.y.toFixed(0)})
                  </span>
                </div>
                <div className="debug-row">
                  <span className="debug-label">Error:</span>
                  <span
                    className="debug-value"
                    style={{
                      color:
                        trajectoryResult.error < 20
                          ? "var(--accent-lime)"
                          : trajectoryResult.error < 50
                          ? "var(--accent-amber)"
                          : "var(--accent-magenta)",
                    }}
                  >
                    {trajectoryResult.error.toFixed(0)}px
                  </span>
                </div>
                <div
                  className="debug-row"
                  style={{
                    marginTop: 8,
                    fontSize: "10px",
                    color: "var(--text-dim)",
                  }}
                >
                  {trajectoryResult.error < 20
                    ? "✓ Agent understands trajectory"
                    : trajectoryResult.error < 50
                    ? "~ Learning dynamics..."
                    : "✗ Trajectory prediction wrong"}
                </div>
              </div>
            )}

            <div className="panel">
              <h3>Controls</h3>
              <div className="control-grid">
                <button className="dir-btn up" onClick={() => pushBall("up")}>
                  ↑
                </button>
                <button
                  className="dir-btn left"
                  onClick={() => pushBall("left")}
                >
                  ←
                </button>
                <button
                  className="dir-btn center"
                  onClick={() => setIsRunning((r) => !r)}
                >
                  {isRunning ? "◼" : "▶"}
                </button>
                <button
                  className="dir-btn right"
                  onClick={() => pushBall("right")}
                >
                  →
                </button>
                <button
                  className="dir-btn down"
                  onClick={() => pushBall("down")}
                >
                  ↓
                </button>
              </div>
            </div>

            <div className="panel">
              <h3>Actions</h3>
              <div className="actions">
                <button className="action-btn train" onClick={() => train(50)}>
                  Train (+50 steps)
                </button>
                <button className="action-btn add" onClick={addNewBall}>
                  Add Ball
                </button>
                <button className="action-btn reset" onClick={reset}>
                  Reset World
                </button>
              </div>
            </div>

            <div className="panel">
              <h3>Mode</h3>
              <div className="toggle-row">
                <label>Autonomous (gravity)</label>
                <div
                  className={`toggle ${autonomous ? "on" : ""}`}
                  onClick={() => {
                    setAutonomous((a) => !a);
                    // Reset when changing modes
                    setTimeout(() => reset(), 0);
                  }}
                />
              </div>
            </div>
          </aside>

          <footer>
            <div className="legend-item">
              <span className="legend-dot ball" />
              Ball
            </div>
            <div className="legend-item">
              <span className="legend-dot prediction" />
              1s Prediction
            </div>
            <div className="legend-item">
              <span className="legend-dot ice" />
              Ice
            </div>
            <div className="legend-item">
              <span className="legend-dot mud" />
              Mud
            </div>
          </footer>
        </>
      )}
    </div>
  );
}

export default App;

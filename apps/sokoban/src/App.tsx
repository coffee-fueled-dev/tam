import { useState, useCallback, useEffect, useRef } from "react";
import "./index.css";
import {
  type GameState,
  type Direction,
  parseLevel,
  stateToGrid,
  move,
  isWon,
  cloneState,
  LEVELS,
  createSokobanAgent,
  trainAgent,
  solve,
  getActionValues,
  gameStateToSituation,
  type SokobanSituation,
} from "./game";
import { type TAMAgent } from "tam";

const CELL_SIZE = 40;

const CELL_COLORS: Record<string, { bg: string; fg: string; char?: string }> = {
  "#": { bg: "#2d2d2d", fg: "#2d2d2d" },
  " ": { bg: "#1a1a2e", fg: "#1a1a2e" },
  ".": { bg: "#1a1a2e", fg: "#4fd1c5", char: "◎" },
  "@": { bg: "#1a1a2e", fg: "#f6ad55", char: "◉" },
  "+": { bg: "#1a1a2e", fg: "#68d391", char: "◉" },
  $: { bg: "#1a1a2e", fg: "#fc8181", char: "■" },
  "*": { bg: "#1a1a2e", fg: "#68d391", char: "★" },
};

function Cell({ char }: { char: string }) {
  const style = CELL_COLORS[char] || CELL_COLORS[" "]!;
  return (
    <div
      style={{
        width: CELL_SIZE,
        height: CELL_SIZE,
        background: style.bg,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: CELL_SIZE * 0.7,
        color: style.fg,
        fontWeight: "bold",
        borderRadius: char === "#" ? 4 : 0,
        boxShadow: char === "#" ? "inset 0 0 8px rgba(0,0,0,0.5)" : "none",
      }}
    >
      {style.char || ""}
    </div>
  );
}

function GameBoard({ state }: { state: GameState }) {
  const grid = stateToGrid(state);
  return (
    <div
      style={{
        display: "inline-grid",
        gridTemplateColumns: `repeat(${state.width}, ${CELL_SIZE}px)`,
        gap: 1,
        background: "#0d0d14",
        padding: 8,
        borderRadius: 8,
        boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
      }}
    >
      {grid.map((row, y) =>
        row.map((cell, x) => <Cell key={`${x}-${y}`} char={cell} />)
      )}
    </div>
  );
}

function ValueBar({
  value,
  label,
  agency,
}: {
  value: number;
  label: string;
  agency: number;
}) {
  // Normalize value for display (assume -10 to 10 range)
  const normalizedValue = Math.max(0, Math.min(1, (value + 5) / 10));
  const agencyColor =
    agency > 0.7 ? "#68d391" : agency > 0.4 ? "#f6ad55" : "#fc8181";

  return (
    <div style={{ marginBottom: 8 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 12,
          color: "#888",
          marginBottom: 2,
        }}
      >
        <span style={{ textTransform: "uppercase", fontWeight: "bold" }}>
          {label}
        </span>
        <span>
          V={value.toFixed(2)} | A={(agency * 100).toFixed(0)}%
        </span>
      </div>
      <div
        style={{
          height: 6,
          background: "#2d2d2d",
          borderRadius: 3,
          overflow: "hidden",
          display: "flex",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${normalizedValue * 70}%`,
            background: `linear-gradient(90deg, #4fd1c5, #68d391)`,
            transition: "width 0.3s",
          }}
        />
        <div
          style={{
            height: "100%",
            width: `${agency * 30}%`,
            background: agencyColor,
            transition: "width 0.3s",
          }}
        />
      </div>
    </div>
  );
}

interface AgentState {
  player: { x: number; y: number };
  boxPositions: string[];
  goalPositions: string[];
  boxesOnGoals: number;
}

interface AgentContext {
  wallAhead: boolean;
  boxAhead: boolean;
  wallBehindBox: boolean;
  boxBehindBox: boolean;
}

function AgentStats({
  agent,
  state,
  trainingStats,
}: {
  agent: TAMAgent<AgentState, AgentContext>;
  state: GameState;
  trainingStats: { episodes: number; wins: number; avgReward: number } | null;
}) {
  const snapshot = agent.snapshot();
  const actionValues = getActionValues(agent, state);
  const stateValue = agent.getValue(gameStateToSituation(state, "up"));

  return (
    <div
      style={{
        background: "#161621",
        padding: 16,
        borderRadius: 8,
        minWidth: 220,
      }}
    >
      <h3 style={{ margin: "0 0 12px", color: "#4fd1c5", fontSize: 14 }}>
        TAM Agent
      </h3>

      {/* Training stats */}
      {trainingStats && (
        <div
          style={{
            background: "#1a1a2e",
            padding: 8,
            borderRadius: 4,
            marginBottom: 12,
            fontSize: 11,
          }}
        >
          <div style={{ color: "#68d391", fontWeight: "bold" }}>
            Last Training:
          </div>
          <div style={{ color: "#888" }}>
            Episodes: {trainingStats.episodes} | Wins: {trainingStats.wins}
          </div>
          <div style={{ color: "#888" }}>
            Avg Reward: {trainingStats.avgReward.toFixed(3)}
          </div>
        </div>
      )}

      {/* Current state value */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>
          State Value:{" "}
          <span style={{ color: "#4fd1c5" }}>{stateValue.toFixed(3)}</span>
        </div>
        <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>
          States Explored:{" "}
          <span style={{ color: "#68d391" }}>
            {snapshot.uniqueStatesVisited}
          </span>
        </div>
        <div style={{ fontSize: 11, color: "#888" }}>
          Total Agency:{" "}
          <span style={{ color: "#f6ad55" }}>
            {(snapshot.totalAgency * 100).toFixed(0)}%
          </span>
        </div>
        <div style={{ fontSize: 11, color: "#888" }}>
          Steps: {snapshot.steps} | Avg Reward: {snapshot.avgReward.toFixed(3)}
        </div>
      </div>

      {/* Action values */}
      <div style={{ marginBottom: 12 }}>
        <div
          style={{
            fontSize: 12,
            fontWeight: "bold",
            color: "#f6ad55",
            marginBottom: 8,
          }}
        >
          Action Values
        </div>
        {actionValues.map(({ action, value, agency }) => (
          <ValueBar key={action} label={action} value={value} agency={agency} />
        ))}
      </div>

      {/* Port info */}
      <div style={{ fontSize: 10, color: "#666" }}>
        <div>
          Value params: w[0]={snapshot.valueParams.w[0]?.toFixed(2) ?? "?"}, b=
          {snapshot.valueParams.b.toFixed(2)}
        </div>
      </div>
    </div>
  );
}

function Controls({
  onMove,
  onReset,
  onTrain,
  onSolve,
  isTraining,
  isSolving,
  won,
}: {
  onMove: (dir: Direction) => void;
  onReset: () => void;
  onTrain: () => void;
  onSolve: () => void;
  isTraining: boolean;
  isSolving: boolean;
  won: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 48px)",
          gridTemplateRows: "repeat(2, 48px)",
          gap: 4,
          justifyContent: "center",
        }}
      >
        <div />
        <button className="dir-btn" onClick={() => onMove("up")}>
          ↑
        </button>
        <div />
        <button className="dir-btn" onClick={() => onMove("left")}>
          ←
        </button>
        <button className="dir-btn" onClick={() => onMove("down")}>
          ↓
        </button>
        <button className="dir-btn" onClick={() => onMove("right")}>
          →
        </button>
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button className="action-btn" onClick={onReset}>
          Reset
        </button>
        <button
          className="action-btn train"
          onClick={onTrain}
          disabled={isTraining}
        >
          {isTraining ? "Training..." : "Train (200 eps)"}
        </button>
        <button
          className="action-btn solve"
          onClick={onSolve}
          disabled={isSolving}
        >
          {isSolving ? "Solving..." : "TAM Solve"}
        </button>
      </div>

      {won && (
        <div
          style={{
            padding: 12,
            background: "#22543d",
            borderRadius: 8,
            color: "#68d391",
            textAlign: "center",
            fontWeight: "bold",
          }}
        >
          🎉 Level Complete!
        </div>
      )}
    </div>
  );
}

function LevelSelect({
  current,
  onChange,
}: {
  current: string;
  onChange: (level: string) => void;
}) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ color: "#888", fontSize: 12, marginRight: 8 }}>
        Level:
      </label>
      <select
        value={current}
        onChange={(e) => onChange(e.target.value)}
        style={{
          background: "#2d2d2d",
          border: "none",
          color: "#fff",
          padding: "6px 12px",
          borderRadius: 4,
          fontSize: 14,
        }}
      >
        {Object.keys(LEVELS).map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
    </div>
  );
}

export function App() {
  const [levelName, setLevelName] = useState<keyof typeof LEVELS>("trivial");
  const [state, setState] = useState<GameState>(() =>
    parseLevel(LEVELS[levelName])
  );
  const [agent] = useState(() => createSokobanAgent());
  const [isTraining, setIsTraining] = useState(false);
  const [isSolving, setIsSolving] = useState(false);
  const [moveHistory, setMoveHistory] = useState<Direction[]>([]);
  const [trainingStats, setTrainingStats] = useState<{
    episodes: number;
    wins: number;
    avgReward: number;
  } | null>(null);
  const [, forceUpdate] = useState(0);
  const solveRef = useRef<{ cancelled: boolean }>({ cancelled: false });

  // Change level
  const handleLevelChange = useCallback((name: string) => {
    setLevelName(name as keyof typeof LEVELS);
    setState(parseLevel(LEVELS[name as keyof typeof LEVELS]));
    setMoveHistory([]);
    solveRef.current.cancelled = true;
  }, []);

  // Reset level
  const handleReset = useCallback(() => {
    setState(parseLevel(LEVELS[levelName]));
    setMoveHistory([]);
    solveRef.current.cancelled = true;
  }, [levelName]);

  // Manual move - also teaches the agent
  const handleMove = useCallback(
    (dir: Direction) => {
      setState((prev) => {
        const result = move(prev, dir);

        // Create transition for agent learning
        const beforeSit = gameStateToSituation(prev, dir);
        const afterSit = gameStateToSituation(result.newState, dir);

        // Compute reward
        const beforeBoxesOnGoals = Array.from(prev.boxes).filter((b) =>
          prev.goals.has(b)
        ).length;
        const afterBoxesOnGoals = Array.from(result.newState.boxes).filter(
          (b) => result.newState.goals.has(b)
        ).length;

        let reward = afterBoxesOnGoals - beforeBoxesOnGoals;
        if (isWon(result.newState)) reward = 10;
        else if (result.success) reward = reward > 0 ? 1 : -0.01;
        else reward = -0.05;

        // Learn from this transition
        agent.observe(
          { action: dir, before: beforeSit, after: afterSit },
          reward
        );

        if (result.success) {
          setMoveHistory((h) => [...h, dir]);
        }

        // Trigger re-render to update stats
        forceUpdate((n) => n + 1);

        return result.newState;
      });
    },
    [agent]
  );

  // Train agent
  const handleTrain = useCallback(() => {
    setIsTraining(true);
    setTimeout(() => {
      const stats = trainAgent(
        agent,
        parseLevel(LEVELS[levelName]),
        200, // episodes
        150, // max steps
        0.4 // epsilon
      );
      setTrainingStats(stats);
      setIsTraining(false);
      forceUpdate((n) => n + 1);
    }, 50);
  }, [agent, levelName]);

  // Auto-solve using TAM agent
  const handleSolve = useCallback(() => {
    setIsSolving(true);
    solveRef.current.cancelled = false;

    const initialState = parseLevel(LEVELS[levelName]);
    setState(initialState);
    setMoveHistory([]);

    setTimeout(() => {
      const result = solve(agent, initialState, 200);

      if (!result.solved && result.moves.length === 0) {
        setIsSolving(false);
        return;
      }

      // Animate solution
      let i = 0;
      let current = cloneState(initialState);

      const step = () => {
        if (i >= result.moves.length || solveRef.current.cancelled) {
          setIsSolving(false);
          return;
        }

        const dir = result.moves[i]!;
        const moveResult = move(current, dir);

        current = moveResult.newState;
        setState(cloneState(current));
        setMoveHistory((h) => [...h, dir]);

        i++;
        setTimeout(step, 200);
      };

      step();
    }, 50);
  }, [agent, levelName]);

  // Keyboard controls
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const keyMap: Record<string, Direction> = {
        ArrowUp: "up",
        ArrowDown: "down",
        ArrowLeft: "left",
        ArrowRight: "right",
        w: "up",
        s: "down",
        a: "left",
        d: "right",
      };
      const dir = keyMap[e.key];
      if (dir) {
        e.preventDefault();
        handleMove(dir);
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleMove]);

  const won = isWon(state);

  return (
    <div className="app">
      <header>
        <h1>
          <span className="highlight">TAM</span> Sokoban
        </h1>
        <p className="subtitle">
          Commitment-based learning through affordance cones
        </p>
      </header>

      <main>
        <div className="game-section">
          <LevelSelect current={levelName} onChange={handleLevelChange} />
          <GameBoard state={state} />
          <div className="move-count">Moves: {moveHistory.length}</div>
        </div>

        <aside>
          <Controls
            onMove={handleMove}
            onReset={handleReset}
            onTrain={handleTrain}
            onSolve={handleSolve}
            isTraining={isTraining}
            isSolving={isSolving}
            won={won}
          />
          <AgentStats
            agent={agent}
            state={state}
            trainingStats={trainingStats}
          />
        </aside>
      </main>

      <footer>
        <div className="legend">
          <span>
            <span className="cell player">◉</span> Player
          </span>
          <span>
            <span className="cell box">■</span> Box
          </span>
          <span>
            <span className="cell goal">◎</span> Goal
          </span>
          <span>
            <span className="cell done">★</span> Box on Goal
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;

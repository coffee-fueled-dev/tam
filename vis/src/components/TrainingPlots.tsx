import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { TrainingMetrics, Frame } from "@/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

interface TrainingPlotsProps {
  trainingMetrics?: TrainingMetrics;
  frames?: Frame[];
  reachedGoalsByFrame?: Record<number, number[][]>;
}

export function TrainingPlots({ trainingMetrics, frames, reachedGoalsByFrame }: TrainingPlotsProps) {
  // Prepare data for moves per goal chart (goal number vs moves taken)
  const movesPerGoalData = useMemo(() => {
    if (!trainingMetrics?.goal_stats || trainingMetrics.goal_stats.length === 0) {
      return [];
    }

    const goalStats = trainingMetrics.goal_stats;
    return goalStats.map((g, index) => ({
      goalNumber: index + 1,
      movesPerGoal: g.moves_taken,
    }));
  }, [trainingMetrics]);

  // Prepare data for agency over moves chart (from frames)
  const agencyData = useMemo(() => {
    if (!frames || frames.length === 0) {
      return [];
    }

    return frames.map((frame) => {
      if (frame.agency) {
        return {
          move: frame.step,
          mean: frame.agency.mean,
          upper: frame.agency.mean + frame.agency.std,
          lower: Math.max(0, frame.agency.mean - frame.agency.std),
        };
      } else {
        return {
          move: frame.step,
          mean: null,
          upper: null,
          lower: null,
        };
      }
    });
  }, [frames]);

  // Prepare data for training loss chart (per move from frames)
  const lossData = useMemo(() => {
    if (!frames || frames.length === 0) {
      return [];
    }

    return frames
      .map((frame) => {
        if (frame.loss !== null && frame.loss !== undefined) {
          return {
            move: frame.step,
            loss: frame.loss,
          };
        }
        return null;
      })
      .filter((d) => d !== null);
  }, [frames]);

  // Prepare data for cumulative goals over moves
  const cumulativeGoalsData = useMemo(() => {
    if (!frames || !reachedGoalsByFrame) {
      return [];
    }

    return frames.map((frame, index) => {
      const goalsReached = reachedGoalsByFrame[index]?.length || 0;
      return {
        move: frame.step,
        cumulativeGoals: goalsReached,
      };
    });
  }, [frames, reachedGoalsByFrame]);

  // Prepare data for energy over moves
  const energyData = useMemo(() => {
    if (!frames) {
      return [];
    }

    return frames
      .map((frame, index) => {
        if (frame.energy === null || frame.energy === undefined) {
          return null;
        }
        return {
          move: frame.step,
          energy: frame.energy,
          maxEnergy: frame.max_energy || null,
          energyPercent: frame.max_energy 
            ? (frame.energy / frame.max_energy) * 100 
            : null,
        };
      })
      .filter((d) => d !== null);
  }, [frames]);

  const hasData = (trainingMetrics && (
    (trainingMetrics.goal_stats && trainingMetrics.goal_stats.length > 0)
  )) || (frames && frames.length > 0 && (
    frames.some(f => f.agency !== undefined) || 
    frames.some(f => f.loss !== null && f.loss !== undefined)
  ));

  if (!hasData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Training Progress</CardTitle>
          <CardDescription>
            No training metrics data available. Upload goal_stats JSONL file to see plots.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {trainingMetrics && (
            <p className="text-sm text-muted-foreground">
              Debug: goal_stats length: {trainingMetrics.goal_stats?.length || 0}
            </p>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-4">Training Progress</h2>
      </div>

      {/* Moves per Goal Chart */}
      {movesPerGoalData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Moves per Goal (Lower = Better)</CardTitle>
            <CardDescription>
              Number of moves taken to reach each goal - shows efficiency improvement over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={movesPerGoalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="goalNumber" 
                  label={{ value: "Goal Number", position: "insideBottom", offset: -5 }}
                />
                <YAxis 
                  label={{ value: "Moves Taken", angle: -90, position: "insideLeft" }}
                />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="movesPerGoal" 
                  stroke="#2563EB" 
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Moves per Goal"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Agency over Moves Chart */}
      {agencyData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Agency (σ) Over Moves</CardTitle>
            <CardDescription>
              Mean agency with standard deviation error bars (forward-filled between goals)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={agencyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="move" 
                  label={{ value: "Move Number", position: "insideBottom", offset: -5 }}
                />
                <YAxis 
                  label={{ value: "Agency (σ)", angle: -90, position: "insideLeft" }}
                />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="mean" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  dot={false}
                  connectNulls={true}
                  name="Mean Agency"
                />
                <Line 
                  type="monotone" 
                  dataKey="upper" 
                  stroke="#10B981" 
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                  connectNulls={true}
                  name="Mean + Std"
                />
                <Line 
                  type="monotone" 
                  dataKey="lower" 
                  stroke="#10B981" 
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                  connectNulls={true}
                  name="Mean - Std"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Training Loss Chart */}
      {lossData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Training Loss Over Time (Log Scale)</CardTitle>
            <CardDescription>
              Loss values during training
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="move" 
                  label={{ value: "Move Number", position: "insideBottom", offset: -5 }}
                />
                <YAxis 
                  scale="log"
                  domain={['auto', 'auto']}
                  label={{ value: "Loss (Log Scale)", angle: -90, position: "insideLeft" }}
                />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#EC4899" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Training Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Cumulative Goals Chart */}
      {cumulativeGoalsData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Cumulative Goals Over Moves</CardTitle>
            <CardDescription>
              Total number of goals reached over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={cumulativeGoalsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="move" 
                  label={{ value: "Move Number", position: "insideBottom", offset: -5 }}
                />
                <YAxis 
                  label={{ value: "Cumulative Goals", angle: -90, position: "insideLeft" }}
                />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="cumulativeGoals" 
                  stroke="#F59E0B" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Cumulative Goals"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Energy Over Moves Chart */}
      {energyData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Energy Over Moves</CardTitle>
            <CardDescription>
              Energy level throughout training
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={energyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="move" 
                  label={{ value: "Move Number", position: "insideBottom", offset: -5 }}
                />
                <YAxis 
                  label={{ value: "Energy", angle: -90, position: "insideLeft" }}
                />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="energy" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Energy"
                />
                {energyData.some((d: any) => d.maxEnergy !== null) && (
                  <Line 
                    type="monotone" 
                    dataKey="maxEnergy" 
                    stroke="#6B7280" 
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Max Energy"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

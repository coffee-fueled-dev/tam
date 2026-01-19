import { useState, type ChangeEvent } from "react";
import {
  parseVisualizationData,
  readFileAsText,
} from "../lib/jsonlParser";
import type { VisualizationData } from "@/types";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

interface FileUploadProps {
  onDataLoaded: (data: VisualizationData) => void;
}

export function FileUpload({ onDataLoaded }: FileUploadProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      // Find JSONL file (prioritize visualization_data files)
      const jsonlFile = Array.from(files).find(
        (f) => (f.name.includes("visualization_data") || f.name.includes("visualization-data")) && 
               (f.name.endsWith(".jsonl") || f.name.endsWith(".jsonl.gz"))
      ) || Array.from(files).find(
        (f) => f.name.endsWith(".jsonl") || f.name.endsWith(".jsonl.gz")
      );
      if (!jsonlFile) {
        throw new Error("No JSONL file found. Please select a visualization_data_*.jsonl file.");
      }

      // Find metadata file (optional)
      const metadataFile = Array.from(files).find(
        (f) => f.name.includes("visualization_metadata") && f.name.endsWith(".json")
      );

      // Find training metrics file (optional) - try multiple patterns
      const trainingMetricsFile = Array.from(files).find(
        (f) => {
          const name = f.name.toLowerCase();
          return (name.includes("goal_stats") || name.includes("goal-stats")) && 
                 (name.endsWith(".jsonl") || name.endsWith(".json"));
        }
      );

      // Read files
      const jsonlContent = await readFileAsText(jsonlFile);
      const metadataContent = metadataFile
        ? await readFileAsText(metadataFile)
        : null;
      
      let trainingMetricsContent: string | null = null;
      if (trainingMetricsFile) {
        try {
          const goalStatsContent = await readFileAsText(trainingMetricsFile);
          // Parse goal stats JSONL and map to expected format
          const goalStats = goalStatsContent
            .split("\n")
            .map((line) => line.trim())
            .filter((line) => line.length > 0)
            .map((line, index) => {
              try {
                const rawStat = JSON.parse(line);
                // Map goal_stats format to match GoalStat interface
                return {
                  moves_taken: rawStat.moves_taken || 0,
                  agency: {
                    mean: rawStat.agency?.mean || 0,
                    std: rawStat.agency?.std || 0,
                  },
                };
              } catch (e) {
                console.warn("Failed to parse goal_stats line:", line.substring(0, 100));
                return null;
              }
            })
            .filter((stat) => stat !== null);
          
          console.log(`Loaded ${goalStats.length} goal stats entries`);
          
          // Loss is now per-frame in visualization_data, not in metadata or goal_stats
          trainingMetricsContent = JSON.stringify({
            goal_stats: goalStats,
            loss_history: [], // Loss is per-frame, not aggregated here
          });
          
          console.log("Training metrics prepared:", {
            goalStatsCount: goalStats.length,
            sampleGoalStat: goalStats[0],
          });
        } catch (error) {
          console.error("Error loading goal_stats file:", error);
          setError(`Failed to parse goal_stats file: ${error instanceof Error ? error.message : String(error)}`);
        }
      } else {
        const availableFiles = Array.from(files).map(f => f.name);
        console.log("No goal_stats file found. Available files:", availableFiles);
        console.log("To see training plots, upload a goal_stats_*.jsonl file along with the visualization files.");
      }

      // Parse data
      const data = await parseVisualizationData(
        jsonlContent,
        metadataContent,
        trainingMetricsContent
      );

      onDataLoaded(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load files");
      console.error("File upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Upload Visualization Files</CardTitle>
        <CardDescription>
          Select visualization_data_*.jsonl and optionally visualization_metadata_*.json
          <br />
          <span className="text-xs text-muted-foreground">
            Tip: Also upload goal_stats_*.jsonl to see training plots
          </span>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col gap-4">
          <input
            id="file-input"
            type="file"
            multiple
            accept=".jsonl,.json,.gz"
            onChange={handleFileChange}
            disabled={loading}
            className="hidden"
          />
          <Button
            onClick={() => document.getElementById("file-input")?.click()}
            disabled={loading}
            className="w-full"
          >
            {loading ? "Loading..." : "Choose Files"}
          </Button>
        </div>
        {error && (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3 text-sm text-destructive">
            {error}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

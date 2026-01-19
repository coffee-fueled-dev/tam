import { useState, useEffect, useRef } from "react";
import { FileUpload } from "./components/FileUpload";
import { Scene3D } from "./components/Scene3D";
import { Controls } from "./components/Controls";
import { TrainingPlots } from "./components/TrainingPlots";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import { Slider } from "./components/ui/slider";
import type { VisualizationData, Frame } from "./types";
import "./index.css";
import { Progress } from "./components/ui/progress";

const MAX_HISTORY = 5;

export function App() {
  const [data, setData] = useState<VisualizationData | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const lastFrameTimeRef = useRef<number>(0);

  const handleDataLoaded = (loadedData: VisualizationData) => {
    setData(loadedData);
    setCurrentFrame(0);
    setIsPlaying(false);
  };

  const handlePlayPause = () => {
    setIsPlaying((prev) => !prev);
    if (!isPlaying) {
      lastFrameTimeRef.current = performance.now();
    }
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
  };

  const handleFrameChange = (frame: number) => {
    setCurrentFrame(frame);
    setIsPlaying(false);
  };

  // Debug logging
  useEffect(() => {
    if (data) {
      console.log("Visualization data loaded:", {
        frameCount: data.frames.length,
        metadata: data.metadata,
        hasTrainingMetrics: !!data.trainingMetrics,
        reachedGoalsByFrameKeys: Object.keys(data.reachedGoalsByFrame).length,
        firstFrame: data.frames[0],
      });
    }
  }, [data]);

  // Animation loop for playback
  useEffect(() => {
    if (!isPlaying || !data || data.frames.length === 0) return;

    let animationId: number;
    const animate = (timestamp: number) => {
      const deltaTime = timestamp - lastFrameTimeRef.current;
      const frameDelay = 1000 / (30 * playbackSpeed); // 30 FPS base

      if (deltaTime >= frameDelay) {
        setCurrentFrame((prev) => (prev + 1) % data.frames.length);
        lastFrameTimeRef.current = timestamp;
      }

      animationId = requestAnimationFrame(animate);
    };

    lastFrameTimeRef.current = performance.now();
    animationId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationId);
  }, [isPlaying, data, playbackSpeed]);

  if (!data) {
    return (
      <div className="min-h-screen w-full flex items-center justify-center bg-background">
        <div className="max-w-2xl w-full px-4">
          <div className="text-center space-y-6">
            <h1 className="text-4xl font-bold text-foreground">TAM Training Visualization</h1>
            <p className="text-muted-foreground">Upload visualization files to begin</p>
            <FileUpload onDataLoaded={handleDataLoaded} />
          </div>
        </div>
      </div>
    );
  }

  const currentFrameData = data.frames[currentFrame] || null;
  const goalCount = data.reachedGoalsByFrame[currentFrame]?.length || 0;

  return (
    <div className="h-screen w-full flex flex-col bg-background">
      {/* Controls Bar - Always visible */}
      <div className="flex-shrink-0 border-b bg-card">
        <Controls
          currentFrame={currentFrame}
          totalFrames={data.frames.length}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
          currentFrameData={currentFrameData}
          goalCount={goalCount}
          onPlayPause={handlePlayPause}
          onSpeedChange={handleSpeedChange}
          onFrameChange={handleFrameChange}
        />
      </div>

      {/* Main Content Area with Tabs */}
      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="visualization" className="h-full flex flex-col">
          <div className="flex-shrink-0 border-b px-4">
            <TabsList>
              <TabsTrigger value="visualization">3D Visualization</TabsTrigger>
              <TabsTrigger value="plots">Training Plots</TabsTrigger>
            </TabsList>
          </div>
          
          <TabsContent value="visualization" className="flex-1 m-0 overflow-hidden flex flex-col">
            {/* Energy Slider - Full width above 3D scene */}
            <div className="w-full px-4 py-2 border-b bg-card flex-shrink-0">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-muted-foreground whitespace-nowrap">
                  Energy:
                </span>
                <div className="flex-1">
                <Progress value={currentFrameData?.energy !== null &&
                  currentFrameData?.energy !== undefined &&
                  currentFrameData?.max_energy !== null &&
                  currentFrameData?.max_energy !== undefined
                    ? (currentFrameData.energy / currentFrameData.max_energy) * 100
                        : 0} 
                        className="w-full"
                />
                  
                </div>
                <span className="text-sm font-medium text-foreground whitespace-nowrap min-w-[120px] text-right">
                  {currentFrameData?.energy !== null &&
                  currentFrameData?.energy !== undefined
                    ? `${currentFrameData.energy.toFixed(1)}/${
                        currentFrameData.max_energy
                          ? currentFrameData.max_energy.toFixed(1)
                          : "--"
                      } (${
                        currentFrameData.max_energy
                          ? Math.round(
                              (currentFrameData.energy / currentFrameData.max_energy) * 100
                            )
                          : 0
                      }%)`
                    : "--"}
                </span>
              </div>
            </div>
            
            {/* 3D Scene */}
            <div className="flex-1 w-full relative overflow-hidden">
              <Scene3D
                frames={data.frames}
                metadata={data.metadata}
                currentFrameIndex={currentFrame}
                maxHistory={MAX_HISTORY}
                reachedGoalsByFrame={data.reachedGoalsByFrame}
              />
            </div>
          </TabsContent>
          
          <TabsContent value="plots" className="flex-1 m-0 overflow-auto">
            <div className="h-full p-6">
              <TrainingPlots 
                trainingMetrics={data.trainingMetrics}
                frames={data.frames}
                reachedGoalsByFrame={data.reachedGoalsByFrame}
              />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default App;

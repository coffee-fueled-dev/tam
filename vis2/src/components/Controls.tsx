import type { Frame } from "@/types";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";

interface ControlsProps {
  currentFrame: number;
  totalFrames: number;
  isPlaying: boolean;
  playbackSpeed: number;
  currentFrameData: Frame | null;
  goalCount: number;
  onPlayPause: () => void;
  onSpeedChange: (speed: number) => void;
  onFrameChange: (frame: number) => void;
}

export function Controls({
  currentFrame,
  totalFrames,
  isPlaying,
  playbackSpeed,
  currentFrameData,
  goalCount,
  onPlayPause,
  onSpeedChange,
  onFrameChange,
}: ControlsProps) {
  const handleSliderChange = (value: number[]) => {
    onFrameChange(value[0] ?? 0);
  };

  return (
    <div className="flex items-center justify-between gap-4 p-4">
      {/* Left side - Goals */}
      <div className="flex items-center gap-4">
        <div className="px-3 py-1.5 bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-md text-sm font-semibold text-amber-900 dark:text-amber-100">
          Goals: {goalCount}
        </div>
      </div>

      {/* Center - Playback Controls */}
      <div className="flex items-center gap-4 flex-1 max-w-2xl">
        <Button onClick={onPlayPause} variant="default" size="default">
          {isPlaying ? "Pause" : "Play"}
        </Button>
        
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground whitespace-nowrap">Speed:</span>
          <Select
            value={playbackSpeed.toString()}
            onValueChange={(value) => onSpeedChange(parseFloat(value))}
          >
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0.25">0.25x</SelectItem>
              <SelectItem value="0.5">0.5x</SelectItem>
              <SelectItem value="1">1x</SelectItem>
              <SelectItem value="2">2x</SelectItem>
              <SelectItem value="4">4x</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2 flex-1">
          <span className="text-sm text-muted-foreground whitespace-nowrap min-w-[100px]">
            Frame: {currentFrame + 1} / {totalFrames}
          </span>
          <Slider
            value={[currentFrame]}
            onValueChange={handleSliderChange}
            min={0}
            max={Math.max(0, totalFrames - 1)}
            step={1}
            className="flex-1"
          />
        </div>
      </div>
    </div>
  );
}

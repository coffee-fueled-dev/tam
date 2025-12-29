import type { Key } from "./sequencer";

export interface IGateSnapshot<
  TCustomMetrics extends Record<string, number> = Record<string, number>
> {
  /** The name of the gate */
  name: string;
  /** Number of inputs processed */
  ingested?: number;
  /** Pass rate of the gate */
  passRate?: number;
  /** Custom metrics */
  customMetrics?: TCustomMetrics;
}

export interface IGateConfig {
  name?: string;
}

export interface IGate<
  TCustomMetrics extends Record<string, number> = Record<string, number>
> {
  /** Evaluates the current state of the gate
   * @returns true if the pattern should continue extending, false if it should segment here and emit the last known pattern
   */
  evaluate(current: Key, previous: Key): boolean;
  /** Resets the gate state */
  reset(): void;
  /** Gets the current state of the gate */
  snapshot(): Promise<IGateSnapshot<TCustomMetrics>>;
}

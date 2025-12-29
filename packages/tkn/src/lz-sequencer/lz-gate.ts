import type { IGate, IGateConfig } from "../sequencer";
import type { IDictionary } from "./dictionary/dictionary";

export interface LZGateConfig extends IGateConfig {
  cache: IDictionary;
  stats?: boolean;
}

export type LZCustomMetrics = {
  cacheUtilization: number;
};

export class LZGate implements IGate<LZCustomMetrics> {
  private _cache: IDictionary;
  private _stats?: boolean;

  constructor({ name = "LZGate", cache, stats }: LZGateConfig) {
    this._name = name ?? this.constructor.name;
    this._cache = cache;
    this._stats = stats;
  }

  // Simple LZ-style inclusion heuristic
  private _currentWasKnown = false;
  evaluate: IGate<LZCustomMetrics>["evaluate"] = (current) => {
    if (!this._stats) return this._cache.merge(current);
    this._ingested += 1;
    this._currentWasKnown = this._cache.merge(current);
    if (this._currentWasKnown) this._passed += 1;
    return this._currentWasKnown;
  };

  reset = (): void => {
    this._cache.clear();
  };

  private _ingested = 0;
  private _passed = 0;
  private _name: string;
  snapshot: IGate<LZCustomMetrics>["snapshot"] = async () => ({
    name: this._name,
    passRate: this._passed / this._ingested,
    ingested: this._ingested,
    customMetrics: {
      cacheUtilization: this._cache.size,
    },
  });
}

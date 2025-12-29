import { Queue, Sequencer } from "../sequencer";
import { Bounded } from "./dictionary/bounded";
import { Unbounded } from "./dictionary/unbounded";
import { LZGate } from "./lz-gate";
import { isDictionary, type IDictionary } from "./dictionary/dictionary";

export interface LZSequencerProperties {
  cacheOptions?:
    | { bounded: true; max: number }
    | { bounded: false }
    | IDictionary
    | undefined;
  historyOptions?: { bounded: true; maxLength: number } | { bounded: false };
  emissionPolicy?: "immediate"; // TODO: add other policies
}
export const createLZSequencer = (
  properties?: LZSequencerProperties
): Sequencer<LZGate[]> => {
  const cache = isDictionary(properties?.cacheOptions)
    ? properties.cacheOptions
    : properties?.cacheOptions?.bounded
    ? new Bounded(properties.cacheOptions.max)
    : new Unbounded();

  const gate = new LZGate({ cache });

  const queue = new Queue({
    historyOptions: properties?.historyOptions,
  });

  const sequencer = new Sequencer({
    gates: [gate],
    queue,
  });

  return sequencer;
};

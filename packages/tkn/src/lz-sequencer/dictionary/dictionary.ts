import type { Key } from "../../sequencer";

export interface IDictionary {
  /**
   * Upsert a key in the cache
   * @param input The key to upsert
   *
   * @returns true if the input was already present in the dictionary
   */
  merge(input: Key): boolean;
  clear(): void;
  size: number;
}

export function isDictionary(value: unknown): value is IDictionary {
  return (
    typeof value === "object" &&
    value !== null &&
    "merge" in value &&
    typeof value.merge === "function" &&
    "clear" in value &&
    typeof value.clear === "function" &&
    "size" in value &&
    typeof value.size === "number"
  );
}

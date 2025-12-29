import type { Key } from "../../sequencer";
import type { IDictionary } from "./dictionary";

/**
 *
 */
export class Unbounded implements IDictionary {
  private _set = new Set<Key>();

  constructor() {}

  merge(input: Key): boolean {
    if (this._set.has(input)) return true;
    this._set.add(input);
    return false;
  }

  clear = (): void => {
    this._set.clear();
  };

  get size(): number {
    return this._set.size;
  }
}

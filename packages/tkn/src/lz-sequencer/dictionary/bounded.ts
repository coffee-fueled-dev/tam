import type { Key } from "../../sequencer";
import type { IDictionary } from "./dictionary";

/**
 * The Bounded dictionary respects a maximum size using a FIFO queue
 */
export class Bounded implements IDictionary {
  private _set = new Set<Key>();
  private _buf: Key[];
  private _head = 0; // index of the oldest element
  private _size = 0; // number of valid elements in buffer
  readonly _max: number;

  constructor(max: number) {
    if (max <= 0 || !Number.isFinite(max)) {
      throw new Error(`Bounded max must be a positive integer; got ${max}`);
    }
    this._max = max | 0;
    this._buf = new Array(this._max);
  }

  merge(input: Key): boolean {
    if (this._set.has(input)) return true;

    if (this._size < this._max) {
      // append at tail
      const tail = (this._head + this._size) % this._max;
      this._buf[tail] = input;
      this._size++;
    } else {
      // evict oldest, overwrite its slot, then advance head
      const oldest = this._buf[this._head];
      // oldest is always defined when _size === _max
      this._set.delete(oldest!);
      this._buf[this._head] = input;
      this._head = (this._head + 1) % this._max;
    }

    this._set.add(input);
    return false;
  }

  clear = (): void => {
    this._set.clear();
    this._head = 0;
    this._size = 0;
  };

  get size(): number {
    return this._size;
  }
}

import type { SequencerOutput } from "../sequencer";

export interface IQueueHistory {
  push(output: SequencerOutput): void;
  get(): SequencerOutput[];
  clear(): void;
  readonly size: number;
}

export class UnboundedHistory implements IQueueHistory {
  private _history: SequencerOutput[] = [];

  push(output: SequencerOutput): void {
    this._history.push(output);
  }

  get(): SequencerOutput[] {
    return this._history;
  }

  clear(): void {
    this._history = [];
  }

  get size(): number {
    return this._history.length;
  }
}

export class BoundedHistory implements IQueueHistory {
  private _buffer: SequencerOutput[];
  private _head = 0;
  private _size = 0;
  readonly _max: number;

  constructor(max: number) {
    if (max <= 0 || !Number.isFinite(max)) {
      throw new Error(`History max must be a positive integer; got ${max}`);
    }
    this._max = max | 0;
    this._buffer = new Array(this._max);
  }

  push(output: SequencerOutput): void {
    if (this._size < this._max) {
      const tail = (this._head + this._size) % this._max;
      this._buffer[tail] = output;
      this._size++;
    } else {
      // Overwrite oldest entry and advance head
      this._buffer[this._head] = output;
      this._head = (this._head + 1) % this._max;
    }
  }

  get(): SequencerOutput[] {
    const result: SequencerOutput[] = [];
    for (let i = 0; i < this._size; i++) {
      result.push(this._buffer[(this._head + i) % this._max]);
    }
    return result;
  }

  clear(): void {
    this._head = 0;
    this._size = 0;
  }

  get size(): number {
    return this._size;
  }
}

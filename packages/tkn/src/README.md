# tkn

A fast, greedy pattern discovery algorithm for sequential data, based on LZ-style compression heuristics.

## What is tkn?

**tkn is not a tokenizer.** Instead, it focuses on discovering patterns in sequential data through greedy segmentation. By tracking which patterns have been seen before (using an LZ-style inclusion heuristic), tkn naturally segments data at boundaries where patterns become novel.

The algorithm processes input one item at a time, growing sequences as long as they remain "known" and emitting them when they become "unknown." Over time, the emission frequencies and co-occurrence patterns encode the structure of your data, which can then be used for tasks like tokenization, compression, or analysis.

## Goals

- **Fast pattern discovery**: Greedy, single-pass algorithm with minimal overhead
- **No training required**: Discovers patterns online as data arrives
- **Flexible**: Works with any sequential data (text, events, tokens, etc.)
- **Composable**: Gate-based architecture allows custom heuristics

## When to use tkn

✅ **Good for:**

- Discovering repetitive patterns in sequential data
- Building adaptive segmentation systems
- Analyzing data structure through emission frequencies
- Online/streaming pattern discovery
- Building tokenizers or compression schemes

❌ **Not for:**

- Optimal segmentation (tkn is greedy, not optimal)
- Direct replacement for trained tokenizers
- Fixed vocabulary requirements

## Installation

```bash
bun add @very-coffee/tkn
```

## Quick Start

```typescript
import { createLZSequencer } from "@very-coffee/tkn";

// Create a sequencer with default settings
const sequencer = createLZSequencer();

// Process input
const text = "hello world hello world";
for (const char of text) {
  sequencer.push(char);
}
sequencer.flush(); // Emit remaining buffer

// Read discovered patterns
for await (const { sequence, key } of sequencer.read()) {
  console.log("Pattern:", sequence, "Key:", key);
}
```

## API Reference

### `createLZSequencer(properties?)`

Factory function to create a pre-configured LZ-style sequencer.

**Parameters:**

```typescript
{
  cacheOptions?: { bounded: true; max: number } | { bounded: false };
  historyOptions?: { bounded: true; maxLength: number } | { bounded: false };
  emissionPolicy?: "immediate";
}
```

**Returns:** `Sequencer<LZGate[]>`

### `Sequencer`

The core pattern discovery engine.

**Methods:**

- `push(input: string | Sentinel)` - Process a single input
- `flush()` - Emit the current buffer as a pattern
- `reset()` - Clear all internal state
- `read()` - AsyncGenerator that yields discovered patterns
- `snapshot()` - Get current state and statistics

**Properties:**

- `history` - Array of all emitted patterns
- `durationMS` - Time since first input

### `LZGate`

Implements the LZ-style inclusion heuristic.

**Constructor:**

```typescript
new LZGate({
  name?: string;
  cache: IDictionary;  // Bounded or Unbounded
  stats?: boolean;
});
```

**Interface:**

- `evaluate(current: Key, previous: Key): boolean` - Returns true if pattern should continue growing
- `reset()` - Clear gate state
- `snapshot()` - Get gate statistics and cache utilization

### `Queue`

Manages the output queue and coordinates resegmentation.

**Constructor:**

```typescript
new Queue({
  resegmenters?: IResegmenter[];
  historyOptions?: { bounded: true; maxLength: number } | { bounded: false };
});
```

**Methods:**

- `push(output: SequencerOutput)` - Add an emitted pattern to the queue
- `read()` - AsyncGenerator that yields queued patterns
- `history` - Array of all consumed patterns

### `Resegmenter`

Abstract base class for implementing resegmentation logic.

**Abstract Methods:**

```typescript
abstract class Resegmenter {
  protected abstract transform(segments: SequencerOutput[]): SequencerOutput[];
  protected abstract shouldEmit(segments: SequencerOutput[]): boolean;
}
```

- `transform()` - Reorganize or modify the segments
- `shouldEmit()` - Return true to emit the queue without resegmentation (when all resegmenters return true, the queue flushes)

### `Unicode`

Utilities for Unicode-aware text processing.

**Static Methods:**

- `toCodepoints(text: string): number[]` - Convert text to codepoint array
- `toString(codepoints: number[]): string` - Convert codepoints back to text
- `streamFile(file: BunFile)` - Async generator for streaming file contents character-by-character (NFC normalized)

## Core Concepts

### Pattern Discovery

The algorithm maintains a "dictionary" of seen patterns and a "buffer" of the current sequence:

```
for each input:
  extended ← pattern + input

  if dictionary contains extended:
    // Known pattern - keep growing
    pattern ← extended
    add input to buffer
  else:
    // Unknown pattern - emit and reset
    emit buffer as discovered sequence
    dictionary learns extended
    buffer ← [input]
```

### Gates

Gates implement heuristics for deciding when to segment. The `LZGate` uses dictionary membership (have we seen this pattern before?) as its heuristic. You can implement custom gates by implementing the `IGate` interface.

### Sentinels

Special markers in the format `<number>` that can be used to inject boundaries or metadata into the sequence without being treated as regular input.

### Resegmenters

Resegmenters allow you to refine or reorganize the emitted patterns after initial segmentation. Each time a new pattern is emitted, resegmenters can examine the current queue and transform it based on custom logic.

**Use cases:**

- Merge adjacent patterns that meet certain criteria
- Split patterns based on higher-level rules
- Filter or reorder segments
- Apply post-processing transformations

A resegmenter runs on every emission until its `shouldEmit` condition is met. When all resegmenters signal they're ready to emit, the queue flushes its contents.

**Example resegmenter:**

```typescript
class MergeShortPatterns extends Resegmenter {
  protected shouldEmit(segments: SequencerOutput[]): boolean {
    // Emit when we don't have enough segments to merge
    return segments.length < 2;
  }

  protected transform(segments: SequencerOutput[]): SequencerOutput[] {
    const merged: SequencerOutput[] = [];
    let buffer: SequencerOutput | null = null;

    for (const segment of segments) {
      if (segment.sequence.length < 3) {
        // Accumulate short patterns
        if (!buffer) {
          buffer = segment;
        } else {
          buffer = {
            sequence: [...buffer.sequence, ...segment.sequence],
            key: buffer.key + segment.key,
          };
        }
      } else {
        // Flush accumulated patterns before long one
        if (buffer) {
          merged.push(buffer);
          buffer = null;
        }
        merged.push(segment);
      }
    }

    if (buffer) merged.push(buffer);
    return merged;
  }
}

// Use with a Queue
const queue = new Queue({
  resegmenters: [new MergeShortPatterns()],
});
```

Multiple resegmenters can be chained, with each operating on the output of the previous one.

## Advanced Usage

### Custom Gates

```typescript
import { Sequencer, Queue, type IGate } from "@very-coffee/tkn";

class MyCustomGate implements IGate {
  evaluate(current: string, previous: string): boolean {
    // Return true to continue growing, false to segment
    return myCustomLogic(current, previous);
  }

  reset() {
    // Clear any internal state
  }

  async snapshot() {
    return {
      name: "MyCustomGate",
      ingested: this.inputCount,
      passRate: this.passRate,
    };
  }
}

const sequencer = new Sequencer({
  gates: [new MyCustomGate()],
  queue: new Queue({}),
});
```

### Using Resegmenters

Resegmenters operate on the queue level, transforming patterns after emission:

```typescript
import {
  Sequencer,
  Queue,
  Resegmenter,
  LZGate,
  Unbounded,
} from "@very-coffee/tkn";

class MergeAdjacentDuplicates extends Resegmenter {
  protected shouldEmit(segments: SequencerOutput[]): boolean {
    // Emit when there's nothing to merge
    return segments.length < 2;
  }

  protected transform(segments: SequencerOutput[]): SequencerOutput[] {
    const result: SequencerOutput[] = [];
    let prev: SequencerOutput | null = null;

    for (const segment of segments) {
      if (prev && prev.key === segment.key) {
        // Merge with previous
        prev = {
          sequence: [...prev.sequence, ...segment.sequence],
          key: prev.key,
        };
      } else {
        if (prev) result.push(prev);
        prev = segment;
      }
    }
    if (prev) result.push(prev);
    return result;
  }
}

const sequencer = new Sequencer({
  gates: [new LZGate({ cache: new Unbounded() })],
  queue: new Queue({
    resegmenters: [new MergeAdjacentDuplicates()],
  }),
});
```

### Bounded Dictionary

For memory-constrained environments:

```typescript
const sequencer = createLZSequencer({
  cacheOptions: { bounded: true, max: 10000 },
  historyOptions: { bounded: true, maxLength: 1000 },
});
```

## License

MIT

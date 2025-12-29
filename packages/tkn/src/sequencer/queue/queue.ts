import type { SequencerOutput } from "../sequencer";
import {
  BoundedHistory,
  UnboundedHistory,
  type IQueueHistory,
} from "./history";
import type { IResegmenter } from "./resegmenter";

export interface IQueue {
  push(input: SequencerOutput | void): void;
  read(): AsyncGenerator<SequencerOutput, void, unknown>;
  close(): void;
  history: SequencerOutput[];
}

type Resolver = (value: SequencerOutput) => void;

export class Queue implements IQueue {
  private _resegmenters?: IResegmenter[];
  private _queue: SequencerOutput[] = [];
  private _resolvers: Resolver[] = [];
  private _history?: IQueueHistory;
  private _closed = false;

  /**
   * @param resegmenters
   * @param history
   */
  constructor({
    resegmenters,
    historyOptions,
  }: {
    resegmenters?: IResegmenter[];
    historyOptions?: { bounded: true; maxLength: number } | { bounded: false };
  }) {
    this._resegmenters = resegmenters;

    if (historyOptions) {
      this._history = historyOptions.bounded
        ? new BoundedHistory(historyOptions.maxLength)
        : new UnboundedHistory();
    }
  }

  push: IQueue["push"] = (input) => {
    if (!input) return;
    this._queue.push(input);

    if (this._resegmenters) {
      const { skipped, segments: resegmentedQueue } = Queue.resegment(
        this._queue,
        this._resegmenters
      );
      if (!skipped) return void 0;
      return Queue.drain(resegmentedQueue, this._resolvers, this._history);
    } else {
      return Queue.drain(this._queue, this._resolvers, this._history);
    }
  };

  async *read(): ReturnType<IQueue["read"]> {
    while (true) {
      const item = Queue.consumeNext(this._queue, this._history);
      if (item) {
        yield item;
      } else if (this._closed) {
        // No more items and queue is closed, exit the generator
        return;
      } else {
        yield await new Promise<SequencerOutput>((resolve) => {
          this._resolvers.push(resolve);
        });
      }
    }
  }

  close(): void {
    this._closed = true;
    // Reject any pending resolvers since no more data is coming
    while (this._resolvers.length > 0) {
      this._resolvers.shift();
    }
  }

  get history(): SequencerOutput[] {
    return this._history?.get() ?? [];
  }

  private static consumeNext(
    queue: SequencerOutput[],
    history?: IQueueHistory
  ): SequencerOutput | undefined {
    const item = queue.shift();
    if (item && history) history.push(item);
    return item;
  }

  private static drain(
    queue: SequencerOutput[],
    resolvers: Resolver[],
    history?: IQueueHistory
  ) {
    while (queue.length > 0 && resolvers.length > 0) {
      const item = Queue.consumeNext(queue, history)!;
      resolvers.shift()!(item);
    }
  }

  private static resegment = (
    initialSegments: SequencerOutput[],
    resegmenters: IResegmenter[]
  ): ReturnType<IResegmenter["evaluate"]> =>
    resegmenters.reduce<ReturnType<IResegmenter["evaluate"]>>(
      (lastSegmentation, resegmenter) => {
        const evaluation = resegmenter.evaluate(lastSegmentation.segments);
        if (evaluation.skipped) return lastSegmentation;
        return evaluation;
      },
      {
        skipped: true,
        segments: initialSegments,
      }
    );
}

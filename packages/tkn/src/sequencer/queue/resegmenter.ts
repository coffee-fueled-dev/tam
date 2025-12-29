import type { SequencerOutput } from "../sequencer";

export interface IResegmenter {
  evaluate(segments: SequencerOutput[]): {
    segments: SequencerOutput[];
    skipped: boolean;
  };
}

export abstract class Resegmenter implements IResegmenter {
  evaluate: IResegmenter["evaluate"] = (segments) => {
    if (this.shouldEmit(segments)) return { skipped: true, segments };
    return { skipped: false, segments: this.transform(segments) };
  };
  protected abstract transform(input: SequencerOutput[]): SequencerOutput[];
  protected abstract shouldEmit(segments: SequencerOutput[]): boolean;
}

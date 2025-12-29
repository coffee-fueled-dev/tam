/**
 * Unicode-aware text reader that yields codepoints instead of UTF-8 bytes
 */
export class Unicode {
  /**
   * Convert a text string into an array of Unicode codepoints
   * Each codepoint is a number that represents one logical character
   */
  static toCodepoints(text: string): number[] {
    const codepoints: number[] = [];
    for (let i = 0; i < text.length; i++) {
      codepoints.push(text[i].codePointAt(0)!);
    }
    return codepoints;
  }

  /**
   * Convert an array of Unicode codepoints back to a string
   */
  static toString(codepoints: number[]): string {
    return String.fromCodePoint(...codepoints);
  }

  /**
   * Stream a file, character by character, normalizing the text to NFC
   * @param file
   */
  static async *streamFile(file: Bun.BunFile | File) {
    const textStream = file.stream().pipeThrough(new TextDecoderStream());
    const reader = textStream.getReader();

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        const text = value.normalize("NFC"); // Normalize for cannonical stability
        for (let i = 0; i < text.length; i++) {
          yield text[i];
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

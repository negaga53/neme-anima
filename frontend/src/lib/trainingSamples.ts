import type { TrainingSampleEpoch } from "./types";

export interface SampleCell {
  url: string;
  file: string;
}

export interface SampleGrid {
  /** Columns, ascending by epoch. */
  epochs: { epoch: number; error: string | null }[];
  /** One row per distinct prompt (first-seen order); a cell is null when
   *  that epoch has no image for the prompt (prompts edited mid-run, or the
   *  epoch failed). */
  rows: { prompt: string; cells: (SampleCell | null)[] }[];
}

export function buildSampleGrid(epochs: TrainingSampleEpoch[]): SampleGrid {
  const sorted = [...epochs].sort((a, b) => a.epoch - b.epoch);
  const prompts: string[] = [];
  for (const e of sorted) {
    for (const img of e.images) if (!prompts.includes(img.prompt)) prompts.push(img.prompt);
  }
  return {
    epochs: sorted.map((e) => ({ epoch: e.epoch, error: e.error })),
    rows: prompts.map((prompt) => ({
      prompt,
      cells: sorted.map((e) => {
        const img = e.images.find((i) => i.prompt === prompt);
        return img ? { url: img.url, file: img.file } : null;
      }),
    })),
  };
}

/** Textarea → prompt list: one prompt per line, trimmed, blanks dropped. */
export function parsePromptLines(text: string): string[] {
  return text.split("\n").map((s) => s.trim()).filter(Boolean);
}

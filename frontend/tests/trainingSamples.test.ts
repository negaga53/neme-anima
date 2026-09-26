import { describe, expect, it } from "vitest";
import { buildSampleGrid, parsePromptLines } from "../src/lib/trainingSamples";
import type { TrainingSampleEpoch } from "../src/lib/types";

function ep(epoch: number, images: [string, string][], error: string | null = null): TrainingSampleEpoch {
  return {
    epoch, dir: `epoch${String(epoch).padStart(4, "0")}`,
    prompts: images.map(([p]) => p), negative_prompt: "", settings: {},
    images: images.map(([prompt, file]) => ({ prompt, file, url: `/u/${epoch}/${file}`, mtime: 1 })),
    error,
  };
}

describe("buildSampleGrid", () => {
  it("orders epochs and builds prompt rows", () => {
    const g = buildSampleGrid([ep(20, [["a", "p00.png"]]), ep(10, [["a", "p00.png"], ["b", "p01.png"]])]);
    expect(g.epochs.map((e) => e.epoch)).toEqual([10, 20]);
    expect(g.rows.map((r) => r.prompt)).toEqual(["a", "b"]);
    expect(g.rows[0].cells.map((c) => c?.url)).toEqual(["/u/10/p00.png", "/u/20/p00.png"]);
    expect(g.rows[1].cells).toEqual([{ url: "/u/10/p01.png", file: "p01.png" }, null]);
  });

  it("keeps errored epochs as columns", () => {
    const g = buildSampleGrid([ep(10, [], "boom")]);
    expect(g.epochs).toEqual([{ epoch: 10, error: "boom" }]);
    expect(g.rows).toEqual([]);
  });
});

describe("parsePromptLines", () => {
  it("trims and drops blanks", () => {
    expect(parsePromptLines(" a \n\n  \nb, c\n")).toEqual(["a", "b, c"]);
  });

  it("drops duplicate prompts", () => {
    expect(parsePromptLines("a\nb\n a ")).toEqual(["a", "b"]);
  });
});

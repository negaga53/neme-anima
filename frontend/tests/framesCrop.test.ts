import { beforeEach, describe, expect, it } from "vitest";
import { framesStore } from "../src/lib/stores/frames.svelte";
import type { FrameRecord } from "../src/lib/types";

function frame(filename: string, overrides: Partial<FrameRecord> = {}): FrameRecord {
  return {
    filename,
    kept: true,
    video_stem: "ep01",
    scene_idx: 0,
    tracklet_id: 1,
    frame_idx: 10,
    timestamp_seconds: 0.4,
    ccip_distance: 0.1,
    score: 0.9,
    character_slug: "default",
    has_description: false,
    has_tags: true,
    has_crop: false,
    ...overrides,
  };
}

describe("framesStore.markCropped", () => {
  beforeEach(() => {
    // Module singleton — reset the slices these tests touch for independence.
    framesStore.items = [];
    framesStore.cropVersions = new Map();
  });

  it("flips has_crop on the row in place so the grid needs no full refresh", () => {
    // A full refresh toggles `loading`, which empties the grid DOM and scrolls
    // the page to the top after a crop save — the bug this guards against.
    framesStore.items = [frame("a"), frame("b")];
    framesStore.markCropped("a");
    expect(framesStore.items[0].has_crop).toBe(true);
    expect(framesStore.cropVersion("a")).toBe(1);
    // Siblings are untouched.
    expect(framesStore.items[1].has_crop).toBe(false);
  });

  it("re-cropping an already-cropped frame just bumps the cache-bust version", () => {
    framesStore.items = [frame("a", { has_crop: true })];
    framesStore.markCropped("a");
    framesStore.markCropped("a");
    expect(framesStore.cropVersion("a")).toBe(2);
    expect(framesStore.items[0].has_crop).toBe(true);
  });

  it("bumps the version even when the filename isn't in the current view", () => {
    framesStore.markCropped("ghost");
    expect(framesStore.cropVersion("ghost")).toBe(1);
  });
});

import { describe, it, expect } from "vitest";
import {
  computeAutocompletePosition,
  AC_MARGIN,
  type AnchorRect,
} from "$lib/tagAutocompletePosition";

const VW = 1280;
const VH = 800;

function rect(partial: Partial<AnchorRect>): AnchorRect {
  return { top: 100, bottom: 124, left: 100, width: 120, ...partial };
}

describe("computeAutocompletePosition", () => {
  it("anchors below-left of the input when there's room", () => {
    const p = computeAutocompletePosition(rect({}), 5, VW, VH);
    expect(p.flipUp).toBe(false);
    expect(p.top).toBe(124 + 2); // anchor.bottom + 2
    expect(p.left).toBe(100); // == anchor.left, no clamp needed
    expect(p.width).toBe(220); // max(anchor.width, 220)
  });

  it("never spills past the right edge when anchored near it", () => {
    // Input sitting in the modal's right-hand panel: its left is well within
    // 220px of the viewport's right edge, so an un-clamped dropdown would
    // overflow off-screen. This is the reported bug.
    const anchor = rect({ left: VW - 140, width: 120 });
    const p = computeAutocompletePosition(anchor, 8, VW, VH);
    expect(p.left).toBeLessThan(anchor.left); // pulled inward
    expect(p.left + p.width).toBeLessThanOrEqual(VW - AC_MARGIN);
    expect(p.left).toBeGreaterThanOrEqual(AC_MARGIN);
  });

  it("never spills past the left edge", () => {
    const p = computeAutocompletePosition(rect({ left: 2, width: 40 }), 5, VW, VH);
    expect(p.left).toBeGreaterThanOrEqual(AC_MARGIN);
  });

  it("caps width and max-width so it stays inside a narrow viewport", () => {
    const narrow = 200;
    const p = computeAutocompletePosition(rect({ left: 0, width: 120 }), 5, narrow, VH);
    expect(p.width).toBeLessThanOrEqual(narrow - 2 * AC_MARGIN);
    expect(p.left + p.maxWidth).toBeLessThanOrEqual(narrow - AC_MARGIN);
    expect(p.maxWidth).toBeGreaterThanOrEqual(p.width); // min-width never exceeds max-width
  });

  it("flips above the input when there's no room below but room above", () => {
    // Anchor near the bottom of the viewport with plenty of space above.
    const anchor = rect({ top: VH - 30, bottom: VH - 6 });
    const p = computeAutocompletePosition(anchor, 10, VW, VH);
    expect(p.flipUp).toBe(true);
    expect(p.top).toBeLessThan(anchor.top); // sits above the input
    expect(p.top).toBeGreaterThanOrEqual(0);
  });

  it("stays below (does not flip) when there's no room above either", () => {
    // Tiny viewport: anchor near the bottom but also near the top.
    const anchor = rect({ top: 10, bottom: 34 });
    const p = computeAutocompletePosition(anchor, 10, VW, 60);
    expect(p.flipUp).toBe(false);
    expect(p.top).toBe(34 + 2);
  });
});

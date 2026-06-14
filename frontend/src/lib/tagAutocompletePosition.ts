// Pure positioning math for the tag autocomplete dropdown (TagAutocomplete.svelte).
//
// The dropdown renders in a `position:fixed` layer off the anchor input's
// bounding rect so it escapes the overflow-hidden grid-hover panel. The anchor
// often sits in the modal's right-hand panel or a narrow grid tile, so we must
// clamp the box to the viewport — otherwise it spills off the right edge ("out
// of frame"). Kept pure (no DOM) so it's unit-testable; the component just feeds
// it `getBoundingClientRect()` + `window.inner{Width,Height}`.

/** Minimal shape we need from a DOMRect — kept narrow so tests can fake it. */
export interface AnchorRect {
  top: number;
  bottom: number;
  left: number;
  width: number;
}

export interface DropdownPosition {
  top: number;
  left: number;
  /** Desired (min) width in px. */
  width: number;
  /** Hard cap so long suggestions can't grow the box past the viewport. */
  maxWidth: number;
  flipUp: boolean;
}

const ROW_PX = 26; // approx height per suggestion row, for the flip-up estimate
const MIN_WIDTH = 220;
/** Gap kept between the dropdown and the viewport edges. */
export const AC_MARGIN = 8;

export function computeAutocompletePosition(
  anchor: AnchorRect,
  count: number,
  viewportW: number,
  viewportH: number,
): DropdownPosition {
  const estHeight = Math.min(count, 10) * ROW_PX + 8;
  // Flip above the input only when there's no room below AND room above.
  const flipUp = anchor.bottom + estHeight > viewportH && anchor.top > estHeight;
  const top = flipUp ? anchor.top - estHeight : anchor.bottom + 2;

  // Width never exceeds the viewport, so min-width can't force an overflow.
  const width = Math.min(Math.max(anchor.width, MIN_WIDTH), viewportW - 2 * AC_MARGIN);
  // Prefer aligning to the anchor's left, but pull inward when that would push
  // the right edge off-screen, and never cross the left edge.
  const left = Math.max(AC_MARGIN, Math.min(anchor.left, viewportW - width - AC_MARGIN));
  // Available space to the right of `left` — bounds content-driven growth so
  // the right edge always lands at most `AC_MARGIN` from the viewport edge.
  const maxWidth = viewportW - left - AC_MARGIN;

  return { top, left, width, maxWidth, flipUp };
}

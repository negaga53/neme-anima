import * as api from "$lib/api";
import { SelectionModel } from "$lib/selection";
import type { FrameRecord, ServerEvent } from "$lib/types";

/** Per-filename UI state the grid watches. `retagged`/`described` are
 *  monotonic counters (not booleans) so re-tagging or re-describing an
 *  already-processed frame still ticks — FrameThumb keys its cache-bust and
 *  pop animation off an increase. `processing` is the in-flight flag for the
 *  bulk spinner. */
export type FrameState = {
  retagged: number;
  described: number;
  processing: boolean;
};

class FramesStore {
  items = $state<FrameRecord[]>([]);
  // Unfiltered count for the current source/kept_only view — used by the
  // top-bar count badge to render "X / total" when a tag query is active.
  totalInView = $state<number>(0);
  // On-disk kept frames whose slug isn't a current character — drives whether
  // the top bar shows the "Unsorted" chip. Refreshed on every list fetch so it
  // stays in step with moves/deletes/uploads/crops that re-list the grid.
  unsortedTotal = $state<number>(0);
  loading = $state(false);
  selection = new SelectionModel();
  selectionVersion = $state(0); // bump to force reactivity for selection changes
  // One per-filename state map (was three separate Map/Set fields). Svelte 5's
  // $state Map proxy tracks reads at the PER-KEY level — so a `set(fn, …)` only
  // invalidates effects that read `.get(fn)`, not effects keyed to another
  // filename. That is what lets FrameThumb cache-bust / pop a single tile
  // without disturbing the frame the user is editing in the modal.
  //
  // TWO INVARIANTS keep that working (breaking either reintroduces the
  // mid-edit TagPill-unmount bug fixed in commit 901aeaf):
  //   1. Mutate by IMMUTABLE PER-KEY REPLACEMENT — `states.set(fn, {...prev})`.
  //      Never mutate the stored object in place (`states.get(fn).retagged++`
  //      won't trigger — the inner object isn't a tracked proxy), and never
  //      reassign the whole map except the intentional reset in refresh().
  //   2. refresh() reassigns `states = new Map()` to broadly invalidate on a
  //      new view — coincidental filename reuse must not leak a stale counter
  //      or a stranded spinner.
  states = $state<Map<string, FrameState>>(new Map());

  // Monotonic per-filename crop counter, kept SEPARATE from `states` because
  // `refresh()` wipes `states` on every filter change — a crop cache-bust must
  // survive that. Each `markCropped` bump appends `?v=N` to the grid's crop
  // URL so re-cropping (which overwrites `<name>_crop.png` at the same URL)
  // shows the new pixels instead of the browser's cached copy. Reset only when
  // the active project changes (see `#lastSlug`).
  cropVersions = $state<Map<string, number>>(new Map());
  #lastSlug: string | null = null;

  /** Current state for a filename, or the zero state if none recorded yet. */
  #stateOf(filename: string): FrameState {
    return this.states.get(filename) ?? { retagged: 0, described: 0, processing: false };
  }

  async refresh(
    slug: string,
    opts: { source?: string; query?: string; characterSlug?: string } = {},
  ) {
    this.loading = true;
    // Crop cache-bust counters are per-project, not per-view — only clear them
    // when the project itself changes so a filter switch doesn't drop a bump.
    if (slug !== this.#lastSlug) {
      this.cropVersions = new Map();
      this.#lastSlug = slug;
    }
    try {
      const page = await api.listFrames(slug, opts);
      this.items = page.items;
      this.totalInView = page.total;
      this.unsortedTotal = page.unsorted_total ?? 0;
      // Drop the per-filename state map on refresh — a different filter or
      // project could reuse filenames coincidentally, and we never want a
      // stale bump or a stranded spinner to leak into a fresh view.
      this.states = new Map();
    } finally {
      this.loading = false;
    }
  }

  /** Bump a frame's crop counter so the grid re-fetches its crop derivative.
   *  Per-key immutable mutation (like the `states` map) keeps the reactivity
   *  scoped to the one tile whose crop changed. */
  markCropped(filename: string) {
    this.cropVersions.set(filename, (this.cropVersions.get(filename) ?? 0) + 1);
  }

  /** Monotonic crop counter for a filename (0 if never cropped this session). */
  cropVersion(filename: string): number {
    return this.cropVersions.get(filename) ?? 0;
  }

  /** Mark a frame as freshly described: flip its has_description and bump the
   *  per-filename counter so FrameThumb runs its pop animation.
   *
   *  IMPORTANT: mutate the states map by immutable per-key replacement rather
   *  than reassigning the field. Svelte 5's reactive Map proxy tracks reads at
   *  the per-key level — so a `set(filename, …)` on this map only
   *  invalidates effects that read `.get(filename)`, not effects keyed to
   *  some other filename. Reassigning the whole map (`this.states = new Map(…)`)
   *  instead invalidates the field broadly, which then re-runs every
   *  FrameThumb's tag-cache-busting effect — including the one for the frame
   *  the user is currently editing. The user's TagPill unmounts mid-edit and
   *  the in-progress text is lost. */
  markDescribed(filename: string) {
    const idx = this.items.findIndex((i) => i.filename === filename);
    if (idx >= 0 && !this.items[idx].has_description) {
      this.items[idx] = { ...this.items[idx], has_description: true };
    }
    const prev = this.#stateOf(filename);
    this.states.set(filename, { ...prev, described: prev.described + 1 });
  }

  /** Bump a frame's retag counter so FrameThumb invalidates its tag cache.
   *  Mutates in place — see the note on markDescribed above. */
  markRetagged(filename: string) {
    const idx = this.items.findIndex((i) => i.filename === filename);
    if (idx >= 0 && !this.items[idx].has_tags) {
      this.items[idx] = { ...this.items[idx], has_tags: true };
    }
    const prev = this.#stateOf(filename);
    this.states.set(filename, { ...prev, retagged: prev.retagged + 1 });
  }

  setSidecarFlags(
    filename: string,
    flags: Partial<Pick<FrameRecord, "has_tags" | "has_description">>,
  ) {
    const idx = this.items.findIndex((i) => i.filename === filename);
    if (idx < 0) return;
    const current = this.items[idx];
    const next = { ...current, ...flags };
    if (
      next.has_tags !== current.has_tags ||
      next.has_description !== current.has_description
    ) {
      this.items[idx] = next;
    }
  }

  /** Add filenames to the in-flight set so their tiles show a spinner. */
  markProcessing(filenames: Iterable<string>) {
    for (const f of filenames) {
      this.states.set(f, { ...this.#stateOf(f), processing: true });
    }
  }

  /** Clear a single filename's in-flight flag (per-frame finish). */
  unmarkProcessing(filename: string) {
    const prev = this.states.get(filename);
    if (prev) this.states.set(filename, { ...prev, processing: false });
  }

  /** Monotonic retag counter for a filename (0 if never retagged). */
  retaggedVersion(filename: string): number {
    return this.states.get(filename)?.retagged ?? 0;
  }

  /** Monotonic describe counter for a filename (0 if never described). */
  describedVersion(filename: string): number {
    return this.states.get(filename)?.described ?? 0;
  }

  /** Whether a filename is currently in flight for a bulk action. */
  isProcessing(filename: string): boolean {
    return this.states.get(filename)?.processing ?? false;
  }

  /** Drop filenames from the selection without removing the underlying rows.
   *  Used by bulk-retag flows to clear each frame from selection as soon as
   *  it finishes processing. */
  deselect(filenames: Iterable<string>) {
    this.selection.remove(filenames);
    this.selectionVersion++;
  }

  ingest(event: ServerEvent) {
    if (event.type === "job.frame") {
      // A frame was just written by a running job — we'd ideally splice it in,
      // but the simpler invariant is: refresh on job.done. For now, no-op.
    }
  }

  click(index: number, mods: { shift: boolean; ctrl: boolean }) {
    this.selection.click(this.items.map(i => i.filename), index, mods);
    this.selectionVersion++;
  }

  selectAll() {
    this.selection.selectAll(this.items.map(i => i.filename));
    this.selectionVersion++;
  }

  clear() {
    this.selection.clear();
    this.selectionVersion++;
  }

  selectedFilenames(): string[] {
    return [...this.selection.selected()];
  }

  removeLocal(filenames: string[]) {
    const set = new Set(filenames);
    this.items = this.items.filter(i => !set.has(i.filename));
    this.selection.remove(set);
    this.selectionVersion++;
  }
}

export const framesStore = new FramesStore();

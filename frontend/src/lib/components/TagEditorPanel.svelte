<script lang="ts">
  import { onDestroy, untrack } from "svelte";
  import * as api from "$lib/api";
  import type { TagReview } from "$lib/types";
  import { createAsyncLoad } from "$lib/composables/asyncLoad.svelte";
  import { createFlash } from "$lib/composables/flash.svelte";
  import { reportDirty } from "$lib/composables/reportDirty.svelte";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { tagClipboard } from "$lib/stores/tagClipboard.svelte";
  import { parseTags, splitSidecar } from "$lib/sidecar";
  import { tagsEqual } from "$lib/tagList";
  import { getFrameOverwriteConfirm } from "$lib/frameOverwriteContext";
  import TagList from "./TagList.svelte";

  type Props = {
    filename: string;
    /** Report unsaved-edit state up to the modal's discard guard. */
    ondirty?: (dirty: boolean) => void;
    /** Close the whole modal (routed through the modal's discard guard). */
    onclose?: () => void;
    /** Delete the current frame (parent confirms + removes; bypasses the
     *  discard guard since the frame is being discarded anyway). */
    ondelete?: () => void;
  };
  const { filename, ondirty, onclose, ondelete }: Props = $props();

  const confirmFrameOverwrite = getFrameOverwriteConfirm();

  let autocompleteOn = $derived(projectsStore.active?.tag_autocomplete ?? true);
  // Same gate the Describe button uses: the Review button only appears once a
  // model has been picked in Settings.
  let llmModelSelected = $derived(!!projectsStore.active?.llm?.model);

  let savedTags = $state<string[]>([]); // last-persisted baseline
  let tags = $state<string[]>([]); // local working copy
  const loader = createAsyncLoad();
  let saving = $state(false);
  const savedFlash = createFlash();

  let tagging = $state(false);

  // ---- LLM tag review (staged: applies into `tags`, not the server) ----
  let reviewing = $state(false);
  let reviewResult = $state<TagReview | null>(null);
  // Which suggested removals/additions are currently checked. Keyed by tag
  // text; default all-checked when a review arrives.
  let acceptRemove = $state<Set<string>>(new Set());
  let acceptAdd = $state<Set<string>>(new Set());

  // ---- tag selection (surfaced from TagList; staging-only) ----
  // TagList owns the selection ring (left-click / Ctrl/Cmd / Shift) and reports
  // the selected subset here so we can render the Copy/Paste buttons. We hold an
  // imperative clear that TagList binds, used after a save / retag / review
  // replaces the tag set (the editor itself clears on click-away / Esc).
  let selectedTags = $state<string[]>([]);
  let clearTagSelection = $state<(() => void) | undefined>(undefined);
  // True once the current selection has been copied; Copy stays disabled until
  // the selection changes again.
  let copied = $state(false);
  // The clipboard tags that would actually be added to this frame — i.e. those
  // not already present (paste dedupes the rest away). Drives both the Paste
  // count and the hover preview so they match what the paste really does.
  let pasteTags = $derived(tagClipboard.tags.filter((t) => !tags.includes(t)));

  // Reset the "copied" latch whenever the surfaced selection changes.
  function onSelectionChange(next: string[]) {
    // TagList's surfacing $effect re-emits a FRESH array on every one of its
    // runs (its `selectedTags` is `tags.filter(...)`). We must bail on an
    // equal selection WITHOUT writing `selectedTags` — assigning a new array
    // identity here would re-render this panel, re-run TagList's effect, and
    // loop forever (effect_update_depth_exceeded). Only a genuine content
    // change updates state.
    const changed =
      next.length !== selectedTags.length ||
      next.some((t, i) => t !== selectedTags[i]);
    if (!changed) return;
    selectedTags = next;
    copied = false;
  }

  function copySelection() {
    if (selectedTags.length === 0 || copied) return;
    tagClipboard.set(selectedTags); // order-preserving; keeps selection intact
    copied = true;
  }
  function pasteClipboard() {
    if (tagClipboard.size === 0) return;
    tags = dedupe([...tags, ...tagClipboard.tags]); // append + dedupe → dirty
  }

  let dirty = $derived(!tagsEqual(tags, savedTags));
  reportDirty(() => dirty, () => ondirty);

  // Reload whenever the displayed frame changes (arrow-key nav in the modal).
  // This effect must depend on `filename` ONLY. Everything in the body is a
  // write or imperative call, wrapped in untrack() so it isn't a tracked read:
  //   - `clearTagSelection?.()` reads the `bind:clearSelection` $state, whose
  //     function identity TagList rewrites on every render. `load(fn)` reassigns
  //     `tags`, TagList re-renders, `clearSelection` gets a fresh identity, the
  //     binding pushes it back here — and if that read were tracked it would
  //     re-fire this effect → reload → re-render → new identity → infinite
  //     `GET .../tags` loop (the P3 regression). untrack() severs that.
  $effect(() => {
    const fn = filename;
    untrack(() => {
      reviewResult = null; // discard stale suggestions for the previous frame
      copied = false;
      clearTagSelection?.(); // selection is per-frame
      void load(fn);
    });
  });

  async function load(fn: string) {
    const slug = projectsStore.active?.slug;
    if (!slug || !fn) { loader.settle(); return; }
    await loader.run(
      () => api.getTags(slug, fn),
      (r) => {
        const parsed = parseTags(splitSidecar(r.text).danbooru);
        savedTags = parsed;
        tags = [...parsed];
      },
      () => {
        savedTags = [];
        tags = [];
      },
    );
  }

  function dedupe(list: string[]): string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const t of list) {
      if (seen.has(t)) continue;
      seen.add(t);
      out.push(t);
    }
    return out;
  }

  async function save() {
    const slug = projectsStore.active?.slug;
    if (!slug || saving || !dirty) return;
    saving = true;
    loader.error = null;
    try {
      // Single-line body → the server preserves the on-disk description and
      // applies its own dedupe/normalization.
      const r = await api.putTags(slug, filename, tags.join(", "));
      const parsed = parseTags(splitSidecar(r.text).danbooru);
      savedTags = parsed;
      tags = [...parsed];
      clearTagSelection?.(); // server normalization may reorder — drop stale index selection
      framesStore.markRetagged(filename);
      framesStore.setSidecarFlags(filename, { has_tags: parsed.length > 0 });
      savedFlash.trigger();
    } catch (e) {
      loader.error = e instanceof Error ? e.message : String(e);
    } finally {
      saving = false;
    }
  }

  // Re-run the WD14 tagger on this one frame. Mirrors the bulk image-selection
  // flow (api.bulkRetagDanbooru + the shared overwrite popup) but scoped to a
  // single filename, then reloads the freshly-written tag line into the panel.
  async function retagNow() {
    const slug = projectsStore.active?.slug;
    if (!slug || tagging || saving || loader.loading) return;
    const affected = savedTags.length > 0 ? 1 : 0;
    if (
      affected > 0 &&
      !(await confirmFrameOverwrite("retag", 1, affected))
    ) {
      return;
    }
    tagging = true;
    loader.error = null;
    try {
      const res = await api.bulkRetagDanbooru(slug, [filename]);
      if (res.retagged > 0) {
        await load(filename);
        clearTagSelection?.(); // tag set replaced — stale index selection
        framesStore.markRetagged(filename);
        framesStore.setSidecarFlags(filename, { has_tags: savedTags.length > 0 });
      }
    } catch (e) {
      loader.error = e instanceof Error ? e.message : String(e);
    } finally {
      tagging = false;
    }
  }

  // Ask the LLM to review the current tags against the (cropped) image. The
  // result is a proposed diff the user accepts/rejects; applying it only stages
  // into `tags` — nothing is written until the user hits "Save tags".
  async function reviewNow() {
    const slug = projectsStore.active?.slug;
    if (!slug || reviewing || saving || loader.loading || tagging) return;
    const fn = filename; // review is slow; guard against arrow-key nav meanwhile
    reviewing = true;
    loader.error = null;
    reviewResult = null;
    try {
      const r = await api.reviewTags(slug, fn);
      if (fn !== filename) return; // user navigated away — drop stale result
      reviewResult = r;
      acceptRemove = new Set(r.remove.map((x) => x.tag));
      acceptAdd = new Set(r.add.map((x) => x.tag));
    } catch (e) {
      if (fn !== filename) return;
      loader.error = e instanceof Error ? e.message : String(e);
    } finally {
      if (fn === filename) reviewing = false;
    }
  }

  function toggleAccept(set: Set<string>, tag: string): Set<string> {
    const next = new Set(set);
    if (next.has(tag)) next.delete(tag);
    else next.add(tag);
    return next;
  }

  let acceptedCount = $derived(acceptRemove.size + acceptAdd.size);

  function applyReview() {
    if (!reviewResult) return;
    let next = tags.filter((t) => !acceptRemove.has(t));
    for (const a of reviewResult.add) {
      if (acceptAdd.has(a.tag)) next.push(a.tag);
    }
    tags = dedupe(next);
    clearTagSelection?.(); // working list changed — stale index selection
    reviewResult = null;
  }

  onDestroy(() => savedFlash.destroy());
</script>

<div class="flex flex-col gap-2 flex-1 min-h-0">
  <div class="flex items-center justify-between">
    <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-400">
      Tags {#if tags.length}<span class="text-slate-600">({tags.length})</span>{/if}
    </h3>
    <div class="flex items-center gap-2">
      {#if savedFlash.active}
        <span class="text-[10px] text-emerald-400">Saved ✓</span>
      {/if}
      {#if ondelete}
        <button
          type="button"
          onclick={ondelete}
          aria-label="Delete image"
          title="Delete image"
          class="w-6 h-6 rounded text-slate-400 hover:text-rose-300 hover:bg-rose-950/40 flex items-center justify-center transition-colors"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" class="w-4 h-4" aria-hidden="true">
            <path d="M3 6h18" />
            <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
            <path d="M19 6l-1 14a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1L5 6" />
            <path d="M10 11v6M14 11v6" />
          </svg>
        </button>
      {/if}
      {#if onclose}
        <button
          type="button"
          onclick={onclose}
          aria-label="Close"
          title="Close"
          class="w-6 h-6 rounded text-slate-400 hover:text-slate-100 hover:bg-ink-800 flex items-center justify-center text-lg leading-none transition-colors"
        >×</button>
      {/if}
    </div>
  </div>

  {#if loader.loading}
    <p class="text-slate-500 text-xs py-4 text-center">Loading…</p>
  {:else}
    <TagList
      {tags}
      onchange={(next) => (tags = next)}
      size="md"
      autocomplete={autocompleteOn}
      reorderable
      searchable
      selectable
      onselectionchange={onSelectionChange}
      bind:clearSelection={clearTagSelection}
    />
  {/if}

  {#if reviewResult}
    <!-- LLM review diff: accept/reject removals + additions, then "Apply" to
         stage them into the tag list (still requires Save tags to persist). -->
    <div class="border border-violet-700/50 rounded bg-violet-950/20 p-2 flex flex-col gap-2 text-xs max-h-[45%] overflow-y-auto">
      <div class="flex items-center justify-between">
        <span class="text-[10px] font-semibold uppercase tracking-wide text-violet-300">LLM review</span>
        <button
          type="button"
          onclick={() => (reviewResult = null)}
          aria-label="Dismiss review"
          class="w-5 h-5 rounded text-slate-400 hover:text-slate-100 flex items-center justify-center leading-none"
        >×</button>
      </div>

      {#if reviewResult.remove.length === 0 && reviewResult.add.length === 0}
        <p class="text-slate-500">No changes suggested — the tags look accurate.</p>
      {/if}

      {#if reviewResult.remove.length}
        <div class="flex flex-col gap-0.5">
          <p class="text-[10px] uppercase tracking-wide text-rose-400/80">Remove</p>
          {#each reviewResult.remove as r (r.tag)}
            <label class="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={acceptRemove.has(r.tag)}
                onchange={() => (acceptRemove = toggleAccept(acceptRemove, r.tag))}
                class="mt-0.5 accent-rose-500"
              />
              <span>
                <span class="line-through text-rose-300">{r.tag}</span>
                <span class="text-slate-500"> — {r.reason}</span>
              </span>
            </label>
          {/each}
        </div>
      {/if}

      {#if reviewResult.add.length}
        <div class="flex flex-col gap-0.5">
          <p class="text-[10px] uppercase tracking-wide text-emerald-400/80">Add</p>
          {#each reviewResult.add as a (a.tag)}
            <label class="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={acceptAdd.has(a.tag)}
                onchange={() => (acceptAdd = toggleAccept(acceptAdd, a.tag))}
                class="mt-0.5 accent-emerald-500"
              />
              <span>
                <span class="text-emerald-300">{a.tag}</span>
                <span class="text-slate-500"> — {a.reason}</span>
              </span>
            </label>
          {/each}
        </div>
      {/if}

      {#if reviewResult.notes.length}
        <details class="text-slate-500">
          <summary class="cursor-pointer text-[10px]">{reviewResult.notes.length} note{reviewResult.notes.length === 1 ? "" : "s"}</summary>
          <ul class="list-disc pl-4 mt-1 space-y-0.5">
            {#each reviewResult.notes as n}<li>{n}</li>{/each}
          </ul>
        </details>
      {/if}

      {#if reviewResult.remove.length || reviewResult.add.length}
        <div class="flex justify-end gap-2 pt-1">
          <button
            type="button"
            onclick={() => (reviewResult = null)}
            class="px-3 py-1 rounded bg-ink-800 hover:bg-ink-700 text-slate-300"
          >Dismiss</button>
          <button
            type="button"
            onclick={applyReview}
            disabled={acceptedCount === 0}
            class="px-3 py-1 rounded bg-violet-600 hover:bg-violet-500 text-white btn-disabled"
          >Apply {acceptedCount} change{acceptedCount === 1 ? "" : "s"}</button>
        </div>
      {/if}
    </div>
  {/if}

  {#if loader.error}
    <p class="text-xs text-red-400 break-all">{loader.error}</p>
  {/if}

  <div class="flex justify-end gap-2">
    {#if selectedTags.length > 0}
      <!-- A tag selection exists: copy it to the clipboard. (Clearing the
           selection is handled in the editor itself — click-away / Esc.) -->
      <button
        type="button"
        onclick={copySelection}
        disabled={copied}
        title={copied
          ? "Copied — select or deselect a tag to copy again"
          : `Copy the ${selectedTags.length} selected tag${selectedTags.length === 1 ? "" : "s"} to the clipboard`}
        class="px-4 py-1.5 text-xs rounded bg-sky-600 hover:bg-sky-500 text-white btn-disabled"
      >{copied ? "Copied ✓" : `Copy ${selectedTags.length}`}</button>
    {:else if tagClipboard.size > 0}
      <!-- Nothing selected here but the clipboard holds tags from another frame:
           offer to append them (deduped) to this frame's tags. Hovering the
           button previews the exact tags that will be pasted. -->
      <div class="relative group">
        <button
          type="button"
          onclick={pasteClipboard}
          disabled={loader.loading || pasteTags.length === 0}
          class="px-4 py-1.5 text-xs rounded bg-sky-600 hover:bg-sky-500 text-white btn-disabled"
        >Paste {pasteTags.length}</button>
        <div
          class="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 z-20 hidden group-hover:block"
        >
          <div class="w-56 max-h-48 overflow-hidden rounded border border-ink-700 bg-ink-900 shadow-lg p-2">
            {#if pasteTags.length}
              <p class="text-[10px] uppercase tracking-wide text-slate-400 mb-1">
                Will paste ({pasteTags.length})
              </p>
              <div class="flex flex-wrap gap-1">
                {#each pasteTags as t}
                  <span class="px-1.5 py-0.5 text-[10px] leading-none rounded-full bg-white/10 text-slate-200">{t}</span>
                {/each}
              </div>
            {:else}
              <p class="text-[10px] text-slate-400">All {tagClipboard.size} copied tag{tagClipboard.size === 1 ? "" : "s"} are already on this frame.</p>
            {/if}
          </div>
        </div>
      </div>
    {/if}
    {#if llmModelSelected}
      <button
        type="button"
        onclick={reviewNow}
        disabled={loader.loading || saving || tagging || reviewing}
        title="Ask the LLM to review these tags against the image (proposes removals and additions)"
        class="px-4 py-1.5 text-xs rounded bg-violet-600 hover:bg-violet-500 text-white btn-disabled"
      >{reviewing ? "Reviewing…" : "Review"}</button>
    {/if}
    <button
      type="button"
      onclick={retagNow}
      disabled={loader.loading || saving || tagging}
      title="Re-run the WD14 tagger on this frame (preserves the description)"
      class="px-4 py-1.5 text-xs rounded bg-emerald-600 hover:bg-emerald-500 text-white btn-disabled"
    >{tagging ? "Tagging…" : "Tag"}</button>
    <button
      type="button"
      onclick={save}
      disabled={loader.loading || saving || !dirty}
      class="px-4 py-1.5 text-xs rounded gradient-accent text-white btn-disabled"
    >{saving ? "Saving…" : "Save tags"}</button>
  </div>
</div>

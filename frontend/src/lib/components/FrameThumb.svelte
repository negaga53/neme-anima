<script lang="ts">
  import * as api from "$lib/api";
  import { colorForSlug } from "$lib/characterColors";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import type { FrameRecord } from "$lib/types";
  import { parseTags, splitSidecar } from "$lib/sidecar";
  import { tagClipboard } from "$lib/stores/tagClipboard.svelte";
  import TagList from "./TagList.svelte";

  type Props = {
    frame: FrameRecord;
    selected: boolean;
    /** Left-click on the image area opens the preview modal. */
    onpreview: () => void;
    /** Middle-click on the image area, OR clicking the toggle pill, toggles
     *  selection. Mods are forwarded so shift-range still works on middle-click. */
    onselect: (mods: { shift: boolean; ctrl: boolean }) => void;
  };
  const { frame, selected, onpreview, onselect }: Props = $props();

  let tagText = $state<string | null>(null);
  let hovered = $state(false);
  // Mirrors `hovered` for keyboard focus: while a TagPill input or the
  // "+" pill is focused inside this tile, the bottom panel must stay
  // mounted even after the cursor leaves the frame. Without it the panel
  // unmounts on mouseleave, the input gets destroyed, and the user's
  // in-progress edit (and its focus) goes with it.
  let focusWithin = $state(false);

  // Show the per-character badge only in "All" view, and only when the
  // project has more than one character — single-character workflows stay
  // visually identical to the pre-multi-character UI. The badge text
  // prefers the character's display name with the slug as a fallback for
  // orphan frames whose slug no longer matches any current character.
  let characterBadge = $derived.by(() => {
    if (viewStore.characterFilter !== "all") return null;
    const chars = projectsStore.active?.characters ?? [];
    if (chars.length <= 1) return null;
    const match = chars.find((c) => c.slug === frame.character_slug);
    return match?.name ?? frame.character_slug;
  });
  // Per-character color so two characters' badges in the same grid don't
  // both read as "the indigo one". Looked up against the project's
  // characters list — orphan slugs fall back to palette[0].
  let characterColor = $derived(
    colorForSlug(frame.character_slug, projectsStore.active?.characters ?? []),
  );
  // Local override so a fresh save through the description modal flips the
  // badge without us having to refetch the whole frames list.
  let hasDescriptionLocal = $state<boolean | null>(null);
  let hasDescription = $derived(hasDescriptionLocal ?? frame.has_description);

  // Reset the local override whenever the underlying prop changes — that
  // catches both filename changes (parent swapped the row) and the
  // server-side flag flipping after a list refresh post-bulk-retag.
  // Also re-runs when either per-filename version counter ticks so the
  // cached sidecar gets dropped after a WD14 retag *or* an LLM describe
  // finishes — the FrameRecord doesn't change for in-place rewrites, only
  // the .txt does, and the badge tooltip reads from this cache.
  $effect(() => {
    void frame.filename;
    void frame.has_description;
    void framesStore.retaggedVersion(frame.filename);
    void framesStore.describedVersion(frame.filename);
    hasDescriptionLocal = null;
    tagText = null; // force a fresh tag fetch on next hover
  });

  // Frames currently in flight for a bulk re-tag / re-describe — render a
  // spinner overlay and swallow clicks so the user can't kick off a preview
  // or toggle selection on a tile that's actively being written to.
  let processing = $derived(framesStore.isProcessing(frame.filename));

  // One-shot pop animation on the description badge. Driven by a per-filename
  // version counter in framesStore that ActionBar bumps when a single-frame
  // LLM describe call completes. We capture the counter the first time we
  // see a given filename so already-described frames don't pop on initial
  // render or when this component instance is reused for a different tile.
  let popping = $state(false);
  let popTimer: ReturnType<typeof setTimeout> | null = null;
  let lastSeenFilename: string | null = null;
  let popBaseline = 0;

  $effect(() => {
    const fn = frame.filename;
    const v = framesStore.describedVersion(fn);
    if (lastSeenFilename !== fn) {
      // First run for this filename — capture baseline silently.
      lastSeenFilename = fn;
      popBaseline = v;
      return;
    }
    if (v > popBaseline) {
      popBaseline = v;
      if (popTimer) clearTimeout(popTimer);
      popping = false;
      // Force a tick so the class toggle restarts the animation when the
      // same frame is popped twice in quick succession.
      requestAnimationFrame(() => { popping = true; });
      popTimer = setTimeout(() => { popping = false; }, 560);
    }
  });

  async function loadTags() {
    if (tagText !== null) return;
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    try {
      const r = await api.getTags(slug, frame.filename);
      tagText = r.text;
    } catch {
      tagText = "";
    }
  }

  async function saveTagsLine(nextDanbooru: string) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const { description } = splitSidecar(tagText ?? "");
    const next = description
      ? `${nextDanbooru}\n${description}`
      : nextDanbooru;
    // Use the server's response so any normalization the route applied
    // (dedupe, whitespace collapse) flows back into the local cache —
    // otherwise the next render would still show the raw text we sent.
    const r = await api.putTags(slug, frame.filename, next);
    tagText = r.text;
    framesStore.setSidecarFlags(frame.filename, {
      has_tags: Boolean(splitSidecar(r.text).danbooru),
    });
  }

  // While the hover/focus tag panel is up, Ctrl/Cmd+V appends the shared tag
  // clipboard to this frame (auto-saved). The modal's other selection-based
  // shortcuts need a selection this surface doesn't have. Skipped while a tag
  // input is focused so the input's own text-paste wins.
  $effect(() => {
    if (!(hovered || focusWithin)) return;
    function onKeydown(e: KeyboardEvent) {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "v") return;
      if (!hovered) return; // paste targets the tile under the pointer (not a focus-only tile)
      const el = document.activeElement as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      if (tagClipboard.size === 0) return;
      const merged = parseTags([...realPills, ...tagClipboard.tags].join(", "));
      if (merged.length === realPills.length) return; // all duplicates → no-op
      e.preventDefault();
      void saveTagsLine(merged.join(", "));
    }
    document.addEventListener("keydown", onKeydown);
    return () => document.removeEventListener("keydown", onKeydown);
  });

  // When the frame has a crop derivative, the grid shows the cropped pixels in
  // place of the original (selection/sidecar still key off the original
  // filename). `cropVersion` busts the browser cache after a re-crop, which
  // overwrites `<name>_crop.png` at the same URL.
  let imageUrl = $derived.by(() => {
    const proj = projectsStore.active;
    if (!proj) return "";
    if (frame.has_crop) {
      const v = framesStore.cropVersion(frame.filename);
      const base = api.frameImageUrl(proj.slug, frame.filename + api.CROP_SUFFIX);
      return v > 0 ? `${base}?v=${v}` : base;
    }
    return api.frameImageUrl(proj.slug, frame.filename);
  });

  // Dedupe defensively — the server normalizes on write, but pre-existing
  // sidecars may still have duplicates and the keyed `{#each}` requires unique
  // keys. `parseTags` handles split + trim + dedupe.
  let realPills = $derived(parseTags(splitSidecar(tagText ?? "").danbooru));

  let autocompleteOn = $derived(projectsStore.active?.tag_autocomplete ?? true);

  // Tooltip text for the top-left "described" badge. Tags get loaded the
  // first time the thumb is hovered (which always happens before/while the
  // user reaches the small badge), so by the time the native tooltip kicks
  // in we usually have the real description ready. The fallback strings
  // cover the brief window before the fetch resolves and the rare case of
  // an empty 2nd line slipping past has_description.
  let badgeTitle = $derived.by(() => {
    if (!hasDescription) return "";
    if (tagText === null) return "Loading description…";
    const desc = splitSidecar(tagText).description;
    return desc || "Has description";
  });

  // Auto-save surface: every edit/add/delete/(reorder—disabled here) from the
  // shared TagList fires the save round-trip immediately. saveTagsLine takes a
  // comma-joined danbooru line and writes back the server-normalized response
  // into tagText. The HTTP roundtrip even for an exact-duplicate add is
  // intentional (server dedupes; response keeps the keyed list in sync).
  function onTagsChange(next: string[]) {
    void saveTagsLine(next.join(", "));
  }

  // Middle-click (auxclick with button === 1) on the image toggles selection.
  // We handle it via mousedown to suppress the browser's auto-scroll cursor.
  function onMouseDown(ev: MouseEvent) {
    if (ev.button === 1) {
      ev.preventDefault();
      onselect({ shift: ev.shiftKey, ctrl: ev.ctrlKey || ev.metaKey });
    }
  }

  // Left click → preview. Shift-click left button still calls onselect so
  // the shift-range bulk-select shortcut keeps working without forcing the
  // user onto the middle button.
  function onMainClick(ev: MouseEvent) {
    if (ev.button !== 0) return;
    if (ev.shiftKey || ev.ctrlKey || ev.metaKey) {
      onselect({ shift: ev.shiftKey, ctrl: ev.ctrlKey || ev.metaKey });
      return;
    }
    onpreview();
  }
</script>

<!-- Wrapper div lets the toggle pill and the action pills live as real
     <button> siblings of the image button instead of nesting inside it
     (invalid HTML). -->
<div
  class="relative aspect-[3/4] group"
  onmouseenter={() => { hovered = true; void loadTags(); }}
  onmouseleave={() => (hovered = false)}
  onfocusin={() => { focusWithin = true; void loadTags(); }}
  onfocusout={(e) => {
    // relatedTarget is where focus is heading. If it's still somewhere
    // inside this tile (e.g. tabbing between tag pills), don't tear the
    // panel down. Otherwise the bottom panel collapses naturally.
    const next = e.relatedTarget as Node | null;
    if (!next || !(e.currentTarget as Node).contains(next)) {
      focusWithin = false;
    }
  }}
  role="presentation"
>
  <button
    type="button"
    class="absolute inset-0 rounded-lg overflow-hidden cursor-pointer transition-transform duration-150 hover:scale-[1.02] hover:z-10
      {selected ? 'shadow-[0_0_20px_rgba(99,102,241,0.4)]' : 'shadow-md'}"
    onclick={onMainClick}
    onmousedown={onMouseDown}
    onauxclick={(e) => { if (e.button === 1) e.preventDefault(); }}
    aria-label="Open frame {frame.filename} preview"
  >
    <!-- draggable=false stops the browser's native image-drag: holding the
         mouse button a beat too long on a thumb would otherwise start dragging
         a ghost of the image, which feels like an accidental move and can
         swallow the intended click. There's no drag-to-reorder feature here. -->
    <img src={imageUrl} alt="" draggable="false" class="w-full h-full select-none {viewStore.fitContain ? 'object-contain bg-black' : 'object-cover'}" loading="lazy" />

    {#if selected}
      <!-- Inset border overlay drawn on top of the image, with a faint tint
           so the selection state is unmissable on any image. -->
      <span class="absolute inset-0 rounded-lg border-[3px] border-accent-500 bg-accent-500/20 pointer-events-none"></span>
    {/if}
  </button>

  <!-- Top-LEFT "described" indicator: a small chat-bubble shown only when the
       .txt sidecar already has a non-empty 2nd line. Semitransparent so it
       reads as a soft glance-cue rather than competing with the image. We
       render it as a <button> so the native browser tooltip works on hover
       (it can't on a pointer-events-none span). Clicking it opens the
       description editor — same action as the bottom action-pill, but
       reachable directly from the badge for users who already see it. -->
  {#if hasDescription}
    <span
      title={badgeTitle}
      aria-label="Has description"
      class="absolute top-1.5 left-1.5 w-5 h-5 rounded-full bg-accent-500/40 text-white/90 flex items-center justify-center z-20 cursor-help {popping ? 'badge-pop' : ''}"
    >
      <svg viewBox="0 0 16 16" class="w-3 h-3" fill="currentColor" aria-hidden="true">
        <path d="M3 3h10a1.5 1.5 0 0 1 1.5 1.5v6A1.5 1.5 0 0 1 13 12H7.4l-3 2.4a.5.5 0 0 1-.8-.4V12H3a1.5 1.5 0 0 1-1.5-1.5v-6A1.5 1.5 0 0 1 3 3z"/>
      </svg>
    </span>
  {/if}

  <!-- Video stem badge, hover-only. Sits at the top-left, offset right of
       the description badge if present so the two don't collide. -->
  <span
    class="absolute top-1.5 px-1.5 py-0.5 text-[9px] bg-black/60 backdrop-blur-sm rounded text-white opacity-0 transition-opacity z-20 pointer-events-none {hovered ? 'opacity-100' : ''}"
    style="left: {hasDescription ? 28 : 6}px;"
  >
    {frame.video_stem}
  </span>

  <!-- Character badge (bottom-left, "All" view only). Always visible in
       multi-character projects so the user can scan owners across the grid
       without hovering each tile. Hidden in single-character projects and
       in per-character filters where every visible tile has the same owner. -->
  {#if characterBadge}
    <span
      class="absolute bottom-1.5 left-1.5 px-1.5 py-0.5 text-[9px] {characterColor.bgSoft} backdrop-blur-sm rounded text-white z-20 pointer-events-none shadow-[0_1px_4px_rgba(0,0,0,0.4)]"
      title="Routed to {characterBadge}"
    >
      {characterBadge}
    </span>
  {/if}

  <!-- Spinner overlay: covers the whole tile during a bulk re-tag /
       re-describe. Sits above every other badge (z-30) and absorbs pointer
       events so the image button underneath can't fire while busy. The dim
       backdrop is just enough to make the spinner read against any image. -->
  {#if processing}
    <div
      class="absolute inset-0 rounded-lg bg-ink-950/55 backdrop-blur-[1px] flex items-center justify-center z-30"
      role="status"
      aria-live="polite"
      aria-label="Processing frame"
    >
      <span
        class="block w-7 h-7 rounded-full border-2 border-white/25 border-t-accent-500 animate-spin"
        aria-hidden="true"
      ></span>
    </div>
  {/if}

  <!-- Top-RIGHT toggle pill: emerald + checkmark when selected, neutral
       otherwise. Always visible; on top of the image button via z-20. -->
  <button
    type="button"
    onclick={(e) => {
      e.stopPropagation();
      onselect({ shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey });
    }}
    title={selected ? "Deselect" : "Select"}
    aria-label={selected ? "Deselect frame" : "Select frame"}
    aria-pressed={selected}
    class="absolute top-1.5 right-1.5 w-6 h-6 rounded-full text-xs leading-none flex items-center justify-center z-20 transition-all border
      {selected
        ? 'bg-emerald-500 border-emerald-300 text-white shadow-[0_0_10px_rgba(16,185,129,0.6)] opacity-100'
        : 'bg-black/60 border-white/30 text-white/80 hover:bg-black/80 opacity-0 group-hover:opacity-100 focus:opacity-100'}"
  >{selected ? "✓" : "○"}</button>

  <!-- Bottom hover panel: tag pills + (+) pill + description pill, all
       inline. Sits above the image button (z-20) so clicks don't bubble
       into the preview-open handler. The TagPills' own e.stopPropagation
       handles the rest. Stays mounted while focus is inside the tile so
       the user's tag edit isn't torn down when the cursor strays out. -->
  {#if hovered || focusWithin}
    <div
      class="absolute inset-x-0 bottom-0 max-h-full overflow-y-auto p-1.5 pt-6 flex flex-wrap gap-1 items-center z-20
        bg-gradient-to-t from-black/90 via-black/75 to-transparent pointer-events-none"
    >
      <!-- pointer-events-none on the gradient + auto on each pill keeps the
           dim area between pills from intercepting clicks meant for the
           image preview. -->
      <div class="contents pointer-events-auto">
        <!-- Auto-save surface: TagList renders the dense pills + "+" and fires
             onTagsChange (immediate save) on every edit. Drag/search/select
             stay off so the grid hover panel keeps its minimal shape. -->
        <TagList
          tags={realPills}
          onchange={onTagsChange}
          size="sm"
          autocomplete={autocompleteOn}
        />
      </div>
    </div>
  {/if}
</div>

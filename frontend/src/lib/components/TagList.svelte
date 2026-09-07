<script lang="ts">
  import {
    matchesQuery,
    moveSelection,
    movingIndices,
    rangeBetween,
    dropIndexAtPoint,
    type SimpleRect,
  } from "$lib/tagList";
  import { tagClipboard } from "$lib/stores/tagClipboard.svelte";
  import { writeSystemClipboard } from "$lib/tagClipboard";
  import TagPill from "./TagPill.svelte";

  type Props = {
    tags: string[];
    onchange: (next: string[]) => void;
    size?: "sm" | "md";
    autocomplete?: boolean;
    reorderable?: boolean;
    searchable?: boolean;
    selectable?: boolean;
    onselectionchange?: (selectedTags: string[]) => void;
    clearSelection?: () => void;
  };
  let {
    tags,
    onchange,
    size = "sm",
    autocomplete = false,
    reorderable = false,
    searchable = false,
    selectable = false,
    onselectionchange,
    clearSelection = $bindable(),
  }: Props = $props();

  // Rich mode = the crop modal (drives selection/edit/drag/keyboard). The grid
  // hover passes none of these flags → simple mode (self-managing TagPill).
  let rich = $derived(selectable);

  const PLACEHOLDER = " __new__"; // simple-mode "+" draft sentinel

  let search = $state("");

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

  // ===== simple mode (grid hover) =====
  let addingTag = $state(false);
  let pills = $derived(addingTag ? [...tags, PLACEHOLDER] : tags);

  function replaceTag(oldTag: string, next: string) {
    onchange(dedupe(tags.map((t) => (t === oldTag ? next : t))));
  }
  function deleteTag(oldTag: string) {
    onchange(tags.filter((t) => t !== oldTag));
  }
  function addTag(next: string) {
    addingTag = false;
    const trimmed = next.trim();
    if (!trimmed) return;
    onchange(dedupe([...tags, trimmed]));
  }

  // ===== rich mode (crop modal) =====
  let containerEl = $state<HTMLDivElement | null>(null);
  let selected = $state<Set<number>>(new Set());
  let editingIndex = $state<number | null>(null);
  let inserting = $state<number | null>(null);
  let draftSeq = $state(0);
  let dropIndex = $state<number | null>(null);
  // Non-reactive gesture/selection scratch.
  let anchor: number | null = null;
  let pressRef: { index: number; inSel: boolean; sole: boolean } | null = null;
  let didDrag = false;
  // Reactive mirror of "a drag is in progress", used only for the drag-ghost
  // opacity in the template (didDrag stays non-reactive — it is a click latch).
  let dragging = $state(false);
  let gesture: { mode: "pending" | "move"; startX: number; startY: number; dragIndex: number } | null = null;

  const DRAG_THRESHOLD = 4; // px before a press becomes a drag

  // Surface the selection (as tag text) to the parent. `selectedTags` is a
  // fresh array each recompute, so the parent's onselectionchange MUST treat an
  // equal selection as a no-op (TagEditorPanel.onSelectionChange does) — else
  // effect → parent write → effect re-run loops (effect_update_depth_exceeded).
  let selectedTags = $derived(
    rich
      ? [...selected]
          .filter((i) => i >= 0 && i < tags.length)
          .sort((a, b) => a - b)
          .map((i) => tags[i])
      : [],
  );
  $effect(() => {
    if (selectable) onselectionchange?.(selectedTags);
  });

  // Imperative clear for the parent (bind:clearSelection). Identity churn here
  // is why the parent reads it under untrack() in its per-filename reload effect.
  clearSelection = () => {
    selected = new Set();
    anchor = null;
  };

  // Click-away OUTSIDE the editor clears the selection. Attached only while
  // something is selected, so it is a no-op the rest of the time.
  $effect(() => {
    if (!rich || selected.size === 0) return;
    function onDocPointerDown(e: PointerEvent) {
      if (containerEl && !containerEl.contains(e.target as Node)) {
        selected = new Set();
      }
    }
    document.addEventListener("pointerdown", onDocPointerDown);
    return () => document.removeEventListener("pointerdown", onDocPointerDown);
  });

  // Tear down any window drag listeners if the editor unmounts mid-gesture
  // (e.g. the modal closes between pointerdown and pointerup).
  $effect(() => () => {
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
  });

  function pillRects(): Array<{ index: number; rect: SimpleRect }> {
    const out: Array<{ index: number; rect: SimpleRect }> = [];
    containerEl?.querySelectorAll<HTMLElement>("[data-tag-index]").forEach((el) => {
      const idx = Number(el.dataset.tagIndex);
      const r = el.getBoundingClientRect();
      out.push({ index: idx, rect: { left: r.left, top: r.top, right: r.right, bottom: r.bottom } });
    });
    return out;
  }
  function computeDrop(x: number, y: number): number {
    return dropIndexAtPoint(pillRects(), x, y, tags.length);
  }

  function onPointerMove(e: PointerEvent) {
    const g = gesture;
    if (!g) return;
    const dx = Math.abs(e.clientX - g.startX);
    const dy = Math.abs(e.clientY - g.startY);
    if (g.mode === "pending") {
      if (dx < DRAG_THRESHOLD && dy < DRAG_THRESHOLD) return;
      g.mode = "move";
      didDrag = true;
      dragging = true;
    }
    if (g.mode === "move") dropIndex = computeDrop(e.clientX, e.clientY);
  }
  function onPointerUp(e: PointerEvent) {
    const g = gesture;
    if (g && g.mode === "move") {
      const drop = computeDrop(e.clientX, e.clientY);
      const sel = movingIndices(selected, g.dragIndex);
      const { next, selection } = moveSelection(tags, sel, drop);
      onchange(next);
      selected = new Set(selection);
    }
    endGesture();
  }
  function beginGesture(x: number, y: number, dragIndex: number) {
    didDrag = false;
    gesture = { mode: "pending", startX: x, startY: y, dragIndex };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }
  function endGesture() {
    gesture = null;
    dragging = false;
    dropIndex = null;
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
  }

  // Select on pointer-down so the pressed pill is the visible subject of a drag.
  function onPillPointerDown(index: number, e: PointerEvent) {
    if (e.button !== 0 || editingIndex === index) return;
    // Focus first: this blurs any editing pill so its blur-commit runs BEFORE
    // we overwrite the selection below (otherwise the commit would clear it).
    containerEl?.focus();
    if (e.metaKey || e.ctrlKey) {
      const next = new Set(selected);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      selected = next;
      anchor = index;
      pressRef = null;
    } else if (e.shiftKey && anchor != null) {
      selected = new Set(rangeBetween(anchor, index));
      pressRef = null;
    } else {
      const inSel = selected.has(index);
      const sole = selected.size === 1 && inSel;
      if (!inSel) selected = new Set([index]); // exclusive select on press
      anchor = index;
      pressRef = { index, inSel, sole };
    }
    if (reorderable) beginGesture(e.clientX, e.clientY, index);
  }

  // No-drag click resolution: collapse a multi-selection to the clicked pill, or
  // toggle a sole selection back off (Ctrl/Cmd/Shift were handled on press).
  function onPillClick(index: number) {
    if (didDrag) {
      didDrag = false;
      pressRef = null;
      return;
    }
    const press = pressRef;
    pressRef = null;
    if (editingIndex === index || !press || press.index !== index) return;
    if (press.inSel) selected = press.sole ? new Set() : new Set([index]);
  }

  function onPillDblClick(index: number) {
    if (editingIndex === index) return;
    editingIndex = index;
    inserting = null;
    selected = new Set();
  }

  function commitEdit(index: number, next: string, chain: boolean) {
    const result = dedupe(tags.map((t, i) => (i === index ? next : t)));
    onchange(result);
    selected = new Set();
    editingIndex = null;
    // Validating the last tag chains a fresh draft for fast entry.
    if (chain && index === tags.length - 1) {
      inserting = result.length;
      draftSeq += 1;
    }
  }
  function deleteAt(index: number) {
    onchange(tags.filter((_, i) => i !== index));
    editingIndex = null;
    selected = new Set();
  }
  function splitPasteAt(index: number, parts: string[]) {
    const [first, ...rest] = parts;
    const updated = [...tags];
    updated[index] = first;
    const withRest = [...updated.slice(0, index + 1), ...rest, ...updated.slice(index + 1)];
    onchange(dedupe(withRest));
    editingIndex = null;
    selected = new Set();
  }

  // "+" opens a draft at the end. Enter on the draft chains another draft.
  function startInsert() {
    inserting = tags.length;
    draftSeq += 1;
    editingIndex = null;
    selected = new Set();
  }
  function cancelInsert() {
    inserting = null;
  }
  function commitInsert(next: string, chain: boolean) {
    if (inserting == null) return;
    const at = inserting;
    const result = dedupe([...tags.slice(0, at), next, ...tags.slice(at)]);
    onchange(result);
    selected = new Set();
    if (chain && at === tags.length) {
      inserting = result.length;
      draftSeq += 1;
    } else {
      inserting = null;
    }
  }
  function splitInsert(parts: string[]) {
    if (inserting == null) return;
    const at = inserting;
    onchange(dedupe([...tags.slice(0, at), ...parts, ...tags.slice(at)]));
    inserting = null;
  }

  function onContainerKeydown(e: KeyboardEvent) {
    if (editingIndex != null || inserting != null) return; // a pill owns its keys
    const mod = e.metaKey || e.ctrlKey;
    const k = e.key.toLowerCase();
    if (mod && k === "a") {
      e.preventDefault();
      selected = new Set(tags.map((_, i) => i));
      return;
    }
    // Bound to in-range indices: a selection can outlive a parent list-shrink
    // (apply-review / paste) that didn't clear it — never map a stale index.
    const sel = [...selected].filter((i) => i >= 0 && i < tags.length).sort((a, b) => a - b);
    if (sel.length === 0) return;
    const selSet = new Set(sel);
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      onchange(tags.filter((_, i) => !selSet.has(i)));
      selected = new Set();
    } else if (mod && k === "c") {
      e.preventDefault();
      const copied = sel.map((i) => tags[i]);
      tagClipboard.set(copied);
      writeSystemClipboard(tagClipboard.text);
    } else if (mod && k === "x") {
      e.preventDefault();
      const copied = sel.map((i) => tags[i]);
      tagClipboard.set(copied);
      writeSystemClipboard(tagClipboard.text);
      onchange(tags.filter((_, i) => !selSet.has(i)));
      selected = new Set();
    } else if (mod && k === "v") {
      e.preventDefault();
      const clip = tagClipboard.tags;
      if (clip.length) {
        const at = sel[sel.length - 1] + 1;
        onchange(dedupe([...tags.slice(0, at), ...clip, ...tags.slice(at)]));
        selected = new Set();
      }
    } else if (e.key === "Escape") {
      selected = new Set();
    }
  }

  function onContainerClick(e: MouseEvent) {
    if (didDrag) {
      didDrag = false;
      return;
    }
    const t = e.target as HTMLElement;
    if (t.closest("[data-tag-index]") || t.closest("input") || t.closest("button")) return;
    selected = new Set(); // empty-space click clears the selection
  }

  function ringClass(i: number, tag: string): string {
    if (selected.has(i)) return "ring-2 ring-sky-400";
    if (searchable && matchesQuery(tag, search)) return "ring-2 ring-amber-400";
    return "";
  }

  let markerCls = $derived(
    size === "md"
      ? "w-0.5 self-stretch min-h-[1.5rem] bg-amber-400 rounded-full"
      : "w-0.5 self-stretch min-h-[1rem] bg-amber-400 rounded-full",
  );
  let addBtnCls = $derived(
    size === "md"
      ? "px-2 py-0.5 text-[11.25px] leading-none rounded-full bg-white/15 hover:bg-white/25 text-white border border-white/10 transition-colors self-start"
      : "px-2 py-0.5 text-[9.5px] leading-none rounded-full bg-white/15 hover:bg-white/25 text-white border border-white/10 transition-colors",
  );
</script>

{#snippet draft()}
  {#key draftSeq}
    <span class="inline-flex">
      <TagPill
        text=""
        editing
        {size}
        {autocomplete}
        existingTags={tags}
        oncommit={(next, chain) => commitInsert(next, chain)}
        ondelete={cancelInsert}
        oncancel={cancelInsert}
        onsplitpaste={(parts) => splitInsert(parts)}
      />
    </span>
  {/key}
{/snippet}

{#if searchable}
  <!-- Search: highlights matching tags in place (no filtering). -->
  <div class="relative mt-1.5">
    <input
      bind:value={search}
      placeholder="Search tags to highlight…"
      class="w-full pl-2 pr-7 py-1 text-xs bg-ink-950 border border-ink-700 rounded focus:outline-none focus:border-accent-500"
    />
    {#if search}
      <button
        type="button"
        onclick={() => (search = "")}
        aria-label="Clear search"
        class="absolute right-1 top-1/2 -translate-y-1/2 w-5 h-5 rounded text-slate-400 hover:text-slate-100 flex items-center justify-center"
      >×</button>
    {/if}
  </div>
{/if}

{#if rich}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div
    bind:this={containerEl}
    data-testid="tag-editor-surface"
    role="listbox"
    tabindex="0"
    onkeydown={onContainerKeydown}
    onclick={onContainerClick}
    class="flex flex-wrap gap-1.5 content-start overflow-y-auto flex-1 min-h-0 pr-1 outline-none"
  >
    {#each tags as tag, i (i + ":" + tag)}
      {#if inserting === i}{@render draft()}{/if}
      {#if reorderable && dropIndex === i}
        <span class={markerCls} aria-hidden="true"></span>
      {/if}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <span
        data-tag-index={i}
        onpointerdown={(e) => onPillPointerDown(i, e)}
        onclick={() => onPillClick(i)}
        ondblclick={() => onPillDblClick(i)}
        class="inline-flex rounded-full transition-shadow {ringClass(i, tag)}
          {reorderable && editingIndex !== i ? 'cursor-grab active:cursor-grabbing' : ''}
          {dragging && selected.has(i) ? 'opacity-40' : ''}"
      >
        <TagPill
          text={tag}
          editing={editingIndex === i}
          {size}
          {autocomplete}
          existingTags={tags}
          oncommit={(next, chain) => commitEdit(i, next, chain)}
          ondelete={() => deleteAt(i)}
          oncancel={() => (editingIndex = null)}
          onsplitpaste={(parts) => splitPasteAt(i, parts)}
        />
      </span>
    {/each}
    {#if inserting === tags.length}{@render draft()}{/if}
    {#if reorderable && dropIndex === tags.length}
      <span class={markerCls} aria-hidden="true"></span>
    {/if}
    <button
      type="button"
      onclick={(e) => { e.stopPropagation(); startInsert(); }}
      title="Add tag"
      aria-label="Add tag"
      class={addBtnCls}
    >+</button>
  </div>
{:else}
  <!-- Simple mode (grid hover): self-managing TagPill, single-click edits,
       no selection / drag. Unchanged from the pre-port behavior. -->
  <div class={size === "md" ? "flex flex-wrap gap-1.5 content-start overflow-y-auto flex-1 min-h-0 pr-1" : "contents"}>
    {#each pills as p, i (p === PLACEHOLDER ? `__new__${i}` : p)}
      {#if p === PLACEHOLDER}
        <TagPill
          text=""
          startEditing
          {size}
          {autocomplete}
          existingTags={tags}
          onreplace={(next) => addTag(next)}
          oncancel={() => { addingTag = false; }}
        />
      {:else}
        <TagPill
          text={p}
          {size}
          {autocomplete}
          existingTags={tags}
          onreplace={(next) => replaceTag(p, next)}
          ondelete={() => deleteTag(p)}
        />
      {/if}
    {/each}
    <button
      type="button"
      onclick={() => (addingTag = true)}
      title="Add tag"
      aria-label="Add tag"
      class={addBtnCls}
    >+</button>
  </div>
{/if}

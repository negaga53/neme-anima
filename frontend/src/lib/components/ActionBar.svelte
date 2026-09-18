<script lang="ts">
  import * as api from "$lib/api";
  import { colorForIndex } from "$lib/characterColors";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { toasts } from "$lib/stores/toasts.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import type { FrameRecord } from "$lib/types";
  import { getFrameOverwriteConfirm } from "$lib/frameOverwriteContext";
  import { runBulkRetag } from "$lib/bulkRetag";
  import { tagActions, describeActions } from "$lib/bulkRetagActions";

  type Props = {
    onopenRegex: () => void;
  };
  const { onopenRegex }: Props = $props();

  const confirmFrameOverwrite = getFrameOverwriteConfirm();

  // Multi-character UI is only relevant when the project actually has more
  // than one character — single-character projects keep the pre-multi-
  // character action set unchanged.
  let characters = $derived(projectsStore.active?.characters ?? []);
  let multiCharacter = $derived(characters.length > 1);

  // Single combined "Characters ▾" dropdown replaces the prior pair.
  // Per-row UX:
  //   - Click the row name → MOVE selected frames to that character
  //     (single-owner reassignment, the common "fix a misroute" case).
  //   - Click the small "+ Also" affordance on the right → ALSO ASSIGN
  //     (duplicate so the frame lives under both characters with their
  //     own captions).
  //   - The character whose slug every selected frame already shares
  //     shows a "● current" badge and is non-clickable for both actions
  //     — no point moving frames to where they already are.
  // For mixed selections, "current" is per-frame and we don't surface a
  // single badge — we just enable both actions for every character.
  let charactersOpen = $state(false);
  let menuBusy = $state(false);

  /** The slug shared by every selected frame, or null if mixed/unknown.
   *  Used to highlight "● current" in the dropdown so the user can see
   *  which row is a no-op before they click. */
  let sharedOwnerSlug = $derived.by(() => {
    void framesStore.selectionVersion;
    const sel = framesStore.selection.selected();
    if (sel.size === 0) return null;
    const slugs = framesStore.items
      .filter((it) => sel.has(it.filename))
      .map((it) => it.character_slug);
    if (slugs.length === 0) return null;
    const first = slugs[0];
    return slugs.every((s) => s === first) ? first : null;
  });

  function currentFilterQuery(): string | undefined {
    if (viewStore.characterFilter === "all") return undefined;
    if (viewStore.characterFilter === "unsorted") return "__unsorted__";
    return viewStore.characterFilter;
  }

  async function refreshAfterAssignment(slug: string) {
    await framesStore.refresh(slug, {
      source: viewStore.sourceFilter ?? undefined,
      query: viewStore.tagQuery || undefined,
      characterSlug: currentFilterQuery(),
    });
  }

  async function moveSelectedTo(targetSlug: string) {
    const slug = projectsStore.active?.slug;
    if (!slug || menuBusy) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) {
      charactersOpen = false;
      return;
    }
    menuBusy = true;
    try {
      await api.bulkMoveFrames(slug, filenames, targetSlug);
      await refreshAfterAssignment(slug);
      framesStore.clear();
    } catch (e) {
      toasts.error(`Move failed: ${e}`);
    } finally {
      menuBusy = false;
      charactersOpen = false;
    }
  }

  async function alsoAssignSelectedTo(targetSlug: string) {
    const slug = projectsStore.active?.slug;
    if (!slug || menuBusy) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) {
      charactersOpen = false;
      return;
    }
    menuBusy = true;
    try {
      const res = await api.bulkDuplicateFrames(slug, filenames, targetSlug);
      // Originals stay; duplicates appear. Refresh shows them inline
      // under "All" or the target's filter.
      await refreshAfterAssignment(slug);
      if (res.missing.length > 0) {
        toasts.info(
          `Also-assigned ${res.duplicated.length}; ` +
          `${res.missing.length} skipped (no metadata).`,
        );
      }
    } catch (e) {
      toasts.error(`Also-assign failed: ${e}`);
    } finally {
      menuBusy = false;
      charactersOpen = false;
    }
  }

  let count = $derived.by(() => {
    framesStore.selectionVersion; // reactive dependency
    return framesStore.selection.count();
  });

  let onFramesTab = $derived(viewStore.tab === "frames");
  let total = $derived(framesStore.items.length);
  // When a tag query is active, the count badge renders "X / Y" — Y is the
  // unfiltered count for the current source/kept_only view, served by the
  // backend alongside the filtered list.
  let totalInView = $derived(framesStore.totalInView);
  let queryActive = $derived(viewStore.tagQuery.trim().length > 0);
  let allSelected = $derived(count > 0 && count === total);

  // The LLM describe button only renders when the project has a model picked —
  // matches Settings logic so a user can't fire a doomed describe against an
  // unconfigured endpoint.
  let llmModelSelected = $derived(!!projectsStore.active?.llm?.model);

  let retagBusy = $state(false);
  let exportBusy = $state(false);

  function selectedItems(filenames: string[]): FrameRecord[] {
    const selected = new Set(filenames);
    return framesStore.items.filter((it) => selected.has(it.filename));
  }

  function countExistingTags(filenames: string[]): number {
    return selectedItems(filenames).filter((it) => it.has_tags).length;
  }

  function countExistingDescriptions(filenames: string[]): number {
    return selectedItems(filenames).filter((it) => it.has_description).length;
  }

  async function deleteSelected() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    if (!confirm(`Delete ${filenames.length} frame${filenames.length === 1 ? "" : "s"}?`)) return;
    await api.bulkDeleteFrames(slug, filenames);
    framesStore.removeLocal(filenames);
  }

  async function retagDanbooru() {
    const slug = projectsStore.active?.slug;
    if (!slug || retagBusy) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    const affectedCount = countExistingTags(filenames);
    if (
      affectedCount > 0 &&
      !(await confirmFrameOverwrite("retag", filenames.length, affectedCount))
    ) {
      return;
    }
    await runRetagDanbooru(slug, filenames);
  }

  async function runRetagDanbooru(slug: string, filenames: string[]) {
    retagBusy = true;
    try {
      await runBulkRetag("tag", slug, filenames, tagActions);
    } finally {
      retagBusy = false;
    }
  }

  async function retagLLM() {
    const slug = projectsStore.active?.slug;
    if (!slug || retagBusy) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    const affectedCount = countExistingDescriptions(filenames);
    if (
      affectedCount > 0 &&
      !(await confirmFrameOverwrite("describe", filenames.length, affectedCount))
    ) {
      return;
    }
    await runRetagLLM(slug, filenames);
  }

  async function runRetagLLM(slug: string, filenames: string[]) {
    retagBusy = true;
    try {
      await runBulkRetag("describe", slug, filenames, describeActions);
    } finally {
      retagBusy = false;
    }
  }

  async function exportSelected() {
    const slug = projectsStore.active?.slug;
    if (!slug || exportBusy) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    exportBusy = true;
    try {
      const blob = await api.bulkExportFrames(slug, filenames);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${slug}-export.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      toasts.error(`Export failed: ${e}`);
    } finally {
      exportBusy = false;
    }
  }

  function toggleSelectAll() {
    if (allSelected) framesStore.clear();
    else framesStore.selectAll();
  }

  // ---- tag search (right of the frames-count badge) ----
  // The committed value lives in viewStore.tagQuery (FramesTab depends on
  // it to refresh the list). The input maintains its own draft and pushes
  // to the store after a short debounce so each keystroke doesn't trigger
  // a fresh /api/frames call.
  let queryDraft = $state(viewStore.tagQuery);
  let queryTimer: ReturnType<typeof setTimeout> | null = null;
  const QUERY_DEBOUNCE_MS = 250;

  // Sync inbound changes (e.g. project switch resets the query) into the
  // draft. Crucially, we only READ viewStore.tagQuery here so that's the
  // sole dependency — touching queryDraft inside the effect would track
  // it too, and every keystroke from the input handler would re-fire the
  // effect and overwrite what the user just typed before the debounce
  // could push it to the store. Equal writes are no-ops in Svelte 5,
  // so the unconditional assignment is safe.
  $effect(() => {
    queryDraft = viewStore.tagQuery;
  });

  function onQueryInput(ev: Event) {
    queryDraft = (ev.target as HTMLInputElement).value;
    if (queryTimer) clearTimeout(queryTimer);
    queryTimer = setTimeout(() => {
      viewStore.tagQuery = queryDraft;
    }, QUERY_DEBOUNCE_MS);
  }

  function clearQuery() {
    if (queryTimer) clearTimeout(queryTimer);
    queryDraft = "";
    viewStore.tagQuery = "";
  }
</script>

<!-- Order, left → right: selected-pill (visible when count > 0), Select all,
     N frames. The purple pill leads so the destructive cluster sits at the
     visual edge of the row, with the static select/count on its right. -->
{#if onFramesTab}
  <div class="flex items-center gap-2 h-7">
    {#if count > 0}
      <!-- Action chips inside the purple selection pill carry tinted
           translucent backgrounds so each action is distinct at a glance
           while still reading as part of the cluster:
             Delete  → red (destructive, conventional warning hue)
             Regex…  → amber (text-edit / "transform the tag line")
             Tag     → emerald (write/rewrite from a model)
             Describe → teal (also model-driven, sibling colour to emerald)
           All sit on the purple gradient via /30 alpha so the parent pill
           still reads as a single unit. -->
      <div class="gradient-accent text-white h-7 pl-3 pr-1 rounded-full inline-flex items-center gap-1.5 text-xs font-medium border border-white/10 shadow-[0_2px_12px_rgba(99,102,241,0.4)]">
        <span class="bg-white/20 px-2 py-0.5 rounded-full leading-none">{count} selected</span>
        <button
          type="button"
          onclick={deleteSelected}
          class="bg-red-500/35 hover:bg-red-500/60 rounded-full px-2.5 h-5 transition-colors inline-flex items-center"
        >Delete</button>
        <button
          type="button"
          onclick={onopenRegex}
          class="bg-amber-400/30 hover:bg-amber-400/55 rounded-full px-2.5 h-5 transition-colors inline-flex items-center"
        >Regex…</button>
        <button
          type="button"
          onclick={retagDanbooru}
          disabled={retagBusy}
          title="Run WD14 tagger on selected frames (preserves the LLM description line)"
          class="bg-emerald-500/30 hover:bg-emerald-500/55 rounded-full px-2.5 h-5 transition-colors inline-flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
        >Tag</button>
        {#if llmModelSelected}
          <button
            type="button"
            onclick={retagLLM}
            disabled={retagBusy}
            title="Run LLM description on selected frames (preserves WD14 tags)"
            class="bg-teal-500/30 hover:bg-teal-500/55 rounded-full px-2.5 h-5 transition-colors inline-flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          >Describe</button>
        {/if}
        <button
          type="button"
          onclick={exportSelected}
          disabled={exportBusy}
          title="Download selected frames as a zip of image + caption pairs"
          class="bg-sky-500/30 hover:bg-sky-500/55 rounded-full px-2.5 h-5 transition-colors inline-flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
        >Export</button>
        {#if multiCharacter}
          <!-- Single "Characters ▾" dropdown unifies move + also-assign.
               Inside, each row exposes both actions:
                 - clicking the row name MOVES (single-owner reassignment)
                 - clicking the small "+ Also" pill DUPLICATES (additive)
               The current shared owner (when every selected frame agrees)
               is shown with a "● current" badge and both of its actions
               disable, so the user can see at a glance which row is the
               no-op. -->
          <div class="relative">
            <button
              type="button"
              onclick={() => (charactersOpen = !charactersOpen)}
              disabled={menuBusy}
              title="Reassign or also-assign these frames to a different character"
              class="bg-fuchsia-500/30 hover:bg-fuchsia-500/55 rounded-full px-2.5 h-5 transition-colors inline-flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
            >Characters ▾</button>
            {#if charactersOpen}
              <div
                class="absolute top-full mt-1 left-0 bg-ink-900 border border-ink-700 rounded-lg shadow-xl py-1 min-w-[14rem] z-50"
                role="menu"
              >
                <div class="px-3 py-1 text-[10px] uppercase tracking-wide text-slate-500 border-b border-ink-800 mb-1">
                  Click name to move · + Also to duplicate
                </div>
                {#each characters as c, i (c.slug)}
                  {@const isCurrent = sharedOwnerSlug === c.slug}
                  {@const color = colorForIndex(i)}
                  <div
                    class="flex items-center gap-1 px-1 py-0.5
                      {isCurrent ? 'opacity-60' : ''}"
                    role="menuitem"
                  >
                    <button
                      type="button"
                      onclick={() => !isCurrent && moveSelectedTo(c.slug)}
                      disabled={isCurrent || menuBusy}
                      title={isCurrent
                        ? "Already routed here — nothing to move"
                        : `Move selected frames to ${c.name}`}
                      class="flex-1 text-left px-2 py-1 text-xs rounded hover:bg-ink-800 disabled:hover:bg-transparent disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span
                        class="w-2 h-2 rounded-full flex-shrink-0 {color.dot}"
                        aria-hidden="true"
                      ></span>
                      <span class="text-slate-200 truncate">{c.name}</span>
                      <span class="text-slate-500 tabular-nums text-[10px]">
                        ({c.ref_count})
                      </span>
                      {#if isCurrent}
                        <span class="ml-auto text-[10px] text-emerald-400 inline-flex items-center gap-1">
                          ● current
                        </span>
                      {/if}
                    </button>
                    <button
                      type="button"
                      onclick={() => !isCurrent && alsoAssignSelectedTo(c.slug)}
                      disabled={isCurrent || menuBusy}
                      title={isCurrent
                        ? "Already routed here — duplicating would create another copy with no purpose"
                        : `Also assign selected frames to ${c.name} (duplicate)`}
                      class="flex-shrink-0 px-2 py-0.5 text-[10px] rounded bg-cyan-500/20 hover:bg-cyan-500/40 text-cyan-200 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                    >+ Also</button>
                  </div>
                {/each}
              </div>
            {/if}

          </div>
        {/if}
        <button
          type="button"
          onclick={() => framesStore.clear()}
          class="opacity-70 hover:opacity-100 h-5 w-5 inline-flex items-center justify-center"
          title="Clear selection"
        >✕</button>
      </div>
    {/if}

    <button
      type="button"
      onclick={toggleSelectAll}
      disabled={total === 0}
      class="h-7 px-3 rounded-full text-xs bg-ink-900 border border-ink-700 text-slate-300 hover:bg-ink-800 hover:text-slate-100 btn-disabled inline-flex items-center"
    >{allSelected ? "Deselect all" : "Select all"}</button>

    <span
      class="h-7 px-3 rounded-full text-xs bg-ink-900 border border-ink-700 text-slate-400 inline-flex items-center"
      title={queryActive
        ? `${total} of ${totalInView} matching the search`
        : "Total frames in the current view"}
    >{#if queryActive}{total} / {totalInView}{:else}{total}{/if} frame{(queryActive ? totalInView : total) === 1 ? "" : "s"}</span>

    <!-- Tag search: substring match on the danbooru line, case-insensitive.
         Whitespace-separated tokens are AND-ed; a leading `~` negates a
         token. The input commits to viewStore.tagQuery on a debounce. -->
    <div class="h-7 inline-flex items-center bg-ink-900 border border-ink-700 rounded-full pl-3 pr-1 focus-within:border-accent-500 transition-colors">
      <svg
        viewBox="0 0 16 16"
        class="w-3 h-3 text-slate-500 flex-shrink-0"
        fill="currentColor"
        aria-hidden="true"
      >
        <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"/>
      </svg>
      <input
        type="search"
        value={queryDraft}
        oninput={onQueryInput}
        placeholder="Filter by tag — try `red ~hat`"
        title={"Whitespace-separated substrings, case-insensitive.\n" +
               "Tokens prefixed with ~ are excluded.\n" +
               "Example: red ~hat — has 'red', not 'hat'."}
        class="bg-transparent border-0 outline-none px-2 text-xs text-slate-200 placeholder:text-slate-500 w-44"
        aria-label="Filter frames by tag"
      />
      {#if queryDraft}
        <button
          type="button"
          onclick={clearQuery}
          title="Clear filter"
          aria-label="Clear filter"
          class="opacity-60 hover:opacity-100 h-5 w-5 inline-flex items-center justify-center text-slate-400 hover:text-slate-200"
        >✕</button>
      {/if}
    </div>
  </div>
{/if}

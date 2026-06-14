<script lang="ts">
  import { UNSORTED_FILTER_SENTINEL } from "$lib/api";
  import * as api from "$lib/api";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { toasts } from "$lib/stores/toasts.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import FrameThumb from "./FrameThumb.svelte";
  import FullSizeModal from "./FullSizeModal.svelte";

  let previewIndex = $state<number | null>(null);

  // One id per dropped file; the skeleton tile renders until that file's
  // upload promise resolves and the frames list is refreshed.
  let pendingDrops = $state<{ id: number; name: string }[]>([]);
  let nextDropId = 0;

  let dragDepth = $state(0); // counter so child enter/leave doesn't flicker
  let dragActive = $derived(dragDepth > 0);

  /** Translate the UI's character-filter chip into the API query value.
   *  - "all" → undefined (no filter, server returns every kept frame)
   *  - "unsorted" → server sentinel for orphan rows
   *  - any other → real character slug */
  function filterToQuery(filter: string): string | undefined {
    if (filter === "all") return undefined;
    if (filter === "unsorted") return UNSORTED_FILTER_SENTINEL;
    return filter;
  }

  $effect(() => {
    const slug = projectsStore.active?.slug;
    if (slug) {
      framesStore.refresh(slug, {
        source: viewStore.sourceFilter ?? undefined,
        query: viewStore.tagQuery || undefined,
        characterSlug: filterToQuery(viewStore.characterFilter),
      });
    }
  });


  function handleSelect(index: number, mods: { shift: boolean; ctrl: boolean }) {
    // Plain middle/toggle click toggles a single tile; shift extends a range.
    framesStore.click(index, {
      shift: mods.shift,
      ctrl: !mods.shift || mods.ctrl,
    });
  }

  function openPreview(index: number) {
    previewIndex = index;
  }

  function navPreview(next: number) {
    if (next < 0 || next >= framesStore.items.length) return;
    previewIndex = next;
  }

  async function deletePreviewFrame(filename: string) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    // Same confirm style as the bulk delete in ActionBar.
    if (!confirm("Delete this frame? This removes it from the dataset.")) return;
    const wasAt = framesStore.items.findIndex((i) => i.filename === filename);
    try {
      await api.deleteFrame(slug, filename);
    } catch (e) {
      console.error("delete failed", e);
      toasts.error("Delete failed — see console for details.");
      return;
    }
    framesStore.removeLocal([filename]);
    // Re-point the modal: stay open on the frame that slid into this slot
    // (curate-and-continue), step back when we deleted the last one, and close
    // when nothing is left. `filenames`/the modal guard derive from the store,
    // so they update with the splice automatically.
    const remaining = framesStore.items.length;
    if (remaining === 0) {
      previewIndex = null;
    } else {
      previewIndex = Math.min(wasAt < 0 ? 0 : wasAt, remaining - 1);
    }
  }

  async function refreshFramesAfterCrop() {
    const slug = projectsStore.active?.slug;
    if (slug) {
      await framesStore.refresh(slug, {
        source: viewStore.sourceFilter ?? undefined,
        query: viewStore.tagQuery || undefined,
        characterSlug: filterToQuery(viewStore.characterFilter),
      });
    }
  }

  // ---------------- drag-and-drop image import ----------------

  function isImageDrag(ev: DragEvent): boolean {
    const items = ev.dataTransfer?.items;
    if (!items || items.length === 0) {
      // Some browsers don't expose `items` on dragenter — fall back to types.
      const types = ev.dataTransfer?.types ?? [];
      return Array.from(types).includes("Files");
    }
    return Array.from(items).some((it) => it.kind === "file");
  }

  function onDragEnter(ev: DragEvent) {
    if (!isImageDrag(ev)) return;
    ev.preventDefault();
    dragDepth++;
  }

  function onDragOver(ev: DragEvent) {
    if (!isImageDrag(ev)) return;
    ev.preventDefault();
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = "copy";
  }

  function onDragLeave(ev: DragEvent) {
    if (!isImageDrag(ev)) return;
    ev.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
  }

  async function onDrop(ev: DragEvent) {
    ev.preventDefault();
    dragDepth = 0;
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const all = Array.from(ev.dataTransfer?.files ?? []);
    const images = all.filter(
      (f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name),
    );
    if (images.length === 0) return;

    // Spawn one skeleton per dropped image so the user has immediate feedback
    // while the server downscales + auto-tags + writes the frame.
    const placeholders = images.map((f) => ({ id: ++nextDropId, name: f.name }));
    pendingDrops = [...pendingDrops, ...placeholders];

    // Drag-drop routing follows the active filter:
    //   - single character selected → drop onto that character (deterministic)
    //   - "All" or "Unsorted" → fall back to the project's first character;
    //     a future revision can CCIP-route per the design doc, but explicit
    //     beats implicit while there's no UI to surface a misroute.
    const filter = viewStore.characterFilter;
    const explicitTarget =
      filter !== "all" && filter !== "unsorted" ? filter : undefined;

    try {
      const resp = await api.uploadFrames(slug, images, explicitTarget);
      await framesStore.refresh(slug, {
        source: viewStore.sourceFilter ?? undefined,
        query: viewStore.tagQuery || undefined,
        characterSlug: filterToQuery(viewStore.characterFilter),
      });
      // Upload succeeded, but the per-image LLM-describe call may have
      // failed silently (endpoint timeout, wrong model, auth refused).
      // Surface that so the user knows their dropped image came back
      // tag-only — otherwise it looks like LLM tagging is configured but
      // simply not running.
      if (resp.llm_error) {
        toasts.error(
          `Image${images.length === 1 ? "" : "s"} uploaded and tagged, ` +
          `but the LLM description call failed:\n\n${resp.llm_error}`,
        );
      }
    } catch (e) {
      console.error("upload failed", e);
      toasts.error("Upload failed — see console for details.");
    } finally {
      const ids = new Set(placeholders.map((p) => p.id));
      pendingDrops = pendingDrops.filter((p) => !ids.has(p.id));
    }
  }

  let cols = $derived(viewStore.density);
  let filenames = $derived(framesStore.items.map((i) => i.filename));
  // Snapshot the selection on every bump so each FrameThumb sees a stable
  // boolean and doesn't have to read `selectionVersion` itself.
  let selectedSet = $derived.by(() => {
    framesStore.selectionVersion;
    return framesStore.selection.selected();
  });
</script>

<div
  class="mt-4 relative"
  ondragenter={onDragEnter}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  ondrop={onDrop}
  role="region"
  aria-label="Frames grid"
>
  {#if framesStore.loading}
    <p class="text-slate-500 py-12 text-center">Loading frames…</p>
  {:else if framesStore.items.length === 0 && pendingDrops.length === 0}
    <div class="py-24 text-center text-slate-500">
      <p class="text-lg mb-1">No frames yet.</p>
      <p class="text-sm">Add a video in Sources and run extract — or drop image files here.</p>
    </div>
  {:else}
    <div
      class="grid gap-2"
      style="grid-template-columns: repeat({cols}, minmax(0, 1fr));"
    >
      {#each framesStore.items as f, i (f.filename)}
        <FrameThumb
          frame={f}
          selected={selectedSet.has(f.filename)}
          onpreview={() => openPreview(i)}
          onselect={(mods) => handleSelect(i, mods)}
        />
      {/each}

      {#each pendingDrops as p (p.id)}
        <div
          class="relative aspect-[3/4] rounded-lg overflow-hidden bg-ink-900 border border-ink-700 flex flex-col items-center justify-center gap-2"
          title="Uploading {p.name}…"
        >
          <span
            class="block w-8 h-8 rounded-full border-2 border-ink-700 border-t-accent-500 animate-spin"
            aria-hidden="true"
          ></span>
          <span class="text-[10px] text-slate-500 px-2 text-center truncate w-full">
            {p.name}
          </span>
        </div>
      {/each}
    </div>
  {/if}

  {#if dragActive}
    <!-- Translucent drop overlay — pointer-events:none so the underlying
         drop handler still fires; pure visual affordance. -->
    <div
      class="absolute inset-0 rounded-xl border-2 border-dashed border-emerald-400 bg-emerald-500/10 pointer-events-none flex items-center justify-center"
    >
      <p class="text-emerald-200 text-sm font-medium">Drop images to add as custom frames</p>
    </div>
  {/if}
</div>

{#if previewIndex !== null && framesStore.items[previewIndex]}
  <FullSizeModal
    {filenames}
    index={previewIndex}
    onnav={navPreview}
    onclose={() => (previewIndex = null)}
    ondelete={deletePreviewFrame}
    oncropped={(filename) => {
      // Bump first (survives the refresh's states reset) so the grid tile
      // re-fetches the freshly-written crop derivative.
      framesStore.markCropped(filename);
      void refreshFramesAfterCrop();
    }}
  />
{/if}

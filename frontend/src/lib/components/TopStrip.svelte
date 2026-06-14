<script lang="ts">
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import ActionBar from "./ActionBar.svelte";
  import CharacterStrip from "./CharacterStrip.svelte";
  import DensitySlider from "./DensitySlider.svelte";
  import ProjectPills from "./ProjectPills.svelte";
  import QueuePill from "./QueuePill.svelte";
  import ViewTabs from "./ViewTabs.svelte";

  type Props = {
    onopenRegex: () => void;
    onopenCreate: () => void;
    onopenDelete: () => void;
  };
  const {
    onopenRegex,
    onopenCreate,
    onopenDelete,
  }: Props = $props();

  // Character filter chips appear in the top bar (next to project pills)
  // when the user is on the Frames tab and the project has more than one
  // character. Single-character workflows render the top bar exactly as
  // before this change so the legacy UX is preserved.
  let showCharacterFilter = $derived(
    viewStore.tab === "frames"
      && (projectsStore.active?.characters.length ?? 0) > 1,
  );

  /** Pseudo-chips in front of the real characters: "All" (no filter) and
   *  "Unsorted" (server sentinel for orphan rows). The "Unsorted" chip only
   *  appears when the project actually has orphan frames — `unsortedTotal` is
   *  refreshed on every grid fetch. */
  let leadingChips = $derived(
    framesStore.unsortedTotal > 0
      ? [
          { key: "all", label: "All" },
          { key: "unsorted", label: "Unsorted" },
        ]
      : [{ key: "all", label: "All" }],
  );

  // If the user was sitting on the "Unsorted" filter and the last orphan
  // frame just left (moved/deleted), the chip disappears — fall back to "All"
  // so the grid isn't stranded on an empty, unselectable filter.
  $effect(() => {
    if (framesStore.unsortedTotal === 0 && viewStore.characterFilter === "unsorted") {
      viewStore.characterFilter = "all";
    }
  });

  function selectCharacterFilter(key: string) {
    viewStore.characterFilter = key;
  }
</script>

<!-- Top bar uses flex-wrap so when the window narrows the children
     wrap onto additional rows instead of overflowing. The min-h is
     dropped (no longer a strict single-row constraint), and a small
     row gap keeps stacked rows from butting against each other. -->
<header class="sticky top-0 z-30 px-4 py-3 bg-ink-950/95 backdrop-blur-sm border-b border-ink-700">
  <div class="flex flex-wrap items-center gap-x-3 gap-y-2 bg-ink-950 border border-ink-700 rounded-xl px-4 py-2.5 shadow-md">
    <!-- Left cluster: brand dot + project pills + (when applicable) the
         character filter strip. Wrapped in its own flex container so the
         filter strip rides next to ProjectPills and breaks together when
         the row gets tight. -->
    <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
      <ProjectPills {onopenCreate} {onopenDelete} />
      {#if showCharacterFilter}
        <!-- Small left padding (pl-2) separates the filter chips from the
             project pills' "+" button. The strip itself wraps internally
             via flex-wrap so a project with many characters splits
             cleanly. -->
        <div class="pl-2">
          <CharacterStrip
            leadingChips={leadingChips}
            activeKey={viewStore.characterFilter}
            onselect={selectCharacterFilter}
          />
        </div>
      {/if}
    </div>

    <!-- Spacer pushes the right cluster (action bar + view tabs + density
         + queue) to the far edge on wide screens. On narrow screens it
         collapses to zero width and the right cluster wraps to the next
         row underneath the project/filter cluster. -->
    <div class="flex-1 min-w-0"></div>

    <!-- Right cluster: action bar + view tabs + density + queue pill.
         Allowed to wrap as a unit so they don't get truncated mid-control. -->
    <div class="flex flex-wrap items-center gap-x-3 gap-y-2 justify-end">
      <ActionBar {onopenRegex} />
      <ViewTabs />
      <DensitySlider />
      <!-- Grid fit toggle, paired with the density slider. Off = thumbnails
           crop to fill the tile (object-cover); on = whole image fits inside
           with black bars (object-contain), never cropped. -->
      <button
        type="button"
        onclick={() => (viewStore.fitContain = !viewStore.fitContain)}
        aria-label="Toggle image fit"
        aria-pressed={viewStore.fitContain}
        title={viewStore.fitContain
          ? "Fit: whole image shown with black bars — click to crop-fill tiles"
          : "Fill: thumbnails cropped to the tile — click to fit the whole image"}
        class="w-7 h-7 inline-flex items-center justify-center rounded-full border transition-colors
          {viewStore.fitContain
            ? 'gradient-accent text-white border-white/10 shadow-[0_2px_8px_rgba(99,102,241,0.35)]'
            : 'bg-ink-900 border-ink-700 text-slate-400 hover:bg-ink-800 hover:text-slate-200'}"
      >
        <svg
          viewBox="0 0 24 24" class="w-4 h-4"
          fill="none" stroke="currentColor" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"
        >
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <rect x="3" y="8.5" width="18" height="7" rx="1" fill="currentColor" stroke="none" opacity="0.55" />
        </svg>
      </button>
      <QueuePill />
    </div>
  </div>
</header>

<script lang="ts">
  import { projectsStore } from "$lib/stores/projects.svelte";

  // Delete modal is intentionally mounted in App.svelte, not here — the
  // top bar uses backdrop-blur, which creates a CSS containing block for
  // position:fixed descendants and would clip the modal to header bounds.
  type Props = { onopenCreate: () => void; onopenDelete: () => void };
  const { onopenCreate, onopenDelete }: Props = $props();

  // Collapsed by default: show only the latest project. The registry returns
  // the list ordered by `last_opened_at DESC`, so `list[0]` is the most
  // recently opened. Expanding reveals every project; the chevron flips.
  let expanded = $state(false);

  let canToggle = $derived(projectsStore.list.length > 1);

  // Collapsed view shows the active project (whichever is open), falling back
  // to the most-recently-opened when nothing is loaded yet.
  let displayList = $derived.by(() => {
    const list = projectsStore.list;
    if (expanded || list.length <= 1) return list;
    const activeSlug = projectsStore.active?.slug;
    const collapsed =
      (activeSlug ? list.find((p) => p.slug === activeSlug) : undefined) ?? list[0];
    return [collapsed];
  });

  function isActive(slug: string): boolean {
    return projectsStore.active?.slug === slug;
  }

  async function selectProject(slug: string) {
    await projectsStore.load(slug);
  }
</script>

<div class="flex items-center gap-1">
  {#each displayList as p (p.slug)}
    {@const active = isActive(p.slug)}
    <span
      class="inline-flex items-center rounded-full overflow-hidden
             transition-all border border-transparent
             {active
               ? 'gradient-accent text-white border-white/10 shadow-[0_2px_8px_rgba(99,102,241,0.35)]'
               : 'bg-ink-800 text-slate-400 hover:bg-ink-700 hover:text-slate-200'}"
    >
      <button
        class="px-3 py-1 text-xs"
        onclick={() => selectProject(p.slug)}
        title={p.folder}
      >{p.name}</button>
      {#if active}
        <button
          type="button"
          onclick={(e) => { e.stopPropagation(); onopenDelete(); }}
          aria-label="delete project"
          title="Delete project"
          class="pr-2 pl-1 py-1 text-white/70 hover:text-white"
        >
          <!-- Inline trash icon (heroicons outline trash, 14×14). -->
          <svg
            xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round"
            class="w-3.5 h-3.5"
          >
            <path d="M3 6h18" />
            <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
            <path d="M10 11v6" />
            <path d="M14 11v6" />
          </svg>
        </button>
      {/if}
    </span>
  {/each}
  {#if canToggle}
    <button
      type="button"
      onclick={() => (expanded = !expanded)}
      aria-expanded={expanded}
      aria-label={expanded ? "Show only the latest project" : "Show all projects"}
      title={expanded ? "Collapse to the latest project" : "Show all projects"}
      class="w-6 py-1 text-xs rounded-full border border-ink-600 leading-none
             text-slate-400 hover:border-accent-500 hover:text-accent-400"
    >{expanded ? "‹" : "›"}</button>
  {/if}
  <button
    class="w-6 py-1 text-xs rounded-full border border-dashed border-ink-600
           text-slate-500 hover:border-accent-500 hover:text-accent-400"
    onclick={onopenCreate}
    title="New project"
  >+</button>
</div>

<!-- frontend/src/lib/components/TagAutocomplete.svelte -->
<script lang="ts">
  import { categoryColor, formatCount, type Suggestion } from "$lib/tagSearch";
  import { computeAutocompletePosition } from "$lib/tagAutocompletePosition";

  type Props = {
    suggestions: Suggestion[];
    activeIndex: number;
    /** The tag input this dropdown is anchored to (for positioning). */
    anchor: HTMLElement;
    onaccept: (s: Suggestion) => void;
    onhover: (index: number) => void;
  };
  const { suggestions, activeIndex, anchor, onaccept, onhover }: Props = $props();

  // Position in a fixed layer off the anchor's rect so we escape the
  // overflow-hidden grid-hover panel. Clamps to the viewport (flip above + keep
  // within the left/right edges); recomputed whenever the suggestion set
  // changes. See `tagAutocompletePosition.ts` for the (tested) math.
  let top = $state(0);
  let left = $state(0);
  let width = $state(0);
  let maxWidth = $state(0);

  $effect(() => {
    // Touch `suggestions` so this recomputes as the list grows/shrinks.
    const count = suggestions.length;
    const r = anchor.getBoundingClientRect();
    const pos = computeAutocompletePosition(r, count, window.innerWidth, window.innerHeight);
    top = pos.top;
    left = pos.left;
    width = pos.width;
    maxWidth = pos.maxWidth;
  });
</script>

<ul
  role="listbox"
  style="position:fixed; top:{top}px; left:{left}px; min-width:{width}px; max-width:{maxWidth}px;"
  class="z-50 max-h-64 overflow-y-auto rounded-lg border border-ink-700 bg-ink-950/95 backdrop-blur-sm shadow-xl py-1 text-[11px]"
>
  {#each suggestions as s, i (s.entry.name)}
    <li role="option" aria-selected={i === activeIndex}>
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <button
        type="button"
        onmousedown={(e) => e.preventDefault()}
        onclick={() => onaccept(s)}
        onmouseenter={() => onhover(i)}
        class="flex w-full items-center gap-2 px-2 py-1 text-left
          {i === activeIndex ? 'bg-accent-500/30' : 'hover:bg-white/5'}"
      >
        <span class="flex-1 truncate {categoryColor(s.entry.category)}">{s.entry.name}</span>
        {#if s.viaAlias}
          <span class="text-[9px] text-slate-500">alias</span>
        {/if}
        <span class="text-[9px] tabular-nums text-slate-500">{formatCount(s.entry.count)}</span>
      </button>
    </li>
  {/each}
</ul>

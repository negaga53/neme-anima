<script lang="ts">
  import * as api from "$lib/api";
  import { trainingStore } from "$lib/stores/training.svelte";
  import { buildSampleGrid } from "$lib/trainingSamples";
  import type { TrainingSamplesResponse } from "$lib/types";

  type Props = {
    runName: string;
    /** Show the "none yet" hint (the active run, with sampling configured);
     *  otherwise a run without samples renders nothing. */
    samplingEnabled: boolean;
  };
  const { runName, samplingEnabled }: Props = $props();

  let data = $state<TrainingSamplesResponse | null>(null);
  let error = $state<string | null>(null);
  let zoomed = $state<{ url: string; prompt: string; epoch: number } | null>(null);
  let grid = $derived(data ? buildSampleGrid(data.epochs) : null);

  // Refetch on mount, on run change, and whenever new samples land.
  $effect(() => {
    const slug = trainingStore.slug;
    const name = runName;
    void trainingStore.samplesVersion;
    if (!slug) return;
    api.listTrainingSamples(slug, name)
      .then((r) => { data = r; error = null; })
      .catch((e) => { error = e instanceof Error ? e.message : String(e); });
  });
</script>

<svelte:window onkeydown={(e) => { if (zoomed && e.key === "Escape") zoomed = null; }} />

{#if error || (grid && (grid.epochs.length > 0 || samplingEnabled))}
  <div class="mt-2 border-t border-ink-800 pt-2">
    <h4 class="text-[10px] uppercase tracking-wide text-slate-500 px-2 mb-1">Samples</h4>
    {#if error}
      <p class="text-red-400 text-[11px] px-2">{error}</p>
    {:else if grid && grid.epochs.length === 0}
      <p class="text-slate-500 text-[11px] px-2">No samples for this run yet.</p>
    {:else if grid}
      <div class="overflow-x-auto px-2 pb-1">
        <table class="border-separate border-spacing-1 text-[10px]">
          <thead>
            <tr>
              <th></th>
              {#each grid.epochs as e (e.epoch)}
                <th class="font-mono font-normal text-slate-400" title={e.error ?? ""}>
                  ep {e.epoch}{#if e.error}<span class="text-red-400"> ⚠</span>{/if}
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each grid.rows as row, ri (ri)}
              <tr>
                <th class="max-w-[10rem] truncate text-left align-top font-normal text-slate-400" title={row.prompt}>
                  {row.prompt}
                </th>
                {#each row.cells as cell, ci (ci)}
                  {@const epoch = grid.epochs[ci].epoch}
                  <td class="align-top">
                    {#if cell}
                      <button
                        type="button"
                        class="block"
                        onclick={() => (zoomed = { url: cell.url, prompt: row.prompt, epoch })}
                        title={`epoch ${epoch}`}
                      >
                        <img
                          src={cell.url}
                          alt={`epoch ${epoch}: ${row.prompt}`}
                          loading="lazy"
                          class="w-28 h-28 object-cover rounded border border-ink-700 hover:border-accent-500"
                        />
                      </button>
                    {:else}
                      <div class="w-28 h-28 rounded border border-dashed border-ink-800"></div>
                    {/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
{/if}

{#if zoomed}
  <div
    class="fixed inset-0 z-50 bg-black/85 flex flex-col items-center justify-center p-6 cursor-zoom-out"
    role="dialog"
    aria-modal="true"
    aria-label="Sample image"
    tabindex="-1"
    onclick={() => (zoomed = null)}
    onkeydown={(e) => { if (e.key === "Escape") zoomed = null; }}
  >
    <img src={zoomed.url} alt={zoomed.prompt} class="max-w-full max-h-[85vh] rounded" />
    <p class="mt-2 text-xs text-slate-300 max-w-3xl text-center">
      <span class="font-mono text-slate-400">epoch {zoomed.epoch}</span> — {zoomed.prompt}
    </p>
  </div>
{/if}

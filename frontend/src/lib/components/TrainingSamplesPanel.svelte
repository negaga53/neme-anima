<script lang="ts">
  import { untrack } from "svelte";
  import { trainingStore } from "$lib/stores/training.svelte";
  import { parsePromptLines } from "$lib/trainingSamples";
  import type { TrainingConfig } from "$lib/types";

  type Props = { cfg: TrainingConfig };
  const { cfg }: Props = $props();

  let enabled = $derived(cfg.sample_prompts.some((p) => p.trim()));
  let samplers = $derived(trainingStore.configResp?.sample_options.samplers ?? [cfg.sample_sampler]);
  let schedulers = $derived(trainingStore.configResp?.sample_options.schedulers ?? [cfg.sample_scheduler]);

  // Free-typing drafts for the two text areas, committed on blur. Re-seeded
  // from the server copy whenever it changes — except while the user is in
  // the field, so a PATCH echo from another input can't eat their typing.
  let promptsDraft = $state("");
  let negativeDraft = $state("");
  let promptsFocused = false;
  let negativeFocused = false;
  $effect(() => {
    const p = cfg.sample_prompts.join("\n");
    untrack(() => { if (!promptsFocused) promptsDraft = p; });
  });
  $effect(() => {
    const n = cfg.sample_negative_prompt;
    untrack(() => { if (!negativeFocused) negativeDraft = n; });
  });

  async function patchField<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) {
    await trainingStore.patch({ [key]: value } as Partial<TrainingConfig>);
  }

  function commitPrompts() {
    promptsFocused = false;
    const next = parsePromptLines(promptsDraft);
    if (next.join("\n") !== cfg.sample_prompts.join("\n")) patchField("sample_prompts", next);
    else promptsDraft = next.join("\n");
  }

  function commitNegative() {
    negativeFocused = false;
    if (negativeDraft !== cfg.sample_negative_prompt) {
      patchField("sample_negative_prompt", negativeDraft.trim());
    }
  }

  const num = (e: Event) => Number((e.target as HTMLInputElement).value);
</script>

<div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
  <div class="flex items-center justify-between mb-3">
    <h3 class="text-sm font-medium text-slate-200">Samples</h3>
    <span
      class="text-[10px] px-2 py-0.5 rounded {enabled
        ? 'bg-emerald-900/50 text-emerald-300 border border-emerald-800'
        : 'bg-slate-800 text-slate-400 border border-slate-700'}"
    >{enabled ? "on" : "off — add a prompt to enable"}</span>
  </div>
  <div class="space-y-3 text-xs">
    <label
      class="block"
      title="One prompt per line. Each saved sampling epoch renders one image per prompt with that epoch's LoRA. Include the trigger token yourself. Leave empty to disable sampling."
    >
      <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
        <span>prompts (one per line)</span>
        <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
      </span>
      <textarea
        rows="3"
        bind:value={promptsDraft}
        onfocus={() => (promptsFocused = true)}
        onblur={commitPrompts}
        placeholder="masterpiece, best quality, mytrigger, 1girl, solo, smile, upper body"
        class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono resize-y"
      ></textarea>
    </label>
    <label class="block" title="Negative prompt shared by every sample (optional).">
      <span class="text-[10px] uppercase tracking-wide text-slate-500">negative prompt</span>
      <textarea
        rows="2"
        bind:value={negativeDraft}
        onfocus={() => (negativeFocused = true)}
        onblur={commitNegative}
        placeholder="worst quality, low quality, blurry"
        class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono resize-y"
      ></textarea>
    </label>
    <div class="grid grid-cols-2 md:grid-cols-3 gap-3 items-end">
      <label
        class="block"
        title="Render samples every N epochs (the final epoch is always sampled). Must be a multiple of save_every_n_epochs — samples are made from saved LoRAs."
      >
        <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
          <span>every N epochs</span>
          <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
        </span>
        <input
          type="number" min="1" step="1" value={cfg.sample_every_n_epochs}
          onchange={(e) => patchField("sample_every_n_epochs", num(e))}
          class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
        />
        <span class="block text-[10px] text-slate-600 mt-1">
          multiple of save_every_n_epochs ({cfg.save_every_n_epochs})
        </span>
      </label>
      <label
        class="flex items-start gap-2 col-span-2 pb-5"
        title="Low-VRAM setups: don't sample while training holds the GPU — render every sample epoch after the trainer exits instead."
      >
        <input
          type="checkbox" checked={cfg.sample_defer_to_end}
          onchange={(e) => patchField("sample_defer_to_end", (e.target as HTMLInputElement).checked)}
          class="mt-0.5"
        />
        <span class="text-slate-300">
          Defer sampling to end of run
          <span class="block text-[10px] text-slate-500">for low-VRAM setups — no GPU sharing with the trainer</span>
        </span>
      </label>
    </div>
    <details>
      <summary class="cursor-pointer text-[10px] uppercase tracking-wide text-slate-500 hover:text-slate-300">
        Advanced sampler settings
      </summary>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">steps</span>
          <input
            type="number" min="1" step="1" value={cfg.sample_steps}
            onchange={(e) => patchField("sample_steps", num(e))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">sampler</span>
          <select
            value={cfg.sample_sampler}
            onchange={(e) => patchField("sample_sampler", (e.target as HTMLSelectElement).value)}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          >
            {#each samplers as s (s)}<option value={s}>{s}</option>{/each}
          </select>
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">scheduler</span>
          <select
            value={cfg.sample_scheduler}
            onchange={(e) => patchField("sample_scheduler", (e.target as HTMLSelectElement).value)}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          >
            {#each schedulers as s (s)}<option value={s}>{s}</option>{/each}
          </select>
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">cfg</span>
          <input
            type="number" min="0" step="0.1" value={cfg.sample_cfg}
            onchange={(e) => patchField("sample_cfg", num(e))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">width</span>
          <input
            type="number" min="256" max="2048" step="16" value={cfg.sample_width}
            onchange={(e) => patchField("sample_width", num(e))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">height</span>
          <input
            type="number" min="256" max="2048" step="16" value={cfg.sample_height}
            onchange={(e) => patchField("sample_height", num(e))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block" title="Fixed seed so each epoch's images are directly comparable.">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">seed</span>
          <input
            type="number" min="0" step="1" value={cfg.sample_seed}
            onchange={(e) => patchField("sample_seed", num(e))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
      </div>
    </details>
  </div>
</div>

<script lang="ts">
  import { onDestroy, untrack } from "svelte";
  import { trainingStore } from "$lib/stores/training.svelte";
  import type { TrainingConfig, TrainingPathCheck } from "$lib/types";
  import TrainingSamplesPanel from "./TrainingSamplesPanel.svelte";
  import {
    PRESETS,
    LOW_VRAM_PROFILE,
    LOW_VRAM_RESET,
    PATH_FIELDS,
    RESOLUTION_PRESETS,
    type PathField,
  } from "$lib/trainingPresets";

  type Props = {
    /** The live config. Never null here — the parent only renders this panel
     *  inside its `{#if cfg}` guard. */
    cfg: TrainingConfig;
    /** Switch the parent to the Identity sub-tab (the trigger-token shortcut). */
    onopenidentity: () => void;
  };
  const { cfg, onopenidentity }: Props = $props();

  // ---- per-path live debounce -----------------------------------------
  // We debounce the check-path POST so the user doesn't fire an HTTP
  // request on every keystroke. The check-path endpoint is cheap, but
  // 60 round-trips while pasting a long path is still wasteful.
  const debounceTimers: Partial<Record<PathField, ReturnType<typeof setTimeout>>> = {};
  let pathChecks = $state<Record<PathField, TrainingPathCheck | null>>({
    diffusion_pipe_dir: null, dit_path: null, vae_path: null, llm_path: null,
  });

  // Local edit buffers for path inputs so the user can type freely; we
  // only commit (PATCH) on blur or Save.
  let pathDraft = $state<Record<PathField, string>>({
    diffusion_pipe_dir: "", dit_path: "", vae_path: "", llm_path: "",
  });
  let pathDirty = $state<Record<PathField, boolean>>({
    diffusion_pipe_dir: false, dit_path: false, vae_path: false, llm_path: false,
  });

  // Seed the path drafts/checks from the loaded config the first time it's
  // available, and whenever the active project's config identity changes.
  $effect(() => {
    const c = cfg; // dependency
    pathDraft = {
      diffusion_pipe_dir: c.diffusion_pipe_dir,
      dit_path: c.dit_path,
      vae_path: c.vae_path,
      llm_path: c.llm_path,
    };
    pathDirty = { diffusion_pipe_dir: false, dit_path: false, vae_path: false, llm_path: false };
    const pc = trainingStore.configResp?.path_checks;
    if (pc) pathChecks = { ...untrack(() => pathChecks), ...pc };
  });

  onDestroy(() => {
    for (const t of Object.values(debounceTimers)) if (t) clearTimeout(t);
  });

  function schedulePathCheck(field: PathField, value: string) {
    pathDraft = { ...pathDraft, [field]: value };
    pathDirty = { ...pathDirty, [field]: true };
    if (debounceTimers[field]) clearTimeout(debounceTimers[field]!);
    debounceTimers[field] = setTimeout(async () => {
      try {
        const res = await trainingStore.checkPath(field, value);
        pathChecks = { ...pathChecks, [field]: res };
      } catch (e) {
        pathChecks = {
          ...pathChecks,
          [field]: { path: value, exists: false, is_file: false, is_dir: false, error: String(e) },
        };
      }
    }, 300);
  }

  async function commitPath(field: PathField) {
    if (!pathDirty[field]) return;
    await trainingStore.patch({ [field]: pathDraft[field] } as Partial<TrainingConfig>);
    pathDirty = { ...pathDirty, [field]: false };
    const pc = trainingStore.configResp?.path_checks;
    if (pc) pathChecks = { ...pathChecks, ...pc };
  }

  async function patchField<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) {
    await trainingStore.patch({ [key]: value } as Partial<TrainingConfig>);
  }

  function applyPreset(preset: string) {
    const overrides = PRESETS[preset];
    if (!overrides) return;
    trainingStore.patch({ ...overrides, preset } as Partial<TrainingConfig>);
  }

  function pickResolutionPreset(values: number[]) {
    trainingStore.patch({ resolutions: values });
  }

  function isLowVramActive(c: TrainingConfig | null): boolean {
    if (!c) return false;
    // Detect the preset by the two values the recipe defaults could never
    // have on their own: a single-512 bucket *and* non-zero block-swap.
    // (rank could be 16 by user preference even outside the preset, so we
    // don't gate on it.)
    return (
      c.resolutions.length === 1 &&
      c.resolutions[0] === 512 &&
      (c.blocks_to_swap ?? 0) > 0
    );
  }

  function toggleLowVram() {
    const patch = isLowVramActive(cfg) ? LOW_VRAM_RESET : LOW_VRAM_PROFILE;
    trainingStore.patch(patch as Partial<TrainingConfig>);
  }

  function pathBadge(check: TrainingPathCheck | null): { ok: boolean; text: string } {
    if (!check) return { ok: false, text: "not checked" };
    if (check.error) return { ok: false, text: check.error };
    return { ok: true, text: check.is_dir ? "directory" : "file" };
  }
</script>

      <!-- ====================================================== -->
      <!-- ================= SETTINGS SUB-TAB =================== -->
      <!-- ====================================================== -->

      <!-- Trainer paths -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <h3 class="text-sm font-medium text-slate-200 mb-1">Trainer paths</h3>
        <p class="text-xs text-slate-500 mb-3">
          Point at locally-installed assets so we don't have to download the
          Anima weights for you. Each path is checked as you type — green
          means the file or directory exists.
        </p>
        <div class="grid grid-cols-1 gap-3">
          {#each PATH_FIELDS as f (f.key)}
            {@const check = pathChecks[f.key]}
            {@const badge = pathBadge(check)}
            <label class="block" title={f.tooltip}>
              <div class="flex items-center justify-between mb-1">
                <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
                  <span>{f.label}</span>
                  <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
                </span>
                <span class="text-[10px] {badge.ok ? 'text-emerald-400' : 'text-red-400'}">
                  {badge.ok ? "✓" : "✗"} {badge.text}
                </span>
              </div>
              <input
                value={pathDraft[f.key]}
                oninput={(e) => schedulePathCheck(f.key, (e.target as HTMLInputElement).value)}
                onblur={() => commitPath(f.key)}
                placeholder={f.placeholder}
                class="w-full px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-xs font-mono focus:outline-none focus:border-accent-500"
              />
              <span class="block text-[10px] text-slate-600 mt-1">{f.hint}</span>
            </label>
          {/each}
        </div>

        <details class="mt-4">
          <summary class="text-xs text-slate-400 cursor-pointer hover:text-slate-200">
            Advanced: launcher command override
          </summary>
          <p class="text-[11px] text-slate-500 mt-2 mb-1">
            Default: <code class="text-slate-400">deepspeed --num_gpus=1 train.py --deepspeed --config &lcub;config&rcub;</code>.
            Override only if you need a custom launcher or wrapper script. Use <code class="text-slate-400">&lcub;config&rcub;</code> as a placeholder for the run TOML path.
          </p>
          <input
            value={cfg.launcher_override}
            onchange={(e) => patchField("launcher_override", (e.target as HTMLInputElement).value)}
            placeholder="(empty = built-in default)"
            class="w-full px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-xs font-mono focus:outline-none focus:border-accent-500"
          />
        </details>
      </div>

      <!-- Preset -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-sm font-medium text-slate-200">Preset</h3>
          <div class="flex items-center gap-1">
            {#each ["style", "character"] as p}
              <button
                type="button"
                onclick={() => applyPreset(p)}
                class="toggle-chip"
                class:is-active={cfg.preset === p}
              >{p}</button>
            {/each}
            <span class="mx-1 text-slate-700" aria-hidden="true">|</span>
            <button
              type="button"
              onclick={toggleLowVram}
              title="Switches to a known-good ≤8 GB VRAM profile (~6.4 GB measured stable on Anima): single 512 bucket, rank 16, 26 of 28 transformer blocks CPU-offloaded (the max), AdamW8bitKahan optimizer (8-bit state), unsloth activation checkpointing. Click again to restore recipe defaults. Orthogonal to style/character — pick that first, then this if needed."
              class="toggle-chip"
              class:is-active={isLowVramActive(cfg)}
            >Fit in 8 GB</button>
          </div>
        </div>
        <p class="text-[11px] text-slate-500">
          <strong>style</strong> = recipe defaults (lr 2e-5, grad-accum 4, 40 epochs).
          <strong>character</strong> bumps lr to 5e-5 and epochs to 60 — community
          reports indicate 2e-5 is too low for character identity.
          <strong>Fit in 8 GB</strong> is independent: it crunches the VRAM-heavy
          knobs (single 512 bucket, rank 16, 26 of Anima's 28 transformer blocks
          CPU-offloaded, 8-bit AdamW optimizer state, unsloth activation
          checkpointing) so the run fits on an 8 GB card — measured stable at
          ~6.4 GB on Anima. Expect ~10× slower steps and slightly less fine
          detail. fp8 is intentionally not used — it crashes Anima's LLM adapter.
        </p>
      </div>

      <!-- Adapter / optimizer -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <h3 class="text-sm font-medium text-slate-200 mb-3">Adapter &amp; optimizer</h3>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
          <label
            class="block"
            title="LoRA rank — the inner dimension of the low-rank adapter matrices. Higher = more capacity and bigger output file. 16–32 is typical for character/style LoRAs; 64+ overfits small datasets."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>rank</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="1" step="1" value={cfg.rank}
              onchange={(e) => patchField("rank", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="LoRA alpha scaling factor. Effective adapter strength ≈ alpha / rank. Used by kohya-style trainers; diffusion-pipe ignores it (it bakes the scale into the optimizer instead). Set equal to rank if unsure."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>alpha (kohya only)</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="1" step="1" value={cfg.alpha}
              onchange={(e) => patchField("alpha", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="Optimizer learning rate. Anima recipe: 2e-5 for styles, 5e-5 for characters. Higher = trains faster but risks instability and forgetting the base model."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>learning_rate</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" step="any" value={cfg.learning_rate}
              onchange={(e) => patchField("learning_rate", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="AdamW weight decay (L2 regularization). Higher = stronger pull toward zero, fights overfitting but can also dampen the LoRA's effect. 0.01 is standard."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>weight_decay</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" step="any" value={cfg.weight_decay}
              onchange={(e) => patchField("weight_decay", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="AdamW epsilon — small constant for numerical stability in the denominator. Default 1e-8; only worth tuning if you see NaNs or training-loss spikes."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>eps</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" step="any" value={cfg.eps}
              onchange={(e) => patchField("eps", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="Steps spent linearly ramping the LR from 0 up to learning_rate at the start of training. 0 = no warmup. A small warmup (e.g. 10–100) helps stability on small datasets."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>warmup_steps</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="0" step="1" value={cfg.warmup_steps}
              onchange={(e) => patchField("warmup_steps", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="Maximum global L2 norm of gradients before clipping. 1.0 is standard. 0 = disabled. Lower values trade speed for stability when gradients spike."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>gradient_clipping</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" step="any" value={cfg.gradient_clipping}
              onchange={(e) => patchField("gradient_clipping", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="AdamW momentum coefficients. β₁ controls the running mean of the gradient (typical 0.9), β₂ the running variance (0.99 is the diffusion-pipe default; 0.999 is the AdamW default)."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>betas (β₁,β₂)</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              value={cfg.optimizer_betas.join(",")}
              onchange={(e) => {
                const parts = (e.target as HTMLInputElement).value.split(",").map(s => Number(s.trim())).filter(n => !Number.isNaN(n));
                if (parts.length === 2) patchField("optimizer_betas", parts);
              }}
              placeholder="0.9, 0.99"
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
        </div>
      </div>

      <!-- Batching / resolution -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <h3 class="text-sm font-medium text-slate-200 mb-3">Batching &amp; resolution</h3>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs mb-3">
          <label
            class="block"
            title="Per-GPU samples processed in one forward/backward pass. Increase only if you have VRAM headroom — this is the main lever for OOM errors."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>micro_batch_size</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="1" step="1" value={cfg.micro_batch_size}
              onchange={(e) => patchField("micro_batch_size", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="How many micro-batches to accumulate before each optimizer step. Effective batch = micro_batch_size × this. Lets you simulate a larger batch without using more VRAM, at the cost of training speed."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>grad_accum_steps</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="1" step="1" value={cfg.gradient_accumulation_steps}
              onchange={(e) => patchField("gradient_accumulation_steps", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <div
            class="block"
            title="Computed: micro_batch_size × grad_accum_steps. This is the effective batch size the optimizer sees per step."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>effective batch</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <div class="mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono text-slate-400">
              {cfg.micro_batch_size * cfg.gradient_accumulation_steps}
            </div>
          </div>
        </div>

        <div class="mb-3" title="Image edge sizes (pixels) the dataset is bucketed into. 512+1024 is the recipe default — gives the model both a low- and high-res view of each frame. Adding 1536 helps fine detail but costs VRAM and time.">
          <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
            <span>resolution buckets</span>
            <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
          </span>
          <div class="flex flex-wrap gap-1 mt-1">
            {#each RESOLUTION_PRESETS as p (p.key)}
              <button
                type="button"
                onclick={() => pickResolutionPreset(p.values)}
                class="toggle-chip"
                class:is-active={cfg.resolutions.join(",") === p.values.join(",")}
              >{p.label}</button>
            {/each}
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <label
            class="flex items-center gap-2"
            title="Group images by aspect ratio so non-square frames aren't center-cropped. Recommended on; turn off only if your dataset is already uniform."
          >
            <input
              type="checkbox" checked={cfg.enable_ar_bucket}
              onchange={(e) => patchField("enable_ar_bucket", (e.target as HTMLInputElement).checked)}
              class="w-4 h-4 rounded bg-ink-950 border-ink-700 accent-accent-500"
            />
            <span class="flex items-center gap-1 text-slate-300">
              <span>enable_ar_bucket</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
          </label>
          <label
            class="block"
            title="Smallest aspect ratio (width / height) accepted into a bucket. 0.5 = portrait 1:2. Frames outside [min_ar, max_ar] are clamped to the nearest bucket."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>min_ar</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" step="any" value={cfg.min_ar}
              onchange={(e) => patchField("min_ar", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="Largest aspect ratio (width / height) accepted into a bucket. 2.0 = landscape 2:1."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>max_ar</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" step="any" value={cfg.max_ar}
              onchange={(e) => patchField("max_ar", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="Number of aspect-ratio buckets to split [min_ar, max_ar] into, per resolution. More = finer matching to native frame shape, but smaller buckets train less efficiently."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>num_ar_buckets</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="1" step="1" value={cfg.num_ar_buckets}
              onchange={(e) => patchField("num_ar_buckets", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
        </div>
      </div>

      <!-- Schedule + Anima specifics -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <h3 class="text-sm font-medium text-slate-200 mb-3">Schedule &amp; Anima specifics</h3>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
          <label
            class="block"
            title="Total epochs to train. Anima recipe targets 40 (style) / 60 (character). 'Continue last' from the Run tab can extend a finished run if you raise this."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>epochs</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="1" step="1" value={cfg.epochs}
              onchange={(e) => patchField("epochs", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="Run a held-out eval pass every N epochs to log validation loss. Doesn't affect training; purely a metric for spotting overfitting."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>eval_every_n_epochs</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="1" step="1" value={cfg.eval_every_n_epochs}
              onchange={(e) => patchField("eval_every_n_epochs", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="Write a LoRA checkpoint to disk every N epochs. Lower = more snapshots to compare, but more disk used. Combine with 'keep last N' below to cap retention."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>save_every_n_epochs</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="1" step="1" value={cfg.save_every_n_epochs}
              onchange={(e) => patchField("save_every_n_epochs", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block"
            title="Anima-specific timestep weighting. Higher = more loss weight on mid-noise timesteps where the model learns global structure. Leave at the recipe default unless you know what you're doing."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>sigmoid_scale</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" step="any" value={cfg.sigmoid_scale}
              onchange={(e) => patchField("sigmoid_scale", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <label
            class="block col-span-2"
            title="Learning rate for an optional LoRA on the text encoder. Anima's recipe explicitly recommends 0 — non-zero values bleed into the text-conditioning and dilute the style. Only touch if you're experimenting."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>llm_adapter_lr (keep at 0)</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" step="any" value={cfg.llm_adapter_lr}
              onchange={(e) => patchField("llm_adapter_lr", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono {cfg.llm_adapter_lr !== 0 ? 'border-amber-700' : ''}"
            />
            {#if cfg.llm_adapter_lr !== 0}
              <span class="block text-[10px] text-amber-400 mt-1">
                ⚠ Non-zero llm_adapter_lr causes "style dilution" on Anima — recommended value is 0.
              </span>
            {/if}
          </label>
        </div>
      </div>

      <!-- Captioning -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <h3 class="text-sm font-medium text-slate-200 mb-3">Captioning</h3>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
          <label
            class="block"
            title="How tags + LLM descriptions combine into the training caption. tags = WD14 tags only; nl = LLM description only; mixed = description first, tags appended (recommended — gives the model both signals)."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>caption_mode</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <select
              value={cfg.caption_mode}
              onchange={(e) => patchField("caption_mode", (e.target as HTMLSelectElement).value)}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            >
              <option value="tags">tags only</option>
              <option value="nl">natural language only</option>
              <option value="mixed">mixed (recommended)</option>
            </select>
          </label>
          <label
            class="block"
            title="Per-step probability (0–100) of dropping each WD14 tag from the caption. Acts as regularization — prevents the model from binding the concept to a specific tag. 10–15% is typical."
          >
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>tag_dropout %</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              type="number" min="0" max="100" step="1" value={cfg.tag_dropout_pct}
              onchange={(e) => patchField("tag_dropout_pct", Number((e.target as HTMLInputElement).value))}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
            />
          </label>
          <div
            class="block text-xs col-span-2 md:col-span-1"
            title="trigger_token is now per-character — edit it under Training → Identity."
          >
            <span class="text-[10px] uppercase tracking-wide text-slate-500">
              trigger_token
            </span>
            <button
              type="button"
              onclick={onopenidentity}
              class="block w-full mt-1 px-3 py-1.5 bg-ink-950 border border-dashed border-ink-700 rounded text-slate-400 text-left hover:border-accent-500 hover:text-slate-200"
            >Edit per character → Identity tab</button>
          </div>
        </div>
      </div>

      <TrainingSamplesPanel {cfg} />

      <!-- Retention -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <h3 class="text-sm font-medium text-slate-200 mb-3">Checkpoint retention</h3>
        <label
          class="block text-xs"
          title="Cap on how many epoch (LoRA) checkpoints survive after a run finishes. The newest N are kept; older ones are deleted. 0 keeps every epoch. DeepSpeed resume state is always trimmed to the latest, regardless of this number."
        >
          <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
            <span>keep last N checkpoints (0 = keep all)</span>
            <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
          </span>
          <input
            type="number" min="0" step="1" value={cfg.keep_last_n_checkpoints}
            onchange={(e) => patchField("keep_last_n_checkpoints", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono max-w-[12rem]"
          />
          <span class="block text-[10px] text-slate-600 mt-1">
            Older checkpoints are pruned at the end of each run. Default
            (<code class="text-slate-400">0</code>) keeps every epoch adapter;
            DeepSpeed resume state is always trimmed to the latest.
          </span>
        </label>
      </div>

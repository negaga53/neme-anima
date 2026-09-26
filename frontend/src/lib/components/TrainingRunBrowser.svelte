<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { toasts } from "$lib/stores/toasts.svelte";
  import { trainingStore } from "$lib/stores/training.svelte";
  import TrainingSamplesGrid from "./TrainingSamplesGrid.svelte";
  import type { TrainingConfig, TrainingRun, TrainingRunState } from "$lib/types";

  type Props = {
    /** Live config (for `hasRoomToTrain` epoch comparison + the Continue tooltip). */
    cfg: TrainingConfig;
    /** Validation problems (gates the per-run Continue + drives the banner). */
    problems: string[];
    /** Current run status — `trainingStore.status?.state ?? null`. */
    runState: TrainingRunState | null;
    /** Whether a run is in flight (disables delete/continue, drives badges). */
    isRunning: boolean;
    /** Header busy flag (disables the per-run Continue mid start/stop/resume). */
    busy: boolean;
    /** Resume a run by name (header's continueRun). */
    oncontinue: (runName: string) => void;
    /** Jump to the Settings sub-tab (the problems banner's "Open Settings →"). */
    onopensettings: () => void;
  };
  const { cfg, problems, runState, isRunning, busy, oncontinue, onopensettings }: Props = $props();

  let runs = $derived(trainingStore.runs);
  let log = $derived(trainingStore.log);

  async function copyPath(path: string) {
    try {
      await navigator.clipboard.writeText(path);
    } catch (e) {
      // Clipboard API can fail outside secure contexts — fall back to a prompt.
      window.prompt("Copy path:", path);
    }
  }

  // The download name the export endpoint will produce. Mirrors the backend's
  // _sanitize_filename (training.py): non [A-Za-z0-9._-] runs → "_", trim
  // leading/trailing "._-", fall back to the slug when empty.
  function exportFilename(ckptName: string): string {
    const name = projectsStore.active?.name ?? "";
    const cleaned = name.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[._-]+|[._-]+$/g, "");
    const stem = cleaned || (projectsStore.active?.slug ?? "");
    return `${stem}-${ckptName}.safetensors`;
  }

  // Shortcut: export a run's newest LoRA epoch without expanding it. We fetch
  // the checkpoint list on demand to get the exact name + diffusion-pipe subdir
  // the export URL needs, pick the highest-epoch LoRA, then trigger a download.
  let exportingRun = $state<string | null>(null);
  async function exportLatestEpoch(runName: string) {
    const slug = projectsStore.active?.slug;
    if (!slug || exportingRun) return;
    exportingRun = runName;
    try {
      const data = await api.listTrainingCheckpoints(slug, runName);
      const loras = data.checkpoints.filter((c) => c.epoch != null);
      if (loras.length === 0) {
        toasts.info("This run has no saved LoRA epoch to export yet.");
        return;
      }
      const latest = loras.reduce((a, b) => ((b.epoch ?? -1) > (a.epoch ?? -1) ? b : a));
      const link = document.createElement("a");
      link.href = api.exportCheckpointUrl(slug, runName, latest.name, latest.subdir);
      link.download = "";
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (e) {
      toasts.error("Export failed: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      exportingRun = null;
    }
  }

  // A run is only worth resuming if cfg.epochs is higher than what's
  // already been trained — otherwise diffusion-pipe would still grind out
  // one more epoch and save it. Returns true when there is real work left.
  function hasRoomToTrain(r: TrainingRun | null): boolean {
    if (!r || r.latest_epoch == null) return true; // unknown → allow
    return cfg.epochs > r.latest_epoch;
  }

  // ---- run/checkpoint browser -----------------------------------------
  let expandedRun = $state<string | null>(null);
  let runCheckpoints = $state<Record<string, Awaited<ReturnType<typeof api.listTrainingCheckpoints>>>>({});
  async function toggleRun(name: string) {
    if (expandedRun === name) { expandedRun = null; return; }
    expandedRun = name;
    if (!runCheckpoints[name] && trainingStore.slug) {
      runCheckpoints[name] = await api.listTrainingCheckpoints(trainingStore.slug, name);
    }
  }
  async function refreshExpandedRun() {
    if (expandedRun && trainingStore.slug) {
      runCheckpoints[expandedRun] = await api.listTrainingCheckpoints(trainingStore.slug, expandedRun);
    }
  }
  async function deleteCheckpoint(runName: string, ckptName: string, subdir: string) {
    if (!confirm(`Delete checkpoint ${subdir ? subdir + "/" : ""}${ckptName}?`)) return;
    await trainingStore.deleteCheckpoint(runName, ckptName, subdir);
    await refreshExpandedRun();
  }
  async function deleteRun(runName: string) {
    if (!confirm(`Delete the entire run ${runName} and all its checkpoints?`)) return;
    await trainingStore.deleteRun(runName);
    if (expandedRun === runName) expandedRun = null;
  }

  // Auto-scroll the log panel.
  let logPanel: HTMLDivElement | null = $state(null);
  $effect(() => {
    void log.length;
    if (logPanel) logPanel.scrollTop = logPanel.scrollHeight;
  });

  // Color theme keyed off run status. Returns CSS class fragments so we
  // can drive both the badge and the surrounding card border in lockstep.
  function statusTheme(s: string): { tone: string; pill: string; bar: string; border: string; label: string } {
    switch (s) {
      case "running":
        return {
          tone: "emerald",
          pill: "bg-emerald-900/50 text-emerald-300 border border-emerald-800",
          bar: "bg-emerald-500",
          border: "border-emerald-800/60",
          label: "Running",
        };
      case "starting":
        return {
          tone: "amber",
          pill: "bg-amber-900/50 text-amber-300 border border-amber-800",
          bar: "bg-amber-500",
          border: "border-amber-800/60",
          label: "Starting…",
        };
      case "sampling":
        return {
          tone: "sky",
          pill: "bg-sky-900/50 text-sky-300 border border-sky-800",
          bar: "bg-sky-500",
          border: "border-sky-800/60",
          label: "Sampling…",
        };
      case "stopping":
        return {
          tone: "amber",
          pill: "bg-amber-900/50 text-amber-300 border border-amber-800",
          bar: "bg-amber-500",
          border: "border-amber-800/60",
          label: "Stopping…",
        };
      case "stopped":
        return {
          tone: "slate",
          pill: "bg-slate-800 text-slate-300 border border-slate-700",
          bar: "bg-slate-500",
          border: "border-slate-700",
          label: "Stopped",
        };
      case "finished":
        return {
          tone: "emerald",
          pill: "bg-emerald-900/60 text-emerald-200 border border-emerald-700",
          bar: "bg-emerald-500",
          border: "border-emerald-800/60",
          label: "✓ Finished",
        };
      case "failed":
        return {
          tone: "red",
          pill: "bg-red-900/60 text-red-200 border border-red-700",
          bar: "bg-red-500",
          border: "border-red-800/60",
          label: "✗ Failed",
        };
      default:
        return {
          tone: "slate",
          pill: "bg-slate-800 text-slate-400 border border-slate-700",
          bar: "bg-slate-500",
          border: "border-ink-700",
          label: s,
        };
    }
  }

  // Compute (epoch%, label) for the current-run progress bar.
  function progressInfo(rs: NonNullable<typeof runState>): { pct: number; label: string } {
    if (rs.sampling) {
      const queued = Math.max(0, rs.sampling.pending - 1);
      const pct = rs.total_epochs ? Math.min(100, Math.round(((rs.epoch ?? 0) / rs.total_epochs) * 100)) : 0;
      return { pct: rs.status === "sampling" ? 100 : pct, label: `sampling epoch ${rs.sampling.epoch}${queued ? ` (${queued} queued)` : ""}` };
    }
    const total = rs.total_epochs ?? 0;
    const cur = rs.epoch ?? 0;
    if (rs.status === "finished") return { pct: 100, label: total ? `${total} / ${total} epochs` : "completed" };
    if (!total) return { pct: 0, label: cur ? `epoch ${cur}` : "preparing…" };
    const pct = Math.max(0, Math.min(100, Math.round((cur / total) * 100)));
    return { pct, label: `${cur} / ${total} epochs` };
  }

  function fmtBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
  }

  function fmtElapsed(startIso: string, endIso?: string | null): string {
    const t0 = new Date(startIso).getTime();
    const t1 = endIso ? new Date(endIso).getTime() : Date.now();
    const sec = Math.max(0, Math.round((t1 - t0) / 1000));
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
  }

  let pollHandle: ReturnType<typeof setInterval> | null = null;
  onMount(() => {
    pollHandle = setInterval(async () => {
      if (!trainingStore.slug) return;
      try {
        const [st] = await Promise.all([
          api.getTrainingStatus(trainingStore.slug),
          trainingStore.refreshRuns(),
        ]);
        trainingStore.status = st;
        if (expandedRun) {
          // Refresh the open run's checkpoints in-place.
          runCheckpoints[expandedRun] = await api.listTrainingCheckpoints(
            trainingStore.slug, expandedRun,
          );
        }
      } catch {
        // ignore — poll resumes next tick
      }
    }, 5000);
  });
  onDestroy(() => { if (pollHandle) clearInterval(pollHandle); });
</script>

      <!-- ====================================================== -->
      <!-- ==================== RUN SUB-TAB ===================== -->
      <!-- ====================================================== -->

      {#if problems.length > 0}
        <div class="bg-amber-900/30 border border-amber-800 rounded-xl p-3 mb-3 text-xs text-amber-200">
          <div class="font-medium mb-1">Cannot start a run yet:</div>
          <ul class="list-disc list-inside space-y-0.5">
            {#each problems as p}
              <li>{p}</li>
            {/each}
          </ul>
          <button
            type="button"
            onclick={onopensettings}
            class="mt-2 text-amber-300 hover:text-amber-100 underline text-[11px]"
          >Open Settings →</button>
        </div>
      {/if}

      <!-- ============ Current run card ============ -->
      {#if runState}
        {@const theme = statusTheme(runState.status)}
        {@const prog = progressInfo(runState)}
        <div class="bg-ink-900 border-2 {theme.border} rounded-xl p-4 mb-4">
          <div class="flex items-start justify-between gap-3 mb-3">
            <div class="min-w-0 flex-1">
              <div class="flex items-center gap-2 mb-1">
                <span class="text-[10px] uppercase tracking-wide text-slate-500">Current run</span>
                <span class="text-[10px] font-mono text-slate-400 truncate">{runState.run_name}</span>
              </div>
              <div class="flex items-center gap-3 flex-wrap">
                <span class="text-sm px-2.5 py-0.5 rounded-full {theme.pill}">
                  {theme.label}
                </span>
                {#if runState.resumed_from}
                  <span class="text-[11px] text-slate-500">
                    resumed from <span class="font-mono text-slate-400">{runState.resumed_from}</span>
                  </span>
                {/if}
              </div>
            </div>
            <div class="text-right text-[11px] text-slate-500 shrink-0">
              {#if runState.finished_at}
                <div>
                  finished {new Date(runState.finished_at).toLocaleString()}
                </div>
                <div>
                  ran for {fmtElapsed(runState.started_at, runState.finished_at)}
                </div>
              {:else}
                <div>
                  started {new Date(runState.started_at).toLocaleString()}
                </div>
                <div>
                  elapsed {fmtElapsed(runState.started_at)}
                </div>
              {/if}
            </div>
          </div>

          <!-- Progress bar -->
          <div class="mb-3">
            <div class="flex items-center justify-between text-[11px] mb-1">
              <span class="text-slate-400 font-mono">{prog.label}</span>
              <span class="text-slate-300 font-mono">{prog.pct}%</span>
            </div>
            <div class="w-full h-2 bg-ink-950 rounded-full overflow-hidden">
              <div
                class="h-full {theme.bar} transition-all duration-500"
                style="width: {prog.pct}%"
              ></div>
            </div>
          </div>

          <!-- Live numerics -->
          <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">step</div>
              <div class="font-mono text-slate-200">{runState.step ?? "—"}</div>
            </div>
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">loss</div>
              <div class="font-mono text-slate-200">{runState.loss != null ? runState.loss.toFixed(4) : "—"}</div>
            </div>
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">epoch</div>
              <div class="font-mono text-slate-200">
                {runState.epoch ?? "—"}{runState.total_epochs ? ` / ${runState.total_epochs}` : ""}
              </div>
            </div>
          </div>

          {#if runState.error}
            <div class="mt-3 px-3 py-2 bg-red-950/40 border border-red-900 rounded text-xs text-red-300 break-words">
              {runState.error}
            </div>
          {/if}
          {#if runState.last_log_line}
            <div class="mt-2 text-[11px] font-mono text-slate-500 truncate">{runState.last_log_line}</div>
          {/if}
        </div>
      {/if}

      <!-- ============ Run history ============ -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-4">
        <h3 class="text-sm font-medium text-slate-200 mb-3">Run history</h3>

        {#if runs.length === 0}
          <p class="text-xs text-slate-500">
            No runs yet. Click <em>Start training</em> above to launch one.
          </p>
        {:else}
          <ul class="space-y-2 text-xs">
            {#each runs as r (r.name)}
              {@const expanded = expandedRun === r.name}
              {@const cur = runState && runState.run_name === r.name}
              {@const cps = runCheckpoints[r.name]}
              <li class="bg-ink-950 border border-ink-800 rounded-lg overflow-hidden">
                <!-- Run header row -->
                <div class="flex items-center gap-2 p-2.5">
                  <button
                    type="button"
                    onclick={() => toggleRun(r.name)}
                    class="flex-1 text-left flex items-center gap-2 min-w-0"
                  >
                    <span class="text-slate-500 text-[10px] w-3">{expanded ? "▼" : "▶"}</span>
                    <span class="font-mono text-slate-200 truncate">{r.name}</span>
                    {#if cur}
                      {@const t = statusTheme(runState.status)}
                      <span class="text-[10px] px-1.5 py-0.5 rounded {t.pill}">{t.label}</span>
                    {/if}
                  </button>

                  <div class="flex items-center gap-3 text-[11px] text-slate-500 shrink-0">
                    {#if r.latest_epoch != null}
                      <span title="Highest saved epoch">
                        epoch <span class="text-slate-300 font-mono">{r.latest_epoch}</span>
                      </span>
                    {/if}
                    <span title="Number of epoch checkpoints saved">
                      {r.checkpoints} ckpt{r.checkpoints === 1 ? "" : "s"}
                    </span>
                    <span title="Total disk usage including DeepSpeed state">
                      {fmtBytes(r.total_size_bytes)}
                    </span>

                    {#if r.resumable_subdir && !isRunning}
                      {@const room = hasRoomToTrain(r)}
                      <button
                        type="button"
                        onclick={() => oncontinue(r.name)}
                        disabled={busy || problems.length > 0 || !room}
                        class="px-2.5 py-1 text-[11px] rounded bg-accent-700 hover:bg-accent-600 text-white btn-disabled"
                        title={room
                          ? "Continue training from this run's last saved state"
                          : `Already at epoch ${r.latest_epoch} / ${cfg?.epochs} — raise 'epochs' in Settings to continue`}
                      >Continue</button>
                    {:else if !r.resumable_subdir}
                      <span class="text-[10px] text-slate-600 italic" title="No DeepSpeed save — can't resume">
                        not resumable
                      </span>
                    {/if}

                    {#if r.checkpoints > 0}
                      <button
                        type="button"
                        onclick={() => exportLatestEpoch(r.name)}
                        disabled={exportingRun === r.name}
                        class="w-6 h-6 inline-flex items-center justify-center rounded text-slate-400 hover:text-slate-100 hover:bg-ink-800 disabled:opacity-40"
                        title={r.latest_checkpoint
                          ? `Export latest epoch as ${exportFilename(r.latest_checkpoint)}`
                          : "Export latest epoch"}
                        aria-label="Export latest epoch"
                      >
                        <svg viewBox="0 0 24 24" class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                          <polyline points="7 10 12 15 17 10"/>
                          <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                      </button>
                    {/if}
                    <button
                      type="button"
                      onclick={() => copyPath(r.path)}
                      class="w-6 h-6 inline-flex items-center justify-center rounded text-slate-400 hover:text-slate-100 hover:bg-ink-800"
                      title="Copy run path: {r.path}"
                      aria-label="Copy run path"
                    >
                      <svg viewBox="0 0 24 24" class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="11" height="11" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                      </svg>
                    </button>
                    <button
                      type="button"
                      onclick={() => deleteRun(r.name)}
                      disabled={cur && isRunning}
                      class="w-6 h-6 inline-flex items-center justify-center rounded text-red-400 hover:text-white hover:bg-red-700 disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-red-400"
                      title="Delete this run"
                      aria-label="Delete this run"
                    >
                      <svg viewBox="0 0 24 24" class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                        <path d="M18 6L6 18M6 6l12 12"/>
                      </svg>
                    </button>
                  </div>
                </div>

                {#if expanded}
                  <div class="border-t border-ink-800 px-2 py-2 bg-black/20">
                    {#if !cps}
                      <p class="text-slate-500 text-[11px] px-2">Loading…</p>
                    {:else if cps.checkpoints.length === 0}
                      <p class="text-slate-500 text-[11px] px-2">No checkpoints saved in this run yet.</p>
                    {:else}
                      <ul class="space-y-1">
                        {#each cps.checkpoints.slice().reverse() as cp (cp.subdir + "/" + cp.name)}
                          {@const isStep = cp.epoch == null && cp.step != null}
                          <li class="flex items-center justify-between gap-2 px-2 py-1 hover:bg-ink-900 rounded">
                            <div class="flex items-center gap-2 min-w-0">
                              {#if isStep}
                                <span class="text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-slate-800 text-slate-400" title="DeepSpeed pipeline state — used for resume">
                                  state
                                </span>
                              {:else}
                                <span class="text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-emerald-900/40 text-emerald-300" title="LoRA adapter output — usable for inference">
                                  lora
                                </span>
                              {/if}
                              <span class="font-mono text-slate-300 truncate">{cp.name}</span>
                            </div>
                            <div class="flex items-center gap-2 text-[10px] text-slate-500 shrink-0">
                              <span>{fmtBytes(cp.size_bytes)}</span>
                              <span>{new Date(cp.modified_at).toLocaleString()}</span>
                              {#if !isStep}
                                <a
                                  href={api.exportCheckpointUrl(projectsStore.active!.slug, r.name, cp.name, cp.subdir)}
                                  download
                                  class="w-5 h-5 inline-flex items-center justify-center rounded text-slate-400 hover:text-slate-100 hover:bg-ink-800"
                                  title={`Export as ${exportFilename(cp.name)}`}
                                  aria-label="Export checkpoint"
                                >
                                  <svg viewBox="0 0 24 24" class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="7 10 12 15 17 10"/>
                                    <line x1="12" y1="15" x2="12" y2="3"/>
                                  </svg>
                                </a>
                              {/if}
                              <button
                                type="button"
                                onclick={() => copyPath(cp.path)}
                                class="w-5 h-5 inline-flex items-center justify-center rounded text-slate-400 hover:text-slate-100 hover:bg-ink-800"
                                title="Copy checkpoint path: {cp.path}"
                                aria-label="Copy checkpoint path"
                              >
                                <svg viewBox="0 0 24 24" class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                  <rect x="9" y="9" width="11" height="11" rx="2" ry="2"/>
                                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                                </svg>
                              </button>
                              <button
                                type="button"
                                disabled={isRunning}
                                onclick={() => deleteCheckpoint(r.name, cp.name, cp.subdir)}
                                class="w-5 h-5 inline-flex items-center justify-center rounded text-red-400 hover:text-white hover:bg-red-700 disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-red-400"
                                title="Delete this checkpoint"
                                aria-label="Delete this checkpoint"
                              >
                                <svg viewBox="0 0 24 24" class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                                  <path d="M18 6L6 18M6 6l12 12"/>
                                </svg>
                              </button>
                            </div>
                          </li>
                        {/each}
                      </ul>
                    {/if}
                    <TrainingSamplesGrid
                      runName={r.name}
                      samplingEnabled={cfg.sample_prompts.some((p) => p.trim())}
                    />
                  </div>
                {/if}
              </li>
            {/each}
          </ul>
        {/if}
      </div>

      <!-- ============ Live log ============ -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-medium text-slate-200">Trainer log</h3>
          <button
            type="button"
            onclick={() => trainingStore.refreshLog()}
            class="text-xs text-slate-500 hover:text-slate-300"
          >Reload</button>
        </div>
        <div
          bind:this={logPanel}
          class="bg-black/60 border border-ink-800 rounded p-2 h-64 overflow-y-auto font-mono text-[11px] leading-snug"
        >
          {#if log.length === 0}
            <span class="text-slate-600">No log lines yet.</span>
          {:else}
            {#each log as l, i (i)}
              <div class="{l.stream === 'stderr' ? 'text-amber-300' : 'text-slate-400'}">{l.line}</div>
            {/each}
          {/if}
        </div>
      </div>

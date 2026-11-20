<script lang="ts">
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { toasts } from "$lib/stores/toasts.svelte";
  import { trainingStore } from "$lib/stores/training.svelte";
  import type { TrainingConfig, TrainingRun, TrainingRunState } from "$lib/types";
  import TrainingConfigPanel from "./TrainingConfigPanel.svelte";
  import TrainingRunBrowser from "./TrainingRunBrowser.svelte";
  import TrainingDatasetPreview from "./TrainingDatasetPreview.svelte";
  import TrainingIdentityPanel from "./TrainingIdentityPanel.svelte";

  // Sub-tab state. Persisted on the component, not the store, since the
  // user's choice is per-session.
  type SubTab = "run" | "dataset" | "settings" | "identity";
  let subtab = $state<SubTab>("run");

  let cfg = $derived(trainingStore.configResp?.config ?? null);
  let problems = $derived(trainingStore.configResp?.problems ?? []);
  let status = $derived(trainingStore.status);
  let runs = $derived(trainingStore.runs);
  let runState = $derived(status?.state ?? null);
  let isRunning = $derived(!!status?.running);
  // A first Stop asks the trainer to save resume state and exit; while that's
  // pending the button escalates to a force-kill.
  let isStopping = $derived(isRunning && runState?.status === "stopping");
  let canStart = $derived(!isRunning && problems.length === 0 && !!cfg);
  // The first run that has a resumable DeepSpeed state. Used to enable
  // the global "Continue last run" affordance in the header.
  let resumableRun = $derived<TrainingRun | null>(
    runs.find((r) => !!r.resumable_subdir) ?? null,
  );
  // A run is only worth resuming if cfg.epochs is higher than what's
  // already been trained — otherwise diffusion-pipe would still grind out
  // one more epoch and save it. Returns true when there is real work left.
  function hasRoomToTrain(r: TrainingRun | null): boolean {
    if (!r || r.latest_epoch == null) return true; // unknown → allow
    if (!cfg) return false;
    return cfg.epochs > r.latest_epoch;
  }
  let canResume = $derived(
    !isRunning
      && !!resumableRun
      && problems.length === 0
      && hasRoomToTrain(resumableRun),
  );

  let busy = $state(false);
  async function doStart() {
    busy = true;
    try { await trainingStore.start(); }
    catch (e) { toasts.error(`Start failed: ${e}`); }
    finally { busy = false; }
  }
  async function doStop() {
    busy = true;
    try { await trainingStore.stop(); }
    catch (e) { toasts.error(`Stop failed: ${e}`); }
    finally { busy = false; }
  }
  async function continueRun(runName?: string) {
    busy = true;
    try {
      await trainingStore.resume(runName ? { run_dir_name: runName } : {});
      // Switch to the Run sub-tab so the user immediately sees what's happening.
      subtab = "run";
    } catch (e) {
      toasts.error(`Continue failed: ${e}`);
    } finally {
      busy = false;
    }
  }

  // Trigger the store load whenever the active project changes.
  let activeSlug = $derived(projectsStore.active?.slug ?? null);
  $effect(() => {
    const slug = activeSlug;
    if (!slug) return;
    void trainingStore.setProject(slug);
  });
</script>

<div class="mt-4 max-w-4xl mx-auto">
  <div class="flex items-center justify-between mb-4 gap-4">
    <h2 class="text-base font-semibold text-slate-200">LoRA training</h2>

    <!-- top-right control row -->
    <div class="flex items-center gap-2 shrink-0">
      {#if isRunning}
        <button
          type="button"
          onclick={doStop}
          disabled={busy}
          class="px-4 py-1.5 text-xs rounded bg-red-700 hover:bg-red-600 text-white disabled:opacity-50"
          title={isStopping
            ? "The trainer is saving resume state before exiting. Click to kill it now (loses progress since the last snapshot)."
            : "Save resume state after the current step, then stop. Continue picks up from there."}
        >{isStopping ? "Force stop" : "Stop"}</button>
      {:else}
        <button
          type="button"
          onclick={() => continueRun(resumableRun?.name)}
          disabled={!canResume || busy}
          class="px-3 py-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-100 border border-ink-700 btn-disabled"
          title={canResume
            ? `Continue ${resumableRun?.name} from epoch ${resumableRun?.latest_epoch ?? "?"}`
            : !resumableRun
              ? "No prior run with a resumable state"
              : !hasRoomToTrain(resumableRun)
                ? `Already at epoch ${resumableRun?.latest_epoch} / ${cfg?.epochs} — raise 'epochs' in Settings to continue`
                : (problems[0] ?? "Cannot continue right now")}
        >Continue last</button>
        <button
          type="button"
          onclick={doStart}
          disabled={!canStart || busy}
          class="px-4 py-1.5 text-xs rounded gradient-accent text-white btn-disabled"
          title={canStart ? "Start a new training run" : (problems[0] ?? "Provide all model paths first")}
        >{busy ? "Starting…" : "Start training"}</button>
      {/if}
    </div>
  </div>

  {#if !cfg}
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 text-sm text-slate-400">
      Loading training config…
    </div>
  {:else}
    <!-- ============ sub-tabs ============ -->
    <div class="flex gap-1 mb-3 border-b border-ink-800">
      {#each [{ k: "run", label: "Run" }, { k: "dataset", label: "Dataset" }, { k: "identity", label: "Identity" }, { k: "settings", label: "Settings" }] as t}
        <button
          type="button"
          onclick={() => (subtab = t.k as SubTab)}
          class="px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors
            {subtab === t.k
              ? 'border-accent-500 text-slate-100'
              : 'border-transparent text-slate-500 hover:text-slate-300'}"
        >{t.label}{t.k === "settings" && problems.length > 0 ? ` (${problems.length})` : ""}</button>
      {/each}
    </div>

    {#if subtab === "run"}
      <TrainingRunBrowser
        {cfg}
        {problems}
        {runState}
        {isRunning}
        {busy}
        oncontinue={(name) => continueRun(name)}
        onopensettings={() => (subtab = "settings")}
      />
    {:else if subtab === "dataset"}
      <TrainingDatasetPreview />
    {:else if subtab === "identity"}
      <TrainingIdentityPanel />
    {:else if subtab === "settings"}
      <TrainingConfigPanel {cfg} onopenidentity={() => (subtab = "identity")} />
    {/if}
  {/if}
</div>

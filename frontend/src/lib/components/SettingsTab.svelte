<script lang="ts">
  import { onMount, untrack } from "svelte";
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import TagBlacklistInput from "$lib/components/TagBlacklistInput.svelte";

  // Mirror of llm.DEFAULT_PROMPT in the backend. Kept here so the textarea
  // can pre-fill with the actual prompt instead of an empty placeholder —
  // the user wanted to be able to edit it as a starting point. Keep these
  // two strings in sync; the backend treats an empty saved prompt as
  // "use default" so the constant only matters for the editing UX.
  const DEFAULT_LLM_PROMPT =
    "Describe this image in 1-2 sentences for a LoRA training caption. " +
    "Focus on the subject's pose, clothing, expression, background, lighting, " +
    "and any distinctive details. Be concise, factual, and avoid speculating " +
    "about names, intent, or off-camera context.";

  // The threshold sections we expose. Pulled from src/neme_anima/config.py.
  const SECTIONS: {
    key: string;
    label: string;
    fields: {
      name: string;
      type: "number" | "boolean";
      placeholder: string;
      help: string;
    }[];
  }[] = [
    { key: "scene", label: "Scene detection", fields: [
        { name: "threshold", type: "number", placeholder: "27.0",
          help: "PySceneDetect content threshold. Higher = fewer scene cuts (only big visual changes split). Default 27." },
        { name: "min_scene_len_frames", type: "number", placeholder: "8",
          help: "Minimum scene length in frames. Shorter scenes are merged into the previous one." },
      ]},
    { key: "detect", label: "Detection", fields: [
        { name: "person_score_min", type: "number", placeholder: "0.35",
          help: "Minimum YOLO confidence (0–1) to keep a person detection. Higher = fewer false positives but more missed people." },
        { name: "frame_stride", type: "number", placeholder: "4",
          help: "Run detection every Nth frame. 4 at 24 fps = 6 effective fps. Higher = faster but coarser." },
        { name: "detect_faces", type: "boolean", placeholder: "false",
          help: "Also run face detection. Adds ~45% to detect time and isn't used by the current matcher; leave off unless you're experimenting." },
      ]},
    { key: "track", label: "Tracking", fields: [
        { name: "track_thresh", type: "number", placeholder: "0.25",
          help: "ByteTrack detection threshold for starting new tracks. Higher = fewer, more confident tracks." },
        { name: "match_thresh", type: "number", placeholder: "0.8",
          help: "IoU threshold for associating detections with existing tracks. Higher = stricter matching, more track breaks." },
        { name: "min_tracklet_len", type: "number", placeholder: "3",
          help: "Discard tracklets shorter than this many frames. Filters out flickers and one-off detections." },
      ]},
    { key: "identify", label: "Identification", fields: [
        { name: "body_max_distance_strict", type: "number", placeholder: "0.15",
          help: "CCIP body-embedding distance below this counts as a high-confidence match to a reference. Lower = stricter." },
        { name: "body_max_distance_loose", type: "number", placeholder: "0.20",
          help: "Looser CCIP distance for medium-confidence matches. Should be ≥ strict." },
        { name: "sample_frames_per_tracklet", type: "number", placeholder: "5",
          help: "Frames sampled per tracklet to compute its averaged body embedding. More = stabler match, slower." },
      ]},
    { key: "frame_select", label: "Frame selection", fields: [
        { name: "candidate_cap", type: "number", placeholder: "20",
          help: "For long tracklets, score this many evenly-spaced frames as candidates. Caps the per-tracklet cost." },
        { name: "dedup_min_frame_gap", type: "number", placeholder: "4",
          help: "Picked frames must be at least this many frames apart, so kept frames don't all land on the same instant." },
        { name: "top_k_short", type: "number", placeholder: "1",
          help: "How many frames to keep from a short tracklet (≈ ≤1s of footage)." },
        { name: "top_k_long", type: "number", placeholder: "3",
          help: "How many frames to keep from a long tracklet (≥ 5s of footage)." },
      ]},
    { key: "crop", label: "Crop", fields: [
        { name: "longest_side", type: "number", placeholder: "1024",
          help: "Resize each saved crop so its longest edge is this many pixels. Aspect ratio is preserved." },
        { name: "pad_ratio", type: "number", placeholder: "0.10",
          help: "Extra padding around the person bbox, as a fraction of bbox size. 0.10 = 10% margin on each side." },
      ]},
    { key: "tag", label: "Tagging", fields: [
        { name: "general_threshold", type: "number", placeholder: "0.35",
          help: "WD14 confidence cutoff for general tags. Lower = more tags (noisier); higher = fewer (cleaner)." },
        { name: "character_threshold", type: "number", placeholder: "0.85",
          help: "WD14 confidence cutoff for character tags. Kept high to avoid false character matches." },
      ]},
    { key: "dedup", label: "Dedup (kept crops)", fields: [
        { name: "distance_threshold", type: "number", placeholder: "0.02",
          help: "CCIP distance below which two crops are considered duplicates. Always-on dedup runs between identify and tag — this knob just tunes how aggressive it is. Default 0.02 only collapses near-pixel-identical crops; well below the 0.15 same-character threshold so different poses survive." },
        { name: "lookback_frames", type: "number", placeholder: "1000",
          help: "Maximum frame_idx delta between two crops for them to be duplicate-eligible. Restricts dedup to a sliding temporal window so visually similar but temporally distant shots stay distinct. Default 1000 ≈ 40 seconds at 24 fps. 0 = compare across the whole video (legacy)." },
        { name: "move_to_rejected", type: "boolean", placeholder: "true",
          help: "When on, duplicates move to rejected/ so you can recover them. Off = duplicates are deleted outright." },
      ]},
  ];

  let overrides = $state<Record<string, Record<string, unknown>>>(
    projectsStore.active?.thresholds_overrides ?? {},
  );

  $effect(() => {
    overrides = { ...(projectsStore.active?.thresholds_overrides ?? {}) };
  });

  function getValue(section: string, field: string): string {
    const v = overrides[section]?.[field];
    return v === undefined || v === null ? "" : String(v);
  }

  function setValue(section: string, field: string, raw: string, type: "number" | "boolean") {
    if (raw === "") {
      if (overrides[section]) delete overrides[section][field];
      if (overrides[section] && Object.keys(overrides[section]).length === 0) delete overrides[section];
      overrides = { ...overrides };
      return;
    }
    const value = type === "boolean" ? raw.toLowerCase() === "true" : Number(raw);
    overrides = {
      ...overrides,
      [section]: { ...(overrides[section] ?? {}), [field]: value },
    };
  }

  let saving = $state(false);
  let savedAt = $state<number | null>(null);

  let appVersion = $state<string>("");
  onMount(async () => {
    try {
      appVersion = (await api.getVersion()).version;
    } catch {
      /* leave blank if the endpoint is unavailable */
    }
  });

  let pauseBeforeTag = $state<boolean>(
    projectsStore.active?.pause_before_tag ?? true,
  );
  $effect(() => {
    pauseBeforeTag = projectsStore.active?.pause_before_tag ?? true;
  });

  let autoDeleteRejected = $state<boolean>(
    projectsStore.active?.auto_delete_rejected ?? false,
  );
  $effect(() => {
    autoDeleteRejected = projectsStore.active?.auto_delete_rejected ?? false;
  });

  let tagAutocomplete = $state<boolean>(
    projectsStore.active?.tag_autocomplete ?? true,
  );
  $effect(() => {
    tagAutocomplete = projectsStore.active?.tag_autocomplete ?? true;
  });

  // Probe whether the tag list has been downloaded so we can nudge the user to
  // run `neme-anima tags fetch` if it's missing.
  let tagListPresent = $state<boolean>(true);
  onMount(async () => {
    try {
      const resp = await fetch("/api/tags/vocabulary", { method: "HEAD" });
      tagListPresent = resp.ok;
    } catch {
      tagListPresent = false;
    }
  });

  let projectName = $state<string>(projectsStore.active?.name ?? "");
  $effect(() => {
    projectName = projectsStore.active?.name ?? "";
  });

  let blacklistTags = $derived<string[]>(
    (overrides.tag?.exclude_tags as string[] | undefined) ?? [],
  );

  function setBlacklist(next: string[]) {
    if (next.length === 0) {
      if (overrides.tag) {
        delete (overrides.tag as Record<string, unknown>).exclude_tags;
        if (Object.keys(overrides.tag).length === 0) delete overrides.tag;
      }
      overrides = { ...overrides };
      return;
    }
    overrides = {
      ...overrides,
      tag: { ...(overrides.tag ?? {}), exclude_tags: next },
    };
  }

  // ---------------- LLM tagging ----------------

  let llmEnabled = $state<boolean>(
    projectsStore.active?.llm?.enabled ?? false,
  );
  let llmEndpoint = $state<string>(
    projectsStore.active?.llm?.endpoint || "http://localhost:1234",
  );
  let llmModel = $state<string>(projectsStore.active?.llm?.model ?? "");
  // Optional bearer token. LMStudio doesn't need one — the input is left blank
  // by default — but OpenAI/OpenRouter/hosted vLLM do. Stored as plain text
  // in project.json (same trust boundary as the rest of the project config).
  let llmApiKey = $state<string>(projectsStore.active?.llm?.api_key ?? "");
  let llmApiKeyVisible = $state<boolean>(false);
  // Reasoning models spend part of the token budget on a think pass before the
  // caption. Opt-in because the request field that suppresses it isn't
  // universal — a strict server can reject the whole call. See LLMConfig.
  let llmDisableThinking = $state<boolean>(
    projectsStore.active?.llm?.disable_thinking ?? false,
  );
  // If the project hasn't customized the prompt, surface the default so the
  // user has something to edit rather than a blank textarea — they can save
  // it verbatim or customize.
  let llmPrompt = $state<string>(
    projectsStore.active?.llm?.prompt || DEFAULT_LLM_PROMPT,
  );
  let llmModelsAvailable = $state<string[]>(
    projectsStore.active?.llm?.model ? [projectsStore.active.llm.model] : [],
  );
  let llmDiscovering = $state(false);
  let llmDiscoverError = $state<string | null>(null);
  let llmDiscoverOk = $state<boolean>(false);

  $effect(() => {
    const llm = projectsStore.active?.llm;
    if (!llm) return;
    llmEnabled = llm.enabled;
    llmEndpoint = llm.endpoint || "http://localhost:1234";
    llmModel = llm.model || "";
    llmApiKey = llm.api_key || "";
    llmDisableThinking = llm.disable_thinking ?? false;
    // Empty saved prompt = "use the default", so show the default text in
    // the editor instead of a blank box; non-empty = user-customized.
    llmPrompt = llm.prompt || DEFAULT_LLM_PROMPT;
    // Seed the dropdown with the saved model so it has at least one option
    // before the user clicks Discover. Untracked so this effect's only
    // dependency is `projectsStore.active` — without that, mutating
    // llmModelsAvailable from discoverLLMModels() would re-fire the
    // effect and slam llmEndpoint back to the project's stored value,
    // wiping the URL the user just typed.
    untrack(() => {
      if (llm.model && !llmModelsAvailable.includes(llm.model)) {
        llmModelsAvailable = [llm.model];
      }
    });
  });

  // Toggle is gated on having picked a model — avoids the "enabled but no
  // model selected" footgun and matches the user's "disable if no model"
  // requirement without storing it as a separate flag.
  let llmCanEnable = $derived(!!llmModel.trim());

  async function discoverLLMModels() {
    llmDiscovering = true;
    llmDiscoverError = null;
    llmDiscoverOk = false;
    try {
      const resp = await api.discoverLLMModels(llmEndpoint.trim(), llmApiKey.trim());
      llmModelsAvailable = resp.models;
      llmDiscoverOk = true;
      if (resp.models.length === 1) {
        // Single-model server (common for self-hosted vLLM/ninfer boxes):
        // there is nothing to choose between, so take it even when a stale
        // model from a previous endpoint is still saved — otherwise the
        // dropdown lands on blank and the user has to pick the only option.
        llmModel = resp.models[0];
      } else if (llmModel && !resp.models.includes(llmModel)) {
        // If the previously-saved model isn't in the new list, blank it so the
        // user has to pick again — keeps the dropdown honest.
        llmModel = "";
      } else if (!llmModel && resp.models.length > 0) {
        // First-time discovery convenience: preselect the first model.
        llmModel = resp.models[0];
      }
    } catch (e) {
      llmDiscoverError = e instanceof Error ? e.message : String(e);
      llmModelsAvailable = [];
    } finally {
      llmDiscovering = false;
    }
  }

  async function save() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    saving = true;
    try {
      const wasAutoDeleteOff = !(projectsStore.active?.auto_delete_rejected ?? false);
      const turningAutoDeleteOn = autoDeleteRejected && wasAutoDeleteOff;
      const existingRejected = projectsStore.active?.rejected_count ?? 0;
      await api.patchProject(slug, {
        name: projectName.trim() || projectsStore.active?.name,
        thresholds_overrides: overrides,
        pause_before_tag: pauseBeforeTag,
        auto_delete_rejected: autoDeleteRejected,
        tag_autocomplete: tagAutocomplete,
        llm: {
          // Force disabled if no model selected — server-side has the same
          // guard but enforcing it here keeps the saved state self-consistent.
          enabled: llmEnabled && !!llmModel.trim(),
          endpoint: llmEndpoint.trim(),
          model: llmModel.trim(),
          // Save empty when the user hasn't customized — that lets future
          // upstream changes to DEFAULT_PROMPT take effect automatically
          // for projects that never edited the prompt.
          prompt: llmPrompt === DEFAULT_LLM_PROMPT ? "" : llmPrompt,
          api_key: llmApiKey.trim(),
          disable_thinking: llmDisableThinking,
        },
      });
      if (turningAutoDeleteOn && existingRejected > 0) {
        if (confirm(`Delete ${existingRejected} existing rejected frame${existingRejected === 1 ? "" : "s"}?`)) {
          await api.deleteRejectedFrames(slug);
        }
        // Cancel: keep the setting saved, skip the one-time deletion.
      }
      await projectsStore.load(slug);
      await projectsStore.refresh();
      savedAt = Date.now();
    } finally {
      saving = false;
    }
  }

  function resetSection(section: string) {
    if (overrides[section]) {
      delete overrides[section];
      overrides = { ...overrides };
    }
  }

  function resetAll() {
    overrides = {};
  }
</script>

<div class="mt-4 max-w-3xl mx-auto">
  <div class="flex items-center justify-between mb-4">
    <div class="flex items-baseline gap-2">
      <h2 class="text-base font-semibold text-slate-200">Per-project settings</h2>
      {#if appVersion}
        <span class="text-xs text-slate-500">v{appVersion}</span>
      {/if}
    </div>
    <div class="flex gap-2 items-center">
      {#if savedAt}
        <span class="text-xs text-emerald-400">saved</span>
      {/if}
      <button type="button" onclick={resetAll} class="text-xs text-slate-500 hover:text-slate-300">Reset thresholds</button>
      <button
        type="button"
        onclick={save}
        disabled={saving}
        class="px-4 py-1.5 text-xs rounded gradient-accent text-white disabled:opacity-50"
      >{saving ? "Saving…" : "Save"}</button>
    </div>
  </div>

  <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
    <h3 class="text-sm font-medium text-slate-200 mb-3">Project</h3>
    <label class="block">
      <span class="block text-[10px] uppercase tracking-wide text-slate-500">Display name</span>
      <input
        bind:value={projectName}
        placeholder="Project name"
        class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-sm focus:outline-none focus:border-accent-500"
      />
      <span class="block text-[10px] text-slate-600 mt-1">
        Renames the project's display name only. The folder on disk and its URL
        are unchanged. Saved with the Save button above.
      </span>
    </label>
  </div>

  <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
    <h3 class="text-sm font-medium text-slate-200 mb-3">Workflow</h3>
    <label class="flex items-start gap-3 cursor-pointer">
      <input
        type="checkbox"
        bind:checked={pauseBeforeTag}
        class="mt-0.5 w-4 h-4 rounded bg-ink-950 border-ink-700 accent-accent-500"
      />
      <span class="flex-1">
        <span class="block text-sm text-slate-200">Pause before tagging</span>
        <span class="block text-xs text-slate-500 mt-0.5">
          When on, the pipeline waits after writing kept frames so you can
          delete unwanted ones before they're tagged. Click the yellow
          ⏸ pill on a running pipeline to resume tagging. Off = the
          pipeline tags inline as it runs.
        </span>
      </span>
    </label>
    <label class="flex items-start gap-3 cursor-pointer mt-3">
      <input
        type="checkbox"
        bind:checked={autoDeleteRejected}
        class="mt-0.5 w-4 h-4 rounded bg-ink-950 border-ink-700 accent-accent-500"
      />
      <span class="flex-1">
        <span class="block text-sm text-slate-200">Auto-delete rejected frames</span>
        <span class="block text-xs text-slate-500 mt-0.5">
          Frames that didn't match any character are deleted instead of saved
          to <code>output/rejected/</code>. Useful once you trust the matching
          thresholds; off by default so you can audit rejections.
        </span>
      </span>
    </label>
    <label class="flex items-start gap-3 cursor-pointer mt-3">
      <input
        type="checkbox"
        bind:checked={tagAutocomplete}
        class="mt-0.5 w-4 h-4 rounded bg-ink-950 border-ink-700 accent-accent-500"
      />
      <span class="flex-1">
        <span class="block text-sm text-slate-200">Tag autocomplete</span>
        <span class="block text-xs text-slate-500 mt-0.5">
          Show a danbooru tag suggestion dropdown while editing tags. Use
          <kbd>↑</kbd>/<kbd>↓</kbd> to navigate and <kbd>Tab</kbd> or
          <kbd>Enter</kbd> to accept.
          {#if !tagListPresent}
            <span class="block text-amber-400/90 mt-1">
              Tag list not downloaded — run <code>neme-anima tags fetch</code>.
            </span>
          {/if}
        </span>
      </span>
    </label>
    <div class="mt-4">
      <span class="block text-sm text-slate-200 mb-1">Blacklist tags</span>
      <span class="block text-xs text-slate-500 mb-2">
        WD14 tags listed here are stripped from every caption written by
        future tagging runs. Existing <code>.txt</code> sidecars are
        untouched. Press Enter or comma after each tag.
      </span>
      <TagBlacklistInput tags={blacklistTags} ontagsChange={setBlacklist} />
    </div>
  </div>

  <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
    <div class="flex items-center justify-between mb-3">
      <h3 class="text-sm font-medium text-slate-200">LLM image description</h3>
      <span class="text-[10px] uppercase tracking-wide text-slate-500">
        adds a 2nd row to each .txt
      </span>
    </div>

    <p class="text-xs text-slate-500 mb-3">
      After WD14 tagging, optionally call an OpenAI-compatible vision endpoint
      (e.g. LMStudio at <code class="text-slate-400">http://localhost:1234</code>)
      to write a 1-2 sentence description as the second line of each caption file.
      Disabled by default and stays off until a model is selected.
    </p>

    <label class="flex items-start gap-3 cursor-pointer mb-4">
      <input
        type="checkbox"
        bind:checked={llmEnabled}
        disabled={!llmCanEnable}
        class="mt-0.5 w-4 h-4 rounded bg-ink-950 border-ink-700 accent-accent-500 disabled:opacity-40"
      />
      <span class="flex-1">
        <span class="block text-sm text-slate-200">Enable LLM tagging</span>
        <span class="block text-xs text-slate-500 mt-0.5">
          {llmCanEnable
            ? "Each tagged frame will also receive a generated description."
            : "Pick a model below first — the toggle unlocks once a model is selected."}
        </span>
      </span>
    </label>

    <div class="grid grid-cols-[1fr_auto] gap-2 items-end mb-3">
      <label
        class="block"
        title="Base URL of the OpenAI-compatible server. e.g. http://localhost:1234 for LMStudio, https://api.openai.com for OpenAI. The /v1 suffix is added automatically if missing."
      >
        <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
          <span>Endpoint</span>
          <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
        </span>
        <input
          bind:value={llmEndpoint}
          placeholder="http://localhost:1234"
          class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-sm font-mono focus:outline-none focus:border-accent-500"
        />
      </label>
      <button
        type="button"
        onclick={discoverLLMModels}
        disabled={llmDiscovering || !llmEndpoint.trim()}
        class="px-3 py-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-200 border border-ink-700 btn-disabled"
      >{llmDiscovering ? "Probing…" : "Discover models"}</button>
    </div>

    <label
      class="block mb-3"
      title="Bearer token sent as Authorization: Bearer <key>. Required for OpenAI, OpenRouter, and hosted vLLM. LMStudio and most local servers ignore it — leave blank."
    >
      <span class="flex items-center justify-between">
        <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
          <span>API key</span>
          <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
          <span class="text-slate-600 normal-case tracking-normal">(optional — leave blank for LMStudio)</span>
        </span>
        <button
          type="button"
          onclick={() => (llmApiKeyVisible = !llmApiKeyVisible)}
          class="text-[10px] text-slate-500 hover:text-slate-300"
        >{llmApiKeyVisible ? "hide" : "show"}</button>
      </span>
      <input
        bind:value={llmApiKey}
        type={llmApiKeyVisible ? "text" : "password"}
        autocomplete="off"
        spellcheck="false"
        placeholder="sk-… (only needed for OpenAI / OpenRouter / hosted vLLM)"
        class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-sm font-mono focus:outline-none focus:border-accent-500"
      />
    </label>

    {#if llmDiscoverError}
      <p class="text-xs text-red-400 mb-2 break-all">{llmDiscoverError}</p>
    {:else if llmDiscoverOk}
      <p class="text-xs text-emerald-400 mb-2">
        Endpoint reachable — {llmModelsAvailable.length} model{llmModelsAvailable.length === 1 ? "" : "s"} found.
      </p>
    {/if}

    <label
      class="block mb-3"
      title="Vision-capable model to call on each kept frame. Click Discover models to populate the list from the endpoint's /v1/models. The toggle above only unlocks once a model is picked."
    >
      <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
        <span>Model</span>
        <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
      </span>
      {#if llmModelsAvailable.length === 0}
        <select
          disabled
          class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-sm font-mono opacity-50"
        >
          <option>— Discover models first —</option>
        </select>
      {:else}
        <select
          bind:value={llmModel}
          class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-sm font-mono focus:outline-none focus:border-accent-500"
        >
          <option value="">— Pick a model —</option>
          {#each llmModelsAvailable as m (m)}
            <option value={m}>{m}</option>
          {/each}
        </select>
      {/if}
    </label>

    <label
      class="flex items-start gap-3 cursor-pointer mb-3"
      title="Sends reasoning_effort: none so a reasoning model skips its thinking pass. Off by default: not every server understands the field, and a strict one (or OpenAI, for a non-reasoning model) will reject the request outright. Turn it on if captions come back truncated or each frame is slow."
    >
      <input
        type="checkbox"
        bind:checked={llmDisableThinking}
        class="mt-0.5 w-4 h-4 rounded bg-ink-950 border-ink-700 accent-accent-500"
      />
      <span class="flex-1">
        <span class="block text-sm text-slate-200">Disable model thinking</span>
        <span class="block text-xs text-slate-500 mt-0.5">
          Skips the reasoning pass on models that have one, so the whole token
          budget goes to the caption. Not supported by every server — if
          describing starts failing, turn this back off.
        </span>
      </span>
    </label>

    <label
      class="block"
      title="Instruction sent with each image. Pre-filled with the built-in LoRA caption prompt. Save unchanged to follow upstream changes to the default; edit to lock in a custom prompt for this project."
    >
      <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
        <span>Prompt (optional override)</span>
        <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
      </span>
      <textarea
        bind:value={llmPrompt}
        rows="4"
        class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-xs font-mono focus:outline-none focus:border-accent-500 resize-y"
      ></textarea>
      <span class="block text-[10px] text-slate-600 mt-1">
        Pre-filled with the built-in LoRA caption prompt — edit to customize,
        or save as-is to track upstream changes to the default.
      </span>
    </label>
  </div>

  {#each SECTIONS as section (section.key)}
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-medium text-slate-200">{section.label}</h3>
        <button type="button" onclick={() => resetSection(section.key)} class="text-xs text-slate-500 hover:text-slate-300">Reset</button>
      </div>
      <div class="grid grid-cols-2 gap-3">
        {#each section.fields as f}
          <label class="block" title={f.help}>
            <span class="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
              <span>{f.name}</span>
              <span aria-hidden="true" class="text-slate-600 cursor-help">ⓘ</span>
            </span>
            <input
              value={getValue(section.key, f.name)}
              oninput={(e) => setValue(section.key, f.name, (e.target as HTMLInputElement).value, f.type)}
              placeholder={f.placeholder}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-sm font-mono focus:outline-none focus:border-accent-500"
            />
          </label>
        {/each}
      </div>
    </div>
  {/each}
</div>

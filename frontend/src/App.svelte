<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import { queueStore } from "$lib/stores/queue.svelte";
  import { jobsStore } from "$lib/stores/jobs.svelte";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { trainingStore } from "$lib/stores/training.svelte";
  import { connectEvents, type Connection } from "$lib/ws";
  import { setFrameOverwriteConfirm } from "$lib/frameOverwriteContext";
  import { matchShortcut } from "$lib/shortcuts.svelte";
  import { toasts } from "$lib/stores/toasts.svelte";
  import { runBulkRetag } from "$lib/bulkRetag";
  import { tagActions, describeActions } from "$lib/bulkRetagActions";
  import TopStrip from "$lib/components/TopStrip.svelte";
  import FramesTab from "$lib/components/FramesTab.svelte";
  import RegexModal from "$lib/components/RegexModal.svelte";
  import CreateProjectModal from "$lib/components/CreateProjectModal.svelte";
  import DeleteProjectModal from "$lib/components/DeleteProjectModal.svelte";
  import ConfirmFrameOverwriteModal from "$lib/components/ConfirmFrameOverwriteModal.svelte";
  import ToastHost from "$lib/components/ToastHost.svelte";
  import ShortcutHelpModal from "$lib/components/ShortcutHelpModal.svelte";
  import SourcesTab from "$lib/components/SourcesTab.svelte";
  import SettingsTab from "$lib/components/SettingsTab.svelte";
  import TrainingTab from "$lib/components/TrainingTab.svelte";

  let conn: Connection | null = null;
  let regexOpen = $state(false);
  let createOpen = $state(false);
  // Modal lives at the App root, not inside ProjectPills — the top bar
  // uses backdrop-blur which creates a CSS containing block for
  // position:fixed descendants, clipping any modal mounted under it.
  let deleteProjectOpen = $state(false);
  let helpOpen = $state(false);
  type FrameOverwriteRequest = {
    action: "retag" | "describe";
    selectedCount: number;
    affectedCount: number;
    resolve: (confirmed: boolean) => void;
  };
  let frameOverwriteRequest = $state<FrameOverwriteRequest | null>(null);

  async function confirmDeleteActiveProject() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    await projectsStore.delete(slug);
    deleteProjectOpen = false;
  }

  function confirmFrameOverwrite(
    action: "retag" | "describe",
    selectedCount: number,
    affectedCount: number,
  ): Promise<boolean> {
    frameOverwriteRequest?.resolve(false);
    return new Promise((resolve) => {
      frameOverwriteRequest = { action, selectedCount, affectedCount, resolve };
    });
  }

  setFrameOverwriteConfirm(confirmFrameOverwrite);

  function resolveFrameOverwrite(confirmed: boolean) {
    const req = frameOverwriteRequest;
    if (!req) return;
    frameOverwriteRequest = null;
    req.resolve(confirmed);
  }

  onMount(async () => {
    await projectsStore.refresh();
    if (projectsStore.list.length > 0) {
      await projectsStore.load(projectsStore.list[0].slug);
    }
    await queueStore.refresh();
    conn = connectEvents({
      url: `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/ws`,
      onEvent: (ev) => {
        queueStore.ingest(ev);
        jobsStore.ingest(ev);
        trainingStore.ingest(ev);
      },
      onStatus: (s) => queueStore.setStatus(s),
    });
    window.addEventListener("keydown", onKey);
  });

  onDestroy(() => {
    conn?.close();
    window.removeEventListener("keydown", onKey);
  });

  function onKey(ev: KeyboardEvent) {
    const target = ev.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
    // Don't run shortcuts while a modal is open — Escape must close the modal,
    // not clear the selection, and FullSizeModal owns its own arrow nav.
    if (document.querySelector('[role="dialog"]')) return;

    const sc = matchShortcut(ev);
    if (!sc) return;
    // Frames-only shortcuts are inert on other tabs; `?` (global) still works.
    if (!sc.global && viewStore.tab !== "frames") return;

    switch (sc.action) {
      case "select-all":
        framesStore.selectAll();
        break;
      case "clear-selection":
        framesStore.clear();
        break;
      case "delete-selection":
        if (framesStore.selectedFilenames().length > 0) void deleteSelection();
        break;
      case "bulk-tag":
        if (framesStore.selectedFilenames().length > 0) bulkTagSelection();
        break;
      case "bulk-describe":
        if (framesStore.selectedFilenames().length > 0) bulkDescribeSelection();
        break;
      case "open-regex":
        if (framesStore.selectedFilenames().length > 0) regexOpen = true;
        break;
      case "open-help":
        helpOpen = true;
        break;
    }
    ev.preventDefault();
  }

  // Mirrors ActionBar's Delete button: native confirm, bulk-delete, then drop
  // the rows locally. Kept here (not delegated to ActionBar) so the keyboard
  // path matches the other shortcut handlers in this file.
  async function deleteSelection() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    if (!confirm(`Delete ${filenames.length} frame${filenames.length === 1 ? "" : "s"}?`)) return;
    await api.bulkDeleteFrames(slug, filenames);
    framesStore.removeLocal(filenames);
  }

  async function bulkTagSelection() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    const affected = framesStore.items.filter(
      (it) => filenames.includes(it.filename) && it.has_tags,
    ).length;
    if (
      affected > 0 &&
      !(await confirmFrameOverwrite("retag", filenames.length, affected))
    ) {
      return;
    }
    await runBulkRetag("tag", slug, filenames, tagActions);
  }

  async function bulkDescribeSelection() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    if (!projectsStore.active?.llm?.model) {
      toasts.info("Pick an LLM model in Settings to describe frames.");
      return;
    }
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    const affected = framesStore.items.filter(
      (it) => filenames.includes(it.filename) && it.has_description,
    ).length;
    if (
      affected > 0 &&
      !(await confirmFrameOverwrite("describe", filenames.length, affected))
    ) {
      return;
    }
    await runBulkRetag("describe", slug, filenames, describeActions);
  }
</script>

<div class="min-h-screen bg-ink-950 text-slate-100">
  <TopStrip
    onopenRegex={() => (regexOpen = true)}
    onopenCreate={() => (createOpen = true)}
    onopenDelete={() => (deleteProjectOpen = true)}
  />
  <main class="px-4 pb-12">
    {#if projectsStore.active}
      {#if viewStore.tab === "sources"}
        <SourcesTab />
      {:else if viewStore.tab === "frames"}
        <FramesTab />
      {:else if viewStore.tab === "training"}
        <TrainingTab />
      {:else if viewStore.tab === "settings"}
        <SettingsTab />
      {/if}
    {:else}
      <div class="flex flex-col items-center justify-center py-32 text-slate-400">
        <p class="text-lg mb-2">No project selected.</p>
        <p class="text-sm">Click "+" in the top bar to create one.</p>
      </div>
    {/if}
  </main>

  {#if regexOpen}
    <RegexModal onclose={() => (regexOpen = false)} />
  {/if}
  {#if createOpen}
    <CreateProjectModal onclose={() => (createOpen = false)} />
  {/if}
  {#if deleteProjectOpen && projectsStore.active}
    <DeleteProjectModal
      project={projectsStore.active}
      onconfirm={confirmDeleteActiveProject}
      oncancel={() => (deleteProjectOpen = false)}
    />
  {/if}
  {#if frameOverwriteRequest}
    <ConfirmFrameOverwriteModal
      action={frameOverwriteRequest.action}
      selectedCount={frameOverwriteRequest.selectedCount}
      affectedCount={frameOverwriteRequest.affectedCount}
      onconfirm={() => resolveFrameOverwrite(true)}
      oncancel={() => resolveFrameOverwrite(false)}
    />
  {/if}
  {#if helpOpen}
    <ShortcutHelpModal onclose={() => (helpOpen = false)} />
  {/if}
  <ToastHost />
</div>

<script lang="ts">
  import { onDestroy } from "svelte";
  import * as api from "$lib/api";
  import { focusTrap } from "$lib/actions/focusTrap";
  import { createFlash } from "$lib/composables/flash.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { toasts } from "$lib/stores/toasts.svelte";
  import TagEditorPanel from "./TagEditorPanel.svelte";
  import DescriptionEditorPanel from "./DescriptionEditorPanel.svelte";
  import DiscardChangesDialog from "./DiscardChangesDialog.svelte";

  type Props = {
    /** Ordered list of frame filenames in the active view, used for ←/→ nav. */
    filenames: string[];
    /** Index into `filenames` of the frame currently shown. */
    index: number;
    /** Parent owns the index; modal asks it to step. */
    onnav: (next: number) => void;
    onclose: () => void;
    /** Called after a crop derivative has been saved server-side. */
    /** Fired after a crop saves, with the cropped original's filename so the
     *  grid can swap in / cache-bust the crop derivative. */
    oncropped: (filename: string) => void;
  };
  const { filenames, index, onnav, onclose, oncropped }: Props = $props();

  // ---- LoRA-friendly crop guardrails (kept in sync with the brief) ----
  // Anima's bucket range is [0.5, 2.0]; the model was largely trained at 512px,
  // so cropping below that on the short side actively hurts.
  const AR_MIN = 0.5;
  const AR_MAX = 2.0;
  const MIN_SHORT_SIDE_PX = 512;

  let filename = $derived(filenames[index] ?? "");
  let imageUrl = $derived(
    projectsStore.active && filename
      ? api.frameImageUrl(projectsStore.active.slug, filename)
      : "",
  );

  // Image natural size (set after load); display container is sized to fit
  // viewport while preserving AR, so display→image px is a single scale.
  let natW = $state(0);
  let natH = $state(0);
  let viewportW = $state(0);
  let viewportH = $state(0);
  let imgEl: HTMLImageElement | undefined = $state();
  let viewportEl: HTMLDivElement | undefined = $state();

  // Crop rect in IMAGE pixel space (not display). On modal open we look up
  // any persisted crop sidecar via /frames/{filename}/crop — if one exists
  // the rect starts at the saved values, otherwise full-image.
  let cropX = $state(0);
  let cropY = $state(0);
  let cropW = $state(0);
  let cropH = $state(0);
  // The "as last saved" rectangle — used both to initialize the overlay and
  // as the reference `modified` compares against. null = no saved crop yet.
  let savedRect = $state<{ x: number; y: number; width: number; height: number } | null>(null);
  // Snapshot of the rect at the moment we apply (saved or full-image), so
  // `modified` flips to true only when the user actually moves something.
  let initialX = $state(0);
  let initialY = $state(0);
  let initialW = $state(0);
  let initialH = $state(0);
  // Track which filename the load belongs to so a slow request that arrives
  // after the user navigated away doesn't clobber the new image's state.
  let savedRectFor = $state<string>("");
  let imageLoadedFor = $state<string>("");
  let modified = $state(false);
  let saving = $state(false);

  // Unsaved-edit state reported up by the two side-panel editors. The exit
  // guard consults `dirty` before honoring a close / navigate request.
  let tagsDirty = $state(false);
  let descDirty = $state(false);
  let dirty = $derived(tagsDirty || descDirty);
  // A close / nav action deferred behind the discard confirmation; null when
  // nothing is pending.
  let pendingExit = $state<
    { kind: "close" } | { kind: "nav"; target: number } | null
  >(null);
  // Transient "Crop saved ✓" toast — the modal now stays open after a crop.
  const cropSavedFlash = createFlash();

  let scale = $derived.by(() => {
    if (!natW || !natH || !viewportW || !viewportH) return 1;
    return Math.min(viewportW / natW, viewportH / natH);
  });
  let displayW = $derived(natW * scale);
  let displayH = $derived(natH * scale);

  function applyRect() {
    // Need both pieces (image dimensions AND saved-rect lookup) to have
    // landed for the SAME filename before we paint. Either side arriving
    // first just waits.
    if (imageLoadedFor !== filename || savedRectFor !== filename) return;
    if (!natW || !natH) return;
    if (savedRect) {
      cropX = savedRect.x;
      cropY = savedRect.y;
      cropW = savedRect.width;
      cropH = savedRect.height;
    } else {
      cropX = 0;
      cropY = 0;
      cropW = natW;
      cropH = natH;
    }
    initialX = cropX;
    initialY = cropY;
    initialW = cropW;
    initialH = cropH;
    modified = false;
  }

  function onImgLoad() {
    if (!imgEl) return;
    natW = imgEl.naturalWidth;
    natH = imgEl.naturalHeight;
    imageLoadedFor = filename;
    applyRect();
  }

  async function loadSavedRect(forFilename: string) {
    const slug = projectsStore.active?.slug;
    if (!slug || !forFilename) {
      savedRect = null;
      savedRectFor = forFilename;
      applyRect();
      return;
    }
    try {
      const r = await api.getCropRect(slug, forFilename);
      // Drop the response if the user already navigated away — otherwise we'd
      // paint stale state for a different image.
      if (forFilename !== filename) return;
      savedRect = r;
    } catch (e) {
      if (forFilename !== filename) return;
      savedRect = null;
      console.warn("crop rect lookup failed", e);
    }
    savedRectFor = forFilename;
    applyRect();
  }

  // Reset rect any time the displayed filename changes (arrow-key nav).
  // Kicks off the saved-rect lookup; the image's onload completes the
  // other half of `applyRect()`.
  $effect(() => {
    const fn = filename;
    natW = 0;
    natH = 0;
    cropX = 0;
    cropY = 0;
    cropW = 0;
    cropH = 0;
    savedRect = null;
    savedRectFor = "";
    imageLoadedFor = "";
    modified = false;
    void loadSavedRect(fn);
  });

  function measureViewport() {
    if (!viewportEl) return;
    viewportW = viewportEl.clientWidth;
    viewportH = viewportEl.clientHeight;
  }

  $effect(() => {
    measureViewport();
    const ro = new ResizeObserver(measureViewport);
    if (viewportEl) ro.observe(viewportEl);
    return () => ro.disconnect();
  });

  // ---------------- crop validity ----------------

  let cropAR = $derived(cropH > 0 ? cropW / cropH : 1);
  let cropShortSide = $derived(Math.min(cropW, cropH));
  let arInRange = $derived(cropAR >= AR_MIN && cropAR <= AR_MAX);
  let sizeOk = $derived(cropShortSide >= MIN_SHORT_SIDE_PX);
  let canConfirm = $derived(modified && arInRange && sizeOk && !saving);

  let invalidReason = $derived.by(() => {
    if (!arInRange) {
      return `Aspect ratio ${cropAR.toFixed(2)} is outside the trainer's bucket range [0.5, 2.0]`;
    }
    if (!sizeOk) {
      return `Short side ${cropShortSide}px is below ${MIN_SHORT_SIDE_PX}px (Anima trains at ≥512px)`;
    }
    return "";
  });

  // ---------------- pointer drag (resize / move) ----------------

  type DragMode =
    | "move"
    | "n" | "s" | "e" | "w"
    | "ne" | "nw" | "se" | "sw"
    | null;

  let dragMode = $state<DragMode>(null);
  let dragStart = $state<{
    px: number; py: number;
    x: number; y: number; w: number; h: number;
  } | null>(null);

  function startDrag(mode: DragMode, ev: PointerEvent) {
    if (!mode) return;
    ev.preventDefault();
    ev.stopPropagation();
    (ev.currentTarget as Element).setPointerCapture(ev.pointerId);
    dragMode = mode;
    dragStart = {
      px: ev.clientX, py: ev.clientY,
      x: cropX, y: cropY, w: cropW, h: cropH,
    };
  }

  function onPointerMove(ev: PointerEvent) {
    if (!dragMode || !dragStart) return;
    const dxImg = (ev.clientX - dragStart.px) / scale;
    const dyImg = (ev.clientY - dragStart.py) / scale;

    let { x, y, w, h } = dragStart;
    if (dragMode === "move") {
      x = Math.max(0, Math.min(natW - w, x + dxImg));
      y = Math.max(0, Math.min(natH - h, y + dyImg));
    } else {
      // Edge / corner resize. Anchor the OPPOSITE side(s) so the rect stays
      // within image bounds and the user's pointer drives one corner.
      let nx = x, ny = y, nw = w, nh = h;
      if (dragMode.includes("e")) nw = Math.max(1, w + dxImg);
      if (dragMode.includes("w")) {
        const newX = Math.max(0, Math.min(x + w - 1, x + dxImg));
        nw = Math.max(1, w - (newX - x));
        nx = newX;
      }
      if (dragMode.includes("s")) nh = Math.max(1, h + dyImg);
      if (dragMode.includes("n")) {
        const newY = Math.max(0, Math.min(y + h - 1, y + dyImg));
        nh = Math.max(1, h - (newY - y));
        ny = newY;
      }
      // Clamp inside image.
      nw = Math.min(nw, natW - nx);
      nh = Math.min(nh, natH - ny);
      x = nx; y = ny; w = nw; h = nh;
    }
    cropX = Math.round(x);
    cropY = Math.round(y);
    cropW = Math.round(w);
    cropH = Math.round(h);
    // Compare against the initial rect — which is the saved crop (if any)
    // or the full-image rect — so reopening an already-cropped frame doesn't
    // immediately appear "modified" just because cropX/Y aren't zero.
    if (
      cropX !== initialX || cropY !== initialY ||
      cropW !== initialW || cropH !== initialH
    ) {
      modified = true;
    }
  }

  function onPointerUp(ev: PointerEvent) {
    if (!dragMode) return;
    try { (ev.currentTarget as Element).releasePointerCapture(ev.pointerId); }
    catch { /* not captured */ }
    dragMode = null;
    dragStart = null;
  }

  // ---------------- exit guard ----------------

  function requestClose() {
    if (dirty) { pendingExit = { kind: "close" }; return; }
    onclose();
  }
  function requestNav(target: number) {
    if (target < 0 || target >= filenames.length) return;
    if (dirty) { pendingExit = { kind: "nav", target }; return; }
    onnav(target);
  }
  function confirmExit() {
    const ex = pendingExit;
    pendingExit = null;
    if (!ex) return;
    if (ex.kind === "close") onclose();
    else onnav(ex.target);
  }
  function cancelExit() {
    pendingExit = null;
  }

  // ---------------- key handling ----------------

  function onKey(ev: KeyboardEvent) {
    // While the discard dialog is up it owns the keyboard (Escape there
    // cancels the exit), so ignore everything to avoid double-handling.
    if (pendingExit) return;
    const target = ev.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
    if (ev.key === "Escape") { requestClose(); return; }
    if (ev.key === "ArrowLeft" && index > 0) {
      requestNav(index - 1);
      ev.preventDefault();
    } else if (ev.key === "ArrowRight" && index < filenames.length - 1) {
      requestNav(index + 1);
      ev.preventDefault();
    } else if (ev.key === "Enter" && canConfirm) {
      void confirmCrop();
      ev.preventDefault();
    }
  }

  // ---------------- save ----------------

  async function confirmCrop() {
    const slug = projectsStore.active?.slug;
    if (!slug || !canConfirm) return;
    saving = true;
    try {
      await api.cropFrame(slug, filename, {
        x: cropX, y: cropY, width: cropW, height: cropH,
      });
      // Keep the modal open: a crop never touches the original image (the
      // derivative is a hidden sibling), so rebase the overlay to the saved
      // rect, clear `modified`, and flash a confirmation instead of closing.
      savedRect = { x: cropX, y: cropY, width: cropW, height: cropH };
      initialX = cropX; initialY = cropY; initialW = cropW; initialH = cropH;
      modified = false;
      cropSavedFlash.trigger();
      oncropped(filename);
    } catch (e) {
      console.error("crop failed", e);
      toasts.error("Crop failed — see console for details.");
    } finally {
      saving = false;
    }
  }

  onDestroy(() => cropSavedFlash.destroy());

  // Display-space helpers (used by the rect overlay).
  let rectDispX = $derived(cropX * scale);
  let rectDispY = $derived(cropY * scale);
  let rectDispW = $derived(cropW * scale);
  let rectDispH = $derived(cropH * scale);
</script>

<svelte:window onkeydown={onKey} />

<!-- Backdrop: now a two-pane row (image viewport | metadata panel). Clicking
     the backdrop or the empty viewport area requests close (guarded when there
     are unsaved tag/description edits). Keyboard close is via Escape on the
     window listener above. -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div
  class="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex gap-4 p-6"
  role="dialog"
  aria-modal="true"
  tabindex="-1"
  use:focusTrap={{}}
  onmousedown={(e) => { if (e.target === e.currentTarget) requestClose(); }}
  onpointermove={onPointerMove}
  onpointerup={onPointerUp}
>
  <!-- LEFT: image + crop viewport -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    bind:this={viewportEl}
    role="presentation"
    class="relative flex-1 min-w-0 h-full flex items-center justify-center"
    onmousedown={(e) => { if (e.target === e.currentTarget) requestClose(); }}
  >
    {#if displayW > 0 && displayH > 0}
      <div
        class="relative shadow-2xl rounded overflow-hidden select-none"
        style="width: {displayW}px; height: {displayH}px;"
      >
        <img
          bind:this={imgEl}
          src={imageUrl}
          alt={filename}
          draggable="false"
          onload={onImgLoad}
          class="block w-full h-full select-none"
        />

        <!-- Dim overlay outside the rect (4 strips). -->
        {#if cropW > 0 && cropH > 0}
          <div class="absolute inset-x-0 top-0 bg-black/60 pointer-events-none"
               style="height: {rectDispY}px;"></div>
          <div class="absolute inset-x-0 bottom-0 bg-black/60 pointer-events-none"
               style="top: {rectDispY + rectDispH}px;"></div>
          <div class="absolute bg-black/60 pointer-events-none"
               style="left: 0; top: {rectDispY}px; width: {rectDispX}px; height: {rectDispH}px;"></div>
          <div class="absolute bg-black/60 pointer-events-none"
               style="left: {rectDispX + rectDispW}px; top: {rectDispY}px; right: 0; height: {rectDispH}px;"></div>

          <!-- Crop rectangle: drag-to-move + 8 handles. -->
          <div
            role="application"
            aria-label="Crop rectangle, drag to move"
            tabindex="-1"
            class="absolute border-2 border-emerald-400 cursor-move"
            style="left: {rectDispX}px; top: {rectDispY}px; width: {rectDispW}px; height: {rectDispH}px;"
            onpointerdown={(e) => startDrag("move", e)}
          >
            {#each [
              ["nw", "top-0 left-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize"],
              ["n",  "top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 cursor-ns-resize"],
              ["ne", "top-0 right-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize"],
              ["e",  "top-1/2 right-0 translate-x-1/2 -translate-y-1/2 cursor-ew-resize"],
              ["se", "bottom-0 right-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize"],
              ["s",  "bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 cursor-ns-resize"],
              ["sw", "bottom-0 left-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize"],
              ["w",  "top-1/2 left-0 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize"],
            ] as [pos, cls] (pos)}
              <div
                role="button"
                tabindex="-1"
                aria-label="Resize {pos}"
                class="absolute w-3 h-3 bg-emerald-400 border border-emerald-900 rounded-sm {cls}"
                onpointerdown={(e) => startDrag(pos as DragMode, e)}
              ></div>
            {/each}
          </div>

          <!-- Live size label inside the rect, top-left. -->
          <div
            class="absolute pointer-events-none px-1.5 py-0.5 text-[10px] font-mono rounded bg-black/70 text-emerald-200"
            style="left: {rectDispX + 4}px; top: {rectDispY + 4}px;"
          >
            {cropW}×{cropH}  ·  AR {cropAR.toFixed(2)}
          </div>
        {/if}

        <!-- Validate button: only when modified. -->
        {#if modified}
          <button
            type="button"
            onclick={(e) => { e.stopPropagation(); void confirmCrop(); }}
            disabled={!canConfirm}
            title={canConfirm
              ? "Save crop (Enter)"
              : invalidReason || (saving ? "Saving…" : "")}
            aria-label="Confirm crop"
            class="absolute top-2 right-2 w-9 h-9 rounded-full text-base font-bold flex items-center justify-center shadow-lg transition-all
              {canConfirm
                ? 'bg-emerald-500 hover:bg-emerald-400 text-white'
                : 'bg-slate-700 text-slate-400 cursor-not-allowed'}"
          >
            {saving ? "…" : "✓"}
          </button>
        {/if}

        <!-- Crop-saved toast: shown briefly after a save; modal stays open. -->
        {#if cropSavedFlash.active}
          <div
            class="absolute top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-emerald-500 text-white text-xs font-medium shadow-lg pointer-events-none"
          >Crop saved ✓</div>
        {/if}

        <!-- Frame counter, bottom-center. -->
        {#if filenames.length > 1}
          <div class="absolute bottom-2 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-black/60 text-[10px] text-slate-300 pointer-events-none">
            {index + 1} / {filenames.length}  ·  ←/→ to navigate
          </div>
        {/if}
      </div>
    {:else}
      <!-- Hidden img to trigger load + populate naturalWidth/Height. -->
      <img
        bind:this={imgEl}
        src={imageUrl}
        alt={filename}
        onload={onImgLoad}
        class="opacity-0 pointer-events-none absolute"
      />
    {/if}
  </div>

  <!-- RIGHT: metadata panel (tags + description). Clicks here never reach the
       backdrop's close handler because they target panel children. The panels
       reload themselves when `filename` changes via arrow-key nav. -->
  <aside
    class="w-96 shrink-0 h-full bg-ink-900 border border-ink-700 rounded-lg flex flex-col gap-4 p-4 overflow-y-auto"
  >
    <TagEditorPanel {filename} ondirty={(d) => (tagsDirty = d)} onclose={requestClose} />
    <DescriptionEditorPanel {filename} ondirty={(d) => (descDirty = d)} />
  </aside>
</div>

{#if pendingExit}
  <DiscardChangesDialog
    {tagsDirty}
    {descDirty}
    onconfirm={confirmExit}
    oncancel={cancelExit}
  />
{/if}

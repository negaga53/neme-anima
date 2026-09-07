<script lang="ts">
  import { tagVocabulary } from "$lib/tagVocabulary.svelte";
  import { normalizeTagKey, type Suggestion } from "$lib/tagSearch";
  import { parseTagsFromClipboard } from "$lib/tagClipboard";
  import TagAutocomplete from "./TagAutocomplete.svelte";

  type Props = {
    text: string;
    /** Uncontrolled commit (grid surface). Optional so the controlled path
     *  (which uses `oncommit`) need not pass it. */
    onreplace?: (next: string) => void;
    /** Called when an existing pill is committed empty — parent should drop the
     *  tag. If omitted, empty commits revert to the old text. */
    ondelete?: () => void;
    /** Called when a placeholder/draft pill is dismissed without typing, or on
     *  Escape in controlled mode — parent removes the slot / closes the editor. */
    oncancel?: () => void;
    /** Mount the pill directly in edit mode (uncontrolled "+" placeholder). */
    startEditing?: boolean;
    /** Visual size. "sm" (grid) / "md" (crop modal). */
    size?: "sm" | "md";
    /** Enable the danbooru autocomplete dropdown while editing. */
    autocomplete?: boolean;
    /** Other tags already on the frame — excluded from suggestions. */
    existingTags?: string[];
    // ---- controlled mode (rich TagList) ----
    /** When provided, the PARENT controls edit state. The pill shows its input
     *  when true and a static label + ✕ when false, and never self-toggles.
     *  Omit for the uncontrolled grid behavior. */
    editing?: boolean;
    /** Commit a non-empty edit (controlled). `chain` is true for an explicit
     *  validate (Enter / accept-suggestion), false for a blur. */
    oncommit?: (next: string, chain: boolean) => void;
    /** A comma list was pasted while editing (controlled) — split into parts. */
    onsplitpaste?: (parts: string[]) => void;
  };
  const {
    text,
    onreplace,
    ondelete,
    oncancel,
    startEditing = false,
    size = "sm",
    autocomplete = false,
    existingTags = [],
    editing: editingProp = undefined,
    oncommit,
    onsplitpaste,
  }: Props = $props();

  // Controlled when the parent passes an `editing` boolean.
  let controlled = $derived(editingProp !== undefined);
  let internalEditing = $state<boolean>(startEditing);
  let isEditing = $derived(controlled ? (editingProp as boolean) : internalEditing);

  let value = $state(text);
  let inputEl = $state<HTMLInputElement | null>(null);
  // Commit guard: Enter flips out of edit and the trailing blur must not
  // re-fire the commit. Plain (non-reactive) so toggling it never re-renders.
  let committed = false;

  // Autocomplete: dismissed closes the dropdown without leaving edit mode
  // (Escape / after accepting); typing re-opens it.
  let acDismissed = $state(false);
  let acIndex = $state(0);

  // One-time vocabulary load the first time an autocomplete-enabled pill edits.
  $effect(() => {
    if (autocomplete && isEditing) tagVocabulary.ensureLoaded();
  });

  // Exclude every other tag on the frame (allow re-typing this pill's own text).
  let excludeKeys = $derived(
    new Set(existingTags.filter((t) => t !== text).map(normalizeTagKey)),
  );

  let suggestions = $derived<Suggestion[]>(
    autocomplete && isEditing && !acDismissed && value.trim()
      ? tagVocabulary.search(value, { exclude: excludeKeys })
      : [],
  );
  let acOpen = $derived(suggestions.length > 0);

  // Sync the buffer to the canonical text whenever not editing (external
  // updates after a save). While editing, `value` holds the user's input.
  $effect(() => {
    if (!isEditing) value = text;
  });

  // On entering edit: reset the commit guard, focus + select the input.
  $effect(() => {
    if (!isEditing) return;
    committed = false;
    const el = inputEl;
    if (!el) return;
    queueMicrotask(() => {
      el.focus();
      if (el.value.length > 0) el.select();
    });
  });

  function finalize(chain: boolean) {
    if (committed) return;
    committed = true;
    const trimmed = value.trim();
    if (!controlled) internalEditing = false;
    if (trimmed === "") {
      // Empty existing pill = delete; empty placeholder/draft = discard.
      if (text === "") oncancel?.();
      else ondelete?.();
      return;
    }
    if (controlled) oncommit?.(trimmed, chain);
    else if (trimmed !== text) onreplace?.(trimmed);
  }

  function cancel() {
    if (committed) return;
    committed = true;
    if (!controlled) {
      value = text;
      internalEditing = false;
    }
    oncancel?.();
  }

  function acceptSuggestion(s: Suggestion) {
    value = s.entry.name; // canonical space form
    acDismissed = true;
    finalize(true); // explicit validate — clicking a suggestion commits it
  }

  // Fill the field with a suggestion WITHOUT committing, so the user can keep
  // editing and then validate with Enter. Tab and → (cursor at end) use this;
  // only Enter (verbatim) or clicking a suggestion actually commits.
  function acceptInline(s: Suggestion) {
    value = s.entry.name; // canonical space form; closing the dropdown next
    acDismissed = true;
    acIndex = 0;
  }

  function onKeydown(ev: KeyboardEvent) {
    if (acOpen) {
      if (ev.key === "ArrowDown") {
        ev.preventDefault();
        acIndex = (acIndex + 1) % suggestions.length;
        return;
      }
      if (ev.key === "ArrowUp") {
        ev.preventDefault();
        acIndex = (acIndex - 1 + suggestions.length) % suggestions.length;
        return;
      }
      if (ev.key === "Tab") {
        // Fill the field, keep editing — does NOT validate (Enter does).
        ev.preventDefault();
        acceptInline(suggestions[acIndex]);
        return;
      }
      if (ev.key === "ArrowRight") {
        // Complete inline only when the cursor is at the very end, so mid-text
        // → still moves the caret. Like Tab, this fills without validating.
        const el = ev.currentTarget as HTMLInputElement;
        if (el.selectionStart === el.value.length && el.selectionEnd === el.value.length) {
          ev.preventDefault();
          acceptInline(suggestions[acIndex]);
          return;
        }
      }
      if (ev.key === "Escape") {
        // Close the dropdown only; keep editing. A second Escape reverts.
        ev.preventDefault();
        ev.stopPropagation();
        acDismissed = true;
        return;
      }
      // Enter falls through → commit the typed text verbatim.
    }
    if (ev.key === "Enter") {
      ev.preventDefault();
      finalize(true);
      return;
    }
    if (ev.key === "Tab") {
      // Only ever completes a suggestion (handled above while open) — never validates.
      ev.preventDefault();
      return;
    }
    if (ev.key === "Escape") {
      ev.preventDefault();
      cancel();
    }
  }

  function onInput() {
    // Re-open the dropdown and reset the highlight as the user types.
    acDismissed = false;
    acIndex = 0;
  }

  function onPaste(ev: ClipboardEvent) {
    if (!controlled) return; // grid keeps default paste-into-input
    const pasted = ev.clipboardData?.getData("text") ?? "";
    if (!pasted.includes(",")) return;
    ev.preventDefault();
    const parts = parseTagsFromClipboard(pasted);
    if (parts.length) {
      committed = true;
      onsplitpaste?.(parts);
    }
  }

  // Heuristic "important" tag styling (unchanged): looks like a character name.
  let isCharacter = $derived(text.toLowerCase().includes("character"));

  let sz = $derived(
    size === "md"
      ? { text: "text-[11.25px]", btn: "px-2 py-0.5", input: "px-2.5 py-0.5 w-28" }
      : { text: "text-[9.5px]", btn: "px-1.5 py-0.5", input: "px-2 py-0.5 w-24" },
  );
</script>

{#if isEditing}
  <input
    bind:this={inputEl}
    bind:value
    oninput={onInput}
    onkeydown={onKeydown}
    onblur={() => finalize(false)}
    onpaste={onPaste}
    onclick={(e) => e.stopPropagation()}
    onpointerdown={(e) => e.stopPropagation()}
    ondblclick={(e) => e.stopPropagation()}
    placeholder={text === "" ? "new tag…" : ""}
    class="{sz.input} {sz.text} rounded-full bg-accent-500 text-white shadow-[0_0_0_1.5px_rgba(199,210,254,1),0_0_12px_rgba(99,102,241,0.6)] outline-none placeholder-white/60"
  />
  {#if acOpen && inputEl}
    <TagAutocomplete
      {suggestions}
      activeIndex={acIndex}
      anchor={inputEl}
      onaccept={acceptSuggestion}
      onhover={(i) => (acIndex = i)}
    />
  {/if}
{:else if controlled}
  <!-- Controlled static label: text + ✕. No self-edit — the container's
       wrapper handles double-click to edit and pointer-down to select/drag. -->
  <span
    class="{sz.btn} {sz.text} inline-flex items-center gap-1 rounded-full backdrop-blur-sm border border-white/5 select-none
      {isCharacter ? 'bg-amber2-500/85 text-amber-950 font-medium' : 'bg-white/15 text-white'}"
  >
    <span>{text}</span>
    <button
      type="button"
      aria-label="Delete tag"
      tabindex="-1"
      onpointerdown={(e) => e.stopPropagation()}
      onclick={(e) => { e.stopPropagation(); ondelete?.(); }}
      class="leading-none opacity-60 hover:opacity-100 hover:text-rose-300 transition-opacity"
    >×</button>
  </span>
{:else}
  <button
    type="button"
    onclick={(e) => { e.stopPropagation(); internalEditing = true; }}
    class="{sz.btn} {sz.text} rounded-full backdrop-blur-sm border border-white/5 transition-colors
      {isCharacter ? 'bg-amber2-500/85 text-amber-950 font-medium' : 'bg-white/15 text-white hover:bg-white/25'}"
  >
    {text}
  </button>
{/if}

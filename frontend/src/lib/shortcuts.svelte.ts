// Keyboard-shortcut registry. The matcher is pure (Vitest-covered); App.svelte
// owns dispatch (it keeps the input/modal/tab guards). The .svelte.ts suffix is
// future-proofing — no runes are exported today, but a later phase could add
// per-project rebinding here without moving the file.

export type ShortcutAction =
  | "select-all"
  | "clear-selection"
  | "delete-selection"
  | "bulk-tag"
  | "bulk-describe"
  | "open-regex"
  | "open-help";

export type Shortcut = {
  key: string;
  label: string;
  description: string;
  action: ShortcutAction;
  /** When true, fires even off the frames tab (only the help shortcut). */
  global?: boolean;
};

export const defaultShortcuts: Shortcut[] = [
  { key: "a", label: "A", description: "Select all frames", action: "select-all" },
  { key: "d", label: "D", description: "Delete selection", action: "delete-selection" },
  { key: "Escape", label: "Esc", description: "Clear selection", action: "clear-selection" },
  { key: "t", label: "T", description: "Tag selected frames (WD14)", action: "bulk-tag" },
  { key: "s", label: "S", description: "Describe selected frames (LLM)", action: "bulk-describe" },
  { key: "r", label: "R", description: "Open regex tag-replace", action: "open-regex" },
  { key: "?", label: "?", description: "Show this shortcuts help", action: "open-help", global: true },
];

/** Returns the matching Shortcut for a keydown, or null. Events with ctrl/meta
 *  held are ignored so browser combos (Ctrl+A, ⌘R, …) fall through. Matching is
 *  exact on KeyboardEvent.key. */
export function matchShortcut(
  ev: Pick<KeyboardEvent, "key" | "ctrlKey" | "metaKey">,
  shortcuts: Shortcut[] = defaultShortcuts,
): Shortcut | null {
  if (ev.ctrlKey || ev.metaKey) return null;
  return shortcuts.find((s) => s.key === ev.key) ?? null;
}

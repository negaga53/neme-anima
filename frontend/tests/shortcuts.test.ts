import { describe, expect, it } from "vitest";
import {
  defaultShortcuts,
  matchShortcut,
  type Shortcut,
} from "../src/lib/shortcuts.svelte";

function ev(key: string, mods: Partial<KeyboardEvent> = {}) {
  return { key, ctrlKey: false, metaKey: false, ...mods };
}

describe("matchShortcut", () => {
  it("matches the plain frames keys", () => {
    expect(matchShortcut(ev("a"))?.action).toBe("select-all");
    expect(matchShortcut(ev("d"))?.action).toBe("delete-selection");
    expect(matchShortcut(ev("Escape"))?.action).toBe("clear-selection");
    expect(matchShortcut(ev("t"))?.action).toBe("bulk-tag");
    expect(matchShortcut(ev("s"))?.action).toBe("bulk-describe");
    expect(matchShortcut(ev("r"))?.action).toBe("open-regex");
    expect(matchShortcut(ev("?"))?.action).toBe("open-help");
  });

  it("ignores events with ctrl or meta held", () => {
    expect(matchShortcut(ev("a", { ctrlKey: true }))).toBeNull();
    expect(matchShortcut(ev("a", { metaKey: true }))).toBeNull();
    expect(matchShortcut(ev("r", { ctrlKey: true }))).toBeNull();
  });

  it("returns null for unmapped keys", () => {
    expect(matchShortcut(ev("z"))).toBeNull();
    expect(matchShortcut(ev("ArrowLeft"))).toBeNull();
  });

  it("is case-sensitive on KeyboardEvent.key", () => {
    // Capital A (Shift+A) is a different key than 'a' and is unmapped.
    expect(matchShortcut(ev("A"))).toBeNull();
  });

  it("flags only the help shortcut as global", () => {
    const help = defaultShortcuts.find((s) => s.action === "open-help");
    expect(help?.global).toBe(true);
    const others = defaultShortcuts.filter((s) => s.action !== "open-help");
    expect(others.every((s: Shortcut) => !s.global)).toBe(true);
  });

  it("binds every action to exactly one key", () => {
    // After D was repurposed to delete-selection, clear-selection is reachable
    // only via Escape — every action is now 1:1 with a key.
    const actions = defaultShortcuts.map((s) => s.action);
    expect(new Set(actions).size).toBe(actions.length);
    expect(matchShortcut(ev("d"))?.action).toBe("delete-selection");
  });
});

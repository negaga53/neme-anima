import { serializeTagsForClipboard } from "$lib/tagClipboard";

// A tiny in-memory clipboard for copying a tag selection from one frame and
// pasting it onto another in the crop modal's tag editor. Module-scoped so it
// survives both arrow-key navigation (the panel reloads in place) and closing
// /reopening the modal (the panel is recreated). Purely ephemeral — it is not
// persisted to disk and resets on page reload.
class TagClipboard {
  tags = $state<string[]>([]);
  text = $state<string>("");

  get size(): number {
    return this.tags.length;
  }

  /** Replace the clipboard contents with a copy of `next`. */
  set(next: string[]): void {
    this.tags = [...next];
    this.text = serializeTagsForClipboard(next);
  }

  clear(): void {
    this.tags = [];
    this.text = "";
  }
}

export const tagClipboard = new TagClipboard();

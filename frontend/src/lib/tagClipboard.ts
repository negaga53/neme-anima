// Serializes/parses the plain-text clipboard payload for copy/paste of tag
// selections in the crop modal. Tags are a comma-separated list (matching the
// on-disk sidecar format), with any literal parentheses inside a tag name
// backslash-escaped — the same convention danbooru/kohya tags and prompt
// syntax use (e.g. `hair_ornament_\(object\)`) so a paste round-trips exactly
// and stray `(`/`)` never get mistaken for prompt-weighting syntax elsewhere.
export function escapeTagForClipboard(tag: string): string {
  return tag
    .trim()
    .replace(/\\/g, "\\\\")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)");
}

export function serializeTagsForClipboard(tags: string[]): string {
  return tags.map((tag) => escapeTagForClipboard(tag)).join(", ");
}

/** Write text to the system clipboard, best-effort. Must call
 *  navigator.clipboard.writeText(...) directly rather than through a stored
 *  reference — the native implementation validates `this` and throws
 *  "Illegal invocation" if the method is detached from its object. */
export function writeSystemClipboard(text: string): void {
  try {
    void navigator.clipboard?.writeText(text)?.catch(() => undefined);
  } catch {
    // Clipboard API unavailable (insecure context, permissions, etc.) — the
    // in-memory tagClipboard store still works for in-app paste.
  }
}

/** Split a comma-separated clipboard payload back into tags, unescaping any
 *  backslash-escaped characters as they're consumed. Escaping is resolved
 *  inline while scanning so a comma can never appear mid-tag from
 *  serializeTagsForClipboard's output — a plain (unescaped) comma always
 *  separates tags. */
export function parseTagsFromClipboard(text: string): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  const out: string[] = [];
  let current = "";
  let escaped = false;

  for (let i = 0; i < trimmed.length; i += 1) {
    const char = trimmed[i];

    if (escaped) {
      current += char;
      escaped = false;
      continue;
    }

    if (char === "\\") {
      escaped = true;
      continue;
    }

    if (char === ",") {
      const token = current.trim();
      if (token) out.push(token);
      current = "";
      continue;
    }

    current += char;
  }

  const tail = current.trim();
  if (tail) out.push(tail);
  return out.filter(Boolean);
}

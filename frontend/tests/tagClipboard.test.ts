import { describe, expect, it, vi } from "vitest";
import { parseTagsFromClipboard, serializeTagsForClipboard, writeSystemClipboard } from "../src/lib/tagClipboard";

describe("tag clipboard helpers", () => {
  it("serializes tags as a plain comma-separated list", () => {
    expect(serializeTagsForClipboard(["1girl", "head out of frame", "solo"])).toBe(
      "1girl, head out of frame, solo",
    );
  });

  it("parses a plain comma-separated clipboard list", () => {
    expect(parseTagsFromClipboard("1girl, head out of frame, solo")).toEqual([
      "1girl",
      "head out of frame",
      "solo",
    ]);
  });

  it("escapes and unescapes literal parentheses inside a tag", () => {
    const serialized = serializeTagsForClipboard(["hair_ornament_(object)", "baz"]);
    expect(serialized).toBe("hair_ornament_\\(object\\), baz");
    expect(parseTagsFromClipboard(serialized)).toEqual(["hair_ornament_(object)", "baz"]);
  });

  it("writes to navigator.clipboard.writeText without an Illegal invocation error", () => {
    // Regression test: navigator.clipboard.writeText is unbound natively — if
    // writeSystemClipboard ever stores the method in a variable and calls it
    // detached from `navigator.clipboard`, real browsers throw
    // "TypeError: Illegal invocation". This fake mimics that behavior so a
    // regression fails the test instead of only showing up in production.
    const writeText = vi.fn(function (this: unknown, _text: string) {
      if (this !== fakeClipboard) throw new TypeError("Illegal invocation");
      return Promise.resolve();
    });
    const fakeClipboard = { writeText };
    vi.stubGlobal("navigator", { clipboard: fakeClipboard });

    expect(() => writeSystemClipboard("tag")).not.toThrow();
    expect(writeText).toHaveBeenCalledWith("tag");

    vi.unstubAllGlobals();
  });
});

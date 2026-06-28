"""WD14 (SmilingWolf v3) auto-tagging via imgutils.

Produces kohya-style comma-separated tag strings. Output is written as a sibling
``.txt`` file alongside each ``.png`` (the standard convention for kohya-ss /
OneTrainer / sd-scripts).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from neme_anima.config import TagConfig


def _norm_tag(s: str) -> str:
    """Normalize a tag for blacklist matching: underscores->spaces, trimmed, lowercased.

    Danbooru uses ``long_hair`` but our sidecars (and WD14 under
    ``TagConfig.no_underline=True``) use ``long hair``. Matching the blacklist
    underscore/space/case-insensitively keeps it aligned with what is actually
    emitted — mirrors ``tag_vocabulary._norm`` without importing that module.
    """
    return s.replace("_", " ").strip().lower()


@dataclass(frozen=True)
class TagResult:
    rating: dict[str, float]
    general: dict[str, float]
    character: dict[str, float]
    text: str  # comma-separated, ready to write to disk


class Tagger:
    """Holds tagging settings; calling ``tag(image)`` returns a TagResult."""

    def __init__(self, cfg: TagConfig | None = None) -> None:
        self.cfg = cfg or TagConfig()

    def tag(self, image_rgb: np.ndarray) -> TagResult:
        from imgutils.tagging import get_wd14_tags

        pil = Image.fromarray(image_rgb) if not isinstance(image_rgb, Image.Image) else image_rgb
        rating, general, character = get_wd14_tags(
            pil,
            model_name=self.cfg.model_name,
            general_threshold=self.cfg.general_threshold,
            character_threshold=self.cfg.character_threshold,
            no_underline=self.cfg.no_underline,
            drop_overlap=self.cfg.drop_overlap,
        )
        text = self._compose_text(general, character)
        return TagResult(rating=rating, general=general, character=character, text=text)

    def _compose_text(
        self,
        general: dict[str, float],
        character: dict[str, float],
    ) -> str:
        # Sort by descending confidence; character tags first (kohya convention is
        # often "char_name, general_tag, general_tag, ..." for character LoRAs).
        char_tags = [t for t, _ in sorted(character.items(), key=lambda kv: -kv[1])]
        general_tags = [t for t, _ in sorted(general.items(), key=lambda kv: -kv[1])]
        # The blacklist is matched underscore/space/case-insensitively. Danbooru
        # tags are conventionally underscore form (``simple_background``) and the
        # Settings blacklist input accepts them verbatim, but WD14 emits the
        # on-disk space form (``simple background``) under ``no_underline=True``.
        # An exact-string exclusion silently ignored every multi-word blacklist
        # entry; normalizing both sides makes either form the user types match.
        excluded = {_norm_tag(t) for t in self.cfg.exclude_tags}
        all_tags = [t for t in (char_tags + general_tags) if _norm_tag(t) not in excluded]
        return ", ".join(all_tags)


def write_tags_sidecar(image_path: Path, tag_text: str) -> Path:
    """Write tag text to a ``.txt`` file next to the image. Returns the txt path."""
    txt = image_path.with_suffix(".txt")
    txt.write_text(tag_text + "\n", encoding="utf-8")
    return txt


def split_sidecar(text: str) -> tuple[str, str]:
    """Return ``(danbooru_line, llm_description)`` from a sidecar's text.

    The sidecar layout is two-line: comma-separated WD14 tags on row 1, then
    an optional LLM description on row 2. Anything past the second non-empty
    line is preserved into the description so we never silently drop content.
    """
    if not text:
        return "", ""
    # rstrip only the trailing newline(s) — internal blank lines are unusual
    # but we tolerate them by skipping leading blanks before the description.
    lines = text.splitlines()
    danbooru = lines[0].strip() if lines else ""
    rest = lines[1:]
    # Drop a single leading blank between rows so editors that double-newline
    # don't blow up the description with a phantom empty line.
    while rest and not rest[0].strip():
        rest = rest[1:]
    description = "\n".join(rest).rstrip()
    return danbooru, description


def _dedupe_tags(danbooru: str) -> str:
    """Trim, drop blanks, and remove duplicate tags preserving first-occurrence order.

    Centralized in :func:`join_sidecar` so every sidecar write normalizes the
    danbooru line once, regardless of who produced it (auto-tagger, manual
    pill edit, regex replace, LLM retag). Two reasons:

    * The frontend's keyed ``{#each}`` over tag pills uses tag text as the
      key — duplicates collapse the rendered list and the user sees the row
      go blank after committing a repeat.
    * Duplicate tags add no signal for the trainer; they just inflate tokens.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in danbooru.split(","):
        t = raw.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return ", ".join(out)


def join_sidecar(danbooru: str, description: str) -> str:
    """Render ``(danbooru, description)`` back into the two-line sidecar form.

    Always trailing-newline terminated for POSIX-friendly diffs. An empty
    description collapses to a single line so files written before LLM
    tagging existed stay byte-identical when round-tripped. The danbooru
    line is also deduped here — it's the single chokepoint every sidecar
    write goes through.
    """
    danbooru = _dedupe_tags(danbooru)
    description = description.strip()
    if description:
        return f"{danbooru}\n{description}\n"
    return f"{danbooru}\n"

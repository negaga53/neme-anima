"""Project: a folder of input videos + reference images + extracted output.

Layout under the project root:

    project.json
    refs/                    (link targets; thumbnails cached under .thumbnails/)
    output/
      kept/                  (all kept frames, prefixed with <video_stem>__)
      rejected/
      metadata.jsonl         (each row carries character_slug)
      cache/<video_stem>/    (per-video detection cache, parquet)

Multi-character data model:
    A project owns a list of ``Character`` records. Each character carries
    its own reference images and training config. A single-character project
    (the historical shape) auto-migrates to one ``"default"`` character.
    Files on disk stay flat under ``kept/`` — the per-frame metadata row
    carries ``character_slug`` so per-character grouping happens at read time
    (e.g. training dataset assembly) rather than at write time. This keeps
    the existing ``project.kept_dir / filename.png`` lookups intact.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

VIDEO_EXTENSIONS = frozenset({
    ".mkv", ".mp4", ".webm", ".mov", ".avi", ".m4v", ".ts", ".wmv",
})

# Suffix appended to a kept frame's filename to mark its crop derivative
# (`<filename>_crop.png`). Defined here because the layout is shared by the
# API (which writes/deletes derivatives) and the trainer (which pairs each
# derivative with the original's `.txt` at staging time). There's only ever
# one derivative per original; re-cropping overwrites it.
CROP_SUFFIX = "_crop"


def refs_dir_contains(project_root: Path, candidate: Path) -> bool:
    """True iff ``candidate`` resolves to a file under ``project_root/refs/``."""
    try:
        candidate.resolve().relative_to((project_root / "refs").resolve())
        return True
    except (ValueError, OSError):
        return False


def list_videos(folder: Path) -> list[Path]:
    """Return a sorted list of video files directly under ``folder`` (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


DEFAULT_CHARACTER_SLUG = "default"


def _slugify_character_name(name: str) -> str:
    """Lower-case, ASCII-safe slug for use as a directory/dict key.

    Mirrors the project-slug convention: only alphanumerics + hyphens, with
    runs of unsafe characters collapsed to a single hyphen. Empty input maps
    to ``"character"`` so callers always get a non-empty slug.
    """
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "")).strip("-").lower()
    return s or "character"


@dataclass
class Segment:
    """A time-range [start_seconds, end_seconds) inside a source video that the
    user wants the pipeline to process. Stored in seconds (not frames) so the
    record stays meaningful even if the file is re-encoded with a different
    framerate; pipeline converts to frame indices at extraction time via the
    video's actual fps.
    """
    start_seconds: float
    end_seconds: float


@dataclass
class Source:
    """An input video tracked by the project.

    ``excluded_refs`` is a per-character map: ``{character_slug: [ref_path, ...]}``
    listing the refs to skip for *that* character on this source. Old projects
    persisted a flat list (no character context) — auto-migrated on load to
    ``{"default": [...]}``.

    ``segments`` is an optional list of time-ranges to restrict extraction to.
    Empty list = process the whole video (legacy / default behaviour).
    ``duration_seconds`` / ``fps`` are convenience caches populated on first
    ffprobe so the segment-editor UI doesn't pay the probe cost on every open.
    """
    path: str                         # absolute path to the video file
    added_at: str                     # ISO-8601 UTC
    excluded_refs: dict[str, list[str]] = field(default_factory=dict)
    extraction_runs: list[dict] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    duration_seconds: float | None = None
    fps: float | None = None
    vcodec: str | None = None         # primary video codec (e.g. "hevc"), cached


@dataclass
class RefImage:
    """A reference image used for character matching."""
    path: str
    added_at: str


@dataclass
class LLMConfig:
    """Optional second-pass image-description tagger using an OpenAI-compatible
    chat-completions endpoint (LMStudio by default).

    ``enabled`` only flips on once the user has both pointed at a reachable
    server and picked a model from its discovery response — see
    :func:`neme_anima.llm.discover_models`. The pipeline treats
    ``enabled=False`` *or* ``model==""`` as off, so the disabled-by-default
    behaviour falls out without an extra check.

    ``api_key`` is empty by default — LMStudio doesn't require auth. Set it
    when targeting providers that gate ``/v1/models`` and ``/v1/chat/completions``
    behind a bearer token (OpenAI, OpenRouter, hosted vLLM, etc.).

    ``disable_thinking`` asks a reasoning model to skip its thinking pass. Off
    by default because it sends a non-universal request field: the knob is
    server-specific and a strict server may reject the whole request, so it is
    opt-in per project rather than something every endpoint pays for.
    """
    enabled: bool = False
    endpoint: str = "http://localhost:1234"
    model: str = ""
    prompt: str = ""  # empty = use llm.DEFAULT_PROMPT
    api_key: str = ""  # empty = no Authorization header (LMStudio default)
    disable_thinking: bool = False


@dataclass
class TrainingConfig:
    """Anima LoRA-training settings, persisted alongside the project.

    Defaults match the official tdrussell style-LoRA recipe (see
    docs/anima-lora-training-notes.md). Three groups: trainer paths
    (validated to actually exist on disk before a run is allowed to start),
    hyperparameters (faithful to the reference TOML), and captioning +
    checkpoint retention. ``keep_last_n_checkpoints == 0`` means
    "keep every epoch adapter" — the user-requested default; DeepSpeed
    resume state is always trimmed to the latest regardless (see
    ``training.prune_checkpoints``).
    """

    preset: str = "style"  # "style" | "character"

    # Trainer paths — none of these are auto-downloaded.
    diffusion_pipe_dir: str = ""
    dit_path: str = ""        # anima-base-v1.0.safetensors
    vae_path: str = ""        # qwen_image_vae.safetensors
    llm_path: str = ""        # qwen_3_06b_base.safetensors
    launcher_override: str = ""  # empty -> built-in deepspeed command

    # Adapter
    rank: int = 32
    alpha: int = 16  # kohya-only; ignored by canonical tdrussell schema

    # Optimizer / schedule
    learning_rate: float = 2e-5
    optimizer_betas: list[float] = field(default_factory=lambda: [0.9, 0.99])
    weight_decay: float = 0.01
    eps: float = 1e-8
    warmup_steps: int = 100
    gradient_clipping: float = 1.0

    # Batching
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # VRAM levers — usually only set by the "Fit in 8 GB" preset, hidden
    # from the regular form. Defaults match the recipe so the 24 GB-class
    # path stays byte-identical to its pre-preset output.
    #
    # ``transformer_dtype`` controls the storage dtype of the diffusion
    # transformer ("bfloat16" = recipe default, "float8" = ~half the VRAM
    # at a small quality cost — note: structurally broken for Anima,
    # rejected by validate_for_run, see comment there).
    # ``blocks_to_swap`` asks diffusion-pipe to keep N transformer blocks
    # on CPU and stream them in/out per step — heavy on CPU↔GPU traffic
    # but the difference between OOM and a usable run on small cards.
    # ``optimizer_type`` is the diffusion-pipe ``[optimizer] type`` value
    # — defaults to the recipe's "adamw_optimi"; "AdamW8bitKahan" is the
    # 8-bit-state variant that saves ~75% of optimizer-state VRAM (used
    # by the canonical wan_14b_min_vram example). The optimizer kwargs
    # (lr / betas / weight_decay / eps) flow through unchanged — train.py
    # builds the optimizer kwargs by stripping ``type`` and forwarding the
    # rest, so a swap is purely additive.
    # ``activation_checkpointing_mode`` chooses the recompute strategy:
    # "default" → ``activation_checkpointing = true`` (PyTorch native
    # checkpoint), "unsloth" → unsloth's more aggressive variant (less
    # GPU memory at a small CPU cost).
    transformer_dtype: str = "bfloat16"
    blocks_to_swap: int = 0
    optimizer_type: str = "adamw_optimi"
    activation_checkpointing_mode: str = "default"

    # Resolution / bucketing
    resolutions: list[int] = field(default_factory=lambda: [512, 1024])
    enable_ar_bucket: bool = True
    min_ar: float = 0.5
    max_ar: float = 2.0
    num_ar_buckets: int = 9

    # Duration
    epochs: int = 40
    eval_every_n_epochs: int = 5
    save_every_n_epochs: int = 10

    # Anima specifics — llm_adapter_lr=0 prevents style dilution per the
    # reference recipe; we expose it but the UI warns against changing it.
    sigmoid_scale: float = 1.3
    llm_adapter_lr: float = 0.0

    # Captioning
    caption_mode: str = "mixed"  # "tags" | "nl" | "mixed"
    tag_dropout_pct: int = 10
    trigger_token: str = ""

    # Retention: how many epoch (LoRA) checkpoints to keep; 0 = keep all
    # (the user-requested default). DeepSpeed resume state (global_step dirs)
    # is always trimmed to the latest, independent of this number.
    keep_last_n_checkpoints: int = 0

    # Sample generation: preview images rendered from each qualifying epoch
    # LoRA by ``_anima_sampler.py`` (vendored ComfyUI in diffusion-pipe's
    # venv). Off while ``sample_prompts`` has no non-blank entry. Which
    # epochs qualify: see ``training.sample_epoch_qualifies``.
    sample_prompts: list[str] = field(default_factory=list)
    sample_negative_prompt: str = ""
    sample_every_n_epochs: int = 10
    sample_defer_to_end: bool = False  # low-VRAM: sample after training exits
    sample_steps: int = 30
    sample_sampler: str = "euler_ancestral"
    sample_scheduler: str = "simple"
    sample_cfg: float = 4.5
    sample_width: int = 1024
    sample_height: int = 1024
    sample_seed: int = 42  # fixed so epochs compare image-to-image

    def __post_init__(self) -> None:
        # Apply install-time prefill defaults for any path field the user
        # hasn't filled in. install_and_run.sh writes a JSON file at
        # ~/.neme-anima/training-defaults.json with paths it set up
        # (diffusion-pipe clone, downloaded model weights). This hook lets
        # those defaults flow into newly-created projects without having
        # to teach the UI a separate "global settings" page. Paths that
        # the user has explicitly filled in are never overridden.
        defaults_path = Path.home() / ".neme-anima" / "training-defaults.json"
        if not defaults_path.is_file():
            return
        try:
            globals_ = json.loads(defaults_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for key in ("diffusion_pipe_dir", "dit_path", "vae_path", "llm_path"):
            if not getattr(self, key, "") and globals_.get(key):
                setattr(self, key, str(globals_[key]))


@dataclass
class Character:
    """One trainable character within a project.

    Each character carries its own references and training config. The slug
    is the primary key within the project (used in metadata rows and as the
    directory key for staged training datasets). The trigger token is what
    the user puts in their LoRA prompt to invoke this character; if empty
    the trainer falls back to the slug.

    Core-tag pruning fields:
      * ``core_tags`` is a persisted list of tags that the user has
        approved as "implied by the trigger word". When pruning is enabled,
        the dataset-staging step strips these from each frame's caption.
      * ``core_tags_freq_threshold`` is the frequency cutoff used by the
        compute-preview step. Defaults to 0.35 — anime2sd's value.
      * ``core_tags_enabled`` gates whether pruning is applied at staging
        time. Off by default so users opt in once they've reviewed the
        suggested list.

    Balancing:
      * ``multiply`` is a per-character training-set repeat multiplier.
        ``0.0`` means "auto" — the balancing pass computes a value from
        relative frame counts. A positive value is a manual override.
    """
    slug: str
    name: str
    refs: list[RefImage] = field(default_factory=list)
    trigger_token: str = ""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    core_tags: list[str] = field(default_factory=list)
    core_tags_freq_threshold: float = 0.35
    core_tags_enabled: bool = False
    multiply: float = 0.0


@dataclass
class Project:
    name: str
    slug: str
    root: Path
    created_at: datetime
    sources: list[Source] = field(default_factory=list)
    # Characters in display order. Always non-empty after ``Project.load`` —
    # legacy single-character projects auto-migrate into one default character
    # whose slug is :data:`DEFAULT_CHARACTER_SLUG`.
    characters: list[Character] = field(default_factory=list)
    thresholds_overrides: dict = field(default_factory=dict)
    source_root: str | None = None
    # When True, extract/rerun pipelines pause after writing kept frames to
    # disk and wait for an explicit resume signal before tagging — giving the
    # user a chance to delete unwanted frames so they don't pay the tagging
    # cost on them. False = tag inline like the original pipeline.
    pause_before_tag: bool = True
    # When True, the pipeline deletes every file in ``output/rejected/``
    # matching the current ``<video_stem>__*`` prefix after each video
    # finishes. Off by default so users can audit rejections; flip on
    # once you trust the matching thresholds.
    auto_delete_rejected: bool = False
    # When True (default), the tag editor shows a danbooru autocomplete
    # dropdown. Purely a UI nicety — has no effect on the pipeline.
    tag_autocomplete: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)

    # ---------------- factory methods ----------------

    @classmethod
    def create(cls, root: Path, *, name: str) -> Project:
        root = Path(root)
        if root.exists():
            raise FileExistsError(f"refusing to overwrite existing folder {root}")
        slug = root.name
        now = datetime.now(UTC)
        project = cls(
            name=name,
            slug=slug,
            root=root,
            created_at=now,
            characters=[Character(slug=DEFAULT_CHARACTER_SLUG, name=name)],
        )
        # Folder skeleton.
        (root / "refs" / ".thumbnails").mkdir(parents=True)
        (root / "output" / "kept").mkdir(parents=True)
        (root / "output" / "rejected").mkdir(parents=True)
        (root / "output" / "cache").mkdir(parents=True)
        project.save()
        return project

    @classmethod
    def load(cls, root: Path) -> Project:
        root = Path(root)
        with open(root / "project.json") as f:
            data = json.load(f)
        llm_raw = data.get("llm") or {}

        characters = cls._load_characters(data)
        sources = cls._load_sources(data, characters)

        return cls(
            name=data["name"],
            slug=data["slug"],
            root=root,
            created_at=datetime.fromisoformat(data["created_at"]),
            sources=sources,
            characters=characters,
            thresholds_overrides=data.get("thresholds_overrides", {}),
            source_root=data.get("source_root"),
            pause_before_tag=bool(data.get("pause_before_tag", True)),
            auto_delete_rejected=bool(data.get("auto_delete_rejected", False)),
            tag_autocomplete=bool(data.get("tag_autocomplete", True)),
            llm=LLMConfig(
                enabled=bool(llm_raw.get("enabled", False)),
                endpoint=str(llm_raw.get("endpoint") or "http://localhost:1234"),
                model=str(llm_raw.get("model") or ""),
                prompt=str(llm_raw.get("prompt") or ""),
                api_key=str(llm_raw.get("api_key") or ""),
            ),
        )

    @staticmethod
    def _load_characters(data: dict) -> list[Character]:
        """Build the character list from a parsed project.json.

        Handles three shapes:
          1. New: ``characters: [{slug, name, refs, trigger_token, training}, ...]``
          2. Old: top-level ``refs: [...]`` and ``training: {...}`` → synthesized
             into one ``"default"`` character. The project's display name is
             used as the character name, since old projects = one character.
          3. Empty: no refs and no training → still emit one default character
             so callers can rely on ``project.characters[0]`` existing.

        Tolerates missing keys throughout so older / partial files keep loading.
        """
        if "characters" in data and data["characters"]:
            chars: list[Character] = []
            for raw in data["characters"]:
                training_raw = raw.get("training") or {}
                training = TrainingConfig(**{
                    f.name: training_raw[f.name]
                    for f in fields(TrainingConfig())
                    if f.name in training_raw
                })
                chars.append(Character(
                    slug=str(raw.get("slug") or DEFAULT_CHARACTER_SLUG),
                    name=str(raw.get("name") or raw.get("slug") or "Character"),
                    refs=[RefImage(**r) for r in raw.get("refs", [])],
                    trigger_token=str(raw.get("trigger_token") or ""),
                    training=training,
                    # Core-tag fields default-tolerantly so a project saved
                    # before this change loads with sane defaults rather
                    # than crashing on the new keys.
                    core_tags=[str(t) for t in raw.get("core_tags", [])],
                    core_tags_freq_threshold=float(
                        raw.get("core_tags_freq_threshold", 0.35),
                    ),
                    core_tags_enabled=bool(raw.get("core_tags_enabled", False)),
                    multiply=float(raw.get("multiply", 0.0)),
                ))
            return chars
        # Legacy single-character migration.
        old_training = data.get("training") or {}
        training = TrainingConfig(**{
            f.name: old_training[f.name]
            for f in fields(TrainingConfig())
            if f.name in old_training
        })
        return [Character(
            slug=DEFAULT_CHARACTER_SLUG,
            name=str(data.get("name") or "default"),
            refs=[RefImage(**r) for r in data.get("refs", [])],
            training=training,
        )]

    @staticmethod
    def _load_sources(data: dict, characters: list[Character]) -> list[Source]:
        """Load sources, migrating per-source ``excluded_refs`` shape.

        Old shape was a flat list of ref paths; new shape is a dict keyed by
        character slug. A flat list is interpreted as "the default character's
        opt-outs" and lifted into ``{"default": [...]}`` so per-character
        opt-outs work for every character thereafter.
        """
        default_slug = (
            characters[0].slug if characters else DEFAULT_CHARACTER_SLUG
        )
        out: list[Source] = []
        for s in data.get("sources", []):
            excluded = s.get("excluded_refs", {})
            if isinstance(excluded, list):
                excluded = {default_slug: list(excluded)} if excluded else {}
            segments_raw = s.get("segments") or []
            segments: list[Segment] = []
            for seg in segments_raw:
                try:
                    segments.append(Segment(
                        start_seconds=float(seg["start_seconds"]),
                        end_seconds=float(seg["end_seconds"]),
                    ))
                except (KeyError, TypeError, ValueError):
                    # Silently drop malformed rows — the field is optional
                    # and a corrupt entry shouldn't prevent project load.
                    continue
            duration_raw = s.get("duration_seconds")
            fps_raw = s.get("fps")
            out.append(Source(
                path=s["path"],
                added_at=s["added_at"],
                excluded_refs=dict(excluded),
                extraction_runs=s.get("extraction_runs", []),
                segments=segments,
                duration_seconds=(
                    float(duration_raw) if duration_raw is not None else None
                ),
                fps=float(fps_raw) if fps_raw is not None else None,
                vcodec=s.get("vcodec"),
            ))
        return out

    def save(self) -> None:
        out = {
            "name": self.name,
            "slug": self.slug,
            "created_at": self.created_at.isoformat(),
            "sources": [asdict(s) for s in self.sources],
            "characters": [asdict(c) for c in self.characters],
            "thresholds_overrides": self.thresholds_overrides,
            "source_root": self.source_root,
            "pause_before_tag": self.pause_before_tag,
            "auto_delete_rejected": self.auto_delete_rejected,
            "tag_autocomplete": self.tag_autocomplete,
            "llm": asdict(self.llm),
        }
        tmp = self.root / "project.json.tmp"
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(self.root / "project.json")

    # ---------------- character helpers ----------------

    @property
    def refs(self) -> list[RefImage]:
        """Backwards-compat alias: refs of the default (first) character.

        Existing API endpoints and the mono-character UI read this. New
        character-aware code should index ``project.characters`` directly.
        """
        return self.characters[0].refs if self.characters else []

    @refs.setter
    def refs(self, value: list[RefImage]) -> None:
        if not self.characters:
            self.characters = [Character(slug=DEFAULT_CHARACTER_SLUG,
                                         name=self.name, refs=list(value))]
        else:
            self.characters[0].refs = list(value)

    @property
    def training(self) -> TrainingConfig:
        """Backwards-compat alias: training config of the default character."""
        if not self.characters:
            self.characters = [Character(slug=DEFAULT_CHARACTER_SLUG, name=self.name)]
        return self.characters[0].training

    @training.setter
    def training(self, value: TrainingConfig) -> None:
        if not self.characters:
            self.characters = [Character(slug=DEFAULT_CHARACTER_SLUG,
                                         name=self.name, training=value)]
        else:
            self.characters[0].training = value

    def character_by_slug(self, slug: str) -> Character | None:
        for c in self.characters:
            if c.slug == slug:
                return c
        return None

    def add_character(self, *, name: str, slug: str | None = None) -> Character:
        """Append a new character to this project.

        Slug is derived from ``name`` if not given, then made unique within
        the project by appending ``-2``, ``-3``, ... so two characters named
        "Yui" never collide on disk.
        """
        base = _slugify_character_name(slug or name)
        candidate = base
        n = 2
        existing = {c.slug for c in self.characters}
        while candidate in existing:
            candidate = f"{base}-{n}"
            n += 1
        c = Character(slug=candidate, name=name)
        self.characters.append(c)
        self.save()
        return c

    def remove_character(self, slug: str) -> None:
        """Drop a character from the project.

        Refuses to remove the last remaining character — every project must
        carry at least one. Cleans up per-source opt-out maps so dangling
        keys don't accumulate.
        """
        if len(self.characters) <= 1:
            raise ValueError("cannot remove the last character — projects need at least one")
        self.characters = [c for c in self.characters if c.slug != slug]
        for s in self.sources:
            s.excluded_refs.pop(slug, None)
        self.save()

    def _resolve_character(self, slug: str | None) -> Character:
        """Return the named character, or the default (first) one when ``slug``
        is None. Raises ``KeyError`` if a non-None slug doesn't exist — caller
        bugs (e.g. UI hands us a stale slug) shouldn't be silently rerouted to
        the default character.
        """
        if slug is None:
            if not self.characters:
                self.characters = [Character(slug=DEFAULT_CHARACTER_SLUG, name=self.name)]
            return self.characters[0]
        c = self.character_by_slug(slug)
        if c is None:
            raise KeyError(f"unknown character slug: {slug!r}")
        return c

    # ---------------- mutations ----------------

    def add_source(self, video_path: Path) -> Source:
        video_path = Path(video_path).resolve()
        if any(Path(s.path) == video_path for s in self.sources):
            raise ValueError(f"video already in project: {video_path}")
        s = Source(
            path=str(video_path),
            added_at=datetime.now(UTC).isoformat(),
        )
        self.sources.append(s)
        self.save()
        return s

    def add_ref(self, ref_path: Path, *, character_slug: str | None = None) -> RefImage:
        """Copy an external image into the project's refs/ folder and track it
        for the given character (default character when ``character_slug`` is
        ``None``)."""
        ref_path = Path(ref_path)
        if not ref_path.is_file():
            raise FileNotFoundError(ref_path)
        return self._ingest_ref(ref_path.name, ref_path.read_bytes(),
                                character_slug=character_slug)

    def add_ref_bytes(
        self, filename: str, data: bytes, *, character_slug: str | None = None,
    ) -> RefImage:
        """Save uploaded image bytes into the project's refs/ folder and track
        it for the given character (default character when ``character_slug``
        is ``None``)."""
        return self._ingest_ref(filename, data, character_slug=character_slug)

    def _ingest_ref(
        self, filename: str, data: bytes, *, character_slug: str | None = None,
    ) -> RefImage:
        character = self._resolve_character(character_slug)
        refs_dir = self.root / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        dest = self._unique_ref_path(filename)
        dest.write_bytes(data)
        r = RefImage(
            path=str(dest.resolve()),
            added_at=datetime.now(UTC).isoformat(),
        )
        character.refs.append(r)
        self.save()
        return r

    def _unique_ref_path(self, filename: str) -> Path:
        """Return a refs/ destination path that doesn't collide with an existing ref."""
        # Sanitize: drop any path components, keep only basename.
        name = Path(filename).name or "ref"
        dest = self.root / "refs" / name
        if not dest.exists():
            return dest
        stem, suffix = dest.stem, dest.suffix
        for n in range(2, 10_000):
            candidate = self.root / "refs" / f"{stem}-{n}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"too many copies of ref named {name!r}")

    def remove_source(self, source_idx: int) -> None:
        del self.sources[source_idx]
        self.save()

    def remove_ref(self, ref_path: str) -> None:
        """Drop a ref path from every character that owns it.

        We don't require the caller to know which character owned the ref —
        the path is unique inside the project's refs/ folder, so removing it
        means it's gone for *all* characters. The on-disk file is then deleted
        only if no character still references it (defence against future
        sharing semantics) and only if the file lives under the project's own
        refs/ folder.
        """
        ref_path = str(Path(ref_path).resolve())
        deleted_paths: set[str] = set()
        for c in self.characters:
            new_refs: list[RefImage] = []
            for r in c.refs:
                if r.path == ref_path:
                    deleted_paths.add(r.path)
                else:
                    new_refs.append(r)
            c.refs = new_refs
        # Strip dangling opt-outs across every (source, character) pair.
        for s in self.sources:
            for slug, paths in list(s.excluded_refs.items()):
                s.excluded_refs[slug] = [p for p in paths if p != ref_path]
                if not s.excluded_refs[slug]:
                    del s.excluded_refs[slug]
        self.save()
        # Delete the on-disk file only if it's gone from every character now,
        # and only if it lives under our refs/ folder.
        still_referenced = any(
            r.path == ref_path for c in self.characters for r in c.refs
        )
        if still_referenced:
            return
        for d in {Path(p) for p in deleted_paths}:
            try:
                if d.is_file() and refs_dir_contains(self.root, d):
                    d.unlink()
            except OSError:
                pass

    # ---------------- folder-based source import ----------------

    def import_videos_from_folder(
        self, folder: Path, *, set_root: bool = True
    ) -> tuple[list[Source], list[str]]:
        """Add every video file in ``folder`` as a source.

        Returns ``(added, skipped)`` where ``skipped`` contains the resolved paths
        that were already in the project.
        """
        folder = Path(folder)
        added: list[Source] = []
        skipped: list[str] = []
        for vid in list_videos(folder):
            try:
                added.append(self.add_source(vid))
            except ValueError:
                skipped.append(str(vid.resolve()))
        if set_root:
            self.source_root = str(folder.resolve())
            self.save()
        return added, skipped

    def set_excluded_refs(
        self, source_idx: int, excluded: list[str], *, character_slug: str | None = None,
    ) -> None:
        """Replace the opt-out list for ``(source, character)`` with ``excluded``.

        ``character_slug=None`` updates the default (first) character. An empty
        list clears the entry so the underlying dict doesn't accumulate empty
        lists across saves.
        """
        character = self._resolve_character(character_slug)
        normalized = [str(Path(p).resolve()) for p in excluded]
        src = self.sources[source_idx]
        if normalized:
            src.excluded_refs[character.slug] = normalized
        else:
            src.excluded_refs.pop(character.slug, None)
        self.save()

    # ---------------- ref-set + path helpers ----------------

    def effective_refs_for(
        self, source_idx: int, *, character_slug: str | None = None,
    ) -> list[str]:
        """Ref paths for ``(source, character)`` minus per-video opt-outs.

        ``character_slug=None`` returns the default character's effective set,
        which is what every existing call site wants while the UI is still
        single-character.
        """
        character = self._resolve_character(character_slug)
        excluded = set(self.sources[source_idx].excluded_refs.get(character.slug, []))
        return [r.path for r in character.refs if r.path not in excluded]

    def video_stem(self, source_idx: int) -> str:
        return Path(self.sources[source_idx].path).stem

    @property
    def kept_dir(self) -> Path:
        return self.root / "output" / "kept"

    @property
    def rejected_dir(self) -> Path:
        return self.root / "output" / "rejected"

    @property
    def metadata_path(self) -> Path:
        return self.root / "output" / "metadata.jsonl"

    def cache_dir_for(self, video_stem: str) -> Path:
        return self.root / "output" / "cache" / video_stem

    # ---------------- training ----------------

    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def training_runs_dir(self) -> Path:
        return self.training_dir / "runs"

    @property
    def training_state_path(self) -> Path:
        """Where the live runner persists its state across server restarts."""
        return self.training_dir / "state.json"

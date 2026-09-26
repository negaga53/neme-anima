"""Anima LoRA-training helpers.

This module is **pure plumbing**: it knows how to validate the user's
configuration, render dataset/run TOML files for tdrussell/diffusion-pipe,
discover checkpoints on disk, and prune old ones according to retention
policy. The actual training subprocess lives in
``neme_anima.server.training_runner``.

The TOML output mirrors the reference recipe documented in
``docs/anima-lora-training-notes.md``. We deliberately avoid a TOML library
dependency — the schema is simple, stable, and easier to read in plain
strings than rebuilt through ``tomli_w``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from neme_anima.storage.project import CROP_SUFFIX, Project, TrainingConfig

# ----- caption defaults -----------------------------------------------------

# 2-sentence quality prefix from the Anima model card (tag-ordered structure).
DEFAULT_QUALITY_PREFIX = "masterpiece, best quality, score_7, safe,"

# Caption suffix for the natural-language portion when the user has no LLM
# description on disk yet. Empty string => no NL appended.
DEFAULT_NL_FALLBACK = ""


# ----- dataclasses ----------------------------------------------------------


@dataclass
class CheckpointInfo:
    """One on-disk training checkpoint produced by diffusion-pipe."""

    name: str               # e.g. "epoch20" or "global_step5000"
    path: str               # absolute path to the directory
    epoch: int | None       # parsed epoch number if the name encodes it
    step: int | None        # parsed global step if encoded
    size_bytes: int         # cumulative size on disk (best-effort)
    modified_at: str        # ISO-8601
    # Path of the parent directory relative to ``run_dir`` ("" for direct
    # children). diffusion-pipe writes its checkpoints into a timestamped
    # subdirectory under our run wrapper, so this is usually non-empty.
    subdir: str = ""


@dataclass
class PathCheck:
    path: str
    exists: bool
    is_file: bool
    is_dir: bool
    error: str | None = None


class SafetensorsMetadataError(ValueError):
    """Raised when a file cannot be safely parsed as safetensors."""


# ----- path validation -------------------------------------------------------


def check_path(raw: str, *, expect: str = "any") -> PathCheck:
    """Stat ``raw`` and report what it is.

    ``expect`` ∈ {"any", "file", "dir"} — if set, ``error`` is populated when
    the resolved path doesn't match. Empty/whitespace input returns
    ``exists=False`` with a friendly error so the UI can render a red X
    without an exception.
    """
    raw = (raw or "").strip()
    if not raw:
        return PathCheck(
            path=raw, exists=False, is_file=False, is_dir=False,
            error="path is empty",
        )
    p = Path(raw).expanduser()
    try:
        exists = p.exists()
    except OSError as e:
        return PathCheck(
            path=str(p), exists=False, is_file=False, is_dir=False, error=str(e),
        )
    is_file = p.is_file()
    is_dir = p.is_dir()
    error: str | None = None
    if not exists:
        error = "no such file or directory"
    elif expect == "file" and not is_file:
        error = "expected a file, got a directory"
    elif expect == "dir" and not is_dir:
        error = "expected a directory, got a file"
    return PathCheck(
        path=str(p), exists=exists, is_file=is_file, is_dir=is_dir, error=error,
    )


def _diffusion_pipe_python_problem(diffusion_pipe_dir: str) -> str | None:
    """Return a problem string when diffusion-pipe's local venv is too old."""
    if not diffusion_pipe_dir:
        return None
    dp = Path(diffusion_pipe_dir).expanduser()
    python = next(
        (
            candidate
            for candidate in (
                dp / ".venv" / "bin" / "python",
                dp / "venv" / "bin" / "python",
            )
            if candidate.exists()
        ),
        None,
    )
    if python is None:
        return None
    if not python.is_file():
        return f"diffusion-pipe venv python is not a file: {python}"
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import sys; "
                    "print(f'{sys.version_info.major}.{sys.version_info.minor}')"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError as e:
        return f"could not run diffusion-pipe venv python at {python}: {e}"
    except subprocess.TimeoutExpired:
        return f"timed out checking diffusion-pipe venv python at {python}"
    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        details = result.stderr.strip() or f"exit code {result.returncode}"
        return (
            f"could not determine diffusion-pipe venv Python version at {python}: "
            f"{details}"
        )
    major_minor = tuple(
        int(part) for part in version.split(".", 2)[:2] if part.isdigit()
    )
    if major_minor < (3, 12):
        return (
            f"diffusion-pipe venv uses Python {version}; diffusion-pipe requires "
            "Python 3.12+ because it calls sqlite3.connect(..., autocommit=...). "
            f"Recreate it with: cd '{dp}' && "
            "rm -rf .venv && uv venv --python 3.12 && "
            "uv pip install --python .venv/bin/python -r requirements.txt"
        )
    return None


def validate_for_run(config: TrainingConfig) -> list[str]:
    """Return a list of human-readable problems blocking a training run.

    Empty list = good to go. Used at the API layer to gate POST /start.
    """
    problems: list[str] = []
    checks = {
        "diffusion-pipe directory": check_path(
            config.diffusion_pipe_dir, expect="dir",
        ),
        "Anima DiT (transformer) file": check_path(
            config.dit_path, expect="file",
        ),
        "Qwen image VAE file": check_path(config.vae_path, expect="file"),
        "Qwen 3 0.6B base text encoder file": check_path(
            config.llm_path, expect="file",
        ),
    }
    for label, c in checks.items():
        if c.error:
            problems.append(f"{label}: {c.error} ({c.path or '<empty>'})")

    train_py = Path(config.diffusion_pipe_dir).expanduser() / "train.py"
    if config.diffusion_pipe_dir and not train_py.is_file():
        problems.append(
            f"diffusion-pipe directory does not contain train.py at {train_py}",
        )
    python_problem = _diffusion_pipe_python_problem(config.diffusion_pipe_dir)
    if python_problem:
        problems.append(python_problem)

    # Pre-flight the launcher binary so we surface a clear UI message instead
    # of a generic FileNotFoundError out of subprocess_exec at run time.
    token, resolved = resolve_launcher_binary(config)
    if token and resolved is None:
        problems.append(
            f"launcher binary not found: {token!r}. Install diffusion-pipe "
            f"dependencies into a venv at "
            f"'{config.diffusion_pipe_dir or '<diffusion_pipe_dir>'}/.venv' "
            f"(see its README), or set launcher_override to an absolute "
            f"path like '/path/to/env/bin/deepspeed --num_gpus=1 "
            f"train.py --deepspeed --config {{config}}'",
        )

    if not config.resolutions:
        problems.append("at least one training resolution is required")
    if config.epochs <= 0:
        problems.append("epochs must be > 0")
    if config.rank <= 0:
        problems.append("rank must be > 0")
    if config.learning_rate <= 0:
        problems.append("learning_rate must be > 0")
    # Anima's LLM adapter is structurally incompatible with diffusion-pipe's
    # current fp8 path: the adapter's nn.Embedding has ndim==2, so the
    # transformer-dtype cast quantizes its weights to fp8; embedding lookup
    # then returns fp8 tensors that flow into RMSNorm, which crashes
    # promoting fp8*fp32 in the rsqrt branch. Block training before the
    # subprocess gets there so the user sees a UI-actionable message
    # instead of a long deepspeed traceback. Drop this guard if/when
    # diffusion-pipe adds llm_adapter.embed/in_proj/out_proj to its
    # KEEP_IN_HIGH_PRECISION list (see models/cosmos_predict2.py).
    if config.transformer_dtype not in ("", "bfloat16"):
        problems.append(
            f"transformer_dtype={config.transformer_dtype!r} is not "
            "supported for Anima: diffusion-pipe's fp8 path crashes "
            "Anima's LLM adapter (RMSNorm fp8/fp32 promotion error). "
            "Use 'bfloat16' — toggling the 'Fit in 8 GB' preset off and "
            "on resets this field."
        )
    problems.extend(_sample_problems(config))
    return problems


# ----- TOML generation -------------------------------------------------------


def _toml_value(v: object) -> str:
    """Tiny TOML emitter — handles the subset we use (int/float/bool/str/list)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        # Avoid scientific-notation floats for clarity — diffusion-pipe is
        # tolerant of either, but readable TOML is nicer for the user.
        if isinstance(v, float):
            return repr(v)
        return str(v)
    if isinstance(v, str):
        # Use double-quoted strings, escape backslashes/double-quotes.
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value type: {type(v).__name__}")


def _toml_kv(key: str, value: object) -> str:
    return f"{key} = {_toml_value(value)}"


def render_dataset_toml(project: Project, *, dataset_root: Path | None = None) -> str:
    """Build the dataset.toml that points diffusion-pipe at the training
    dataset directory.

    ``dataset_root`` is the directory diffusion-pipe will scan for image +
    ``.txt`` sidecar pairs. Callers should normally pass the staging
    directory built by :func:`build_dataset_staging`, which substitutes
    each cropped frame's image while keeping the original sidecar. When
    omitted, falls back to ``project.kept_dir`` — useful for previews and
    backwards compatibility, but training will then see both the original
    and the ``_crop`` derivative as separate samples (the historical bug).

    Multi-character projects emit one ``[[directory]]`` block per
    character pointing at the per-character staging subdirectory built by
    :func:`build_dataset_staging`, with ``num_repeats`` set to that
    character's effective balancing multiplier. Single-character projects
    keep a single block pointing at the staging root for backwards
    compatibility.

    AR bucketing values come from the project's ``TrainingConfig`` so
    ``num_ar_buckets`` and ``min_ar``/``max_ar`` stay in sync with the run
    config.
    """
    from neme_anima.balancing import compute_character_balancing

    cfg = project.training
    root = dataset_root if dataset_root is not None else project.kept_dir
    lines = [
        "# Auto-generated by neme-anima — do not hand-edit.",
        f'# Project: {project.slug}',
        "",
        "resolutions = " + _toml_value(cfg.resolutions),
        _toml_kv("enable_ar_bucket", cfg.enable_ar_bucket),
        _toml_kv("min_ar", cfg.min_ar),
        _toml_kv("max_ar", cfg.max_ar),
        _toml_kv("num_ar_buckets", cfg.num_ar_buckets),
        "",
    ]
    multi_character = len(project.characters) > 1
    if multi_character:
        # One block per character. Each subdir under ``root`` carries that
        # character's frames; ``num_repeats`` is the balancing multiplier.
        # Characters with zero frames get skipped — emitting an empty
        # directory would make diffusion-pipe scan an empty path and warn.
        rows = compute_character_balancing(project=project)
        any_emitted = False
        for row in rows:
            if row.frame_count <= 0:
                continue
            char_dir = Path(root) / row.character_slug
            lines += [
                f"# Character: {row.name} ({row.character_slug}) — "
                f"{row.frame_count} frame{'s' if row.frame_count != 1 else ''}, "
                f"multiply = {row.effective_multiply}",
                "[[directory]]",
                _toml_kv("path", str(char_dir.resolve())),
                _toml_kv("num_repeats", row.effective_multiply),
                "",
            ]
            any_emitted = True
        if not any_emitted:
            # No character has frames yet — emit a single placeholder
            # block at the root so diffusion-pipe doesn't choke on an
            # empty TOML; the user will see a "no images" error which is
            # the right signal here.
            lines += [
                "[[directory]]",
                _toml_kv("path", str(Path(root).resolve())),
                _toml_kv("num_repeats", 1),
                "",
            ]
    else:
        # Single-character: keep the historical flat layout so existing
        # callers and the dataset preview UI keep working unchanged.
        lines += [
            "[[directory]]",
            _toml_kv("path", str(Path(root).resolve())),
            _toml_kv("num_repeats", 1),
            "",
        ]
    return "\n".join(lines)


# ----- training dataset staging ---------------------------------------------


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def build_dataset_staging(project: Project, dest: Path) -> dict:
    """Materialize the trainer's view of the project as a directory of
    symlinks under ``dest``.

    For each kept frame ``<stem>``:
      * the staged image ``<dest>/<stem>.<ext>`` points at
        ``<stem>_crop.png`` if a crop derivative exists, else at
        ``<stem>.<ext>``. The crop's pixels are what the model should see.
      * the staged sidecar ``<dest>/<stem>.txt`` always points at the
        original ``<stem>.txt`` — there is only ever one .txt per kept
        frame, regardless of cropping. This is what makes "edit tags on
        the original; train on the crop" work end-to-end.

    ``_crop`` images in ``kept_dir`` are skipped as standalone samples;
    they are only used as the link target for their parent. Any legacy
    ``<stem>_crop.txt`` files on disk are ignored.

    The destination is wiped and recreated on every call so a re-run picks
    up the latest crop set without dragging stale links forward.
    """
    kept = project.kept_dir
    dest = Path(dest)
    if dest.exists():
        for child in dest.iterdir():
            try:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except OSError:
                pass
    dest.mkdir(parents=True, exist_ok=True)

    paired = 0
    with_crop = 0
    missing_txt = 0
    if not kept.is_dir():
        return {
            "images": 0, "with_crop": 0, "missing_txt": 0,
            "dest": str(dest.resolve()),
        }

    # Pre-compute per-frame core-tag pruning context. We do this once per
    # staging call (an O(n_frames) scan of the metadata log) so each
    # frame's link_or_copy doesn't re-walk it. The map is empty when no
    # character has core_tags_enabled — short-circuiting back to the
    # symlink-only path so single-character no-pruning runs are
    # byte-identical to the pre-core-tags pipeline.
    prune_map = _build_prune_map(project)
    trigger_map = _build_trigger_map(project)
    pruned_count = 0
    # Per-character staging only kicks in for multi-character projects;
    # single-character projects keep the flat layout so existing callers
    # (and tests) remain byte-identical. The owner-of-frame lookup is the
    # same metadata-driven map used by core-tag pruning, just keyed by
    # filename → slug instead of filename → core_tags.
    multi_character = len(project.characters) > 1
    owner_map = _build_owner_map(project) if multi_character else {}

    for img in sorted(kept.iterdir()):
        if not img.is_file():
            continue
        if img.suffix.lower() not in _IMAGE_EXTS:
            continue
        stem = img.stem
        if stem.endswith(CROP_SUFFIX):
            continue  # handled via its parent
        crop = kept / f"{stem}{CROP_SUFFIX}.png"
        src_img = crop if crop.is_file() else img
        # Choose the per-frame staging directory: a character subdir for
        # multi-character projects (so dataset.toml can emit one
        # ``[[directory]]`` block per character with its own num_repeats),
        # the flat root for single-character projects.
        if multi_character:
            slug = owner_map.get(stem)
            if slug is None:
                # Frames whose owner can't be resolved drop into an
                # ``unsorted`` subdir so they don't pollute any one
                # character's training set.
                slug = "unsorted"
            frame_dir = dest / slug
            frame_dir.mkdir(parents=True, exist_ok=True)
        else:
            frame_dir = dest
        # Stage the image under the original stem so the trainer pairs it
        # with the original sidecar by stem-match.
        link_img = frame_dir / f"{stem}{src_img.suffix.lower()}"
        _link_or_copy(src_img, link_img)
        src_txt = kept / f"{stem}.txt"
        if src_txt.is_file():
            core_tags = prune_map.get(stem)
            trigger = trigger_map.get(stem)
            if core_tags or trigger:
                text = src_txt.read_text(encoding="utf-8")
                if core_tags:
                    from neme_anima.core_tags import prune_sidecar_text
                    text = prune_sidecar_text(text, core_tags)
                    pruned_count += 1
                if trigger:
                    text = _prepend_trigger(text, trigger)
                (frame_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
            else:
                _link_or_copy(src_txt, frame_dir / f"{stem}.txt")
        else:
            missing_txt += 1
        paired += 1
        if crop.is_file():
            with_crop += 1
    return {
        "images": paired,
        "with_crop": with_crop,
        "missing_txt": missing_txt,
        "pruned": pruned_count,
        "dest": str(dest.resolve()),
    }


def _build_owner_map(project: Project) -> dict[str, str]:
    """Return ``{filename_stem: character_slug}`` for kept frames.

    Last-write-wins per filename so a moved frame is staged under its
    new owner. Used only when staging multi-character projects.
    """
    from neme_anima.storage.metadata import MetadataLog

    log = MetadataLog(project.metadata_path)
    owner: dict[str, str] = {}
    for rec in log.iter_records():
        if rec.kept:
            owner[rec.filename] = rec.character_slug
    return owner


def _build_prune_map(project: Project) -> dict[str, list[str]]:
    """Build ``{filename: core_tags_to_drop}`` for the staging pass.

    Walks the metadata log once to determine each frame's owning
    character (last-write-wins), then materialises that character's
    core_tags list when pruning is enabled. Frames whose owners aren't
    pruning-enabled don't appear in the map — callers fall through to
    the symlink path.
    """
    from neme_anima.storage.metadata import MetadataLog

    log = MetadataLog(project.metadata_path)
    latest_owner: dict[str, str] = {}
    for rec in log.iter_records():
        if rec.kept:
            latest_owner[rec.filename] = rec.character_slug
    # Cache the per-character list lookup so an N-frame project is O(N+C)
    # instead of O(N*C).
    by_slug: dict[str, list[str]] = {}
    for c in project.characters:
        if c.core_tags_enabled and c.core_tags:
            by_slug[c.slug] = list(c.core_tags)
    out: dict[str, list[str]] = {}
    for filename, slug in latest_owner.items():
        tags = by_slug.get(slug)
        if tags:
            out[filename] = tags
    return out


def _build_trigger_map(project: Project) -> dict[str, str]:
    """Return ``{filename: trigger_token}`` for kept frames whose owning
    character has a non-empty ``trigger_token``.

    Mirrors :func:`_build_prune_map`'s walk: last-write-wins per filename
    so a moved frame is staged under its new owner's trigger. Frames whose
    owners have an empty trigger are absent from the map — callers fall
    through to the symlink path (no rewrite needed).
    """
    from neme_anima.storage.metadata import MetadataLog

    log = MetadataLog(project.metadata_path)
    latest_owner: dict[str, str] = {}
    for rec in log.iter_records():
        if rec.kept:
            latest_owner[rec.filename] = rec.character_slug
    triggers_by_slug = {
        c.slug: c.trigger_token.strip()
        for c in project.characters
        if c.trigger_token.strip()
    }
    out: dict[str, str] = {}
    for filename, slug in latest_owner.items():
        t = triggers_by_slug.get(slug)
        if t:
            out[filename] = t
    return out


def _prepend_trigger(sidecar_text: str, trigger: str) -> str:
    """Return a new sidecar with ``trigger, `` prepended to line 1.

    Line 2 (LLM description) is untouched. Pass through ``split_sidecar``
    / ``join_sidecar`` so the staged file's trailing-newline + CRLF + tag
    dedup behaviour matches the rest of the pipeline (mirrors
    ``core_tags.prune_sidecar_text``).
    """
    from neme_anima.tag import join_sidecar, split_sidecar

    danbooru, description = split_sidecar(sidecar_text)
    triggered = f"{trigger}, {danbooru}" if danbooru else trigger
    return join_sidecar(triggered, description)


def _link_or_copy(src: Path, dest: Path) -> None:
    """Symlink ``dest`` → ``src`` (absolute target); fall back to a copy
    on platforms or filesystems that reject symlinks (e.g. Windows without
    developer mode)."""
    src_abs = src.resolve()
    try:
        os.symlink(src_abs, dest)
    except OSError:
        shutil.copy2(src_abs, dest)


def render_run_toml(
    project: Project,
    *,
    run_dir: Path,
    dataset_toml_path: Path,
    resume_from_checkpoint: str | None = None,
) -> str:
    """Build the run TOML for ``train.py --config``.

    ``resume_from_checkpoint`` is the *name* (not full path) of a checkpoint
    directory under ``run_dir`` — diffusion-pipe resolves resume paths
    relative to ``output_dir``.
    """
    cfg = project.training
    lines = [
        "# Auto-generated by neme-anima — do not hand-edit.",
        f"# Project: {project.slug}",
        f"# Generated at {datetime.now(timezone.utc).isoformat()}",
        "",
        _toml_kv("output_dir", str(run_dir.resolve())),
        _toml_kv("dataset", str(dataset_toml_path.resolve())),
    ]
    # ``resume_from_checkpoint`` MUST be a top-level key in the TOML — if
    # we let it fall after a ``[section]`` header, train.py would read it
    # off the wrong table (and DeepSpeed forwarded it as a kwarg to AdamW,
    # blowing up with TypeError).
    if resume_from_checkpoint:
        lines.append(
            _toml_kv("resume_from_checkpoint", resume_from_checkpoint),
        )
    # ``blocks_to_swap`` is a top-level diffusion-pipe option — like
    # ``resume_from_checkpoint`` it must not slip under a [section] header
    # or train.py won't see it.
    if cfg.blocks_to_swap > 0:
        lines.append(_toml_kv("blocks_to_swap", cfg.blocks_to_swap))
    lines += [
        "",
        # Training settings
        _toml_kv("epochs", cfg.epochs),
        _toml_kv("micro_batch_size_per_gpu", cfg.micro_batch_size),
        _toml_kv("pipeline_stages", 1),
        _toml_kv("gradient_accumulation_steps", cfg.gradient_accumulation_steps),
        _toml_kv("gradient_clipping", cfg.gradient_clipping),
        _toml_kv("warmup_steps", cfg.warmup_steps),
        "",
        # Eval
        _toml_kv("eval_every_n_epochs", cfg.eval_every_n_epochs),
        _toml_kv("eval_before_first_step", True),
        _toml_kv("eval_micro_batch_size_per_gpu", 1),
        _toml_kv("eval_gradient_accumulation_steps", 1),
        "",
        # Checkpointing
        _toml_kv("save_every_n_epochs", cfg.save_every_n_epochs),
        # ``activation_checkpointing`` accepts either ``true`` (PyTorch
        # native checkpoint, the recipe default) or the string ``"unsloth"``
        # (unsloth's variant, more aggressive memory savings — used by the
        # canonical wan_14b_min_vram example). diffusion-pipe's train.py
        # branches on ``== True`` vs ``== 'unsloth'``, so we must emit a
        # bare boolean for the default case, not the string ``"true"``.
        _toml_kv(
            "activation_checkpointing",
            "unsloth" if cfg.activation_checkpointing_mode == "unsloth" else True,
        ),
        _toml_kv("partition_method", "parameters"),
        _toml_kv("save_dtype", "bfloat16"),
        _toml_kv("caching_batch_size", 1),
        _toml_kv("map_num_proc", 8),
        _toml_kv("steps_per_print", 1),
        # torch.compile is incompatible with the low-VRAM levers: Inductor
        # fails to lower fused kernels that touch fp8 weights (Triton MLIR
        # PassManager error on embedding+RMSNorm fusions), and block_swap
        # moves params between CPU/GPU mid-step, breaking compile's
        # static-parameter assumption. The canonical wan_14b_min_vram.toml
        # / qwen_image_24gb_vram.toml examples shipped with diffusion-pipe
        # omit ``compile`` for the same reason.
        *(
            [_toml_kv("compile", True)]
            if cfg.transformer_dtype == "bfloat16" and cfg.blocks_to_swap == 0
            else []
        ),
        "",
        "[model]",
        _toml_kv("type", "anima"),
        _toml_kv("transformer_path", str(Path(cfg.dit_path).expanduser().resolve())),
        _toml_kv("vae_path", str(Path(cfg.vae_path).expanduser().resolve())),
        _toml_kv("llm_path", str(Path(cfg.llm_path).expanduser().resolve())),
        # ``dtype`` is the base storage/compute dtype — must be a real fp
        # type. The VAE init does ``1.0 / std`` on this dtype, which fails
        # on float8 (NotImplementedError "reciprocal_cpu" for Float8_e4m3fn).
        # Keep it pinned to bfloat16; the optional fp8 quantization layer
        # is a separate ``transformer_dtype`` key applied on top.
        _toml_kv("dtype", "bfloat16"),
        # Only emit transformer_dtype when it differs from the base — that
        # keeps default-recipe TOMLs unchanged and matches diffusion-pipe's
        # own ``model_config.get('transformer_dtype', dtype)`` fallback.
        *(
            [_toml_kv("transformer_dtype", cfg.transformer_dtype)]
            if cfg.transformer_dtype and cfg.transformer_dtype != "bfloat16"
            else []
        ),
        _toml_kv("llm_adapter_lr", cfg.llm_adapter_lr),
        _toml_kv("sigmoid_scale", cfg.sigmoid_scale),
        "",
        "[adapter]",
        _toml_kv("type", "lora"),
        _toml_kv("rank", cfg.rank),
        _toml_kv("dtype", "bfloat16"),
        "",
        "[optimizer]",
        # ``optimizer_type`` is forwarded as the diffusion-pipe ``type`` key.
        # train.py lowercases it before matching, so casing is free; we
        # preserve user casing for readability ("AdamW8bitKahan" vs
        # "adamw_optimi"). Optimizer kwargs (lr / betas / weight_decay /
        # eps) flow through to all supported types unchanged — see
        # train.py's ``kwargs = {k: v for k, v in optim_config.items()
        # if k not in ['type', 'gradient_release']}``.
        _toml_kv("type", cfg.optimizer_type),
        _toml_kv("lr", cfg.learning_rate),
        _toml_kv("betas", cfg.optimizer_betas),
        _toml_kv("weight_decay", cfg.weight_decay),
        _toml_kv("eps", cfg.eps),
        "",
    ]
    return "\n".join(lines)


# ----- caption preview -------------------------------------------------------


def _read_sidecar(image_path: Path) -> tuple[str, str]:
    """Return (tag_line, nl_line) from the ``.txt`` sidecar of an image.

    The pipeline writes one or two lines: line 1 = comma-separated tags,
    line 2 (optional) = LLM natural-language description. Missing file =
    both empty.
    """
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.is_file():
        return ("", "")
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError:
        return ("", "")
    parts = text.splitlines()
    tags = parts[0].strip() if len(parts) >= 1 else ""
    nl = parts[1].strip() if len(parts) >= 2 else ""
    return (tags, nl)


def render_training_caption(
    *, tags: str, nl: str, config: TrainingConfig,
) -> str:
    """Render the caption that the trainer will see for one image.

    This does NOT mutate the on-disk sidecars — the trainer reads them as-is.
    The function is for *previewing* what the chosen ``caption_mode`` and
    ``trigger_token`` will produce so the user can sanity-check before
    starting a run.
    """
    trigger = (config.trigger_token or "").strip()
    tags = tags.strip()
    nl = nl.strip()
    if config.caption_mode == "tags":
        body = tags
    elif config.caption_mode == "nl":
        body = nl
    else:  # "mixed"
        if tags and nl:
            body = f"{tags}. {nl}"
        else:
            body = tags or nl
    if trigger:
        return f"{trigger}, {body}".rstrip(", ")
    return body


def dataset_preview(project: Project, *, sample_n: int = 5) -> dict:
    """Summarize the project's training dataset for the UI.

    The ``samples`` block renders each frame's caption with that frame's
    owning character's ``trigger_token`` so multi-character projects show
    per-character previews. Project-level fields (``caption_mode``) come
    from the project-wide config (still ``project.training``).
    """
    cfg = project.training
    kept = project.kept_dir
    if not kept.is_dir():
        return {
            "total_images": 0, "with_tags": 0, "with_descriptions": 0,
            "samples": [], "kept_dir": str(kept),
        }
    # `_crop` derivatives are internal training-target replacements, not
    # standalone samples — counting them would double the visible total
    # whenever the user has cropped a frame.
    images = sorted(
        p for p in kept.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        and not p.stem.endswith(CROP_SUFFIX)
    )
    total = len(images)
    with_tags = 0
    with_nl = 0

    # Resolve each sampled frame to its owning character's trigger. Built
    # once per call (O(metadata_rows + characters)).
    owner_map = _build_owner_map(project)
    triggers_by_slug = {
        c.slug: c.trigger_token for c in project.characters
    }
    fallback_trigger = (
        project.characters[0].trigger_token if project.characters else ""
    )

    samples: list[dict] = []
    for i, img in enumerate(images):
        tags, nl = _read_sidecar(img)
        if tags:
            with_tags += 1
        if nl:
            with_nl += 1
        if i < sample_n:
            slug = owner_map.get(img.stem)
            trigger = (
                triggers_by_slug.get(slug, fallback_trigger)
                if slug is not None else fallback_trigger
            )
            sample_cfg = TrainingConfig(
                caption_mode=cfg.caption_mode,
                trigger_token=trigger,
            )
            samples.append({
                "filename": img.name,
                "tags": tags,
                "nl": nl,
                "rendered": render_training_caption(
                    tags=tags, nl=nl, config=sample_cfg,
                ),
            })
    return {
        "total_images": total,
        "with_tags": with_tags,
        "with_descriptions": with_nl,
        "samples": samples,
        "kept_dir": str(kept),
    }


# ----- checkpoint discovery + retention --------------------------------------

# diffusion-pipe writes per-epoch directories named ``epoch{N}`` under the
# run's output_dir; some forks also use ``global_step{N}``. Match either.
_CKPT_PATTERNS = (
    re.compile(r"^epoch(\d+)$"),
    re.compile(r"^global_step(\d+)$"),
    re.compile(r"^step(\d+)$"),
)


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _parse_checkpoint_name(name: str) -> tuple[int | None, int | None]:
    """Return (epoch, step) parsed from a checkpoint directory name.

    Either component may be None depending on the trainer's naming scheme.
    """
    for pat in _CKPT_PATTERNS:
        m = pat.match(name)
        if m:
            n = int(m.group(1))
            if pat.pattern.startswith("^epoch"):
                return (n, None)
            return (None, n)
    return (None, None)


def discover_checkpoints(run_dir: Path) -> list[CheckpointInfo]:
    """List every checkpoint directory under ``run_dir``, oldest first.

    diffusion-pipe creates a per-launch timestamped subdirectory inside
    ``run_dir`` and writes its ``epoch{N}`` / ``global_step{N}`` checkpoints
    *there*, not directly in ``run_dir``. We therefore walk both levels: a
    checkpoint can live as a direct child of ``run_dir`` (legacy / unusual
    layouts) or one level deeper (the common case).

    Sort key: epoch when present, else step, else mtime. This is the order
    the retention pruner uses to decide what to delete.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    out: list[CheckpointInfo] = []

    def _record(entry: Path, rel_subdir: str) -> None:
        epoch, step = _parse_checkpoint_name(entry.name)
        if epoch is None and step is None:
            return
        try:
            stat = entry.stat()
        except OSError:
            return
        out.append(CheckpointInfo(
            name=entry.name,
            path=str(entry.resolve()),
            epoch=epoch,
            step=step,
            size_bytes=_dir_size(entry),
            modified_at=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc,
            ).isoformat(),
            subdir=rel_subdir,
        ))

    # Skip directories we know aren't checkpoints (the staged training
    # dataset, and the sample-image tree; we'd just waste a sysstat scan
    # walking them).
    SKIP = {"dataset", SAMPLES_DIRNAME}
    for entry in run_dir.iterdir():
        if not entry.is_dir() or entry.name in SKIP:
            continue
        if _parse_checkpoint_name(entry.name) != (None, None):
            _record(entry, "")
            continue
        # Treat anything else as a candidate diffusion-pipe sub-run-dir.
        try:
            for inner in entry.iterdir():
                if inner.is_dir():
                    _record(inner, entry.name)
        except OSError:
            continue

    def _key(c: CheckpointInfo) -> tuple[int, int, str]:
        return (
            c.epoch if c.epoch is not None else -1,
            c.step if c.step is not None else -1,
            c.modified_at,
        )

    out.sort(key=_key)
    return out


def find_resumable_subdir(run_dir: Path) -> str | None:
    """Return the diffusion-pipe sub-run-dir name we can resume from.

    train.py reads ``<output_dir>/<resume_from_checkpoint>/latest`` to find
    its DeepSpeed state, so the value to pass back to ``train.py`` is *not*
    a checkpoint name but the timestamped subdirectory created on a prior
    launch. We pick the most recent subdir that has a ``latest`` file.
    Returns ``None`` if nothing resumable exists.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    candidates: list[tuple[float, str]] = []
    for entry in run_dir.iterdir():
        if not entry.is_dir() or entry.name == "dataset":
            continue
        latest = entry / "latest"
        if not latest.is_file():
            continue
        try:
            mtime = latest.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, entry.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def latest_checkpoint(run_dir: Path) -> CheckpointInfo | None:
    """Most recent checkpoint, or None if there are none."""
    cps = discover_checkpoints(run_dir)
    return cps[-1] if cps else None


# ----- safetensors provenance tagging ----------------------------------------


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _nearest_file(candidates: Iterable[Path]) -> Path | None:
    return next((p for p in candidates if p.is_file()), None)


def _checkpoint_weights_path(checkpoint_dir: Path) -> Path | None:
    preferred = checkpoint_dir / "adapter_model.safetensors"
    if preferred.is_file():
        return preferred
    return next(iter(sorted(checkpoint_dir.glob("*.safetensors"))), None)


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                raise SafetensorsMetadataError(
                    f"{path} is too short to contain a safetensors header",
                )
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len <= 0 or header_len > size - 8:
                raise SafetensorsMetadataError(
                    f"{path} has an invalid safetensors header length",
                )
            raw_header = f.read(header_len)
    except OSError as e:
        raise SafetensorsMetadataError(f"could not read {path}: {e}") from e

    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as e:
        raise SafetensorsMetadataError(
            f"{path} does not contain a valid JSON safetensors header",
        ) from e
    if not isinstance(header, dict):
        raise SafetensorsMetadataError(
            f"{path} safetensors header must be a JSON object",
        )
    return header, header_len


def _rewrite_safetensors_header(path: Path, header: dict, old_header_len: int) -> None:
    try:
        stat = path.stat()
    except OSError as e:
        raise SafetensorsMetadataError(f"could not stat {path}: {e}") from e

    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    tmp_name: str | None = None
    try:
        with open(path, "rb") as src, tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as tmp:
            tmp_name = tmp.name
            src.seek(8 + old_header_len)
            tmp.write(struct.pack("<Q", len(header_bytes)))
            tmp.write(header_bytes)
            shutil.copyfileobj(src, tmp, length=1024 * 1024)
        os.chmod(tmp_name, stat.st_mode)
        os.replace(tmp_name, path)
    except OSError as e:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise SafetensorsMetadataError(f"could not rewrite {path}: {e}") from e


def tag_safetensors_metadata(path: Path, metadata: dict[str, str]) -> bool:
    """Merge string metadata into a safetensors file without loading tensors.

    Returns True when the file was rewritten. Tensor byte ranges stay unchanged
    because safetensors ``data_offsets`` are relative to the post-header buffer,
    not absolute file offsets.
    """
    path = Path(path)
    header, old_header_len = _read_safetensors_header(path)
    raw_existing = header.get("__metadata__", {})
    if raw_existing is None:
        raw_existing = {}
    if not isinstance(raw_existing, dict):
        raise SafetensorsMetadataError(
            f"{path} safetensors __metadata__ must be a JSON object",
        )
    existing = {str(k): str(v) for k, v in raw_existing.items()}
    additions = {str(k): str(v) for k, v in metadata.items()}
    additions.setdefault(
        "neme_anima_tagged_at",
        datetime.now(timezone.utc).isoformat(),
    )
    merged = {**existing, **additions}
    if existing == merged:
        return False
    header["__metadata__"] = merged
    _rewrite_safetensors_header(path, header, old_header_len)
    return True


def build_safetensors_provenance_metadata(
    project: Project,
    *,
    run_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, str]:
    """Metadata embedded in exported/finished LoRA files for provenance."""
    from neme_anima import __version__

    run_dir = Path(run_dir)
    checkpoint_dir = Path(checkpoint_dir)
    run_toml = _nearest_file((
        checkpoint_dir / "run.toml",
        checkpoint_dir.parent / "run.toml",
        run_dir / "run.toml",
    ))
    dataset_toml = _nearest_file((
        checkpoint_dir / "dataset.toml",
        checkpoint_dir.parent / "dataset.toml",
        run_dir / "dataset.toml",
    ))
    cfg = project.training
    out = {
        "neme_anima": "true",
        "neme_anima_schema": "1",
        "neme_anima_generated_by": "neme-anima",
        "neme_anima_version": __version__,
        "neme_anima_project_slug": project.slug,
        "neme_anima_project_name": project.name,
        "neme_anima_run_name": run_dir.name,
        "neme_anima_checkpoint": checkpoint_dir.name,
        "neme_anima_trainer": "diffusion-pipe",
        "neme_anima_model_type": "anima",
        "neme_anima_adapter_type": "lora",
        "neme_anima_rank": str(cfg.rank),
        "neme_anima_caption_mode": cfg.caption_mode,
    }
    trigger = (cfg.trigger_token or "").strip()
    if trigger:
        out["neme_anima_trigger_token"] = trigger
    if run_toml is not None:
        out["neme_anima_run_toml_sha256"] = _file_sha256(run_toml) or ""
    if dataset_toml is not None:
        out["neme_anima_dataset_toml_sha256"] = _file_sha256(dataset_toml) or ""
    return out


def tag_checkpoint_safetensors(
    project: Project,
    *,
    run_dir: Path,
    checkpoint_dir: Path,
) -> Path | None:
    """Tag one checkpoint's LoRA safetensors and return its path if changed."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = _checkpoint_weights_path(checkpoint_dir)
    if weights is None:
        return None
    changed = tag_safetensors_metadata(
        weights,
        build_safetensors_provenance_metadata(
            project, run_dir=run_dir, checkpoint_dir=checkpoint_dir,
        ),
    )
    return weights if changed else None


def tag_run_safetensors(project: Project, run_dir: Path) -> list[str]:
    """Tag every discovered LoRA checkpoint in ``run_dir``."""
    changed: list[str] = []
    for cp in discover_checkpoints(run_dir):
        path = tag_checkpoint_safetensors(
            project, run_dir=Path(run_dir), checkpoint_dir=Path(cp.path),
        )
        if path is not None:
            changed.append(str(path.resolve()))
    return changed


def prune_checkpoints(
    run_dir: Path, *, keep_last_n: int,
) -> list[str]:
    """Trim a finished run's checkpoints, treating the two artifact kinds
    separately.

    diffusion-pipe leaves behind two very different things (see
    ``discover_checkpoints``): ``epoch{N}`` dirs holding the LoRA
    ``adapter_model.safetensors`` (the *deliverables* — you keep several to
    compare and pick the best epoch) and ``global_step{N}`` dirs holding the
    heavy DeepSpeed optimizer + model state (the *resume* plumbing — ~4x the
    size of an adapter, and only the latest is ever resumed from).

    - **Resume state** is always trimmed to the single latest checkpoint
      (highest step). This happens regardless of ``keep_last_n`` — once a run
      is over there is no value in older resume snapshots.
    - **Epoch deliverables** honour ``keep_last_n``: keep the most recent N,
      delete the rest. ``keep_last_n == 0`` keeps every epoch (the
      user-requested default).

    ``discover_checkpoints`` returns the list already sorted ascending (epoch,
    then step), so the latest of each kind is simply the last element. Returns
    the deleted directory names so the caller can log / surface them.
    """
    cps = discover_checkpoints(run_dir)
    # epoch is None == resume plumbing (global_step / step); else a deliverable.
    deliverables = [c for c in cps if c.epoch is not None]
    resume = [c for c in cps if c.epoch is None]

    to_delete: list[CheckpointInfo] = []
    if len(resume) > 1:
        to_delete.extend(resume[:-1])
    if keep_last_n > 0 and len(deliverables) > keep_last_n:
        to_delete.extend(deliverables[: len(deliverables) - keep_last_n])

    deleted: list[str] = []
    for cp in to_delete:
        target = Path(cp.path)
        try:
            _rmtree(target)
            deleted.append(cp.name)
        except OSError:
            # Best-effort — log via the runner; do not crash the run.
            pass
    return deleted


def _rmtree(path: Path) -> None:
    """Tolerant rmtree — ignore individual file errors so a partially-locked
    checkpoint dir doesn't take down the whole prune."""
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    for child in path.iterdir():
        _rmtree(child)
    try:
        path.rmdir()
    except OSError:
        pass


# ----- sample generation -----------------------------------------------------

# Curated ComfyUI KSampler names offered in the UI. All run with the ComfyUI
# vendored in diffusion-pipe's venv as-is; the SDE samplers are left out
# because they need torchsde, which that venv lacks (see _anima_sampler.py).
SAMPLE_SAMPLERS = (
    "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms",
    "dpmpp_2s_ancestral", "dpmpp_2m", "ddim", "uni_pc", "res_multistep",
)
SAMPLE_SCHEDULERS = (
    "simple", "normal", "karras", "exponential", "sgm_uniform", "beta",
    "ddim_uniform",
)

# Sample images live under ``<run_dir>/samples/epoch{N:04d}/``.
SAMPLES_DIRNAME = "samples"
_SAMPLE_DIR_RE = re.compile(r"^epoch(\d+)$")


def sample_prompts(config: TrainingConfig) -> list[str]:
    """The non-blank sample prompts, stripped."""
    return [p.strip() for p in config.sample_prompts if p.strip()]


def sampling_enabled(config: TrainingConfig) -> bool:
    return bool(sample_prompts(config))


def diffusion_pipe_python(diffusion_pipe_dir: str) -> Path | None:
    """diffusion-pipe's venv interpreter (``.venv`` preferred), if present."""
    if not diffusion_pipe_dir:
        return None
    dp = Path(diffusion_pipe_dir).expanduser()
    for candidate in (dp / ".venv" / "bin" / "python", dp / "venv" / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def sampler_script_path() -> Path:
    """The standalone sampler run inside diffusion-pipe's venv."""
    return Path(__file__).parent / "_anima_sampler.py"


def _sample_problems(config: TrainingConfig) -> list[str]:
    """Sample-generation settings that would block a run. None while
    sampling is off, so a stale sampler setting never blocks training."""
    if not sampling_enabled(config):
        return []
    problems: list[str] = []
    every, save_every = config.sample_every_n_epochs, config.save_every_n_epochs
    if every <= 0:
        problems.append("sample_every_n_epochs must be > 0")
    elif save_every > 0 and every % save_every != 0:
        problems.append(
            f"sample_every_n_epochs ({every}) must be a multiple of "
            f"save_every_n_epochs ({save_every}) — samples are rendered from "
            "saved epoch LoRAs",
        )
    if config.sample_steps <= 0:
        problems.append("sample_steps must be > 0")
    if config.sample_cfg < 0:
        problems.append("sample_cfg must be >= 0")
    for label, value in (("sample_width", config.sample_width),
                         ("sample_height", config.sample_height)):
        if value % 16 or not 256 <= value <= 2048:
            problems.append(f"{label} must be a multiple of 16 between 256 and 2048")
    if config.sample_sampler not in SAMPLE_SAMPLERS:
        problems.append(f"unknown sample_sampler {config.sample_sampler!r}")
    if config.sample_scheduler not in SAMPLE_SCHEDULERS:
        problems.append(f"unknown sample_scheduler {config.sample_scheduler!r}")
    if config.diffusion_pipe_dir:
        dp = Path(config.diffusion_pipe_dir).expanduser()
        if diffusion_pipe_python(config.diffusion_pipe_dir) is None:
            problems.append(
                f"sample generation needs diffusion-pipe's venv python at "
                f"{dp / '.venv' / 'bin' / 'python'}",
            )
        if not (dp / "submodules" / "ComfyUI" / "comfy").is_dir():
            problems.append(
                f"sample generation needs diffusion-pipe's vendored ComfyUI at "
                f"{dp / 'submodules' / 'ComfyUI'} (git submodule update --init)",
            )
    return problems


def sample_epoch_qualifies(epoch: int, config: TrainingConfig) -> bool:
    """Every ``sample_every_n_epochs``-th epoch, plus the final one
    (diffusion-pipe always saves it)."""
    every = config.sample_every_n_epochs
    return epoch == config.epochs or (every > 0 and epoch % every == 0)


def sample_dir(run_dir: Path, epoch: int) -> Path:
    return Path(run_dir) / SAMPLES_DIRNAME / f"epoch{epoch:04d}"


def sample_epochs_ready(run_dir: Path, config: TrainingConfig) -> list[tuple[int, Path]]:
    """``(epoch, adapter path)`` for every saved epoch LoRA that should be
    sampled and hasn't been yet, ascending.

    A checkpoint counts as complete once ``run.toml`` sits next to the
    adapter: diffusion-pipe copies it in only after the safetensors is fully
    written (``utils/saver.py``). An existing ``manifest.json`` (written last
    by the sampler, success or failure) marks an epoch done, so resumed runs
    skip what was already sampled. When a resume leaves the same epoch in two
    sub-run dirs, the newer one wins (``discover_checkpoints`` sorts by mtime
    within an epoch).
    """
    ready: dict[int, Path] = {}
    for cp in discover_checkpoints(run_dir):
        if cp.epoch is None or not sample_epoch_qualifies(cp.epoch, config):
            continue
        ckpt = Path(cp.path)
        # A newer duplicate supersedes an older one even while it's still
        # being written — don't sample the stale LoRA in the meantime.
        ready.pop(cp.epoch, None)
        adapter = ckpt / "adapter_model.safetensors"
        if not (adapter.is_file() and (ckpt / "run.toml").is_file()):
            continue
        if (sample_dir(run_dir, cp.epoch) / "manifest.json").is_file():
            continue
        ready[cp.epoch] = adapter
    return sorted(ready.items())


def build_sample_job(
    config: TrainingConfig, *, lora_path: Path, epoch: int, output_dir: Path,
) -> dict:
    """The JSON job ``_anima_sampler.py --job`` consumes."""
    def _p(raw: str) -> str:
        # Keep an empty path empty so the sampler fails loudly instead of
        # loading from the resolved cwd.
        return str(Path(raw).expanduser().resolve()) if raw else ""

    return {
        "diffusion_pipe_dir": _p(config.diffusion_pipe_dir),
        "dit_path": _p(config.dit_path),
        "vae_path": _p(config.vae_path),
        "llm_path": _p(config.llm_path),
        "lora_path": str(Path(lora_path).resolve()),
        "epoch": epoch,
        "checkpoint": Path(lora_path).parent.name,
        "prompts": sample_prompts(config),
        "negative_prompt": config.sample_negative_prompt.strip(),
        "settings": {
            "steps": config.sample_steps,
            "sampler": config.sample_sampler,
            "scheduler": config.sample_scheduler,
            "cfg": config.sample_cfg,
            "width": config.sample_width,
            "height": config.sample_height,
            "seed": config.sample_seed,
        },
        "output_dir": str(Path(output_dir).resolve()),
    }


def read_sample_manifests(run_dir: Path) -> list[dict]:
    """Every finished sample epoch of a run (those with a manifest), by
    epoch. Images whose file is missing are dropped; ``mtime`` rides along
    so the API can cache-bust URLs."""
    root = Path(run_dir) / SAMPLES_DIRNAME
    if not root.is_dir():
        return []
    out: list[dict] = []
    for entry in root.iterdir():
        m = _SAMPLE_DIR_RE.match(entry.name)
        manifest = entry / "manifest.json"
        if not m or not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        images = []
        for img in data.get("images") or []:
            if not isinstance(img, dict):
                continue
            f = entry / str(img.get("file", ""))
            if img.get("file") and f.is_file():
                images.append({
                    "file": img["file"],
                    "prompt": img.get("prompt", ""),
                    "mtime": int(f.stat().st_mtime),
                })
        out.append({
            "epoch": int(m.group(1)),
            "dir": entry.name,
            "prompts": data.get("prompts") or [],
            "negative_prompt": data.get("negative_prompt") or "",
            "settings": data.get("settings") or {},
            "images": images,
            "error": data.get("error"),
        })
    out.sort(key=lambda e: e["epoch"])
    return out


# ----- run-folder helpers ----------------------------------------------------


def new_run_dir(project: Project, *, label: str | None = None) -> Path:
    """Mint a fresh directory under ``training/runs/`` for a new run.

    Naming: ``YYYYMMDD-HHMMSS[-label]``. The directory is created on disk;
    ``train.py`` is allowed to populate it with epoch checkpoints, eval
    samples, etc.
    """
    runs = project.training_runs_dir
    runs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{label}" if label else ts
    candidate = runs / name
    n = 2
    while candidate.exists():
        candidate = runs / f"{name}-{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def list_runs(project: Project) -> list[dict]:
    """Return all runs found on disk, newest first, with checkpoint counts."""
    runs = project.training_runs_dir
    if not runs.is_dir():
        return []
    out: list[dict] = []
    for entry in sorted(runs.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        cps = discover_checkpoints(entry)
        # Split LoRA outputs from DeepSpeed plumbing — the UI shows the
        # former as artifacts, while the latter only matters for resume.
        epoch_cps = [c for c in cps if c.epoch is not None]
        latest_epoch = max(
            (c.epoch for c in epoch_cps if c.epoch is not None),
            default=None,
        )
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            mtime = 0.0
        out.append({
            "name": entry.name,
            "path": str(entry.resolve()),
            "checkpoints": len(epoch_cps),
            "latest_checkpoint": epoch_cps[-1].name if epoch_cps else None,
            "latest_epoch": latest_epoch,
            "resumable_subdir": find_resumable_subdir(entry),
            "total_size_bytes": sum(c.size_bytes for c in cps),
            "modified_at": datetime.fromtimestamp(
                mtime, tz=timezone.utc,
            ).isoformat() if mtime else "",
        })
    return out


# ----- launcher command ------------------------------------------------------


DEFAULT_LAUNCHER_TEMPLATE = (
    "deepspeed --num_gpus=1 train.py --deepspeed --config {config}"
)


def resolve_launcher_binary(config: TrainingConfig) -> tuple[str, str | None]:
    """Find the absolute path of the launcher's first token.

    Returns ``(token, resolved_path_or_None)``. We look in the system PATH
    first, then in ``<diffusion_pipe_dir>/.venv/bin`` and ``venv/bin`` so a
    user who configured the diffusion-pipe directory and built its venv
    there doesn't also have to set ``launcher_override`` — the natural
    install layout just works.
    """
    template = (config.launcher_override or "").strip() or DEFAULT_LAUNCHER_TEMPLATE
    token = next((tok for tok in template.split() if tok), "")
    if not token:
        return ("", None)
    found = shutil.which(token)
    if found is None and config.diffusion_pipe_dir:
        dp = Path(config.diffusion_pipe_dir).expanduser()
        for venv_bin in (dp / ".venv" / "bin", dp / "venv" / "bin"):
            cand = venv_bin / token
            if cand.is_file() and os.access(cand, os.X_OK):
                found = str(cand)
                break
    return (token, found)


def _shim_path() -> Path:
    """Absolute path to the diffusion-pipe pre-train shim shipped with this
    package. The shim patches diffusion-pipe's offloader to use blocking
    GPU↔CPU transfers — the workaround for WSL2's pinned-memory ceiling.
    See the shim file's docstring for the full root-cause analysis."""
    return Path(__file__).parent / "_diffusion_pipe_shim.py"


def build_launcher_argv(config: TrainingConfig, *, run_toml: Path) -> list[str]:
    """Build the argv list to launch the trainer.

    The user can override the template via ``launcher_override``. ``{config}``
    is substituted with the run TOML path; if the template lacks the marker,
    we append it as a final positional arg so a bare ``python train.py``
    style override still works.

    The launcher binary's first token is resolved to an absolute path when
    possible so the spawned subprocess finds it regardless of the parent
    process's PATH (e.g. the server is started without the diffusion-pipe
    venv activated).

    When ``blocks_to_swap > 0``, the trainer script (``train.py``) is
    transparently replaced with our pre-train shim. The shim monkey-patches
    diffusion-pipe's offloader to use blocking GPU↔CPU transfers (which
    don't need pinned host memory) before exec'ing train.py — fixing the
    misleading "CUDA out of memory" error that hits WSL2 users with plenty
    of free VRAM. The substitution is keyed on the bare token "train.py"
    so user-supplied launcher_override templates that name train.py
    explicitly are still upgraded; templates that point at a custom
    trainer script are left alone (the user knows what they're doing).
    """
    template = (config.launcher_override or "").strip() or DEFAULT_LAUNCHER_TEMPLATE
    sub = template.replace("{config}", str(run_toml.resolve()))
    if "{config}" not in template and str(run_toml.resolve()) not in sub:
        sub = f"{sub} {str(run_toml.resolve())}"
    # Naive shell split — good enough for our launcher templates which are
    # space-separated tokens with no embedded quoting.
    argv = [tok for tok in sub.split() if tok]
    if argv:
        _, resolved = resolve_launcher_binary(config)
        if resolved:
            argv[0] = resolved
        # Swap train.py for the shim when block-swap is on. The shim is
        # invoked by deepspeed exactly like train.py; argv[1:] (--deepspeed
        # --config <toml>) is forwarded unchanged because the shim exec's
        # train.py with the same sys.argv.
        if config.blocks_to_swap > 0:
            shim = str(_shim_path().resolve())
            argv = [shim if tok == "train.py" else tok for tok in argv]
    return argv

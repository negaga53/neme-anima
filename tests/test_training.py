"""Pure-function tests for the training module — TOML rendering, path
validation, and checkpoint pruning. The subprocess machinery in
``server.training_runner`` is not exercised here (it requires a real
diffusion-pipe install + GPU).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from neme_anima import training
from neme_anima.storage.project import Project, TrainingConfig


def _write_tiny_safetensors(path: Path, metadata: dict[str, str] | None = None) -> None:
    header = {
        "__metadata__": metadata or {"format": "pt"},
        "weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
    }
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"DATA")


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header["__metadata__"]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "p1", name="p1")


# ----- path validation ------------------------------------------------------


def test_check_path_empty():
    c = training.check_path("")
    assert not c.exists
    assert c.error and "empty" in c.error.lower()


def test_check_path_missing(tmp_path: Path):
    c = training.check_path(str(tmp_path / "nope.bin"), expect="file")
    assert not c.exists
    assert c.error and "no such" in c.error.lower()


def test_check_path_file_vs_dir(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    assert training.check_path(str(f), expect="file").error is None
    # Asking for a dir on a file must report an error.
    c = training.check_path(str(f), expect="dir")
    assert c.exists and c.error and "directory" in c.error.lower()


def test_validate_for_run_complains_about_missing_paths(project: Project):
    problems = training.validate_for_run(project.training)
    assert any("diffusion-pipe" in p for p in problems)
    assert any("DiT" in p or "transformer" in p for p in problems)
    assert any("VAE" in p for p in problems)
    assert any("text encoder" in p for p in problems)


def test_validate_for_run_passes_when_paths_resolve(
    project: Project, tmp_path: Path,
):
    # Build a fake diffusion-pipe install with a train.py + dummy weight files.
    dp_dir = tmp_path / "diffusion-pipe"
    dp_dir.mkdir()
    (dp_dir / "train.py").write_text("# fake")
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"")
    llm = tmp_path / "llm.safetensors"
    llm.write_bytes(b"")

    cfg = project.training
    cfg.diffusion_pipe_dir = str(dp_dir)
    cfg.dit_path = str(dit)
    cfg.vae_path = str(vae)
    cfg.llm_path = str(llm)
    # Default launcher uses `deepspeed`, which isn't necessarily on the test
    # host's PATH. Override with a binary we know exists.
    cfg.launcher_override = "/bin/sh -c true {config}"
    assert training.validate_for_run(cfg) == []


@pytest.mark.parametrize("venv_name", [".venv", "venv"])
def test_validate_for_run_rejects_diffusion_pipe_python_before_312(
    project: Project, tmp_path: Path, venv_name: str,
):
    dp_dir = tmp_path / "diffusion-pipe"
    dp_dir.mkdir()
    (dp_dir / "train.py").write_text("# fake")
    venv_bin = dp_dir / venv_name / "bin"
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / "python"
    fake_python.write_text("#!/usr/bin/env sh\nprintf '3.11\\n'\n")
    fake_python.chmod(0o755)
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"")
    llm = tmp_path / "llm.safetensors"
    llm.write_bytes(b"")

    cfg = project.training
    cfg.diffusion_pipe_dir = str(dp_dir)
    cfg.dit_path = str(dit)
    cfg.vae_path = str(vae)
    cfg.llm_path = str(llm)
    cfg.launcher_override = "/bin/sh -c true {config}"

    problems = training.validate_for_run(cfg)
    assert any("Python 3.11" in p and "autocommit" in p for p in problems)


def test_validate_for_run_requires_train_py(project: Project, tmp_path: Path):
    dp_dir = tmp_path / "diffusion-pipe-no-train"
    dp_dir.mkdir()
    # Note: no train.py
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"")
    llm = tmp_path / "llm.safetensors"
    llm.write_bytes(b"")
    cfg = project.training
    cfg.diffusion_pipe_dir = str(dp_dir)
    cfg.dit_path = str(dit)
    cfg.vae_path = str(vae)
    cfg.llm_path = str(llm)
    problems = training.validate_for_run(cfg)
    assert any("train.py" in p for p in problems)


def test_validate_for_run_rejects_fp8_transformer_dtype(
    project: Project, tmp_path: Path,
):
    """Stale on-disk configs from an older version of the 'Fit in 8 GB'
    preset still carry transformer_dtype=float8. Diffusion-pipe's current
    Anima path crashes on it (RMSNorm fp8/fp32 promotion in the LLM
    adapter). The Start button's gate must surface a clear, actionable
    message instead of letting the user wait through caching only to hit
    the deepspeed traceback."""
    cfg = project.training
    cfg.transformer_dtype = "float8"
    problems = training.validate_for_run(cfg)
    assert any("transformer_dtype" in p and "float8" in p for p in problems)
    # Bfloat16 (the recipe default) must not be flagged.
    cfg.transformer_dtype = "bfloat16"
    problems = training.validate_for_run(cfg)
    assert not any("transformer_dtype" in p for p in problems)


# ----- TOML rendering -------------------------------------------------------


def test_render_dataset_toml_defaults_to_kept_dir(project: Project):
    text = training.render_dataset_toml(project)
    assert "[[directory]]" in text
    assert str(project.kept_dir.resolve()) in text
    # AR bucket fields tracked from config.
    assert "enable_ar_bucket = true" in text
    assert "min_ar = 0.5" in text
    assert "num_ar_buckets = 9" in text
    # Default mixed resolutions.
    assert "[512, 1024]" in text


def test_render_dataset_toml_uses_passed_dataset_root(
    project: Project, tmp_path: Path,
):
    """The runner passes a per-run staging dir so the TOML must point at
    that path, not at kept_dir — otherwise diffusion-pipe would see the
    raw `_crop` derivatives as separate samples."""
    staged = tmp_path / "staged"
    staged.mkdir()
    text = training.render_dataset_toml(project, dataset_root=staged)
    assert str(staged.resolve()) in text
    assert str(project.kept_dir.resolve()) not in text


def test_render_run_toml_matches_reference_recipe(
    project: Project, tmp_path: Path,
):
    project.training.dit_path = str(tmp_path / "dit.bin")
    project.training.vae_path = str(tmp_path / "vae.bin")
    project.training.llm_path = str(tmp_path / "llm.bin")
    run_dir = tmp_path / "run"
    ds = tmp_path / "ds.toml"
    text = training.render_run_toml(
        project, run_dir=run_dir, dataset_toml_path=ds,
    )
    # Anima-specific knobs from the reference recipe.
    assert 'type = "anima"' in text
    assert "llm_adapter_lr = 0.0" in text
    assert "sigmoid_scale = 1.3" in text
    # LoRA adapter rank.
    assert "rank = 32" in text
    # Optimizer.
    assert 'type = "adamw_optimi"' in text
    assert "betas = [0.9, 0.99]" in text
    # Recipe-default transformer dtype.
    assert 'dtype = "bfloat16"' in text
    # Recipe-default activation checkpointing: bare boolean ``true``,
    # NOT the string ``"true"`` and NOT ``"unsloth"`` — diffusion-pipe's
    # train.py branches on ``== True``.
    assert "activation_checkpointing = true" in text
    assert 'activation_checkpointing = "unsloth"' not in text
    # blocks_to_swap is suppressed entirely when 0 — diffusion-pipe treats
    # the absence of the key the same as 0, and emitting `= 0` would just
    # be visual noise in the generated TOML.
    assert "blocks_to_swap" not in text
    # torch.compile speeds up the recipe-default path materially; the
    # constant-token bucketing is specifically arranged to keep Inductor
    # on a single static shape (see anima-lora-training-notes.md).
    assert "compile = true" in text
    # No resume flag when not requested.
    assert "resume_from_checkpoint" not in text


def test_render_run_toml_low_vram_profile(project: Project, tmp_path: Path):
    """The 'Fit in 8 GB' preset on the frontend stamps these values; the
    rendered TOML must surface them where diffusion-pipe expects them:
    ``blocks_to_swap`` at the top level, base ``dtype = bfloat16`` inside
    ``[model]``, ``activation_checkpointing = "unsloth"``, optimizer type
    ``AdamW8bitKahan``, and torch.compile suppressed (incompatible with
    block_swap's mid-step parameter movement)."""
    import tomllib
    project.training.dit_path = str(tmp_path / "dit.bin")
    project.training.vae_path = str(tmp_path / "vae.bin")
    project.training.llm_path = str(tmp_path / "llm.bin")
    # The current preset does NOT enable fp8 — see test below for why.
    project.training.blocks_to_swap = 26
    project.training.optimizer_type = "AdamW8bitKahan"
    project.training.activation_checkpointing_mode = "unsloth"
    text = training.render_run_toml(
        project, run_dir=tmp_path / "run", dataset_toml_path=tmp_path / "ds.toml",
    )
    parsed = tomllib.loads(text)
    assert parsed["blocks_to_swap"] == 26
    assert parsed["model"]["dtype"] == "bfloat16"
    # compile is off whenever block_swap is on (compile assumes static
    # param location; block_swap moves params mid-step).
    assert "compile" not in parsed
    # blocks_to_swap must NOT leak into [model] or any other section.
    for section in ("optimizer", "adapter", "model"):
        if section in parsed:
            assert "blocks_to_swap" not in parsed[section]
    # 8-bit optimizer + aggressive checkpointing — the levers that close
    # the last bit of headroom on a small card. Optimizer kwargs (lr,
    # betas, weight_decay, eps) must still be present unchanged: train.py
    # forwards everything except ``type`` to the optimizer constructor.
    assert parsed["optimizer"]["type"] == "AdamW8bitKahan"
    assert parsed["optimizer"]["lr"] == project.training.learning_rate
    assert parsed["optimizer"]["eps"] == project.training.eps
    assert parsed["activation_checkpointing"] == "unsloth"


def test_render_run_toml_recipe_defaults_unchanged_by_new_fields(
    project: Project, tmp_path: Path,
):
    """Regression guard: the 24 GB-class recipe-default path must stay
    byte-identical to the pre-preset output. If a future schema change
    breaks this, anyone running the recipe defaults loses their proven
    config — exactly what the user told us not to disturb."""
    import tomllib
    project.training.dit_path = str(tmp_path / "dit.bin")
    project.training.vae_path = str(tmp_path / "vae.bin")
    project.training.llm_path = str(tmp_path / "llm.bin")
    # Spot-check the defaults so this test catches an accidental change
    # to the dataclass defaults too.
    assert project.training.optimizer_type == "adamw_optimi"
    assert project.training.activation_checkpointing_mode == "default"
    assert project.training.blocks_to_swap == 0
    text = training.render_run_toml(
        project, run_dir=tmp_path / "run", dataset_toml_path=tmp_path / "ds.toml",
    )
    parsed = tomllib.loads(text)
    assert parsed["optimizer"]["type"] == "adamw_optimi"
    assert parsed["activation_checkpointing"] is True
    assert "blocks_to_swap" not in parsed
    assert parsed["compile"] is True


def test_render_run_toml_emits_fp8_when_set(project: Project, tmp_path: Path):
    """The schema field is preserved for forked diffusion-pipe builds that
    fix the upstream LLM-adapter fp8 bug (their llm_adapter.embed has
    ndim==2 and gets quantized, then RMSNorm crashes promoting fp8*fp32).
    The renderer must emit transformer_dtype as a separate ``[model]`` key,
    NOT clobber the base ``dtype`` (which the VAE init reads directly and
    breaks on float8 — NotImplementedError "reciprocal_cpu")."""
    import tomllib
    project.training.dit_path = str(tmp_path / "dit.bin")
    project.training.vae_path = str(tmp_path / "vae.bin")
    project.training.llm_path = str(tmp_path / "llm.bin")
    project.training.transformer_dtype = "float8"
    text = training.render_run_toml(
        project, run_dir=tmp_path / "run", dataset_toml_path=tmp_path / "ds.toml",
    )
    parsed = tomllib.loads(text)
    assert parsed["model"]["dtype"] == "bfloat16"
    assert parsed["model"]["transformer_dtype"] == "float8"
    # fp8 also crashes Inductor fusion — compile must be off.
    assert "compile" not in parsed


def test_render_run_toml_omits_transformer_dtype_when_default(
    project: Project, tmp_path: Path,
):
    """When transformer_dtype matches the base bfloat16 dtype, we suppress
    the redundant key so default-recipe TOMLs stay byte-identical to the
    pre-feature output. Diffusion-pipe falls back to ``dtype`` automatically
    via ``model_config.get('transformer_dtype', dtype)``."""
    project.training.dit_path = str(tmp_path / "dit.bin")
    project.training.vae_path = str(tmp_path / "vae.bin")
    project.training.llm_path = str(tmp_path / "llm.bin")
    # Default is "bfloat16"; assert explicitly so a future default change
    # would force this test to be revisited.
    assert project.training.transformer_dtype == "bfloat16"
    text = training.render_run_toml(
        project, run_dir=tmp_path / "run", dataset_toml_path=tmp_path / "ds.toml",
    )
    assert "transformer_dtype" not in text


def test_render_run_toml_includes_resume_flag(project: Project, tmp_path: Path):
    text = training.render_run_toml(
        project,
        run_dir=tmp_path / "run",
        dataset_toml_path=tmp_path / "ds.toml",
        resume_from_checkpoint="epoch20",
    )
    assert 'resume_from_checkpoint = "epoch20"' in text


def test_render_run_toml_emits_resume_at_top_level(
    project: Project, tmp_path: Path,
):
    """``resume_from_checkpoint`` must be a top-level key, not nested under a
    section. If it lands under [optimizer] (the previous bug), DeepSpeed
    forwards it to AdamW as a kwarg and training crashes with TypeError."""
    import tomllib
    project.training.dit_path = str(tmp_path / "dit.bin")
    project.training.vae_path = str(tmp_path / "vae.bin")
    project.training.llm_path = str(tmp_path / "llm.bin")
    text = training.render_run_toml(
        project,
        run_dir=tmp_path / "run",
        dataset_toml_path=tmp_path / "ds.toml",
        resume_from_checkpoint="20260501_11-51-58",
    )
    parsed = tomllib.loads(text)
    assert parsed.get("resume_from_checkpoint") == "20260501_11-51-58"
    # Sanity: not bleeding into any subsection.
    for section in ("optimizer", "adapter", "model"):
        if section in parsed:
            assert "resume_from_checkpoint" not in parsed[section]


def test_run_toml_quotes_paths_with_special_chars(
    project: Project, tmp_path: Path,
):
    weird = tmp_path / 'name with "quotes".bin'
    weird.write_bytes(b"")
    project.training.dit_path = str(weird)
    text = training.render_run_toml(
        project, run_dir=tmp_path, dataset_toml_path=tmp_path / "ds.toml",
    )
    # Embedded double-quotes must be escaped, not bare.
    assert r'\"quotes\"' in text


# ----- dataset staging ------------------------------------------------------


def _png(path: Path, value: int = 0) -> None:
    """Write a 4×4 solid-color PNG at ``path``."""
    import numpy as np
    from PIL import Image
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def test_build_dataset_staging_pairs_originals(project: Project, tmp_path: Path):
    """Frames without a crop are staged as plain symlinks to the original
    image and its sidecar."""
    _png(project.kept_dir / "f1.png")
    (project.kept_dir / "f1.txt").write_text("tag_a, tag_b\n", encoding="utf-8")

    dest = tmp_path / "ds"
    info = training.build_dataset_staging(project, dest)
    assert info["images"] == 1
    assert info["with_crop"] == 0
    assert info["missing_txt"] == 0

    # Both pair members exist at the staging path.
    assert (dest / "f1.png").exists()
    assert (dest / "f1.txt").exists()
    # Sidecar content reads back through the link.
    assert (dest / "f1.txt").read_text(encoding="utf-8") == "tag_a, tag_b\n"


def test_build_dataset_staging_substitutes_crop_image(
    project: Project, tmp_path: Path,
):
    """When a `_crop` derivative exists, the staged image points at the
    crop's pixels but the sidecar still points at the original `.txt`.
    This is the on-disk realization of "edit tags on the original; train
    on the crop"."""
    _png(project.kept_dir / "f1.png", value=200)        # original (light)
    _png(project.kept_dir / "f1_crop.png", value=50)    # crop (dark)
    (project.kept_dir / "f1.txt").write_text("orig_tags\n", encoding="utf-8")

    dest = tmp_path / "ds"
    info = training.build_dataset_staging(project, dest)
    assert info["images"] == 1
    assert info["with_crop"] == 1
    # The trainer sees `f1.png`, not `f1_crop.png` — pairing is by stem.
    assert sorted(p.name for p in dest.iterdir()) == ["f1.png", "f1.txt"]
    # The staged image's bytes are the crop's (dark pixels).
    import numpy as np
    from PIL import Image
    with Image.open(dest / "f1.png") as im:
        assert int(np.array(im).mean()) < 100
    # The staged sidecar's content is the original's tags.
    assert (dest / "f1.txt").read_text(encoding="utf-8") == "orig_tags\n"


def test_build_dataset_staging_ignores_legacy_crop_txt(
    project: Project, tmp_path: Path,
):
    """A leftover `<name>_crop.txt` from older project layouts must not
    leak into the trainer's view — only the original sidecar is staged."""
    _png(project.kept_dir / "f1.png")
    _png(project.kept_dir / "f1_crop.png")
    (project.kept_dir / "f1.txt").write_text("orig\n", encoding="utf-8")
    (project.kept_dir / "f1_crop.txt").write_text("legacy\n", encoding="utf-8")

    dest = tmp_path / "ds"
    training.build_dataset_staging(project, dest)
    assert (dest / "f1.txt").read_text(encoding="utf-8") == "orig\n"
    # No `f1_crop.*` shadow files in the staging dir.
    assert not any(p.name.endswith("_crop.png") for p in dest.iterdir())
    assert not any(p.name.endswith("_crop.txt") for p in dest.iterdir())


def test_build_dataset_staging_rebuilds_dest(
    project: Project, tmp_path: Path,
):
    """Re-running staging must wipe stale links so a removed/renamed
    frame doesn't linger in the trainer's view."""
    _png(project.kept_dir / "f1.png")
    (project.kept_dir / "f1.txt").write_text("a\n", encoding="utf-8")

    dest = tmp_path / "ds"
    training.build_dataset_staging(project, dest)
    assert (dest / "f1.png").exists()

    # Drop f1; add f2. The next staging pass should reflect that.
    (project.kept_dir / "f1.png").unlink()
    (project.kept_dir / "f1.txt").unlink()
    _png(project.kept_dir / "f2.png")
    (project.kept_dir / "f2.txt").write_text("b\n", encoding="utf-8")

    info = training.build_dataset_staging(project, dest)
    assert info["images"] == 1
    assert not (dest / "f1.png").exists()
    assert (dest / "f2.png").exists()


def test_build_dataset_staging_counts_missing_txt(
    project: Project, tmp_path: Path,
):
    """A sidecar-less image is still staged (the trainer can train on it
    if the user is OK with empty captions) but the count surfaces so the
    caller can warn."""
    _png(project.kept_dir / "f1.png")
    # No f1.txt on disk.
    info = training.build_dataset_staging(project, tmp_path / "ds")
    assert info["images"] == 1
    assert info["missing_txt"] == 1


def test_build_dataset_staging_prepends_trigger_token(tmp_path):
    """Each character's trigger_token is prepended to the danbooru line of
    its frames' staged sidecars. The on-disk source sidecar in kept/ is
    untouched."""
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.training import build_dataset_staging

    p = Project.create(tmp_path / "proj", name="proj")
    p.characters[0].trigger_token = "mychar"
    p.save()

    # One kept frame for the default character.
    fname = "vid__s0_t0_f0"
    (p.kept_dir / f"{fname}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (p.kept_dir / f"{fname}.txt").write_text(
        "tag1, tag2\na natural language description\n", encoding="utf-8",
    )
    MetadataLog(p.metadata_path).append(FrameRecord(
        filename=fname, kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
        sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
        video_stem="vid", character_slug=p.characters[0].slug,
    ))

    dest = tmp_path / "stage"
    build_dataset_staging(p, dest)

    staged = (dest / f"{fname}.txt").read_text(encoding="utf-8")
    assert staged.startswith("mychar, tag1, tag2"), staged
    # On-disk source untouched.
    src = (p.kept_dir / f"{fname}.txt").read_text(encoding="utf-8")
    assert src.startswith("tag1, tag2"), src
    # Sidecar shape: trailing newline preserved (POSIX-friendly diffs).
    assert staged.endswith("\n"), staged


def test_build_dataset_staging_prunes_then_prepends_trigger(tmp_path):
    """When both core_tags pruning and trigger_token are configured for a
    character, pruning runs first, then the trigger is prepended. Order
    matters: trigger must not be a candidate for pruning."""
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.training import build_dataset_staging

    p = Project.create(tmp_path / "proj", name="proj")
    c = p.characters[0]
    c.trigger_token = "mychar"
    c.core_tags = ["red eyes"]
    c.core_tags_enabled = True
    p.save()

    fname = "vid__s0_t0_f0"
    (p.kept_dir / f"{fname}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (p.kept_dir / f"{fname}.txt").write_text(
        "red eyes, blue hair, smile\nbackground description\n",
        encoding="utf-8",
    )
    MetadataLog(p.metadata_path).append(FrameRecord(
        filename=fname, kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
        sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
        video_stem="vid", character_slug=c.slug,
    ))

    dest = tmp_path / "stage"
    build_dataset_staging(p, dest)
    staged = (dest / f"{fname}.txt").read_text(encoding="utf-8")
    # red eyes pruned, then trigger prepended
    assert staged.startswith("mychar, blue hair, smile"), staged
    assert "red eyes" not in staged.split("\n", 1)[0]


def test_build_dataset_staging_per_character_trigger(tmp_path):
    """Two characters with distinct trigger_tokens — each character's
    frames get only their own trigger."""
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.training import build_dataset_staging

    p = Project.create(tmp_path / "proj", name="proj")
    p.characters[0].trigger_token = "alpha"
    p.add_character(name="Beta", slug="beta")
    p.character_by_slug("beta").trigger_token = "beta"
    p.save()

    log = MetadataLog(p.metadata_path)
    for slug in ("default", "beta"):
        fn = f"vid__{slug}"
        (p.kept_dir / f"{fn}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (p.kept_dir / f"{fn}.txt").write_text("tag\n", encoding="utf-8")
        log.append(FrameRecord(
            filename=fn, kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
            timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
            sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
            video_stem="vid", character_slug=slug,
        ))

    dest = tmp_path / "stage"
    build_dataset_staging(p, dest)
    # Multi-character mode → per-slug subdirs
    a = (dest / "default" / "vid__default.txt").read_text(encoding="utf-8")
    b = (dest / "beta" / "vid__beta.txt").read_text(encoding="utf-8")
    assert a.startswith("alpha, tag"), a
    assert b.startswith("beta, tag"), b
    assert "beta" not in a.split("\n", 1)[0]
    assert "alpha" not in b.split("\n", 1)[0]


def test_dataset_preview_uses_per_character_trigger(tmp_path):
    """Two characters with distinct trigger tokens. dataset_preview's
    samples must render each frame's caption with that frame's owning
    character's trigger — not characters[0]'s for everyone."""
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.training import dataset_preview

    p = Project.create(tmp_path / "proj", name="proj")
    p.characters[0].trigger_token = "alpha"
    p.add_character(name="Beta", slug="beta")
    p.character_by_slug("beta").trigger_token = "beta"
    p.save()

    log = MetadataLog(p.metadata_path)
    for slug in ("default", "beta"):
        fn = f"vid__{slug}"
        (p.kept_dir / f"{fn}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (p.kept_dir / f"{fn}.txt").write_text("tag\n", encoding="utf-8")
        log.append(FrameRecord(
            filename=fn, kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
            timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
            sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
            video_stem="vid", character_slug=slug,
        ))

    preview = dataset_preview(p, sample_n=10)
    by_filename = {s["filename"]: s for s in preview["samples"]}
    assert by_filename["vid__default.png"]["rendered"].startswith("alpha,")
    assert by_filename["vid__beta.png"]["rendered"].startswith("beta,")


# ----- launcher argv --------------------------------------------------------


def test_default_launcher_argv(project: Project, tmp_path: Path):
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    # First token is the launcher binary, possibly resolved to an absolute
    # path if 'deepspeed' is in PATH or the diffusion-pipe venv.
    assert argv[0].endswith("deepspeed")
    assert "--num_gpus=1" in argv
    assert str(run_toml.resolve()) in argv
    # No block_swap → trainer is plain train.py, not the shim.
    assert "train.py" in argv
    assert "_diffusion_pipe_shim.py" not in " ".join(argv)


def test_launcher_swaps_in_shim_when_block_swap_enabled(
    project: Project, tmp_path: Path,
):
    """When block_swap > 0, the launcher must invoke our pre-train shim
    instead of train.py directly. The shim patches diffusion-pipe's
    offloader to use blocking GPU↔CPU transfers — without it, WSL2 users
    OOM on the pinned-memory ceiling regardless of available VRAM."""
    project.training.blocks_to_swap = 16
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    # train.py replaced with the shim path; argv shape otherwise unchanged.
    assert "train.py" not in argv
    assert any(tok.endswith("_diffusion_pipe_shim.py") for tok in argv)
    # The shim's path must be absolute — the launcher subprocess runs with
    # cwd=diffusion_pipe_dir, so a relative path wouldn't resolve.
    shim_tok = next(tok for tok in argv if tok.endswith("_diffusion_pipe_shim.py"))
    assert Path(shim_tok).is_absolute()
    assert Path(shim_tok).is_file()


def test_launcher_does_not_swap_shim_for_custom_template(
    project: Project, tmp_path: Path,
):
    """When the user has set launcher_override to a custom template that
    doesn't name train.py, leave it alone — they're running a different
    trainer and our shim wouldn't apply."""
    project.training.blocks_to_swap = 16
    project.training.launcher_override = "/bin/sh custom_trainer.sh {config}"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv == ["/bin/sh", "custom_trainer.sh", str(run_toml.resolve())]


def test_launcher_override_with_placeholder(
    project: Project, tmp_path: Path,
):
    # Use a binary we know exists at a stable path so the resolution is
    # deterministic across hosts.
    project.training.launcher_override = "/bin/sh wrapper.py {config}"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv == ["/bin/sh", "wrapper.py", str(run_toml.resolve())]


def test_launcher_override_without_placeholder_appends(
    project: Project, tmp_path: Path,
):
    project.training.launcher_override = "/bin/sh wrapper.py"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv == ["/bin/sh", "wrapper.py", str(run_toml.resolve())]


def test_launcher_resolves_via_diffusion_pipe_venv(
    project: Project, tmp_path: Path,
):
    """A binary in <diffusion_pipe_dir>/.venv/bin should be discovered even
    when not on the system PATH."""
    dp = tmp_path / "dp"
    venv_bin = dp / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake = venv_bin / "fake-launcher"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    project.training.diffusion_pipe_dir = str(dp)
    project.training.launcher_override = "fake-launcher {config}"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv[0] == str(fake)


# ----- checkpoint discovery + retention -------------------------------------


def _make_ckpt(parent: Path, name: str) -> Path:
    p = parent / name
    p.mkdir(parents=True)
    (p / "weights.safetensors").write_bytes(b"x" * 1024)
    return p


def test_discover_checkpoints_sorts_by_epoch(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "epoch10")
    _make_ckpt(run_dir, "epoch2")
    _make_ckpt(run_dir, "epoch5")
    cps = training.discover_checkpoints(run_dir)
    assert [c.name for c in cps] == ["epoch2", "epoch5", "epoch10"]
    for c in cps:
        assert c.size_bytes >= 1024


def test_discover_checkpoints_handles_global_step(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "global_step1000")
    _make_ckpt(run_dir, "global_step500")
    cps = training.discover_checkpoints(run_dir)
    assert [c.name for c in cps] == ["global_step500", "global_step1000"]


def test_discover_checkpoints_ignores_non_ckpt_dirs(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "epoch1")
    (run_dir / "logs").mkdir()
    (run_dir / "config.json").write_text("{}")
    cps = training.discover_checkpoints(run_dir)
    assert [c.name for c in cps] == ["epoch1"]


def test_discover_checkpoints_finds_nested_diffusion_pipe_layout(tmp_path: Path):
    """diffusion-pipe writes its checkpoints into a per-launch timestamped
    subdirectory under ``output_dir``, not directly into ``output_dir``."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sub = run_dir / "20260501_11-51-58"
    sub.mkdir()
    _make_ckpt(sub, "epoch10")
    _make_ckpt(sub, "epoch20")
    _make_ckpt(sub, "global_step720")
    cps = training.discover_checkpoints(run_dir)
    assert {c.name for c in cps} == {"epoch10", "epoch20", "global_step720"}
    for c in cps:
        assert c.subdir == "20260501_11-51-58"
        assert str(sub) in c.path


def test_discover_checkpoints_skips_dataset_dir(tmp_path: Path):
    """The staged training dataset lives under run_dir/dataset/ and must
    never be treated as a checkpoint container."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dataset").mkdir()
    # If we were to recurse here we'd find nothing, but we still want to
    # avoid the wasted scan: just assert no spurious checkpoints surface.
    sub = run_dir / "20260501_11-51-58"
    sub.mkdir()
    _make_ckpt(sub, "epoch1")
    cps = training.discover_checkpoints(run_dir)
    assert [c.name for c in cps] == ["epoch1"]


def test_find_resumable_subdir_returns_subdir_with_latest_marker(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sub = run_dir / "20260501_11-51-58"
    sub.mkdir()
    (sub / "latest").write_text("global_step720")
    assert training.find_resumable_subdir(run_dir) == "20260501_11-51-58"


def test_find_resumable_subdir_picks_most_recent(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    older = run_dir / "20260501_11-51-58"
    older.mkdir()
    (older / "latest").write_text("global_step100")
    newer = run_dir / "20260501_14-22-10"
    newer.mkdir()
    newer_latest = newer / "latest"
    newer_latest.write_text("global_step200")
    # Force mtime ordering so the assertion is robust on fast filesystems.
    import os
    import time
    os.utime(older / "latest", (time.time() - 60, time.time() - 60))
    assert training.find_resumable_subdir(run_dir) == "20260501_14-22-10"


def test_find_resumable_subdir_none_when_no_latest(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sub = run_dir / "20260501_11-51-58"
    sub.mkdir()
    # No 'latest' marker = nothing to resume from.
    assert training.find_resumable_subdir(run_dir) is None


def test_tag_safetensors_metadata_rewrites_header_only(tmp_path: Path):
    weights = tmp_path / "adapter_model.safetensors"
    _write_tiny_safetensors(weights)
    before = weights.read_bytes()

    changed = training.tag_safetensors_metadata(
        weights, {"neme_anima_generated_by": "neme-anima"},
    )

    assert changed is True
    assert weights.read_bytes().endswith(b"DATA")
    assert weights.read_bytes() != before
    meta = _read_safetensors_metadata(weights)
    assert meta["format"] == "pt"
    assert meta["neme_anima_generated_by"] == "neme-anima"
    assert meta["neme_anima_tagged_at"]


def test_tag_run_safetensors_adds_project_provenance(
    project: Project, tmp_path: Path,
):
    run_dir = tmp_path / "run"
    sub = run_dir / "20260501_11-51-58"
    ckpt = sub / "epoch10"
    ckpt.mkdir(parents=True)
    (run_dir / "run.toml").write_text("# Auto-generated by neme-anima\n")
    (run_dir / "dataset.toml").write_text("# Auto-generated by neme-anima\n")
    weights = ckpt / "adapter_model.safetensors"
    _write_tiny_safetensors(weights)

    tagged = training.tag_run_safetensors(project, run_dir)

    assert tagged == [str(weights.resolve())]
    meta = _read_safetensors_metadata(weights)
    assert meta["neme_anima"] == "true"
    assert meta["neme_anima_generated_by"] == "neme-anima"
    assert meta["neme_anima_project_slug"] == project.slug
    assert meta["neme_anima_run_name"] == run_dir.name
    assert meta["neme_anima_checkpoint"] == "epoch10"
    assert meta["neme_anima_trainer"] == "diffusion-pipe"
    assert len(meta["neme_anima_run_toml_sha256"]) == 64
    assert len(meta["neme_anima_dataset_toml_sha256"]) == 64


def test_prune_keeps_all_when_zero(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for i in (1, 2, 3, 4):
        _make_ckpt(run_dir, f"epoch{i}")
    deleted = training.prune_checkpoints(run_dir, keep_last_n=0)
    assert deleted == []
    assert len(training.discover_checkpoints(run_dir)) == 4


def test_prune_keeps_last_n(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for i in (1, 2, 3, 4, 5):
        _make_ckpt(run_dir, f"epoch{i}")
    deleted = training.prune_checkpoints(run_dir, keep_last_n=2)
    assert sorted(deleted) == ["epoch1", "epoch2", "epoch3"]
    remaining = [c.name for c in training.discover_checkpoints(run_dir)]
    assert remaining == ["epoch4", "epoch5"]


def test_prune_spares_protected_epochs(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for i in (1, 2, 3, 4, 5):
        _make_ckpt(run_dir, f"epoch{i}")
    deleted = training.prune_checkpoints(run_dir, keep_last_n=2, protect={2})
    assert sorted(deleted) == ["epoch1", "epoch3"]
    remaining = [c.name for c in training.discover_checkpoints(run_dir)]
    assert remaining == ["epoch2", "epoch4", "epoch5"]


def test_prune_no_op_when_under_limit(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "epoch1")
    _make_ckpt(run_dir, "epoch2")
    assert training.prune_checkpoints(run_dir, keep_last_n=5) == []


def test_prune_trims_resume_state_to_latest_keeping_all_epochs(tmp_path: Path):
    """keep_last_n=0 keeps every epoch adapter (the deliverables) but still
    trims the heavy global_step resume state down to the latest one."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for i in (1, 2, 3):
        _make_ckpt(run_dir, f"epoch{i}")
    for step in (100, 200, 300):
        _make_ckpt(run_dir, f"global_step{step}")
    deleted = training.prune_checkpoints(run_dir, keep_last_n=0)
    assert sorted(deleted) == ["global_step100", "global_step200"]
    remaining = {c.name for c in training.discover_checkpoints(run_dir)}
    assert remaining == {"epoch1", "epoch2", "epoch3", "global_step300"}


def test_prune_resume_and_epochs_pruned_independently(tmp_path: Path):
    """keep_last_n counts only epoch deliverables; resume state is always
    trimmed to the latest regardless of that number."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for i in (1, 2, 3, 4, 5):
        _make_ckpt(run_dir, f"epoch{i}")
    for step in (100, 200, 300):
        _make_ckpt(run_dir, f"global_step{step}")
    deleted = training.prune_checkpoints(run_dir, keep_last_n=2)
    assert sorted(deleted) == [
        "epoch1", "epoch2", "epoch3", "global_step100", "global_step200",
    ]
    remaining = {c.name for c in training.discover_checkpoints(run_dir)}
    assert remaining == {"epoch4", "epoch5", "global_step300"}


def test_prune_leaves_single_resume_checkpoint_untouched(tmp_path: Path):
    """One resume checkpoint is the latest by definition — nothing to trim."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "epoch1")
    _make_ckpt(run_dir, "global_step500")
    assert training.prune_checkpoints(run_dir, keep_last_n=0) == []
    remaining = {c.name for c in training.discover_checkpoints(run_dir)}
    assert remaining == {"epoch1", "global_step500"}


# ----- caption rendering ----------------------------------------------------


def test_render_training_caption_modes():
    cfg = TrainingConfig()
    cfg.caption_mode = "tags"
    assert training.render_training_caption(tags="1girl, blue eyes", nl="A girl smiles.", config=cfg) == "1girl, blue eyes"

    cfg.caption_mode = "nl"
    assert training.render_training_caption(tags="1girl", nl="A girl.", config=cfg) == "A girl."

    cfg.caption_mode = "mixed"
    assert training.render_training_caption(tags="1girl", nl="A girl.", config=cfg) == "1girl. A girl."


def test_render_training_caption_with_trigger():
    cfg = TrainingConfig()
    cfg.trigger_token = "mychar"
    cfg.caption_mode = "mixed"
    out = training.render_training_caption(tags="1girl", nl="A girl.", config=cfg)
    assert out == "mychar, 1girl. A girl."


# ----- run-dir helpers + project storage round-trip ------------------------


def test_new_run_dir_creates_unique_dirs(project: Project):
    a = training.new_run_dir(project, label="x")
    b = training.new_run_dir(project, label="x")
    assert a.exists() and b.exists()
    assert a != b


def test_training_config_round_trips(project: Project):
    project.training.dit_path = "/foo/dit.safetensors"
    project.training.keep_last_n_checkpoints = 5
    project.training.preset = "character"
    project.training.learning_rate = 5e-5
    project.training.resolutions = [768, 1024]
    project.save()

    reloaded = Project.load(project.root)
    assert reloaded.training.dit_path == "/foo/dit.safetensors"
    assert reloaded.training.keep_last_n_checkpoints == 5
    assert reloaded.training.preset == "character"
    assert reloaded.training.learning_rate == 5e-5
    assert reloaded.training.resolutions == [768, 1024]


def test_training_config_default_round_trip_keep_all(project: Project):
    project.save()
    reloaded = Project.load(project.root)
    # The user-requested default is 0 — keep all checkpoints.
    assert reloaded.training.keep_last_n_checkpoints == 0

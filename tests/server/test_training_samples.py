"""Pure sample-generation helpers in ``neme_anima.training``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from neme_anima import training
from neme_anima.storage.project import Project, TrainingConfig


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "p", name="p")


def _fake_dp(tmp_path: Path, *, venv: bool = True, comfy: bool = True) -> Path:
    dp = tmp_path / "diffusion-pipe"
    dp.mkdir()
    (dp / "train.py").write_text("# fake")
    if venv:
        py = dp / ".venv" / "bin" / "python"
        py.parent.mkdir(parents=True)
        # Answers the Python-version probe in _diffusion_pipe_python_problem.
        py.write_text("#!/bin/sh\necho 3.12\n")
        py.chmod(0o755)
    if comfy:
        (dp / "submodules" / "ComfyUI" / "comfy").mkdir(parents=True)
    return dp


def _sampling_cfg(tmp_path: Path, **kw) -> TrainingConfig:
    dp = _fake_dp(tmp_path)
    cfg = TrainingConfig(diffusion_pipe_dir=str(dp), sample_prompts=["1girl"])
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _sample_problems(cfg: TrainingConfig) -> list[str]:
    return training._sample_problems(cfg)


def _make_ckpt(run_dir: Path, epoch: int, *, complete: bool = True,
               subdir: str = "20260926_10-00-00") -> Path:
    d = run_dir / subdir / f"epoch{epoch}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"x")
    if complete:
        (d / "run.toml").write_text("# copied last by diffusion-pipe")
    return d


def test_sample_defaults():
    cfg = TrainingConfig()
    assert cfg.sample_prompts == []
    assert cfg.sample_negative_prompt == ""
    assert cfg.sample_every_n_epochs == 10
    assert cfg.sample_defer_to_end is False
    assert (cfg.sample_steps, cfg.sample_sampler, cfg.sample_scheduler) == (
        30, "euler_ancestral", "simple",
    )
    assert cfg.sample_cfg == 4.5
    assert (cfg.sample_width, cfg.sample_height, cfg.sample_seed) == (1024, 1024, 42)


def test_sample_defaults_survive_project_roundtrip(project: Project):
    project.training.sample_prompts = ["a", "b"]
    project.training.sample_defer_to_end = True
    project.save()
    loaded = Project.load(project.root)
    assert loaded.training.sample_prompts == ["a", "b"]
    assert loaded.training.sample_defer_to_end is True
    assert loaded.training.sample_cfg == 4.5


def test_sampling_disabled_when_prompts_blank():
    cfg = TrainingConfig(sample_prompts=["", "   "])
    assert training.sampling_enabled(cfg) is False
    assert training.sample_prompts(cfg) == []
    cfg.sample_prompts = [" 1girl ", ""]
    assert training.sampling_enabled(cfg) is True
    assert training.sample_prompts(cfg) == ["1girl"]


def test_no_sample_problems_when_disabled(tmp_path: Path):
    cfg = _sampling_cfg(tmp_path, sample_prompts=[], sample_every_n_epochs=7,
                        sample_sampler="nope")
    assert _sample_problems(cfg) == []


def test_valid_sampling_config_has_no_problems(tmp_path: Path):
    assert _sample_problems(_sampling_cfg(tmp_path)) == []


@pytest.mark.parametrize(("field", "value", "needle"), [
    ("sample_every_n_epochs", 0, "sample_every_n_epochs must be > 0"),
    ("sample_every_n_epochs", 15, "multiple of save_every_n_epochs"),
    ("sample_steps", 0, "sample_steps"),
    ("sample_cfg", -1.0, "sample_cfg"),
    ("sample_width", 1000, "sample_width"),
    ("sample_height", 4096, "sample_height"),
    ("sample_sampler", "dpmpp_sde", "sample_sampler"),
    ("sample_scheduler", "bogus", "sample_scheduler"),
])
def test_sample_validation_rejects(tmp_path: Path, field, value, needle):
    cfg = _sampling_cfg(tmp_path, **{field: value})
    assert any(needle in p for p in _sample_problems(cfg)), _sample_problems(cfg)


def test_sample_validation_needs_venv_python(tmp_path: Path):
    dp = _fake_dp(tmp_path, venv=False)
    cfg = TrainingConfig(diffusion_pipe_dir=str(dp), sample_prompts=["x"])
    assert any("venv python" in p for p in _sample_problems(cfg))


def test_sample_validation_needs_vendored_comfyui(tmp_path: Path):
    dp = _fake_dp(tmp_path, comfy=False)
    cfg = TrainingConfig(diffusion_pipe_dir=str(dp), sample_prompts=["x"])
    assert any("ComfyUI" in p for p in _sample_problems(cfg))


def test_validate_for_run_includes_sample_problems(tmp_path: Path):
    cfg = _sampling_cfg(tmp_path, sample_steps=0)
    assert "sample_steps must be > 0" in training.validate_for_run(cfg)


def test_sample_epoch_qualifies():
    cfg = TrainingConfig(sample_every_n_epochs=10, epochs=25)
    assert [e for e in range(1, 26) if training.sample_epoch_qualifies(e, cfg)] == [10, 20, 25]


def test_sample_epochs_ready_filters(tmp_path: Path):
    run_dir = tmp_path / "run"
    cfg = TrainingConfig(sample_every_n_epochs=10, epochs=40)
    _make_ckpt(run_dir, 5)                      # not a sampling epoch
    _make_ckpt(run_dir, 10)                     # ready
    _make_ckpt(run_dir, 20, complete=False)     # adapter still being written
    _make_ckpt(run_dir, 30)                     # already sampled
    done = training.sample_dir(run_dir, 30)
    done.mkdir(parents=True)
    (done / "manifest.json").write_text("{}")
    _make_ckpt(run_dir, 40)                     # final epoch, ready
    ready = training.sample_epochs_ready(run_dir, cfg)
    assert [e for e, _ in ready] == [10, 40]
    assert ready[0][1].name == "adapter_model.safetensors"


def test_sample_epochs_ready_newer_duplicate_supersedes(tmp_path: Path):
    """A resume re-saving epoch 10 in a new sub-run dir: while the new copy is
    still being written, the stale older copy must not be sampled."""
    run_dir = tmp_path / "run"
    cfg = TrainingConfig(sample_every_n_epochs=10, epochs=40)
    old = _make_ckpt(run_dir, 10, subdir="a_old")
    new = _make_ckpt(run_dir, 10, complete=False, subdir="b_new")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    assert training.sample_epochs_ready(run_dir, cfg) == []
    (new / "run.toml").write_text("")
    os.utime(new, (2, 2))
    assert training.sample_epochs_ready(run_dir, cfg) == [
        (10, new / "adapter_model.safetensors"),
    ]


def test_discover_checkpoints_ignores_samples_dir(tmp_path: Path):
    run_dir = tmp_path / "run"
    _make_ckpt(run_dir, 10)
    training.sample_dir(run_dir, 10).mkdir(parents=True)
    names = [c.name for c in training.discover_checkpoints(run_dir)]
    assert names == ["epoch10"]


def test_build_sample_job(tmp_path: Path):
    cfg = _sampling_cfg(tmp_path, sample_prompts=["a", " ", "b"],
                        sample_negative_prompt=" bad ", dit_path="/m/dit.st",
                        vae_path="/m/vae.st", llm_path="/m/llm.st")
    lora = _make_ckpt(tmp_path / "run", 10) / "adapter_model.safetensors"
    out = tmp_path / "out"
    job = training.build_sample_job(cfg, lora_path=lora, epoch=10, output_dir=out)
    assert job["prompts"] == ["a", "b"]
    assert job["negative_prompt"] == "bad"
    assert job["epoch"] == 10
    assert job["checkpoint"] == "epoch10"
    assert job["lora_path"] == str(lora.resolve())
    assert job["output_dir"] == str(out.resolve())
    assert job["dit_path"] == "/m/dit.st"
    assert job["settings"] == {
        "steps": 30, "sampler": "euler_ancestral", "scheduler": "simple",
        "cfg": 4.5, "width": 1024, "height": 1024, "seed": 42,
    }


def test_read_sample_manifests(tmp_path: Path):
    run_dir = tmp_path / "run"
    for epoch, err in ((20, None), (10, None), (30, "boom")):
        d = training.sample_dir(run_dir, epoch)
        d.mkdir(parents=True)
        images = []
        if err is None:
            (d / "p00.png").write_bytes(b"png")
            images = [{"file": "p00.png", "prompt": "a", "seconds": 1.0},
                      {"file": "p01.png", "prompt": "gone", "seconds": 1.0}]
        (d / "manifest.json").write_text(json.dumps({
            "prompts": ["a"], "negative_prompt": "", "settings": {"steps": 30},
            "images": images, "error": err,
        }))
    # Malformed manifests are skipped, not fatal.
    bad = training.sample_dir(run_dir, 50)
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("[1, 2]")
    odd = training.sample_dir(run_dir, 60)
    odd.mkdir(parents=True)
    (odd / "manifest.json").write_text(json.dumps({"images": ["p00.png"]}))
    # An in-flight epoch (no manifest yet) is not listed.
    training.sample_dir(run_dir, 40).mkdir(parents=True)
    got = training.read_sample_manifests(run_dir)
    assert [e["epoch"] for e in got] == [10, 20, 30, 60]
    assert got[3]["images"] == []
    assert got[0]["dir"] == "epoch0010"
    assert [i["file"] for i in got[0]["images"]] == ["p00.png"]  # missing file dropped
    assert "mtime" in got[0]["images"][0]
    assert got[2]["error"] == "boom"

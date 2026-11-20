"""Stop → Continue with a fake trainer that mimics diffusion-pipe's resume
plumbing: a timestamped sub-run-dir, the ``save_quit`` signal file, and
``global_step{N}`` + ``latest`` DeepSpeed snapshots."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from neme_anima import training
from neme_anima.server import training_runner
from neme_anima.server.events import Broadcaster
from neme_anima.server.training_runner import TrainingManager
from neme_anima.storage.project import Project

# argv: <mode> <run.toml>. mode "honour" saves + exits on save_quit (like
# train.py); "ignore" never looks at it (stuck in model load / eval).
FAKE_TRAINER = r"""
import sys, time, tomllib
from datetime import UTC, datetime
from pathlib import Path
mode, cfg = sys.argv[1], tomllib.loads(Path(sys.argv[2]).read_text())
resume = cfg.get("resume_from_checkpoint")
sub = Path(cfg["output_dir"]) / (resume or datetime.now(UTC).strftime("%Y%m%d_%H-%M-%S"))
sub.mkdir(parents=True, exist_ok=True)
step = 1
if resume:
    step = int((sub / "latest").read_text().removeprefix("global_step")) + 1
    print(f"Resuming at step {step}", flush=True)
while True:
    if mode == "honour" and (sub / "save_quit").exists():
        (sub / "save_quit").unlink()
        snap = sub / f"global_step{step}"
        snap.mkdir()
        (snap / "mp_rank_00_model_states.pt").write_bytes(b"x")
        (sub / "latest").write_text(snap.name)
        sys.exit(0)
    print(f"step: {step}", flush=True)
    step += 1
    time.sleep(0.05)
"""


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "p", name="p")
    dp = tmp_path / "diffusion-pipe"
    dp.mkdir()
    (dp / "train.py").write_text("# fake")
    py = dp / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\necho 3.12\n")
    py.chmod(0o755)
    for name in ("dit", "vae", "llm"):
        (tmp_path / f"{name}.safetensors").write_bytes(b"")
    script = tmp_path / "fake_trainer.py"
    script.write_text(FAKE_TRAINER)
    cfg = p.training
    cfg.diffusion_pipe_dir = str(dp)
    cfg.dit_path = str(tmp_path / "dit.safetensors")
    cfg.vae_path = str(tmp_path / "vae.safetensors")
    cfg.llm_path = str(tmp_path / "llm.safetensors")
    cfg.launcher_override = f"{sys.executable} {script} honour {{config}}"
    p.save()
    return p


def _use_mode(project: Project, mode: str) -> None:
    cfg = project.training
    cfg.launcher_override = cfg.launcher_override.replace(" honour ", f" {mode} ")


async def _wait(cond, timeout=10.0, what="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.05)


def _state(mgr: TrainingManager, project: Project) -> dict:
    return mgr.status(project)["state"] or {}


async def _wait_training(mgr: TrainingManager, project: Project) -> None:
    await _wait(lambda: _state(mgr, project).get("step") is not None, what="first step")


async def _wait_idle(mgr: TrainingManager, project: Project, timeout=10.0) -> dict:
    await _wait(lambda: not mgr._is_active(), timeout=timeout, what="run to end")
    return _state(mgr, project)


async def test_graceful_stop_saves_resume_state_then_continue(project: Project):
    mgr = TrainingManager(broadcaster=Broadcaster())
    await mgr.start(project)
    await _wait_training(mgr, project)

    resp = await mgr.stop(project)
    # Returns at once while the trainer finishes its step and saves.
    assert resp["state"]["status"] == "stopping"
    st = await _wait_idle(mgr, project)
    assert st["status"] == "stopped"

    run_dir = Path(st["run_dir"])
    sub = training.find_resumable_subdir(run_dir)
    assert sub is not None
    assert not (run_dir / sub / "save_quit").exists()
    saved_step = int((run_dir / sub / "latest").read_text().removeprefix("global_step"))

    # Continue from the snapshot — the trainer picks up after the saved step.
    await mgr.start(project, resume_from_checkpoint=sub, run_dir_name=run_dir.name)
    await _wait(
        lambda: f"Resuming at step {saved_step + 1}" in "\n".join(
            ln["line"] for ln in mgr.get_log_buffer(project)
        ),
        what="resume log line",
    )
    assert _state(mgr, project)["resumed_from"] == sub
    await mgr.stop(project)
    assert (await _wait_idle(mgr, project))["status"] == "stopped"
    # End-of-run retention keeps only the snapshot ``latest`` names.
    snaps = [c.name for c in training.discover_checkpoints(run_dir) if c.epoch is None]
    assert snaps == [(run_dir / sub / "latest").read_text()]


async def test_relaunch_clears_unconsumed_save_quit(project: Project):
    """A save_quit left behind (Stop escalated to SIGTERM before the trainer
    read it) must not make the resumed run quit at its first step."""
    run_dir = project.training_runs_dir / "run1"
    sub = run_dir / "20260501_11-51-58"
    (sub / "global_step10").mkdir(parents=True)
    (sub / "latest").write_text("global_step10")
    (sub / "save_quit").write_text("")

    mgr = TrainingManager(broadcaster=Broadcaster())
    await mgr.start(project, resume_from_checkpoint=sub.name, run_dir_name="run1")
    await _wait(lambda: (_state(mgr, project).get("step") or 0) >= 15, what="training past resume")
    assert mgr._is_active()
    await mgr.stop(project)
    await _wait_idle(mgr, project)


async def test_stop_escalates_when_trainer_ignores_save_quit(
    project: Project, monkeypatch,
):
    monkeypatch.setattr(training_runner, "GRACEFUL_STOP_TIMEOUT_S", 0.5)
    _use_mode(project, "ignore")
    mgr = TrainingManager(broadcaster=Broadcaster())
    await mgr.start(project)
    await _wait_training(mgr, project)
    assert (await mgr.stop(project))["state"]["status"] == "stopping"
    st = await _wait_idle(mgr, project, timeout=15.0)
    assert st["status"] == "stopped"
    assert any("terminating" in ln["line"] for ln in mgr._log_buffer)


async def test_second_stop_force_kills(project: Project):
    _use_mode(project, "ignore")
    mgr = TrainingManager(broadcaster=Broadcaster())
    await mgr.start(project)
    await _wait_training(mgr, project)
    await mgr.stop(project)  # graceful request; trainer ignores it
    assert mgr._is_active()
    await mgr.stop(project)  # force
    st = await _wait_idle(mgr, project, timeout=12.0)
    assert st["status"] == "stopped"

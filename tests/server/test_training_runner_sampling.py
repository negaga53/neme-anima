"""TrainingManager + SampleScheduler with a fake trainer and fake sampler."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from neme_anima import training
from neme_anima.server.events import Broadcaster
from neme_anima.server.training_runner import TrainingManager
from neme_anima.storage.project import Project

FAKE_SAMPLER = r"""
import json, os, sys, time
from pathlib import Path
job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text())
out = Path(job["output_dir"])
print(f"epoch 999 step 999 loss 9.9 fake sampler for {job['epoch']}", flush=True)
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
(out / "p00.png").write_bytes(b"png")
(out / "manifest.json").write_text(json.dumps({
    "images": [{"file": "p00.png", "prompt": job["prompts"][0]}], "error": None,
}))
"""


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "p", name="p")
    dp = tmp_path / "diffusion-pipe"
    (dp / "submodules" / "ComfyUI" / "comfy").mkdir(parents=True)
    (dp / "train.py").write_text("# fake")
    py = dp / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\necho 3.12\n")
    py.chmod(0o755)
    for name in ("dit", "vae", "llm"):
        (tmp_path / f"{name}.safetensors").write_bytes(b"")
    cfg = p.training
    cfg.diffusion_pipe_dir = str(dp)
    cfg.dit_path = str(tmp_path / "dit.safetensors")
    cfg.vae_path = str(tmp_path / "vae.safetensors")
    cfg.llm_path = str(tmp_path / "llm.safetensors")
    cfg.launcher_override = "/bin/true {config}"  # "trainer" exits at once
    cfg.epochs = 30
    cfg.save_every_n_epochs = 10
    cfg.sample_every_n_epochs = 10
    cfg.sample_prompts = ["1girl"]
    cfg.keep_last_n_checkpoints = 1
    p.save()
    return p


class CollectingBroadcaster(Broadcaster):
    def __init__(self) -> None:
        super().__init__()
        self.events = []

    async def publish(self, event) -> None:
        self.events.append(event)
        await super().publish(event)


def _manager(tmp_path: Path, bc: Broadcaster, sleep: float = 0.0) -> TrainingManager:
    script = tmp_path / "fake_sampler.py"
    script.write_text(FAKE_SAMPLER)

    async def spawn(argv, cwd):
        return await asyncio.create_subprocess_exec(
            sys.executable, str(script), *argv[argv.index("--job"):], cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "FAKE_SLEEP": str(sleep)},
        )

    return TrainingManager(broadcaster=bc, sampler_spawn=spawn)


def _pre_seed_checkpoints(project: Project, run_name: str) -> Path:
    """Stand in for diffusion-pipe output: the fake trainer writes nothing,
    so plant finished epoch LoRAs in the run dir it will reuse."""
    run_dir = project.training_runs_dir / run_name
    for epoch in (10, 20, 30):
        d = run_dir / "20260926_10-00-00" / f"epoch{epoch}"
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(b"x")
        (d / "run.toml").write_text("")
    return run_dir


async def _wait_status(mgr: TrainingManager, project: Project, statuses, timeout=15.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        st = mgr.status(project)["state"]
        if st and st["status"] in statuses and not mgr._is_active():
            return st
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out; last state {st}")
        await asyncio.sleep(0.05)


async def test_samples_every_epoch_before_pruning(tmp_path: Path, project: Project):
    run_dir = _pre_seed_checkpoints(project, "run1")
    bc = CollectingBroadcaster()
    mgr = _manager(tmp_path, bc)
    await mgr.start(project, run_dir_name="run1")
    st = await _wait_status(mgr, project, {"finished"})
    for epoch in (10, 20, 30):
        assert (training.sample_dir(run_dir, epoch) / "p00.png").is_file()
    # keep_last_n_checkpoints=1 pruned epochs 10/20 — but only after sampling.
    assert [c.name for c in training.discover_checkpoints(run_dir)] == ["epoch30"]
    sampled = sorted(e.payload["epoch"] for e in bc.events if e.type == "training.sample")
    assert sampled == [10, 20, 30]
    assert any(e.type == "training.status" and e.payload["state"]["status"] == "sampling"
               for e in bc.events)
    sample_logs = [e for e in bc.events
                   if e.type == "training.log" and e.payload["stream"] == "sample"]
    assert sample_logs
    # The fake sampler prints "epoch 999 step 999 loss 9.9" — must not leak
    # into the trainer's progress fields.
    assert st["epoch"] != 999 and st["step"] != 999
    assert st["sampling"] is None
    # _finalizing is cleared before the last broadcast, so it reports idle.
    last_status = [e for e in bc.events if e.type == "training.status"][-1]
    assert last_status.payload["running"] is False
    assert last_status.payload["state"]["status"] == "finished"


async def test_run_stays_active_while_sampling_and_stop_cancels(
    tmp_path: Path, project: Project,
):
    run_dir = _pre_seed_checkpoints(project, "run1")
    mgr = _manager(tmp_path, CollectingBroadcaster(), sleep=30)
    await mgr.start(project, run_dir_name="run1")
    deadline = asyncio.get_running_loop().time() + 10
    while (mgr.status(project)["state"] or {}).get("status") != "sampling":
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.05)
    assert mgr.status(project)["running"] is True
    with pytest.raises(RuntimeError, match="already active"):
        await mgr.start(project)
    await asyncio.wait_for(mgr.stop(project), timeout=15)
    st = await _wait_status(mgr, project, {"stopped"})
    assert st["status"] == "stopped"
    assert not (training.sample_dir(run_dir, 10) / "manifest.json").exists()


async def test_no_sampling_when_prompts_empty(tmp_path: Path, project: Project):
    project.training.sample_prompts = []
    project.save()
    run_dir = _pre_seed_checkpoints(project, "run1")
    bc = CollectingBroadcaster()
    mgr = _manager(tmp_path, bc)
    await mgr.start(project, run_dir_name="run1")
    await _wait_status(mgr, project, {"finished"})
    assert not (run_dir / "samples").exists()
    assert not any(e.type == "training.sample" for e in bc.events)

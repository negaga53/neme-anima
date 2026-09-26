"""SampleScheduler with a fake sampler subprocess (no GPU)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from neme_anima import training
from neme_anima.server.sampling_runner import SampleScheduler
from neme_anima.storage.project import TrainingConfig

# Stand-in for _anima_sampler.py: reads the job, writes one PNG + manifest.
# FAKE_MODE env var selects crash (exit 3, no manifest) or hang behaviour.
FAKE_SAMPLER = r"""
import json, os, sys, time
from pathlib import Path
job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text())
out = Path(job["output_dir"])
mode = os.environ.get("FAKE_MODE", "ok")
print(f"fake sampling epoch {job['epoch']}", flush=True)
if mode == "crash":
    sys.exit(3)
if mode == "hang":
    time.sleep(60)
(out / "p00.png").write_bytes(b"png")
(out / "manifest.json").write_text(json.dumps({
    "epoch": job["epoch"], "prompts": job["prompts"], "settings": job["settings"],
    "images": [{"file": "p00.png", "prompt": job["prompts"][0]}], "error": None,
}))
"""


def _make_ckpt(run_dir: Path, epoch: int) -> None:
    d = run_dir / "sub" / f"epoch{epoch}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"x")
    (d / "run.toml").write_text("")


class Harness:
    def __init__(self, tmp_path: Path, *, mode: str = "ok", **cfg_kw):
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir()
        script = tmp_path / "fake_sampler.py"
        script.write_text(FAKE_SAMPLER)
        self.cfg = TrainingConfig(
            diffusion_pipe_dir=str(tmp_path), sample_prompts=["1girl"],
            sample_every_n_epochs=10, epochs=40, **cfg_kw,
        )
        self.logs: list[str] = []
        self.sampled: list[int] = []
        self.spawned: list[list[str]] = []

        async def spawn(argv: list[str], cwd: str):
            self.spawned.append(argv)
            job_arg = argv[argv.index("--job"):]
            return await asyncio.create_subprocess_exec(
                sys.executable, str(script), *job_arg, cwd=cwd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "FAKE_MODE": mode},
            )

        async def on_log(line: str) -> None:
            self.logs.append(line)

        async def on_change() -> None:
            pass

        async def on_sampled(epoch: int) -> None:
            self.sampled.append(epoch)

        self.sched = SampleScheduler(
            run_dir=self.run_dir, cfg=self.cfg, python="/unused/python",
            on_log=on_log, on_change=on_change, on_sampled=on_sampled,
            spawn=spawn, poll_interval=0.05,
        )


async def _wait_for(pred, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not pred():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


async def test_concurrent_mode_samples_new_checkpoints(tmp_path: Path):
    h = Harness(tmp_path)
    h.sched.start()
    _make_ckpt(h.run_dir, 10)
    await _wait_for(lambda: h.sampled == [10])
    assert (training.sample_dir(h.run_dir, 10) / "p00.png").is_file()
    job = json.loads((training.sample_dir(h.run_dir, 10) / "job.json").read_text())
    assert job["epoch"] == 10
    assert any("fake sampling epoch 10" in line for line in h.logs)
    await h.sched.drain()


async def test_defer_mode_waits_for_drain(tmp_path: Path):
    h = Harness(tmp_path, sample_defer_to_end=True)
    h.sched.start()
    _make_ckpt(h.run_dir, 20)
    _make_ckpt(h.run_dir, 10)
    _make_ckpt(h.run_dir, 15)  # not a sampling epoch
    await asyncio.sleep(0.3)
    assert h.spawned == []
    await h.sched.drain()
    assert h.sampled == [10, 20]


async def test_crash_writes_error_manifest(tmp_path: Path):
    h = Harness(tmp_path, mode="crash")
    _make_ckpt(h.run_dir, 10)
    await h.sched.drain()
    manifest = json.loads(
        (training.sample_dir(h.run_dir, 10) / "manifest.json").read_text(),
    )
    assert "exited with code 3" in manifest["error"]
    assert h.sampled == []
    assert len(h.spawned) == 1  # not retried


async def test_cancel_kills_running_sampler(tmp_path: Path):
    h = Harness(tmp_path, mode="hang")
    h.sched.start()
    _make_ckpt(h.run_dir, 10)
    await _wait_for(lambda: h.sched.current_epoch == 10)
    await asyncio.wait_for(h.sched.cancel(), timeout=10)
    assert not (training.sample_dir(h.run_dir, 10) / "manifest.json").exists()
    assert h.sched.current_epoch is None


async def test_cancel_during_spawn_kills_sampler(tmp_path: Path):
    # cancel() landing while spawn() is still in flight (current_epoch set,
    # _proc not yet) must still kill the process rather than wait it out.
    h = Harness(tmp_path, mode="hang")
    real_spawn = h.sched._spawn
    procs: list[asyncio.subprocess.Process] = []

    async def slow_spawn(argv, cwd):
        await asyncio.sleep(0.3)
        procs.append(await real_spawn(argv, cwd))
        return procs[-1]

    h.sched._spawn = slow_spawn
    h.sched.start()
    _make_ckpt(h.run_dir, 10)
    await _wait_for(lambda: h.sched.current_epoch == 10)
    await asyncio.wait_for(h.sched.cancel(), timeout=10)
    assert len(procs) == 1 and procs[0].returncode is not None  # killed, not orphaned
    assert not (training.sample_dir(h.run_dir, 10) / "manifest.json").exists()
    assert h.sched.current_epoch is None


async def test_disabled_never_spawns(tmp_path: Path):
    h = Harness(tmp_path)
    h.cfg.sample_prompts = ["  "]
    h.sched.start()
    _make_ckpt(h.run_dir, 10)
    await h.sched.drain()
    assert h.spawned == []


async def test_spawn_failure_records_error(tmp_path: Path):
    h = Harness(tmp_path)

    async def boom(argv, cwd):
        raise FileNotFoundError("no python")

    h.sched._spawn = boom
    _make_ckpt(h.run_dir, 10)
    await h.sched.drain()
    manifest = json.loads(
        (training.sample_dir(h.run_dir, 10) / "manifest.json").read_text(),
    )
    assert "no python" in manifest["error"]


@pytest.mark.parametrize("defer", [False, True])
async def test_pending_epochs(tmp_path: Path, defer: bool):
    h = Harness(tmp_path, sample_defer_to_end=defer)
    _make_ckpt(h.run_dir, 10)
    assert h.sched.pending_epochs() == [10]

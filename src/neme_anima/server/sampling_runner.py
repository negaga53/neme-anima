"""Per-run sample-image scheduler.

A :class:`SampleScheduler` belongs to one training run (created by
:class:`~neme_anima.server.training_runner.TrainingManager`). It watches the
run directory for finished epoch LoRAs that should be sampled
(``training.sample_epochs_ready``) and renders them by running
``_anima_sampler.py`` in diffusion-pipe's venv — one subprocess at a time,
lowest epoch first.

* **Concurrent mode** (default): ``start()`` launches a poll loop that checks
  every ``poll_interval`` seconds while training runs. The sampler shares the
  GPU with the trainer; ComfyUI's memory manager offloads to fit.
* **Defer mode** (``sample_defer_to_end``): nothing runs during training.

Either way the manager calls ``drain()`` once training exits — it stops the
poll loop (letting an in-flight job finish) and renders everything left. The
manager awaits it *before* checkpoint pruning. ``cancel()`` (user Stop) kills
the sampler and drops the queue; the killed epoch gets no manifest so a
resumed run samples it again.

A sampler failure never propagates: the error is logged and stored in the
epoch's ``manifest.json`` (written here if the process died before writing
its own), which also stops that epoch from being retried in a loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from neme_anima import training as training_lib
from neme_anima.storage.project import TrainingConfig

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 15.0

SpawnFn = Callable[[list[str], str], Awaitable[asyncio.subprocess.Process]]


async def _default_spawn(argv: list[str], cwd: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # Own process group so cancel() can take down any children too.
        start_new_session=True,
        # ComfyUI/tqdm can emit long carriage-return progress runs with no
        # newline; the default 64 KiB line limit would abort the reader.
        limit=4 * 1024 * 1024,
    )


class SampleScheduler:
    def __init__(
        self,
        *,
        run_dir: Path,
        cfg: TrainingConfig,
        python: str,
        on_log: Callable[[str], Awaitable[None]],
        on_change: Callable[[], Awaitable[None]],
        on_sampled: Callable[[int], Awaitable[None]],
        spawn: SpawnFn | None = None,
        poll_interval: float = POLL_INTERVAL_S,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._cfg = cfg
        self._python = python
        self._on_log = on_log
        self._on_change = on_change
        self._on_sampled = on_sampled
        self._spawn = spawn or _default_spawn
        self._poll_interval = poll_interval
        self._job_lock = asyncio.Lock()  # one sampler process at a time
        self._wake = asyncio.Event()
        self._poll_task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._stopping = False   # poll loop should exit (drain or cancel)
        self._cancelled = False  # drop everything, kill the sampler
        # Progress, read by the manager for RunState.sampling.
        self.current_epoch: int | None = None
        self.pending: int = 0

    @property
    def enabled(self) -> bool:
        return training_lib.sampling_enabled(self._cfg)

    def pending_epochs(self) -> list[int]:
        if not self.enabled:
            return []
        return [e for e, _ in training_lib.sample_epochs_ready(self._run_dir, self._cfg)]

    def start(self) -> None:
        if self.enabled and not self._cfg.sample_defer_to_end:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def drain(self) -> None:
        """Stop polling (an in-flight job finishes) and sample what's left."""
        self._stopping = True
        self._wake.set()
        await self._join_poll_task()
        if self.enabled and not self._cancelled:
            await self._process_ready()

    async def cancel(self) -> None:
        """Kill the running sampler and drop the queue."""
        self._cancelled = True
        self._stopping = True
        self._wake.set()
        if self._proc is not None:
            self._kill(self._proc)
        await self._join_poll_task()
        # Wait out a job still unwinding under the lock.
        async with self._job_lock:
            pass

    # ----- internals -------------------------------------------------------

    async def _join_poll_task(self) -> None:
        # asyncio.wait (not ``await task`` under suppress(CancelledError)) so a
        # cancellation of *our caller* still propagates instead of being eaten.
        if self._poll_task is not None:
            await asyncio.wait({self._poll_task})
            self._poll_task = None

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    async def _poll_loop(self) -> None:
        while not self._stopping:
            try:
                await self._process_ready()
            except Exception:
                logger.exception("sampling: poll iteration failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_interval)

    async def _process_ready(self) -> None:
        async with self._job_lock:
            try:
                while not self._cancelled:
                    ready = training_lib.sample_epochs_ready(self._run_dir, self._cfg)
                    if not ready:
                        break
                    self.pending = len(ready)
                    epoch, adapter = ready[0]
                    await self._sample_one(epoch, adapter)
            finally:
                if self.current_epoch is not None or self.pending:
                    self.current_epoch = None
                    self.pending = 0
                    await self._on_change()

    async def _sample_one(self, epoch: int, adapter: Path) -> None:
        out_dir = training_lib.sample_dir(self._run_dir, epoch)
        out_dir.mkdir(parents=True, exist_ok=True)
        job = training_lib.build_sample_job(
            self._cfg, lora_path=adapter, epoch=epoch, output_dir=out_dir,
        )
        job_path = out_dir / "job.json"
        job_path.write_text(json.dumps(job, indent=2))
        argv = [
            self._python, str(training_lib.sampler_script_path()),
            "--job", str(job_path),
        ]
        self.current_epoch = epoch
        await self._on_change()
        await self._on_log(
            f"sampling epoch {epoch}: {len(job['prompts'])} prompt(s), "
            f"{self.pending - 1} more queued",
        )
        if self._cancelled:
            return
        t0 = time.time()
        try:
            proc = await self._spawn(argv, job["diffusion_pipe_dir"])
        except OSError as e:
            await self._fail(out_dir, epoch, f"could not launch sampler: {e}")
            return
        self._proc = proc
        if self._cancelled:
            # cancel() ran while spawn() was in flight, before _proc was set.
            self._kill(proc)
        try:
            if proc.stdout is not None:
                async for raw in proc.stdout:
                    await self._on_log(raw.decode("utf-8", errors="replace").rstrip("\n"))
            rc = await proc.wait()
        finally:
            self._proc = None
        if self._cancelled:
            await self._on_log(f"sampling epoch {epoch}: cancelled")
            return
        manifest = out_dir / "manifest.json"
        if not manifest.is_file():
            await self._fail(out_dir, epoch, f"sampler exited with code {rc}")
            return
        try:
            error = json.loads(manifest.read_text()).get("error")
        except (OSError, ValueError):
            error = "unreadable manifest.json"
        if error:
            await self._on_log(f"sampling epoch {epoch} failed: {error}")
            return
        await self._on_log(f"sampling epoch {epoch}: done in {time.time() - t0:.0f}s")
        await self._on_sampled(epoch)

    async def _fail(self, out_dir: Path, epoch: int, error: str) -> None:
        await self._on_log(f"sampling epoch {epoch} failed: {error}")
        tmp = out_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps({"epoch": epoch, "images": [], "error": error}, indent=2))
        tmp.replace(out_dir / "manifest.json")

"""Async subprocess manager for LoRA training runs.

One :class:`TrainingManager` per server instance handles all projects but
allows at most one active run at a time (training is GPU-bound and would
fight the extraction queue if both ran in parallel). It owns:

* The currently running ``asyncio.subprocess.Process`` (if any).
* A bounded in-memory ring buffer of stdout/stderr lines for the UI's log
  panel (kept small — the on-disk ``run.log`` is the source of truth).
* The persisted run state, so the UI can show "last run was XYZ, stopped at
  epoch N" after a server restart.

Events broadcast on the WebSocket:

* ``training.status`` — payload contains the full status snapshot. Sent on
  every state transition (start, stop, finish) plus periodically while
  training is running.
* ``training.log`` — payload ``{slug, line, stream}`` for one log line
  (``stream`` is ``"sample"`` for sample-generator output).
* ``training.sample`` — payload ``{slug, run_name, epoch}`` once an epoch's
  sample images are written.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import shlex
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neme_anima import training as training_lib
from neme_anima.server.events import Broadcaster, Event
from neme_anima.server.sampling_runner import SampleScheduler, SpawnFn
from neme_anima.storage.project import Project, TrainingConfig

logger = logging.getLogger(__name__)

# Keep at most this many log lines in-memory for the UI. The on-disk run.log
# captures everything; this buffer is just for fast reconnects.
LOG_BUFFER_MAX = 2000

# Guards the persisted-state file I/O. The pump tasks, the start/stop
# handlers, and status() reads all share one event loop, but the
# tmp-write-then-replace in _persist_state must stay atomic against any
# future to_thread offload — and the lock documents the invariant.
_STATE_LOCK = threading.Lock()


@dataclass
class RunState:
    """Persisted snapshot for one training run.

    Stored at ``project.training_state_path``; rewritten atomically on each
    state transition. Survives server restarts; on startup we reload it so
    the UI can pick up where it left off.
    """

    project_slug: str
    run_dir: str
    # "starting" | "running" | "sampling" | "stopping" | "stopped" | "finished" | "failed"
    status: str
    started_at: str
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    # Best-effort progress parsed from stdout. The trainer prints epoch and
    # step counts; we extract them with simple regexes (see _parse_progress).
    epoch: int | None = None
    step: int | None = None
    loss: float | None = None
    last_log_line: str = ""
    # The checkpoint we resumed from, if any. Useful for the UI to show a
    # "resumed from epoch 20" pill on the run.
    resumed_from: str | None = None
    # Flag that the user clicked Stop. Distinguishes voluntary stop from
    # a crash (status=failed).
    stop_requested: bool = False
    # Snapshot of cfg.epochs at run-launch so the UI can render a progress
    # bar without having to query the live config (which the user may have
    # edited mid-run).
    total_epochs: int | None = None
    # Sample generation progress while a sampler job is queued/running:
    # ``{"epoch": int | None, "pending": int}``; None when idle.
    sampling: dict | None = None


_PROGRESS_PATTERNS = [
    # diffusion-pipe deepspeed prints lines like:
    #   "epoch: 5, step: 1234, loss: 0.1234"
    # plus tqdm-ish "Epoch 5/40" forms. Extract epoch/step/loss best-effort.
    ("epoch", r"epoch[\s:=]+(\d+)"),
    ("step",  r"\b(?:global[_ ]?step|step)[\s:=]+(\d+)"),
    ("loss",  r"\bloss[\s:=]+([0-9]*\.?[0-9]+)"),
]

_PROG_RE = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _PROGRESS_PATTERNS]


def _parse_progress(line: str) -> dict[str, Any]:
    """Pull epoch/step/loss out of a log line, if present."""
    out: dict[str, Any] = {}
    for name, rx in _PROG_RE:
        m = rx.search(line)
        if not m:
            continue
        try:
            out[name] = float(m.group(1)) if name == "loss" else int(m.group(1))
        except ValueError:
            pass
    return out


class TrainingManager:
    """Single-active-run training coordinator."""

    def __init__(
        self,
        *,
        broadcaster: Broadcaster,
        sampler_spawn: SpawnFn | None = None,
    ) -> None:
        self._broadcaster = broadcaster
        self._sampler_spawn = sampler_spawn
        self._lock = asyncio.Lock()
        # The active run, if any. Only one at a time.
        self._project: Project | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._state: RunState | None = None
        self._tasks: list[asyncio.Task] = []
        self._log_buffer: deque[dict[str, Any]] = deque(maxlen=LOG_BUFFER_MAX)
        self._log_path: Path | None = None
        self._log_file: Any = None  # text-mode file handle
        # Tracks the TrainingConfig snapshot at run time so retention pruning
        # uses the correct ``keep_last_n_checkpoints`` even if the user edits
        # config mid-run.
        self._cfg_snapshot: TrainingConfig | None = None
        # The active run's sample-image scheduler (see sampling_runner).
        self._sampler: SampleScheduler | None = None
        # True for the whole of _wait_for_exit (log drain, sampling drain,
        # pruning). The run still counts as active so nothing else grabs the GPU.
        self._finalizing = False

    # ----- public API ------------------------------------------------------

    @property
    def active_slug(self) -> str | None:
        return self._state.project_slug if self._state and self._is_active() else None

    def status(self, project: Project) -> dict:
        """Snapshot for the API/UI. ``running`` is True iff this project owns
        the active subprocess; otherwise we report the last persisted state."""
        is_ours = (
            self._is_active() and self._project is not None and self._project.slug == project.slug
        )
        if is_ours and self._state is not None:
            persisted = self._state
        else:
            persisted = _load_persisted_state(project)
        running = is_ours
        return {
            "slug": project.slug,
            "running": running,
            "global_active_slug": self.active_slug,
            "state": _state_to_dict(persisted) if persisted else None,
            "log_lines": list(self._log_buffer) if is_ours else [],
        }

    def get_log_buffer(self, project: Project) -> list[dict[str, Any]]:
        if self._is_active() and self._project and self._project.slug == project.slug:
            return list(self._log_buffer)
        return []

    async def start(
        self,
        project: Project,
        *,
        resume_from_checkpoint: str | None = None,
        run_dir_name: str | None = None,
    ) -> dict:
        """Kick off a new training run for ``project``.

        Pass ``resume_from_checkpoint`` (a checkpoint *name*) to continue a
        prior run. ``run_dir_name`` reuses an existing run directory; the
        common case is to pass the directory of the run whose checkpoint
        we're resuming from.
        """
        async with self._lock:
            if self._is_active():
                raise RuntimeError(
                    f"another training run is already active "
                    f"(project={self.active_slug})",
                )
            problems = training_lib.validate_for_run(project.training)
            if problems:
                raise RuntimeError("; ".join(problems))

            # Pick / create the run directory.
            project.training_dir.mkdir(parents=True, exist_ok=True)
            project.training_runs_dir.mkdir(parents=True, exist_ok=True)
            if run_dir_name:
                run_dir = project.training_runs_dir / run_dir_name
                run_dir.mkdir(parents=True, exist_ok=True)
            else:
                label = "resume" if resume_from_checkpoint else project.training.preset
                run_dir = training_lib.new_run_dir(project, label=label)

            # Materialize the training dataset as symlinks under the run
            # directory: each kept frame's image points at its crop derivative
            # when one exists, while the sidecar always points at the
            # original `.txt` (so "edit tags on the original; train on the
            # crop" is the on-disk reality the trainer sees). Rebuilt on
            # every start so a resume picks up the latest crop set.
            dataset_dir = run_dir / "dataset"
            staging = training_lib.build_dataset_staging(project, dataset_dir)
            logger.info(
                "training: staged dataset at %s (%d images, %d cropped, %d missing txt)",
                staging["dest"], staging["images"],
                staging["with_crop"], staging["missing_txt"],
            )

            # Render TOML files into the run dir.
            dataset_toml = run_dir / "dataset.toml"
            run_toml = run_dir / "run.toml"
            dataset_toml.write_text(training_lib.render_dataset_toml(
                project, dataset_root=dataset_dir,
            ))
            run_toml.write_text(training_lib.render_run_toml(
                project,
                run_dir=run_dir,
                dataset_toml_path=dataset_toml,
                resume_from_checkpoint=resume_from_checkpoint,
            ))

            # Build argv. Run with cwd=diffusion-pipe-dir so train.py imports
            # its sibling modules correctly.
            argv = training_lib.build_launcher_argv(project.training, run_toml=run_toml)
            cwd = str(Path(project.training.diffusion_pipe_dir).expanduser().resolve())

            # Deep copy: the sampler reads it for the whole run, so mid-run
            # config edits must not leak in.
            self._cfg_snapshot = copy.deepcopy(project.training)
            self._project = project
            self._log_buffer.clear()
            self._log_path = run_dir / "run.log"
            self._log_file = open(self._log_path, "a", encoding="utf-8", buffering=1)
            self._log_file.write(
                f"\n=== run start {datetime.now(UTC).isoformat()} "
                f"argv={shlex.join(argv)} cwd={cwd}\n",
            )

            self._state = RunState(
                project_slug=project.slug,
                run_dir=str(run_dir.resolve()),
                status="starting",
                started_at=datetime.now(UTC).isoformat(),
                resumed_from=resume_from_checkpoint,
                total_epochs=project.training.epochs,
            )
            _persist_state(project, self._state)

            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    # New process group so we can signal the whole tree on stop.
                    start_new_session=True,
                )
            except FileNotFoundError as e:
                self._state.status = "failed"
                self._state.error = f"launcher not found: {e}"
                self._state.finished_at = datetime.now(UTC).isoformat()
                _persist_state(project, self._state)
                self._close_log_file()
                self._project = None
                self._cfg_snapshot = None
                raise

            self._state.pid = self._proc.pid
            self._state.status = "running"
            _persist_state(project, self._state)

            # Spawn pump tasks.
            self._tasks = [
                asyncio.create_task(self._pump_stream(self._proc.stdout, "stdout")),
                asyncio.create_task(self._pump_stream(self._proc.stderr, "stderr")),
                asyncio.create_task(self._wait_for_exit()),
            ]

            python = training_lib.diffusion_pipe_python(project.training.diffusion_pipe_dir)
            self._sampler = SampleScheduler(
                run_dir=run_dir,
                cfg=self._cfg_snapshot,
                python=str(python) if python else "python",
                on_log=self._on_sample_log,
                on_change=self._on_sample_change,
                on_sampled=self._on_sampled,
                spawn=self._sampler_spawn,
            )
            self._sampler.start()

            await self._broadcast_status()
            return self.status(project)

    async def stop(self, project: Project) -> dict:
        """Signal the active run to terminate. Returns the new status."""
        async with self._lock:
            if not self._is_active() or self._project is None or self._project.slug != project.slug:
                raise RuntimeError(
                    "no active training run for this project",
                )
            assert self._proc is not None and self._state is not None
            # Local refs: _wait_for_exit may finish (and clear these) while we
            # await below.
            proc = self._proc
            sampler = self._sampler
            self._state.stop_requested = True
            self._state.status = "stopping"
            _persist_state(project, self._state)
            await self._broadcast_status()

            if sampler is not None:
                await sampler.cancel()

            # Kill the whole process group — deepspeed spawns child processes.
            # Skipped when the trainer already exited (stop during the drain).
            if proc.returncode is None:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    # Already gone, or process-group lookup failed; fall back.
                    try:
                        proc.terminate()
                    except ProcessLookupError:
                        pass

        # Outside the lock — wait for the wait task to mark things finished.
        await asyncio.sleep(0)  # let scheduler progress
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except TimeoutError:
                # Force-kill if it didn't go down cleanly.
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
        return self.status(project)

    async def shutdown(self) -> None:
        """Best-effort: kill the active run on server shutdown.

        We don't wait long — the lifespan tear-down should not block the
        user's Ctrl-C. The on-disk state already captures "running with
        pid=X" so the next startup can clean up.
        """
        if self._state is not None and self._is_active():
            # An interrupted run (or sampling drain) must end "stopped", not
            # "finished" — _wait_for_exit keys off this flag.
            self._state.stop_requested = True
        if self._sampler is not None:
            await self._sampler.cancel()
        proc = self._proc
        if proc is None or proc.returncode is not None:
            self._close_log_file()
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except TimeoutError:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        self._close_log_file()

    # ----- internals -------------------------------------------------------

    def _is_active(self) -> bool:
        return self._finalizing or (self._proc is not None and self._proc.returncode is None)

    async def _pump_stream(
        self,
        stream: asyncio.StreamReader | None,
        kind: str,
    ) -> None:
        if stream is None:
            return
        try:
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    line = repr(raw)
                await self._emit_log(line, kind)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("training log pump (%s) crashed", kind)

    async def _emit_log(self, line: str, kind: str) -> None:
        """Record one log line (buffer, run.log, WS). Only trainer output
        feeds the epoch/step/loss parser — sampler lines ("sample") would
        otherwise be misread as progress."""
        ts = time.time()
        self._log_buffer.append({"t": ts, "stream": kind, "line": line})
        if self._log_file is not None:
            try:
                self._log_file.write(f"[{kind}] {line}\n")
            except Exception:
                pass
        if self._state is not None:
            if kind != "sample":
                self._state.last_log_line = line
                parsed = _parse_progress(line)
                if parsed:
                    if "epoch" in parsed:
                        self._state.epoch = parsed["epoch"]
                    if "step" in parsed:
                        self._state.step = parsed["step"]
                    if "loss" in parsed:
                        self._state.loss = parsed["loss"]
                    if self._project is not None:
                        _persist_state(self._project, self._state)
        await self._broadcaster.publish(Event(
            type="training.log",
            payload={
                "slug": self._state.project_slug if self._state else "",
                "stream": kind,
                "line": line,
                "t": ts,
            },
        ))

    async def _on_sample_log(self, line: str) -> None:
        await self._emit_log(line, "sample")

    async def _on_sample_change(self) -> None:
        if self._state is None or self._sampler is None:
            return
        s = self._sampler
        self._state.sampling = (
            {"epoch": s.current_epoch, "pending": s.pending}
            if s.current_epoch is not None else None
        )
        if self._project is not None:
            _persist_state(self._project, self._state)
        await self._broadcast_status()

    async def _on_sampled(self, epoch: int) -> None:
        if self._state is None:
            return
        await self._broadcaster.publish(Event(
            type="training.sample",
            payload={
                "slug": self._state.project_slug,
                "run_name": Path(self._state.run_dir).name,
                "epoch": epoch,
            },
        ))

    async def _wait_for_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        # Set while the trainer is still alive: once it exits, returncode is
        # set before the pumps below finish draining, and the run must not
        # look idle (start() could grab the GPU) during that window.
        self._finalizing = True
        try:
            rc = await proc.wait()
        except asyncio.CancelledError:
            self._finalizing = False
            raise
        finally:
            # Make sure both stream pumps drain even if wait() returned early.
            for t in self._tasks:
                if t is not asyncio.current_task() and not t.done():
                    try:
                        await asyncio.wait_for(t, timeout=2.0)
                    except (TimeoutError, asyncio.CancelledError):
                        pass

        state = self._state
        project = self._project
        sampler = self._sampler
        # Everything below runs under try/finally: an exception here must not
        # leave _finalizing stuck True, or every future start() would be
        # refused as "already active" until the server restarts.
        try:
            if state is not None:
                state.exit_code = rc
            # Render samples for every LoRA not sampled yet — before pruning
            # can delete one. A user Stop already cancelled the scheduler.
            if sampler is not None and state is not None and not state.stop_requested:
                if sampler.pending_epochs():
                    state.status = "sampling"
                    if project is not None:
                        _persist_state(project, state)
                    await self._broadcast_status()
                try:
                    await sampler.drain()
                except asyncio.CancelledError:
                    # Our own cancellation leaves the scheduler's poll task
                    # running; kill it so no sampler outlives the manager.
                    await sampler.cancel()
                    raise
                except Exception:
                    logger.exception("training: sample drain failed")
            if state is not None:
                state.sampling = None
                state.finished_at = datetime.now(UTC).isoformat()
                if state.stop_requested:
                    state.status = "stopped"
                elif rc == 0:
                    state.status = "finished"
                else:
                    state.status = "failed"
                    if not state.error:
                        state.error = f"trainer exited with code {rc}"
                if project is not None:
                    _persist_state(project, state)

            # Apply checkpoint retention now that the run is over, then tag
            # the remaining LoRA files with neme-anima provenance metadata.
            if state is not None and project is not None and self._cfg_snapshot is not None:
                try:
                    # Stop/shutdown may interrupt sampling: never let
                    # retention delete a LoRA whose samples weren't rendered.
                    deleted = training_lib.prune_checkpoints(
                        Path(state.run_dir),
                        keep_last_n=self._cfg_snapshot.keep_last_n_checkpoints,
                        protect=sampler.pending_epochs() if sampler is not None else (),
                    )
                    if deleted:
                        logger.info(
                            "training: pruned %d checkpoints from %s: %s",
                            len(deleted), state.run_dir, deleted,
                        )
                except Exception:
                    logger.exception("training: prune_checkpoints failed")
                try:
                    tagged = training_lib.tag_run_safetensors(
                        project, Path(state.run_dir),
                    )
                    if tagged:
                        logger.info(
                            "training: tagged %d LoRA safetensors in %s",
                            len(tagged), state.run_dir,
                        )
                except Exception:
                    logger.exception("training: tag_run_safetensors failed")
        finally:
            self._finalizing = False
            try:
                await self._broadcast_status()
            except Exception:
                logger.exception("training: final status broadcast failed")
            self._close_log_file()
            self._proc = None
            self._project = None
            self._cfg_snapshot = None
            self._sampler = None
            self._tasks = []

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
        self._log_file = None

    async def _broadcast_status(self) -> None:
        if self._state is None or self._project is None:
            return
        await self._broadcaster.publish(Event(
            type="training.status",
            payload={
                "slug": self._state.project_slug,
                "running": self._is_active(),
                "state": _state_to_dict(self._state),
            },
        ))


# ----- state persistence helpers --------------------------------------------


def _state_to_dict(state: RunState) -> dict:
    return {
        "project_slug": state.project_slug,
        "run_dir": state.run_dir,
        "run_name": Path(state.run_dir).name,
        "status": state.status,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "pid": state.pid,
        "exit_code": state.exit_code,
        "error": state.error,
        "epoch": state.epoch,
        "step": state.step,
        "loss": state.loss,
        "last_log_line": state.last_log_line,
        "resumed_from": state.resumed_from,
        "stop_requested": state.stop_requested,
        "total_epochs": state.total_epochs,
        "sampling": state.sampling,
    }


def _persist_state(project: Project, state: RunState) -> None:
    with _STATE_LOCK:
        project.training_dir.mkdir(parents=True, exist_ok=True)
        tmp = project.training_state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_state_to_dict(state), indent=2))
        tmp.replace(project.training_state_path)


def _load_persisted_state(project: Project) -> RunState | None:
    p = project.training_state_path
    with _STATE_LOCK:
        if not p.is_file():
            return None
        try:
            raw = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("training: corrupt persisted state at %s: %s", p, exc)
            return None
    try:
        return RunState(
            project_slug=raw.get("project_slug") or project.slug,
            run_dir=raw["run_dir"],
            status=raw.get("status") or "stopped",
            started_at=raw.get("started_at") or "",
            finished_at=raw.get("finished_at"),
            pid=raw.get("pid"),
            exit_code=raw.get("exit_code"),
            error=raw.get("error"),
            epoch=raw.get("epoch"),
            step=raw.get("step"),
            loss=raw.get("loss"),
            last_log_line=raw.get("last_log_line", ""),
            resumed_from=raw.get("resumed_from"),
            stop_requested=bool(raw.get("stop_requested", False)),
            total_epochs=raw.get("total_epochs"),
            sampling=raw.get("sampling"),
        )
    except KeyError as exc:
        logger.warning(
            "training: persisted state at %s is missing required key %s", p, exc,
        )
        return None

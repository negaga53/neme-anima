"""WebSocket events + broadcaster.

A `Broadcaster` is a fan-out queue: each subscriber gets its own asyncio.Queue,
and `publish()` puts the event on every live subscriber's queue. WebSocket
endpoints subscribe on connect and unsubscribe on disconnect.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    """One server-pushed event over the WebSocket channel.

    `type` is one of:
      - "queue.update"     payload: { queue: [...] }                — queue contents changed
      - "job.progress"     payload: { job_id, project, pct, stage } — extraction progress
      - "job.frame"        payload: { job_id, filename, kept }      — a new frame was written
      - "job.log"          payload: { job_id, level, line }         — log line emitted by the worker
      - "job.done"         payload: { job_id, kept, rejected }      — job finished
      - "training.status"  payload: { slug, running, state }        — training run state changed
      - "training.log"     payload: { slug, stream, line, t }       — one trainer log line
      - "training.sample"  payload: { slug, run_name, epoch }       — sample images written
    """
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> Event:
        d = json.loads(s)
        return cls(type=d["type"], payload=d.get("payload", {}))


class Broadcaster:
    """Fan-out queue over asyncio.Queue subscribers."""

    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[Event]] = []

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue()
        self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        try:
            self._subs.remove(q)
        except ValueError:
            pass  # Already gone — no-op.

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    async def publish(self, event: Event) -> None:
        for q in list(self._subs):  # snapshot so unsubscribe-during-publish is safe
            await q.put(event)

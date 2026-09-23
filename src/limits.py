"""Concurrency limits for a public deployment of the prototype.

A single connection cap cannot express the load this service actually sees, so
this module shapes the limits like the work:

* **Live streams** are capped per process and per client address. One stream is
  one viewer watching one game, and it holds both a Server-Sent Events response
  and an outbound Lichess connection open for the length of that game. The
  per-client cap matters more than the total: without it one visitor can open
  dozens of streams and spend the deployment's Lichess API budget by itself.
* **Analysis requests** are capped by how many may be *waiting*, not by how many
  may run. One rating inference is a few hundred milliseconds of synchronous
  torch work, so requests serialize whatever the cap says. Bounding the queue
  means an ordinary burst waits its turn while a flood is shed quickly, instead
  of building a backlog that times out every request in it.
* **Page and static-asset requests are not counted at all.** They are cheap, and
  one visitor opening the page is already a dozen of them: counting those is what
  makes a site return 503 for its own stylesheet under mild load.

Every limit is read from the environment, so a deployment retunes without a code
change. Set any of them to 0 to disable that limit.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

# One stream is one viewer. The default is deliberately well above the number of
# concurrent inferences the CPU can sustain, because a stream spends nearly all
# of its life idle, waiting for the opponent to move.
MAX_LIVE_STREAMS = int(os.environ.get("RATINGNET_MAX_LIVE_STREAMS", "20"))
MAX_LIVE_STREAMS_PER_CLIENT = int(os.environ.get("RATINGNET_MAX_LIVE_STREAMS_PER_CLIENT", "2"))

# How many analysis requests may run at once, and how many may queue behind them.
MAX_ANALYSIS_RUNNING = int(os.environ.get("RATINGNET_MAX_ANALYSIS_RUNNING", "2"))
MAX_ANALYSIS_WAITING = int(os.environ.get("RATINGNET_MAX_ANALYSIS_WAITING", "12"))


class LiveStreamLimit(Exception):
    """Raised when a live stream would exceed a configured cap.

    The message is written for a visitor, because the SSE endpoints report setup
    problems to the browser as an error event carrying exactly this text.
    """


class AnalysisBusy(Exception):
    """Raised when the analysis queue is full and the request should be shed."""


class LiveStreamLimiter:
    """Counts live streams in total and per client address."""

    def __init__(self, total: int = MAX_LIVE_STREAMS, per_client: int = MAX_LIVE_STREAMS_PER_CLIENT):
        self._total_cap = total
        self._per_client_cap = per_client
        self._active = 0
        self._per_client: dict[str, int] = {}

    @property
    def active(self) -> int:
        return self._active

    @contextmanager
    def hold(self, client: str | None) -> Iterator[None]:
        """Reserve one stream slot for the body of the `with`, then release it.

        Releasing happens in a `finally`, so a viewer closing the tab, a dropped
        connection and a finished game all return the slot the same way.
        """
        key = client or "unknown"
        if self._total_cap and self._active >= self._total_cap:
            raise LiveStreamLimit(
                f"This deployment is already following {self._active} live games, which is its limit. "
                "Try again in a few minutes, or analyse a finished game by pasting its PGN."
            )
        if self._per_client_cap and self._per_client.get(key, 0) >= self._per_client_cap:
            raise LiveStreamLimit(
                f"You already have {self._per_client[key]} live games open, which is the limit per "
                "visitor. Close one of them before starting another."
            )
        self._active += 1
        self._per_client[key] = self._per_client.get(key, 0) + 1
        try:
            yield
        finally:
            self._active -= 1
            remaining = self._per_client.get(key, 1) - 1
            if remaining > 0:
                self._per_client[key] = remaining
            else:
                # Drop the key rather than leaving a zero behind, so the mapping
                # cannot grow without bound across many one-off visitors.
                self._per_client.pop(key, None)


class AnalysisQueue:
    """Bounds how many analysis requests run, and how many wait to run."""

    def __init__(self, running: int = MAX_ANALYSIS_RUNNING, waiting: int = MAX_ANALYSIS_WAITING):
        self._waiting_cap = waiting
        self._semaphore = asyncio.Semaphore(running) if running else None
        self._waiting = 0

    @property
    def waiting(self) -> int:
        return self._waiting

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Wait for a turn, or raise `AnalysisBusy` if the queue is already full."""
        if self._semaphore is None:
            yield
            return
        if self._waiting_cap and self._waiting >= self._waiting_cap:
            raise AnalysisBusy(
                f"{self._waiting} analysis requests are already queued. Please retry shortly."
            )
        self._waiting += 1
        try:
            async with self._semaphore:
                yield
        finally:
            self._waiting -= 1

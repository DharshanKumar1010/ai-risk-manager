"""The live scoring feed's broadcast fan-out — transport-agnostic on purpose.

Every scored decision that commits is published here, and every connected analyst socket
reads from a queue this module hands out. Kept as a plain asyncio primitive rather than
anything websocket-specific so the transport (``app/api/feed.py``'s WS route today) is a thin
adapter over it -- if the WebSocket route ever misbehaves in a deployed environment, an SSE
route reading the same :class:`FeedBroadcaster` is a small addition, not a rewrite.

**In-process, and that is a stated limit, not an oversight.** ``backend/Dockerfile`` runs a
single ``uvicorn`` process, so one broadcaster in one process's memory is correct today. If a
later phase adds ``--workers 2`` for throughput, this broadcaster silently shows each socket
only the decisions its own worker happened to handle -- half the traffic, with no error and no
warning. The fix is Redis pub/sub, and ``Settings.redis_url``'s own docstring already promises
"rate limiting and the live scoring feed" for exactly this reason. Not built now because a
single worker is what actually runs; named here and in ``BUILD_LOG.md`` so it is a decision
carried forward rather than a gap discovered later.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Queue depth per subscriber. Small: this is a live feed, not a durable log -- a socket that
#: falls this far behind is better served a "you missed some" marker than a growing backlog.
SUBSCRIBER_QUEUE_SIZE = 100

#: Concurrent connections admitted per principal. Bounds one analyst opening many tabs (or a
#: misbehaving client reconnect-looping) from growing the broadcaster's subscriber set without
#: limit.
MAX_CONNECTIONS_PER_PRINCIPAL = 4


class FeedBroadcaster:
    """Fan out published events to every subscribed queue.

    A slow or absent subscriber must never be able to block :func:`publish` -- in particular,
    it must never be able to slow down ``POST /score``'s commit, which is what publishes here.
    ``put_nowait`` plus a bounded queue is what gives that guarantee: a full queue drops its
    oldest event to make room for the newest, rather than the publisher blocking on a slow
    reader. See :meth:`publish` for why that swap is a single evict-then-insert rather than
    also queuing a "you missed some" marker.
    """

    def __init__(self) -> None:
        """Start with no subscribers."""
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._connections_by_principal: dict[str, int] = {}

    def subscriber_count(self) -> int:
        """Return how many sockets are currently subscribed. For diagnostics/tests."""
        return len(self._subscribers)

    def admit(self, principal_subject: str) -> bool:
        """Return whether another connection may be opened for this principal.

        Call before creating the queue in :meth:`subscribe`; a caller that gets ``False``
        should refuse the connection (websocket close 1008) rather than subscribing anyway.
        """
        return self._connections_by_principal.get(principal_subject, 0) < (
            MAX_CONNECTIONS_PER_PRINCIPAL
        )

    def subscribe(self, principal_subject: str) -> "asyncio.Queue[dict[str, Any]]":
        """Register a new subscriber and return its queue.

        Raises:
            RuntimeError: If :meth:`admit` would have refused this principal. Callers must
                check :meth:`admit` first; this is a defensive re-check, not the primary gate.
        """
        if not self.admit(principal_subject):
            raise RuntimeError(f"{principal_subject} already holds the maximum connections")
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        self._connections_by_principal[principal_subject] = (
            self._connections_by_principal.get(principal_subject, 0) + 1
        )
        return queue

    def unsubscribe(self, principal_subject: str, queue: "asyncio.Queue[dict[str, Any]]") -> None:
        """Remove a subscriber. Safe to call more than once for the same queue."""
        self._subscribers.discard(queue)
        remaining = self._connections_by_principal.get(principal_subject, 0) - 1
        if remaining > 0:
            self._connections_by_principal[principal_subject] = remaining
        else:
            self._connections_by_principal.pop(principal_subject, None)

    def publish(self, event: dict[str, Any]) -> None:
        """Push ``event`` onto every subscriber's queue, dropping the oldest on overflow.

        A full queue is handled by evicting exactly one old event and inserting the new one --
        a straight swap, so the queue's size never changes here and a second `QueueFull` can
        never occur from this call. An earlier version instead inserted a separate "you missed
        one" marker alongside the new event, which needed *two* free slots after freeing only
        one; on a queue that had been full for a while, the marker consumed the freed slot and
        the real event was the thing dropped -- the opposite of "never lose the newest
        decision", which is the property this method exists to guarantee. The client-visible
        cost of the simpler policy is that a lagging subscriber is not told a specific count of
        skipped events; it still eventually sees the current picture, which for a live feed is
        the property that actually matters.

        Never awaits, never raises for a slow subscriber -- the one property the caller
        (``POST /score``, after its commit) actually depends on.
        """
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    logger.warning("feed subscriber queue could not accept an event; skipped")

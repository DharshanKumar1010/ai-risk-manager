"""``app.core.feed.FeedBroadcaster`` — the live feed's in-process fan-out.

The property that matters most here is the one a slow-consumer test proves directly: a full
subscriber queue must never make :meth:`publish` block or raise. ``POST /score`` calls publish
after its commit, on the same coroutine that is about to return a response -- a broadcaster
that could stall there would put the live feed's failure mode on the scoring path's latency.
"""

import asyncio

import pytest

from app.core.feed import MAX_CONNECTIONS_PER_PRINCIPAL, SUBSCRIBER_QUEUE_SIZE, FeedBroadcaster


class TestSubscribeAndPublish:
    def test_a_subscriber_receives_a_published_event(self) -> None:
        broadcaster = FeedBroadcaster()
        queue = broadcaster.subscribe("analyst-1")
        broadcaster.publish({"type": "decision", "audit_id": 1})
        assert queue.get_nowait() == {"type": "decision", "audit_id": 1}

    def test_every_subscriber_receives_the_same_event(self) -> None:
        broadcaster = FeedBroadcaster()
        first = broadcaster.subscribe("analyst-1")
        second = broadcaster.subscribe("analyst-2")
        broadcaster.publish({"type": "decision", "audit_id": 1})
        assert first.get_nowait() == second.get_nowait() == {"type": "decision", "audit_id": 1}

    def test_publishing_with_no_subscribers_does_not_raise(self) -> None:
        FeedBroadcaster().publish({"type": "decision"})

    def test_unsubscribing_stops_further_events(self) -> None:
        broadcaster = FeedBroadcaster()
        queue = broadcaster.subscribe("analyst-1")
        broadcaster.unsubscribe("analyst-1", queue)
        broadcaster.publish({"type": "decision"})
        assert queue.empty()

    def test_unsubscribing_twice_is_safe(self) -> None:
        broadcaster = FeedBroadcaster()
        queue = broadcaster.subscribe("analyst-1")
        broadcaster.unsubscribe("analyst-1", queue)
        broadcaster.unsubscribe("analyst-1", queue)  # must not raise


class TestConnectionLimit:
    """One principal opening unbounded connections must not grow the subscriber set forever."""

    def test_admits_up_to_the_limit(self) -> None:
        broadcaster = FeedBroadcaster()
        for _ in range(MAX_CONNECTIONS_PER_PRINCIPAL):
            assert broadcaster.admit("analyst-1")
            broadcaster.subscribe("analyst-1")

    def test_refuses_beyond_the_limit(self) -> None:
        broadcaster = FeedBroadcaster()
        for _ in range(MAX_CONNECTIONS_PER_PRINCIPAL):
            broadcaster.subscribe("analyst-1")
        assert not broadcaster.admit("analyst-1")

    def test_subscribe_raises_when_admit_would_have_refused(self) -> None:
        broadcaster = FeedBroadcaster()
        for _ in range(MAX_CONNECTIONS_PER_PRINCIPAL):
            broadcaster.subscribe("analyst-1")
        with pytest.raises(RuntimeError):
            broadcaster.subscribe("analyst-1")

    def test_a_different_principal_is_unaffected_by_anothers_limit(self) -> None:
        broadcaster = FeedBroadcaster()
        for _ in range(MAX_CONNECTIONS_PER_PRINCIPAL):
            broadcaster.subscribe("analyst-1")
        assert broadcaster.admit("analyst-2")

    def test_unsubscribing_frees_a_slot(self) -> None:
        broadcaster = FeedBroadcaster()
        queues = [broadcaster.subscribe("analyst-1") for _ in range(MAX_CONNECTIONS_PER_PRINCIPAL)]
        broadcaster.unsubscribe("analyst-1", queues[0])
        assert broadcaster.admit("analyst-1")


class TestSlowSubscriberCannotBlockPublish:
    """The property POST /score's commit path actually depends on."""

    def test_publish_does_not_raise_when_a_queue_is_full(self) -> None:
        broadcaster = FeedBroadcaster()
        queue = broadcaster.subscribe("analyst-1")
        for i in range(SUBSCRIBER_QUEUE_SIZE + 5):
            broadcaster.publish({"type": "decision", "audit_id": i})  # must never raise
        assert queue.qsize() <= SUBSCRIBER_QUEUE_SIZE

    def test_a_full_queue_drops_the_oldest_event_to_admit_the_newest(self) -> None:
        broadcaster = FeedBroadcaster()
        queue: asyncio.Queue = broadcaster.subscribe("analyst-1")
        for i in range(SUBSCRIBER_QUEUE_SIZE):
            broadcaster.publish({"type": "decision", "audit_id": i})
        broadcaster.publish({"type": "decision", "audit_id": "overflow"})

        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())
        # The queue never grows past its bound: exactly one evict per overflowing publish, so
        # the oldest event is gone and the newest made it through.
        assert len(drained) == SUBSCRIBER_QUEUE_SIZE
        assert {"type": "decision", "audit_id": 0} not in drained
        assert {"type": "decision", "audit_id": "overflow"} in drained

    def test_one_slow_subscriber_does_not_affect_a_healthy_one(self) -> None:
        broadcaster = FeedBroadcaster()
        slow = broadcaster.subscribe("slow-analyst")
        healthy = broadcaster.subscribe("healthy-analyst")

        for i in range(SUBSCRIBER_QUEUE_SIZE + 10):
            broadcaster.publish({"type": "decision", "audit_id": i})
            if i < 3:
                # The healthy subscriber keeps draining; the slow one never does. Both still
                # end up at the queue's bound: `asyncio.Queue(maxsize=...)` enforces that cap
                # itself regardless of the eviction policy, so briefly draining early does not
                # let the healthy queue end up any less full than the one that never drained.
                healthy.get_nowait()

        assert slow.qsize() == SUBSCRIBER_QUEUE_SIZE
        assert healthy.qsize() == SUBSCRIBER_QUEUE_SIZE


class TestSubscriberCount:
    def test_reflects_active_subscriptions(self) -> None:
        broadcaster = FeedBroadcaster()
        assert broadcaster.subscriber_count() == 0
        queue = broadcaster.subscribe("analyst-1")
        assert broadcaster.subscriber_count() == 1
        broadcaster.unsubscribe("analyst-1", queue)
        assert broadcaster.subscriber_count() == 0

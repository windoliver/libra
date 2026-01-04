"""Tests for the MessageBus."""

from __future__ import annotations

import asyncio
import time

import pytest

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus, MessageBusConfig


class TestSubscription:
    """Tests for subscription management."""

    def test_subscribe_returns_id(self) -> None:
        """Subscribe should return a unique ID."""
        bus = MessageBus()

        async def handler(event: Event) -> None:
            pass

        id1 = bus.subscribe(EventType.TICK, handler)
        id2 = bus.subscribe(EventType.TICK, handler)

        assert id1 != id2
        assert isinstance(id1, int)

    def test_unsubscribe_by_id(self) -> None:
        """Unsubscribe should remove handler."""
        bus = MessageBus()

        async def handler(event: Event) -> None:
            pass

        sub_id = bus.subscribe(EventType.TICK, handler)
        assert bus.stats["handlers"] == 1

        result = bus.unsubscribe(sub_id)
        assert result is True
        assert bus.stats["handlers"] == 0

    def test_unsubscribe_unknown_id(self) -> None:
        """Unsubscribe with unknown ID should return False."""
        bus = MessageBus()
        result = bus.unsubscribe(99999)
        assert result is False

    def test_subscribe_with_filter(self) -> None:
        """Filter function should control event delivery."""
        bus = MessageBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        def only_btc(event: Event) -> bool:
            return event.payload.get("symbol") == "BTC/USDT"

        bus.subscribe(EventType.TICK, handler, filter_fn=only_btc)

        # Publish events
        eth_event = Event.create(EventType.TICK, "test", {"symbol": "ETH/USDT"})
        btc_event = Event.create(EventType.TICK, "test", {"symbol": "BTC/USDT"})

        bus.publish(eth_event)
        bus.publish(btc_event)

        # Dispatch synchronously for testing
        async def dispatch() -> None:
            await bus._dispatch_batch()

        asyncio.run(dispatch())

        # Only BTC event should be received
        assert len(received) == 1
        assert received[0].payload["symbol"] == "BTC/USDT"


class TestPublish:
    """Tests for event publishing."""

    def test_publish_increments_counter(self) -> None:
        """Publish should increment the published counter."""
        bus = MessageBus()
        event = Event.create(EventType.TICK, "test")

        assert bus.stats["published"] == 0
        bus.publish(event)
        assert bus.stats["published"] == 1

    def test_publish_routes_to_correct_queue(self) -> None:
        """Events should be routed based on priority."""
        bus = MessageBus()

        risk_event = Event.create(EventType.CIRCUIT_BREAKER, "test")
        tick_event = Event.create(EventType.TICK, "test")

        bus.publish(risk_event)
        bus.publish(tick_event)

        assert bus.queue_sizes["RISK"] == 1
        assert bus.queue_sizes["MARKET_DATA"] == 1
        assert bus.queue_sizes["ORDERS"] == 0
        assert bus.queue_sizes["SIGNALS"] == 0

    def test_publish_rejected_when_stopped(self) -> None:
        """Publish should return False when bus is stopped."""
        bus = MessageBus()
        bus._accepting = False

        event = Event.create(EventType.TICK, "test")
        result = bus.publish(event)

        assert result is False
        assert bus.stats["published"] == 0


class TestPriorityRouting:
    """Tests for priority-based event routing."""

    @pytest.mark.asyncio
    async def test_risk_events_processed_first(self) -> None:
        """RISK events should always be processed before lower priority."""
        bus = MessageBus()
        order: list[str] = []

        async def handler(event: Event) -> None:
            order.append(event.event_type.name)

        bus.subscribe(EventType.TICK, handler)
        bus.subscribe(EventType.SIGNAL, handler)
        bus.subscribe(EventType.ORDER_NEW, handler)
        bus.subscribe(EventType.CIRCUIT_BREAKER, handler)

        # Publish in reverse priority order
        bus.publish(Event.create(EventType.TICK, "1"))
        bus.publish(Event.create(EventType.SIGNAL, "2"))
        bus.publish(Event.create(EventType.ORDER_NEW, "3"))
        bus.publish(Event.create(EventType.CIRCUIT_BREAKER, "4"))

        # Dispatch all
        await bus._dispatch_batch()

        # Allow tasks to complete
        await asyncio.sleep(0.01)

        # Should be processed in priority order
        assert order[0] == "CIRCUIT_BREAKER"  # RISK
        assert order[1] == "ORDER_NEW"  # ORDERS
        assert order[2] == "SIGNAL"  # SIGNALS
        assert order[3] == "TICK"  # MARKET_DATA

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self) -> None:
        """Events with same priority should be FIFO."""
        bus = MessageBus()
        order: list[int] = []

        async def handler(event: Event) -> None:
            order.append(event.payload["seq"])

        bus.subscribe(EventType.TICK, handler)

        # Publish multiple ticks
        for i in range(5):
            bus.publish(Event.create(EventType.TICK, "test", {"seq": i}))

        await bus._dispatch_batch()
        await asyncio.sleep(0.01)

        assert order == [0, 1, 2, 3, 4]


class TestDispatch:
    """Tests for event dispatching."""

    @pytest.mark.asyncio
    async def test_handler_receives_event(self) -> None:
        """Handler should receive published events."""
        bus = MessageBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.TICK, handler)

        async with bus:
            event = Event.create(EventType.TICK, "test", {"price": 50000.0})
            bus.publish(event)
            await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0].payload["price"] == 50000.0

    @pytest.mark.asyncio
    async def test_multiple_handlers(self) -> None:
        """Multiple handlers should all receive the event."""
        bus = MessageBus()
        results: list[str] = []

        async def handler1(event: Event) -> None:
            results.append("h1")

        async def handler2(event: Event) -> None:
            results.append("h2")

        bus.subscribe(EventType.TICK, handler1)
        bus.subscribe(EventType.TICK, handler2)

        async with bus:
            bus.publish(Event.create(EventType.TICK, "test"))
            await asyncio.sleep(0.05)

        assert "h1" in results
        assert "h2" in results

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self) -> None:
        """Handler errors should not affect other handlers."""
        bus = MessageBus()
        received: list[Event] = []

        async def bad_handler(event: Event) -> None:
            raise ValueError("Handler error!")

        async def good_handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.TICK, bad_handler)
        bus.subscribe(EventType.TICK, good_handler)

        async with bus:
            bus.publish(Event.create(EventType.TICK, "test"))
            await asyncio.sleep(0.05)

        # Good handler should still receive event
        assert len(received) == 1
        assert bus.stats["errors"] == 1


class TestLifecycle:
    """Tests for message bus lifecycle."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Async context manager should start and stop bus."""
        bus = MessageBus()

        async with bus:
            assert bus.is_running is True

        assert bus.is_running is False

    @pytest.mark.asyncio
    async def test_graceful_shutdown_drains_queue(self) -> None:
        """Stop should drain pending events."""
        bus = MessageBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.TICK, handler)

        # Publish before starting
        bus.publish(Event.create(EventType.TICK, "1"))
        bus.publish(Event.create(EventType.TICK, "2"))

        async with bus:
            await asyncio.sleep(0.05)

        # Should have drained all events
        assert len(received) == 2
        assert bus.total_pending == 0

    @pytest.mark.asyncio
    async def test_shutdown_timeout(self) -> None:
        """Shutdown should respect timeout."""
        config = MessageBusConfig(drain_timeout=0.1)
        bus = MessageBus(config)

        async def slow_handler(event: Event) -> None:
            await asyncio.sleep(10)  # Way longer than timeout

        bus.subscribe(EventType.TICK, slow_handler)
        bus.publish(Event.create(EventType.TICK, "1"))

        # Start and stop quickly
        async with bus:
            await asyncio.sleep(0.01)

        # Shutdown should complete within timeout


class TestBackpressure:
    """Tests for queue backpressure."""

    def test_queue_maxlen(self) -> None:
        """Queue should enforce maxlen."""
        config = MessageBusConfig(data_queue_size=3)
        bus = MessageBus(config)

        # Publish more than maxlen
        for i in range(5):
            bus.publish(Event.create(EventType.TICK, "test", {"seq": i}))

        # Queue should only have 3 events
        assert bus.queue_sizes["MARKET_DATA"] == 3
        # 2 should be dropped
        assert bus.stats["dropped"] == 2


class TestMetrics:
    """Tests for bus metrics."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Stats should track all metrics."""
        bus = MessageBus()

        async def handler(event: Event) -> None:
            pass

        bus.subscribe(EventType.TICK, handler)

        async with bus:
            bus.publish(Event.create(EventType.TICK, "test"))
            await asyncio.sleep(0.05)

        stats = bus.stats
        assert stats["published"] == 1
        assert stats["dispatched"] == 1
        assert stats["handlers"] == 1
        assert stats["errors"] == 0


class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_publish_throughput(self) -> None:
        """Measure publish throughput."""
        bus = MessageBus()
        iterations = 100_000

        start = time.perf_counter_ns()
        for _ in range(iterations):
            bus.publish(Event.create(EventType.TICK, "bench", {"price": 50000.0}))
        elapsed = time.perf_counter_ns() - start

        per_event = elapsed / iterations
        throughput = 1e9 / per_event

        print(f"\nPublish: {per_event:.0f}ns/event = {throughput:,.0f} events/sec")

        # Should exceed 100K events/sec
        assert throughput > 100_000

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_full_cycle_throughput(self) -> None:
        """Measure full publish + dispatch throughput."""
        bus = MessageBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.TICK, handler)

        iterations = 10_000

        start = time.perf_counter_ns()

        for _ in range(iterations):
            bus.publish(Event.create(EventType.TICK, "bench", {"price": 50000.0}))

        # Dispatch all
        while bus.total_pending > 0:
            await bus._dispatch_batch()
            await asyncio.sleep(0)

        # Wait for handlers
        await asyncio.sleep(0.1)

        elapsed = time.perf_counter_ns() - start

        per_event = elapsed / iterations
        throughput = 1e9 / per_event

        print(f"\nFull cycle: {per_event:.0f}ns/event = {throughput:,.0f} events/sec")
        print(f"Processed: {len(received)} events")

        # Should exceed 30K events/sec (lower threshold for coverage overhead)
        # Real throughput is 400K+ events/sec (see benchmarks/)
        assert throughput > 30_000

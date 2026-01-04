"""
Throughput benchmarks for MessageBus publish operations.

Uses pytest-benchmark with pedantic mode for reliable measurements:
- Explicit warmup rounds
- GC disabled during measurement
- Multiple payload sizes
- Statistical reporting (min, max, mean, stddev)

Run: pytest benchmarks/bench_publish.py --benchmark-only -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus, MessageBusConfig


if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


# =============================================================================
# Publish Throughput Benchmarks
# =============================================================================


class TestPublishThroughput:
    """Benchmarks for raw publish throughput (no dispatch)."""

    @pytest.mark.bench_throughput
    def test_publish_single_event(
        self,
        benchmark: BenchmarkFixture,
        tick_event: Event,
        gc_disabled: None,
    ) -> None:
        """Measure single event publish latency.

        Target: <1μs per publish (>1M events/sec)
        """
        bus = MessageBus()

        # Use pedantic mode for precise control
        benchmark.pedantic(
            bus.publish,
            args=(tick_event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_publish_batch_1k(
        self,
        benchmark: BenchmarkFixture,
        tick_events_1k: list[Event],
        gc_disabled: None,
    ) -> None:
        """Measure batch publish throughput (1K events).

        Target: >1M events/sec sustained
        """
        bus = MessageBus()

        def publish_batch() -> None:
            for event in tick_events_1k:
                bus.publish(event)

        benchmark.pedantic(
            publish_batch,
            rounds=50,
            iterations=1,
            warmup_rounds=3,
        )

    @pytest.mark.bench_throughput
    def test_publish_batch_10k(
        self,
        benchmark: BenchmarkFixture,
        tick_events_10k: list[Event],
        gc_disabled: None,
    ) -> None:
        """Measure batch publish throughput (10K events).

        Tests sustained throughput with larger batches.
        """
        bus = MessageBus()

        def publish_batch() -> None:
            for event in tick_events_10k:
                bus.publish(event)

        benchmark.pedantic(
            publish_batch,
            rounds=30,
            iterations=1,
            warmup_rounds=3,
        )


# =============================================================================
# Publish with Different Payload Sizes
# =============================================================================


class TestPublishPayloadSizes:
    """Benchmarks for publish with varying payload sizes."""

    @pytest.mark.bench_throughput
    def test_publish_tiny_payload(
        self,
        benchmark: BenchmarkFixture,
        tiny_payload: dict[str, Any],
        gc_disabled: None,
    ) -> None:
        """Publish with ~10 byte payload."""
        bus = MessageBus()
        event = Event.create(EventType.TICK, "bench", tiny_payload)

        benchmark.pedantic(
            bus.publish,
            args=(event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_publish_small_payload(
        self,
        benchmark: BenchmarkFixture,
        small_payload: dict[str, Any],
        gc_disabled: None,
    ) -> None:
        """Publish with ~100 byte payload."""
        bus = MessageBus()
        event = Event.create(EventType.TICK, "bench", small_payload)

        benchmark.pedantic(
            bus.publish,
            args=(event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_publish_medium_payload(
        self,
        benchmark: BenchmarkFixture,
        medium_payload: dict[str, Any],
        gc_disabled: None,
    ) -> None:
        """Publish with ~1 KB payload."""
        bus = MessageBus()
        event = Event.create(EventType.TICK, "bench", medium_payload)

        benchmark.pedantic(
            bus.publish,
            args=(event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_publish_large_payload(
        self,
        benchmark: BenchmarkFixture,
        large_payload: dict[str, Any],
        gc_disabled: None,
    ) -> None:
        """Publish with ~10 KB payload."""
        bus = MessageBus()
        event = Event.create(EventType.TICK, "bench", large_payload)

        benchmark.pedantic(
            bus.publish,
            args=(event,),
            rounds=100,
            iterations=5_000,
            warmup_rounds=5,
        )


# =============================================================================
# Priority Queue Routing
# =============================================================================


class TestPublishPriorityRouting:
    """Benchmarks for publish to different priority queues."""

    @pytest.mark.bench_throughput
    def test_publish_risk_priority(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Publish to RISK queue (highest priority)."""
        bus = MessageBus()
        event = Event.create(EventType.CIRCUIT_BREAKER, "bench", {"triggered": True})

        benchmark.pedantic(
            bus.publish,
            args=(event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_publish_orders_priority(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Publish to ORDERS queue."""
        bus = MessageBus()
        event = Event.create(EventType.ORDER_NEW, "bench", {"order_id": "123"})

        benchmark.pedantic(
            bus.publish,
            args=(event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_publish_market_data_priority(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Publish to MARKET_DATA queue (highest volume)."""
        bus = MessageBus()
        event = Event.create(EventType.TICK, "bench", {"price": 50000.0})

        benchmark.pedantic(
            bus.publish,
            args=(event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )


# =============================================================================
# Backpressure Behavior
# =============================================================================


class TestPublishBackpressure:
    """Benchmarks for publish under backpressure (full queues)."""

    @pytest.mark.bench_throughput
    def test_publish_with_drops(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Publish with small queue (measures drop overhead)."""
        config = MessageBusConfig(data_queue_size=100)
        bus = MessageBus(config)

        event = Event.create(EventType.TICK, "bench", {"price": 50000.0})

        # Pre-fill to create backpressure
        for _ in range(100):
            bus.publish(event)

        benchmark.pedantic(
            bus.publish,
            args=(event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )


# =============================================================================
# Event Creation Benchmarks (for reference)
# =============================================================================


class TestEventCreation:
    """Benchmarks for event creation (separate from publish)."""

    @pytest.mark.bench_throughput
    def test_event_create(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Measure Event.create() overhead."""
        payload = {"symbol": "BTC/USDT", "price": 50000.0}

        def create_event() -> Event:
            return Event.create(EventType.TICK, "bench", payload)

        benchmark.pedantic(
            create_event,
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_event_create_batch_1k(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Measure batch Event.create() throughput."""
        payload = {"symbol": "BTC/USDT", "price": 50000.0}

        def create_batch() -> list[Event]:
            return [Event.create(EventType.TICK, "bench", payload) for _ in range(1000)]

        benchmark.pedantic(
            create_batch,
            rounds=50,
            iterations=1,
            warmup_rounds=3,
        )

"""
End-to-end benchmarks for full MessageBus cycle.

Measures the complete path: publish -> dispatch -> handler execution.

These benchmarks are more realistic but also more variable due to:
- Async scheduling overhead
- Handler execution time
- Event loop interaction

Run: pytest benchmarks/bench_full_cycle.py -v -s
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import time
from dataclasses import dataclass

import pytest
from hdrh.histogram import HdrHistogram

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus, MessageBusConfig


# =============================================================================
# Result Container
# =============================================================================


@dataclass
class E2EResult:
    """Container for end-to-end benchmark results."""

    name: str
    events_published: int = 0
    events_received: int = 0
    elapsed_ns: int = 0
    throughput_eps: float = 0.0
    latency_histogram: HdrHistogram | None = None

    def compute_throughput(self) -> None:
        """Compute throughput from elapsed time."""
        if self.elapsed_ns > 0:
            self.throughput_eps = self.events_received / (self.elapsed_ns / 1e9)

    def report(self) -> str:
        """Generate human-readable report."""
        lines = [
            f"\n{self.name}",
            "=" * 60,
            f"  Published:  {self.events_published:>12,} events",
            f"  Received:   {self.events_received:>12,} events",
            f"  Elapsed:    {self.elapsed_ns / 1e6:>12,.2f} ms",
            f"  Throughput: {self.throughput_eps:>12,.0f} events/sec",
        ]

        if self.latency_histogram and self.latency_histogram.total_count > 0:
            h = self.latency_histogram
            lines.extend(
                [
                    "",
                    "  End-to-End Latency:",
                    f"    p50:    {h.get_value_at_percentile(50) / 1000:>10,.2f} us",
                    f"    p95:    {h.get_value_at_percentile(95) / 1000:>10,.2f} us",
                    f"    p99:    {h.get_value_at_percentile(99) / 1000:>10,.2f} us",
                    f"    p99.9:  {h.get_value_at_percentile(99.9) / 1000:>10,.2f} us",
                    f"    max:    {h.max_value / 1000:>10,.2f} us",
                ]
            )

        return "\n".join(lines)


# =============================================================================
# Full Cycle Throughput
# =============================================================================


class TestFullCycleThroughput:
    """End-to-end throughput benchmarks."""

    @pytest.mark.bench_e2e
    @pytest.mark.asyncio
    async def test_single_handler_throughput(self) -> None:
        """Measure throughput with single handler.

        Target: >100K events/sec end-to-end
        """
        gc.disable()
        gc.collect()

        bus = MessageBus()
        result = E2EResult(name="Single Handler Throughput")
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.TICK, handler)

        # Pre-create events to isolate publish+dispatch
        events = [
            Event.create(EventType.TICK, "bench", {"price": 50000.0 + i}) for i in range(10_000)
        ]

        # Warmup
        for e in events[:1000]:
            bus.publish(e)
        async with bus:
            await asyncio.sleep(0.1)
        received.clear()

        # Measure
        bus = MessageBus()  # Fresh bus
        bus.subscribe(EventType.TICK, handler)

        start = time.perf_counter_ns()

        for e in events:
            bus.publish(e)
        result.events_published = len(events)

        async with bus:
            # Wait for all handlers to complete (with timeout)
            deadline = time.perf_counter_ns() + 5_000_000_000  # 5 second timeout
            while len(received) < len(events) and time.perf_counter_ns() < deadline:
                await asyncio.sleep(0.001)

        result.elapsed_ns = time.perf_counter_ns() - start
        result.events_received = len(received)
        result.compute_throughput()

        print(result.report())

        gc.enable()

        assert result.events_received == result.events_published
        assert result.throughput_eps > 50_000, f"Throughput too low: {result.throughput_eps:.0f}"

    @pytest.mark.bench_e2e
    @pytest.mark.asyncio
    async def test_multiple_handlers_throughput(self) -> None:
        """Measure throughput with multiple handlers per event.

        Each event triggers 3 handlers.
        """
        gc.disable()
        gc.collect()

        bus = MessageBus()
        result = E2EResult(name="Multiple Handlers (3x) Throughput")
        received: list[Event] = []

        async def handler1(event: Event) -> None:
            received.append(event)

        async def handler2(event: Event) -> None:
            received.append(event)

        async def handler3(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.TICK, handler1)
        bus.subscribe(EventType.TICK, handler2)
        bus.subscribe(EventType.TICK, handler3)

        events = [
            Event.create(EventType.TICK, "bench", {"price": 50000.0 + i}) for i in range(5_000)
        ]
        expected_received = len(events) * 3

        start = time.perf_counter_ns()

        for e in events:
            bus.publish(e)
        result.events_published = len(events)

        async with bus:
            deadline = time.perf_counter_ns() + 5_000_000_000  # 5 second timeout
            while len(received) < expected_received and time.perf_counter_ns() < deadline:
                await asyncio.sleep(0.001)

        result.elapsed_ns = time.perf_counter_ns() - start
        result.events_received = len(received)
        result.compute_throughput()

        print(result.report())

        gc.enable()

        assert result.events_received == expected_received

    @pytest.mark.bench_e2e
    @pytest.mark.asyncio
    async def test_priority_routing_throughput(self) -> None:
        """Measure throughput with mixed priority events."""
        gc.disable()
        gc.collect()

        # Use larger queues for this test (2500 events per priority)
        config = MessageBusConfig(
            risk_queue_size=5000,
            orders_queue_size=5000,
            signals_queue_size=5000,
            data_queue_size=5000,
        )
        bus = MessageBus(config)
        result = E2EResult(name="Mixed Priority Throughput")
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        # Subscribe to all event types
        bus.subscribe(EventType.TICK, handler)
        bus.subscribe(EventType.SIGNAL, handler)
        bus.subscribe(EventType.ORDER_NEW, handler)
        bus.subscribe(EventType.CIRCUIT_BREAKER, handler)

        # Create mixed priority events
        events = []
        for i in range(2500):
            events.append(Event.create(EventType.TICK, "bench", {"seq": i}))
            events.append(Event.create(EventType.SIGNAL, "bench", {"seq": i}))
            events.append(Event.create(EventType.ORDER_NEW, "bench", {"seq": i}))
            events.append(Event.create(EventType.CIRCUIT_BREAKER, "bench", {"seq": i}))

        start = time.perf_counter_ns()

        for e in events:
            bus.publish(e)
        result.events_published = len(events)

        async with bus:
            deadline = time.perf_counter_ns() + 5_000_000_000  # 5 second timeout
            while len(received) < len(events) and time.perf_counter_ns() < deadline:
                await asyncio.sleep(0.001)

        result.elapsed_ns = time.perf_counter_ns() - start
        result.events_received = len(received)
        result.compute_throughput()

        print(result.report())

        gc.enable()

        assert result.events_received == result.events_published


# =============================================================================
# End-to-End Latency
# =============================================================================


class TestFullCycleLatency:
    """End-to-end latency benchmarks with HDR Histogram."""

    @pytest.mark.bench_e2e
    @pytest.mark.asyncio
    async def test_e2e_latency_distribution(self) -> None:
        """Measure end-to-end latency distribution.

        Latency = time from publish to handler completion.

        Target:
          - p50 < 100μs
          - p99 < 1ms
        """
        gc.disable()
        gc.collect()

        bus = MessageBus()
        histogram = HdrHistogram(1, 1_000_000_000, 3)  # 1ns to 1s
        result = E2EResult(name="E2E Latency Distribution", latency_histogram=histogram)
        latency_done = asyncio.Event()

        async def latency_handler(event: Event) -> None:
            elapsed = time.perf_counter_ns() - event.payload["send_time"]
            histogram.record_value(elapsed)
            if histogram.total_count >= 10_000:
                latency_done.set()

        bus.subscribe(EventType.TICK, latency_handler)

        async with bus:
            # Warmup
            for _ in range(1000):
                e = Event.create(EventType.TICK, "bench", {"send_time": time.perf_counter_ns()})
                bus.publish(e)
            await asyncio.sleep(0.1)

            # Reset histogram for measurement
            histogram.reset()

            # Measure
            for _ in range(10_000):
                e = Event.create(EventType.TICK, "bench", {"send_time": time.perf_counter_ns()})
                bus.publish(e)
                await asyncio.sleep(0)  # Yield to allow dispatch

            # Wait for all to be processed (timeout is acceptable)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(latency_done.wait(), timeout=5.0)

        result.events_published = 10_000
        result.events_received = histogram.total_count

        print(result.report())

        gc.enable()

        # Assertions
        p50 = histogram.get_value_at_percentile(50)
        p99 = histogram.get_value_at_percentile(99)

        print(f"\n  Assertions: p50={p50 / 1000:.2f}us, p99={p99 / 1000:.2f}us")
        assert p50 < 500_000, f"p50 too high: {p50 / 1000:.2f}us"  # < 500μs
        assert p99 < 5_000_000, f"p99 too high: {p99 / 1000:.2f}us"  # < 5ms


# =============================================================================
# Sustained Load Testing
# =============================================================================


class TestSustainedLoad:
    """Tests for sustained throughput over time."""

    @pytest.mark.bench_e2e
    @pytest.mark.asyncio
    async def test_sustained_throughput_1sec(self) -> None:
        """Measure sustained throughput over 1 second.

        Publishes continuously and measures actual processing rate.
        """
        gc.disable()
        gc.collect()

        bus = MessageBus()
        result = E2EResult(name="Sustained 1-second Load")
        received_count = 0

        async def counter_handler(event: Event) -> None:
            nonlocal received_count
            received_count += 1

        bus.subscribe(EventType.TICK, counter_handler)

        async with bus:
            start = time.perf_counter_ns()
            published = 0

            # Publish for 1 second
            while (time.perf_counter_ns() - start) < 1_000_000_000:
                event = Event.create(EventType.TICK, "bench", {"seq": published})
                bus.publish(event)
                published += 1

                # Yield periodically
                if published % 1000 == 0:
                    await asyncio.sleep(0)

            result.events_published = published

            # Wait for remaining to be processed
            await asyncio.sleep(0.5)

        result.elapsed_ns = time.perf_counter_ns() - start
        result.events_received = received_count
        result.compute_throughput()

        print(result.report())

        gc.enable()

        # Should maintain high throughput
        assert result.throughput_eps > 50_000, f"Throughput too low: {result.throughput_eps:.0f}"


# =============================================================================
# Backpressure Behavior
# =============================================================================


class TestBackpressureBehavior:
    """Tests for behavior under backpressure."""

    @pytest.mark.bench_e2e
    @pytest.mark.asyncio
    async def test_slow_handler_backpressure(self) -> None:
        """Measure behavior with slow handler (simulates I/O)."""
        gc.disable()
        gc.collect()

        config = MessageBusConfig(data_queue_size=100)
        bus = MessageBus(config)
        result = E2EResult(name="Slow Handler (1ms delay)")
        received_count = 0

        async def slow_handler(event: Event) -> None:
            nonlocal received_count
            await asyncio.sleep(0.001)  # 1ms simulated I/O
            received_count += 1

        bus.subscribe(EventType.TICK, slow_handler)

        async with bus:
            start = time.perf_counter_ns()

            # Publish more than queue can hold
            for i in range(500):
                event = Event.create(EventType.TICK, "bench", {"seq": i})
                bus.publish(event)

            result.events_published = 500

            # Wait for processing
            await asyncio.sleep(1.0)

        result.elapsed_ns = time.perf_counter_ns() - start
        result.events_received = received_count
        result.compute_throughput()

        dropped = bus.stats["dropped"]

        print(result.report())
        print(f"\n  Dropped due to backpressure: {dropped}")
        print(f"  Effective throughput with drops: {result.throughput_eps:.0f} eps")

        gc.enable()


# =============================================================================
# Summary Report
# =============================================================================


class TestBenchmarkSummary:
    """Generate summary report of all benchmarks."""

    @pytest.mark.bench_e2e
    @pytest.mark.asyncio
    async def test_generate_summary(self) -> None:
        """Generate comprehensive benchmark summary."""
        print("\n")
        print("=" * 80)
        print("LIBRA MESSAGE BUS BENCHMARK SUMMARY")
        print("=" * 80)

        gc.disable()
        gc.collect()

        bus = MessageBus()
        payload = {"symbol": "BTC/USDT", "price": 50000.0}
        event = Event.create(EventType.TICK, "bench", payload)

        # 1. Publish-only throughput
        iterations = 100_000
        start = time.perf_counter_ns()
        for _ in range(iterations):
            bus.publish(event)
        publish_elapsed = time.perf_counter_ns() - start
        publish_throughput = iterations / (publish_elapsed / 1e9)
        publish_latency = publish_elapsed / iterations

        print("\n1. PUBLISH ONLY (no dispatch)")
        print(f"   Throughput: {publish_throughput:,.0f} events/sec")
        print(f"   Latency:    {publish_latency:.0f} ns/event")

        # 2. Event creation throughput
        iterations = 100_000
        start = time.perf_counter_ns()
        for _ in range(iterations):
            Event.create(EventType.TICK, "bench", payload)
        create_elapsed = time.perf_counter_ns() - start
        create_throughput = iterations / (create_elapsed / 1e9)
        create_latency = create_elapsed / iterations

        print("\n2. EVENT CREATION")
        print(f"   Throughput: {create_throughput:,.0f} events/sec")
        print(f"   Latency:    {create_latency:.0f} ns/event")

        # 3. Full cycle with handler
        bus = MessageBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.TICK, handler)
        iterations = 10_000

        events = [Event.create(EventType.TICK, "bench", {"seq": i}) for i in range(iterations)]

        start = time.perf_counter_ns()
        for e in events:
            bus.publish(e)

        async with bus:
            while len(received) < iterations:
                await asyncio.sleep(0.001)

        e2e_elapsed = time.perf_counter_ns() - start
        e2e_throughput = iterations / (e2e_elapsed / 1e9)
        e2e_latency = e2e_elapsed / iterations

        print("\n3. FULL CYCLE (publish + dispatch + handler)")
        print(f"   Throughput: {e2e_throughput:,.0f} events/sec")
        print(f"   Latency:    {e2e_latency / 1000:.2f} us/event")

        gc.enable()

        print("\n" + "=" * 80)
        print("TARGETS:")
        print("  - Publish: >1M events/sec")
        print("  - E2E:     >100K events/sec")
        print("=" * 80)

        # Assertions
        assert publish_throughput > 500_000, "Publish throughput below 500K"
        assert e2e_throughput > 50_000, "E2E throughput below 50K"

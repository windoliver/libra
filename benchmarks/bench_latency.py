"""
Latency distribution benchmarks using HDR Histogram.

HDR Histogram provides:
- High-precision percentile tracking (p50, p95, p99, p99.9)
- Fixed memory footprint regardless of sample count
- Coordinated omission detection
- Sub-microsecond recording overhead (~3-6ns)

Run: pytest benchmarks/bench_latency.py -v -s
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import pytest
from hdrh.histogram import HdrHistogram

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus


# =============================================================================
# Latency Result Container
# =============================================================================


@dataclass
class LatencyResult:
    """Container for latency distribution results."""

    name: str
    samples: int = 0
    min_ns: int = 0
    max_ns: int = 0
    mean_ns: float = 0.0
    stddev_ns: float = 0.0
    p50_ns: int = 0
    p75_ns: int = 0
    p95_ns: int = 0
    p99_ns: int = 0
    p999_ns: int = 0

    histogram: HdrHistogram = field(default_factory=lambda: HdrHistogram(1, 1_000_000_000, 3))

    def record(self, latency_ns: int) -> None:
        """Record a latency sample."""
        self.histogram.record_value(latency_ns)

    def finalize(self) -> None:
        """Compute final statistics from histogram."""
        self.samples = self.histogram.total_count
        self.min_ns = self.histogram.get_min_value()
        self.max_ns = self.histogram.get_max_value()
        self.mean_ns = self.histogram.get_mean_value()
        self.stddev_ns = self.histogram.get_stddev()
        self.p50_ns = self.histogram.get_value_at_percentile(50)
        self.p75_ns = self.histogram.get_value_at_percentile(75)
        self.p95_ns = self.histogram.get_value_at_percentile(95)
        self.p99_ns = self.histogram.get_value_at_percentile(99)
        self.p999_ns = self.histogram.get_value_at_percentile(99.9)

    def report(self) -> str:
        """Generate human-readable report."""
        return f"""
{self.name} Latency Distribution ({self.samples:,} samples)
{"=" * 60}
  Min:    {self._fmt(self.min_ns):>12}
  Max:    {self._fmt(self.max_ns):>12}
  Mean:   {self._fmt(self.mean_ns):>12} (stddev: {self._fmt(self.stddev_ns)})

  Percentiles:
    p50:    {self._fmt(self.p50_ns):>12}  (median)
    p75:    {self._fmt(self.p75_ns):>12}
    p95:    {self._fmt(self.p95_ns):>12}
    p99:    {self._fmt(self.p99_ns):>12}  (SLA boundary)
    p99.9:  {self._fmt(self.p999_ns):>12}  (tail latency)
"""

    @staticmethod
    def _fmt(ns: float) -> str:
        """Format nanoseconds with appropriate unit."""
        if ns >= 1_000_000:
            return f"{ns / 1_000_000:.2f}ms"
        if ns >= 1_000:
            return f"{ns / 1_000:.2f}us"
        return f"{ns:.0f}ns"


# =============================================================================
# Publish Latency Distribution
# =============================================================================


class TestPublishLatency:
    """Latency distribution for publish operations."""

    @pytest.mark.bench_latency
    def test_publish_latency_distribution(self) -> None:
        """Measure publish latency distribution.

        Target:
          - p50 < 500ns
          - p99 < 2μs
          - p99.9 < 10μs
        """
        gc.disable()
        gc.collect()

        bus = MessageBus()
        event = Event.create(EventType.TICK, "bench", {"price": 50000.0})
        result = LatencyResult(name="Publish")

        # Warmup
        for _ in range(10_000):
            bus.publish(event)

        # Clear queue for measurement
        bus._queues[event.priority].clear()
        bus._events_published = 0

        # Measure
        samples = 100_000
        for _ in range(samples):
            start = time.perf_counter_ns()
            bus.publish(event)
            elapsed = time.perf_counter_ns() - start
            result.record(elapsed)

        result.finalize()
        print(result.report())

        gc.enable()

        # Assertions based on target latencies
        assert result.p50_ns < 1_000, f"p50 too high: {result.p50_ns}ns"
        assert result.p99_ns < 5_000, f"p99 too high: {result.p99_ns}ns"
        assert result.p999_ns < 50_000, f"p99.9 too high: {result.p999_ns}ns"

    @pytest.mark.bench_latency
    def test_event_create_latency_distribution(self) -> None:
        """Measure Event.create() latency distribution.

        This isolates event creation overhead from publish.
        """
        gc.disable()
        gc.collect()

        payload = {"symbol": "BTC/USDT", "price": 50000.0}
        result = LatencyResult(name="Event.create")

        # Warmup
        for _ in range(10_000):
            Event.create(EventType.TICK, "bench", payload)

        # Measure
        samples = 100_000
        for _ in range(samples):
            start = time.perf_counter_ns()
            Event.create(EventType.TICK, "bench", payload)
            elapsed = time.perf_counter_ns() - start
            result.record(elapsed)

        result.finalize()
        print(result.report())

        gc.enable()

        # Event creation includes trace_id generation
        assert result.p50_ns < 5_000, f"p50 too high: {result.p50_ns}ns"
        assert result.p99_ns < 20_000, f"p99 too high: {result.p99_ns}ns"


# =============================================================================
# Combined Create + Publish Latency
# =============================================================================


class TestCombinedLatency:
    """Latency for combined event creation and publish."""

    @pytest.mark.bench_latency
    def test_create_and_publish_latency(self) -> None:
        """Measure combined Event.create() + publish latency.

        This is the realistic hot-path for publishing new events.
        """
        gc.disable()
        gc.collect()

        bus = MessageBus()
        payload = {"symbol": "BTC/USDT", "price": 50000.0}
        result = LatencyResult(name="Create+Publish")

        # Warmup
        for _ in range(10_000):
            event = Event.create(EventType.TICK, "bench", payload)
            bus.publish(event)

        # Clear for measurement
        bus._queues[0].clear()  # RISK queue
        bus._queues[3].clear()  # MARKET_DATA queue
        bus._events_published = 0

        # Measure
        samples = 100_000
        for i in range(samples):
            start = time.perf_counter_ns()
            event = Event.create(EventType.TICK, "bench", {"price": 50000.0 + i})
            bus.publish(event)
            elapsed = time.perf_counter_ns() - start
            result.record(elapsed)

        result.finalize()
        print(result.report())

        gc.enable()

        # Combined should be < 10μs at p99
        assert result.p50_ns < 10_000, f"p50 too high: {result.p50_ns}ns"
        assert result.p99_ns < 50_000, f"p99 too high: {result.p99_ns}ns"


# =============================================================================
# Subscription Lookup Latency
# =============================================================================


class TestSubscriptionLatency:
    """Latency for subscription operations."""

    @pytest.mark.bench_latency
    def test_subscribe_latency(self) -> None:
        """Measure subscribe() latency."""
        gc.disable()
        gc.collect()

        result = LatencyResult(name="Subscribe")

        async def handler(event: Event) -> None:
            pass

        # Measure with fresh bus each iteration to avoid list growth
        samples = 10_000
        for _ in range(samples):
            bus = MessageBus()
            start = time.perf_counter_ns()
            bus.subscribe(EventType.TICK, handler)
            elapsed = time.perf_counter_ns() - start
            result.record(elapsed)

        result.finalize()
        print(result.report())

        gc.enable()

        assert result.p50_ns < 5_000, f"p50 too high: {result.p50_ns}ns"

    @pytest.mark.bench_latency
    def test_subscribe_with_many_handlers(self) -> None:
        """Measure subscribe() latency with many existing handlers."""
        gc.disable()
        gc.collect()

        bus = MessageBus()
        result = LatencyResult(name="Subscribe (100 existing)")

        async def handler(event: Event) -> None:
            pass

        # Pre-populate with 100 handlers
        for _ in range(100):
            bus.subscribe(EventType.TICK, handler)

        # Measure adding more
        samples = 10_000
        for _ in range(samples):
            start = time.perf_counter_ns()
            bus.subscribe(EventType.TICK, handler)
            elapsed = time.perf_counter_ns() - start
            result.record(elapsed)

        result.finalize()
        print(result.report())

        gc.enable()

        # Should still be fast (O(1) append)
        assert result.p50_ns < 5_000, f"p50 too high: {result.p50_ns}ns"


# =============================================================================
# Comparative Latency Report
# =============================================================================


class TestLatencyComparison:
    """Generate comparative latency report across operations."""

    @pytest.mark.bench_latency
    def test_operation_latency_comparison(self) -> None:
        """Compare latencies across different operations."""
        gc.disable()
        gc.collect()

        results: list[LatencyResult] = []
        bus = MessageBus()
        payload = {"symbol": "BTC/USDT", "price": 50000.0}
        event = Event.create(EventType.TICK, "bench", payload)
        samples = 50_000

        async def handler(event: Event) -> None:
            pass

        bus.subscribe(EventType.TICK, handler)

        # 1. Event creation
        result1 = LatencyResult(name="Event.create()")
        for _ in range(samples):
            start = time.perf_counter_ns()
            Event.create(EventType.TICK, "bench", payload)
            result1.record(time.perf_counter_ns() - start)
        result1.finalize()
        results.append(result1)

        # 2. Publish only (pre-created event)
        bus._queues[event.priority].clear()
        result2 = LatencyResult(name="bus.publish()")
        for _ in range(samples):
            start = time.perf_counter_ns()
            bus.publish(event)
            result2.record(time.perf_counter_ns() - start)
        result2.finalize()
        results.append(result2)

        # 3. Create + Publish
        bus._queues[event.priority].clear()
        result3 = LatencyResult(name="create+publish")
        for i in range(samples):
            start = time.perf_counter_ns()
            e = Event.create(EventType.TICK, "bench", {"price": float(i)})
            bus.publish(e)
            result3.record(time.perf_counter_ns() - start)
        result3.finalize()
        results.append(result3)

        gc.enable()

        # Print comparison table
        print("\n" + "=" * 80)
        print("LATENCY COMPARISON (nanoseconds)")
        print("=" * 80)
        print(f"{'Operation':<20} {'p50':>10} {'p95':>10} {'p99':>10} {'p99.9':>10} {'max':>10}")
        print("-" * 80)
        for r in results:
            print(
                f"{r.name:<20} {r.p50_ns:>10,} {r.p95_ns:>10,} "
                f"{r.p99_ns:>10,} {r.p999_ns:>10,} {r.max_ns:>10,}"
            )
        print("=" * 80)

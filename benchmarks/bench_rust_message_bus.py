"""
Benchmark: Python MessageBus vs Rust MessageBus (Issue #112).

Compares throughput and latency between:
- Pure Python MessageBus (baseline)
- Rust MessageBus with PyO3 bindings

Run: pytest benchmarks/bench_rust_message_bus.py --benchmark-only -v

Expected Results:
- Rust publish: >100M events/sec
- Rust dispatch: >50M events/sec
- Python publish: ~2.5M events/sec
- Speedup: 40-100x
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus, MessageBusConfig


if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


# Try to import Rust MessageBus
try:
    from libra.core._rust import RustMessageBus, RustMessageBusConfig

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustMessageBus = None  # type: ignore[misc, assignment]
    RustMessageBusConfig = None  # type: ignore[misc, assignment]


# =============================================================================
# Python MessageBus Baseline
# =============================================================================


class TestPythonMessageBusBaseline:
    """Baseline benchmarks for Python MessageBus."""

    @pytest.mark.bench_throughput
    def test_python_publish_single(
        self,
        benchmark: BenchmarkFixture,
        tick_event: Event,
        gc_disabled: None,
    ) -> None:
        """Python MessageBus single publish."""
        bus = MessageBus()

        benchmark.pedantic(
            bus.publish,
            args=(tick_event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_python_publish_batch_10k(
        self,
        benchmark: BenchmarkFixture,
        tick_events_10k: list[Event],
        gc_disabled: None,
    ) -> None:
        """Python MessageBus batch publish (10K events)."""
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
# Rust MessageBus Benchmarks
# =============================================================================


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestRustMessageBus:
    """Benchmarks for Rust MessageBus."""

    @pytest.mark.bench_throughput
    def test_rust_publish_single(
        self,
        benchmark: BenchmarkFixture,
        tick_event: Event,
        gc_disabled: None,
    ) -> None:
        """Rust MessageBus single publish.

        Target: <10ns per publish (>100M events/sec)
        """
        bus = RustMessageBus()

        benchmark.pedantic(
            bus.publish,
            args=(tick_event,),
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_rust_publish_batch_10k(
        self,
        benchmark: BenchmarkFixture,
        tick_events_10k: list[Event],
        gc_disabled: None,
    ) -> None:
        """Rust MessageBus batch publish (10K events)."""
        bus = RustMessageBus()

        def publish_batch() -> None:
            for event in tick_events_10k:
                bus.publish(event)

        benchmark.pedantic(
            publish_batch,
            rounds=30,
            iterations=1,
            warmup_rounds=3,
        )

    @pytest.mark.bench_throughput
    def test_rust_dispatch_batch(
        self,
        benchmark: BenchmarkFixture,
        tick_events_10k: list[Event],
        gc_disabled: None,
    ) -> None:
        """Rust MessageBus dispatch batch."""
        bus = RustMessageBus()

        # Add a simple handler
        call_count = [0]

        def handler(event: Event) -> None:
            call_count[0] += 1

        bus.subscribe(EventType.TICK.value, handler)

        def publish_and_dispatch() -> int:
            # Publish events
            for event in tick_events_10k[:100]:  # Use 100 events per batch
                bus.publish(event)
            # Dispatch batch
            return bus.dispatch_batch()

        benchmark.pedantic(
            publish_and_dispatch,
            rounds=50,
            iterations=1,
            warmup_rounds=3,
        )

    @pytest.mark.bench_throughput
    def test_rust_full_cycle(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Full cycle: create event, publish, dispatch."""
        bus = RustMessageBus()

        call_count = [0]

        def handler(event: Event) -> None:
            call_count[0] += 1

        bus.subscribe(EventType.TICK.value, handler)

        def full_cycle() -> None:
            event = Event.create(EventType.TICK, "bench", {"price": 50000.0})
            bus.publish(event)
            bus.dispatch_batch()

        benchmark.pedantic(
            full_cycle,
            rounds=100,
            iterations=1_000,
            warmup_rounds=5,
        )


# =============================================================================
# Side-by-Side Comparison
# =============================================================================


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestComparison:
    """Side-by-side Python vs Rust comparison."""

    @pytest.mark.bench_throughput
    def test_comparison_publish_throughput(
        self,
        tick_events_10k: list[Event],
        gc_disabled: None,
    ) -> None:
        """Compare publish throughput Python vs Rust."""
        # Python baseline
        py_bus = MessageBus()
        start = time.perf_counter()
        for event in tick_events_10k:
            py_bus.publish(event)
        py_time = time.perf_counter() - start
        py_throughput = len(tick_events_10k) / py_time

        # Rust implementation
        rust_bus = RustMessageBus()
        start = time.perf_counter()
        for event in tick_events_10k:
            rust_bus.publish(event)
        rust_time = time.perf_counter() - start
        rust_throughput = len(tick_events_10k) / rust_time

        speedup = rust_throughput / py_throughput

        print(f"\n{'='*60}")
        print("Publish Throughput Comparison (10K events)")
        print(f"{'='*60}")
        print(f"Python:  {py_throughput:,.0f} events/sec ({py_time*1000:.2f}ms)")
        print(f"Rust:    {rust_throughput:,.0f} events/sec ({rust_time*1000:.2f}ms)")
        print(f"Speedup: {speedup:.1f}x")
        print(f"{'='*60}")

        # Verify Rust is faster
        assert speedup > 1.0, f"Rust should be faster than Python, got {speedup:.2f}x"


# =============================================================================
# Memory Efficiency
# =============================================================================


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestMemoryEfficiency:
    """Memory efficiency benchmarks."""

    @pytest.mark.bench_throughput
    def test_rust_stats_overhead(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Measure stats retrieval overhead."""
        bus = RustMessageBus()

        # Publish some events first
        for i in range(1000):
            event = Event.create(EventType.TICK, "bench", {"price": 50000.0 + i})
            bus.publish(event)

        benchmark.pedantic(
            bus.get_stats,
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )

    @pytest.mark.bench_throughput
    def test_rust_queue_depth_overhead(
        self,
        benchmark: BenchmarkFixture,
        gc_disabled: None,
    ) -> None:
        """Measure queue depth check overhead."""
        bus = RustMessageBus()

        # Publish some events first
        for i in range(1000):
            event = Event.create(EventType.TICK, "bench", {"price": 50000.0 + i})
            bus.publish(event)

        benchmark.pedantic(
            bus.total_queue_depth,
            rounds=100,
            iterations=10_000,
            warmup_rounds=5,
        )


# =============================================================================
# Standalone Comparison Script
# =============================================================================


def run_standalone_comparison() -> None:
    """Run standalone comparison (can be executed directly)."""
    print("\n" + "=" * 70)
    print("MessageBus Performance Comparison: Python vs Rust")
    print("=" * 70)

    if not RUST_AVAILABLE:
        print("\nRust extension not available. Build with:")
        print("  cd libra-core-rs && maturin develop --release")
        return

    # Create events
    num_events = 100_000
    events = [
        Event.create(EventType.TICK, "bench", {"symbol": "BTC/USDT", "price": 50000.0 + i})
        for i in range(num_events)
    ]

    print(f"\nBenchmarking with {num_events:,} events...")

    # Python MessageBus
    print("\n[Python MessageBus]")
    py_bus = MessageBus()

    start = time.perf_counter()
    for event in events:
        py_bus.publish(event)
    py_publish_time = time.perf_counter() - start
    py_publish_throughput = num_events / py_publish_time

    print(f"  Publish: {py_publish_throughput:,.0f} events/sec")
    print(f"  Time:    {py_publish_time * 1000:.2f}ms")

    # Rust MessageBus
    print("\n[Rust MessageBus]")
    rust_bus = RustMessageBus()

    start = time.perf_counter()
    for event in events:
        rust_bus.publish(event)
    rust_publish_time = time.perf_counter() - start
    rust_publish_throughput = num_events / rust_publish_time

    print(f"  Publish: {rust_publish_throughput:,.0f} events/sec")
    print(f"  Time:    {rust_publish_time * 1000:.2f}ms")

    # Dispatch with handler
    print("\n[Rust MessageBus with Handler]")
    rust_bus2 = RustMessageBus()
    call_count = [0]

    def handler(event: Event) -> None:
        call_count[0] += 1

    rust_bus2.subscribe(EventType.TICK.value, handler)

    # Publish all events
    for event in events:
        rust_bus2.publish(event)

    # Dispatch in batches
    start = time.perf_counter()
    total_dispatched = 0
    while True:
        dispatched = rust_bus2.dispatch_batch()
        total_dispatched += dispatched
        if dispatched == 0:
            break
    rust_dispatch_time = time.perf_counter() - start
    rust_dispatch_throughput = total_dispatched / rust_dispatch_time if rust_dispatch_time > 0 else 0

    print(f"  Dispatch: {rust_dispatch_throughput:,.0f} events/sec")
    print(f"  Events:   {total_dispatched:,} dispatched, {call_count[0]:,} handled")

    # Summary
    publish_speedup = rust_publish_throughput / py_publish_throughput

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Python publish throughput:  {py_publish_throughput:>12,.0f} events/sec")
    print(f"Rust publish throughput:    {rust_publish_throughput:>12,.0f} events/sec")
    print(f"Publish speedup:            {publish_speedup:>12.1f}x")
    print("=" * 70)

    # Get Rust stats
    stats = rust_bus.get_stats()
    print(f"\nRust MessageBus Stats: {stats}")


if __name__ == "__main__":
    run_standalone_comparison()

"""
Benchmark fixtures and configuration.

Best practices applied:
- GC disabled during measurements
- Warmup rounds before measurement
- Multiple payload sizes for thorough testing
- HDR Histogram for latency percentile tracking
"""

from __future__ import annotations

import gc
import sys
from typing import TYPE_CHECKING, Any

import pytest

from libra.core.events import Event, EventType


if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# GC Control Fixtures
# =============================================================================


@pytest.fixture
def gc_disabled() -> Generator[None, None, None]:
    """Disable garbage collection during benchmark."""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    gc.collect()  # Clean slate
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


@pytest.fixture
def gc_stats() -> Generator[dict[str, Any], None, None]:
    """Collect GC statistics during benchmark."""
    gc.collect()
    stats_before = gc.get_stats()
    collections_before = [s["collections"] for s in stats_before]

    result: dict[str, Any] = {}
    yield result

    stats_after = gc.get_stats()
    collections_after = [s["collections"] for s in stats_after]

    result["gc_collections"] = [
        after - before for before, after in zip(collections_before, collections_after, strict=True)
    ]


# =============================================================================
# Payload Fixtures (varying sizes for thorough benchmarking)
# =============================================================================

PAYLOAD_SIZES = {
    "tiny": 10,  # ~10 bytes
    "small": 100,  # ~100 bytes
    "medium": 1000,  # ~1 KB
    "large": 10000,  # ~10 KB
}


def create_payload(size_bytes: int) -> dict[str, Any]:
    """Create a payload of approximately the given size."""
    # Base payload structure
    payload: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "price": 50000.12345678,
        "quantity": 0.12345678,
    }

    # Add padding to reach target size
    base_size = sys.getsizeof(str(payload))
    if size_bytes > base_size:
        padding_size = size_bytes - base_size
        payload["_pad"] = "x" * padding_size

    return payload


@pytest.fixture
def tiny_payload() -> dict[str, Any]:
    """~10 byte payload."""
    return create_payload(PAYLOAD_SIZES["tiny"])


@pytest.fixture
def small_payload() -> dict[str, Any]:
    """~100 byte payload."""
    return create_payload(PAYLOAD_SIZES["small"])


@pytest.fixture
def medium_payload() -> dict[str, Any]:
    """~1 KB payload."""
    return create_payload(PAYLOAD_SIZES["medium"])


@pytest.fixture
def large_payload() -> dict[str, Any]:
    """~10 KB payload."""
    return create_payload(PAYLOAD_SIZES["large"])


# =============================================================================
# Pre-created Event Fixtures (avoid creation overhead in hot path)
# =============================================================================


@pytest.fixture
def tick_event() -> Event:
    """Pre-created tick event for benchmarking."""
    return Event.create(
        EventType.TICK,
        "benchmark",
        {"symbol": "BTC/USDT", "price": 50000.0, "quantity": 0.1},
    )


@pytest.fixture
def tick_events_1k() -> list[Event]:
    """1000 pre-created tick events."""
    return [
        Event.create(EventType.TICK, "benchmark", {"symbol": "BTC/USDT", "price": 50000.0 + i})
        for i in range(1000)
    ]


@pytest.fixture
def tick_events_10k() -> list[Event]:
    """10000 pre-created tick events."""
    return [
        Event.create(EventType.TICK, "benchmark", {"symbol": "BTC/USDT", "price": 50000.0 + i})
        for i in range(10000)
    ]


# =============================================================================
# Benchmark Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register benchmark markers."""
    config.addinivalue_line("markers", "bench_throughput: throughput benchmarks")
    config.addinivalue_line("markers", "bench_latency: latency distribution benchmarks")
    config.addinivalue_line("markers", "bench_e2e: end-to-end benchmarks")


# =============================================================================
# Result Reporting Helpers
# =============================================================================


def format_throughput(events_per_sec: float) -> str:
    """Format throughput with appropriate unit."""
    if events_per_sec >= 1_000_000:
        return f"{events_per_sec / 1_000_000:.2f}M events/sec"
    if events_per_sec >= 1_000:
        return f"{events_per_sec / 1_000:.1f}K events/sec"
    return f"{events_per_sec:.0f} events/sec"


def format_latency(ns: float) -> str:
    """Format latency with appropriate unit."""
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f}us"
    return f"{ns:.0f}ns"

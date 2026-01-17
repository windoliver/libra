"""
Cache performance benchmarks (Issue #66).

Compares lru-dict C extension vs Python OrderedDict baseline.

Two benchmark scenarios:
1. Raw data structures (no TTL): Shows theoretical ~14x improvement
2. With CacheEntry wrapper (TTL support): Shows real-world performance

Run: pytest benchmarks/bench_cache.py -v -s
"""

from __future__ import annotations

import gc
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from lru import LRU

from libra.data.cache import LRUCache


if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Benchmark Configuration
# =============================================================================

CACHE_SIZE = 10_000
NUM_OPERATIONS = 100_000
WARMUP_ITERATIONS = 1_000


# =============================================================================
# Baseline OrderedDict Implementation (for comparison)
# =============================================================================


class OrderedDictLRUCache:
    """
    Baseline LRU cache using OrderedDict (pre-Issue #66 implementation).

    Used for benchmark comparison only.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        """Get value from cache."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: str) -> None:
        """Set value in cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# =============================================================================
# Benchmark Results
# =============================================================================


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    operations: int = 0
    duration_sec: float = 0.0
    memory_kb: int = 0
    ops_per_sec: float = field(init=False)

    def __post_init__(self) -> None:
        self.ops_per_sec = self.operations / self.duration_sec if self.duration_sec > 0 else 0

    def report(self) -> str:
        """Generate human-readable report."""
        return f"""
{self.name}
{"=" * 50}
  Operations:    {self.operations:>12,}
  Duration:      {self.duration_sec:>12.3f} sec
  Throughput:    {self.ops_per_sec:>12,.0f} ops/sec
  Memory:        {self.memory_kb:>12,} KB
"""


def get_memory_usage() -> int:
    """Get current memory usage in KB."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gc_disabled() -> Generator[None, None, None]:
    """Disable garbage collection during benchmark."""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    gc.collect()
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


# =============================================================================
# LRU Cache Benchmarks
# =============================================================================


class TestRawDataStructureBenchmark:
    """
    Benchmark raw lru-dict vs OrderedDict (no CacheEntry wrapper).

    This shows the theoretical performance improvement of the C extension.
    """

    @pytest.mark.benchmark
    def test_raw_comparison(self, gc_disabled: None) -> None:
        """Compare raw lru-dict vs OrderedDict without TTL wrapper."""
        # Raw lru-dict benchmark
        raw_lru: LRU = LRU(CACHE_SIZE)
        for i in range(WARMUP_ITERATIONS):
            raw_lru[f"w_{i}"] = f"v_{i}"

        gc.collect()
        lru_start = time.perf_counter()
        for i in range(NUM_OPERATIONS):
            key = f"key_{i % CACHE_SIZE}"
            raw_lru[key] = f"value_{i}"
            _ = raw_lru[key]
        lru_duration = time.perf_counter() - lru_start

        # Raw OrderedDict benchmark
        raw_od: OrderedDict[str, str] = OrderedDict()
        for i in range(WARMUP_ITERATIONS):
            raw_od[f"w_{i}"] = f"v_{i}"
            if len(raw_od) > CACHE_SIZE:
                raw_od.popitem(last=False)

        gc.collect()
        od_start = time.perf_counter()
        for i in range(NUM_OPERATIONS):
            key = f"key_{i % CACHE_SIZE}"
            if key in raw_od:
                raw_od.move_to_end(key)
            raw_od[key] = f"value_{i}"
            while len(raw_od) > CACHE_SIZE:
                raw_od.popitem(last=False)
            _ = raw_od.get(key)
            if key in raw_od:
                raw_od.move_to_end(key)
        od_duration = time.perf_counter() - od_start

        speed_improvement = od_duration / lru_duration if lru_duration > 0 else 0

        print(f"""
================================================================================
            RAW Data Structure Comparison (no TTL/CacheEntry overhead)
================================================================================

Configuration:
  Cache size:       {CACHE_SIZE:,} entries
  Operations:       {NUM_OPERATIONS:,} (get + set each)

Results:
                           lru-dict (C)      OrderedDict (Py)   Improvement
  -------------------------------------------------------------------------
  Duration (sec):    {lru_duration:>12.3f}      {od_duration:>12.3f}        {speed_improvement:>6.1f}x faster
  Ops/sec:           {(NUM_OPERATIONS * 2) / lru_duration:>12,.0f}      {(NUM_OPERATIONS * 2) / od_duration:>12,.0f}

This benchmark shows the raw C extension performance benefit.
================================================================================
""")

        # Performance varies by system; expect at least 1.5x improvement
        assert speed_improvement >= 1.5, f"Raw lru-dict should be faster than OrderedDict (got {speed_improvement:.1f}x)"


class TestLRUCacheBenchmark:
    """Benchmark tests for LRUCache with TTL support (real-world usage)."""

    @pytest.mark.benchmark
    def test_lru_dict_throughput(self, gc_disabled: None) -> None:
        """Benchmark lru-dict implementation throughput."""
        cache: LRUCache[str, str] = LRUCache(max_size=CACHE_SIZE)

        # Warmup
        for i in range(WARMUP_ITERATIONS):
            cache.set_sync(f"warmup_{i}", f"value_{i}")
            cache.get_sync(f"warmup_{i}")

        gc.collect()
        memory_before = get_memory_usage()

        # Benchmark: mixed get/set operations
        start = time.perf_counter()
        for i in range(NUM_OPERATIONS):
            key = f"key_{i % CACHE_SIZE}"
            cache.set_sync(key, f"value_{i}")
            cache.get_sync(key)
        duration = time.perf_counter() - start

        memory_after = get_memory_usage()

        result = BenchmarkResult(
            name="LRUCache with lru-dict (TTL enabled)",
            operations=NUM_OPERATIONS * 2,  # get + set
            duration_sec=duration,
            memory_kb=memory_after - memory_before,
        )

        print(result.report())
        assert result.ops_per_sec > 100_000, "lru-dict should exceed 100K ops/sec"

    @pytest.mark.benchmark
    def test_ordered_dict_throughput(self, gc_disabled: None) -> None:
        """Benchmark OrderedDict baseline throughput."""
        cache = OrderedDictLRUCache(max_size=CACHE_SIZE)

        # Warmup
        for i in range(WARMUP_ITERATIONS):
            cache.set(f"warmup_{i}", f"value_{i}")
            cache.get(f"warmup_{i}")

        gc.collect()
        memory_before = get_memory_usage()

        # Benchmark: mixed get/set operations
        start = time.perf_counter()
        for i in range(NUM_OPERATIONS):
            key = f"key_{i % CACHE_SIZE}"
            cache.set(key, f"value_{i}")
            cache.get(key)
        duration = time.perf_counter() - start

        memory_after = get_memory_usage()

        result = BenchmarkResult(
            name="OrderedDict (no TTL, Python baseline)",
            operations=NUM_OPERATIONS * 2,  # get + set
            duration_sec=duration,
            memory_kb=memory_after - memory_before,
        )

        print(result.report())

    @pytest.mark.benchmark
    def test_implementation_summary(self, gc_disabled: None) -> None:
        """Summary of LRUCache implementation performance."""
        # Run lru-dict benchmark
        lru_cache: LRUCache[str, str] = LRUCache(max_size=CACHE_SIZE)
        for i in range(WARMUP_ITERATIONS):
            lru_cache.set_sync(f"w_{i}", f"v_{i}")

        gc.collect()
        lru_start = time.perf_counter()
        for i in range(NUM_OPERATIONS):
            key = f"key_{i % CACHE_SIZE}"
            lru_cache.set_sync(key, f"value_{i}")
            lru_cache.get_sync(key)
        lru_duration = time.perf_counter() - lru_start

        print(f"""
================================================================================
                    LRUCache Implementation Benchmark (Issue #66)
================================================================================

Configuration:
  Cache size:       {CACHE_SIZE:,} entries
  Operations:       {NUM_OPERATIONS:,} (get + set each)

LRUCache with lru-dict C extension:
  Duration:         {lru_duration:.3f} sec
  Throughput:       {(NUM_OPERATIONS * 2) / lru_duration:,.0f} ops/sec

Note: LRUCache includes CacheEntry wrapper for TTL support.
For raw C extension performance, see test_raw_comparison.
================================================================================
""")


class TestLRUCacheEviction:
    """Benchmark LRU eviction behavior."""

    @pytest.mark.benchmark
    def test_eviction_heavy_workload(self, gc_disabled: None) -> None:
        """Benchmark with heavy eviction (more inserts than capacity)."""
        cache: LRUCache[str, str] = LRUCache(max_size=1000)

        # Insert 10x more items than capacity to trigger heavy eviction
        start = time.perf_counter()
        for i in range(10_000):
            cache.set_sync(f"key_{i}", f"value_{i}")
        duration = time.perf_counter() - start

        stats = cache.stats
        print(f"""
Eviction Heavy Workload
=======================
  Inserts:     10,000
  Cache size:   1,000
  Evictions:   {stats.evictions:>6,}
  Duration:    {duration:.3f} sec
  Inserts/sec: {10_000 / duration:,.0f}
""")

        assert stats.evictions == 9000, "Should have evicted 9000 entries"


class TestLRUCacheHitRate:
    """Benchmark cache hit rate scenarios."""

    @pytest.mark.benchmark
    def test_hot_key_access(self, gc_disabled: None) -> None:
        """Benchmark with hot key access pattern (high hit rate expected)."""
        cache: LRUCache[str, str] = LRUCache(max_size=1000)

        # Populate cache
        for i in range(1000):
            cache.set_sync(f"key_{i}", f"value_{i}")

        # Access hot keys repeatedly (keys 0-99 are hot)
        start = time.perf_counter()
        for _ in range(10_000):
            for i in range(100):
                cache.get_sync(f"key_{i}")
        duration = time.perf_counter() - start

        stats = cache.stats
        print(f"""
Hot Key Access Pattern
======================
  Accesses:    1,000,000
  Hit rate:    {stats.hit_rate_percent:.1f}%
  Duration:    {duration:.3f} sec
  Gets/sec:    {1_000_000 / duration:,.0f}
""")

        assert stats.hit_rate > 0.99, "Hot key access should have >99% hit rate"

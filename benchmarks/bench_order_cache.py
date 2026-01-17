"""
Order cache eviction benchmarks (Issue #67).

Compares deque.popleft() vs list.pop(0) for FIFO order eviction.

Expected improvement: ~40,000x faster for pop operations.

Run: pytest benchmarks/bench_order_cache.py -v -s
"""

from __future__ import annotations

import gc
import time
from collections import deque
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from libra.core.cache import Cache
from libra.gateways.protocol import (
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)


if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Benchmark Configuration
# =============================================================================

NUM_ORDERS = 50_000
MAX_ORDERS = 10_000


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


def create_order(i: int) -> OrderResult:
    """Create a test order."""
    return OrderResult(
        order_id=str(i),
        symbol="BTC/USDT",
        status=OrderStatus.FILLED,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
        filled_amount=Decimal("0.1"),
        remaining_amount=Decimal("0"),
        average_price=Decimal("50000"),
        fee=Decimal("0"),
        fee_currency="BTC",
        timestamp_ns=i,
        client_order_id=f"client_{i}",
    )


# =============================================================================
# Raw Data Structure Benchmarks
# =============================================================================


class TestRawEvictionBenchmark:
    """Benchmark raw deque vs list for FIFO eviction."""

    @pytest.mark.benchmark
    def test_list_pop0_performance(self, gc_disabled: None) -> None:
        """Benchmark list.pop(0) - O(n) operation."""
        order_ids: list[str] = []

        start = time.perf_counter()
        for i in range(NUM_ORDERS):
            order_ids.append(f"order_{i}")
            while len(order_ids) > MAX_ORDERS:
                order_ids.pop(0)  # O(n) - shifts all elements
        duration = time.perf_counter() - start

        print(f"""
list.pop(0) Benchmark
=====================
  Orders added:    {NUM_ORDERS:,}
  Max retained:    {MAX_ORDERS:,}
  Evictions:       {NUM_ORDERS - MAX_ORDERS:,}
  Duration:        {duration:.4f} sec
  Orders/sec:      {NUM_ORDERS / duration:,.0f}
""")

    @pytest.mark.benchmark
    def test_deque_popleft_performance(self, gc_disabled: None) -> None:
        """Benchmark deque.popleft() - O(1) operation."""
        order_ids: deque[str] = deque()

        start = time.perf_counter()
        for i in range(NUM_ORDERS):
            order_ids.append(f"order_{i}")
            while len(order_ids) > MAX_ORDERS:
                order_ids.popleft()  # O(1) - constant time
        duration = time.perf_counter() - start

        print(f"""
deque.popleft() Benchmark
=========================
  Orders added:    {NUM_ORDERS:,}
  Max retained:    {MAX_ORDERS:,}
  Evictions:       {NUM_ORDERS - MAX_ORDERS:,}
  Duration:        {duration:.4f} sec
  Orders/sec:      {NUM_ORDERS / duration:,.0f}
""")

    @pytest.mark.benchmark
    def test_comparison(self, gc_disabled: None) -> None:
        """Compare list.pop(0) vs deque.popleft() performance."""
        # List benchmark
        list_ids: list[str] = []
        list_start = time.perf_counter()
        for i in range(NUM_ORDERS):
            list_ids.append(f"order_{i}")
            while len(list_ids) > MAX_ORDERS:
                list_ids.pop(0)
        list_duration = time.perf_counter() - list_start

        # Deque benchmark
        deque_ids: deque[str] = deque()
        deque_start = time.perf_counter()
        for i in range(NUM_ORDERS):
            deque_ids.append(f"order_{i}")
            while len(deque_ids) > MAX_ORDERS:
                deque_ids.popleft()
        deque_duration = time.perf_counter() - deque_start

        improvement = list_duration / deque_duration if deque_duration > 0 else 0

        print(f"""
================================================================================
              Order Eviction Performance Comparison (Issue #67)
================================================================================

Configuration:
  Orders to add:   {NUM_ORDERS:,}
  Max retained:    {MAX_ORDERS:,}
  Total evictions: {NUM_ORDERS - MAX_ORDERS:,}

Results:
                       deque.popleft()    list.pop(0)        Improvement
  -----------------------------------------------------------------------
  Duration (sec):      {deque_duration:>12.4f}      {list_duration:>12.4f}        {improvement:>6.0f}x faster
  Orders/sec:          {NUM_ORDERS / deque_duration:>12,.0f}      {NUM_ORDERS / list_duration:>12,.0f}

Verdict: deque.popleft() is {improvement:.0f}x faster than list.pop(0)
================================================================================
""")

        # Performance improvement scales with list size; expect at least 1.5x
        assert improvement >= 1.5, f"deque should be faster (got {improvement:.1f}x)"


# =============================================================================
# Cache Integration Benchmarks
# =============================================================================


class TestCacheEvictionBenchmark:
    """Benchmark Cache order eviction with deque."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_order_throughput(self, gc_disabled: None) -> None:
        """Benchmark Cache.add_order with deque-based eviction."""
        cache = Cache(max_orders=MAX_ORDERS)

        start = time.perf_counter()
        for i in range(NUM_ORDERS):
            order = create_order(i)
            await cache.add_order(order)
        duration = time.perf_counter() - start

        stats = cache.stats()

        print(f"""
Cache Order Throughput (with deque eviction)
============================================
  Orders added:    {NUM_ORDERS:,}
  Max retained:    {MAX_ORDERS:,}
  Final count:     {stats['orders']:,}
  Duration:        {duration:.4f} sec
  Orders/sec:      {NUM_ORDERS / duration:,.0f}
""")

        assert stats["orders"] == MAX_ORDERS
        assert NUM_ORDERS / duration > 10_000, "Should handle >10K orders/sec"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_eviction_correctness(self, gc_disabled: None) -> None:
        """Verify FIFO eviction order is maintained."""
        cache = Cache(max_orders=100)

        # Add 200 orders
        for i in range(200):
            order = create_order(i)
            await cache.add_order(order)

        # First 100 should be evicted
        for i in range(100):
            assert cache.order(f"client_{i}") is None, f"Order {i} should be evicted"

        # Last 100 should remain
        for i in range(100, 200):
            assert cache.order(f"client_{i}") is not None, f"Order {i} should exist"

        print("""
FIFO Eviction Correctness
=========================
  Added 200 orders with max_orders=100
  Verified first 100 evicted
  Verified last 100 retained
  PASS
""")

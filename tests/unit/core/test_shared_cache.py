"""Tests for the SharedPriceCache."""

from __future__ import annotations

import multiprocessing
import time

import pytest

from libra.core.shared_cache import PriceData, SharedPriceCache


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def symbols() -> list[str]:
    """Create test symbols."""
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


@pytest.fixture
def cache(symbols: list[str]):
    """Create shared price cache for testing."""
    # Cleanup any existing shared memory
    SharedPriceCache.cleanup(name="test_cache")

    cache = SharedPriceCache(symbols, create=True, name="test_cache")
    yield cache
    cache.close()
    cache.unlink()


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBasicFunctionality:
    """Tests for basic cache operations."""

    def test_create_cache(self, symbols: list[str]) -> None:
        """Should create shared memory cache."""
        SharedPriceCache.cleanup(name="test_create")
        cache = SharedPriceCache(symbols, create=True, name="test_create")

        assert len(cache) == 3
        assert cache.symbols == symbols

        cache.close()
        cache.unlink()

    def test_update_and_get_price(self, cache: SharedPriceCache) -> None:
        """Should update and retrieve prices."""
        cache.update_all("BTC/USDT", bid=49999.0, ask=50001.0, last=50000.0, volume=100.0)

        price = cache.get_price("BTC/USDT")

        assert price.bid == 49999.0
        assert price.ask == 50001.0
        assert price.last == 50000.0
        assert price.volume == 100.0
        assert price.timestamp_ns > 0

    def test_update_individual_fields(self, cache: SharedPriceCache) -> None:
        """Should update individual price fields."""
        cache.update_price("ETH/USDT", bid=1999.0)
        cache.update_price("ETH/USDT", ask=2001.0)
        cache.update_price("ETH/USDT", last=2000.0)

        price = cache.get_price("ETH/USDT")

        assert price.bid == 1999.0
        assert price.ask == 2001.0
        assert price.last == 2000.0

    def test_get_last(self, cache: SharedPriceCache) -> None:
        """Should get just the last price."""
        cache.update_price("SOL/USDT", last=150.0)

        last = cache.get_last("SOL/USDT")

        assert last == 150.0

    def test_get_bid_ask(self, cache: SharedPriceCache) -> None:
        """Should get bid and ask prices."""
        cache.update_all("BTC/USDT", bid=49999.0, ask=50001.0, last=50000.0)

        bid, ask = cache.get_bid_ask("BTC/USDT")

        assert bid == 49999.0
        assert ask == 50001.0

    def test_get_all_prices(self, cache: SharedPriceCache) -> None:
        """Should get all prices as dictionary."""
        cache.update_all("BTC/USDT", bid=50000.0, ask=50001.0, last=50000.0)
        cache.update_all("ETH/USDT", bid=2000.0, ask=2001.0, last=2000.0)

        prices = cache.get_all_prices()

        assert len(prices) == 3
        assert "BTC/USDT" in prices
        assert "ETH/USDT" in prices
        assert prices["BTC/USDT"].last == 50000.0
        assert prices["ETH/USDT"].last == 2000.0

    def test_contains(self, cache: SharedPriceCache) -> None:
        """Should check if symbol exists."""
        assert "BTC/USDT" in cache
        assert "INVALID/PAIR" not in cache

    def test_unknown_symbol_raises(self, cache: SharedPriceCache) -> None:
        """Should raise KeyError for unknown symbol."""
        with pytest.raises(KeyError, match="Unknown symbol"):
            cache.get_price("INVALID/PAIR")

        with pytest.raises(KeyError, match="Unknown symbol"):
            cache.update_price("INVALID/PAIR", last=100.0)


# =============================================================================
# Timestamp Tests
# =============================================================================


class TestTimestamps:
    """Tests for timestamp handling."""

    def test_auto_timestamp(self, cache: SharedPriceCache) -> None:
        """Should auto-generate timestamp if not provided."""
        before = time.time_ns()
        cache.update_price("BTC/USDT", last=50000.0)
        after = time.time_ns()

        price = cache.get_price("BTC/USDT")

        assert before <= price.timestamp_ns <= after

    def test_custom_timestamp(self, cache: SharedPriceCache) -> None:
        """Should use custom timestamp when provided."""
        custom_ts = 1234567890000000000

        cache.update_price("BTC/USDT", last=50000.0, timestamp_ns=custom_ts)

        price = cache.get_price("BTC/USDT")
        assert price.timestamp_ns == custom_ts


# =============================================================================
# Multi-Process Tests
# =============================================================================


def _worker_read(name: str, symbols: list[str], results: multiprocessing.Queue) -> None:
    """Worker process that reads from shared cache."""
    try:
        cache = SharedPriceCache(symbols, create=False, name=name)
        price = cache.get_price("BTC/USDT")
        results.put(("success", price.last))
        cache.close()
    except Exception as e:
        results.put(("error", str(e)))


def _worker_write(name: str, symbols: list[str], value: float) -> None:
    """Worker process that writes to shared cache."""
    cache = SharedPriceCache(symbols, create=False, name=name)
    cache.update_price("BTC/USDT", last=value)
    cache.close()


class TestMultiProcess:
    """Tests for multi-process functionality."""

    def test_read_from_worker_process(self, symbols: list[str]) -> None:
        """Worker process should read prices from shared memory."""
        SharedPriceCache.cleanup(name="test_mp_read")
        cache = SharedPriceCache(symbols, create=True, name="test_mp_read")

        # Set price in main process
        cache.update_price("BTC/USDT", last=50000.0)

        # Read from worker process
        results: multiprocessing.Queue = multiprocessing.Queue()
        worker = multiprocessing.Process(
            target=_worker_read,
            args=("test_mp_read", symbols, results),
        )
        worker.start()
        worker.join(timeout=5)

        status, value = results.get(timeout=1)

        assert status == "success"
        assert value == 50000.0

        cache.close()
        cache.unlink()

    def test_write_from_worker_process(self, symbols: list[str]) -> None:
        """Worker process should write prices to shared memory."""
        SharedPriceCache.cleanup(name="test_mp_write")
        cache = SharedPriceCache(symbols, create=True, name="test_mp_write")

        # Write from worker process
        worker = multiprocessing.Process(
            target=_worker_write,
            args=("test_mp_write", symbols, 55000.0),
        )
        worker.start()
        worker.join(timeout=5)

        # Read in main process
        price = cache.get_price("BTC/USDT")

        assert price.last == 55000.0

        cache.close()
        cache.unlink()


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for cache lifecycle management."""

    def test_context_manager(self, symbols: list[str]) -> None:
        """Should work as context manager."""
        SharedPriceCache.cleanup(name="test_ctx")

        with SharedPriceCache(symbols, create=True, name="test_ctx") as cache:
            cache.update_price("BTC/USDT", last=50000.0)
            assert cache.get_last("BTC/USDT") == 50000.0

        # Should be closed after exit
        assert not SharedPriceCache.exists(name="test_ctx")

    def test_closed_cache_raises(self, cache: SharedPriceCache) -> None:
        """Should raise error when accessing closed cache."""
        cache.close()

        with pytest.raises(RuntimeError, match="closed"):
            cache.get_price("BTC/USDT")

        with pytest.raises(RuntimeError, match="closed"):
            cache.update_price("BTC/USDT", last=50000.0)

    def test_exists_check(self, symbols: list[str]) -> None:
        """Should check if shared memory exists."""
        SharedPriceCache.cleanup(name="test_exists")

        assert not SharedPriceCache.exists(name="test_exists")

        cache = SharedPriceCache(symbols, create=True, name="test_exists")
        assert SharedPriceCache.exists(name="test_exists")

        cache.close()
        cache.unlink()
        assert not SharedPriceCache.exists(name="test_exists")

    def test_cleanup_class_method(self, symbols: list[str]) -> None:
        """Should cleanup orphaned shared memory."""
        SharedPriceCache.cleanup(name="test_cleanup")

        # Create without cleanup
        cache = SharedPriceCache(symbols, create=True, name="test_cleanup")
        cache.close()  # Close but don't unlink

        # Should still exist
        assert SharedPriceCache.exists(name="test_cleanup")

        # Force cleanup
        result = SharedPriceCache.cleanup(name="test_cleanup")
        assert result is True

        # Should not exist
        assert not SharedPriceCache.exists(name="test_cleanup")

    def test_cleanup_nonexistent(self) -> None:
        """Cleanup should return False for nonexistent memory."""
        result = SharedPriceCache.cleanup(name="nonexistent_memory_12345")
        assert result is False


# =============================================================================
# Stats Tests
# =============================================================================


class TestStats:
    """Tests for cache statistics."""

    def test_stats_tracking(self, cache: SharedPriceCache) -> None:
        """Should track read/write stats."""
        # Initial stats
        stats = cache.stats
        assert stats["reads"] == 0
        assert stats["writes"] == 0

        # Write
        cache.update_price("BTC/USDT", last=50000.0)
        assert cache.stats["writes"] == 1

        # Read
        cache.get_price("BTC/USDT")
        assert cache.stats["reads"] == 1

        # Multiple reads
        cache.get_last("BTC/USDT")
        cache.get_bid_ask("BTC/USDT")
        assert cache.stats["reads"] == 3

    def test_stats_content(self, cache: SharedPriceCache) -> None:
        """Should include all stat fields."""
        stats = cache.stats

        assert "reads" in stats
        assert "writes" in stats
        assert "symbols" in stats
        assert "size_bytes" in stats
        assert stats["symbols"] == 3


# =============================================================================
# PriceData Tests
# =============================================================================


class TestPriceData:
    """Tests for PriceData named tuple."""

    def test_price_data_fields(self) -> None:
        """PriceData should have correct fields."""
        price = PriceData(
            bid=49999.0,
            ask=50001.0,
            last=50000.0,
            volume=100.0,
            timestamp_ns=1234567890,
        )

        assert price.bid == 49999.0
        assert price.ask == 50001.0
        assert price.last == 50000.0
        assert price.volume == 100.0
        assert price.timestamp_ns == 1234567890

    def test_price_data_immutable(self) -> None:
        """PriceData should be immutable."""
        price = PriceData(bid=1.0, ask=2.0, last=1.5, volume=100.0, timestamp_ns=0)

        with pytest.raises(AttributeError):
            price.bid = 999.0  # type: ignore[misc]


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.benchmark
    def test_read_throughput(self, cache: SharedPriceCache) -> None:
        """Measure read throughput."""
        cache.update_price("BTC/USDT", last=50000.0)
        iterations = 100_000

        start = time.perf_counter_ns()
        for _ in range(iterations):
            cache.get_last("BTC/USDT")
        elapsed = time.perf_counter_ns() - start

        per_read = elapsed / iterations
        throughput = 1e9 / per_read

        print(f"\nRead: {per_read:.0f}ns/op = {throughput:,.0f} ops/sec")

        # Should be sub-microsecond
        assert per_read < 1000, f"Read too slow: {per_read}ns"

    @pytest.mark.benchmark
    def test_write_throughput(self, cache: SharedPriceCache) -> None:
        """Measure write throughput."""
        iterations = 100_000

        start = time.perf_counter_ns()
        for i in range(iterations):
            cache.update_price("BTC/USDT", last=50000.0 + i)
        elapsed = time.perf_counter_ns() - start

        per_write = elapsed / iterations
        throughput = 1e9 / per_write

        print(f"\nWrite: {per_write:.0f}ns/op = {throughput:,.0f} ops/sec")

        # Should be sub-microsecond
        assert per_write < 1000, f"Write too slow: {per_write}ns"

    @pytest.mark.benchmark
    def test_full_update_throughput(self, cache: SharedPriceCache) -> None:
        """Measure full update throughput."""
        iterations = 100_000

        start = time.perf_counter_ns()
        for i in range(iterations):
            cache.update_all(
                "BTC/USDT",
                bid=49999.0 + i,
                ask=50001.0 + i,
                last=50000.0 + i,
                volume=100.0,
            )
        elapsed = time.perf_counter_ns() - start

        per_op = elapsed / iterations
        throughput = 1e9 / per_op

        print(f"\nFull update: {per_op:.0f}ns/op = {throughput:,.0f} ops/sec")

        # Should be sub-microsecond
        assert per_op < 2000, f"Full update too slow: {per_op}ns"

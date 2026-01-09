"""Tests for Data Cache (Issue #23).

Tests:
- LRU cache operations
- TTL expiration
- Cache statistics
- DataFrame-specific caching
- Aggregation cache
"""

import asyncio
import time

import polars as pl
import pytest

from libra.data.cache import (
    AggregationCache,
    CacheEntry,
    CacheStats,
    DataFrameCache,
    LRUCache,
)


class TestCacheStats:
    """Tests for CacheStats."""

    def test_initial_stats(self) -> None:
        """Test initial statistics."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        assert stats.size == 0

    def test_hit_rate_empty(self) -> None:
        """Test hit rate with no accesses."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
        assert stats.hit_rate_percent == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)

        assert stats.hit_rate == 0.8
        assert stats.hit_rate_percent == 80.0

    def test_reset(self) -> None:
        """Test resetting stats."""
        stats = CacheStats(hits=100, misses=50)
        stats.reset()

        assert stats.hits == 0
        assert stats.misses == 0


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_entry_creation(self) -> None:
        """Test creating cache entry."""
        entry = CacheEntry(
            value="test_value",
            created_at=time.time(),
            expires_at=None,
        )

        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert entry.is_expired() is False

    def test_entry_expiration(self) -> None:
        """Test entry expiration."""
        # Expired entry
        entry = CacheEntry(
            value="test",
            created_at=time.time() - 100,
            expires_at=time.time() - 50,
        )

        assert entry.is_expired() is True

        # Not expired
        entry2 = CacheEntry(
            value="test",
            created_at=time.time(),
            expires_at=time.time() + 100,
        )

        assert entry2.is_expired() is False

    def test_entry_touch(self) -> None:
        """Test touch updates metadata."""
        entry = CacheEntry(
            value="test",
            created_at=time.time(),
            expires_at=None,
        )

        initial_access = entry.last_access
        entry.touch()

        assert entry.access_count == 1
        assert entry.last_access >= initial_access


class TestLRUCache:
    """Tests for LRUCache."""

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        """Test basic set and get."""
        cache: LRUCache[str, str] = LRUCache(max_size=10)

        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_miss(self) -> None:
        """Test cache miss returns None."""
        cache: LRUCache[str, str] = LRUCache()

        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """Test LRU eviction when max size reached."""
        cache: LRUCache[str, int] = LRUCache(max_size=3)

        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)

        # Access 'a' to make it recent
        await cache.get("a")

        # Add 'd', should evict 'b' (least recently used)
        await cache.set("d", 4)

        assert await cache.get("a") == 1  # Still there
        assert await cache.get("b") is None  # Evicted
        assert await cache.get("c") == 3  # Still there
        assert await cache.get("d") == 4  # Newly added

    @pytest.mark.asyncio
    async def test_ttl_expiration(self) -> None:
        """Test TTL-based expiration."""
        cache: LRUCache[str, str] = LRUCache(default_ttl=0.1)

        await cache.set("key", "value")
        assert await cache.get("key") == "value"

        # Wait for TTL
        await asyncio.sleep(0.15)

        result = await cache.get("key")
        assert result is None
        assert cache.stats.expirations >= 1

    @pytest.mark.asyncio
    async def test_custom_ttl(self) -> None:
        """Test custom TTL per entry."""
        cache: LRUCache[str, str] = LRUCache(default_ttl=10)

        await cache.set("short", "value", ttl=0.1)
        await asyncio.sleep(0.15)

        assert await cache.get("short") is None

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test delete operation."""
        cache: LRUCache[str, str] = LRUCache()

        await cache.set("key", "value")
        result = await cache.delete("key")

        assert result is True
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        """Test deleting nonexistent key."""
        cache: LRUCache[str, str] = LRUCache()

        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clearing cache."""
        cache: LRUCache[str, int] = LRUCache()

        await cache.set("a", 1)
        await cache.set("b", 2)

        count = await cache.clear()

        assert count == 2
        assert await cache.get("a") is None
        assert await cache.get("b") is None

    @pytest.mark.asyncio
    async def test_contains(self) -> None:
        """Test contains check."""
        cache: LRUCache[str, str] = LRUCache()

        await cache.set("key", "value")

        assert await cache.contains("key") is True
        assert await cache.contains("other") is False

    @pytest.mark.asyncio
    async def test_keys(self) -> None:
        """Test getting all keys."""
        cache: LRUCache[str, int] = LRUCache()

        await cache.set("a", 1)
        await cache.set("b", 2)

        keys = await cache.keys()

        assert "a" in keys
        assert "b" in keys

    @pytest.mark.asyncio
    async def test_cleanup_expired(self) -> None:
        """Test cleanup of expired entries."""
        cache: LRUCache[str, str] = LRUCache(default_ttl=0.1)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await asyncio.sleep(0.15)

        count = await cache.cleanup_expired()
        assert count == 2

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Test statistics are tracked correctly."""
        cache: LRUCache[str, str] = LRUCache(max_size=2)

        await cache.set("a", "1")
        await cache.get("a")  # Hit
        await cache.get("b")  # Miss
        await cache.set("b", "2")
        await cache.set("c", "3")  # Evicts 'a'

        stats = cache.stats

        assert stats.hits >= 1
        assert stats.misses >= 1
        assert stats.evictions >= 1

    def test_sync_get_set(self) -> None:
        """Test synchronous get and set."""
        cache: LRUCache[str, str] = LRUCache()

        cache.set_sync("key", "value")
        result = cache.get_sync("key")

        assert result == "value"


class TestDataFrameCache:
    """Tests for DataFrameCache."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame."""
        return pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "open": [100.0, 101.0, 102.0],
                "close": [101.0, 102.0, 103.0],
            }
        )

    @pytest.mark.asyncio
    async def test_set_and_get_bars(self, sample_df: pl.DataFrame) -> None:
        """Test setting and getting bars."""
        cache = DataFrameCache()

        await cache.set_bars("BTC/USDT", "1m", sample_df)
        result = await cache.get_bars("BTC/USDT", "1m")

        assert result is not None
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_bars_with_time_filter(self, sample_df: pl.DataFrame) -> None:
        """Test getting bars with time filter."""
        cache = DataFrameCache()

        await cache.set_bars("BTC/USDT", "1m", sample_df)
        result = await cache.get_bars("BTC/USDT", "1m", start=2, end=3)

        assert result is not None
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_bars_miss(self) -> None:
        """Test cache miss returns None."""
        cache = DataFrameCache()

        result = await cache.get_bars("UNKNOWN", "1m")
        assert result is None

    @pytest.mark.asyncio
    async def test_append_bars(self, sample_df: pl.DataFrame) -> None:
        """Test appending bars."""
        cache = DataFrameCache()

        await cache.set_bars("BTC/USDT", "1m", sample_df)

        new_bars = pl.DataFrame(
            {
                "timestamp": [4, 5],
                "open": [103.0, 104.0],
                "close": [104.0, 105.0],
            }
        )

        await cache.append_bars("BTC/USDT", "1m", new_bars)

        result = await cache.get_bars("BTC/USDT", "1m")
        assert result is not None
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_append_bars_with_max_rows(self) -> None:
        """Test append respects max rows."""
        cache = DataFrameCache()

        # Create large DataFrame
        df = pl.DataFrame(
            {"timestamp": list(range(100)), "close": [float(i) for i in range(100)]}
        )

        await cache.set_bars("BTC/USDT", "1m", df)

        new_bars = pl.DataFrame(
            {"timestamp": list(range(100, 110)), "close": [float(i) for i in range(100, 110)]}
        )

        await cache.append_bars("BTC/USDT", "1m", new_bars, max_rows=50)

        result = await cache.get_bars("BTC/USDT", "1m")
        assert result is not None
        assert len(result) == 50

    @pytest.mark.asyncio
    async def test_invalidate(self, sample_df: pl.DataFrame) -> None:
        """Test invalidating cached data."""
        cache = DataFrameCache()

        await cache.set_bars("BTC/USDT", "1m", sample_df)
        await cache.set_bars("BTC/USDT", "5m", sample_df)
        await cache.set_bars("ETH/USDT", "1m", sample_df)

        # Invalidate specific timeframe
        count = await cache.invalidate("BTC/USDT", "1m")
        assert count == 1

        assert await cache.get_bars("BTC/USDT", "1m") is None
        assert await cache.get_bars("BTC/USDT", "5m") is not None

    @pytest.mark.asyncio
    async def test_invalidate_all_timeframes(self, sample_df: pl.DataFrame) -> None:
        """Test invalidating all timeframes for symbol."""
        cache = DataFrameCache()

        await cache.set_bars("BTC/USDT", "1m", sample_df)
        await cache.set_bars("BTC/USDT", "5m", sample_df)

        count = await cache.invalidate("BTC/USDT")
        assert count == 2

    def test_stats(self) -> None:
        """Test getting stats."""
        cache = DataFrameCache()
        stats = cache.stats

        assert isinstance(stats, CacheStats)


class TestAggregationCache:
    """Tests for AggregationCache."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame."""
        return pl.DataFrame({"timestamp": [1, 2, 3], "close": [100.0, 101.0, 102.0]})

    @pytest.mark.asyncio
    async def test_set_and_get(self, sample_df: pl.DataFrame) -> None:
        """Test setting and getting aggregation."""
        cache = AggregationCache()

        await cache.set("BTC/USDT", "1m", sample_df)
        result = await cache.get("BTC/USDT", "1m")

        assert result is not None
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_invalid_timeframe(self) -> None:
        """Test getting invalid timeframe returns None."""
        cache = AggregationCache()

        result = await cache.get("BTC/USDT", "invalid")
        assert result is None

    @pytest.mark.asyncio
    async def test_aggregation_levels(self, sample_df: pl.DataFrame) -> None:
        """Test standard aggregation levels are available."""
        cache = AggregationCache()

        # Should support standard levels
        for level in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            await cache.set("BTC/USDT", level, sample_df)
            result = await cache.get("BTC/USDT", level)
            assert result is not None

    def test_get_stats(self) -> None:
        """Test getting stats for all levels."""
        cache = AggregationCache()
        stats = cache.get_stats()

        assert "1m" in stats
        assert "5m" in stats
        assert "1h" in stats
        assert "1d" in stats

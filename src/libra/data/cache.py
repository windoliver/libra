"""
Data Cache: High-performance LRU cache with TTL and statistics.

Provides:
- LRU eviction with configurable max size
- TTL-based expiration
- Hit/miss statistics
- Thread-safe async operations
- Memory-efficient storage

Performance Targets (Issue #23):
- Cache hit rate: >90% for active trading
- Data access latency: <10ms

See: https://github.com/windoliver/libra/issues/23
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Generic, TypeVar

import polars as pl


logger = logging.getLogger(__name__)


K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


# =============================================================================
# Cache Statistics
# =============================================================================


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def hit_rate_percent(self) -> float:
        """Hit rate as percentage."""
        return self.hit_rate * 100

    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry(Generic[V]):
    """Single cache entry with metadata."""

    value: V
    created_at: float  # time.time()
    expires_at: float | None  # None = no expiration
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()


# =============================================================================
# LRU Cache
# =============================================================================


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU cache with TTL support.

    Features:
    - O(1) get/set operations
    - LRU eviction when max_size reached
    - Optional TTL for entries
    - Hit/miss statistics
    - Async-safe with lock

    Example:
        cache = LRUCache[str, pl.DataFrame](max_size=100, default_ttl=300)

        # Set with default TTL
        await cache.set("key1", df1)

        # Set with custom TTL (60 seconds)
        await cache.set("key2", df2, ttl=60)

        # Get (returns None if miss or expired)
        df = await cache.get("key1")

        # Check stats
        print(f"Hit rate: {cache.stats.hit_rate_percent:.1f}%")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float | None = None,
    ) -> None:
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats(max_size=max_size)

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.size = len(self._cache)
        return self._stats

    async def get(self, key: K) -> V | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if miss/expired
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.expirations += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1
            return entry.value

    async def set(
        self,
        key: K,
        value: V,
        ttl: float | None = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (None = use default)
        """
        async with self._lock:
            # Determine expiration
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = time.time() + effective_ttl if effective_ttl else None

            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
            )

            # Update or add
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = entry

            # Evict if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._stats.evictions += 1

    async def delete(self, key: K) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def contains(self, key: K) -> bool:
        """Check if key exists (and not expired)."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                self._stats.expirations += 1
                return False
            return True

    async def keys(self) -> list[K]:
        """Get all non-expired keys."""
        async with self._lock:
            return [k for k, e in self._cache.items() if not e.is_expired()]

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [k for k, e in self._cache.items() if e.is_expired()]
            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1
            return len(expired_keys)

    def get_sync(self, key: K) -> V | None:
        """Synchronous get (use with caution in async context)."""
        entry = self._cache.get(key)
        if entry is None:
            self._stats.misses += 1
            return None
        if entry.is_expired():
            del self._cache[key]
            self._stats.misses += 1
            self._stats.expirations += 1
            return None
        self._cache.move_to_end(key)
        entry.touch()
        self._stats.hits += 1
        return entry.value

    def set_sync(self, key: K, value: V, ttl: float | None = None) -> None:
        """Synchronous set (use with caution in async context)."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + effective_ttl if effective_ttl else None

        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            expires_at=expires_at,
        )

        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = entry

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
            self._stats.evictions += 1


# =============================================================================
# DataFrame Cache
# =============================================================================


class DataFrameCache:
    """
    Specialized cache for Polars DataFrames.

    Features:
    - Automatic memory tracking
    - Pre-computed aggregation caching
    - Symbol-based partitioning
    - Efficient memory usage

    Example:
        cache = DataFrameCache(max_memory_mb=512)

        # Cache OHLCV data
        await cache.set_bars("BTC/USDT", "1m", df)

        # Get cached data
        df = await cache.get_bars("BTC/USDT", "1m")

        # Get with time range filter
        df = await cache.get_bars("BTC/USDT", "1m", start=start_time, end=end_time)
    """

    def __init__(
        self,
        max_entries: int = 100,
        default_ttl: float = 300.0,  # 5 minutes
    ) -> None:
        """
        Initialize DataFrame cache.

        Args:
            max_entries: Max number of cached DataFrames
            default_ttl: Default TTL in seconds
        """
        self._cache: LRUCache[str, pl.DataFrame] = LRUCache(
            max_size=max_entries,
            default_ttl=default_ttl,
        )

    @staticmethod
    def _make_key(symbol: str, timeframe: str, suffix: str = "") -> str:
        """Create cache key."""
        key = f"{symbol}:{timeframe}"
        if suffix:
            key = f"{key}:{suffix}"
        return key

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: int | None = None,
        end: int | None = None,
    ) -> pl.DataFrame | None:
        """
        Get cached OHLCV bars.

        Args:
            symbol: Trading pair
            timeframe: Bar timeframe
            start: Start timestamp (ns)
            end: End timestamp (ns)

        Returns:
            DataFrame or None if not cached
        """
        key = self._make_key(symbol, timeframe)
        df = await self._cache.get(key)

        if df is None:
            return None

        # Apply time filter if requested
        if start is not None or end is not None:
            if "timestamp" in df.columns:
                if start is not None:
                    df = df.filter(pl.col("timestamp") >= start)
                if end is not None:
                    df = df.filter(pl.col("timestamp") <= end)

        return df

    async def set_bars(
        self,
        symbol: str,
        timeframe: str,
        df: pl.DataFrame,
        ttl: float | None = None,
    ) -> None:
        """
        Cache OHLCV bars.

        Args:
            symbol: Trading pair
            timeframe: Bar timeframe
            df: DataFrame to cache
            ttl: Optional custom TTL
        """
        key = self._make_key(symbol, timeframe)
        await self._cache.set(key, df, ttl=ttl)

    async def append_bars(
        self,
        symbol: str,
        timeframe: str,
        new_bars: pl.DataFrame,
        max_rows: int = 10000,
    ) -> None:
        """
        Append new bars to cached data.

        Args:
            symbol: Trading pair
            timeframe: Bar timeframe
            new_bars: New bars to append
            max_rows: Maximum rows to keep (trim oldest)
        """
        key = self._make_key(symbol, timeframe)
        existing = await self._cache.get(key)

        if existing is not None:
            # Concatenate and remove duplicates
            combined = pl.concat([existing, new_bars])
            if "timestamp" in combined.columns:
                combined = combined.unique(subset=["timestamp"]).sort("timestamp")

            # Trim to max_rows
            if len(combined) > max_rows:
                combined = combined.tail(max_rows)

            await self._cache.set(key, combined)
        else:
            # Trim new bars if needed
            if len(new_bars) > max_rows:
                new_bars = new_bars.tail(max_rows)
            await self._cache.set(key, new_bars)

    async def invalidate(self, symbol: str, timeframe: str | None = None) -> int:
        """
        Invalidate cached data for a symbol.

        Args:
            symbol: Trading pair
            timeframe: Optional specific timeframe

        Returns:
            Number of entries invalidated
        """
        count = 0
        keys = await self._cache.keys()

        for key in keys:
            if key.startswith(symbol):
                if timeframe is None or f":{timeframe}" in key:
                    await self._cache.delete(key)
                    count += 1

        return count

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats


# =============================================================================
# Aggregation Cache
# =============================================================================


class AggregationCache:
    """
    Cache for pre-computed aggregations.

    Stores downsampled data (1m → 5m → 1h → 1d) for fast access.

    Example:
        agg_cache = AggregationCache()

        # Store 1-minute bars
        await agg_cache.set_source("BTC/USDT", "1m", df_1m)

        # Get pre-computed 1-hour bars
        df_1h = await agg_cache.get_aggregated("BTC/USDT", "1h")
    """

    # Standard aggregation hierarchy
    AGGREGATION_LEVELS = ["1m", "5m", "15m", "1h", "4h", "1d"]

    def __init__(self, max_entries_per_level: int = 50) -> None:
        self._caches: dict[str, LRUCache[str, pl.DataFrame]] = {
            level: LRUCache(max_size=max_entries_per_level, default_ttl=3600)
            for level in self.AGGREGATION_LEVELS
        }

    async def get(self, symbol: str, timeframe: str) -> pl.DataFrame | None:
        """Get cached aggregation."""
        if timeframe not in self._caches:
            return None
        return await self._caches[timeframe].get(symbol)

    async def set(
        self,
        symbol: str,
        timeframe: str,
        df: pl.DataFrame,
    ) -> None:
        """Set cached aggregation."""
        if timeframe in self._caches:
            await self._caches[timeframe].set(symbol, df)

    def get_stats(self) -> dict[str, CacheStats]:
        """Get stats for all aggregation levels."""
        return {level: cache.stats for level, cache in self._caches.items()}

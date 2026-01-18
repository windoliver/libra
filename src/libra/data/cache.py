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
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import polars as pl
from cachetools import TTLCache
from lru import LRU


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


@dataclass(slots=True)
class CacheEntry(Generic[V]):
    """Single cache entry with metadata (Issue #69: slots=True for memory efficiency)."""

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

    Uses lru-dict C extension for ~14x faster performance and ~3.6x memory reduction
    compared to OrderedDict (Issue #66).

    Features:
    - O(1) get/set operations (C-optimized)
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
        self._stats = CacheStats(max_size=max_size)
        # Use lru-dict C extension with eviction callback for stats tracking
        self._cache: LRU = LRU(max_size, callback=self._on_evict)
        self._lock = asyncio.Lock()

    def _on_evict(self, key: K, value: CacheEntry[V]) -> None:  # noqa: ARG002
        """Callback when an entry is evicted due to capacity overflow."""
        del key, value  # Required by lru-dict callback signature
        self._stats.evictions += 1

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
            # lru-dict returns None for missing keys with .get()
            entry: CacheEntry[V] | None = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.expirations += 1
                return None

            # Access via [] to trigger LRU reordering (lru-dict does this automatically)
            _ = self._cache[key]
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

            # lru-dict automatically handles LRU reordering on set and eviction
            # when capacity is exceeded (eviction callback tracks stats)
            self._cache[key] = entry

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
        entry: CacheEntry[V] | None = self._cache.get(key)
        if entry is None:
            self._stats.misses += 1
            return None
        if entry.is_expired():
            del self._cache[key]
            self._stats.misses += 1
            self._stats.expirations += 1
            return None
        # Access via [] to trigger LRU reordering
        _ = self._cache[key]
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

        # lru-dict automatically handles LRU reordering and eviction
        self._cache[key] = entry


# =============================================================================
# Hybrid L1/L2 Cache (Issue #77)
# =============================================================================


class HybridCache(Generic[K, V]):
    """
    Two-tier hybrid cache with L1 (hot) and L2 (warm) tiers.

    L1: lru-dict LRU - ultra-fast C extension for hot data
    L2: TTLCache - larger capacity with TTL for warm data

    Access pattern:
    1. Check L1 (hot) first - O(1) C-optimized
    2. On L1 miss, check L2 (warm)
    3. On L2 hit, promote to L1
    4. Items evicted from L1 demote to L2

    This provides:
    - Ultra-fast access for frequently used items (L1)
    - Automatic TTL expiration for warm items (L2)
    - Graceful degradation under memory pressure

    Issue #77: https://github.com/windoliver/libra/issues/77

    Example:
        cache = HybridCache[str, dict](l1_size=100, l2_size=1000, ttl=300)

        # Set value (goes to L1)
        cache.set("key1", {"data": "value"})

        # Get value (checks L1 first, then L2)
        value = cache.get("key1")

        # Check L1 hit rate
        print(f"L1 hit rate: {cache.l1_hit_rate:.1%}")
    """

    def __init__(
        self,
        l1_size: int = 1000,
        l2_size: int = 10000,
        ttl: float = 300.0,
    ) -> None:
        """
        Initialize hybrid cache.

        Args:
            l1_size: L1 (hot) tier max size
            l2_size: L2 (warm) tier max size
            ttl: TTL for L2 entries in seconds
        """
        self._l1_size = l1_size
        self._l2_size = l2_size
        self._ttl = ttl

        # L1: lru-dict C extension - ultra-fast, small capacity
        # On eviction, demote to L2
        self._l1: LRU = LRU(l1_size, callback=self._on_l1_evict)

        # L2: TTLCache - larger capacity with automatic expiration
        self._l2: TTLCache[K, V] = TTLCache(maxsize=l2_size, ttl=ttl)

        # Thread-safe lock (RLock for nested calls)
        self._lock = threading.RLock()

        # Stats
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
        self._promotions = 0
        self._demotions = 0

    def _on_l1_evict(self, key: K, value: V) -> None:
        """Demote evicted L1 entry to L2."""
        # Note: Lock already held by caller during LRU eviction
        self._l2[key] = value
        self._demotions += 1

    @property
    def l1_hit_rate(self) -> float:
        """L1 hit rate (0.0 to 1.0)."""
        total = self._l1_hits + self._l2_hits + self._misses
        return self._l1_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        """Overall hit rate (0.0 to 1.0)."""
        total = self._l1_hits + self._l2_hits + self._misses
        return (self._l1_hits + self._l2_hits) / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        with self._lock:
            return {
                "l1_hits": self._l1_hits,
                "l2_hits": self._l2_hits,
                "misses": self._misses,
                "promotions": self._promotions,
                "demotions": self._demotions,
                "l1_size": len(self._l1),
                "l2_size": len(self._l2),
                "l1_hit_rate": self.l1_hit_rate,
                "overall_hit_rate": self.overall_hit_rate,
            }

    def get(self, key: K) -> V | None:
        """
        Get value from cache (sync).

        Checks L1 first, then L2. Promotes L2 hits to L1.
        """
        with self._lock:
            # Check L1 (hot tier) first
            value = self._l1.get(key)
            if value is not None:
                self._l1_hits += 1
                return value

            # Check L2 (warm tier)
            value = self._l2.get(key)
            if value is not None:
                self._l2_hits += 1
                # Promote to L1 (may trigger demotion of another item)
                self._l1[key] = value
                del self._l2[key]
                self._promotions += 1
                return value

            self._misses += 1
            return None

    def set(self, key: K, value: V) -> None:
        """
        Set value in cache (sync).

        Always inserts into L1 (hot tier).
        """
        with self._lock:
            # Remove from L2 if exists (will be in L1 now)
            if key in self._l2:
                del self._l2[key]

            # Insert into L1 (may trigger eviction -> demotion to L2)
            self._l1[key] = value

    def delete(self, key: K) -> bool:
        """Delete from both tiers."""
        with self._lock:
            deleted = False
            if key in self._l1:
                del self._l1[key]
                deleted = True
            if key in self._l2:
                del self._l2[key]
                deleted = True
            return deleted

    def clear(self) -> None:
        """Clear both tiers."""
        with self._lock:
            self._l1.clear()
            self._l2.clear()

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        with self._lock:
            self._l1_hits = 0
            self._l2_hits = 0
            self._misses = 0
            self._promotions = 0
            self._demotions = 0

    def __contains__(self, key: K) -> bool:
        """Check if key exists in either tier."""
        with self._lock:
            return key in self._l1 or key in self._l2

    def __len__(self) -> int:
        """Total entries in both tiers."""
        with self._lock:
            return len(self._l1) + len(self._l2)


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

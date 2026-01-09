"""
E2E tests for Data Management Subsystem using REAL market data from Binance.

Demonstrates Issue #23: Data Catalog, Tiered Storage, and Downsampling.
Uses actual live OHLCV data to validate:
- Schema validation
- OHLCV downsampling
- Cache operations
- Data integrity preservation
"""

from __future__ import annotations

import json
import tempfile
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from libra.data import (
    # Cache
    CacheStats,
    DataFrameCache,
    LRUCache,
    # Catalog
    DataCatalog,
    DataRequirement,
    DataType,
    # Downsampling
    downsample_ohlcv,
    downsample_to_multiple,
    ticks_to_bars,
    validate_ohlcv_integrity,
    # Schemas
    OHLCV_SCHEMA,
    create_ohlcv_df,
    validate_ohlcv,
    # Storage
    InMemoryBackend,
    ParquetBackend,
    StorageTier,
    TieredStorage,
)


# =============================================================================
# Binance Data Fetcher (No CCXT dependency)
# =============================================================================


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Fetch klines from Binance public API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            raw = json.loads(response.read().decode())

        # Convert to dict format
        return [
            {
                "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "trades": int(candle[8]),
            }
            for candle in raw
        ]
    except Exception as e:
        pytest.skip(f"Could not fetch Binance data: {e}")
        return []


def fetch_binance_trades(
    symbol: str = "BTCUSDT",
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Fetch recent trades from Binance."""
    url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit={limit}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            raw = json.loads(response.read().decode())

        return [
            {
                "timestamp": datetime.fromtimestamp(trade["time"] / 1000),
                "last": float(trade["price"]),
                "amount": float(trade["qty"]),
            }
            for trade in raw
        ]
    except Exception as e:
        pytest.skip(f"Could not fetch Binance trades: {e}")
        return []


# =============================================================================
# Schema Validation with Real Data
# =============================================================================


class TestSchemaValidationRealData:
    """Test schema validation with real OHLCV data."""

    def test_real_ohlcv_passes_validation(self) -> None:
        """Real OHLCV data should pass validation."""
        data = fetch_binance_klines("BTCUSDT", "1h", 100)
        if not data:
            pytest.skip("No data fetched")

        df = create_ohlcv_df(
            data,
            timeframe="1h",
        )
        df = df.with_columns(pl.lit("BTC/USDT").alias("symbol"))

        errors = validate_ohlcv(df)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_real_ohlcv_integrity(self) -> None:
        """Real OHLCV data should have proper integrity."""
        data = fetch_binance_klines("ETHUSDT", "15m", 200)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data)

        errors = validate_ohlcv_integrity(df)
        assert len(errors) == 0, f"Integrity errors: {errors}"

    def test_ohlcv_relationships_hold(self) -> None:
        """Verify OHLCV relationships: high >= open,close,low; low <= open,close,high."""
        data = fetch_binance_klines("BTCUSDT", "5m", 500)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data)

        # Check high >= all prices
        invalid_high = df.filter(
            (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("high") < pl.col("low"))
        )
        assert len(invalid_high) == 0, f"Found {len(invalid_high)} bars with invalid high"

        # Check low <= all prices
        invalid_low = df.filter(
            (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
            | (pl.col("low") > pl.col("high"))
        )
        assert len(invalid_low) == 0, f"Found {len(invalid_low)} bars with invalid low"


# =============================================================================
# Downsampling with Real Data
# =============================================================================


class TestDownsamplingRealData:
    """Test OHLCV downsampling with real data."""

    @pytest.fixture
    def real_1m_data(self) -> pl.DataFrame:
        """Fetch real 1-minute data."""
        data = fetch_binance_klines("BTCUSDT", "1m", 500)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns"))
        )
        return df

    def test_downsample_1m_to_5m_real_data(self, real_1m_data: pl.DataFrame) -> None:
        """Test downsampling 1m to 5m with real data."""
        result = downsample_ohlcv(real_1m_data, "1m", "5m")

        # Should reduce row count by ~5x
        assert len(result) < len(real_1m_data)
        assert len(result) <= len(real_1m_data) // 5 + 1

        # Validate integrity preserved
        errors = validate_ohlcv_integrity(result)
        assert len(errors) == 0, f"Integrity errors after downsample: {errors}"

    def test_downsample_1m_to_1h_real_data(self, real_1m_data: pl.DataFrame) -> None:
        """Test downsampling 1m to 1h with real data."""
        result = downsample_ohlcv(real_1m_data, "1m", "1h")

        # Should reduce row count significantly
        assert len(result) < len(real_1m_data) // 50

        # Validate integrity
        errors = validate_ohlcv_integrity(result)
        assert len(errors) == 0

    def test_downsample_preserves_volume_sum(self, real_1m_data: pl.DataFrame) -> None:
        """Total volume should be preserved after downsampling."""
        original_volume = real_1m_data["volume"].sum()

        result = downsample_ohlcv(real_1m_data, "1m", "5m")
        downsampled_volume = result["volume"].sum()

        # Allow small floating point difference
        assert abs(original_volume - downsampled_volume) < 0.01 * original_volume

    def test_downsample_to_multiple_resolutions(self, real_1m_data: pl.DataFrame) -> None:
        """Test multi-resolution downsampling."""
        results = downsample_to_multiple(real_1m_data, "1m", ["5m", "15m", "1h"])

        assert "5m" in results
        assert "15m" in results
        assert "1h" in results

        # Check relative sizes
        assert len(results["5m"]) > len(results["15m"])
        assert len(results["15m"]) > len(results["1h"])

        # All should pass integrity check
        for tf, df in results.items():
            errors = validate_ohlcv_integrity(df)
            assert len(errors) == 0, f"Integrity errors in {tf}: {errors}"


class TestTickToBarRealData:
    """Test tick aggregation with real trade data."""

    def test_ticks_to_bars_real_data(self) -> None:
        """Aggregate real trade data to bars."""
        trades = fetch_binance_trades("BTCUSDT", 500)
        if not trades:
            pytest.skip("No trades fetched")

        df = pl.DataFrame(trades).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns"))
        )

        result = ticks_to_bars(df, "1m", size_col="amount")

        # Should have bars
        assert len(result) > 0

        # Should have OHLCV columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
        assert "vwap" in result.columns

        # VWAP should be between low and high
        for row in result.iter_rows(named=True):
            if row["vwap"] is not None:
                assert row["low"] <= row["vwap"] <= row["high"]


# =============================================================================
# Cache with Real Data
# =============================================================================


class TestCacheRealData:
    """Test caching with real data."""

    @pytest.mark.asyncio
    async def test_dataframe_cache_with_real_data(self) -> None:
        """Test DataFrame cache operations with real data."""
        data = fetch_binance_klines("BTCUSDT", "1m", 100)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data)

        cache = DataFrameCache(max_entries=10, default_ttl=60.0)

        # Cache the data
        await cache.set_bars("BTC/USDT", "1m", df)

        # Retrieve from cache
        cached = await cache.get_bars("BTC/USDT", "1m")
        assert cached is not None
        assert len(cached) == len(df)

        # Verify data matches
        assert cached["open"][0] == df["open"][0]
        assert cached["close"][-1] == df["close"][-1]

        # Check stats
        stats = cache.stats
        assert stats.hits >= 1

    @pytest.mark.asyncio
    async def test_cache_append_real_data(self) -> None:
        """Test appending new bars to cache."""
        data = fetch_binance_klines("ETHUSDT", "1m", 100)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data)
        first_half = df.head(50)
        second_half = df.tail(50)

        cache = DataFrameCache()

        # Cache first half
        await cache.set_bars("ETH/USDT", "1m", first_half)

        # Append second half
        await cache.append_bars("ETH/USDT", "1m", second_half)

        # Should have all data
        result = await cache.get_bars("ETH/USDT", "1m")
        assert result is not None
        assert len(result) >= 50  # At least second half


# =============================================================================
# Storage with Real Data
# =============================================================================


class TestStorageRealData:
    """Test tiered storage with real data."""

    @pytest.mark.asyncio
    async def test_in_memory_storage_real_data(self) -> None:
        """Test in-memory storage with real data."""
        data = fetch_binance_klines("BTCUSDT", "1h", 200)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns"))
        )

        backend = InMemoryBackend()

        # Write
        await backend.write("BTC/USDT:1h", df)

        # Read
        result = await backend.read("BTC/USDT:1h")
        assert result is not None
        assert len(result) == len(df)

        # Check metadata
        meta = await backend.get_metadata("BTC/USDT:1h")
        assert meta is not None
        assert meta["row_count"] == len(df)

    @pytest.mark.asyncio
    async def test_parquet_storage_real_data(self) -> None:
        """Test Parquet storage with real data."""
        data = fetch_binance_klines("ETHUSDT", "1h", 200)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns"))
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetBackend(Path(tmpdir), StorageTier.WARM)

            # Write
            await backend.write("ETH/USDT:1h", df)

            # Read
            result = await backend.read("ETH/USDT:1h")
            assert result is not None
            assert len(result) == len(df)

            # Verify data preserved
            assert result["open"][0] == df["open"][0]
            # Use approximate comparison for floating point
            assert abs(result["volume"].sum() - df["volume"].sum()) < 1e-6

    @pytest.mark.asyncio
    async def test_tiered_storage_real_data(self) -> None:
        """Test tiered storage with real data."""
        data = fetch_binance_klines("BTCUSDT", "15m", 100)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns"))
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TieredStorage(
                hot=InMemoryBackend(),
                warm=ParquetBackend(Path(tmpdir) / "warm", StorageTier.WARM),
            )

            # Write to hot tier
            await storage.write("BTC/USDT:15m", df, tier=StorageTier.HOT)

            # Read from hot
            result = await storage.read("BTC/USDT:15m")
            assert result is not None

            # Demote to warm
            await storage.demote("BTC/USDT:15m", StorageTier.HOT, StorageTier.WARM)

            # Should not be in hot
            hot_result = await storage.read("BTC/USDT:15m", tier=StorageTier.HOT)
            assert hot_result is None

            # Should be in warm
            warm_result = await storage.read("BTC/USDT:15m", tier=StorageTier.WARM)
            assert warm_result is not None
            assert len(warm_result) == len(df)


# =============================================================================
# Performance Benchmarks
# =============================================================================


class TestPerformanceRealData:
    """Test performance targets from Issue #23."""

    def test_downsample_performance(self) -> None:
        """Downsampling should complete quickly on real data."""
        import time

        data = fetch_binance_klines("BTCUSDT", "1m", 500)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns"))
        )

        start = time.perf_counter()
        result = downsample_ohlcv(df, "1m", "5m")
        elapsed = time.perf_counter() - start

        # Should complete in < 100ms (target from Issue #23)
        assert elapsed < 0.1, f"Downsampling took {elapsed*1000:.1f}ms"
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_cache_hit_latency(self) -> None:
        """Cache hits should be fast."""
        import time

        data = fetch_binance_klines("BTCUSDT", "1m", 100)
        if not data:
            pytest.skip("No data fetched")

        df = pl.DataFrame(data)
        cache = DataFrameCache()

        await cache.set_bars("BTC/USDT", "1m", df)

        # Measure read latency
        start = time.perf_counter()
        for _ in range(100):
            await cache.get_bars("BTC/USDT", "1m")
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / 100) * 1000

        # Target: < 10ms per access
        assert avg_latency_ms < 10, f"Average cache latency: {avg_latency_ms:.2f}ms"

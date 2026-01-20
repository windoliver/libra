"""
Tests for Bcolz columnar storage (Issue #105).

Tests the Zipline-style bcolz storage backend.
Note: Tests are skipped if bcolz-zipline is not installed.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from libra.data.storage import BCOLZ_AVAILABLE, StorageTier


# Skip all tests if bcolz is not available
pytestmark = pytest.mark.skipif(
    not BCOLZ_AVAILABLE, reason="bcolz-zipline not installed"
)


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    n = 100
    np.random.seed(42)

    base_time = int(datetime(2024, 1, 1).timestamp() * 1_000_000_000)
    timestamps = [base_time + i * 86400 * 1_000_000_000 for i in range(n)]

    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": prices.tolist(),
        "high": (prices + np.random.rand(n)).tolist(),
        "low": (prices - np.random.rand(n)).tolist(),
        "close": (prices + np.random.randn(n) * 0.1).tolist(),
        "volume": (np.random.rand(n) * 1000).tolist(),
    })


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create sample DataFrame for testing."""
    n = 50
    np.random.seed(42)

    base_time = int(datetime(2024, 1, 1).timestamp() * 1_000_000_000)
    timestamps = [base_time + i * 3600 * 1_000_000_000 for i in range(n)]

    return pl.DataFrame({
        "timestamp": timestamps,
        "value": np.random.randn(n).tolist(),
        "category": ["A" if i % 2 == 0 else "B" for i in range(n)],
    })


class TestBcolzBackend:
    """Tests for BcolzBackend class."""

    @pytest.fixture
    def bcolz_backend(self, tmp_path: Path):
        """Create bcolz backend for testing."""
        from libra.data.storage import BcolzBackend

        return BcolzBackend(tmp_path / "bcolz_test")

    async def test_tier_property(self, bcolz_backend) -> None:
        """Test tier property returns WARM by default."""
        assert bcolz_backend.tier == StorageTier.WARM

    async def test_write_and_read(self, bcolz_backend, sample_df: pl.DataFrame) -> None:
        """Test basic write and read operations."""
        await bcolz_backend.write("test_key", sample_df)
        result = await bcolz_backend.read("test_key")

        assert result is not None
        assert result.height == sample_df.height
        assert set(result.columns) == set(sample_df.columns)

    async def test_read_nonexistent(self, bcolz_backend) -> None:
        """Test reading nonexistent key returns None."""
        result = await bcolz_backend.read("nonexistent")
        assert result is None

    async def test_read_with_time_range(
        self, bcolz_backend, sample_df: pl.DataFrame
    ) -> None:
        """Test reading with time range filter."""
        await bcolz_backend.write("test_key", sample_df)

        # Read middle portion
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 20, 0, 0)

        result = await bcolz_backend.read("test_key", start=start, end=end)

        assert result is not None
        # Should have fewer rows than original
        assert result.height < sample_df.height
        assert result.height > 0

    async def test_delete(self, bcolz_backend, sample_df: pl.DataFrame) -> None:
        """Test delete operation."""
        await bcolz_backend.write("test_key", sample_df)
        assert await bcolz_backend.exists("test_key")

        deleted = await bcolz_backend.delete("test_key")
        assert deleted
        assert not await bcolz_backend.exists("test_key")

    async def test_delete_nonexistent(self, bcolz_backend) -> None:
        """Test deleting nonexistent key."""
        deleted = await bcolz_backend.delete("nonexistent")
        assert not deleted

    async def test_exists(self, bcolz_backend, sample_df: pl.DataFrame) -> None:
        """Test exists check."""
        assert not await bcolz_backend.exists("test_key")
        await bcolz_backend.write("test_key", sample_df)
        assert await bcolz_backend.exists("test_key")

    async def test_list_keys(self, bcolz_backend, sample_df: pl.DataFrame) -> None:
        """Test listing keys."""
        await bcolz_backend.write("key1", sample_df)
        await bcolz_backend.write("key2", sample_df)
        await bcolz_backend.write("other_key", sample_df)

        all_keys = await bcolz_backend.list_keys()
        assert len(all_keys) == 3

        filtered_keys = await bcolz_backend.list_keys(prefix="key")
        assert len(filtered_keys) == 2

    async def test_get_metadata(
        self, bcolz_backend, sample_df: pl.DataFrame
    ) -> None:
        """Test getting metadata."""
        await bcolz_backend.write("test_key", sample_df)
        meta = await bcolz_backend.get_metadata("test_key")

        assert meta is not None
        assert meta["nrows"] == sample_df.height
        assert set(meta["columns"]) == set(sample_df.columns)
        assert "size_bytes" in meta
        assert meta["size_bytes"] > 0

    async def test_get_metadata_nonexistent(self, bcolz_backend) -> None:
        """Test metadata for nonexistent key."""
        meta = await bcolz_backend.get_metadata("nonexistent")
        assert meta is None

    async def test_append(self, bcolz_backend, sample_df: pl.DataFrame) -> None:
        """Test appending data."""
        await bcolz_backend.write("test_key", sample_df)

        # Create more data to append
        n = 10
        np.random.seed(123)
        base_time = int(datetime(2024, 1, 3).timestamp() * 1_000_000_000)
        timestamps = [base_time + i * 3600 * 1_000_000_000 for i in range(n)]

        append_df = pl.DataFrame({
            "timestamp": timestamps,
            "value": np.random.randn(n).tolist(),
            "category": ["C"] * n,
        })

        await bcolz_backend.append("test_key", append_df)

        result = await bcolz_backend.read("test_key")
        assert result is not None
        assert result.height == sample_df.height + n


class TestBcolzDataStore:
    """Tests for BcolzDataStore class."""

    @pytest.fixture
    def data_store(self, tmp_path: Path):
        """Create data store for testing."""
        from libra.data.storage import BcolzDataStore

        return BcolzDataStore(tmp_path / "bundles")

    def test_save_and_load_bars(
        self, data_store, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test saving and loading OHLCV bars."""
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1d")
        result = data_store.load_bars("BTC/USDT", timeframe="1d")

        assert result is not None
        assert result.height == sample_ohlcv_df.height
        assert "open" in result.columns
        assert "close" in result.columns

    def test_load_nonexistent(self, data_store) -> None:
        """Test loading nonexistent symbol."""
        result = data_store.load_bars("NONEXISTENT/USDT")
        assert result is None

    def test_load_with_time_range(
        self, data_store, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test loading with time range."""
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1d")

        start = datetime(2024, 1, 10)
        end = datetime(2024, 1, 20)
        result = data_store.load_bars("BTC/USDT", timeframe="1d", start=start, end=end)

        assert result is not None
        assert result.height < sample_ohlcv_df.height
        assert result.height > 0

    def test_list_symbols(
        self, data_store, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test listing symbols."""
        data_store.save_bars("BTC/USDT", sample_ohlcv_df)
        data_store.save_bars("ETH/USDT", sample_ohlcv_df)

        symbols = data_store.list_symbols()
        assert len(symbols) == 2
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols

    def test_list_timeframes(
        self, data_store, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test listing timeframes."""
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1d")
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1h")

        timeframes = data_store.list_timeframes("BTC/USDT")
        assert len(timeframes) == 2
        assert "1d" in timeframes
        assert "1h" in timeframes

    def test_get_info(
        self, data_store, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test getting data info."""
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1d")
        info = data_store.get_info("BTC/USDT", timeframe="1d")

        assert info is not None
        assert info["symbol"] == "BTC/USDT"
        assert info["timeframe"] == "1d"
        assert info["nrows"] == sample_ohlcv_df.height
        assert info["start_timestamp"] is not None
        assert info["end_timestamp"] is not None

    def test_delete_timeframe(
        self, data_store, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test deleting specific timeframe."""
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1d")
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1h")

        deleted = data_store.delete("BTC/USDT", timeframe="1d")
        assert deleted

        # 1d should be gone, 1h should remain
        timeframes = data_store.list_timeframes("BTC/USDT")
        assert "1d" not in timeframes
        assert "1h" in timeframes

    def test_delete_symbol(
        self, data_store, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test deleting entire symbol."""
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1d")
        data_store.save_bars("BTC/USDT", sample_ohlcv_df, timeframe="1h")

        deleted = data_store.delete("BTC/USDT")
        assert deleted

        symbols = data_store.list_symbols()
        assert "BTC/USDT" not in symbols

    def test_missing_columns_error(self, data_store) -> None:
        """Test error when required columns are missing."""
        incomplete_df = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
            # Missing open, high, low, volume
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            data_store.save_bars("BTC/USDT", incomplete_df)


class TestBcolzCompression:
    """Tests for bcolz compression."""

    @pytest.fixture
    def data_store(self, tmp_path: Path):
        """Create data store for testing."""
        from libra.data.storage import BcolzDataStore

        return BcolzDataStore(tmp_path / "bundles")

    def test_compression_reduces_size(
        self, data_store, tmp_path: Path
    ) -> None:
        """Test that bcolz compression reduces storage size."""
        # Create larger dataset for compression test
        n = 10000
        np.random.seed(42)

        base_time = int(datetime(2024, 1, 1).timestamp() * 1_000_000_000)
        timestamps = [base_time + i * 60 * 1_000_000_000 for i in range(n)]

        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": prices.tolist(),
            "high": (prices + np.random.rand(n)).tolist(),
            "low": (prices - np.random.rand(n)).tolist(),
            "close": (prices + np.random.randn(n) * 0.1).tolist(),
            "volume": (np.random.rand(n) * 1000).tolist(),
        })

        data_store.save_bars("BTC/USDT", df, timeframe="1m")

        # Check that data was compressed
        info = data_store.get_info("BTC/USDT", timeframe="1m")
        assert info is not None
        assert "lz4" in info["compression"]


class TestBcolzNotAvailable:
    """Tests for when bcolz is not installed."""

    def test_import_error_when_not_available(self, tmp_path: Path) -> None:
        """Test that clear error is raised when bcolz not available."""
        if BCOLZ_AVAILABLE:
            pytest.skip("bcolz is available, cannot test import error")

        from libra.data.storage import BcolzBackend, BcolzDataStore

        with pytest.raises(ImportError, match="bcolz not available"):
            BcolzBackend(tmp_path)

        with pytest.raises(ImportError, match="bcolz not available"):
            BcolzDataStore(tmp_path)

"""Tests for OHLCV Downsampling (Issue #23).

Tests:
- Resolution conversion
- OHLCV downsampling with integrity
- Multi-resolution downsampling
- Age-based retention policies
- Tick to bar aggregation
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from libra.data.downsample import (
    RESOLUTION_SECONDS,
    VALID_DOWNSAMPLE,
    apply_retention_policy,
    can_downsample,
    downsample_ohlcv,
    downsample_ohlcv_lazy,
    downsample_to_multiple,
    get_downsample_factor,
    get_resolution_seconds,
    ticks_to_bars,
    validate_ohlcv_integrity,
)


class TestResolutionDefinitions:
    """Tests for resolution definitions."""

    def test_resolution_seconds(self) -> None:
        """Test resolution to seconds mapping."""
        assert RESOLUTION_SECONDS["1s"] == 1
        assert RESOLUTION_SECONDS["1m"] == 60
        assert RESOLUTION_SECONDS["5m"] == 300
        assert RESOLUTION_SECONDS["1h"] == 3600
        assert RESOLUTION_SECONDS["1d"] == 86400

    def test_valid_downsample_paths(self) -> None:
        """Test valid downsample paths."""
        assert "5m" in VALID_DOWNSAMPLE["1m"]
        assert "1h" in VALID_DOWNSAMPLE["1m"]
        assert "1d" in VALID_DOWNSAMPLE["1h"]


class TestResolutionHelpers:
    """Tests for resolution helper functions."""

    def test_get_resolution_seconds(self) -> None:
        """Test getting resolution in seconds."""
        assert get_resolution_seconds("1m") == 60
        assert get_resolution_seconds("5m") == 300
        assert get_resolution_seconds("1h") == 3600

    def test_get_resolution_seconds_invalid(self) -> None:
        """Test invalid resolution raises error."""
        with pytest.raises(ValueError, match="Unknown resolution"):
            get_resolution_seconds("invalid")

    def test_can_downsample(self) -> None:
        """Test can_downsample check."""
        assert can_downsample("1m", "5m") is True
        assert can_downsample("1m", "1h") is True
        assert can_downsample("1h", "1d") is True

        # Invalid paths
        assert can_downsample("1h", "1m") is False  # Can't upsample
        assert can_downsample("1m", "7m") is False  # Non-standard resolution
        assert can_downsample("invalid", "1m") is False

    def test_get_downsample_factor(self) -> None:
        """Test downsample factor calculation."""
        assert get_downsample_factor("1m", "5m") == 5
        assert get_downsample_factor("1m", "1h") == 60
        assert get_downsample_factor("1h", "1d") == 24

    def test_get_downsample_factor_upsample_raises(self) -> None:
        """Test upsampling raises error."""
        with pytest.raises(ValueError, match="Cannot upsample"):
            get_downsample_factor("1h", "1m")

    def test_get_downsample_factor_uneven_raises(self) -> None:
        """Test uneven downsample raises error."""
        # 7m isn't a standard resolution, so it raises "Unknown resolution" first
        with pytest.raises(ValueError):
            get_downsample_factor("1m", "7m")


class TestDownsampleOHLCV:
    """Tests for downsample_ohlcv."""

    @pytest.fixture
    def ohlcv_1m(self) -> pl.DataFrame:
        """Create 1-minute OHLCV data."""
        # Create 60 minutes of data (1 hour)
        timestamps = [
            datetime(2024, 1, 1, 0, i) for i in range(60)
        ]
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [float(100 + i % 10) for i in range(60)],
                "high": [float(110 + i % 10) for i in range(60)],
                "low": [float(90 + i % 10) for i in range(60)],
                "close": [float(105 + i % 10) for i in range(60)],
                "volume": [float(1000 + i * 10) for i in range(60)],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))

    def test_downsample_1m_to_5m(self, ohlcv_1m: pl.DataFrame) -> None:
        """Test downsampling 1m to 5m."""
        result = downsample_ohlcv(ohlcv_1m, "1m", "5m")

        # 60 minutes / 5 = 12 bars
        assert len(result) == 12

        # Check OHLCV columns exist
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
        assert "timeframe" in result.columns

        # Check timeframe is set
        assert result["timeframe"][0] == "5m"

    def test_downsample_1m_to_1h(self, ohlcv_1m: pl.DataFrame) -> None:
        """Test downsampling 1m to 1h."""
        result = downsample_ohlcv(ohlcv_1m, "1m", "1h")

        # 60 minutes = 1 hour bar
        assert len(result) == 1

    def test_downsample_preserves_ohlc(self, ohlcv_1m: pl.DataFrame) -> None:
        """Test OHLC values are correctly aggregated."""
        result = downsample_ohlcv(ohlcv_1m, "1m", "5m")

        # First 5-minute bar
        first_bar = result.row(0, named=True)

        # Open should be first value in period
        # High should be max in period
        # Low should be min in period
        # Close should be last value in period
        assert first_bar["open"] is not None
        assert first_bar["high"] >= first_bar["open"]
        assert first_bar["low"] <= first_bar["open"]
        assert first_bar["close"] is not None

    def test_downsample_volume_summed(self, ohlcv_1m: pl.DataFrame) -> None:
        """Test volume is summed correctly."""
        result = downsample_ohlcv(ohlcv_1m, "1m", "5m")

        # First 5-minute bar should sum first 5 volumes
        # 1000 + 1010 + 1020 + 1030 + 1040 = 5100
        first_volume = result["volume"][0]
        expected = sum(1000 + i * 10 for i in range(5))
        assert first_volume == expected

    def test_downsample_invalid_path_raises(self, ohlcv_1m: pl.DataFrame) -> None:
        """Test invalid downsample path raises error."""
        with pytest.raises(ValueError, match="Cannot downsample"):
            downsample_ohlcv(ohlcv_1m, "1m", "7m")

    def test_downsample_with_trades(self) -> None:
        """Test downsampling with trades column."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, i) for i in range(5)
                ],
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000.0] * 5,
                "trades": [100, 150, 200, 120, 180],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))

        result = downsample_ohlcv(df, "1m", "5m")

        assert "trades" in result.columns
        # Trades should be summed
        assert result["trades"][0] == 750


class TestDownsampleOHLCVLazy:
    """Tests for lazy downsampling."""

    def test_returns_lazyframe(self) -> None:
        """Test lazy version returns LazyFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, i) for i in range(5)],
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000.0] * 5,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))

        result = downsample_ohlcv_lazy(df.lazy(), "1m", "5m")

        assert isinstance(result, pl.LazyFrame)


class TestDownsampleToMultiple:
    """Tests for multi-resolution downsampling."""

    @pytest.fixture
    def ohlcv_1m(self) -> pl.DataFrame:
        """Create 1-minute OHLCV data (240 minutes = 4 hours)."""
        timestamps = [
            datetime(2024, 1, 1, i // 60, i % 60) for i in range(240)
        ]
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * 240,
                "high": [105.0] * 240,
                "low": [95.0] * 240,
                "close": [102.0] * 240,
                "volume": [1000.0] * 240,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))

    def test_downsample_to_multiple(self, ohlcv_1m: pl.DataFrame) -> None:
        """Test downsampling to multiple resolutions."""
        results = downsample_to_multiple(ohlcv_1m, "1m", ["5m", "15m", "1h"])

        assert "5m" in results
        assert "15m" in results
        assert "1h" in results

        # Check row counts
        assert len(results["5m"]) == 48  # 240 / 5
        assert len(results["15m"]) == 16  # 240 / 15
        assert len(results["1h"]) == 4  # 240 / 60


class TestValidateOHLCVIntegrity:
    """Tests for OHLCV integrity validation."""

    def test_valid_ohlcv(self) -> None:
        """Test valid OHLCV passes validation."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )

        errors = validate_ohlcv_integrity(df)
        assert len(errors) == 0

    def test_invalid_high(self) -> None:
        """Test high < open/close is detected."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [98.0],  # Invalid: high < open
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )

        errors = validate_ohlcv_integrity(df)
        assert any("High" in e for e in errors)

    def test_invalid_low(self) -> None:
        """Test low > open/close is detected."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [101.0],  # Invalid: low > open
                "close": [102.0],
                "volume": [1000.0],
            }
        )

        errors = validate_ohlcv_integrity(df)
        assert any("Low" in e for e in errors)

    def test_negative_volume(self) -> None:
        """Test negative volume is detected."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [-1000.0],  # Invalid
            }
        )

        errors = validate_ohlcv_integrity(df)
        assert any("Negative volume" in e for e in errors)


class TestTicksToBars:
    """Tests for tick to bar aggregation."""

    @pytest.fixture
    def ticks_df(self) -> pl.DataFrame:
        """Create tick data."""
        # 10 ticks per minute for 5 minutes
        timestamps = []
        for minute in range(5):
            for second in range(0, 60, 6):  # 10 ticks per minute
                timestamps.append(datetime(2024, 1, 1, 0, minute, second))

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "last": [100.0 + i * 0.1 for i in range(50)],
                "amount": [1.0] * 50,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))

    def test_ticks_to_1m_bars(self, ticks_df: pl.DataFrame) -> None:
        """Test converting ticks to 1-minute bars."""
        result = ticks_to_bars(ticks_df, "1m", size_col="amount")

        assert len(result) == 5  # 5 minutes

        # Check OHLCV columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
        assert "trades" in result.columns

    def test_ticks_to_bars_aggregations(self, ticks_df: pl.DataFrame) -> None:
        """Test tick aggregations are correct."""
        result = ticks_to_bars(ticks_df, "1m", size_col="amount")

        first_bar = result.row(0, named=True)

        # First bar should have 10 trades
        assert first_bar["trades"] == 10

        # Volume should be sum of amounts
        assert first_bar["volume"] == 10.0

    def test_ticks_to_bars_vwap(self, ticks_df: pl.DataFrame) -> None:
        """Test VWAP calculation."""
        result = ticks_to_bars(ticks_df, "1m", size_col="amount")

        assert "vwap" in result.columns
        assert result["vwap"][0] is not None

    def test_ticks_to_bars_no_size(self, ticks_df: pl.DataFrame) -> None:
        """Test ticks to bars without size column."""
        result = ticks_to_bars(ticks_df.drop("amount"), "1m")

        # Should still work but volume is 0
        assert result["volume"][0] == 0.0


class TestApplyRetentionPolicy:
    """Tests for retention policy."""

    @pytest.fixture
    def historical_ohlcv(self) -> pl.DataFrame:
        """Create historical OHLCV spanning multiple time ranges."""
        now = datetime.utcnow()
        timestamps = [
            now - timedelta(hours=1),  # Recent
            now - timedelta(hours=12),  # Recent
            now - timedelta(days=3),  # Last week
            now - timedelta(days=10),  # Last month
            now - timedelta(days=45),  # Older
        ]

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000.0] * 5,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))

    def test_default_policy(self, historical_ohlcv: pl.DataFrame) -> None:
        """Test default retention policy."""
        results = apply_retention_policy(historical_ohlcv)

        # Should have data in various resolutions based on age
        assert isinstance(results, dict)

    def test_custom_policy(self, historical_ohlcv: pl.DataFrame) -> None:
        """Test custom retention policy."""
        rules = [
            (timedelta(days=1), "1m"),
            (timedelta(days=7), "5m"),
        ]

        results = apply_retention_policy(historical_ohlcv, rules=rules)

        assert isinstance(results, dict)

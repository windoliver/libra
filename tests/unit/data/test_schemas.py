"""Tests for Polars Schemas (Issue #23).

Tests:
- OHLCV schema and validation
- Tick schema and validation
- Trade and OrderBook schemas
- Technical indicator helpers
- Conversion utilities
"""

from datetime import datetime

import polars as pl
import pytest

from libra.data.schemas import (
    OHLCV_REQUIRED_COLUMNS,
    OHLCV_SCHEMA,
    TICK_REQUIRED_COLUMNS,
    TICK_SCHEMA,
    TRADE_SCHEMA,
    add_bollinger_bands,
    add_ema,
    add_rsi,
    add_sma,
    create_ohlcv_df,
    create_orderbook_df,
    create_tick_df,
    validate_ohlcv,
    validate_tick,
)


class TestOHLCVSchema:
    """Tests for OHLCV schema."""

    def test_schema_has_required_columns(self) -> None:
        """Test schema has all required columns."""
        for col in OHLCV_REQUIRED_COLUMNS:
            assert col in OHLCV_SCHEMA

    def test_schema_types(self) -> None:
        """Test schema column types."""
        assert OHLCV_SCHEMA["timestamp"] == pl.Datetime("ns")
        assert OHLCV_SCHEMA["symbol"] == pl.Utf8
        assert OHLCV_SCHEMA["open"] == pl.Float64
        assert OHLCV_SCHEMA["high"] == pl.Float64
        assert OHLCV_SCHEMA["low"] == pl.Float64
        assert OHLCV_SCHEMA["close"] == pl.Float64
        assert OHLCV_SCHEMA["volume"] == pl.Float64


class TestCreateOHLCVDF:
    """Tests for create_ohlcv_df."""

    def test_create_from_data(self) -> None:
        """Test creating DataFrame from dict list."""
        data = [
            {
                "timestamp": datetime(2024, 1, 1, 0, 0),
                "symbol": "BTC/USDT",
                "open": 50000.0,
                "high": 50500.0,
                "low": 49800.0,
                "close": 50200.0,
                "volume": 100.0,
                "timeframe": "1h",
            }
        ]

        df = create_ohlcv_df(data)

        assert len(df) == 1
        assert df["open"][0] == 50000.0
        assert df["close"][0] == 50200.0

    def test_create_from_arrays(self) -> None:
        """Test creating DataFrame from arrays."""
        timestamps = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        opens = [100.0, 102.0]
        highs = [105.0, 107.0]
        lows = [99.0, 101.0]
        closes = [104.0, 106.0]
        volumes = [1000.0, 1200.0]

        df = create_ohlcv_df(
            timestamps=timestamps,
            symbol="TEST",
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            timeframe="1d",
        )

        assert len(df) == 2
        assert df["open"][0] == 100.0
        assert df["close"][1] == 106.0
        assert df["symbol"][0] == "TEST"

    def test_create_requires_data(self) -> None:
        """Test that create requires either data or arrays."""
        with pytest.raises(ValueError, match="Must provide"):
            create_ohlcv_df()


class TestValidateOHLCV:
    """Tests for validate_ohlcv."""

    def test_valid_ohlcv(self) -> None:
        """Test valid OHLCV data passes."""
        df = create_ohlcv_df(
            timestamps=[datetime(2024, 1, 1)],
            symbol="TEST",
            opens=[100.0],
            highs=[105.0],
            lows=[99.0],
            closes=[102.0],
            volumes=[1000.0],
        )

        errors = validate_ohlcv(df)
        assert len(errors) == 0

    def test_missing_column(self) -> None:
        """Test missing required column."""
        df = pl.DataFrame({"timestamp": [], "symbol": []})

        errors = validate_ohlcv(df)
        assert any("Missing required column" in e for e in errors)

    def test_invalid_high_low(self) -> None:
        """Test invalid high < low."""
        df = create_ohlcv_df(
            timestamps=[datetime(2024, 1, 1)],
            symbol="TEST",
            opens=[100.0],
            highs=[95.0],  # Invalid: high < low
            lows=[99.0],
            closes=[97.0],
            volumes=[1000.0],
        )

        errors = validate_ohlcv(df)
        assert any("Invalid OHLC" in e for e in errors)

    def test_negative_values(self) -> None:
        """Test negative values detected."""
        df = create_ohlcv_df(
            timestamps=[datetime(2024, 1, 1)],
            symbol="TEST",
            opens=[-100.0],  # Negative
            highs=[105.0],
            lows=[99.0],
            closes=[102.0],
            volumes=[1000.0],
        )

        errors = validate_ohlcv(df)
        assert any("Negative" in e for e in errors)


class TestTickSchema:
    """Tests for Tick schema."""

    def test_schema_has_required_columns(self) -> None:
        """Test tick schema has required columns."""
        for col in TICK_REQUIRED_COLUMNS:
            assert col in TICK_SCHEMA

    def test_create_tick_df(self) -> None:
        """Test creating tick DataFrame."""
        df = create_tick_df(
            timestamps=[datetime(2024, 1, 1)],
            symbol="BTC/USDT",
            bids=[50000.0],
            asks=[50001.0],
            lasts=[50000.5],
        )

        assert len(df) == 1
        assert df["bid"][0] == 50000.0
        assert df["ask"][0] == 50001.0

    def test_validate_tick_valid(self) -> None:
        """Test valid tick data passes."""
        df = create_tick_df(
            timestamps=[datetime(2024, 1, 1)],
            symbol="BTC/USDT",
            bids=[50000.0],
            asks=[50001.0],
        )

        errors = validate_tick(df)
        assert len(errors) == 0

    def test_validate_tick_bid_gt_ask(self) -> None:
        """Test bid > ask is invalid."""
        df = create_tick_df(
            timestamps=[datetime(2024, 1, 1)],
            symbol="BTC/USDT",
            bids=[50002.0],  # Bid > Ask
            asks=[50001.0],
        )

        errors = validate_tick(df)
        assert any("Bid > Ask" in e for e in errors)


class TestOrderBookSchema:
    """Tests for OrderBook schema."""

    def test_create_orderbook_df(self) -> None:
        """Test creating order book DataFrame."""
        bids = [(50000.0, 1.5), (49999.0, 2.0)]
        asks = [(50001.0, 1.0), (50002.0, 3.0)]

        df = create_orderbook_df(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1),
            bids=bids,
            asks=asks,
        )

        assert len(df) == 4
        bid_rows = df.filter(pl.col("side") == "bid")
        ask_rows = df.filter(pl.col("side") == "ask")

        assert len(bid_rows) == 2
        assert len(ask_rows) == 2
        assert bid_rows["level"][0] == 0  # Best bid
        assert ask_rows["level"][0] == 0  # Best ask


class TestTechnicalIndicators:
    """Tests for technical indicator helpers."""

    @pytest.fixture
    def price_df(self) -> pl.LazyFrame:
        """Create sample price data."""
        data = {
            "timestamp": [datetime(2024, 1, i) for i in range(1, 31)],
            "close": [float(100 + i) for i in range(30)],
        }
        return pl.DataFrame(data).lazy()

    def test_add_sma(self, price_df: pl.LazyFrame) -> None:
        """Test adding SMA."""
        result = add_sma(price_df, "close", 5).collect()

        assert "sma_5" in result.columns
        # First 4 values should be null (not enough data)
        assert result["sma_5"][0] is None
        assert result["sma_5"][4] is not None

    def test_add_sma_custom_alias(self, price_df: pl.LazyFrame) -> None:
        """Test SMA with custom alias."""
        result = add_sma(price_df, "close", 10, alias="ma_10").collect()
        assert "ma_10" in result.columns

    def test_add_ema(self, price_df: pl.LazyFrame) -> None:
        """Test adding EMA."""
        result = add_ema(price_df, "close", 12).collect()

        assert "ema_12" in result.columns
        # EMA should have values
        assert result["ema_12"][-1] is not None

    def test_add_rsi(self, price_df: pl.LazyFrame) -> None:
        """Test adding RSI."""
        result = add_rsi(price_df, "close", 14).collect()

        assert "rsi" in result.columns
        # RSI should be between 0 and 100
        rsi_values = result["rsi"].drop_nulls()
        assert all(0 <= v <= 100 for v in rsi_values)

    def test_add_bollinger_bands(self, price_df: pl.LazyFrame) -> None:
        """Test adding Bollinger Bands."""
        result = add_bollinger_bands(price_df, "close", 20, 2.0).collect()

        assert "bb_middle" in result.columns
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns

        # Upper > Middle > Lower
        valid_rows = result.filter(pl.col("bb_middle").is_not_null())
        if len(valid_rows) > 0:
            assert all(
                valid_rows["bb_upper"][i] >= valid_rows["bb_middle"][i]
                for i in range(len(valid_rows))
            )
            assert all(
                valid_rows["bb_middle"][i] >= valid_rows["bb_lower"][i]
                for i in range(len(valid_rows))
            )

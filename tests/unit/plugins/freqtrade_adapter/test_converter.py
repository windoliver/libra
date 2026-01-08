"""Unit tests for FreqtradeSignalConverter."""

from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest

from libra.plugins.freqtrade_adapter.converter import FreqtradeSignalConverter
from libra.strategies.protocol import SignalType


class TestFreqtradeSignalConverter:
    """Tests for FreqtradeSignalConverter."""

    @pytest.fixture
    def converter(self) -> FreqtradeSignalConverter:
        """Create a signal converter instance."""
        return FreqtradeSignalConverter()

    @pytest.fixture
    def sample_ohlcv_df(self) -> pl.DataFrame:
        """Create a sample OHLCV Polars DataFrame."""
        return pl.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0],
            "enter_long": [0, 0, 0],
            "enter_short": [0, 0, 0],
            "exit_long": [0, 0, 0],
            "exit_short": [0, 0, 0],
        })

    def test_convert_empty_dataframe(self, converter: FreqtradeSignalConverter) -> None:
        """Test conversion with empty DataFrame returns None."""
        df = pl.DataFrame()
        result = converter.convert_dataframe(df, "BTC/USDT")
        assert result is None

    def test_convert_no_signal(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test conversion when no signals are present."""
        result = converter.convert_dataframe(sample_ohlcv_df, "BTC/USDT")
        assert result is None

    def test_convert_enter_long_signal(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test conversion of enter_long signal."""
        # Set enter_long=1 on last row
        df = sample_ohlcv_df.with_columns(
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("enter_long"))
            .alias("enter_long")
        )

        result = converter.convert_dataframe(df, "BTC/USDT")

        assert result is not None
        assert result.signal_type == SignalType.LONG
        assert result.symbol == "BTC/USDT"
        assert result.price == Decimal("104.0")
        assert result.metadata["source"] == "freqtrade"
        assert result.metadata["direction"] == "long"

    def test_convert_enter_short_signal(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test conversion of enter_short signal."""
        df = sample_ohlcv_df.with_columns(
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("enter_short"))
            .alias("enter_short")
        )

        result = converter.convert_dataframe(df, "ETH/USDT")

        assert result is not None
        assert result.signal_type == SignalType.SHORT
        assert result.symbol == "ETH/USDT"
        assert result.metadata["direction"] == "short"

    def test_convert_exit_long_signal(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test conversion of exit_long signal."""
        df = sample_ohlcv_df.with_columns(
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("exit_long"))
            .alias("exit_long")
        )

        result = converter.convert_dataframe(df, "BTC/USDT")

        assert result is not None
        assert result.signal_type == SignalType.CLOSE_LONG
        assert result.metadata["direction"] == "long"

    def test_convert_exit_short_signal(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test conversion of exit_short signal."""
        df = sample_ohlcv_df.with_columns(
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("exit_short"))
            .alias("exit_short")
        )

        result = converter.convert_dataframe(df, "BTC/USDT")

        assert result is not None
        assert result.signal_type == SignalType.CLOSE_SHORT
        assert result.metadata["direction"] == "short"

    def test_exit_priority_over_entry(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test that exit signals have priority over entry signals."""
        # Set both enter_long and exit_long on last row
        df = sample_ohlcv_df.with_columns([
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("enter_long"))
            .alias("enter_long"),
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("exit_long"))
            .alias("exit_long"),
        ])

        result = converter.convert_dataframe(df, "BTC/USDT")

        # Exit should take priority (risk management)
        assert result is not None
        assert result.signal_type == SignalType.CLOSE_LONG

    def test_enter_tag_extraction(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test extraction of enter_tag from DataFrame."""
        df = sample_ohlcv_df.with_columns([
            pl.Series("enter_tag", ["", "", "rsi_oversold"]),
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("enter_long"))
            .alias("enter_long"),
        ])

        result = converter.convert_dataframe(df, "BTC/USDT")

        assert result is not None
        assert result.metadata.get("enter_tag") == "rsi_oversold"

    def test_exit_tag_extraction(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test extraction of exit_tag from DataFrame."""
        df = sample_ohlcv_df.with_columns([
            pl.Series("exit_tag", ["", "", "take_profit"]),
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("exit_long"))
            .alias("exit_long"),
        ])

        result = converter.convert_dataframe(df, "BTC/USDT")

        assert result is not None
        assert result.metadata.get("exit_tag") == "take_profit"

    def test_custom_column_extraction(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test extraction of custom_ prefixed columns."""
        df = sample_ohlcv_df.with_columns([
            pl.Series("custom_rsi", [30, 35, 28]),
            pl.Series("custom_trend", ["up", "up", "down"]),
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("enter_long"))
            .alias("enter_long"),
        ])

        result = converter.convert_dataframe(df, "BTC/USDT")

        assert result is not None
        assert result.metadata.get("custom_rsi") == 28
        assert result.metadata.get("custom_trend") == "down"

    def test_convert_row_dict(self, converter: FreqtradeSignalConverter) -> None:
        """Test conversion of dictionary row."""
        row = {
            "enter_long": 1,
            "enter_short": 0,
            "exit_long": 0,
            "exit_short": 0,
            "enter_tag": "golden_cross",
        }

        result = converter.convert_row(row, "BTC/USDT", price=Decimal("50000"))

        assert result is not None
        assert result.signal_type == SignalType.LONG
        assert result.price == Decimal("50000")
        assert result.metadata.get("enter_tag") == "golden_cross"

    def test_convert_row_no_signal(self, converter: FreqtradeSignalConverter) -> None:
        """Test conversion of row with no signal."""
        row = {
            "enter_long": 0,
            "enter_short": 0,
            "exit_long": 0,
            "exit_short": 0,
        }

        result = converter.convert_row(row, "BTC/USDT")
        assert result is None

    def test_legacy_column_support(self) -> None:
        """Test support for legacy buy/sell columns."""
        converter = FreqtradeSignalConverter(use_legacy_columns=True)

        row = {
            "buy": 1,
            "sell": 0,
        }

        result = converter.convert_row(row, "BTC/USDT")

        assert result is not None
        assert result.signal_type == SignalType.LONG

    def test_signal_to_freqtrade_columns(
        self,
        converter: FreqtradeSignalConverter,
    ) -> None:
        """Test conversion from LIBRA Signal back to Freqtrade columns."""
        from libra.strategies.protocol import Signal

        signal = Signal.create(
            signal_type=SignalType.LONG,
            symbol="BTC/USDT",
            metadata={"enter_tag": "test_signal"},
        )

        columns = converter.signal_to_freqtrade_columns(signal)

        assert columns["enter_long"] == 1
        assert columns["enter_short"] == 0
        assert columns["exit_long"] == 0
        assert columns["exit_short"] == 0
        assert columns.get("enter_tag") == "test_signal"

    def test_signal_to_freqtrade_columns_short(
        self,
        converter: FreqtradeSignalConverter,
    ) -> None:
        """Test conversion of SHORT signal to Freqtrade columns."""
        from libra.strategies.protocol import Signal

        signal = Signal.create(signal_type=SignalType.SHORT, symbol="BTC/USDT")

        columns = converter.signal_to_freqtrade_columns(signal)

        assert columns["enter_long"] == 0
        assert columns["enter_short"] == 1
        assert columns["exit_long"] == 0
        assert columns["exit_short"] == 0

    def test_signal_to_freqtrade_columns_close_long(
        self,
        converter: FreqtradeSignalConverter,
    ) -> None:
        """Test conversion of CLOSE_LONG signal to Freqtrade columns."""
        from libra.strategies.protocol import Signal

        signal = Signal.create(signal_type=SignalType.CLOSE_LONG, symbol="BTC/USDT")

        columns = converter.signal_to_freqtrade_columns(signal)

        assert columns["enter_long"] == 0
        assert columns["enter_short"] == 0
        assert columns["exit_long"] == 1
        assert columns["exit_short"] == 0

    def test_signal_to_freqtrade_columns_hold(
        self,
        converter: FreqtradeSignalConverter,
    ) -> None:
        """Test conversion of HOLD signal to Freqtrade columns."""
        from libra.strategies.protocol import Signal

        signal = Signal.create(signal_type=SignalType.HOLD, symbol="BTC/USDT")

        columns = converter.signal_to_freqtrade_columns(signal)

        assert columns["enter_long"] == 0
        assert columns["enter_short"] == 0
        assert columns["exit_long"] == 0
        assert columns["exit_short"] == 0

    def test_timestamp_is_set(
        self,
        converter: FreqtradeSignalConverter,
        sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """Test that timestamp_ns is properly set in converted signal."""
        import time

        df = sample_ohlcv_df.with_columns(
            pl.when(pl.arange(0, sample_ohlcv_df.height) == 2)
            .then(1)
            .otherwise(pl.col("enter_long"))
            .alias("enter_long")
        )

        before = time.time_ns()
        result = converter.convert_dataframe(df, "BTC/USDT")
        after = time.time_ns()

        assert result is not None
        assert before <= result.timestamp_ns <= after

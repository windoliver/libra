"""
Signal converter for Freqtrade to LIBRA.

Converts Freqtrade DataFrame-based signals (enter_long, exit_long, etc.)
to LIBRA's Signal objects with proper type mapping.

Uses Polars as primary DataFrame type (10-3500x faster than pandas).
Converts to pandas only when interfacing with Freqtrade internals.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import polars as pl

from libra.strategies.protocol import Signal, SignalType


if TYPE_CHECKING:
    pass


class FreqtradeSignalConverter:
    """
    Converts Freqtrade signals to LIBRA Signal objects.

    Freqtrade uses DataFrame columns to indicate signals:
    - enter_long: 1 = buy signal for long position
    - enter_short: 1 = sell signal for short position
    - exit_long: 1 = close long position
    - exit_short: 1 = close short position
    - enter_tag: Optional string describing entry reason
    - exit_tag: Optional string describing exit reason

    LIBRA uses Signal objects with SignalType enum:
    - SignalType.LONG: Open long position
    - SignalType.SHORT: Open short position
    - SignalType.CLOSE_LONG: Close long position
    - SignalType.CLOSE_SHORT: Close short position
    - SignalType.HOLD: No action

    This converter uses Polars as the primary DataFrame type for performance.
    Pandas DataFrames are converted to Polars internally when needed.

    Examples:
        converter = FreqtradeSignalConverter()

        # Convert Polars DataFrame with signals
        signal = converter.convert_dataframe(df, symbol="BTC/USDT")
        if signal and signal.signal_type == SignalType.LONG:
            print(f"Long signal: {signal.metadata}")
    """

    # Mapping from Freqtrade columns to LIBRA SignalType
    # Priority order: exits first (safety), then entries
    SIGNAL_PRIORITY: list[tuple[str, int, SignalType]] = [
        # Exit signals have higher priority (risk management)
        ("exit_long", 1, SignalType.CLOSE_LONG),
        ("exit_short", 1, SignalType.CLOSE_SHORT),
        # Entry signals
        ("enter_long", 1, SignalType.LONG),
        ("enter_short", 1, SignalType.SHORT),
    ]

    # Legacy Freqtrade column names (pre-2021)
    LEGACY_COLUMN_MAP: dict[str, str] = {
        "buy": "enter_long",
        "sell": "exit_long",
    }

    def __init__(self, use_legacy_columns: bool = False) -> None:
        """
        Initialize the signal converter.

        Args:
            use_legacy_columns: Whether to check for legacy buy/sell columns.
        """
        self.use_legacy_columns = use_legacy_columns

    def convert_dataframe(
        self,
        df: pl.DataFrame,
        symbol: str,
        price_column: str = "close",
    ) -> Signal | None:
        """
        Convert Freqtrade Polars DataFrame signals to LIBRA Signal.

        Analyzes the last row of the DataFrame for signal columns
        and returns the highest priority signal found.

        Args:
            df: Polars DataFrame with Freqtrade indicator/signal columns.
            symbol: Trading pair symbol (e.g., "BTC/USDT").
            price_column: Column name for reference price.

        Returns:
            Signal object if a signal is present, None otherwise.

        Examples:
            signal = converter.convert_dataframe(df, "BTC/USDT")
        """
        if df.is_empty():
            return None

        # Get last row as dict
        last_row = df.row(-1, named=True)

        # Handle legacy columns
        if self.use_legacy_columns:
            last_row = self._map_legacy_columns_dict(last_row)

        # Check signals in priority order
        for column, trigger_value, signal_type in self.SIGNAL_PRIORITY:
            if column in df.columns and last_row.get(column) == trigger_value:
                return self._create_signal(
                    signal_type=signal_type,
                    symbol=symbol,
                    row=last_row,
                    price_column=price_column,
                )

        return None

    def convert_pandas_dataframe(
        self,
        df: Any,  # pandas.DataFrame
        symbol: str,
        price_column: str = "close",
    ) -> Signal | None:
        """
        Convert pandas DataFrame to Signal (for Freqtrade compatibility).

        This method accepts a pandas DataFrame and converts it to Polars
        internally for processing.

        Args:
            df: Pandas DataFrame with Freqtrade indicator/signal columns.
            symbol: Trading pair symbol.
            price_column: Column name for reference price.

        Returns:
            Signal object if a signal is present, None otherwise.
        """
        if df.empty:
            return None

        # Convert pandas to Polars
        polars_df = pl.from_pandas(df)
        return self.convert_dataframe(polars_df, symbol, price_column)

    def convert_row(
        self,
        row: dict[str, Any],
        symbol: str,
        price: Decimal | None = None,
    ) -> Signal | None:
        """
        Convert a single row/dict of Freqtrade signals to LIBRA Signal.

        Args:
            row: Dictionary with signal columns.
            symbol: Trading pair symbol.
            price: Optional reference price.

        Returns:
            Signal object if a signal is present, None otherwise.
        """
        # Handle legacy columns
        if self.use_legacy_columns:
            row = self._map_legacy_columns_dict(row)

        # Check signals in priority order
        for column, trigger_value, signal_type in self.SIGNAL_PRIORITY:
            if row.get(column) == trigger_value:
                return self._create_signal_from_row(
                    signal_type=signal_type,
                    symbol=symbol,
                    row=row,
                    price=price,
                )

        return None

    def _create_signal(
        self,
        signal_type: SignalType,
        symbol: str,
        row: dict[str, Any],
        price_column: str,
    ) -> Signal:
        """Create Signal from row dict."""
        # Extract price
        price: Decimal | None = None
        if price_column in row:
            try:
                price = Decimal(str(row[price_column]))
            except (ValueError, TypeError):
                pass

        # Build metadata from Freqtrade tags
        metadata = self._extract_metadata(row, signal_type)

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            timestamp_ns=time.time_ns(),
            price=price,
            metadata=metadata,
        )

    def _create_signal_from_row(
        self,
        signal_type: SignalType,
        symbol: str,
        row: dict[str, Any],
        price: Decimal | None,
    ) -> Signal:
        """Create Signal from dict row."""
        metadata = self._extract_metadata(row, signal_type)

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            timestamp_ns=time.time_ns(),
            price=price,
            metadata=metadata,
        )

    def _extract_metadata(
        self,
        row: dict[str, Any],
        signal_type: SignalType,
    ) -> dict[str, Any]:
        """
        Extract signal metadata from Freqtrade row.

        Captures enter_tag, exit_tag, and other relevant fields.
        """
        metadata: dict[str, Any] = {
            "source": "freqtrade",
        }

        # Entry tag
        if "enter_tag" in row:
            tag = row.get("enter_tag")
            if tag is not None and str(tag) not in ("nan", "None", ""):
                metadata["enter_tag"] = str(tag)

        # Exit tag
        if "exit_tag" in row:
            tag = row.get("exit_tag")
            if tag is not None and str(tag) not in ("nan", "None", ""):
                metadata["exit_tag"] = str(tag)

        # Add signal direction info
        if signal_type in (SignalType.LONG, SignalType.CLOSE_LONG):
            metadata["direction"] = "long"
        elif signal_type in (SignalType.SHORT, SignalType.CLOSE_SHORT):
            metadata["direction"] = "short"

        # Capture any custom data columns (prefixed with 'custom_')
        for key, value in row.items():
            if str(key).startswith("custom_"):
                if value is not None and str(value) not in ("nan", "None"):
                    metadata[str(key)] = value

        return metadata

    def _map_legacy_columns_dict(
        self,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Map legacy Freqtrade columns to current format.

        Freqtrade used 'buy'/'sell' before switching to
        'enter_long'/'exit_long' in 2021.
        """
        row = dict(row)
        for old_col, new_col in self.LEGACY_COLUMN_MAP.items():
            if old_col in row and new_col not in row:
                row[new_col] = row[old_col]
        return row

    @staticmethod
    def signal_to_freqtrade_columns(signal: Signal) -> dict[str, int | str]:
        """
        Convert LIBRA Signal back to Freqtrade column format.

        Useful for testing or writing back to DataFrames.

        Args:
            signal: LIBRA Signal object.

        Returns:
            Dictionary with Freqtrade column values.
        """
        result: dict[str, int | str] = {
            "enter_long": 0,
            "enter_short": 0,
            "exit_long": 0,
            "exit_short": 0,
        }

        match signal.signal_type:
            case SignalType.LONG:
                result["enter_long"] = 1
            case SignalType.SHORT:
                result["enter_short"] = 1
            case SignalType.CLOSE_LONG:
                result["exit_long"] = 1
            case SignalType.CLOSE_SHORT:
                result["exit_short"] = 1
            case SignalType.HOLD:
                pass  # All zeros

        # Add tags if present
        if "enter_tag" in signal.metadata:
            result["enter_tag"] = signal.metadata["enter_tag"]
        if "exit_tag" in signal.metadata:
            result["exit_tag"] = signal.metadata["exit_tag"]

        return result

    @staticmethod
    def polars_to_pandas(df: pl.DataFrame) -> Any:
        """
        Convert Polars DataFrame to pandas (for Freqtrade compatibility).

        Args:
            df: Polars DataFrame.

        Returns:
            pandas DataFrame.
        """
        return df.to_pandas()

    @staticmethod
    def pandas_to_polars(df: Any) -> pl.DataFrame:
        """
        Convert pandas DataFrame to Polars.

        Args:
            df: pandas DataFrame.

        Returns:
            Polars DataFrame.
        """
        return pl.from_pandas(df)

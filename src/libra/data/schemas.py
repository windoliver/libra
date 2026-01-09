"""
Polars Schemas: Standard DataFrame schemas for market data.

Provides:
- Type-safe schema definitions for all data types
- Validation functions for DataFrames
- Conversion utilities between formats

Based on Polars best practices:
- Use proper dtypes (Int64, Float64, Datetime, etc.)
- Decimal columns as Float64 (Polars doesn't have Decimal)
- Timestamps as Datetime[ns] for precision

See: https://github.com/windoliver/libra/issues/23
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import polars as pl


# =============================================================================
# OHLCV Schema
# =============================================================================

OHLCV_SCHEMA = {
    "timestamp": pl.Datetime("ns"),  # Nanosecond precision
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "timeframe": pl.Utf8,
    # Optional fields
    "trades": pl.Int64,
    "vwap": pl.Float64,
    "turnover": pl.Float64,
}

OHLCV_REQUIRED_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def create_ohlcv_df(
    data: list[dict[str, Any]] | None = None,
    *,
    timestamps: list[datetime] | None = None,
    symbol: str | None = None,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    closes: list[float] | None = None,
    volumes: list[float] | None = None,
    timeframe: str = "1m",
) -> pl.DataFrame:
    """
    Create an OHLCV DataFrame with proper schema.

    Args:
        data: List of dicts with OHLCV data
        timestamps: List of timestamps
        symbol: Symbol for all rows
        opens/highs/lows/closes/volumes: Price/volume arrays
        timeframe: Timeframe string

    Returns:
        Polars DataFrame with OHLCV schema
    """
    if data is not None:
        df = pl.DataFrame(data)
    else:
        if timestamps is None or opens is None or closes is None:
            raise ValueError("Must provide data or timestamps/opens/closes")

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": [symbol] * len(timestamps) if symbol else None,
                "open": opens,
                "high": highs or opens,
                "low": lows or opens,
                "close": closes,
                "volume": volumes or [0.0] * len(timestamps),
                "timeframe": [timeframe] * len(timestamps),
            }
        )

    return df.cast(
        {
            "timestamp": pl.Datetime("ns"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )


def validate_ohlcv(df: pl.DataFrame) -> list[str]:
    """
    Validate OHLCV DataFrame.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required columns
    for col in OHLCV_REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return errors

    # Check OHLC relationships
    invalid_rows = df.filter(
        (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
        | (pl.col("high") < pl.col("low"))
    )

    if len(invalid_rows) > 0:
        errors.append(f"Invalid OHLC relationships in {len(invalid_rows)} rows")

    # Check for negative values
    negative = df.filter(
        (pl.col("open") < 0)
        | (pl.col("high") < 0)
        | (pl.col("low") < 0)
        | (pl.col("close") < 0)
        | (pl.col("volume") < 0)
    )

    if len(negative) > 0:
        errors.append(f"Negative values in {len(negative)} rows")

    return errors


# =============================================================================
# Tick Schema
# =============================================================================

TICK_SCHEMA = {
    "timestamp": pl.Datetime("ns"),
    "symbol": pl.Utf8,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "last": pl.Float64,
    "bid_size": pl.Float64,
    "ask_size": pl.Float64,
    "volume_24h": pl.Float64,
}

TICK_REQUIRED_COLUMNS = ["timestamp", "symbol", "bid", "ask", "last"]


def create_tick_df(
    data: list[dict[str, Any]] | None = None,
    *,
    timestamps: list[datetime] | None = None,
    symbol: str | None = None,
    bids: list[float] | None = None,
    asks: list[float] | None = None,
    lasts: list[float] | None = None,
) -> pl.DataFrame:
    """Create a Tick DataFrame with proper schema."""
    if data is not None:
        df = pl.DataFrame(data)
    else:
        if timestamps is None or bids is None or asks is None:
            raise ValueError("Must provide data or timestamps/bids/asks")

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": [symbol] * len(timestamps) if symbol else None,
                "bid": bids,
                "ask": asks,
                "last": lasts or [(b + a) / 2 for b, a in zip(bids, asks)],
            }
        )

    return df.cast(
        {
            "timestamp": pl.Datetime("ns"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "last": pl.Float64,
        }
    )


def validate_tick(df: pl.DataFrame) -> list[str]:
    """Validate Tick DataFrame."""
    errors = []

    for col in TICK_REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return errors

    # Check bid <= ask
    invalid = df.filter(pl.col("bid") > pl.col("ask"))
    if len(invalid) > 0:
        errors.append(f"Bid > Ask in {len(invalid)} rows")

    return errors


# =============================================================================
# Trade Schema
# =============================================================================

TRADE_SCHEMA = {
    "timestamp": pl.Datetime("ns"),
    "trade_id": pl.Utf8,
    "symbol": pl.Utf8,
    "side": pl.Utf8,  # "buy" or "sell"
    "price": pl.Float64,
    "amount": pl.Float64,
    "cost": pl.Float64,
    "fee": pl.Float64,
    "fee_currency": pl.Utf8,
}

TRADE_REQUIRED_COLUMNS = ["timestamp", "symbol", "side", "price", "amount"]


def create_trade_df(data: list[dict[str, Any]]) -> pl.DataFrame:
    """Create a Trade DataFrame with proper schema."""
    df = pl.DataFrame(data)
    return df.cast(
        {
            "timestamp": pl.Datetime("ns"),
            "price": pl.Float64,
            "amount": pl.Float64,
            "cost": pl.Float64,
            "fee": pl.Float64,
        }
    )


# =============================================================================
# Order Book Schema
# =============================================================================

ORDERBOOK_SCHEMA = {
    "timestamp": pl.Datetime("ns"),
    "symbol": pl.Utf8,
    "side": pl.Utf8,  # "bid" or "ask"
    "price": pl.Float64,
    "size": pl.Float64,
    "level": pl.Int32,  # 0 = best bid/ask
}


def create_orderbook_df(
    symbol: str,
    timestamp: datetime,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> pl.DataFrame:
    """
    Create Order Book DataFrame from bid/ask levels.

    Args:
        symbol: Trading pair
        timestamp: Snapshot timestamp
        bids: List of (price, size) tuples, best first
        asks: List of (price, size) tuples, best first

    Returns:
        DataFrame with order book levels
    """
    rows = []

    for i, (price, size) in enumerate(bids):
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "bid",
                "price": price,
                "size": size,
                "level": i,
            }
        )

    for i, (price, size) in enumerate(asks):
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "ask",
                "price": price,
                "size": size,
                "level": i,
            }
        )

    return pl.DataFrame(rows).cast(
        {
            "timestamp": pl.Datetime("ns"),
            "price": pl.Float64,
            "size": pl.Float64,
            "level": pl.Int32,
        }
    )


# =============================================================================
# Signal Schema
# =============================================================================

SIGNAL_SCHEMA = {
    "timestamp": pl.Datetime("ns"),
    "symbol": pl.Utf8,
    "signal_type": pl.Utf8,  # "LONG", "SHORT", "CLOSE_LONG", etc.
    "strength": pl.Float64,  # 0.0 to 1.0
    "price": pl.Float64,
    "strategy": pl.Utf8,
    "metadata": pl.Utf8,  # JSON string
}


# =============================================================================
# Conversion Utilities
# =============================================================================


def decimal_to_float(value: Decimal | float | None) -> float | None:
    """Convert Decimal to float for Polars."""
    if value is None:
        return None
    return float(value)


def bars_to_df(bars: list[Any]) -> pl.DataFrame:
    """
    Convert list of Bar objects to Polars DataFrame.

    Args:
        bars: List of Bar msgspec structs

    Returns:
        OHLCV DataFrame
    """
    if not bars:
        return pl.DataFrame(schema=OHLCV_SCHEMA)

    data = []
    for bar in bars:
        data.append(
            {
                "timestamp": datetime.fromtimestamp(bar.timestamp_ns / 1e9),
                "symbol": bar.symbol,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "timeframe": bar.timeframe,
                "trades": bar.trades,
                "vwap": decimal_to_float(bar.vwap),
                "turnover": decimal_to_float(bar.turnover),
            }
        )

    return pl.DataFrame(data).cast(
        {
            "timestamp": pl.Datetime("ns"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )


def ticks_to_df(ticks: list[Any]) -> pl.DataFrame:
    """
    Convert list of Tick objects to Polars DataFrame.

    Args:
        ticks: List of Tick msgspec structs

    Returns:
        Tick DataFrame
    """
    if not ticks:
        return pl.DataFrame(schema=TICK_SCHEMA)

    data = []
    for tick in ticks:
        data.append(
            {
                "timestamp": datetime.fromtimestamp(tick.timestamp_ns / 1e9),
                "symbol": tick.symbol,
                "bid": float(tick.bid),
                "ask": float(tick.ask),
                "last": float(tick.last),
                "bid_size": decimal_to_float(tick.bid_size),
                "ask_size": decimal_to_float(tick.ask_size),
                "volume_24h": decimal_to_float(tick.volume_24h),
            }
        )

    return pl.DataFrame(data).cast(
        {
            "timestamp": pl.Datetime("ns"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "last": pl.Float64,
        }
    )


# =============================================================================
# Technical Indicator Helpers
# =============================================================================


def add_sma(df: pl.LazyFrame, column: str, period: int, alias: str | None = None) -> pl.LazyFrame:
    """
    Add Simple Moving Average column.

    Args:
        df: Input LazyFrame
        column: Column to calculate SMA on
        period: SMA period
        alias: Output column name (default: sma_{period})

    Returns:
        LazyFrame with SMA column added
    """
    name = alias or f"sma_{period}"
    return df.with_columns(pl.col(column).rolling_mean(window_size=period).alias(name))


def add_ema(df: pl.LazyFrame, column: str, period: int, alias: str | None = None) -> pl.LazyFrame:
    """
    Add Exponential Moving Average column.

    Args:
        df: Input LazyFrame
        column: Column to calculate EMA on
        period: EMA period (span)
        alias: Output column name (default: ema_{period})

    Returns:
        LazyFrame with EMA column added
    """
    name = alias or f"ema_{period}"
    return df.with_columns(pl.col(column).ewm_mean(span=period).alias(name))


def add_rsi(df: pl.LazyFrame, column: str = "close", period: int = 14) -> pl.LazyFrame:
    """
    Add Relative Strength Index column.

    Args:
        df: Input LazyFrame with price column
        column: Price column name
        period: RSI period (default 14)

    Returns:
        LazyFrame with rsi column added
    """
    return df.with_columns(
        [
            (pl.col(column) - pl.col(column).shift(1)).alias("_change"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("_change") > 0).then(pl.col("_change")).otherwise(0).alias("_gain"),
            pl.when(pl.col("_change") < 0)
            .then(pl.col("_change").abs())
            .otherwise(0)
            .alias("_loss"),
        ]
    ).with_columns(
        [
            pl.col("_gain").rolling_mean(window_size=period).alias("_avg_gain"),
            pl.col("_loss").rolling_mean(window_size=period).alias("_avg_loss"),
        ]
    ).with_columns(
        [
            (
                100
                - (100 / (1 + pl.col("_avg_gain") / pl.col("_avg_loss").replace(0, 1e-10)))
            ).alias("rsi"),
        ]
    ).drop(
        ["_change", "_gain", "_loss", "_avg_gain", "_avg_loss"]
    )


def add_bollinger_bands(
    df: pl.LazyFrame,
    column: str = "close",
    period: int = 20,
    std_dev: float = 2.0,
) -> pl.LazyFrame:
    """
    Add Bollinger Bands columns.

    Args:
        df: Input LazyFrame
        column: Price column
        period: SMA period
        std_dev: Standard deviation multiplier

    Returns:
        LazyFrame with bb_middle, bb_upper, bb_lower columns
    """
    return df.with_columns(
        [
            pl.col(column).rolling_mean(window_size=period).alias("bb_middle"),
            pl.col(column).rolling_std(window_size=period).alias("_bb_std"),
        ]
    ).with_columns(
        [
            (pl.col("bb_middle") + std_dev * pl.col("_bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - std_dev * pl.col("_bb_std")).alias("bb_lower"),
        ]
    ).drop("_bb_std")

"""
Downsampling: OHLCV aggregation and time-based resampling.

Provides:
- OHLCV-preserving downsampling (1m -> 5m -> 1h -> 1d)
- Configurable aggregation rules
- Automatic resolution reduction for older data
- Data integrity validation

Performance target: 1M rows aggregation < 100ms

See: https://github.com/windoliver/libra/issues/23
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import polars as pl


logger = logging.getLogger(__name__)


# =============================================================================
# Resolution Definitions
# =============================================================================

# Standard resolutions and their durations in seconds
RESOLUTION_SECONDS: dict[str, int] = {
    "1s": 1,
    "5s": 5,
    "10s": 10,
    "30s": 30,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,  # ~30 days
}

# Valid downsampling paths (source -> target)
VALID_DOWNSAMPLE: dict[str, list[str]] = {
    "1s": ["5s", "10s", "30s", "1m"],
    "5s": ["10s", "30s", "1m"],
    "10s": ["30s", "1m"],
    "30s": ["1m", "5m"],
    "1m": ["3m", "5m", "15m", "30m", "1h"],
    "3m": ["15m", "30m", "1h"],
    "5m": ["15m", "30m", "1h"],
    "15m": ["30m", "1h", "4h"],
    "30m": ["1h", "2h", "4h"],
    "1h": ["2h", "4h", "6h", "12h", "1d"],
    "2h": ["4h", "6h", "12h", "1d"],
    "4h": ["8h", "12h", "1d"],
    "6h": ["12h", "1d"],
    "12h": ["1d"],
    "1d": ["3d", "1w"],
    "3d": ["1w"],
    "1w": ["1M"],
}


def get_resolution_seconds(resolution: str) -> int:
    """Get resolution duration in seconds."""
    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(f"Unknown resolution: {resolution}")
    return RESOLUTION_SECONDS[resolution]


def can_downsample(from_res: str, to_res: str) -> bool:
    """Check if downsampling from one resolution to another is valid."""
    if from_res not in VALID_DOWNSAMPLE:
        return False
    return to_res in VALID_DOWNSAMPLE[from_res]


def find_downsample_path(from_res: str, to_res: str) -> list[str]:
    """
    Find a valid downsampling path between resolutions.

    Uses breadth-first search to find shortest path.

    Args:
        from_res: Source resolution
        to_res: Target resolution

    Returns:
        List of resolutions forming the path (including from_res and to_res)

    Raises:
        ValueError: If no valid path exists
    """
    if from_res == to_res:
        return [from_res]

    if from_res not in VALID_DOWNSAMPLE:
        raise ValueError(f"Unknown source resolution: {from_res}")

    # BFS to find shortest path
    from collections import deque

    queue = deque([(from_res, [from_res])])
    visited = {from_res}

    while queue:
        current, path = queue.popleft()

        if current not in VALID_DOWNSAMPLE:
            continue

        for next_res in VALID_DOWNSAMPLE[current]:
            if next_res == to_res:
                return path + [to_res]

            if next_res not in visited:
                visited.add(next_res)
                queue.append((next_res, path + [next_res]))

    raise ValueError(f"No valid downsampling path from {from_res} to {to_res}")


def downsample_multi_step(
    df: pl.DataFrame,
    from_resolution: str,
    to_resolution: str,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """
    Downsample using multiple steps if needed.

    Finds and applies valid downsampling path.

    Args:
        df: Input DataFrame
        from_resolution: Source resolution
        to_resolution: Target resolution
        timestamp_col: Timestamp column name

    Returns:
        Downsampled DataFrame
    """
    path = find_downsample_path(from_resolution, to_resolution)

    result = df
    for i in range(len(path) - 1):
        result = downsample_ohlcv(result, path[i], path[i + 1], timestamp_col)

    return result


def get_downsample_factor(from_res: str, to_res: str) -> int:
    """Calculate the downsampling factor between resolutions."""
    from_sec = get_resolution_seconds(from_res)
    to_sec = get_resolution_seconds(to_res)

    if to_sec < from_sec:
        raise ValueError(f"Cannot upsample from {from_res} to {to_res}")

    if to_sec % from_sec != 0:
        raise ValueError(f"Cannot evenly downsample from {from_res} to {to_res}")

    return to_sec // from_sec


# =============================================================================
# OHLCV Downsampling
# =============================================================================


def downsample_ohlcv(
    df: pl.DataFrame | pl.LazyFrame,
    from_resolution: str,
    to_resolution: str,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """
    Downsample OHLCV data while preserving integrity.

    Aggregation rules:
    - open: First value in period
    - high: Maximum in period
    - low: Minimum in period
    - close: Last value in period
    - volume: Sum of period
    - trades: Sum (if present)
    - vwap: Volume-weighted (if present)

    Args:
        df: Input DataFrame with OHLCV columns
        from_resolution: Source resolution (e.g., "1m")
        to_resolution: Target resolution (e.g., "5m")
        timestamp_col: Name of timestamp column

    Returns:
        Downsampled DataFrame

    Raises:
        ValueError: If downsampling is not valid

    Example:
        # Downsample 1-minute to 5-minute bars
        df_5m = downsample_ohlcv(df_1m, "1m", "5m")
    """
    if not can_downsample(from_resolution, to_resolution):
        raise ValueError(f"Cannot downsample from {from_resolution} to {to_resolution}")

    # Convert to LazyFrame for optimization
    if isinstance(df, pl.DataFrame):
        lf = df.lazy()
    else:
        lf = df

    # Get period duration
    period_seconds = get_resolution_seconds(to_resolution)
    period = f"{period_seconds}s"

    # Build aggregation expressions
    agg_exprs = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]

    # Optional columns - use collect_schema().names() for LazyFrame to avoid performance warning
    if isinstance(df, pl.DataFrame):
        columns = df.columns
    else:
        columns = lf.collect_schema().names()

    if "trades" in columns:
        agg_exprs.append(pl.col("trades").sum().alias("trades"))

    if "vwap" in columns and "volume" in columns:
        # Volume-weighted average price
        agg_exprs.append(
            (pl.col("vwap") * pl.col("volume")).sum()
            / pl.col("volume").sum().alias("vwap")
        )

    if "turnover" in columns:
        agg_exprs.append(pl.col("turnover").sum().alias("turnover"))

    # Group by time period and aggregate
    result = (
        lf.sort(timestamp_col)
        .group_by_dynamic(timestamp_col, every=period)
        .agg(agg_exprs)
        .with_columns(pl.lit(to_resolution).alias("timeframe"))
        .collect()
    )

    input_rows = df.height if isinstance(df, pl.DataFrame) else "?"
    logger.debug(
        f"Downsampled {input_rows} rows ({from_resolution}) to {len(result)} rows ({to_resolution})"
    )

    return result


def downsample_ohlcv_lazy(
    lf: pl.LazyFrame,
    from_resolution: str,
    to_resolution: str,
    timestamp_col: str = "timestamp",
) -> pl.LazyFrame:
    """
    Lazy version of downsample_ohlcv for query optimization.

    Returns LazyFrame instead of collecting immediately.
    """
    if not can_downsample(from_resolution, to_resolution):
        raise ValueError(f"Cannot downsample from {from_resolution} to {to_resolution}")

    period_seconds = get_resolution_seconds(to_resolution)
    period = f"{period_seconds}s"

    agg_exprs = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]

    return (
        lf.sort(timestamp_col)
        .group_by_dynamic(timestamp_col, every=period)
        .agg(agg_exprs)
        .with_columns(pl.lit(to_resolution).alias("timeframe"))
    )


# =============================================================================
# Multi-Resolution Downsampling
# =============================================================================


def downsample_to_multiple(
    df: pl.DataFrame,
    from_resolution: str,
    target_resolutions: list[str],
    timestamp_col: str = "timestamp",
) -> dict[str, pl.DataFrame]:
    """
    Downsample to multiple resolutions efficiently.

    Uses intermediate results when possible for efficiency.

    Args:
        df: Source DataFrame
        from_resolution: Source resolution
        target_resolutions: List of target resolutions
        timestamp_col: Timestamp column name

    Returns:
        Dict mapping resolution to DataFrame

    Example:
        results = downsample_to_multiple(df_1m, "1m", ["5m", "15m", "1h"])
        df_5m = results["5m"]
        df_1h = results["1h"]
    """
    results: dict[str, pl.DataFrame] = {}

    # Sort targets by granularity (finest first)
    sorted_targets = sorted(
        target_resolutions,
        key=lambda r: get_resolution_seconds(r),
    )

    # Track which source to use for each target
    current_df = df
    current_res = from_resolution

    for target_res in sorted_targets:
        # Find best source (either original or intermediate)
        if can_downsample(current_res, target_res):
            results[target_res] = downsample_ohlcv(
                current_df, current_res, target_res, timestamp_col
            )
            # Use this as source for next (if beneficial)
            if len(results[target_res]) < len(current_df) // 2:
                current_df = results[target_res]
                current_res = target_res
        elif can_downsample(from_resolution, target_res):
            # Fall back to original source
            results[target_res] = downsample_ohlcv(
                df, from_resolution, target_res, timestamp_col
            )

    return results


# =============================================================================
# Age-Based Downsampling
# =============================================================================


@pl.api.register_dataframe_namespace("libra")
class LibraDataFrameExtension:
    """Polars DataFrame extension for LIBRA operations."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def downsample(
        self,
        from_res: str,
        to_res: str,
        timestamp_col: str = "timestamp",
    ) -> pl.DataFrame:
        """Downsample OHLCV data."""
        return downsample_ohlcv(self._df, from_res, to_res, timestamp_col)


def apply_retention_policy(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    rules: list[tuple[timedelta, str]] | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Apply retention policy with automatic downsampling.

    Rules specify age thresholds and target resolutions.

    Args:
        df: Source DataFrame (assumed 1m resolution)
        timestamp_col: Timestamp column
        rules: List of (age_threshold, target_resolution) tuples
               Data older than threshold gets downsampled

    Returns:
        Dict of resolution -> DataFrame

    Example:
        rules = [
            (timedelta(days=1), "1m"),    # Last day: keep 1m
            (timedelta(days=7), "5m"),    # Last week: 5m
            (timedelta(days=30), "1h"),   # Last month: 1h
            (timedelta(days=365), "1d"),  # Last year: 1d
        ]
        results = apply_retention_policy(df, rules=rules)
    """
    if rules is None:
        # Default retention policy
        rules = [
            (timedelta(days=1), "1m"),
            (timedelta(days=7), "5m"),
            (timedelta(days=30), "1h"),
            (timedelta(days=365), "1d"),
        ]

    now = datetime.utcnow()
    results: dict[str, pl.DataFrame] = {}

    # Sort rules by age (newest first)
    sorted_rules = sorted(rules, key=lambda r: r[0])

    prev_cutoff = now
    for age_threshold, resolution in sorted_rules:
        cutoff = now - age_threshold

        # Filter data in this age range
        mask = (pl.col(timestamp_col) >= cutoff) & (pl.col(timestamp_col) < prev_cutoff)
        segment = df.filter(mask)

        if len(segment) > 0:
            if resolution == "1m":
                # No downsampling needed
                if resolution in results:
                    results[resolution] = pl.concat([results[resolution], segment])
                else:
                    results[resolution] = segment
            else:
                # Downsample this segment (using multi-step if needed)
                downsampled = downsample_multi_step(segment, "1m", resolution, timestamp_col)
                if resolution in results:
                    results[resolution] = pl.concat([results[resolution], downsampled])
                else:
                    results[resolution] = downsampled

        prev_cutoff = cutoff

    return results


# =============================================================================
# Validation
# =============================================================================


def validate_ohlcv_integrity(df: pl.DataFrame) -> list[str]:
    """
    Validate OHLCV data integrity after downsampling.

    Checks:
    - High >= Open, Close, Low
    - Low <= Open, Close, High
    - Volume >= 0
    - No NaN values in OHLC

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check OHLC relationships
    invalid_high = df.filter(
        (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("high") < pl.col("low"))
    )
    if len(invalid_high) > 0:
        errors.append(f"High < OHLC in {len(invalid_high)} rows")

    invalid_low = df.filter(
        (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
        | (pl.col("low") > pl.col("high"))
    )
    if len(invalid_low) > 0:
        errors.append(f"Low > OHLC in {len(invalid_low)} rows")

    # Check for negative volume
    if "volume" in df.columns:
        neg_vol = df.filter(pl.col("volume") < 0)
        if len(neg_vol) > 0:
            errors.append(f"Negative volume in {len(neg_vol)} rows")

    # Check for nulls in OHLC
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            null_count = df.filter(pl.col(col).is_null()).height
            if null_count > 0:
                errors.append(f"Null values in {col}: {null_count}")

    return errors


# =============================================================================
# Tick to Bar Aggregation
# =============================================================================


def ticks_to_bars(
    df: pl.DataFrame | pl.LazyFrame,
    resolution: str = "1m",
    timestamp_col: str = "timestamp",
    price_col: str = "last",
    size_col: str | None = None,
) -> pl.DataFrame:
    """
    Aggregate tick data to OHLCV bars.

    Args:
        df: Tick DataFrame with timestamp, price, and optionally size
        resolution: Target bar resolution
        timestamp_col: Timestamp column name
        price_col: Price column name (default: "last")
        size_col: Size/volume column name (optional)

    Returns:
        OHLCV DataFrame

    Example:
        bars = ticks_to_bars(ticks_df, "1m", size_col="amount")
    """
    if isinstance(df, pl.DataFrame):
        lf = df.lazy()
    else:
        lf = df

    period_seconds = get_resolution_seconds(resolution)
    period = f"{period_seconds}s"

    agg_exprs = [
        pl.col(price_col).first().alias("open"),
        pl.col(price_col).max().alias("high"),
        pl.col(price_col).min().alias("low"),
        pl.col(price_col).last().alias("close"),
        pl.count().alias("trades"),
    ]

    if size_col:
        agg_exprs.append(pl.col(size_col).sum().alias("volume"))
        # VWAP = sum(price * size) / sum(size)
        agg_exprs.append(
            ((pl.col(price_col) * pl.col(size_col)).sum() / pl.col(size_col).sum()).alias(
                "vwap"
            )
        )
    else:
        agg_exprs.append(pl.lit(0.0).alias("volume"))

    result = (
        lf.sort(timestamp_col)
        .group_by_dynamic(timestamp_col, every=period)
        .agg(agg_exprs)
        .with_columns(pl.lit(resolution).alias("timeframe"))
        .collect()
    )

    return result

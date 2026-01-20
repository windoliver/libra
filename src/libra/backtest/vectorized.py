"""
Vectorized Backtest Engine.

Implements vectorbt-style fully vectorized backtesting for ~1000x speedup
over event-driven approaches. Uses Polars for high-performance computation.

Key Benefits:
- ~1000x faster than event-driven backtesting
- Perfect for parameter optimization (thousands of runs)
- Ideal for simple signal-based strategies

Limitations:
- Less flexible than event-driven (no complex execution logic)
- Cannot model order book dynamics
- Best for research, not production

Issue #104: Implement vectorbt-style vectorized backtest
Reference: https://vectorbt.dev/
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
NS_PER_DAY = 86400 * 1_000_000_000


@dataclass
class VectorizedResult:
    """
    Result from vectorized backtest.

    Contains equity curve as Polars DataFrame and computed metrics.
    """

    # Equity curve DataFrame
    equity_df: pl.DataFrame

    # Core metrics
    initial_capital: float
    final_equity: float
    total_return: float
    total_return_pct: float
    cagr: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # Trade statistics (approximate from position changes)
    num_trades: int
    win_rate: float
    profit_factor: float

    # Metadata
    duration_days: float
    bars_processed: int
    execution_time_ms: float

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 60)
        print("Vectorized Backtest Results")
        print("=" * 60)
        print(f"Duration: {self.duration_days:.1f} days | Bars: {self.bars_processed:,}")
        print(f"Execution Time: {self.execution_time_ms:.2f}ms")
        print("-" * 60)
        print("PERFORMANCE")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Equity:    ${self.final_equity:,.2f}")
        print(f"  Total Return:    ${self.total_return:,.2f} ({self.total_return_pct * 100:.2f}%)")
        print(f"  CAGR:            {self.cagr * 100:.2f}%")
        print("-" * 60)
        print("RISK METRICS")
        print(f"  Sharpe Ratio:    {self.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:   {self.sortino_ratio:.2f}")
        print(f"  Calmar Ratio:    {self.calmar_ratio:.2f}")
        print(f"  Max Drawdown:    {self.max_drawdown_pct * 100:.2f}%")
        print("-" * 60)
        print("TRADE STATISTICS")
        print(f"  Approx Trades:   {self.num_trades}")
        print(f"  Win Rate:        {self.win_rate * 100:.1f}%")
        print(f"  Profit Factor:   {self.profit_factor:.2f}")
        print("=" * 60 + "\n")


@dataclass
class VectorizedConfig:
    """Configuration for vectorized backtest."""

    initial_capital: float = 100_000.0
    commission_pct: float = 0.001  # 0.1% per trade (10 bps)
    slippage_pct: float = 0.0005  # 0.05% slippage (5 bps)
    risk_free_rate: float = 0.0  # Annualized risk-free rate


class VectorizedBacktest:
    """
    Vectorized backtest engine using Polars.

    Implements vectorbt-style fully vectorized backtesting for massive speedup.
    All operations are performed on entire arrays at once using Polars expressions.

    Example:
        backtest = VectorizedBacktest()

        # Create signals DataFrame
        signals = pl.DataFrame({
            "symbol": ["BTC/USDT"] * 100,
            "timestamp": range(100),
            "signal": [1, 0, 0, -1, 0, ...],  # 1=buy, -1=sell, 0=hold
        })

        # Create prices DataFrame
        prices = pl.DataFrame({
            "symbol": ["BTC/USDT"] * 100,
            "timestamp": range(100),
            "close": [100.0, 101.0, 99.5, ...],
        })

        result = backtest.run(signals, prices)
        result.print_summary()
    """

    def __init__(self, config: VectorizedConfig | None = None) -> None:
        """Initialize vectorized backtest engine."""
        self.config = config or VectorizedConfig()

    def run(
        self,
        signals: pl.DataFrame,
        prices: pl.DataFrame,
        initial_capital: float | None = None,
    ) -> VectorizedResult:
        """
        Run vectorized backtest.

        Args:
            signals: DataFrame with columns (symbol, timestamp, signal)
                     signal: 1 = buy/long, -1 = sell/short, 0 = hold
            prices: DataFrame with columns (symbol, timestamp, close)
            initial_capital: Override config initial capital

        Returns:
            VectorizedResult with equity curve and metrics
        """
        start_time = time.perf_counter()
        capital = initial_capital or self.config.initial_capital

        # Validate inputs
        self._validate_inputs(signals, prices)

        # Join signals with prices
        df = signals.join(prices, on=["symbol", "timestamp"], how="inner")

        # Process by symbol for multi-asset support
        symbols = df["symbol"].unique().to_list()

        if len(symbols) == 1:
            # Single symbol - simpler path
            equity_df = self._run_single_symbol(df, capital)
        else:
            # Multi-symbol - aggregate
            equity_df = self._run_multi_symbol(df, capital, symbols)

        # Compute metrics
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        result = self._compute_metrics(equity_df, capital, execution_time_ms)

        logger.info(
            "Vectorized backtest completed: %.2fms, %d bars, Sharpe=%.2f",
            execution_time_ms,
            result.bars_processed,
            result.sharpe_ratio,
        )

        return result

    def _validate_inputs(self, signals: pl.DataFrame, prices: pl.DataFrame) -> None:
        """Validate input DataFrames."""
        required_signal_cols = {"symbol", "timestamp", "signal"}
        required_price_cols = {"symbol", "timestamp", "close"}

        signal_cols = set(signals.columns)
        price_cols = set(prices.columns)

        if not required_signal_cols.issubset(signal_cols):
            missing = required_signal_cols - signal_cols
            raise ValueError(f"signals DataFrame missing columns: {missing}")

        if not required_price_cols.issubset(price_cols):
            missing = required_price_cols - price_cols
            raise ValueError(f"prices DataFrame missing columns: {missing}")

        if signals.height == 0:
            raise ValueError("signals DataFrame is empty")

        if prices.height == 0:
            raise ValueError("prices DataFrame is empty")

    def _run_single_symbol(
        self,
        df: pl.DataFrame,
        initial_capital: float,
    ) -> pl.DataFrame:
        """Run backtest for single symbol."""
        commission = self.config.commission_pct
        slippage = self.config.slippage_pct

        # Sort by timestamp
        df = df.sort("timestamp")

        # Vectorized position calculation (cumulative sum of signals)
        # Position represents number of units held: positive = long, negative = short
        df = df.with_columns([
            pl.col("signal").cum_sum().alias("position"),
        ])

        # Calculate returns
        df = df.with_columns([
            pl.col("close").pct_change().fill_null(0.0).alias("price_return"),
        ])

        # Position-weighted returns (lagged position * current return)
        df = df.with_columns([
            (pl.col("price_return") * pl.col("position").shift(1).fill_null(0))
            .alias("strategy_return"),
        ])

        # Apply transaction costs on position changes
        df = df.with_columns([
            (pl.col("position") - pl.col("position").shift(1).fill_null(0))
            .abs()
            .alias("position_change"),
        ])

        # Transaction cost = |position_change| * (commission + slippage)
        df = df.with_columns([
            (pl.col("position_change") * (commission + slippage)).alias("transaction_cost"),
        ])

        # Net returns after costs
        df = df.with_columns([
            (pl.col("strategy_return") - pl.col("transaction_cost")).alias("net_return"),
        ])

        # Equity curve: initial_capital * cumulative product of (1 + net_return)
        df = df.with_columns([
            ((1 + pl.col("net_return")).cum_prod() * initial_capital).alias("equity"),
        ])

        # Calculate drawdown
        df = df.with_columns([
            pl.col("equity").cum_max().alias("peak_equity"),
        ])

        df = df.with_columns([
            (pl.col("peak_equity") - pl.col("equity")).alias("drawdown"),
            ((pl.col("peak_equity") - pl.col("equity")) / pl.col("peak_equity"))
            .alias("drawdown_pct"),
        ])

        return df.select([
            "timestamp",
            "close",
            "signal",
            "position",
            "net_return",
            "equity",
            "drawdown",
            "drawdown_pct",
        ])

    def _run_multi_symbol(
        self,
        df: pl.DataFrame,
        initial_capital: float,
        symbols: list[str],
    ) -> pl.DataFrame:
        """Run backtest for multiple symbols with equal weight allocation."""
        # Allocate capital equally across symbols
        per_symbol_capital = initial_capital / len(symbols)

        # Process each symbol
        symbol_dfs = []
        for symbol in symbols:
            symbol_df = df.filter(pl.col("symbol") == symbol)
            result_df = self._run_single_symbol(symbol_df, per_symbol_capital)
            result_df = result_df.with_columns([
                pl.lit(symbol).alias("symbol"),
            ])
            symbol_dfs.append(result_df)

        # Combine and aggregate by timestamp
        combined = pl.concat(symbol_dfs)

        # Aggregate equity across symbols
        equity_df = combined.group_by("timestamp").agg([
            pl.col("equity").sum().alias("equity"),
            pl.col("net_return").mean().alias("net_return"),
            pl.col("position").sum().alias("position"),
        ]).sort("timestamp")

        # Recalculate drawdown on aggregated equity
        equity_df = equity_df.with_columns([
            pl.col("equity").cum_max().alias("peak_equity"),
        ])

        equity_df = equity_df.with_columns([
            (pl.col("peak_equity") - pl.col("equity")).alias("drawdown"),
            ((pl.col("peak_equity") - pl.col("equity")) / pl.col("peak_equity"))
            .alias("drawdown_pct"),
        ])

        return equity_df

    def _compute_metrics(
        self,
        equity_df: pl.DataFrame,
        initial_capital: float,
        execution_time_ms: float,
    ) -> VectorizedResult:
        """Compute performance metrics from equity curve."""
        # Extract arrays for computation
        equity = equity_df["equity"].to_numpy()
        returns = equity_df["net_return"].to_numpy()
        drawdown_pct = equity_df["drawdown_pct"].to_numpy()
        positions = equity_df["position"].to_numpy()

        bars = len(equity)
        final_equity = float(equity[-1]) if bars > 0 else initial_capital

        # Basic returns
        total_return = final_equity - initial_capital
        total_return_pct = total_return / initial_capital if initial_capital > 0 else 0.0

        # Duration (assuming daily data, adjust based on actual timestamps)
        timestamps = equity_df["timestamp"].to_numpy()
        if len(timestamps) > 1:
            # Check if timestamps are nanoseconds
            ts_diff = timestamps[-1] - timestamps[0]
            if ts_diff > 1e15:  # Likely nanoseconds
                duration_days = ts_diff / NS_PER_DAY
            else:
                # Assume integer indices or seconds
                duration_days = bars / TRADING_DAYS_PER_YEAR * 365
        else:
            duration_days = 1.0

        # CAGR
        years = duration_days / 365
        if years > 0 and initial_capital > 0 and final_equity > 0:
            cagr = (final_equity / initial_capital) ** (1 / years) - 1
        else:
            cagr = 0.0

        # Sharpe ratio (annualized)
        sharpe = self._calculate_sharpe(returns)

        # Sortino ratio (annualized)
        sortino = self._calculate_sortino(returns)

        # Max drawdown
        max_dd_pct = float(np.max(drawdown_pct)) if len(drawdown_pct) > 0 else 0.0
        max_dd = max_dd_pct * initial_capital

        # Calmar ratio
        calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0

        # Trade statistics (approximate from position changes)
        position_changes = np.diff(positions, prepend=0)
        num_trades = int(np.sum(position_changes != 0))

        # Win rate (approximate from positive returns when in position)
        in_position = np.abs(np.roll(positions, 1)) > 0
        in_position[0] = False
        position_returns = returns[in_position]

        if len(position_returns) > 0:
            wins = np.sum(position_returns > 0)
            win_rate = wins / len(position_returns)

            # Profit factor
            gross_profit = np.sum(position_returns[position_returns > 0])
            gross_loss = abs(np.sum(position_returns[position_returns < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return VectorizedResult(
            equity_df=equity_df,
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float("inf") else 999.99,
            duration_days=duration_days,
            bars_processed=bars,
            execution_time_ms=execution_time_ms,
        )

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        daily_rf = self.config.risk_free_rate / TRADING_DAYS_PER_YEAR
        return float((mean_return - daily_rf) / std_return * math.sqrt(TRADING_DAYS_PER_YEAR))

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        daily_rf = self.config.risk_free_rate / TRADING_DAYS_PER_YEAR

        # Downside deviation (only negative returns)
        downside_returns = np.minimum(returns - daily_rf, 0)
        downside_std = np.sqrt(np.mean(downside_returns**2))

        if downside_std == 0:
            return 0.0

        return float((mean_return - daily_rf) / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR))

    def run_optimization(
        self,
        signals_generator: callable,
        prices: pl.DataFrame,
        param_grid: dict[str, list],
    ) -> pl.DataFrame:
        """
        Run parameter optimization over multiple parameter combinations.

        This is where vectorized backtesting really shines - running thousands
        of backtests in seconds.

        Args:
            signals_generator: Function(params) -> signals DataFrame
            prices: Prices DataFrame
            param_grid: Dict of parameter name -> list of values

        Returns:
            DataFrame with parameter combinations and their metrics
        """
        import itertools

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        results = []
        total = len(combinations)

        logger.info("Running optimization over %d parameter combinations", total)
        start_time = time.perf_counter()

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            signals = signals_generator(params)
            result = self.run(signals, prices)

            row = {
                **params,
                "sharpe": result.sharpe_ratio,
                "sortino": result.sortino_ratio,
                "cagr": result.cagr,
                "max_dd": result.max_drawdown_pct,
                "total_return": result.total_return_pct,
                "num_trades": result.num_trades,
            }
            results.append(row)

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - start_time
                rate = (i + 1) / elapsed
                logger.info("Progress: %d/%d (%.1f runs/sec)", i + 1, total, rate)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Optimization complete: %d combinations in %.2fs (%.1f runs/sec)",
            total,
            total_time,
            total / total_time,
        )

        return pl.DataFrame(results).sort("sharpe", descending=True)


def generate_moving_average_signals(
    prices: pl.DataFrame,
    fast_period: int = 10,
    slow_period: int = 50,
) -> pl.DataFrame:
    """
    Generate moving average crossover signals.

    Utility function for common MA crossover strategy.

    Args:
        prices: DataFrame with columns (symbol, timestamp, close)
        fast_period: Fast MA period
        slow_period: Slow MA period

    Returns:
        signals DataFrame with columns (symbol, timestamp, signal)
    """
    df = prices.sort(["symbol", "timestamp"])

    # Calculate MAs per symbol
    df = df.with_columns([
        pl.col("close")
        .rolling_mean(window_size=fast_period)
        .over("symbol")
        .alias("fast_ma"),
        pl.col("close")
        .rolling_mean(window_size=slow_period)
        .over("symbol")
        .alias("slow_ma"),
    ])

    # Generate signals: 1 when fast > slow, -1 when fast < slow
    df = df.with_columns([
        pl.when(pl.col("fast_ma") > pl.col("slow_ma"))
        .then(1)
        .when(pl.col("fast_ma") < pl.col("slow_ma"))
        .then(-1)
        .otherwise(0)
        .alias("signal"),
    ])

    return df.select(["symbol", "timestamp", "signal"])


def generate_rsi_signals(
    prices: pl.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pl.DataFrame:
    """
    Generate RSI-based signals.

    Buy when RSI crosses above oversold, sell when crosses below overbought.

    Args:
        prices: DataFrame with columns (symbol, timestamp, close)
        period: RSI period
        oversold: Oversold threshold (buy signal)
        overbought: Overbought threshold (sell signal)

    Returns:
        signals DataFrame with columns (symbol, timestamp, signal)
    """
    df = prices.sort(["symbol", "timestamp"])

    # Calculate price changes
    df = df.with_columns([
        pl.col("close").diff().over("symbol").alias("change"),
    ])

    # Separate gains and losses
    df = df.with_columns([
        pl.when(pl.col("change") > 0).then(pl.col("change")).otherwise(0).alias("gain"),
        pl.when(pl.col("change") < 0)
        .then(pl.col("change").abs())
        .otherwise(0)
        .alias("loss"),
    ])

    # Calculate average gain/loss (using SMA for simplicity)
    df = df.with_columns([
        pl.col("gain").rolling_mean(window_size=period).over("symbol").alias("avg_gain"),
        pl.col("loss").rolling_mean(window_size=period).over("symbol").alias("avg_loss"),
    ])

    # Calculate RSI
    df = df.with_columns([
        (100 - 100 / (1 + pl.col("avg_gain") / pl.col("avg_loss").clip(lower_bound=1e-10)))
        .alias("rsi"),
    ])

    # Generate signals
    df = df.with_columns([
        pl.when(pl.col("rsi") < oversold)
        .then(1)
        .when(pl.col("rsi") > overbought)
        .then(-1)
        .otherwise(0)
        .alias("signal"),
    ])

    return df.select(["symbol", "timestamp", "signal"])

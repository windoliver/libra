"""
Backtesting bridge between LIBRA and Freqtrade.

Provides methods to run Freqtrade's native backtesting engine
and convert results to LIBRA's BacktestResult format.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


class FreqtradeBacktestBridge:
    """
    Bridge for running Freqtrade backtests within LIBRA.

    This class provides two modes of operation:
    1. Native Mode: Uses Freqtrade's Backtesting class directly
    2. LIBRA Mode: Uses LIBRA's backtest engine with FT strategy signals

    Native mode provides more accurate results matching Freqtrade's behavior,
    while LIBRA mode integrates better with LIBRA's event-driven architecture.

    Examples:
        bridge = FreqtradeBacktestBridge()

        # Run native Freqtrade backtest
        result = await bridge.run_native_backtest(
            strategy_name="SampleStrategy",
            config_path=Path("config.json"),
            timerange="20230101-20231231",
        )

        # Run using LIBRA's engine
        result = await bridge.run_libra_backtest(
            strategy=strategy_instance,
            data=historical_data,
            initial_capital=Decimal("10000"),
        )
    """

    def __init__(self) -> None:
        """Initialize the backtest bridge."""
        self._freqtrade_available = self._check_freqtrade()

    def _check_freqtrade(self) -> bool:
        """Check if Freqtrade is installed."""
        try:
            from freqtrade.optimize.backtesting import Backtesting  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def freqtrade_available(self) -> bool:
        """Whether Freqtrade backtesting is available."""
        return self._freqtrade_available

    async def run_native_backtest(
        self,
        config: dict[str, Any],
        timerange: str | None = None,
        max_open_trades: int | None = None,
        stake_amount: str | Decimal | None = None,
    ) -> BacktestResultData:
        """
        Run Freqtrade's native backtesting engine.

        Args:
            config: Freqtrade configuration dictionary.
            timerange: Time range string (e.g., "20230101-20231231").
            max_open_trades: Override max open trades.
            stake_amount: Override stake amount.

        Returns:
            BacktestResultData with comprehensive metrics.

        Raises:
            ImportError: If Freqtrade is not installed.
        """
        if not self._freqtrade_available:
            raise ImportError(
                "Freqtrade is not installed. Install with: pip install libra[freqtrade]"
            )

        from freqtrade.optimize.backtesting import Backtesting

        # Apply overrides
        if timerange:
            config["timerange"] = timerange
        if max_open_trades is not None:
            config["max_open_trades"] = max_open_trades
        if stake_amount is not None:
            config["stake_amount"] = str(stake_amount)

        # Run backtest
        backtesting = Backtesting(config)
        try:
            backtesting.start()
            results = backtesting.results
        finally:
            backtesting.cleanup()

        return self._convert_native_results(results, config)

    def run_on_dataframe(
        self,
        strategy: Any,
        data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000"),
        commission: Decimal = Decimal("0.001"),
        slippage: Decimal = Decimal("0.0005"),
    ) -> BacktestResultData:
        """
        Run a simplified backtest on a DataFrame.

        This method runs the strategy's signal generation on historical data
        and simulates trades without using Freqtrade's full backtesting engine.

        Useful when Freqtrade is not installed or for quick strategy validation.

        Args:
            strategy: Freqtrade IStrategy instance.
            data: OHLCV DataFrame with columns: open, high, low, close, volume.
            initial_capital: Starting capital.
            commission: Trading commission as decimal.
            slippage: Slippage as decimal.

        Returns:
            BacktestResultData with calculated metrics.
        """
        import pandas as pd

        # Validate data
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(data.columns)):
            missing = required_cols - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Run strategy to get signals
        metadata = {"pair": "BACKTEST/USDT"}
        df = data.copy()

        # Add indicators
        if hasattr(strategy, "advise_indicators"):
            df = strategy.advise_indicators(df, metadata)
        elif hasattr(strategy, "populate_indicators"):
            df = strategy.populate_indicators(df, metadata)

        # Add entry signals
        if hasattr(strategy, "advise_entry"):
            df = strategy.advise_entry(df, metadata)
        elif hasattr(strategy, "populate_entry_trend"):
            df = strategy.populate_entry_trend(df, metadata)

        # Add exit signals
        if hasattr(strategy, "advise_exit"):
            df = strategy.advise_exit(df, metadata)
        elif hasattr(strategy, "populate_exit_trend"):
            df = strategy.populate_exit_trend(df, metadata)

        # Simulate trades
        trades = self._simulate_trades(df, initial_capital, commission, slippage)

        # Calculate metrics
        return self._calculate_metrics(
            trades=trades,
            equity_curve=self._build_equity_curve(trades, df, initial_capital),
            initial_capital=initial_capital,
            data=df,
        )

    def _simulate_trades(
        self,
        df: pd.DataFrame,
        initial_capital: Decimal,
        commission: Decimal,
        slippage: Decimal,
    ) -> list[dict[str, Any]]:
        """Simulate trades from signals in DataFrame."""
        trades: list[dict[str, Any]] = []
        position: dict[str, Any] | None = None
        capital = initial_capital

        for i, (idx, row) in enumerate(df.iterrows()):
            # Skip if no signals columns
            if "enter_long" not in df.columns:
                continue

            # Check for entry
            if position is None:
                if row.get("enter_long", 0) == 1:
                    entry_price = Decimal(str(row["close"])) * (1 + slippage)
                    size = capital * Decimal("0.95") / entry_price  # 95% of capital
                    cost = size * entry_price * (1 + commission)

                    position = {
                        "entry_idx": i,
                        "entry_time": idx,
                        "entry_price": entry_price,
                        "size": size,
                        "cost": cost,
                        "direction": "long",
                        "entry_tag": row.get("enter_tag", ""),
                    }

            # Check for exit
            elif position is not None:
                should_exit = False
                exit_reason = ""

                if row.get("exit_long", 0) == 1:
                    should_exit = True
                    exit_reason = row.get("exit_tag", "exit_signal")

                # Check stoploss
                if hasattr(df, "stoploss") and position["direction"] == "long":
                    stoploss_price = position["entry_price"] * (
                        1 + Decimal(str(df.stoploss))
                    )
                    if Decimal(str(row["low"])) <= stoploss_price:
                        should_exit = True
                        exit_reason = "stoploss"

                if should_exit:
                    exit_price = Decimal(str(row["close"])) * (1 - slippage)
                    proceeds = position["size"] * exit_price * (1 - commission)
                    pnl = proceeds - position["cost"]
                    pnl_pct = pnl / position["cost"]

                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "entry_time": position["entry_time"],
                        "exit_time": idx,
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "size": position["size"],
                        "direction": position["direction"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "entry_tag": position.get("entry_tag", ""),
                        "exit_reason": exit_reason,
                    })

                    capital = capital + pnl
                    position = None

        return trades

    def _build_equity_curve(
        self,
        trades: list[dict[str, Any]],
        df: pd.DataFrame,
        initial_capital: Decimal,
    ) -> list[Decimal]:
        """Build equity curve from trades."""
        equity = [initial_capital]
        current_equity = initial_capital

        trade_exits = {t["exit_idx"]: t["pnl"] for t in trades}

        for i in range(len(df)):
            if i in trade_exits:
                current_equity = current_equity + trade_exits[i]
            equity.append(current_equity)

        return equity

    def _calculate_metrics(
        self,
        trades: list[dict[str, Any]],
        equity_curve: list[Decimal],
        initial_capital: Decimal,
        data: pd.DataFrame,
    ) -> BacktestResultData:
        """Calculate backtest metrics from trades."""
        import pandas as pd

        if not trades:
            # Return empty result
            return BacktestResultData(
                total_return=Decimal("0"),
                annualized_return=Decimal("0"),
                max_drawdown=Decimal("0"),
                max_drawdown_duration=timedelta(0),
                sharpe_ratio=Decimal("0"),
                sortino_ratio=Decimal("0"),
                calmar_ratio=Decimal("0"),
                total_trades=0,
                win_rate=Decimal("0"),
                profit_factor=Decimal("0"),
                avg_trade_return=Decimal("0"),
                initial_capital=initial_capital,
                final_capital=initial_capital,
                equity_curve=equity_curve,
            )

        final_capital = equity_curve[-1]
        total_return = (final_capital - initial_capital) / initial_capital

        # Calculate metrics
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        win_rate = (
            Decimal(str(len(winning_trades))) / Decimal(str(len(trades)))
            if trades
            else Decimal("0")
        )

        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))

        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else Decimal("999")
        )

        avg_trade_return = (
            sum(t["pnl_pct"] for t in trades) / len(trades) if trades else Decimal("0")
        )

        # Calculate drawdown
        peak = initial_capital
        max_dd = Decimal("0")
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else Decimal("0")
            if dd > max_dd:
                max_dd = dd

        # Calculate returns for Sharpe/Sortino
        equity_series = pd.Series([float(e) for e in equity_curve])
        returns = equity_series.pct_change().dropna()

        if len(returns) > 1 and returns.std() > 0:
            sharpe = Decimal(str(returns.mean() / returns.std() * (252**0.5)))
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino = Decimal(
                    str(returns.mean() / downside_returns.std() * (252**0.5))
                )
            else:
                sortino = sharpe
        else:
            sharpe = Decimal("0")
            sortino = Decimal("0")

        calmar = (
            total_return / max_dd if max_dd > 0 else Decimal("0")
        )

        # Estimate annualized return
        if len(data) > 0:
            days = len(data)
            annualized = (1 + float(total_return)) ** (365 / days) - 1
        else:
            annualized = 0

        return BacktestResultData(
            total_return=total_return,
            annualized_return=Decimal(str(annualized)),
            max_drawdown=max_dd,
            max_drawdown_duration=timedelta(days=0),  # Simplified
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            initial_capital=initial_capital,
            final_capital=final_capital,
            equity_curve=equity_curve,
            trades=trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
        )

    def _convert_native_results(
        self,
        results: dict[str, Any],
        config: dict[str, Any],
    ) -> BacktestResultData:
        """Convert Freqtrade native results to LIBRA format."""
        strategy_name = config.get("strategy", "unknown")
        stats = results.get(strategy_name, {})

        # Extract metrics from Freqtrade results
        total_return = Decimal(str(stats.get("profit_total", 0)))
        max_drawdown = Decimal(str(abs(stats.get("max_drawdown", 0))))

        return BacktestResultData(
            total_return=total_return,
            annualized_return=Decimal(str(stats.get("profit_total_abs", 0))),
            max_drawdown=max_drawdown,
            max_drawdown_duration=timedelta(
                days=stats.get("max_drawdown_duration", 0)
            ),
            sharpe_ratio=Decimal(str(stats.get("sharpe_ratio", 0) or 0)),
            sortino_ratio=Decimal(str(stats.get("sortino_ratio", 0) or 0)),
            calmar_ratio=Decimal(str(stats.get("calmar_ratio", 0) or 0)),
            total_trades=stats.get("total_trades", 0),
            win_rate=Decimal(str(stats.get("wins", 0) / stats.get("total_trades", 1))),
            profit_factor=Decimal(str(stats.get("profit_factor", 0) or 0)),
            avg_trade_return=Decimal(str(stats.get("avg_profit", 0) or 0)),
            initial_capital=Decimal(str(config.get("stake_amount", 10000))),
            final_capital=Decimal(str(stats.get("final_balance", 0))),
            equity_curve=[],
            trades=[],
        )


class BacktestResultData:
    """
    Container for backtest results.

    This class holds the results of a backtest run, providing access
    to all standard performance metrics.

    Can be converted to LIBRA's BacktestResult dataclass via to_libra_result().
    """

    def __init__(
        self,
        total_return: Decimal,
        annualized_return: Decimal,
        max_drawdown: Decimal,
        max_drawdown_duration: timedelta,
        sharpe_ratio: Decimal,
        sortino_ratio: Decimal,
        calmar_ratio: Decimal,
        total_trades: int,
        win_rate: Decimal,
        profit_factor: Decimal,
        avg_trade_return: Decimal,
        initial_capital: Decimal,
        final_capital: Decimal,
        equity_curve: list[Decimal] | None = None,
        trades: list[dict[str, Any]] | None = None,
        winning_trades: int = 0,
        losing_trades: int = 0,
    ) -> None:
        """Initialize backtest results."""
        self.total_return = total_return
        self.annualized_return = annualized_return
        self.max_drawdown = max_drawdown
        self.max_drawdown_duration = max_drawdown_duration
        self.sharpe_ratio = sharpe_ratio
        self.sortino_ratio = sortino_ratio
        self.calmar_ratio = calmar_ratio
        self.total_trades = total_trades
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.avg_trade_return = avg_trade_return
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.equity_curve = equity_curve or []
        self.trades = trades or []
        self.winning_trades = winning_trades
        self.losing_trades = losing_trades

    def to_libra_result(self) -> Any:
        """
        Convert to LIBRA's BacktestResult dataclass.

        Returns:
            BacktestResult instance.
        """
        from libra.strategies.protocol import BacktestResult

        # Need start/end dates - use placeholders if not available
        now = datetime.now()

        return BacktestResult(
            total_return=self.total_return,
            annualized_return=self.annualized_return,
            max_drawdown=self.max_drawdown,
            max_drawdown_duration=self.max_drawdown_duration,
            sharpe_ratio=self.sharpe_ratio,
            sortino_ratio=self.sortino_ratio,
            calmar_ratio=self.calmar_ratio,
            total_trades=self.total_trades,
            win_rate=self.win_rate,
            profit_factor=self.profit_factor,
            avg_trade_return=self.avg_trade_return,
            start_date=now - timedelta(days=365),
            end_date=now,
            initial_capital=self.initial_capital,
            final_capital=self.final_capital,
            equity_curve=self.equity_curve,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
        )

    def summary(self) -> dict[str, str]:
        """Return formatted summary dict."""
        return {
            "total_return": f"{self.total_return * 100:.2f}%",
            "annualized_return": f"{self.annualized_return * 100:.2f}%",
            "max_drawdown": f"{self.max_drawdown * 100:.2f}%",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "total_trades": str(self.total_trades),
            "win_rate": f"{self.win_rate * 100:.1f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "net_profit": f"{self.final_capital - self.initial_capital:.2f}",
        }

"""
BacktestResult: Immutable container for backtest results.

Uses msgspec.Struct for performance:
- Fast serialization to JSON/MessagePack
- Minimal memory footprint
- Immutable (frozen=True)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import msgspec


class BacktestSummary(msgspec.Struct, frozen=True, gc=False):
    """
    Summary statistics from a backtest run.

    All monetary values are in the quote currency (e.g., USDT for BTC/USDT).
    All percentages are expressed as decimals (0.15 = 15%).
    """

    # Identification
    strategy_name: str
    symbol: str
    timeframe: str

    # Time range
    start_time_ns: int
    end_time_ns: int
    duration_days: float

    # Capital
    initial_capital: Decimal
    final_equity: Decimal

    # Returns
    total_return: Decimal  # Absolute return in quote currency
    total_return_pct: float  # Return as percentage (0.15 = 15%)
    cagr: float  # Compound Annual Growth Rate

    # Risk metrics
    sharpe_ratio: float  # Annualized Sharpe (risk-free rate = 0)
    sortino_ratio: float  # Downside deviation version
    calmar_ratio: float  # CAGR / Max Drawdown

    # Drawdown
    max_drawdown: Decimal  # Maximum drawdown amount
    max_drawdown_pct: float  # Maximum drawdown percentage
    max_drawdown_duration_days: float  # Longest drawdown recovery time

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # winning_trades / total_trades

    # Profit metrics
    profit_factor: float  # gross_profit / gross_loss
    avg_win: Decimal  # Average winning trade
    avg_loss: Decimal  # Average losing trade
    avg_trade: Decimal  # Average trade P&L
    largest_win: Decimal
    largest_loss: Decimal

    # Risk/reward
    avg_win_loss_ratio: float  # avg_win / abs(avg_loss)
    expectancy: Decimal  # Expected value per trade

    # Volume
    total_volume: Decimal  # Total traded volume
    total_fees: Decimal  # Total fees paid

    # Bars processed
    bars_processed: int

    @property
    def duration_str(self) -> str:
        """Human-readable duration."""
        days = self.duration_days
        if days >= 365:
            return f"{days / 365:.1f} years"
        elif days >= 30:
            return f"{days / 30:.1f} months"
        else:
            return f"{days:.1f} days"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for display/storage."""
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "duration": self.duration_str,
            "initial_capital": str(self.initial_capital),
            "final_equity": str(self.final_equity),
            "total_return": str(self.total_return),
            "total_return_pct": f"{self.total_return_pct * 100:.2f}%",
            "cagr": f"{self.cagr * 100:.2f}%",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "calmar_ratio": f"{self.calmar_ratio:.2f}",
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": f"{self.max_drawdown_pct * 100:.2f}%",
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate * 100:.1f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "avg_trade": str(self.avg_trade),
            "expectancy": str(self.expectancy),
            "total_fees": str(self.total_fees),
        }


class EquityPoint(msgspec.Struct, frozen=True, gc=False):
    """Single point on the equity curve."""

    timestamp_ns: int
    equity: Decimal
    cash: Decimal
    position_value: Decimal
    drawdown: Decimal
    drawdown_pct: float


class BacktestResult(msgspec.Struct, frozen=True, gc=False):
    """
    Complete backtest result container.

    Contains:
    - Summary statistics
    - Equity curve (list of equity points)
    - Trade log (list of trade records)
    - Parameters used for the run

    This is the return value of BacktestEngine.run().
    """

    # Summary metrics
    summary: BacktestSummary

    # Equity curve (list of EquityPoint)
    equity_curve: list[EquityPoint]

    # Trade log (list of TradeRecord) - forward reference to avoid circular import
    trades: list[Any]  # Actually list[TradeRecord]

    # Daily returns for analysis
    daily_returns: list[float]

    # Configuration used
    config: dict[str, Any]

    # Metadata
    run_timestamp_ns: int
    engine_version: str = "1.0.0"

    def to_dataframe(self) -> Any:
        """
        Convert equity curve to Polars DataFrame.

        Returns:
            Polars DataFrame with columns:
            - timestamp_ns: int (nanoseconds since epoch)
            - equity: float
            - cash: float
            - position_value: float
            - drawdown: float
            - drawdown_pct: float
        """
        import polars as pl

        return pl.DataFrame({
            "timestamp_ns": [e.timestamp_ns for e in self.equity_curve],
            "equity": [float(e.equity) for e in self.equity_curve],
            "cash": [float(e.cash) for e in self.equity_curve],
            "position_value": [float(e.position_value) for e in self.equity_curve],
            "drawdown": [float(e.drawdown) for e in self.equity_curve],
            "drawdown_pct": [e.drawdown_pct for e in self.equity_curve],
        })

    def trades_to_dataframe(self) -> Any:
        """
        Convert trade log to Polars DataFrame.

        Returns:
            Polars DataFrame with trade records
        """
        import polars as pl

        if not self.trades:
            return pl.DataFrame()

        return pl.DataFrame({
            "entry_time_ns": [t.entry_time_ns for t in self.trades],
            "exit_time_ns": [t.exit_time_ns for t in self.trades],
            "symbol": [t.symbol for t in self.trades],
            "side": [t.side for t in self.trades],
            "entry_price": [float(t.entry_price) for t in self.trades],
            "exit_price": [float(t.exit_price) if t.exit_price else None for t in self.trades],
            "quantity": [float(t.quantity) for t in self.trades],
            "pnl": [float(t.pnl) if t.pnl else None for t in self.trades],
            "pnl_pct": [t.pnl_pct for t in self.trades],
            "fees": [float(t.fees) for t in self.trades],
            "duration_bars": [t.duration_bars for t in self.trades],
        })

    def save(self, path: str) -> None:
        """
        Save result to JSON file.

        Args:
            path: File path to save to
        """
        import json
        from pathlib import Path

        data = {
            "summary": self.summary.to_dict(),
            "equity_curve": [
                {
                    "timestamp_ns": e.timestamp_ns,
                    "equity": str(e.equity),
                    "cash": str(e.cash),
                    "position_value": str(e.position_value),
                    "drawdown": str(e.drawdown),
                    "drawdown_pct": e.drawdown_pct,
                }
                for e in self.equity_curve
            ],
            "trades": [
                {
                    "entry_time_ns": t.entry_time_ns,
                    "exit_time_ns": t.exit_time_ns,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": str(t.entry_price),
                    "exit_price": str(t.exit_price) if t.exit_price else None,
                    "quantity": str(t.quantity),
                    "pnl": str(t.pnl) if t.pnl else None,
                    "pnl_pct": t.pnl_pct,
                    "fees": str(t.fees),
                }
                for t in self.trades
            ],
            "daily_returns": self.daily_returns,
            "config": self.config,
            "run_timestamp_ns": self.run_timestamp_ns,
            "engine_version": self.engine_version,
        }

        Path(path).write_text(json.dumps(data, indent=2))

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        s = self.summary
        print("\n" + "=" * 60)
        print(f"Backtest Results: {s.strategy_name}")
        print("=" * 60)
        print(f"Symbol: {s.symbol} | Timeframe: {s.timeframe}")
        print(f"Duration: {s.duration_str}")
        print("-" * 60)
        print("PERFORMANCE")
        print(f"  Initial Capital: ${s.initial_capital:,.2f}")
        print(f"  Final Equity:    ${s.final_equity:,.2f}")
        print(f"  Total Return:    ${s.total_return:,.2f} ({s.total_return_pct * 100:.2f}%)")
        print(f"  CAGR:            {s.cagr * 100:.2f}%")
        print("-" * 60)
        print("RISK METRICS")
        print(f"  Sharpe Ratio:    {s.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:   {s.sortino_ratio:.2f}")
        print(f"  Calmar Ratio:    {s.calmar_ratio:.2f}")
        print(f"  Max Drawdown:    ${s.max_drawdown:,.2f} ({s.max_drawdown_pct * 100:.2f}%)")
        print("-" * 60)
        print("TRADE STATISTICS")
        print(f"  Total Trades:    {s.total_trades}")
        print(f"  Win Rate:        {s.win_rate * 100:.1f}%")
        print(f"  Profit Factor:   {s.profit_factor:.2f}")
        print(f"  Avg Trade:       ${s.avg_trade:,.2f}")
        print(f"  Expectancy:      ${s.expectancy:,.2f}")
        print(f"  Total Fees:      ${s.total_fees:,.2f}")
        print("=" * 60 + "\n")

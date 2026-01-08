"""
MetricsCollector: Real-time performance metrics calculation.

Collects and calculates:
- Equity curve with drawdown tracking
- Trade log with P&L
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Win/loss statistics

Optimized for streaming calculation during backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

import msgspec


if TYPE_CHECKING:
    pass


# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252
SECONDS_PER_DAY = 86400
NS_PER_DAY = SECONDS_PER_DAY * 1_000_000_000


class TradeRecord(msgspec.Struct, gc=False):
    """
    Record of a completed trade (round-trip).

    A trade consists of entry and exit. For simplicity, this
    tracks one entry and one exit (no partial fills in basic version).
    """

    # Identification
    trade_id: str
    symbol: str
    side: str  # "long" or "short"

    # Entry
    entry_time_ns: int
    entry_price: Decimal
    quantity: Decimal

    # Exit (None if position still open)
    exit_time_ns: int | None = None
    exit_price: Decimal | None = None

    # P&L (calculated on exit)
    pnl: Decimal | None = None  # Absolute P&L
    pnl_pct: float | None = None  # Return percentage

    # Costs
    fees: Decimal = Decimal("0")

    # Metadata
    duration_bars: int = 0
    entry_reason: str = ""
    exit_reason: str = ""

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time_ns is not None

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl is not None and self.pnl > 0

    @property
    def is_loser(self) -> bool:
        """Check if trade was unprofitable."""
        return self.pnl is not None and self.pnl < 0


@dataclass
class DrawdownState:
    """Tracks current drawdown state."""

    peak_equity: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")
    current_drawdown_pct: float = 0.0
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0
    drawdown_start_ns: int | None = None
    max_drawdown_duration_ns: int = 0
    current_drawdown_start_ns: int | None = None


@dataclass
class MetricsCollector:
    """
    Real-time performance metrics collector.

    Collects metrics during backtest execution:
    - Equity curve snapshots
    - Trade records
    - Daily returns for Sharpe/Sortino calculation

    Usage:
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # On each bar
        collector.record_equity(timestamp_ns, equity, cash, position_value)

        # On trade close
        collector.record_trade(trade_record)

        # At end of backtest
        summary = collector.calculate_summary(
            strategy_name="MyStrategy",
            symbol="BTC/USDT",
            timeframe="1h",
        )
    """

    initial_capital: Decimal
    risk_free_rate: float = 0.0  # Annualized risk-free rate

    # Equity tracking
    equity_snapshots: list[tuple[int, Decimal, Decimal, Decimal]] = field(
        default_factory=list
    )  # (timestamp_ns, equity, cash, position_value)

    # Trade tracking
    trades: list[TradeRecord] = field(default_factory=list)
    open_trades: dict[str, TradeRecord] = field(default_factory=dict)  # symbol -> trade

    # Drawdown tracking
    drawdown: DrawdownState = field(default_factory=DrawdownState)

    # Daily returns for Sharpe calculation
    daily_returns: list[float] = field(default_factory=list)
    _last_day_equity: Decimal | None = field(default=None, repr=False)
    _last_day_timestamp: int = field(default=0, repr=False)

    # Running totals
    total_fees: Decimal = Decimal("0")
    total_volume: Decimal = Decimal("0")
    bars_processed: int = 0

    def __post_init__(self) -> None:
        """Initialize peak equity."""
        self.drawdown.peak_equity = self.initial_capital

    def record_equity(
        self,
        timestamp_ns: int,
        equity: Decimal,
        cash: Decimal,
        position_value: Decimal,
    ) -> None:
        """
        Record an equity snapshot.

        Call this on each bar or at regular intervals.

        Args:
            timestamp_ns: Current timestamp in nanoseconds
            equity: Total equity (cash + position value)
            cash: Available cash
            position_value: Value of open positions
        """
        self.equity_snapshots.append((timestamp_ns, equity, cash, position_value))
        self.bars_processed += 1

        # Update drawdown tracking
        self._update_drawdown(timestamp_ns, equity)

        # Track daily returns for Sharpe calculation
        self._update_daily_returns(timestamp_ns, equity)

    def _update_drawdown(self, timestamp_ns: int, equity: Decimal) -> None:
        """Update drawdown metrics."""
        dd = self.drawdown

        if equity > dd.peak_equity:
            # New peak - reset drawdown
            dd.peak_equity = equity
            if dd.current_drawdown_start_ns is not None:
                # Record duration of previous drawdown
                duration = timestamp_ns - dd.current_drawdown_start_ns
                if duration > dd.max_drawdown_duration_ns:
                    dd.max_drawdown_duration_ns = duration
            dd.current_drawdown = Decimal("0")
            dd.current_drawdown_pct = 0.0
            dd.current_drawdown_start_ns = None
        else:
            # In drawdown
            dd.current_drawdown = dd.peak_equity - equity
            dd.current_drawdown_pct = (
                float(dd.current_drawdown / dd.peak_equity)
                if dd.peak_equity > 0
                else 0.0
            )

            if dd.current_drawdown_start_ns is None:
                dd.current_drawdown_start_ns = timestamp_ns

            # Track max drawdown
            if dd.current_drawdown > dd.max_drawdown:
                dd.max_drawdown = dd.current_drawdown
                dd.max_drawdown_pct = dd.current_drawdown_pct

    def _update_daily_returns(self, timestamp_ns: int, equity: Decimal) -> None:
        """Track daily returns for Sharpe/Sortino calculation."""
        current_day = timestamp_ns // NS_PER_DAY

        if self._last_day_equity is None:
            # First equity point
            self._last_day_equity = equity
            self._last_day_timestamp = current_day
            return

        if current_day > self._last_day_timestamp:
            # New day - calculate return
            if self._last_day_equity > 0:
                daily_return = float(
                    (equity - self._last_day_equity) / self._last_day_equity
                )
                self.daily_returns.append(daily_return)

            self._last_day_equity = equity
            self._last_day_timestamp = current_day

    def open_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_time_ns: int,
        entry_price: Decimal,
        quantity: Decimal,
        fees: Decimal = Decimal("0"),
        entry_reason: str = "",
    ) -> TradeRecord:
        """
        Record opening a new trade.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: "long" or "short"
            entry_time_ns: Entry timestamp
            entry_price: Entry price
            quantity: Position size
            fees: Entry fees
            entry_reason: Why trade was entered

        Returns:
            TradeRecord for the new trade
        """
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_time_ns=entry_time_ns,
            entry_price=entry_price,
            quantity=quantity,
            fees=fees,
            entry_reason=entry_reason,
        )
        self.open_trades[symbol] = trade
        self.total_fees += fees
        self.total_volume += quantity * entry_price
        return trade

    def close_trade(
        self,
        symbol: str,
        exit_time_ns: int,
        exit_price: Decimal,
        fees: Decimal = Decimal("0"),
        exit_reason: str = "",
        duration_bars: int = 0,
    ) -> TradeRecord | None:
        """
        Close an open trade.

        Args:
            symbol: Trading symbol
            exit_time_ns: Exit timestamp
            exit_price: Exit price
            fees: Exit fees
            exit_reason: Why trade was closed
            duration_bars: Number of bars trade was open

        Returns:
            Closed TradeRecord or None if no open trade
        """
        trade = self.open_trades.pop(symbol, None)
        if trade is None:
            return None

        # Calculate P&L
        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity

        # Subtract total fees
        total_fees = trade.fees + fees
        pnl -= total_fees

        # Calculate percentage return
        cost_basis = trade.entry_price * trade.quantity
        pnl_pct = float(pnl / cost_basis) if cost_basis > 0 else 0.0

        # Update trade record (create new since frozen)
        closed_trade = TradeRecord(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            side=trade.side,
            entry_time_ns=trade.entry_time_ns,
            entry_price=trade.entry_price,
            quantity=trade.quantity,
            exit_time_ns=exit_time_ns,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=total_fees,
            duration_bars=duration_bars,
            entry_reason=trade.entry_reason,
            exit_reason=exit_reason,
        )

        self.trades.append(closed_trade)
        self.total_fees += fees
        self.total_volume += trade.quantity * exit_price

        return closed_trade

    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record a completed trade directly.

        Use this if you're tracking trades externally.

        Args:
            trade: Completed TradeRecord
        """
        self.trades.append(trade)
        self.total_fees += trade.fees
        self.total_volume += trade.quantity * trade.entry_price
        if trade.exit_price:
            self.total_volume += trade.quantity * trade.exit_price

    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (mean_return - risk_free_rate) / std_dev * sqrt(252)

        Returns:
            Annualized Sharpe ratio, or 0.0 if insufficient data
        """
        if len(self.daily_returns) < 2:
            return 0.0

        mean_return = sum(self.daily_returns) / len(self.daily_returns)
        daily_rf = self.risk_free_rate / TRADING_DAYS_PER_YEAR

        excess_returns = [r - daily_rf for r in self.daily_returns]
        variance = sum((r - mean_return) ** 2 for r in excess_returns) / (
            len(excess_returns) - 1
        )
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return 0.0

        return (mean_return - daily_rf) / std_dev * math.sqrt(TRADING_DAYS_PER_YEAR)

    def calculate_sortino_ratio(self) -> float:
        """
        Calculate annualized Sortino ratio.

        Sortino = (mean_return - risk_free_rate) / downside_std * sqrt(252)

        Only uses negative returns for volatility calculation.

        Returns:
            Annualized Sortino ratio, or 0.0 if insufficient data
        """
        if len(self.daily_returns) < 2:
            return 0.0

        mean_return = sum(self.daily_returns) / len(self.daily_returns)
        daily_rf = self.risk_free_rate / TRADING_DAYS_PER_YEAR

        # Only count downside deviation
        downside_returns = [min(0, r - daily_rf) for r in self.daily_returns]
        downside_variance = sum(r**2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0

        if downside_std == 0:
            return 0.0

        return (mean_return - daily_rf) / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    def calculate_summary(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
    ) -> "BacktestSummary":
        """
        Calculate complete summary statistics.

        Call this at the end of the backtest to get final metrics.

        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            timeframe: Bar timeframe

        Returns:
            BacktestSummary with all metrics
        """
        from libra.backtest.result import BacktestSummary

        # Get time range from equity snapshots
        if not self.equity_snapshots:
            raise ValueError("No equity snapshots recorded")

        start_time_ns = self.equity_snapshots[0][0]
        end_time_ns = self.equity_snapshots[-1][0]
        final_equity = self.equity_snapshots[-1][1]

        duration_ns = end_time_ns - start_time_ns
        duration_days = duration_ns / NS_PER_DAY

        # Returns
        total_return = final_equity - self.initial_capital
        total_return_pct = (
            float(total_return / self.initial_capital)
            if self.initial_capital > 0
            else 0.0
        )

        # CAGR
        years = duration_days / 365
        if years > 0 and self.initial_capital > 0:
            cagr = (float(final_equity / self.initial_capital) ** (1 / years)) - 1
        else:
            cagr = 0.0

        # Risk metrics
        sharpe = self.calculate_sharpe_ratio()
        sortino = self.calculate_sortino_ratio()

        # Calmar = CAGR / Max Drawdown
        calmar = (
            cagr / self.drawdown.max_drawdown_pct
            if self.drawdown.max_drawdown_pct > 0
            else 0.0
        )

        # Trade statistics
        closed_trades = [t for t in self.trades if t.is_closed]
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.is_winner)
        losing_trades = sum(1 for t in closed_trades if t.is_loser)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit metrics
        gross_profit = sum(t.pnl for t in closed_trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl and t.pnl < 0))

        profit_factor = (
            float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        )

        # Average trade metrics
        if winning_trades > 0:
            avg_win = gross_profit / Decimal(winning_trades)
        else:
            avg_win = Decimal("0")

        if losing_trades > 0:
            avg_loss = gross_loss / Decimal(losing_trades)
        else:
            avg_loss = Decimal("0")

        if total_trades > 0:
            avg_trade = total_return / Decimal(total_trades)
        else:
            avg_trade = Decimal("0")

        # Largest win/loss
        if closed_trades:
            pnls = [t.pnl for t in closed_trades if t.pnl is not None]
            largest_win = max(pnls) if pnls else Decimal("0")
            largest_loss = min(pnls) if pnls else Decimal("0")
        else:
            largest_win = Decimal("0")
            largest_loss = Decimal("0")

        # Win/loss ratio
        avg_win_loss_ratio = (
            float(avg_win / avg_loss) if avg_loss > 0 else float("inf")
        )

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        loss_rate = 1 - win_rate
        expectancy = Decimal(str(win_rate)) * avg_win - Decimal(str(loss_rate)) * avg_loss

        # Drawdown duration in days
        max_dd_duration_days = self.drawdown.max_drawdown_duration_ns / NS_PER_DAY

        return BacktestSummary(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            duration_days=duration_days,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=self.drawdown.max_drawdown,
            max_drawdown_pct=self.drawdown.max_drawdown_pct,
            max_drawdown_duration_days=max_dd_duration_days,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_win_loss_ratio=avg_win_loss_ratio,
            expectancy=expectancy,
            total_volume=self.total_volume,
            total_fees=self.total_fees,
            bars_processed=self.bars_processed,
        )

    def get_equity_curve(self) -> list:
        """
        Get equity curve as list of EquityPoint.

        Returns:
            List of EquityPoint structs
        """
        from libra.backtest.result import EquityPoint

        curve = []
        for ts, equity, cash, pos_value in self.equity_snapshots:
            # Calculate drawdown at this point
            dd = self.initial_capital  # Simple version - would need to track properly
            peak = max(e[1] for e in self.equity_snapshots[: len(curve) + 1])
            drawdown = peak - equity
            drawdown_pct = float(drawdown / peak) if peak > 0 else 0.0

            curve.append(
                EquityPoint(
                    timestamp_ns=ts,
                    equity=equity,
                    cash=cash,
                    position_value=pos_value,
                    drawdown=drawdown,
                    drawdown_pct=drawdown_pct,
                )
            )
        return curve

    def reset(self) -> None:
        """Reset collector for a new backtest run."""
        self.equity_snapshots.clear()
        self.trades.clear()
        self.open_trades.clear()
        self.daily_returns.clear()
        self.drawdown = DrawdownState()
        self.drawdown.peak_equity = self.initial_capital
        self._last_day_equity = None
        self._last_day_timestamp = 0
        self.total_fees = Decimal("0")
        self.total_volume = Decimal("0")
        self.bars_processed = 0

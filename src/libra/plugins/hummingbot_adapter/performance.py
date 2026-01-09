"""
Performance Tracking Module for Hummingbot Adapter (Issue #12).

Provides comprehensive metrics tracking for market making strategies:
- P&L tracking (realized, unrealized, total)
- Position tracking with entry/exit prices
- Trade statistics (win rate, profit factor, Sharpe ratio)
- Risk metrics (max drawdown, VaR, expected shortfall)
- Order fill analytics
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Protocol


class TradeEventProtocol(Protocol):
    """Protocol for trade events from various sources."""

    symbol: str
    side: str
    quantity: float
    price: float
    timestamp_ns: int


class PositionSide(str, Enum):
    """Position side."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """
    Open position tracking.

    Tracks entry price, size, and unrealized P&L.
    """

    symbol: str
    side: PositionSide
    size: Decimal
    entry_price: Decimal
    entry_time_ns: int
    realized_pnl: Decimal = Decimal("0")
    commission_paid: Decimal = Decimal("0")

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.size > 0

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L at current price."""
        if self.size == 0:
            return Decimal("0")

        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - current_price) * self.size

    def total_pnl(self, current_price: Decimal) -> Decimal:
        """Total P&L including realized and unrealized."""
        return self.realized_pnl + self.unrealized_pnl(current_price)


@dataclass
class Trade:
    """Completed trade record."""

    symbol: str
    side: str  # "buy" or "sell"
    size: Decimal
    entry_price: Decimal
    exit_price: Decimal
    entry_time_ns: int
    exit_time_ns: int
    realized_pnl: Decimal
    commission: Decimal
    holding_time_ns: int = 0

    def __post_init__(self) -> None:
        """Calculate holding time."""
        if self.holding_time_ns == 0:
            self.holding_time_ns = self.exit_time_ns - self.entry_time_ns

    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.entry_price == 0:
            return 0.0
        return float((self.exit_price - self.entry_price) / self.entry_price * 100)

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.realized_pnl > 0


@dataclass
class OrderFill:
    """Order fill record for analytics."""

    order_id: str
    symbol: str
    side: str
    price: Decimal
    size: Decimal
    timestamp_ns: int
    latency_ns: int = 0  # Time from order submission to fill


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""

    timestamp_ns: int
    equity: Decimal
    cash: Decimal
    position_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_pnl: Decimal
    drawdown_pct: float = 0.0


@dataclass
class PerformanceStats:
    """Comprehensive performance statistics."""

    # P&L
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    commission_total: Decimal = Decimal("0")

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    avg_trade: Decimal = Decimal("0")
    avg_holding_time_ms: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_value: Decimal = Decimal("0")
    current_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    var_95: Decimal = Decimal("0")  # Value at Risk (95%)
    expected_shortfall: Decimal = Decimal("0")

    # Order analytics
    total_orders: int = 0
    filled_orders: int = 0
    fill_rate: float = 0.0
    avg_fill_latency_ms: float = 0.0
    avg_slippage_bps: float = 0.0  # Basis points


class PerformanceTracker:
    """
    Comprehensive performance tracking.

    Tracks all trading activity and calculates:
    - P&L (realized, unrealized, total)
    - Trade statistics
    - Risk metrics (drawdown, Sharpe, Sortino)
    - Order fill analytics
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        snapshot_interval_ns: int = 60_000_000_000,  # 1 minute snapshots
    ) -> None:
        """
        Initialize performance tracker.

        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            snapshot_interval_ns: Interval between equity snapshots
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.snapshot_interval_ns = snapshot_interval_ns

        # Current state
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._current_prices: dict[str, Decimal] = {}

        # Historical data
        self._trades: list[Trade] = []
        self._order_fills: list[OrderFill] = []
        self._snapshots: deque[PerformanceSnapshot] = deque(maxlen=10000)

        # Running statistics
        self._peak_equity = initial_capital
        self._last_snapshot_time = 0
        self._returns: deque[float] = deque(maxlen=1000)  # For Sharpe calculation

        # Order tracking
        self._pending_orders: dict[str, tuple[int, Decimal]] = {}  # order_id -> (submit_time, expected_price)

    @property
    def equity(self) -> Decimal:
        """Current total equity."""
        position_value = sum(
            pos.size * self._current_prices.get(pos.symbol, pos.entry_price)
            for pos in self._positions.values()
            if pos.is_open
        )
        return self._cash + position_value

    @property
    def unrealized_pnl(self) -> Decimal:
        """Total unrealized P&L across all positions."""
        return sum(
            (
                pos.unrealized_pnl(self._current_prices.get(pos.symbol, pos.entry_price))
                for pos in self._positions.values()
                if pos.is_open
            ),
            Decimal("0"),
        )

    @property
    def realized_pnl(self) -> Decimal:
        """Total realized P&L from closed trades."""
        return sum((trade.realized_pnl for trade in self._trades), Decimal("0"))

    def update_price(self, symbol: str, price: Decimal) -> None:
        """
        Update market price for a symbol.

        Args:
            symbol: Trading symbol
            price: Current market price
        """
        self._current_prices[symbol] = price
        self._maybe_take_snapshot()

    def on_trade(self, event: TradeEventProtocol) -> None:
        """
        Process a trade event.

        Updates positions and records trade.

        Args:
            event: Trade event from message bus
        """
        symbol = event.symbol
        is_buy = event.side.lower() == "buy"
        size = Decimal(str(event.quantity))
        price = Decimal(str(event.price))
        timestamp = event.timestamp_ns

        # Get or create position
        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol,
                side=PositionSide.FLAT,
                size=Decimal("0"),
                entry_price=Decimal("0"),
                entry_time_ns=timestamp,
            )

        position = self._positions[symbol]

        if is_buy:
            self._process_buy(position, size, price, timestamp)
        else:
            self._process_sell(position, size, price, timestamp)

        # Update cash
        trade_value = size * price
        if is_buy:
            self._cash -= trade_value
        else:
            self._cash += trade_value

        # Update price and snapshot
        self._current_prices[symbol] = price
        self._maybe_take_snapshot()

    def _process_buy(
        self,
        position: Position,
        size: Decimal,
        price: Decimal,
        timestamp: int,
    ) -> None:
        """Process buy trade."""
        if position.side == PositionSide.FLAT:
            # Open new long position
            position.side = PositionSide.LONG
            position.size = size
            position.entry_price = price
            position.entry_time_ns = timestamp

        elif position.side == PositionSide.LONG:
            # Add to long position (average up/down)
            total_value = position.size * position.entry_price + size * price
            position.size += size
            position.entry_price = total_value / position.size

        elif position.side == PositionSide.SHORT:
            # Close or reduce short position
            if size >= position.size:
                # Full close (and maybe flip)
                pnl = (position.entry_price - price) * position.size
                self._record_trade(position, price, timestamp, pnl)

                remaining = size - position.size
                if remaining > 0:
                    # Flip to long
                    position.side = PositionSide.LONG
                    position.size = remaining
                    position.entry_price = price
                    position.entry_time_ns = timestamp
                else:
                    position.side = PositionSide.FLAT
                    position.size = Decimal("0")
            else:
                # Partial close
                pnl = (position.entry_price - price) * size
                position.realized_pnl += pnl
                position.size -= size

    def _process_sell(
        self,
        position: Position,
        size: Decimal,
        price: Decimal,
        timestamp: int,
    ) -> None:
        """Process sell trade."""
        if position.side == PositionSide.FLAT:
            # Open new short position
            position.side = PositionSide.SHORT
            position.size = size
            position.entry_price = price
            position.entry_time_ns = timestamp

        elif position.side == PositionSide.SHORT:
            # Add to short position
            total_value = position.size * position.entry_price + size * price
            position.size += size
            position.entry_price = total_value / position.size

        elif position.side == PositionSide.LONG:
            # Close or reduce long position
            if size >= position.size:
                # Full close (and maybe flip)
                pnl = (price - position.entry_price) * position.size
                self._record_trade(position, price, timestamp, pnl)

                remaining = size - position.size
                if remaining > 0:
                    # Flip to short
                    position.side = PositionSide.SHORT
                    position.size = remaining
                    position.entry_price = price
                    position.entry_time_ns = timestamp
                else:
                    position.side = PositionSide.FLAT
                    position.size = Decimal("0")
            else:
                # Partial close
                pnl = (price - position.entry_price) * size
                position.realized_pnl += pnl
                position.size -= size

    def _record_trade(
        self,
        position: Position,
        exit_price: Decimal,
        exit_time: int,
        pnl: Decimal,
    ) -> None:
        """Record a completed trade."""
        trade = Trade(
            symbol=position.symbol,
            side="long" if position.side == PositionSide.LONG else "short",
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time_ns=position.entry_time_ns,
            exit_time_ns=exit_time,
            realized_pnl=pnl,
            commission=position.commission_paid,
        )
        self._trades.append(trade)
        position.realized_pnl = Decimal("0")
        position.commission_paid = Decimal("0")

    def record_order_submit(
        self,
        order_id: str,
        expected_price: Decimal,
    ) -> None:
        """Record order submission for latency tracking."""
        self._pending_orders[order_id] = (time.time_ns(), expected_price)

    def record_order_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        fill_price: Decimal,
        fill_size: Decimal,
    ) -> None:
        """Record order fill for analytics."""
        timestamp = time.time_ns()

        # Calculate latency if we tracked the order
        latency_ns = 0
        if order_id in self._pending_orders:
            submit_time, _ = self._pending_orders.pop(order_id)  # expected_price for future slippage calc
            latency_ns = timestamp - submit_time

        fill = OrderFill(
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=fill_price,
            size=fill_size,
            timestamp_ns=timestamp,
            latency_ns=latency_ns,
        )
        self._order_fills.append(fill)

    def _maybe_take_snapshot(self) -> None:
        """Take equity snapshot if interval has passed."""
        now = time.time_ns()
        if now - self._last_snapshot_time < self.snapshot_interval_ns:
            return

        equity = self.equity
        unrealized = self.unrealized_pnl
        realized = self.realized_pnl
        total_pnl = equity - self.initial_capital

        # Update peak and calculate drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown_pct = 0.0
        if self._peak_equity > 0:
            drawdown_pct = float(
                (self._peak_equity - equity) / self._peak_equity * 100
            )

        snapshot = PerformanceSnapshot(
            timestamp_ns=now,
            equity=equity,
            cash=self._cash,
            position_value=equity - self._cash,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_pnl=total_pnl,
            drawdown_pct=drawdown_pct,
        )
        self._snapshots.append(snapshot)
        self._last_snapshot_time = now

        # Record return for Sharpe calculation
        if len(self._snapshots) >= 2:
            prev_equity = self._snapshots[-2].equity
            if prev_equity > 0:
                ret = float((equity - prev_equity) / prev_equity)
                self._returns.append(ret)

    def get_stats(self) -> PerformanceStats:
        """
        Calculate comprehensive performance statistics.

        Returns:
            PerformanceStats with all metrics
        """
        stats = PerformanceStats()

        # P&L
        stats.realized_pnl = self.realized_pnl
        stats.unrealized_pnl = self.unrealized_pnl
        stats.total_pnl = self.equity - self.initial_capital

        # Trade statistics
        if self._trades:
            stats.total_trades = len(self._trades)

            winning_trades = [t for t in self._trades if t.is_winner]
            losing_trades = [t for t in self._trades if not t.is_winner]

            stats.winning_trades = len(winning_trades)
            stats.losing_trades = len(losing_trades)

            if stats.total_trades > 0:
                stats.win_rate = stats.winning_trades / stats.total_trades

            # Profit/Loss metrics
            stats.gross_profit = sum(
                (t.realized_pnl for t in winning_trades), Decimal("0")
            )
            stats.gross_loss = abs(
                sum((t.realized_pnl for t in losing_trades), Decimal("0"))
            )
            stats.commission_total = sum(
                (t.commission for t in self._trades), Decimal("0")
            )

            if stats.gross_loss > 0:
                stats.profit_factor = float(stats.gross_profit / stats.gross_loss)

            if winning_trades:
                stats.avg_win = stats.gross_profit / Decimal(len(winning_trades))
                stats.largest_win = max(t.realized_pnl for t in winning_trades)

            if losing_trades:
                stats.avg_loss = stats.gross_loss / Decimal(len(losing_trades))
                stats.largest_loss = min(t.realized_pnl for t in losing_trades)

            stats.avg_trade = sum(
                (t.realized_pnl for t in self._trades), Decimal("0")
            ) / Decimal(stats.total_trades)
            stats.avg_holding_time_ms = (
                sum(t.holding_time_ns for t in self._trades)
                / stats.total_trades
                / 1_000_000
            )

        # Risk metrics
        if self._snapshots:
            max_dd = max(s.drawdown_pct for s in self._snapshots)
            stats.max_drawdown_pct = max_dd
            stats.max_drawdown_value = Decimal(str(max_dd / 100)) * self._peak_equity
            stats.current_drawdown_pct = self._snapshots[-1].drawdown_pct

        # Sharpe ratio (annualized)
        if len(self._returns) >= 2:
            returns_list = list(self._returns)
            avg_return = sum(returns_list) / len(returns_list)
            variance = sum((r - avg_return) ** 2 for r in returns_list) / len(
                returns_list
            )
            std_return = math.sqrt(variance) if variance > 0 else 0.0

            # Annualize (assuming ~252 trading periods per year)
            periods_per_year = 252 * (
                24 * 60 * 60 * 1_000_000_000 / self.snapshot_interval_ns
            )
            annualized_return = avg_return * periods_per_year

            if std_return > 0:
                annualized_std = std_return * math.sqrt(periods_per_year)
                stats.sharpe_ratio = (
                    annualized_return - self.risk_free_rate
                ) / annualized_std

            # Sortino ratio (only downside deviation)
            downside_returns = [r for r in returns_list if r < 0]
            if downside_returns:
                downside_variance = sum(r**2 for r in downside_returns) / len(
                    returns_list
                )
                downside_std = math.sqrt(downside_variance)
                if downside_std > 0:
                    annualized_downside = downside_std * math.sqrt(periods_per_year)
                    stats.sortino_ratio = (
                        annualized_return - self.risk_free_rate
                    ) / annualized_downside

            # Value at Risk (95%)
            sorted_returns = sorted(returns_list)
            var_index = int(len(sorted_returns) * 0.05)
            stats.var_95 = Decimal(str(sorted_returns[var_index])) * self.equity

            # Expected Shortfall (CVaR)
            tail_returns = sorted_returns[: var_index + 1]
            if tail_returns:
                avg_tail = sum(tail_returns) / len(tail_returns)
                stats.expected_shortfall = Decimal(str(avg_tail)) * self.equity

        # Calmar ratio
        if stats.max_drawdown_pct > 0 and self._snapshots:
            # Annualized return / max drawdown
            total_return = float(
                (self.equity - self.initial_capital) / self.initial_capital
            )
            time_span_ns = (
                self._snapshots[-1].timestamp_ns - self._snapshots[0].timestamp_ns
            )
            if time_span_ns > 0:
                years = time_span_ns / (365.25 * 24 * 60 * 60 * 1_000_000_000)
                # Require at least 1 day of data for meaningful Calmar ratio
                if years >= 1 / 365.25:
                    try:
                        annualized_return = (1 + total_return) ** (1 / years) - 1
                        stats.calmar_ratio = annualized_return / (
                            stats.max_drawdown_pct / 100
                        )
                    except (OverflowError, ValueError):
                        # Skip if calculation overflows
                        pass

        # Order analytics
        if self._order_fills:
            stats.total_orders = len(self._pending_orders) + len(self._order_fills)
            stats.filled_orders = len(self._order_fills)
            stats.fill_rate = (
                stats.filled_orders / max(1, stats.total_orders)
            )
            stats.avg_fill_latency_ms = (
                sum(f.latency_ns for f in self._order_fills)
                / len(self._order_fills)
                / 1_000_000
            )

        return stats

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> list[Position]:
        """Get all open positions."""
        return [p for p in self._positions.values() if p.is_open]

    def get_trades(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[Trade]:
        """Get recent trades, optionally filtered by symbol."""
        trades = self._trades
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        return trades[-limit:]

    def get_equity_curve(self) -> list[tuple[int, float]]:
        """Get equity curve as (timestamp_ns, equity) pairs."""
        return [(s.timestamp_ns, float(s.equity)) for s in self._snapshots]

    def get_drawdown_curve(self) -> list[tuple[int, float]]:
        """Get drawdown curve as (timestamp_ns, drawdown_pct) pairs."""
        return [(s.timestamp_ns, s.drawdown_pct) for s in self._snapshots]

    def reset(self) -> None:
        """Reset all tracking data."""
        self._cash = self.initial_capital
        self._positions.clear()
        self._current_prices.clear()
        self._trades.clear()
        self._order_fills.clear()
        self._snapshots.clear()
        self._returns.clear()
        self._peak_equity = self.initial_capital
        self._last_snapshot_time = 0
        self._pending_orders.clear()

    def to_dict(self) -> dict:
        """Export current state as dictionary."""
        stats = self.get_stats()
        return {
            "equity": float(self.equity),
            "cash": float(self._cash),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "total_pnl": float(stats.total_pnl),
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "size": float(p.size),
                    "entry_price": float(p.entry_price),
                    "unrealized_pnl": float(
                        p.unrealized_pnl(
                            self._current_prices.get(p.symbol, p.entry_price)
                        )
                    ),
                }
                for p in self._positions.values()
                if p.is_open
            ],
            "stats": {
                "total_trades": stats.total_trades,
                "win_rate": stats.win_rate,
                "profit_factor": stats.profit_factor,
                "sharpe_ratio": stats.sharpe_ratio,
                "max_drawdown_pct": stats.max_drawdown_pct,
                "avg_holding_time_ms": stats.avg_holding_time_ms,
            },
        }

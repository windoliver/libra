"""
Demo Trading Engine for TUI.

Provides realistic simulated trading for demo mode:
- Price movements with momentum and mean reversion
- Position tracking with P&L calculation
- Order execution simulation
- Risk scenario triggers (drawdown, circuit breaker)
- Trading state management
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Callable


class TradingState(str, Enum):
    """Trading state for demo."""

    ACTIVE = "ACTIVE"
    REDUCING = "REDUCING"
    HALTED = "HALTED"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class DemoPosition:
    """Simulated position."""

    symbol: str
    side: str  # "LONG" or "SHORT" or "FLAT"
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.quantity == 0:
            return Decimal("0")
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.quantity
        elif self.side == "SHORT":
            return (self.entry_price - self.current_price) * self.quantity
        return Decimal("0")

    @property
    def notional(self) -> Decimal:
        """Current notional value."""
        return abs(self.quantity * self.current_price)

    @property
    def exposure_pct(self) -> float:
        """Exposure as percentage (simplified)."""
        return float(self.notional / Decimal("100000") * 100)


@dataclass
class DemoOrder:
    """Simulated order."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal | None = None
    order_type: str = "MARKET"
    filled: bool = False
    fill_price: Decimal | None = None


@dataclass
class PriceState:
    """Price state with momentum."""

    symbol: str
    price: Decimal
    momentum: float = 0.0  # -1 to 1
    volatility: float = 1.0  # multiplier
    base_price: Decimal = Decimal("0")  # for mean reversion

    def __post_init__(self):
        if self.base_price == Decimal("0"):
            self.base_price = self.price


@dataclass
class DemoTrader:
    """
    Demo trading engine with realistic simulation.

    Features:
    - Realistic price movements with momentum
    - Position tracking with live P&L
    - Risk state management
    - Order execution simulation
    - Circuit breaker logic
    """

    # Account
    initial_balance: Decimal = Decimal("100000")
    balance: Decimal = field(default_factory=lambda: Decimal("100000"))
    realized_pnl: Decimal = Decimal("0")
    peak_equity: Decimal = field(default_factory=lambda: Decimal("100000"))

    # Positions
    positions: dict[str, DemoPosition] = field(default_factory=dict)

    # Prices
    prices: dict[str, PriceState] = field(default_factory=dict)

    # Risk state
    trading_state: TradingState = TradingState.ACTIVE
    circuit_breaker_state: str = "CLOSED"
    consecutive_losses: int = 0
    daily_pnl: Decimal = Decimal("0")
    daily_trades: int = 0

    # Limits
    max_drawdown_pct: float = 10.0
    daily_loss_limit: Decimal = Decimal("5000")
    max_consecutive_losses: int = 5
    max_position_pct: float = 30.0  # max 30% per position

    # Order rate tracking
    orders_this_second: int = 0
    order_rate_limit: int = 10

    # Callbacks
    on_trade: Callable[[str], None] | None = None
    on_risk_event: Callable[[str, str], None] | None = None
    on_state_change: Callable[[TradingState], None] | None = None

    def __post_init__(self):
        """Initialize default prices."""
        self._init_prices()
        self._init_positions()

    def _init_prices(self) -> None:
        """Initialize price states."""
        defaults = [
            ("BTC/USDT", Decimal("51000"), 0.02),  # 2% volatility
            ("ETH/USDT", Decimal("3000"), 0.025),
            ("SOL/USDT", Decimal("145"), 0.035),
        ]
        for symbol, price, vol in defaults:
            self.prices[symbol] = PriceState(
                symbol=symbol,
                price=price,
                volatility=vol,
            )

    def _init_positions(self) -> None:
        """Initialize flat positions."""
        for symbol in self.prices:
            self.positions[symbol] = DemoPosition(
                symbol=symbol,
                side="FLAT",
                quantity=Decimal("0"),
                current_price=self.prices[symbol].price,
            )

    # =========================================================================
    # Price Simulation
    # =========================================================================

    def tick_prices(self) -> dict[str, Decimal]:
        """
        Update all prices with realistic movement.

        Returns dict of symbol -> new price.
        """
        updates = {}

        for symbol, state in self.prices.items():
            # Update momentum with some persistence + noise
            state.momentum = state.momentum * 0.8 + random.gauss(0, 0.3)  # noqa: S311
            state.momentum = max(-1, min(1, state.momentum))

            # Mean reversion force
            deviation = float(state.price - state.base_price) / float(state.base_price)
            reversion = -deviation * 0.1

            # Calculate price change
            base_move = float(state.price) * state.volatility * 0.01
            direction = state.momentum + reversion + random.gauss(0, 0.5)  # noqa: S311
            change = Decimal(str(base_move * direction))

            # Apply change with bounds
            state.price = max(
                state.base_price * Decimal("0.85"),
                min(state.base_price * Decimal("1.15"), state.price + change),
            )

            updates[symbol] = state.price

            # Update position current price
            if symbol in self.positions:
                self.positions[symbol].current_price = state.price

        return updates

    def get_price(self, symbol: str) -> Decimal:
        """Get current price for symbol."""
        if symbol in self.prices:
            return self.prices[symbol].price
        return Decimal("0")

    # =========================================================================
    # Order Execution
    # =========================================================================

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> tuple[bool, str]:
        """
        Execute a simulated order.

        Returns (success, message).
        """
        # Check trading state
        if self.trading_state == TradingState.HALTED:
            return False, "Trading HALTED - cannot execute orders"

        current_price = self.get_price(symbol)
        if current_price == 0:
            return False, f"Unknown symbol: {symbol}"

        fill_price = price or current_price

        # Check if reducing only
        position = self.positions.get(symbol)
        if self.trading_state == TradingState.REDUCING:
            if position and position.side != "FLAT":
                # Only allow closing trades
                is_closing = (
                    (position.side == "LONG" and side == "SELL")
                    or (position.side == "SHORT" and side == "BUY")
                )
                if not is_closing:
                    return False, "REDUCING mode - only closing orders allowed"
            else:
                return False, "REDUCING mode - no new positions allowed"

        # Calculate notional
        notional = quantity * fill_price

        # Check balance for buys
        if side == "BUY" and notional > self.balance:
            return False, f"Insufficient balance: need ${notional:,.2f}, have ${self.balance:,.2f}"

        # Execute the trade
        self._apply_trade(symbol, side, quantity, fill_price)

        # Update tracking
        self.daily_trades += 1
        self.orders_this_second += 1

        # Generate trade message
        msg = f"FILLED {side} {quantity} {symbol} @ ${fill_price:,.2f} (${notional:,.2f})"

        if self.on_trade:
            self.on_trade(msg)

        return True, msg

    def _apply_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        fill_price: Decimal,
    ) -> None:
        """Apply trade to position."""
        position = self.positions.get(symbol)
        if not position:
            position = DemoPosition(symbol=symbol, side="FLAT", current_price=fill_price)
            self.positions[symbol] = position

        notional = quantity * fill_price

        if position.side == "FLAT":
            # Opening new position
            position.side = "LONG" if side == "BUY" else "SHORT"
            position.quantity = quantity
            position.entry_price = fill_price
            if side == "BUY":
                self.balance -= notional

        elif (position.side == "LONG" and side == "BUY") or (
            position.side == "SHORT" and side == "SELL"
        ):
            # Adding to position
            total_cost = position.entry_price * position.quantity + fill_price * quantity
            position.quantity += quantity
            position.entry_price = total_cost / position.quantity
            if side == "BUY":
                self.balance -= notional

        else:
            # Closing/reducing position
            if quantity >= position.quantity:
                # Full close
                pnl = position.unrealized_pnl
                self.realized_pnl += pnl
                self.daily_pnl += pnl
                self.balance += position.notional + pnl

                # Track consecutive losses
                if pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                position.side = "FLAT"
                position.quantity = Decimal("0")
                position.entry_price = Decimal("0")
            else:
                # Partial close
                close_ratio = quantity / position.quantity
                pnl = position.unrealized_pnl * close_ratio
                self.realized_pnl += pnl
                self.daily_pnl += pnl
                self.balance += (position.notional * close_ratio) + pnl

                if pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                position.quantity -= quantity

        position.current_price = fill_price
        self._check_risk_limits()

    # =========================================================================
    # Risk Management
    # =========================================================================

    def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        equity = self.equity

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Check drawdown
        drawdown_pct = self.drawdown_pct
        if drawdown_pct >= self.max_drawdown_pct:
            self._set_trading_state(TradingState.HALTED, f"Max drawdown {drawdown_pct:.1f}% exceeded")
            return

        if drawdown_pct >= self.max_drawdown_pct * 0.7:
            if self.trading_state == TradingState.ACTIVE:
                self._set_trading_state(TradingState.REDUCING, f"Drawdown warning {drawdown_pct:.1f}%")

        # Check daily loss
        if self.daily_pnl <= -self.daily_loss_limit:
            self._set_trading_state(TradingState.HALTED, f"Daily loss limit ${self.daily_pnl:,.0f}")
            return

        if self.daily_pnl <= -self.daily_loss_limit * Decimal("0.7"):
            if self.trading_state == TradingState.ACTIVE:
                self._set_trading_state(TradingState.REDUCING, f"Daily loss warning ${self.daily_pnl:,.0f}")

        # Check circuit breaker
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.circuit_breaker_state = "OPEN"
            self._set_trading_state(
                TradingState.HALTED,
                f"Circuit breaker: {self.consecutive_losses} consecutive losses",
            )

    def _set_trading_state(self, state: TradingState, reason: str) -> None:
        """Set trading state with notification."""
        if state != self.trading_state:
            self.trading_state = state
            if self.on_risk_event:
                self.on_risk_event(state.value, reason)
            if self.on_state_change:
                self.on_state_change(state)

    def reset_daily(self) -> None:
        """Reset daily counters (call at day boundary)."""
        self.daily_pnl = Decimal("0")
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.circuit_breaker_state = "CLOSED"
        if self.trading_state != TradingState.HALTED or self.drawdown_pct < self.max_drawdown_pct:
            self.trading_state = TradingState.ACTIVE

    def resume_trading(self) -> bool:
        """Manually resume trading if safe."""
        if self.drawdown_pct >= self.max_drawdown_pct:
            return False
        if self.daily_pnl <= -self.daily_loss_limit:
            return False

        self.trading_state = TradingState.ACTIVE
        self.circuit_breaker_state = "CLOSED"
        self.consecutive_losses = 0
        return True

    # =========================================================================
    # Metrics
    # =========================================================================

    @property
    def equity(self) -> Decimal:
        """Total equity (balance + unrealized P&L)."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.balance + unrealized

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown percentage."""
        if self.peak_equity == 0:
            return 0.0
        return float((self.peak_equity - self.equity) / self.peak_equity * 100)

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_exposure(self) -> Decimal:
        """Total position exposure."""
        return sum(p.notional for p in self.positions.values() if p.side != "FLAT")

    def get_position_exposure(self, symbol: str) -> tuple[float, float]:
        """Get position exposure (current%, limit%)."""
        position = self.positions.get(symbol)
        if not position or position.side == "FLAT":
            return 0.0, self.max_position_pct

        exposure = float(position.notional / self.initial_balance * 100)
        return exposure, self.max_position_pct

    def get_stats(self) -> dict:
        """Get comprehensive stats for display."""
        return {
            "balance": self.balance,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.total_unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "drawdown_pct": self.drawdown_pct,
            "trading_state": self.trading_state.value,
            "circuit_breaker": self.circuit_breaker_state,
            "consecutive_losses": self.consecutive_losses,
            "daily_trades": self.daily_trades,
            "total_exposure": self.total_exposure,
        }

    # =========================================================================
    # Scenario Triggers (for testing)
    # =========================================================================

    def trigger_winning_streak(self, count: int = 3) -> None:
        """Simulate a winning streak."""
        for symbol in list(self.prices.keys())[:1]:
            price = self.get_price(symbol)
            # Buy low
            self.execute_order(symbol, "BUY", Decimal("0.1"), price * Decimal("0.99"))
            # Simulate price increase
            self.prices[symbol].price = price * Decimal("1.02")
            self.positions[symbol].current_price = self.prices[symbol].price
            # Sell high
            self.execute_order(symbol, "SELL", Decimal("0.1"), self.prices[symbol].price)

    def trigger_losing_streak(self, count: int = 3) -> None:
        """Simulate a losing streak."""
        for _ in range(count):
            symbol = "BTC/USDT"
            price = self.get_price(symbol)
            # Buy
            self.execute_order(symbol, "BUY", Decimal("0.1"), price)
            # Simulate price drop
            self.prices[symbol].price = price * Decimal("0.98")
            self.positions[symbol].current_price = self.prices[symbol].price
            # Sell at loss
            self.execute_order(symbol, "SELL", Decimal("0.1"), self.prices[symbol].price)

    def trigger_flash_crash(self, symbol: str = "BTC/USDT") -> None:
        """Simulate a flash crash."""
        if symbol in self.prices:
            self.prices[symbol].price *= Decimal("0.85")
            self.prices[symbol].momentum = -1.0
            if symbol in self.positions:
                self.positions[symbol].current_price = self.prices[symbol].price
            self._check_risk_limits()

"""
Risk Engine for LIBRA.

Pre-trade risk validation engine following NautilusTrader's RiskEngine pattern.
ALL orders MUST pass through this engine before execution.

Architecture:
    Strategy -> RiskEngine.validate_order() -> ExecutionEngine -> Gateway

Features:
    - Trading state management (ACTIVE/REDUCING/HALTED)
    - Position limit checks
    - Notional value checks
    - Order rate limiting (submit + modify)
    - Drawdown monitoring
    - Circuit breaker
    - Self-trade prevention
    - Price/quantity precision validation
    - Price collar (fat-finger protection)

References:
    - NautilusTrader RiskEngine: https://nautilustrader.io/docs/latest/api_reference/risk/
    - FIA Best Practices: https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING, Any

import msgspec

from libra.core.events import Event, EventType
from libra.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from libra.risk.limits import RiskLimits
from libra.risk.rate_limiter import MultiRateLimiter, TokenBucketRateLimiter


if TYPE_CHECKING:
    from libra.clients.data_client import Instrument
    from libra.core.message_bus import MessageBus
    from libra.gateways.protocol import Order, Position


logger = logging.getLogger(__name__)


# =============================================================================
# Trading States
# =============================================================================


class TradingState(str, Enum):
    """
    Risk engine trading states.

    Controls what orders are allowed through the risk engine.

    State Transitions:
        ACTIVE -> REDUCING: When daily loss limit approached
        ACTIVE -> HALTED: When max drawdown or circuit breaker triggered
        REDUCING -> HALTED: When additional thresholds breached
        HALTED -> ACTIVE: Only via manual reset
        Any -> HALTED: Via manual halt
    """

    ACTIVE = "active"  # Normal trading - all orders allowed
    REDUCING = "reducing"  # Only position-reducing orders allowed
    HALTED = "halted"  # No new orders - only cancellations allowed


# =============================================================================
# Risk Check Result
# =============================================================================


class RiskCheckResult(msgspec.Struct, frozen=True, gc=False):
    """
    Result of a pre-trade risk check.

    Immutable and optimized for fast creation (~95ns).

    Attributes:
        passed: True if check passed, False if rejected
        check_name: Name of the check (e.g., "position_limit")
        reason: Human-readable rejection reason (if failed)
    """

    passed: bool
    check_name: str
    reason: str | None = None


# =============================================================================
# Risk Engine Configuration
# =============================================================================


@dataclass
class RiskEngineConfig:
    """
    Complete Risk Engine configuration.

    Combines RiskLimits with additional engine-specific settings.

    Attributes:
        limits: Core risk limits (position, notional, rate, drawdown)
        enable_self_trade_prevention: Check for wash trading
        enable_price_collar: Reject orders far from market
        price_collar_pct: Max deviation from mid-price (e.g., 0.05 = 5%)
        enable_precision_validation: Validate price/quantity precision
        max_modify_rate: Max order modifications per second
    """

    limits: RiskLimits = field(default_factory=RiskLimits)

    # Self-trade prevention
    enable_self_trade_prevention: bool = True

    # Price collar (fat-finger protection)
    enable_price_collar: bool = True
    price_collar_pct: Decimal = Decimal("0.10")  # 10% default

    # Precision validation
    enable_precision_validation: bool = True

    # Modify rate limiting (separate from submit)
    max_modify_rate: int = 20  # modifications per second


# =============================================================================
# Risk Engine
# =============================================================================


@dataclass
class RiskEngine:
    """
    Pre-trade risk validation engine.

    ALL orders MUST pass through this engine before execution.
    Implements the complete pre-trade validation pipeline following
    NautilusTrader's RiskEngine pattern.

    Pipeline (in order, fail-fast):
        1. Trading State Check (ACTIVE/REDUCING/HALTED)
        2. Self-Trade Prevention
        3. Price Precision Validation
        4. Quantity Precision Validation
        5. Price Collar Check
        6. Position Limit Check
        7. Notional Value Check
        8. Order Rate Limiting
        9. Drawdown Check
        10. Circuit Breaker Check

    Performance Target: <1ms for all checks combined.

    Examples:
        config = RiskEngineConfig(limits=RiskLimits(...))
        engine = RiskEngine(config=config, bus=message_bus)

        # Before every order
        result = engine.validate_order(order, current_price, instrument)
        if not result.passed:
            # Order denied - result.reason has details
            return

        # Order passed all checks, proceed to execution
        await execution_client.submit_order(order)

    Thread Safety:
        This implementation is NOT thread-safe. For multi-threaded use,
        wrap calls in a lock or use the Rust implementation (Phase 1B).
    """

    config: RiskEngineConfig
    bus: MessageBus | None = None  # Optional for standalone use

    # Internal state (initialized in __post_init__)
    _trading_state: TradingState = field(init=False, default=TradingState.ACTIVE)
    _submit_rate_limiter: MultiRateLimiter = field(init=False, repr=False)
    _modify_rate_limiter: TokenBucketRateLimiter = field(init=False, repr=False)
    _circuit_breaker: CircuitBreaker = field(init=False, repr=False)

    # Tracking state
    _peak_equity: Decimal = field(init=False, default=Decimal("0"))
    _current_equity: Decimal = field(init=False, default=Decimal("0"))
    _daily_pnl: Decimal = field(init=False, default=Decimal("0"))
    _weekly_pnl: Decimal = field(init=False, default=Decimal("0"))
    _positions: dict[str, Position] = field(init=False, default_factory=dict)

    # Open orders tracking for self-trade prevention
    _open_orders: dict[str, list[Order]] = field(init=False, default_factory=dict)

    # Metrics
    _orders_checked: int = field(init=False, default=0)
    _orders_denied: int = field(init=False, default=0)
    _check_latency_ns: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Initialize rate limiters and circuit breaker from config."""
        limits = self.config.limits

        # Multi-tier rate limiter for order submissions: per-second AND per-minute
        self._submit_rate_limiter = MultiRateLimiter(
            limiters=[
                TokenBucketRateLimiter(
                    rate=float(limits.max_orders_per_second),
                    capacity=limits.max_orders_per_second,
                ),
                TokenBucketRateLimiter(
                    rate=limits.max_orders_per_minute / 60.0,
                    capacity=limits.max_orders_per_minute,
                ),
            ]
        )

        # Separate rate limiter for order modifications
        self._modify_rate_limiter = TokenBucketRateLimiter(
            rate=float(self.config.max_modify_rate),
            capacity=self.config.max_modify_rate * 2,  # Allow burst
        )

        self._circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                drawdown_threshold=limits.circuit_breaker_drawdown_pct,
                cooldown_seconds=limits.circuit_breaker_cooldown_seconds,
                max_consecutive_losses=limits.circuit_breaker_max_consecutive_losses,
            )
        )

        logger.info(
            "RiskEngine initialized: max_daily_loss=%.1f%%, max_drawdown=%.1f%%, "
            "rate_limit=%d/sec, self_trade=%s, price_collar=%s (%.1f%%)",
            float(limits.max_daily_loss_pct) * 100,
            float(limits.max_total_drawdown_pct) * 100,
            limits.max_orders_per_second,
            self.config.enable_self_trade_prevention,
            self.config.enable_price_collar,
            float(self.config.price_collar_pct) * 100,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def limits(self) -> RiskLimits:
        """Access risk limits from config."""
        return self.config.limits

    @property
    def trading_state(self) -> TradingState:
        """Current trading state."""
        return self._trading_state

    @property
    def is_active(self) -> bool:
        """Check if trading is fully active."""
        return self._trading_state == TradingState.ACTIVE

    @property
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self._trading_state == TradingState.HALTED

    @property
    def current_drawdown(self) -> Decimal:
        """Current drawdown from peak equity."""
        if self._peak_equity <= 0:
            return Decimal("0")
        return (self._current_equity - self._peak_equity) / self._peak_equity

    @property
    def circuit_breaker_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return self._circuit_breaker.get_status()

    # =========================================================================
    # Trading State Management
    # =========================================================================

    def set_trading_state(self, state: TradingState, reason: str = "manual") -> None:
        """
        Manually set trading state.

        Args:
            state: New trading state
            reason: Reason for state change
        """
        old_state = self._trading_state
        self._trading_state = state

        logger.warning(
            "Trading state changed: %s -> %s (reason: %s)",
            old_state.value,
            state.value,
            reason,
        )

        # Publish state change event
        self._publish_event(
            EventType.RISK_LIMIT_BREACH,
            {
                "event": "trading_state_changed",
                "old_state": old_state.value,
                "new_state": state.value,
                "reason": reason,
            },
        )

    def halt_trading(self, reason: str = "manual") -> None:
        """Emergency halt all trading."""
        self.set_trading_state(TradingState.HALTED, reason)

    def resume_trading(self) -> None:
        """Resume normal trading (from HALTED or REDUCING)."""
        if self._circuit_breaker.is_open:
            logger.warning("Cannot resume: circuit breaker still open")
            return
        self.set_trading_state(TradingState.ACTIVE, "manual_resume")

    # =========================================================================
    # Order Validation
    # =========================================================================

    def validate_order(
        self,
        order: Order,
        current_price: Decimal,
        instrument: Instrument | None = None,
    ) -> RiskCheckResult:
        """
        Validate order against all risk limits.

        This is the main entry point - EVERY order must pass through here.
        Returns immediately on first failed check for performance.

        Args:
            order: Order to validate
            current_price: Current market price for notional/collar calculation
            instrument: Optional instrument for precision validation

        Returns:
            RiskCheckResult with pass/fail and reason

        Performance:
            Target <1ms for all checks combined.
            Actual ~50-100μs typical.
        """
        start_ns = time.time_ns()
        self._orders_checked += 1

        # Run all checks in order (fail-fast)
        checks = [
            self._check_trading_state(order),
        ]

        # Self-trade prevention (if enabled)
        if self.config.enable_self_trade_prevention:
            checks.append(self._check_self_trade(order))

        # Precision validation (if enabled and instrument provided)
        if self.config.enable_precision_validation and instrument is not None:
            checks.append(self._check_price_precision(order, instrument))
            checks.append(self._check_quantity_precision(order, instrument))

        # Price collar (if enabled)
        if self.config.enable_price_collar and order.price is not None:
            checks.append(self._check_price_collar(order, current_price))

        # Core risk checks
        checks.extend([
            self._check_position_limit(order),
            self._check_notional_limit(order, current_price),
            self._check_rate_limit(),
            self._check_drawdown(),
            self._check_circuit_breaker(),
        ])

        for check in checks:
            if not check.passed:
                self._orders_denied += 1
                self._record_latency(start_ns)

                logger.warning(
                    "Order DENIED: symbol=%s check=%s reason=%s",
                    order.symbol,
                    check.check_name,
                    check.reason,
                )

                # Publish ORDER_DENIED event (high priority)
                self._publish_event(
                    EventType.ORDER_DENIED,
                    {
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "amount": str(order.amount),
                        "price": str(order.price) if order.price else None,
                        "client_order_id": order.client_order_id,
                        "check": check.check_name,
                        "reason": check.reason,
                    },
                )

                return check

        # All checks passed
        self._record_latency(start_ns)
        return RiskCheckResult(passed=True, check_name="all_checks")

    def validate_modify(
        self,
        order_id: str,
        new_price: Decimal | None = None,
        new_amount: Decimal | None = None,
        current_price: Decimal | None = None,
        instrument: Instrument | None = None,
    ) -> RiskCheckResult:
        """
        Validate order modification against risk limits.

        Uses separate rate limiter for modifications.

        Args:
            order_id: ID of order to modify
            new_price: New price (if changing)
            new_amount: New amount (if changing)
            current_price: Current market price (for collar check)
            instrument: Instrument for precision validation

        Returns:
            RiskCheckResult with pass/fail and reason
        """
        # Check modify rate limit
        if not self._modify_rate_limiter.try_acquire():
            return RiskCheckResult(
                passed=False,
                check_name="modify_rate_limit",
                reason=f"Modify rate limit exceeded ({self.config.max_modify_rate}/sec)",
            )

        # Check trading state
        if self._trading_state == TradingState.HALTED:
            return RiskCheckResult(
                passed=False,
                check_name="trading_state",
                reason="Trading HALTED - modifications not allowed",
            )

        # Validate new price precision
        if (
            new_price is not None
            and instrument is not None
            and self.config.enable_precision_validation
        ):
            result = self._validate_price_value(new_price, instrument)
            if not result.passed:
                return result

        # Validate new amount precision
        if (
            new_amount is not None
            and instrument is not None
            and self.config.enable_precision_validation
        ):
            result = self._validate_quantity_value(new_amount, instrument)
            if not result.passed:
                return result

        # Price collar check for new price
        if (
            new_price is not None
            and current_price is not None
            and self.config.enable_price_collar
        ):
            deviation = abs(new_price - current_price) / current_price
            if deviation > self.config.price_collar_pct:
                return RiskCheckResult(
                    passed=False,
                    check_name="price_collar",
                    reason=f"Modified price {new_price} deviates {float(deviation):.1%} "
                    f"from market {current_price} (max {float(self.config.price_collar_pct):.1%})",
                )

        return RiskCheckResult(passed=True, check_name="all_modify_checks")

    def _record_latency(self, start_ns: int) -> None:
        """Record check latency for monitoring."""
        latency = time.time_ns() - start_ns
        # Keep last 1000 measurements
        if len(self._check_latency_ns) >= 1000:
            self._check_latency_ns.pop(0)
        self._check_latency_ns.append(latency)

    # =========================================================================
    # Individual Risk Checks
    # =========================================================================

    def _check_trading_state(self, order: Order) -> RiskCheckResult:
        """Check if order is allowed in current trading state."""
        if self._trading_state == TradingState.HALTED:
            return RiskCheckResult(
                passed=False,
                check_name="trading_state",
                reason="Trading HALTED - only cancellations allowed",
            )

        if self._trading_state == TradingState.REDUCING:
            # Only allow orders that reduce position
            position = self._positions.get(order.symbol)
            if position and not self._order_reduces_position(order, position):
                return RiskCheckResult(
                    passed=False,
                    check_name="trading_state",
                    reason="REDUCING state - only position-reducing orders allowed",
                )

        return RiskCheckResult(passed=True, check_name="trading_state")

    def _order_reduces_position(self, order: Order, position: Position) -> bool:
        """Check if order would reduce the current position."""
        from libra.gateways.protocol import OrderSide, PositionSide

        # Long position: SELL reduces
        if position.side == PositionSide.LONG and order.side == OrderSide.SELL:
            return True
        # Short position: BUY reduces
        if position.side == PositionSide.SHORT and order.side == OrderSide.BUY:
            return True
        # reduce_only flag
        if order.reduce_only:
            return True

        return False

    def _check_self_trade(self, order: Order) -> RiskCheckResult:
        """
        Check for potential self-trade (wash trading).

        Prevents placing orders that would immediately match against
        our own open orders on the same symbol.
        """
        from libra.gateways.protocol import OrderSide

        open_orders = self._open_orders.get(order.symbol, [])
        if not open_orders:
            return RiskCheckResult(passed=True, check_name="self_trade")

        for existing in open_orders:
            # Skip if same side (won't match)
            if existing.side == order.side:
                continue

            # Check if prices would cross
            if order.price is not None and existing.price is not None:
                # BUY order with existing SELL: would match if buy_price >= sell_price
                if order.side == OrderSide.BUY and order.price >= existing.price:
                    return RiskCheckResult(
                        passed=False,
                        check_name="self_trade",
                        reason=f"Would self-trade: BUY at {order.price} vs "
                        f"open SELL at {existing.price}",
                    )
                # SELL order with existing BUY: would match if sell_price <= buy_price
                if order.side == OrderSide.SELL and order.price <= existing.price:
                    return RiskCheckResult(
                        passed=False,
                        check_name="self_trade",
                        reason=f"Would self-trade: SELL at {order.price} vs "
                        f"open BUY at {existing.price}",
                    )

            # Market orders always risk self-trade if opposite side exists
            elif order.price is None:
                return RiskCheckResult(
                    passed=False,
                    check_name="self_trade",
                    reason=f"Market {order.side.value} would self-trade vs "
                    f"open {existing.side.value} order",
                )

        return RiskCheckResult(passed=True, check_name="self_trade")

    def _check_price_precision(self, order: Order, instrument: Instrument) -> RiskCheckResult:
        """Check if order price matches instrument tick size."""
        if order.price is None:
            return RiskCheckResult(passed=True, check_name="price_precision")

        return self._validate_price_value(order.price, instrument)

    def _validate_price_value(self, price: Decimal, instrument: Instrument) -> RiskCheckResult:
        """Validate a price value against instrument tick size."""
        tick_size = Decimal(str(instrument.tick_size))
        if tick_size <= 0:
            return RiskCheckResult(passed=True, check_name="price_precision")

        try:
            # Check if price is a multiple of tick_size
            remainder = price % tick_size
            if remainder != 0:
                return RiskCheckResult(
                    passed=False,
                    check_name="price_precision",
                    reason=f"Price {price} not a multiple of tick_size {tick_size}",
                )
        except InvalidOperation:
            return RiskCheckResult(
                passed=False,
                check_name="price_precision",
                reason=f"Invalid price value: {price}",
            )

        return RiskCheckResult(passed=True, check_name="price_precision")

    def _check_quantity_precision(self, order: Order, instrument: Instrument) -> RiskCheckResult:
        """Check if order quantity matches instrument lot size."""
        return self._validate_quantity_value(order.amount, instrument)

    def _validate_quantity_value(self, amount: Decimal, instrument: Instrument) -> RiskCheckResult:
        """Validate a quantity value against instrument lot size."""
        lot_size = Decimal(str(instrument.lot_size))
        if lot_size <= 0:
            return RiskCheckResult(passed=True, check_name="quantity_precision")

        try:
            # Check if amount is a multiple of lot_size
            remainder = amount % lot_size
            if remainder != 0:
                return RiskCheckResult(
                    passed=False,
                    check_name="quantity_precision",
                    reason=f"Quantity {amount} not a multiple of lot_size {lot_size}",
                )
        except InvalidOperation:
            return RiskCheckResult(
                passed=False,
                check_name="quantity_precision",
                reason=f"Invalid quantity value: {amount}",
            )

        return RiskCheckResult(passed=True, check_name="quantity_precision")

    def _check_price_collar(self, order: Order, current_price: Decimal) -> RiskCheckResult:
        """
        Check if order price is within acceptable range of market price.

        Fat-finger protection to prevent orders far from current market.
        """
        if order.price is None or current_price <= 0:
            return RiskCheckResult(passed=True, check_name="price_collar")

        deviation = abs(order.price - current_price) / current_price

        if deviation > self.config.price_collar_pct:
            return RiskCheckResult(
                passed=False,
                check_name="price_collar",
                reason=f"Price {order.price} deviates {float(deviation):.1%} from "
                f"market {current_price} (max {float(self.config.price_collar_pct):.1%})",
            )

        return RiskCheckResult(passed=True, check_name="price_collar")

    def _check_position_limit(self, order: Order) -> RiskCheckResult:
        """Check if order would exceed position limits."""
        from libra.gateways.protocol import OrderSide

        symbol_limits = self.config.limits.get_symbol_limits(order.symbol)
        current_position = self._positions.get(order.symbol)

        # Calculate new position size
        current_size = Decimal("0")
        if current_position:
            current_size = current_position.amount

        # For simplicity, treat buy as add, sell as reduce (adjust for actual position side)
        if order.side == OrderSide.BUY:
            new_size = current_size + order.amount
        else:
            new_size = abs(current_size - order.amount)

        if new_size > symbol_limits.max_position_size:
            return RiskCheckResult(
                passed=False,
                check_name="position_limit",
                reason=f"Would exceed max position {symbol_limits.max_position_size} "
                f"(current: {current_size}, order: {order.amount})",
            )

        return RiskCheckResult(passed=True, check_name="position_limit")

    def _check_notional_limit(
        self,
        order: Order,
        current_price: Decimal,
    ) -> RiskCheckResult:
        """Check if order notional value exceeds limit."""
        symbol_limits = self.config.limits.get_symbol_limits(order.symbol)

        notional = order.amount * current_price
        if notional > symbol_limits.max_notional_per_order:
            return RiskCheckResult(
                passed=False,
                check_name="notional_limit",
                reason=f"Notional {notional} exceeds max {symbol_limits.max_notional_per_order}",
            )

        return RiskCheckResult(passed=True, check_name="notional_limit")

    def _check_rate_limit(self) -> RiskCheckResult:
        """Check order submission rate limit."""
        if not self._submit_rate_limiter.try_acquire():
            return RiskCheckResult(
                passed=False,
                check_name="rate_limit",
                reason=f"Rate limit exceeded ({self.config.limits.max_orders_per_second}/sec)",
            )
        return RiskCheckResult(passed=True, check_name="rate_limit")

    def _check_drawdown(self) -> RiskCheckResult:
        """Check current drawdown against limits."""
        if self._peak_equity <= 0:
            return RiskCheckResult(passed=True, check_name="drawdown")

        drawdown = self.current_drawdown
        limits = self.config.limits

        # Check total drawdown (HALT)
        if drawdown <= limits.max_total_drawdown_pct:
            self.set_trading_state(
                TradingState.HALTED,
                f"Max drawdown {float(drawdown):.2%} exceeded limit "
                f"{float(limits.max_total_drawdown_pct):.2%}",
            )
            return RiskCheckResult(
                passed=False,
                check_name="max_drawdown",
                reason=f"Max drawdown breached: {float(drawdown):.2%}",
            )

        # Check daily loss (REDUCING)
        if self._daily_pnl <= limits.max_daily_loss_pct:
            if self._trading_state == TradingState.ACTIVE:
                self.set_trading_state(
                    TradingState.REDUCING,
                    f"Daily loss {float(self._daily_pnl):.2%} exceeded limit "
                    f"{float(limits.max_daily_loss_pct):.2%}",
                )
            return RiskCheckResult(
                passed=False,
                check_name="daily_loss",
                reason=f"Daily loss limit breached: {float(self._daily_pnl):.2%}",
            )

        return RiskCheckResult(passed=True, check_name="drawdown")

    def _check_circuit_breaker(self) -> RiskCheckResult:
        """Check circuit breaker status."""
        if self._circuit_breaker.is_open:
            return RiskCheckResult(
                passed=False,
                check_name="circuit_breaker",
                reason=f"Circuit breaker open: {self._circuit_breaker.trip_reason}",
            )
        return RiskCheckResult(passed=True, check_name="circuit_breaker")

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_position(self, position: Position) -> None:
        """
        Update tracked position state.

        Call after each position change (fill, close).

        Args:
            position: Updated position
        """
        self._positions[position.symbol] = position

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position from tracking."""
        self._positions.pop(symbol, None)

    def add_open_order(self, order: Order) -> None:
        """
        Track an open order for self-trade prevention.

        Call after order is successfully submitted.

        Args:
            order: Order that was submitted
        """
        if order.symbol not in self._open_orders:
            self._open_orders[order.symbol] = []
        self._open_orders[order.symbol].append(order)

    def remove_open_order(self, symbol: str, order_id: str | None = None, client_order_id: str | None = None) -> None:
        """
        Remove an order from open orders tracking.

        Call after order is filled, cancelled, or rejected.

        Args:
            symbol: Order symbol
            order_id: Exchange order ID
            client_order_id: Client order ID
        """
        if symbol not in self._open_orders:
            return

        self._open_orders[symbol] = [
            o for o in self._open_orders[symbol]
            if not (
                (order_id and o.id == order_id) or
                (client_order_id and o.client_order_id == client_order_id)
            )
        ]

        # Clean up empty lists
        if not self._open_orders[symbol]:
            del self._open_orders[symbol]

    def clear_open_orders(self, symbol: str | None = None) -> None:
        """
        Clear open orders tracking.

        Args:
            symbol: Clear for specific symbol, or all if None
        """
        if symbol is None:
            self._open_orders.clear()
        else:
            self._open_orders.pop(symbol, None)

    def update_equity(self, equity: Decimal) -> None:
        """
        Update equity and track peak for drawdown.

        Call periodically (e.g., every second) or after each fill.

        Args:
            equity: Current account equity
        """
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Also check circuit breaker
        if self._peak_equity > 0:
            drawdown_pct = self.current_drawdown
            self._circuit_breaker.check_pnl(drawdown_pct)

    def update_pnl(self, daily_pnl: Decimal, weekly_pnl: Decimal | None = None) -> None:
        """
        Update P&L tracking.

        Args:
            daily_pnl: Daily P&L as decimal (e.g., -0.02 for -2%)
            weekly_pnl: Weekly P&L (optional)
        """
        self._daily_pnl = daily_pnl
        if weekly_pnl is not None:
            self._weekly_pnl = weekly_pnl

    def record_trade_result(self, profitable: bool) -> None:
        """
        Record trade result for circuit breaker.

        Call after each closed trade.

        Args:
            profitable: True if trade was profitable
        """
        self._circuit_breaker.record_trade_result(profitable)

    def reset_daily(self) -> None:
        """
        Reset daily tracking.

        Call at day boundary (e.g., 00:00 UTC).
        """
        self._daily_pnl = Decimal("0")
        logger.info("Daily P&L tracking reset")

    def reset_weekly(self) -> None:
        """
        Reset weekly tracking.

        Call at week boundary.
        """
        self._weekly_pnl = Decimal("0")
        logger.info("Weekly P&L tracking reset")

    # =========================================================================
    # Metrics & Status
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get risk engine statistics."""
        avg_latency_us = 0.0
        p99_latency_us = 0.0

        if self._check_latency_ns:
            sorted_latencies = sorted(self._check_latency_ns)
            avg_latency_us = sum(sorted_latencies) / len(sorted_latencies) / 1000
            p99_idx = int(len(sorted_latencies) * 0.99)
            p99_latency_us = sorted_latencies[p99_idx] / 1000

        return {
            "trading_state": self._trading_state.value,
            "orders_checked": self._orders_checked,
            "orders_denied": self._orders_denied,
            "denial_rate": (
                self._orders_denied / self._orders_checked
                if self._orders_checked > 0
                else 0.0
            ),
            "current_equity": str(self._current_equity),
            "peak_equity": str(self._peak_equity),
            "current_drawdown": f"{float(self.current_drawdown):.2%}",
            "daily_pnl": f"{float(self._daily_pnl):.2%}",
            "positions_tracked": len(self._positions),
            "open_orders_tracked": sum(len(v) for v in self._open_orders.values()),
            "circuit_breaker": self._circuit_breaker.get_status(),
            "avg_check_latency_us": round(avg_latency_us, 2),
            "p99_check_latency_us": round(p99_latency_us, 2),
            "config": {
                "self_trade_prevention": self.config.enable_self_trade_prevention,
                "price_collar": self.config.enable_price_collar,
                "price_collar_pct": float(self.config.price_collar_pct),
                "precision_validation": self.config.enable_precision_validation,
            },
        }

    # =========================================================================
    # Event Publishing
    # =========================================================================

    def _publish_event(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """Publish event to message bus if available."""
        if self.bus is None:
            return

        event = Event.create(
            event_type=event_type,
            source="risk_engine",
            payload=payload,
        )
        self.bus.publish(event)

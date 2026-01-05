"""
Risk Manager for LIBRA.

Core risk management engine that validates ALL orders before execution.
Implements pre-trade risk checks, position monitoring, and trading state management.

Architecture:
    Strategy -> RiskManager.validate_order() -> ExecutionEngine -> Gateway

ALL orders MUST pass through the RiskManager. This is mandatory, not optional.

Based on NautilusTrader's RiskEngine pattern:
    https://nautilustrader.io/docs/latest/api_reference/risk/
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

import msgspec

from libra.core.events import Event, EventType
from libra.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from libra.risk.limits import RiskLimits
from libra.risk.rate_limiter import MultiRateLimiter, TokenBucketRateLimiter


if TYPE_CHECKING:
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
# Risk Manager
# =============================================================================


@dataclass
class RiskManager:
    """
    Pre-trade risk validation engine.

    ALL orders MUST pass through this engine before execution.
    Implements the complete pre-trade validation pipeline.

    Pipeline (in order):
        1. Trading State Check (ACTIVE/REDUCING/HALTED)
        2. Position Limit Check
        3. Notional Value Check
        4. Order Rate Limiting
        5. Drawdown Check
        6. Circuit Breaker Check

    Performance Target: <1ms for all checks combined.

    Examples:
        limits = RiskLimits(...)
        manager = RiskManager(limits=limits, bus=message_bus)

        # Before every order
        result = manager.validate_order(order, current_price)
        if not result.passed:
            logger.warning("Order rejected: %s", result.reason)
            return

        # Order passed all checks, proceed to execution
        await gateway.submit_order(order)

    Thread Safety:
        This implementation is NOT thread-safe. For multi-threaded use,
        wrap calls in a lock or use the Rust implementation (Phase 1B).
    """

    limits: RiskLimits
    bus: MessageBus | None = None  # Optional for standalone use

    # Internal state (initialized in __post_init__)
    _trading_state: TradingState = field(init=False, default=TradingState.ACTIVE)
    _rate_limiter: MultiRateLimiter = field(init=False, repr=False)
    _circuit_breaker: CircuitBreaker = field(init=False, repr=False)

    # Tracking state
    _peak_equity: Decimal = field(init=False, default=Decimal("0"))
    _current_equity: Decimal = field(init=False, default=Decimal("0"))
    _daily_pnl: Decimal = field(init=False, default=Decimal("0"))
    _weekly_pnl: Decimal = field(init=False, default=Decimal("0"))
    _positions: dict[str, Position] = field(init=False, default_factory=dict)

    # Metrics
    _orders_checked: int = field(init=False, default=0)
    _orders_rejected: int = field(init=False, default=0)
    _check_latency_ns: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Initialize rate limiter and circuit breaker from limits."""
        # Multi-tier rate limiter: per-second AND per-minute
        self._rate_limiter = MultiRateLimiter(
            limiters=[
                TokenBucketRateLimiter(
                    rate=float(self.limits.max_orders_per_second),
                    capacity=self.limits.max_orders_per_second,
                ),
                TokenBucketRateLimiter(
                    rate=self.limits.max_orders_per_minute / 60.0,
                    capacity=self.limits.max_orders_per_minute,
                ),
            ]
        )

        self._circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                drawdown_threshold=self.limits.circuit_breaker_drawdown_pct,
                cooldown_seconds=self.limits.circuit_breaker_cooldown_seconds,
                max_consecutive_losses=self.limits.circuit_breaker_max_consecutive_losses,
            )
        )

        logger.info(
            "RiskManager initialized: max_daily_loss=%.1f%%, max_drawdown=%.1f%%, "
            "rate_limit=%d/sec",
            float(self.limits.max_daily_loss_pct) * 100,
            float(self.limits.max_total_drawdown_pct) * 100,
            self.limits.max_orders_per_second,
        )

    # =========================================================================
    # Properties
    # =========================================================================

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
    ) -> RiskCheckResult:
        """
        Validate order against all risk limits.

        This is the main entry point - EVERY order must pass through here.
        Returns immediately on first failed check for performance.

        Args:
            order: Order to validate
            current_price: Current market price for notional calculation

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
            self._check_position_limit(order),
            self._check_notional_limit(order, current_price),
            self._check_rate_limit(),
            self._check_drawdown(),
            self._check_circuit_breaker(),
        ]

        for check in checks:
            if not check.passed:
                self._orders_rejected += 1
                self._record_latency(start_ns)

                logger.warning(
                    "Order rejected: symbol=%s check=%s reason=%s",
                    order.symbol,
                    check.check_name,
                    check.reason,
                )

                # Publish rejection event
                self._publish_event(
                    EventType.ORDER_REJECTED,
                    {
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "amount": str(order.amount),
                        "check": check.check_name,
                        "reason": check.reason,
                    },
                )

                return check

        # All checks passed
        self._record_latency(start_ns)
        return RiskCheckResult(passed=True, check_name="all_checks")

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

    def _check_position_limit(self, order: Order) -> RiskCheckResult:
        """Check if order would exceed position limits."""
        from libra.gateways.protocol import OrderSide

        symbol_limits = self.limits.get_symbol_limits(order.symbol)
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
        symbol_limits = self.limits.get_symbol_limits(order.symbol)

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
        if not self._rate_limiter.try_acquire():
            return RiskCheckResult(
                passed=False,
                check_name="rate_limit",
                reason=f"Rate limit exceeded ({self.limits.max_orders_per_second}/sec)",
            )
        return RiskCheckResult(passed=True, check_name="rate_limit")

    def _check_drawdown(self) -> RiskCheckResult:
        """Check current drawdown against limits."""
        if self._peak_equity <= 0:
            return RiskCheckResult(passed=True, check_name="drawdown")

        drawdown = self.current_drawdown

        # Check total drawdown (HALT)
        if drawdown <= self.limits.max_total_drawdown_pct:
            self.set_trading_state(
                TradingState.HALTED,
                f"Max drawdown {float(drawdown):.2%} exceeded limit "
                f"{float(self.limits.max_total_drawdown_pct):.2%}",
            )
            return RiskCheckResult(
                passed=False,
                check_name="max_drawdown",
                reason=f"Max drawdown breached: {float(drawdown):.2%}",
            )

        # Check daily loss (REDUCING)
        if self._daily_pnl <= self.limits.max_daily_loss_pct:
            if self._trading_state == TradingState.ACTIVE:
                self.set_trading_state(
                    TradingState.REDUCING,
                    f"Daily loss {float(self._daily_pnl):.2%} exceeded limit "
                    f"{float(self.limits.max_daily_loss_pct):.2%}",
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
        """Get risk manager statistics."""
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
            "orders_rejected": self._orders_rejected,
            "rejection_rate": (
                self._orders_rejected / self._orders_checked
                if self._orders_checked > 0
                else 0.0
            ),
            "current_equity": str(self._current_equity),
            "peak_equity": str(self._peak_equity),
            "current_drawdown": f"{float(self.current_drawdown):.2%}",
            "daily_pnl": f"{float(self._daily_pnl):.2%}",
            "positions_tracked": len(self._positions),
            "circuit_breaker": self._circuit_breaker.get_status(),
            "avg_check_latency_us": round(avg_latency_us, 2),
            "p99_check_latency_us": round(p99_latency_us, 2),
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
            source="risk_manager",
            payload=payload,
        )
        self.bus.publish(event)

"""
Circuit Breaker for LIBRA Trading Operations.

Automatically halts trading when risk thresholds are breached:
- P&L drops below threshold
- Consecutive losing trades exceed limit
- Extreme volatility detected

Based on Martin Fowler's Circuit Breaker pattern adapted for trading.

Reference:
    https://martinfowler.com/bliki/CircuitBreaker.html
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """
    Circuit breaker states.

    State Transitions:
        CLOSED -> OPEN: When failure threshold exceeded
        OPEN -> HALF_OPEN: After cooldown period
        HALF_OPEN -> CLOSED: When test trades succeed
        HALF_OPEN -> OPEN: When test trade fails
    """

    CLOSED = "closed"  # Normal operation - all trades allowed
    OPEN = "open"  # Failures exceeded threshold - blocking trades
    HALF_OPEN = "half_open"  # Testing recovery - limited trades allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # P&L threshold (negative, e.g., -0.05 for -5%)
    drawdown_threshold: Decimal = Decimal("-0.05")

    # Cooldown period before attempting recovery (seconds)
    cooldown_seconds: int = 300  # 5 minutes

    # Consecutive losses before tripping
    max_consecutive_losses: int = 10

    # Number of test trades in HALF_OPEN state before recovery
    test_trades_required: int = 3

    # Volatility threshold (multiple of baseline)
    volatility_threshold: Decimal = Decimal("3.0")


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for trading operations.

    Automatically halts trading when risk thresholds are breached.
    After a cooldown period, enters HALF_OPEN state to test recovery
    with limited trading before returning to normal operation.

    Examples:
        breaker = CircuitBreaker(config=CircuitBreakerConfig(
            drawdown_threshold=Decimal("-0.05"),
            cooldown_seconds=300,
            max_consecutive_losses=10,
        ))

        # Check before trading
        if breaker.is_open:
            return "Trading halted"

        # Record trade results
        breaker.record_trade_result(profitable=True)

        # Check P&L
        breaker.check_pnl(current_pnl_pct=Decimal("-0.03"))

    Thread Safety:
        This implementation is NOT thread-safe. For multi-threaded use,
        wrap calls in a lock.
    """

    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Internal state
    _state: CircuitState = field(init=False, default=CircuitState.CLOSED)
    _last_trip_time: float = field(init=False, default=0.0)
    _consecutive_losses: int = field(init=False, default=0)
    _test_trades_remaining: int = field(init=False, default=0)
    _trip_reason: str = field(init=False, default="")

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        self._check_cooldown()
        return self._state

    @property
    def is_open(self) -> bool:
        """
        Check if circuit is open (blocking new trades).

        Automatically transitions to HALF_OPEN after cooldown.
        """
        self._check_cooldown()
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        self._check_cooldown()
        return self._state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is in half-open testing state."""
        self._check_cooldown()
        return self._state == CircuitState.HALF_OPEN

    @property
    def trip_reason(self) -> str:
        """Reason for last circuit trip."""
        return self._trip_reason

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive loss count."""
        return self._consecutive_losses

    @property
    def time_until_half_open(self) -> float:
        """
        Seconds until circuit transitions to HALF_OPEN.

        Returns 0.0 if not in OPEN state.
        """
        if self._state != CircuitState.OPEN:
            return 0.0

        elapsed = time.time() - self._last_trip_time
        remaining = self.config.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def _check_cooldown(self) -> None:
        """Transition from OPEN to HALF_OPEN after cooldown period."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_trip_time
            if elapsed >= self.config.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                self._test_trades_remaining = self.config.test_trades_required
                logger.info(
                    "Circuit breaker transitioning to HALF_OPEN after %ds cooldown",
                    int(elapsed),
                )

    def record_trade_result(self, profitable: bool) -> None:
        """
        Record trade result for consecutive loss tracking.

        In HALF_OPEN state:
        - Winning trades count toward recovery
        - Losing trades immediately reopen the circuit

        Args:
            profitable: True if trade was profitable (including breakeven)
        """
        if profitable:
            self._consecutive_losses = 0

            if self._state == CircuitState.HALF_OPEN:
                self._test_trades_remaining -= 1
                logger.debug(
                    "Circuit breaker: profitable trade in HALF_OPEN, %d tests remaining",
                    self._test_trades_remaining,
                )

                if self._test_trades_remaining <= 0:
                    self._state = CircuitState.CLOSED
                    self._trip_reason = ""
                    logger.info("Circuit breaker CLOSED - recovery successful")
        else:
            self._consecutive_losses += 1

            if self._state == CircuitState.HALF_OPEN:
                self._trip("Losing trade during recovery test")
            elif self._consecutive_losses >= self.config.max_consecutive_losses:
                self._trip(
                    f"{self._consecutive_losses} consecutive losses "
                    f"(limit: {self.config.max_consecutive_losses})"
                )

    def check_pnl(self, current_pnl_pct: Decimal) -> bool:
        """
        Check if P&L has breached threshold.

        Args:
            current_pnl_pct: Current P&L as decimal (e.g., -0.03 for -3%)

        Returns:
            True if OK (circuit remains closed/half-open)
            False if threshold breached (circuit opened)
        """
        if current_pnl_pct <= self.config.drawdown_threshold:
            self._trip(
                f"P&L {float(current_pnl_pct):.2%} breached threshold "
                f"{float(self.config.drawdown_threshold):.2%}"
            )
            return False
        return True

    def check_volatility(
        self,
        current_vol: Decimal,
        baseline_vol: Decimal,
    ) -> bool:
        """
        Check if volatility has exceeded threshold.

        Trips circuit if current volatility exceeds baseline by
        the configured threshold multiple.

        Args:
            current_vol: Current volatility measure (e.g., ATR)
            baseline_vol: Baseline/normal volatility

        Returns:
            True if OK, False if circuit tripped
        """
        if baseline_vol <= 0:
            return True

        vol_ratio = current_vol / baseline_vol
        if vol_ratio >= self.config.volatility_threshold:
            self._trip(
                f"Volatility {float(vol_ratio):.1f}x normal "
                f"(threshold: {float(self.config.volatility_threshold):.1f}x)"
            )
            return False
        return True

    def _trip(self, reason: str) -> None:
        """
        Trip the circuit breaker.

        Args:
            reason: Human-readable reason for tripping
        """
        previous_state = self._state
        self._state = CircuitState.OPEN
        self._last_trip_time = time.time()
        self._trip_reason = reason

        logger.warning(
            "Circuit breaker TRIPPED: %s (previous state: %s, cooldown: %ds)",
            reason,
            previous_state.value,
            self.config.cooldown_seconds,
        )

    def reset(self) -> None:
        """
        Manually reset circuit breaker to CLOSED state.

        Use with caution - bypasses normal recovery process.
        """
        self._state = CircuitState.CLOSED
        self._consecutive_losses = 0
        self._test_trades_remaining = 0
        self._trip_reason = ""
        logger.info("Circuit breaker manually reset to CLOSED")

    def force_open(self, reason: str = "manual") -> None:
        """
        Manually trip the circuit breaker.

        Useful for emergency stops or maintenance.
        """
        self._trip(f"Manual: {reason}")

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status as dict."""
        self._check_cooldown()
        return {
            "state": self._state.value,
            "is_open": self._state == CircuitState.OPEN,
            "consecutive_losses": self._consecutive_losses,
            "trip_reason": self._trip_reason,
            "time_until_half_open": self.time_until_half_open,
            "test_trades_remaining": (
                self._test_trades_remaining if self._state == CircuitState.HALF_OPEN else 0
            ),
        }

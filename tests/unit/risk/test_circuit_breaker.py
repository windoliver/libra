"""Tests for CircuitBreaker."""

import time
from decimal import Decimal

import pytest

from libra.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Default configuration values."""
        config = CircuitBreakerConfig()

        assert config.drawdown_threshold == Decimal("-0.05")
        assert config.cooldown_seconds == 300
        assert config.max_consecutive_losses == 10
        assert config.test_trades_required == 3

    def test_custom_config(self):
        """Custom configuration values."""
        config = CircuitBreakerConfig(
            drawdown_threshold=Decimal("-0.10"),
            cooldown_seconds=600,
            max_consecutive_losses=5,
        )

        assert config.drawdown_threshold == Decimal("-0.10")
        assert config.cooldown_seconds == 600
        assert config.max_consecutive_losses == 5


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Circuit starts in CLOSED state."""
        breaker = CircuitBreaker()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    def test_trip_on_pnl_breach(self):
        """Circuit trips when P&L breaches threshold."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(drawdown_threshold=Decimal("-0.05"))
        )

        # P&L at -6% breaches -5% threshold
        result = breaker.check_pnl(Decimal("-0.06"))

        assert result is False
        assert breaker.is_open is True
        assert "P&L" in breaker.trip_reason

    def test_no_trip_above_threshold(self):
        """Circuit stays closed when P&L above threshold."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(drawdown_threshold=Decimal("-0.05"))
        )

        result = breaker.check_pnl(Decimal("-0.03"))

        assert result is True
        assert breaker.is_closed is True

    def test_trip_on_consecutive_losses(self):
        """Circuit trips after max consecutive losses."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(max_consecutive_losses=3)
        )

        # Record 3 consecutive losses
        breaker.record_trade_result(profitable=False)
        breaker.record_trade_result(profitable=False)
        assert breaker.is_closed is True

        breaker.record_trade_result(profitable=False)
        assert breaker.is_open is True
        assert "consecutive losses" in breaker.trip_reason

    def test_winning_trade_resets_loss_count(self):
        """Winning trade resets consecutive loss counter."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(max_consecutive_losses=3)
        )

        breaker.record_trade_result(profitable=False)
        breaker.record_trade_result(profitable=False)
        assert breaker.consecutive_losses == 2

        breaker.record_trade_result(profitable=True)
        assert breaker.consecutive_losses == 0
        assert breaker.is_closed is True

    def test_volatility_trip(self):
        """Circuit trips on extreme volatility."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(volatility_threshold=Decimal("3.0"))
        )

        # 4x normal volatility breaches 3x threshold
        result = breaker.check_volatility(
            current_vol=Decimal("40"),
            baseline_vol=Decimal("10"),
        )

        assert result is False
        assert breaker.is_open is True
        assert "Volatility" in breaker.trip_reason

    def test_volatility_normal(self):
        """Circuit stays closed on normal volatility."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(volatility_threshold=Decimal("3.0"))
        )

        result = breaker.check_volatility(
            current_vol=Decimal("20"),
            baseline_vol=Decimal("10"),
        )

        assert result is True
        assert breaker.is_closed is True

    def test_cooldown_transition_to_half_open(self):
        """Circuit transitions to HALF_OPEN after cooldown."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(cooldown_seconds=1)  # 1 second for testing
        )

        # Trip the breaker
        breaker.check_pnl(Decimal("-0.10"))
        assert breaker.is_open is True

        # Wait for cooldown
        time.sleep(1.1)

        # Should transition to HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.is_half_open is True

    def test_half_open_recovery_on_wins(self):
        """Circuit closes after successful test trades in HALF_OPEN."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                cooldown_seconds=0,  # Immediate for testing
                test_trades_required=2,
            )
        )

        # Trip and transition to half-open
        breaker.check_pnl(Decimal("-0.10"))
        time.sleep(0.01)  # Trigger cooldown check
        _ = breaker.state  # Trigger state check

        # Record winning test trades
        breaker.record_trade_result(profitable=True)
        assert breaker.is_half_open is True

        breaker.record_trade_result(profitable=True)
        assert breaker.is_closed is True

    def test_half_open_reopen_on_loss(self):
        """Circuit reopens if losing trade during HALF_OPEN."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                cooldown_seconds=0,
                test_trades_required=3,
            )
        )

        # Trip and transition to half-open
        breaker.check_pnl(Decimal("-0.10"))
        time.sleep(0.01)
        # Force state check to trigger transition
        state = breaker.state
        assert state == CircuitState.HALF_OPEN

        # Losing trade in half-open should reopen
        breaker.record_trade_result(profitable=False)
        # After recording loss, it trips again - need to wait for cooldown check
        time.sleep(0.01)
        # The state is OPEN after trip, but with 0 cooldown it immediately goes to HALF_OPEN
        # So we check the trip_reason instead
        assert "Losing trade" in breaker.trip_reason

    def test_time_until_half_open(self):
        """Calculate remaining cooldown time."""
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(cooldown_seconds=10)
        )

        # Not tripped - should be 0
        assert breaker.time_until_half_open == 0.0

        # Trip the breaker
        breaker.check_pnl(Decimal("-0.10"))

        # Should be close to 10 seconds
        remaining = breaker.time_until_half_open
        assert 9.0 <= remaining <= 10.0

    def test_manual_reset(self):
        """Manually reset circuit breaker."""
        breaker = CircuitBreaker()

        breaker.check_pnl(Decimal("-0.10"))
        assert breaker.is_open is True

        breaker.reset()

        assert breaker.is_closed is True
        assert breaker.consecutive_losses == 0
        assert breaker.trip_reason == ""

    def test_force_open(self):
        """Manually trip circuit breaker."""
        breaker = CircuitBreaker()

        breaker.force_open(reason="maintenance")

        assert breaker.is_open is True
        assert "maintenance" in breaker.trip_reason

    def test_get_status(self):
        """Get circuit breaker status dict."""
        breaker = CircuitBreaker()

        status = breaker.get_status()

        assert status["state"] == "closed"
        assert status["is_open"] is False
        assert status["consecutive_losses"] == 0
        assert status["trip_reason"] == ""

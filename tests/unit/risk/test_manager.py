"""Tests for RiskManager."""

import time
from decimal import Decimal

import pytest

from libra.gateways.protocol import Order, OrderSide, OrderType, Position, PositionSide
from libra.risk import (
    RiskCheckResult,
    RiskLimits,
    RiskManager,
    SymbolLimits,
    TradingState,
)


@pytest.fixture
def basic_limits() -> RiskLimits:
    """Basic risk limits for testing."""
    return RiskLimits(
        max_total_exposure=Decimal("100000"),
        max_single_position_pct=Decimal("0.20"),
        max_daily_loss_pct=Decimal("-0.03"),
        max_weekly_loss_pct=Decimal("-0.07"),
        max_total_drawdown_pct=Decimal("-0.15"),
        max_orders_per_second=10,
        max_orders_per_minute=60,
        symbol_limits={
            "BTC/USDT": SymbolLimits(
                max_position_size=Decimal("1.0"),
                max_notional_per_order=Decimal("50000"),
                max_order_rate=5,
            ),
        },
    )


@pytest.fixture
def manager(basic_limits: RiskLimits) -> RiskManager:
    """RiskManager instance for testing."""
    return RiskManager(limits=basic_limits, bus=None)


@pytest.fixture
def sample_order() -> Order:
    """Sample order for testing."""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.5"),
    )


class TestRiskManagerCreation:
    """Tests for RiskManager initialization."""

    def test_create_manager(self, basic_limits):
        """Create RiskManager with limits."""
        manager = RiskManager(limits=basic_limits)

        assert manager.limits == basic_limits
        assert manager.trading_state == TradingState.ACTIVE
        assert manager.is_active is True

    def test_initial_state(self, manager):
        """Manager starts in correct initial state."""
        assert manager.is_active is True
        assert manager.is_halted is False
        assert manager.current_drawdown == Decimal("0")


class TestTradingStateManagement:
    """Tests for trading state transitions."""

    def test_halt_trading(self, manager):
        """Halt trading manually."""
        manager.halt_trading(reason="test")

        assert manager.trading_state == TradingState.HALTED
        assert manager.is_halted is True
        assert manager.is_active is False

    def test_resume_trading(self, manager):
        """Resume trading after halt."""
        manager.halt_trading(reason="test")
        manager.resume_trading()

        assert manager.trading_state == TradingState.ACTIVE
        assert manager.is_active is True

    def test_set_reducing_state(self, manager):
        """Set trading to REDUCING state."""
        manager.set_trading_state(TradingState.REDUCING, "daily loss")

        assert manager.trading_state == TradingState.REDUCING


class TestOrderValidation:
    """Tests for order validation."""

    def test_valid_order_passes(self, manager, sample_order):
        """Valid order passes all checks."""
        result = manager.validate_order(sample_order, Decimal("50000"))

        assert result.passed is True
        assert result.check_name == "all_checks"

    def test_halted_state_rejects(self, manager, sample_order):
        """Orders rejected when trading halted."""
        manager.halt_trading(reason="test")

        result = manager.validate_order(sample_order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "trading_state"
        assert "HALTED" in result.reason

    def test_position_limit_exceeded(self, manager):
        """Order rejected when exceeding position limit."""
        # Max BTC position is 1.0
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.5"),  # Exceeds 1.0 limit
        )

        result = manager.validate_order(order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "position_limit"

    def test_notional_limit_exceeded(self, manager):
        """Order rejected when exceeding notional limit."""
        # Max notional for BTC is 50000
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.8"),  # 0.8 * 70000 = 56000 > 50000
        )

        result = manager.validate_order(order, Decimal("70000"))

        assert result.passed is False
        assert result.check_name == "notional_limit"

    def test_rate_limit_exceeded(self, manager, sample_order):
        """Order rejected when rate limit exceeded."""
        # Submit orders up to the limit
        for _ in range(10):
            manager.validate_order(sample_order, Decimal("50000"))

        # Next order should be rate limited
        result = manager.validate_order(sample_order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "rate_limit"

    def test_validation_performance_under_1ms(self, manager, sample_order):
        """All checks complete in under 1ms."""
        # Warm up
        manager.validate_order(sample_order, Decimal("50000"))

        # Time multiple validations
        start = time.perf_counter_ns()
        for _ in range(100):
            manager.validate_order(sample_order, Decimal("50000"))
        elapsed_ns = time.perf_counter_ns() - start

        avg_ns = elapsed_ns / 100
        avg_ms = avg_ns / 1_000_000

        # Should be well under 1ms (typically 50-100μs)
        assert avg_ms < 1.0, f"Average validation time {avg_ms:.3f}ms exceeds 1ms target"


class TestDrawdownMonitoring:
    """Tests for drawdown monitoring."""

    def test_drawdown_calculation(self, manager):
        """Drawdown calculated correctly."""
        manager.update_equity(Decimal("100000"))  # Set peak
        manager.update_equity(Decimal("95000"))  # 5% drawdown

        assert manager.current_drawdown == Decimal("-0.05")

    def test_daily_loss_triggers_reducing(self, manager, sample_order):
        """Daily loss exceeding limit triggers REDUCING state."""
        # Set peak equity first (required for drawdown check)
        manager.update_equity(Decimal("100000"))
        # Set P&L beyond daily limit
        manager.update_pnl(daily_pnl=Decimal("-0.04"))  # Exceeds -3% limit

        result = manager.validate_order(sample_order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "daily_loss"
        assert manager.trading_state == TradingState.REDUCING

    def test_max_drawdown_triggers_halt(self, manager, sample_order):
        """Max drawdown exceeding limit triggers HALTED state."""
        # Set peak and current equity for >15% drawdown
        manager.update_equity(Decimal("100000"))
        manager.update_equity(Decimal("80000"))  # 20% drawdown

        result = manager.validate_order(sample_order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "max_drawdown"
        assert manager.trading_state == TradingState.HALTED


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_status(self, manager):
        """Get circuit breaker status."""
        status = manager.circuit_breaker_status

        assert status["state"] == "closed"
        assert status["is_open"] is False

    def test_consecutive_losses_trip_breaker(self, manager, sample_order):
        """Consecutive losses trip circuit breaker."""
        # Record max consecutive losses (default 10)
        for _ in range(10):
            manager.record_trade_result(profitable=False)

        result = manager.validate_order(sample_order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "circuit_breaker"


class TestPositionTracking:
    """Tests for position tracking."""

    def test_update_position(self, manager):
        """Update tracked position."""
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
        )

        manager.update_position(position)

        assert "BTC/USDT" in manager._positions

    def test_remove_position(self, manager):
        """Remove closed position."""
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
        )

        manager.update_position(position)
        manager.remove_position("BTC/USDT")

        assert "BTC/USDT" not in manager._positions

    def test_reducing_state_allows_position_reduction(self, manager):
        """REDUCING state allows orders that reduce position."""
        # Set up existing long position
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )
        manager.update_position(position)

        # Enter REDUCING state
        manager.set_trading_state(TradingState.REDUCING, "test")

        # SELL order should pass (reduces long position)
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("0.3"),
        )

        result = manager.validate_order(sell_order, Decimal("50000"))
        assert result.passed is True

    def test_reducing_state_blocks_position_increase(self, manager):
        """REDUCING state blocks orders that increase position."""
        # Set up existing long position
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )
        manager.update_position(position)

        # Enter REDUCING state
        manager.set_trading_state(TradingState.REDUCING, "test")

        # BUY order should fail (increases long position)
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.3"),
        )

        result = manager.validate_order(buy_order, Decimal("50000"))
        assert result.passed is False
        assert result.check_name == "trading_state"


class TestMetrics:
    """Tests for risk manager metrics."""

    def test_get_stats(self, manager, sample_order):
        """Get comprehensive stats."""
        # Generate some activity
        manager.update_equity(Decimal("100000"))
        manager.validate_order(sample_order, Decimal("50000"))

        stats = manager.get_stats()

        assert stats["trading_state"] == "active"
        assert stats["orders_checked"] == 1
        assert stats["orders_rejected"] == 0
        assert "current_equity" in stats
        assert "circuit_breaker" in stats
        assert "avg_check_latency_us" in stats

    def test_reset_daily(self, manager):
        """Reset daily tracking."""
        manager.update_pnl(daily_pnl=Decimal("-0.02"))
        manager.reset_daily()

        assert manager._daily_pnl == Decimal("0")

    def test_reset_weekly(self, manager):
        """Reset weekly tracking."""
        manager.update_pnl(
            daily_pnl=Decimal("-0.01"),
            weekly_pnl=Decimal("-0.05"),
        )
        manager.reset_weekly()

        assert manager._weekly_pnl == Decimal("0")


class TestDefaultSymbolLimits:
    """Tests for default symbol limits."""

    def test_unconfigured_symbol_uses_defaults(self, manager):
        """Unconfigured symbols use default limits."""
        order = Order(
            symbol="DOGE/USDT",  # Not configured
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.5"),  # Within default limit of 1.0
        )

        # Should use default limits and pass
        result = manager.validate_order(order, Decimal("0.10"))

        assert result.passed is True


class TestRiskCheckResult:
    """Tests for RiskCheckResult struct."""

    def test_passed_result(self):
        """Create passing result."""
        result = RiskCheckResult(passed=True, check_name="test")

        assert result.passed is True
        assert result.reason is None

    def test_failed_result(self):
        """Create failing result with reason."""
        result = RiskCheckResult(
            passed=False,
            check_name="position_limit",
            reason="Exceeded max position",
        )

        assert result.passed is False
        assert result.reason == "Exceeded max position"

    def test_result_is_frozen(self):
        """Result struct is immutable."""
        result = RiskCheckResult(passed=True, check_name="test")

        with pytest.raises(AttributeError):
            result.passed = False

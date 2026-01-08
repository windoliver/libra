"""Tests for RiskEngine."""

import time
from decimal import Decimal

import pytest

from libra.clients.data_client import Instrument
from libra.gateways.protocol import Order, OrderSide, OrderType, Position, PositionSide
from libra.risk import (
    RiskCheckResult,
    RiskEngine,
    RiskEngineConfig,
    RiskLimits,
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
def config(basic_limits: RiskLimits) -> RiskEngineConfig:
    """RiskEngineConfig for testing."""
    return RiskEngineConfig(
        limits=basic_limits,
        enable_self_trade_prevention=True,
        enable_price_collar=True,
        price_collar_pct=Decimal("0.10"),  # 10%
        enable_precision_validation=True,
        max_modify_rate=20,
    )


@pytest.fixture
def engine(config: RiskEngineConfig) -> RiskEngine:
    """RiskEngine instance for testing."""
    return RiskEngine(config=config, bus=None)


@pytest.fixture
def sample_order() -> Order:
    """Sample order for testing."""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.5"),
    )


@pytest.fixture
def sample_limit_order() -> Order:
    """Sample limit order for testing."""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("0.5"),
        price=Decimal("50000"),
    )


@pytest.fixture
def btc_instrument() -> Instrument:
    """Sample BTC instrument for precision testing."""
    return Instrument(
        symbol="BTC/USDT",
        base="BTC",
        quote="USDT",
        exchange="binance",
        lot_size=Decimal("0.00001"),  # 5 decimal places
        tick_size=Decimal("0.01"),  # 2 decimal places for price
    )


class TestRiskEngineCreation:
    """Tests for RiskEngine initialization."""

    def test_create_engine(self, config):
        """Create RiskEngine with config."""
        engine = RiskEngine(config=config)

        assert engine.config == config
        assert engine.trading_state == TradingState.ACTIVE
        assert engine.is_active is True

    def test_initial_state(self, engine):
        """Engine starts in correct initial state."""
        assert engine.is_active is True
        assert engine.is_halted is False
        assert engine.current_drawdown == Decimal("0")

    def test_config_access(self, engine, basic_limits):
        """Access config and limits through engine."""
        assert engine.limits == basic_limits
        assert engine.config.enable_self_trade_prevention is True
        assert engine.config.enable_price_collar is True


class TestSelfTradePrevention:
    """Tests for self-trade prevention."""

    def test_no_self_trade_without_open_orders(self, engine, sample_limit_order):
        """No self-trade if no open orders."""
        result = engine.validate_order(sample_limit_order, Decimal("50000"))
        assert result.passed is True

    def test_self_trade_detected_buy_vs_sell(self, engine):
        """Detect self-trade: new BUY at price >= existing SELL."""
        # Add open SELL order at 50000
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
            client_order_id="sell-1",
        )
        engine.add_open_order(sell_order)

        # Try to BUY at 50000 or higher - should trigger self-trade
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),  # Same price - would match
        )

        result = engine.validate_order(buy_order, Decimal("50000"))
        assert result.passed is False
        assert result.check_name == "self_trade"

    def test_self_trade_detected_sell_vs_buy(self, engine):
        """Detect self-trade: new SELL at price <= existing BUY."""
        # Add open BUY order at 50000
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
            client_order_id="buy-1",
        )
        engine.add_open_order(buy_order)

        # Try to SELL at 50000 or lower - should trigger self-trade
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("49000"),  # Lower price - would match
        )

        result = engine.validate_order(sell_order, Decimal("50000"))
        assert result.passed is False
        assert result.check_name == "self_trade"

    def test_no_self_trade_same_side(self, engine):
        """No self-trade for same-side orders."""
        # Add open BUY order
        existing_buy = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("49000"),
            client_order_id="buy-1",
        )
        engine.add_open_order(existing_buy)

        # Another BUY is OK
        new_buy = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
        )

        result = engine.validate_order(new_buy, Decimal("50000"))
        assert result.passed is True

    def test_no_self_trade_prices_dont_cross(self, engine):
        """No self-trade if prices don't cross."""
        # Add open SELL at 55000
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("55000"),
            client_order_id="sell-1",
        )
        engine.add_open_order(sell_order)

        # BUY at 50000 won't match SELL at 55000
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
        )

        result = engine.validate_order(buy_order, Decimal("52000"))
        assert result.passed is True

    def test_market_order_self_trade(self, engine):
        """Market orders always risk self-trade if opposite side exists."""
        # Add open SELL order
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
            client_order_id="sell-1",
        )
        engine.add_open_order(sell_order)

        # Market BUY would match
        market_buy = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.5"),
        )

        result = engine.validate_order(market_buy, Decimal("50000"))
        assert result.passed is False
        assert result.check_name == "self_trade"

    def test_remove_open_order(self, engine):
        """Remove open order clears self-trade concern."""
        # Add and then remove open order
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
            client_order_id="sell-1",
        )
        engine.add_open_order(sell_order)
        engine.remove_open_order("BTC/USDT", client_order_id="sell-1")

        # Now BUY should pass
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
        )

        result = engine.validate_order(buy_order, Decimal("50000"))
        assert result.passed is True


class TestPricePrecisionValidation:
    """Tests for price precision validation."""

    def test_valid_price_precision(self, engine, btc_instrument):
        """Price with correct precision passes."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000.01"),  # Valid - multiple of 0.01
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)
        assert result.passed is True

    def test_invalid_price_precision(self, engine, btc_instrument):
        """Price with wrong precision fails."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000.001"),  # Invalid - 3 decimal places
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)
        assert result.passed is False
        assert result.check_name == "price_precision"

    def test_market_order_skips_price_precision(self, engine, btc_instrument):
        """Market orders don't need price precision check."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.5"),
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)
        assert result.passed is True


class TestQuantityPrecisionValidation:
    """Tests for quantity precision validation."""

    def test_valid_quantity_precision(self, engine, btc_instrument):
        """Quantity with correct precision passes."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.50000"),  # Valid - multiple of 0.00001
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)
        assert result.passed is True

    def test_invalid_quantity_precision(self, engine, btc_instrument):
        """Quantity with wrong precision fails."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.500001"),  # Invalid - 6 decimal places
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)
        assert result.passed is False
        assert result.check_name == "quantity_precision"


class TestPriceCollar:
    """Tests for price collar (fat-finger protection)."""

    def test_price_within_collar(self, engine):
        """Price within collar passes."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("52000"),  # 4% above market
        )

        result = engine.validate_order(order, Decimal("50000"))  # Market at 50000
        assert result.passed is True

    def test_price_outside_collar(self, engine):
        """Price outside collar fails."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("60000"),  # 20% above market - exceeds 10% collar
        )

        result = engine.validate_order(order, Decimal("50000"))
        assert result.passed is False
        assert result.check_name == "price_collar"

    def test_price_collar_below_market(self, engine):
        """Price collar also applies below market."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("40000"),  # 20% below market
        )

        result = engine.validate_order(order, Decimal("50000"))
        assert result.passed is False
        assert result.check_name == "price_collar"

    def test_market_order_skips_price_collar(self, engine, sample_order):
        """Market orders don't have price collar check."""
        result = engine.validate_order(sample_order, Decimal("50000"))
        assert result.passed is True


class TestModifyOrderValidation:
    """Tests for order modification validation."""

    def test_modify_passes_all_checks(self, engine, btc_instrument):
        """Valid modification passes."""
        result = engine.validate_modify(
            order_id="order-1",
            new_price=Decimal("51000"),
            new_amount=Decimal("0.5"),
            current_price=Decimal("50000"),
            instrument=btc_instrument,
        )
        assert result.passed is True

    def test_modify_rate_limit(self, engine, btc_instrument):
        """Modification rate limit enforced."""
        # Exhaust rate limit (20/sec with 40 burst capacity)
        for _ in range(40):
            engine.validate_modify("order-1", current_price=Decimal("50000"))

        # Next should fail
        result = engine.validate_modify("order-1", current_price=Decimal("50000"))
        assert result.passed is False
        assert result.check_name == "modify_rate_limit"

    def test_modify_blocked_when_halted(self, engine):
        """Modifications blocked when trading halted."""
        engine.halt_trading("test")

        result = engine.validate_modify("order-1", new_price=Decimal("51000"))
        assert result.passed is False
        assert result.check_name == "trading_state"

    def test_modify_price_precision(self, engine, btc_instrument):
        """Modified price must have correct precision."""
        result = engine.validate_modify(
            order_id="order-1",
            new_price=Decimal("50000.001"),  # Invalid precision
            current_price=Decimal("50000"),
            instrument=btc_instrument,
        )
        assert result.passed is False
        assert result.check_name == "price_precision"

    def test_modify_price_collar(self, engine, btc_instrument):
        """Modified price must be within collar."""
        result = engine.validate_modify(
            order_id="order-1",
            new_price=Decimal("60000"),  # 20% above market
            current_price=Decimal("50000"),
            instrument=btc_instrument,
        )
        assert result.passed is False
        assert result.check_name == "price_collar"


class TestTradingStateManagement:
    """Tests for trading state transitions."""

    def test_halt_trading(self, engine):
        """Halt trading manually."""
        engine.halt_trading(reason="test")

        assert engine.trading_state == TradingState.HALTED
        assert engine.is_halted is True
        assert engine.is_active is False

    def test_halted_state_denies_orders(self, engine, sample_order):
        """Orders denied when trading halted."""
        engine.halt_trading(reason="test")

        result = engine.validate_order(sample_order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "trading_state"
        assert "HALTED" in result.reason

    def test_resume_trading(self, engine):
        """Resume trading after halt."""
        engine.halt_trading(reason="test")
        engine.resume_trading()

        assert engine.trading_state == TradingState.ACTIVE
        assert engine.is_active is True


class TestCoreRiskChecks:
    """Tests for core risk checks (same as RiskManager)."""

    def test_valid_order_passes(self, engine, sample_order):
        """Valid order passes all checks."""
        result = engine.validate_order(sample_order, Decimal("50000"))

        assert result.passed is True
        assert result.check_name == "all_checks"

    def test_position_limit_exceeded(self, engine):
        """Order rejected when exceeding position limit."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.5"),  # Exceeds 1.0 limit
        )

        result = engine.validate_order(order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "position_limit"

    def test_notional_limit_exceeded(self, engine):
        """Order rejected when exceeding notional limit."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.8"),  # 0.8 * 70000 = 56000 > 50000
        )

        result = engine.validate_order(order, Decimal("70000"))

        assert result.passed is False
        assert result.check_name == "notional_limit"

    def test_rate_limit_exceeded(self, engine, sample_order):
        """Order rejected when rate limit exceeded."""
        # Submit orders up to the limit
        for _ in range(10):
            engine.validate_order(sample_order, Decimal("50000"))

        # Next order should be rate limited
        result = engine.validate_order(sample_order, Decimal("50000"))

        assert result.passed is False
        assert result.check_name == "rate_limit"


class TestMetrics:
    """Tests for risk engine metrics."""

    def test_get_stats(self, engine, sample_order):
        """Get comprehensive stats."""
        engine.update_equity(Decimal("100000"))
        engine.validate_order(sample_order, Decimal("50000"))

        stats = engine.get_stats()

        assert stats["trading_state"] == "active"
        assert stats["orders_checked"] == 1
        assert stats["orders_denied"] == 0
        assert "current_equity" in stats
        assert "circuit_breaker" in stats
        assert "avg_check_latency_us" in stats
        assert "config" in stats
        assert stats["config"]["self_trade_prevention"] is True
        assert stats["config"]["price_collar"] is True

    def test_open_orders_tracked_in_stats(self, engine):
        """Open orders count appears in stats."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
        )
        engine.add_open_order(order)

        stats = engine.get_stats()
        assert stats["open_orders_tracked"] == 1


class TestPerformance:
    """Tests for performance requirements."""

    def test_validation_performance_under_1ms(self, engine, sample_order, btc_instrument):
        """All checks complete in under 1ms."""
        # Warm up
        engine.validate_order(sample_order, Decimal("50000"), btc_instrument)

        # Time multiple validations
        start = time.perf_counter_ns()
        for _ in range(100):
            engine.validate_order(sample_order, Decimal("50000"), btc_instrument)
        elapsed_ns = time.perf_counter_ns() - start

        avg_ns = elapsed_ns / 100
        avg_ms = avg_ns / 1_000_000

        # Should be well under 1ms (typically 50-100us)
        assert avg_ms < 1.0, f"Average validation time {avg_ms:.3f}ms exceeds 1ms target"


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
            check_name="self_trade",
            reason="Would self-trade",
        )

        assert result.passed is False
        assert result.reason == "Would self-trade"

    def test_result_is_frozen(self):
        """Result struct is immutable."""
        result = RiskCheckResult(passed=True, check_name="test")

        with pytest.raises(AttributeError):
            result.passed = False

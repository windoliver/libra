"""Tests for instrument min/max quantity validation (Issue #113)."""

from decimal import Decimal

import pytest

from libra.clients.data_client import Instrument
from libra.gateways.protocol import Order, OrderSide, OrderType
from libra.risk import RiskEngine, RiskEngineConfig, RiskLimits, SymbolLimits


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_limits() -> RiskLimits:
    """Basic risk limits for testing."""
    return RiskLimits(
        max_total_exposure=Decimal("100000"),
        max_single_position_pct=Decimal("0.20"),
        max_daily_loss_pct=Decimal("-0.03"),
        max_weekly_loss_pct=Decimal("-0.07"),
        max_total_drawdown_pct=Decimal("-0.15"),
        max_orders_per_second=100,  # High for testing
        max_orders_per_minute=600,
        symbol_limits={
            "BTC/USDT": SymbolLimits(
                max_position_size=Decimal("100.0"),
                max_notional_per_order=Decimal("5000000"),
                max_order_rate=100,
            ),
        },
    )


@pytest.fixture
def config(basic_limits: RiskLimits) -> RiskEngineConfig:
    """RiskEngineConfig with quantity limits enabled."""
    return RiskEngineConfig(
        limits=basic_limits,
        enable_self_trade_prevention=False,  # Disable for focused testing
        enable_price_collar=False,
        enable_precision_validation=False,
        enable_quantity_limits_check=True,  # Enable quantity limits
    )


@pytest.fixture
def engine(config: RiskEngineConfig) -> RiskEngine:
    """RiskEngine instance for testing."""
    return RiskEngine(config=config, bus=None)


@pytest.fixture
def btc_instrument() -> Instrument:
    """BTC instrument with min/max quantity limits."""
    return Instrument(
        symbol="BTC/USDT",
        base="BTC",
        quote="USDT",
        exchange="binance",
        lot_size=Decimal("0.00001"),
        tick_size=Decimal("0.01"),
        min_quantity=Decimal("0.0001"),  # Min: 0.0001 BTC
        max_quantity=Decimal("100.0"),  # Max: 100 BTC
    )


@pytest.fixture
def eth_instrument_no_limits() -> Instrument:
    """ETH instrument without quantity limits (None values)."""
    return Instrument(
        symbol="ETH/USDT",
        base="ETH",
        quote="USDT",
        exchange="binance",
        lot_size=Decimal("0.0001"),
        tick_size=Decimal("0.01"),
        min_quantity=None,  # No min limit
        max_quantity=None,  # No max limit
    )


@pytest.fixture
def small_quantity_instrument() -> Instrument:
    """Instrument with very small quantity limits (for testing edge cases)."""
    return Instrument(
        symbol="SHIB/USDT",
        base="SHIB",
        quote="USDT",
        exchange="binance",
        lot_size=Decimal("1"),
        tick_size=Decimal("0.00000001"),
        min_quantity=Decimal("100000"),  # Min: 100,000 SHIB
        max_quantity=Decimal("10000000000"),  # Max: 10B SHIB
    )


# =============================================================================
# Tests for Minimum Quantity
# =============================================================================


class TestMinQuantityValidation:
    """Tests for minimum quantity validation."""

    def test_order_below_min_quantity_rejected(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Order below minimum quantity is rejected."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.00005"),  # Below min 0.0001
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is False
        assert result.check_name == "min_quantity"
        assert "below minimum" in result.reason
        assert "0.0001" in result.reason

    def test_order_at_min_quantity_passes(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Order exactly at minimum quantity passes."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.0001"),  # Exactly at min
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is True

    def test_order_above_min_quantity_passes(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Order above minimum quantity passes."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),  # Well above min
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is True

    def test_sell_order_below_min_quantity_rejected(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Sell order below minimum quantity is also rejected."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("0.00001"),  # Below min
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is False
        assert result.check_name == "min_quantity"


# =============================================================================
# Tests for Maximum Quantity
# =============================================================================


class TestMaxQuantityValidation:
    """Tests for maximum quantity validation."""

    def test_order_above_max_quantity_rejected(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Order above maximum quantity is rejected."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("150.0"),  # Above max 100.0
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is False
        assert result.check_name == "max_quantity"
        assert "exceeds maximum" in result.reason
        assert "100" in result.reason

    def test_order_at_max_quantity_passes(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Order exactly at maximum quantity passes."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100.0"),  # Exactly at max
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is True

    def test_order_below_max_quantity_passes(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Order below maximum quantity passes."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("50.0"),  # Below max
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is True


# =============================================================================
# Tests for No Limits (None values)
# =============================================================================


class TestNoQuantityLimits:
    """Tests when min/max quantity are None."""

    def test_no_min_limit_allows_small_quantity(
        self, engine: RiskEngine, eth_instrument_no_limits: Instrument
    ) -> None:
        """Without min_quantity, very small orders pass."""
        order = Order(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.000001"),  # Very small
        )

        result = engine.validate_order(order, Decimal("3000"), eth_instrument_no_limits)

        assert result.passed is True

    def test_no_max_limit_allows_large_quantity(
        self, engine: RiskEngine, eth_instrument_no_limits: Instrument
    ) -> None:
        """Without max_quantity, very large orders pass (may fail other checks)."""
        # Note: Need to update symbol limits for ETH/USDT
        engine.config.limits.symbol_limits["ETH/USDT"] = SymbolLimits(
            max_position_size=Decimal("100000000"),
            max_notional_per_order=Decimal("100000000000"),
            max_order_rate=100,
        )

        order = Order(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1000000"),  # Very large
        )

        result = engine.validate_order(order, Decimal("3000"), eth_instrument_no_limits)

        # Should pass quantity limits (may fail position limits)
        # We're specifically testing quantity_limits check passes
        if not result.passed:
            # If it fails, it should NOT be due to quantity limits
            assert result.check_name not in ("min_quantity", "max_quantity")


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestQuantityLimitsEdgeCases:
    """Edge case tests for quantity limits."""

    def test_zero_quantity_fails_min_check(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Zero quantity order fails minimum check."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0"),
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is False
        assert result.check_name == "min_quantity"

    def test_very_small_decimal_quantity(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Very small decimal quantity is properly compared."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.00009999"),  # Just below 0.0001
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is False
        assert result.check_name == "min_quantity"

    def test_large_quantity_with_decimals(
        self, engine: RiskEngine, btc_instrument: Instrument
    ) -> None:
        """Large quantity with decimal precision handled correctly."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100.00001"),  # Just above max 100.0
        )

        result = engine.validate_order(order, Decimal("50000"), btc_instrument)

        assert result.passed is False
        assert result.check_name == "max_quantity"

    def test_high_volume_token_limits(
        self, engine: RiskEngine, small_quantity_instrument: Instrument
    ) -> None:
        """High-volume tokens with large quantity limits work correctly."""
        # Update symbol limits
        engine.config.limits.symbol_limits["SHIB/USDT"] = SymbolLimits(
            max_position_size=Decimal("100000000000"),
            max_notional_per_order=Decimal("1000000"),
            max_order_rate=100,
        )

        # Order below minimum (100,000 SHIB)
        order = Order(
            symbol="SHIB/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("50000"),  # Below 100,000 min
        )

        result = engine.validate_order(
            order, Decimal("0.00001"), small_quantity_instrument
        )

        assert result.passed is False
        assert result.check_name == "min_quantity"


# =============================================================================
# Tests for Config Toggle
# =============================================================================


class TestQuantityLimitsConfigToggle:
    """Tests for enabling/disabling quantity limits check."""

    def test_quantity_limits_disabled(self, basic_limits: RiskLimits) -> None:
        """When quantity limits check is disabled, orders bypass the check."""
        config = RiskEngineConfig(
            limits=basic_limits,
            enable_self_trade_prevention=False,
            enable_price_collar=False,
            enable_precision_validation=False,
            enable_quantity_limits_check=False,  # Disabled
        )
        engine = RiskEngine(config=config, bus=None)

        instrument = Instrument(
            symbol="BTC/USDT",
            base="BTC",
            quote="USDT",
            exchange="binance",
            lot_size=Decimal("0.00001"),
            tick_size=Decimal("0.01"),
            min_quantity=Decimal("1.0"),  # Min 1 BTC
            max_quantity=Decimal("10.0"),  # Max 10 BTC
        )

        # Order below min should pass when check is disabled
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.001"),  # Below min 1.0
        )

        result = engine.validate_order(order, Decimal("50000"), instrument)

        # Should pass (or fail on other checks, but NOT min_quantity)
        if not result.passed:
            assert result.check_name != "min_quantity"
            assert result.check_name != "max_quantity"


# =============================================================================
# Tests for Instrument Without Limits in RiskEngine
# =============================================================================


class TestInstrumentNotProvided:
    """Tests when instrument is not provided to validate_order."""

    def test_no_instrument_skips_quantity_check(
        self, engine: RiskEngine
    ) -> None:
        """Without instrument, quantity limits check is skipped."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.0000001"),  # Would fail if checked
        )

        # No instrument provided
        result = engine.validate_order(order, Decimal("50000"), instrument=None)

        # Should pass (or fail on other checks, but NOT quantity limits)
        if not result.passed:
            assert result.check_name not in ("min_quantity", "max_quantity", "quantity_limits")


# =============================================================================
# Tests for Stats Reporting
# =============================================================================


class TestQuantityLimitsStats:
    """Tests for stats reporting."""

    def test_stats_include_quantity_limits_config(self, engine: RiskEngine) -> None:
        """Stats include quantity_limits_check config."""
        stats = engine.get_stats()

        assert "config" in stats
        assert "quantity_limits_check" in stats["config"]
        assert stats["config"]["quantity_limits_check"] is True

"""
Tests for InstrumentStatus events and RiskEngine integration (Issue #110).
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from libra.gateways.protocol import (
    HaltReason,
    InstrumentStatus,
    InstrumentStatusEvent,
    Order,
    OrderSide,
    OrderType,
)
from libra.risk.engine import RiskEngine, RiskEngineConfig
from libra.risk.limits import RiskLimits


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def risk_engine():
    """Create a RiskEngine for testing."""
    # Use high limits to ensure tests focus on instrument status
    limits = RiskLimits(
        default_max_position_size=Decimal("1000"),
        default_max_notional_per_order=Decimal("1000000"),
    )
    config = RiskEngineConfig(
        limits=limits,
        enable_instrument_status_check=True,
        enable_self_trade_prevention=False,  # Simplify tests
        enable_price_collar=False,  # Simplify tests
    )
    return RiskEngine(config=config)


@pytest.fixture
def test_order():
    """Create a test order."""
    return Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("10"),
        price=Decimal("150.00"),
        client_order_id="test-order-1",
    )


# =============================================================================
# InstrumentStatus Enum Tests
# =============================================================================


class TestInstrumentStatus:
    """Tests for InstrumentStatus enum."""

    def test_status_values(self):
        """Test enum values."""
        assert InstrumentStatus.PRE_OPEN.value == "pre_open"
        assert InstrumentStatus.OPEN.value == "open"
        assert InstrumentStatus.PAUSE.value == "pause"
        assert InstrumentStatus.HALT.value == "halt"
        assert InstrumentStatus.CLOSE.value == "close"
        assert InstrumentStatus.POST_CLOSE.value == "post_close"
        assert InstrumentStatus.NOT_AVAILABLE.value == "not_available"
        assert InstrumentStatus.DELISTED.value == "delisted"


class TestHaltReason:
    """Tests for HaltReason enum."""

    def test_halt_reason_values(self):
        """Test halt reason values."""
        assert HaltReason.CIRCUIT_BREAKER.value == "circuit_breaker"
        assert HaltReason.VOLATILITY.value == "volatility"
        assert HaltReason.NEWS_PENDING.value == "news_pending"
        assert HaltReason.REGULATORY.value == "regulatory"
        assert HaltReason.LULD.value == "luld"


# =============================================================================
# InstrumentStatusEvent Tests
# =============================================================================


class TestInstrumentStatusEvent:
    """Tests for InstrumentStatusEvent."""

    def test_create_halt_event(self):
        """Test creating a halt event."""
        event = InstrumentStatusEvent(
            symbol="AAPL",
            status=InstrumentStatus.HALT,
            timestamp_ns=time.time_ns(),
            previous_status=InstrumentStatus.OPEN,
            reason=HaltReason.NEWS_PENDING,
            halt_reason_text="Pending news announcement",
        )

        assert event.symbol == "AAPL"
        assert event.status == InstrumentStatus.HALT
        assert event.previous_status == InstrumentStatus.OPEN
        assert event.reason == HaltReason.NEWS_PENDING
        assert event.halt_reason_text == "Pending news announcement"

    def test_create_close_event(self):
        """Test creating a market close event."""
        event = InstrumentStatusEvent(
            symbol="BTC/USDT",
            status=InstrumentStatus.CLOSE,
            timestamp_ns=time.time_ns(),
        )

        assert event.symbol == "BTC/USDT"
        assert event.status == InstrumentStatus.CLOSE
        assert event.previous_status is None
        assert event.reason is None

    def test_is_tradeable(self):
        """Test is_tradeable property."""
        # Tradeable statuses
        for status in [InstrumentStatus.OPEN, InstrumentStatus.PRE_OPEN, InstrumentStatus.POST_CLOSE]:
            event = InstrumentStatusEvent(
                symbol="AAPL",
                status=status,
                timestamp_ns=time.time_ns(),
            )
            assert event.is_tradeable is True, f"{status} should be tradeable"

        # Non-tradeable statuses
        for status in [InstrumentStatus.HALT, InstrumentStatus.PAUSE, InstrumentStatus.CLOSE]:
            event = InstrumentStatusEvent(
                symbol="AAPL",
                status=status,
                timestamp_ns=time.time_ns(),
            )
            assert event.is_tradeable is False, f"{status} should not be tradeable"

    def test_is_halted(self):
        """Test is_halted property."""
        # Halted statuses
        for status in [
            InstrumentStatus.HALT,
            InstrumentStatus.PAUSE,
            InstrumentStatus.NOT_AVAILABLE,
            InstrumentStatus.DELISTED,
        ]:
            event = InstrumentStatusEvent(
                symbol="AAPL",
                status=status,
                timestamp_ns=time.time_ns(),
            )
            assert event.is_halted is True, f"{status} should be halted"

        # Non-halted statuses
        for status in [InstrumentStatus.OPEN, InstrumentStatus.PRE_OPEN, InstrumentStatus.CLOSE]:
            event = InstrumentStatusEvent(
                symbol="AAPL",
                status=status,
                timestamp_ns=time.time_ns(),
            )
            assert event.is_halted is False, f"{status} should not be halted"

    def test_timestamp_sec(self):
        """Test timestamp_sec property."""
        ts_ns = 1000000000000000000  # 1 second in ns
        event = InstrumentStatusEvent(
            symbol="AAPL",
            status=InstrumentStatus.OPEN,
            timestamp_ns=ts_ns,
        )
        assert event.timestamp_sec == 1000000000.0


# =============================================================================
# RiskEngine Instrument Status Tests
# =============================================================================


class TestRiskEngineInstrumentStatus:
    """Tests for RiskEngine instrument status check."""

    def test_order_allowed_no_status_tracked(self, risk_engine, test_order):
        """Test order is allowed when no status is tracked."""
        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

    def test_order_allowed_when_open(self, risk_engine, test_order):
        """Test order is allowed when instrument is OPEN."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.OPEN)

        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

    def test_order_allowed_when_pre_open(self, risk_engine, test_order):
        """Test order is allowed when instrument is PRE_OPEN."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.PRE_OPEN)

        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

    def test_order_allowed_when_post_close(self, risk_engine, test_order):
        """Test order is allowed when instrument is POST_CLOSE."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.POST_CLOSE)

        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

    def test_order_rejected_when_halted(self, risk_engine, test_order):
        """Test order is rejected when instrument is HALT."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.HALT)

        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is False
        assert result.check_name == "instrument_status"
        assert "halt" in result.reason.lower()

    def test_order_rejected_when_paused(self, risk_engine, test_order):
        """Test order is rejected when instrument is PAUSE."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.PAUSE)

        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is False
        assert result.check_name == "instrument_status"
        assert "pause" in result.reason.lower()

    def test_order_rejected_when_closed(self, risk_engine, test_order):
        """Test order is rejected when instrument is CLOSE."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.CLOSE)

        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is False
        assert result.check_name == "instrument_status"
        assert "close" in result.reason.lower()

    def test_order_rejected_when_not_available(self, risk_engine, test_order):
        """Test order is rejected when instrument is NOT_AVAILABLE."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.NOT_AVAILABLE)

        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is False
        assert result.check_name == "instrument_status"
        assert "not_available" in result.reason.lower()

    def test_order_rejected_when_delisted(self, risk_engine, test_order):
        """Test order is rejected when instrument is DELISTED."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.DELISTED)

        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is False
        assert result.check_name == "instrument_status"
        assert "delisted" in result.reason.lower()

    def test_status_check_can_be_disabled(self, test_order):
        """Test that instrument status check can be disabled."""
        config = RiskEngineConfig(
            limits=RiskLimits(),
            enable_instrument_status_check=False,
        )
        engine = RiskEngine(config=config)

        # Set halted status
        engine.set_instrument_status("AAPL", InstrumentStatus.HALT)

        # Order should still be allowed
        result = engine.validate_order(test_order, current_price=Decimal("150.00"))
        # May fail other checks, but not instrument_status
        assert result.check_name != "instrument_status" or result.passed is True

    def test_different_symbols_have_independent_status(self, risk_engine):
        """Test that different symbols have independent status."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.HALT)
        risk_engine.set_instrument_status("GOOGL", InstrumentStatus.OPEN)

        # AAPL order should be rejected
        aapl_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("10"),
        )
        result = risk_engine.validate_order(aapl_order, current_price=Decimal("150.00"))
        assert result.passed is False
        assert result.check_name == "instrument_status"

        # GOOGL order should be allowed
        googl_order = Order(
            symbol="GOOGL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("10"),
        )
        result = risk_engine.validate_order(googl_order, current_price=Decimal("150.00"))
        assert result.passed is True


class TestRiskEngineStatusUpdate:
    """Tests for RiskEngine status update methods."""

    def test_set_instrument_status(self, risk_engine):
        """Test setting instrument status directly."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.HALT)

        assert risk_engine.get_instrument_status("AAPL") == InstrumentStatus.HALT

    def test_get_instrument_status_unknown(self, risk_engine):
        """Test getting status for unknown symbol."""
        assert risk_engine.get_instrument_status("UNKNOWN") is None

    def test_update_instrument_status_from_event(self, risk_engine):
        """Test updating status from InstrumentStatusEvent."""
        event = InstrumentStatusEvent(
            symbol="AAPL",
            status=InstrumentStatus.HALT,
            timestamp_ns=time.time_ns(),
            reason=HaltReason.NEWS_PENDING,
            halt_reason_text="Pending earnings announcement",
        )

        risk_engine.update_instrument_status(event)

        assert risk_engine.get_instrument_status("AAPL") == InstrumentStatus.HALT

    def test_clear_instrument_status_single(self, risk_engine):
        """Test clearing status for a single symbol."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.HALT)
        risk_engine.set_instrument_status("GOOGL", InstrumentStatus.PAUSE)

        risk_engine.clear_instrument_status("AAPL")

        assert risk_engine.get_instrument_status("AAPL") is None
        assert risk_engine.get_instrument_status("GOOGL") == InstrumentStatus.PAUSE

    def test_clear_instrument_status_all(self, risk_engine):
        """Test clearing all instrument statuses."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.HALT)
        risk_engine.set_instrument_status("GOOGL", InstrumentStatus.PAUSE)

        risk_engine.clear_instrument_status()

        assert risk_engine.get_instrument_status("AAPL") is None
        assert risk_engine.get_instrument_status("GOOGL") is None

    def test_stats_include_instrument_status(self, risk_engine):
        """Test that get_stats includes instrument status info."""
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.HALT)
        risk_engine.set_instrument_status("GOOGL", InstrumentStatus.OPEN)
        risk_engine.set_instrument_status("MSFT", InstrumentStatus.PAUSE)

        stats = risk_engine.get_stats()

        assert stats["instruments_tracked"] == 3
        assert "AAPL" in stats["halted_instruments"]
        assert "MSFT" in stats["halted_instruments"]
        assert "GOOGL" not in stats["halted_instruments"]
        assert stats["config"]["instrument_status_check"] is True


class TestInstrumentStatusTransitions:
    """Tests for instrument status transitions."""

    def test_halt_to_resume_transition(self, risk_engine, test_order):
        """Test halting and resuming an instrument."""
        # Initially open
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.OPEN)
        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

        # Halt
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.HALT)
        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is False

        # Resume
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.OPEN)
        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

    def test_session_transition(self, risk_engine, test_order):
        """Test market session transitions."""
        # Pre-market
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.PRE_OPEN)
        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

        # Market open
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.OPEN)
        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

        # Market close
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.CLOSE)
        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is False

        # Post-market
        risk_engine.set_instrument_status("AAPL", InstrumentStatus.POST_CLOSE)
        result = risk_engine.validate_order(test_order, current_price=Decimal("150.00"))
        assert result.passed is True

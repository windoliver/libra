"""
Unit tests for Option Models.

Tests OptionType, OptionStyle, OptionContract, and OptionPosition.

Issue #63: Options Data Models
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from libra.core.options import (
    OptionContract,
    OptionPosition,
    OptionStyle,
    OptionType,
    decode_option_contract,
    encode_option_contract,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestOptionType:
    """Tests for OptionType enum."""

    def test_values(self) -> None:
        """Option types have correct values."""
        assert OptionType.CALL.value == "call"
        assert OptionType.PUT.value == "put"

    def test_string_serialization(self) -> None:
        """Option types serialize to strings."""
        assert str(OptionType.CALL) == "OptionType.CALL"
        assert OptionType.CALL == "call"


class TestOptionStyle:
    """Tests for OptionStyle enum."""

    def test_values(self) -> None:
        """Option styles have correct values."""
        assert OptionStyle.AMERICAN.value == "american"
        assert OptionStyle.EUROPEAN.value == "european"


# =============================================================================
# OptionContract Tests
# =============================================================================


class TestOptionContract:
    """Tests for OptionContract struct."""

    @pytest.fixture
    def call_contract(self) -> OptionContract:
        """Create a test call contract."""
        return OptionContract(
            symbol="AAPL250117C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("150.00"),
            expiration=date(2025, 1, 17),
        )

    @pytest.fixture
    def put_contract(self) -> OptionContract:
        """Create a test put contract."""
        return OptionContract(
            symbol="AAPL250117P00150000",
            underlying="AAPL",
            option_type=OptionType.PUT,
            strike=Decimal("150.00"),
            expiration=date(2025, 1, 17),
        )

    def test_contract_creation(self, call_contract: OptionContract) -> None:
        """Contract fields are set correctly."""
        assert call_contract.symbol == "AAPL250117C00150000"
        assert call_contract.underlying == "AAPL"
        assert call_contract.option_type == OptionType.CALL
        assert call_contract.strike == Decimal("150.00")
        assert call_contract.expiration == date(2025, 1, 17)
        assert call_contract.style == OptionStyle.AMERICAN
        assert call_contract.multiplier == 100
        assert call_contract.exchange is None

    def test_is_call_property(
        self, call_contract: OptionContract, put_contract: OptionContract
    ) -> None:
        """is_call returns correct value."""
        assert call_contract.is_call is True
        assert call_contract.is_put is False
        assert put_contract.is_call is False
        assert put_contract.is_put is True

    def test_intrinsic_value_call_itm(self, call_contract: OptionContract) -> None:
        """Call intrinsic value when ITM."""
        # Strike = 150, underlying = 160 -> intrinsic = 10
        value = call_contract.intrinsic_value(Decimal("160.00"))
        assert value == Decimal("10.00")

    def test_intrinsic_value_call_otm(self, call_contract: OptionContract) -> None:
        """Call intrinsic value when OTM."""
        # Strike = 150, underlying = 140 -> intrinsic = 0
        value = call_contract.intrinsic_value(Decimal("140.00"))
        assert value == Decimal("0")

    def test_intrinsic_value_put_itm(self, put_contract: OptionContract) -> None:
        """Put intrinsic value when ITM."""
        # Strike = 150, underlying = 140 -> intrinsic = 10
        value = put_contract.intrinsic_value(Decimal("140.00"))
        assert value == Decimal("10.00")

    def test_intrinsic_value_put_otm(self, put_contract: OptionContract) -> None:
        """Put intrinsic value when OTM."""
        # Strike = 150, underlying = 160 -> intrinsic = 0
        value = put_contract.intrinsic_value(Decimal("160.00"))
        assert value == Decimal("0")

    def test_is_itm(self, call_contract: OptionContract) -> None:
        """is_itm returns True when in-the-money."""
        assert call_contract.is_itm(Decimal("160.00")) is True
        assert call_contract.is_itm(Decimal("140.00")) is False
        assert call_contract.is_itm(Decimal("150.00")) is False  # ATM is not ITM

    def test_is_otm(self, call_contract: OptionContract) -> None:
        """is_otm returns True when out-of-the-money."""
        assert call_contract.is_otm(Decimal("140.00")) is True
        assert call_contract.is_otm(Decimal("160.00")) is False
        assert call_contract.is_otm(Decimal("150.00")) is True  # ATM is OTM

    def test_is_atm(self, call_contract: OptionContract) -> None:
        """is_atm returns True when at-the-money within tolerance."""
        # Exactly at strike
        assert call_contract.is_atm(Decimal("150.00")) is True
        # Within 1% tolerance
        assert call_contract.is_atm(Decimal("151.00")) is True
        # Outside 1% tolerance
        assert call_contract.is_atm(Decimal("155.00")) is False

    def test_is_atm_zero_price(self, call_contract: OptionContract) -> None:
        """is_atm handles zero price."""
        assert call_contract.is_atm(Decimal("0")) is False

    def test_days_to_expiry(self) -> None:
        """days_to_expiry calculated correctly."""
        # Create contract with known future date
        from datetime import timedelta

        future_date = date.today() + timedelta(days=30)
        contract = OptionContract(
            symbol="TEST",
            underlying="TEST",
            option_type=OptionType.CALL,
            strike=Decimal("100"),
            expiration=future_date,
        )
        assert contract.days_to_expiry == 30

    def test_is_expired(self) -> None:
        """is_expired correctly detects expired contracts."""
        # Past date
        past_contract = OptionContract(
            symbol="TEST",
            underlying="TEST",
            option_type=OptionType.CALL,
            strike=Decimal("100"),
            expiration=date(2020, 1, 1),
        )
        assert past_contract.is_expired is True

        # Future date
        future_contract = OptionContract(
            symbol="TEST",
            underlying="TEST",
            option_type=OptionType.CALL,
            strike=Decimal("100"),
            expiration=date(2030, 1, 1),
        )
        assert future_contract.is_expired is False

    def test_immutability(self, call_contract: OptionContract) -> None:
        """Contract is immutable."""
        with pytest.raises(AttributeError):
            call_contract.strike = Decimal("200.00")  # type: ignore

    def test_serialization_roundtrip(self, call_contract: OptionContract) -> None:
        """Contract can be serialized and deserialized."""
        encoded = encode_option_contract(call_contract)
        decoded = decode_option_contract(encoded)
        assert decoded.symbol == call_contract.symbol
        assert decoded.strike == call_contract.strike
        assert decoded.option_type == call_contract.option_type


# =============================================================================
# OptionPosition Tests
# =============================================================================


class TestOptionPosition:
    """Tests for OptionPosition struct."""

    @pytest.fixture
    def contract(self) -> OptionContract:
        """Create a test contract."""
        return OptionContract(
            symbol="AAPL250117C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("150.00"),
            expiration=date(2025, 1, 17),
        )

    @pytest.fixture
    def long_position(self, contract: OptionContract) -> OptionPosition:
        """Create a long position."""
        return OptionPosition(
            contract=contract,
            quantity=10,
            avg_price=Decimal("5.00"),
            current_price=Decimal("6.00"),
            opened_at=datetime(2025, 1, 1, 10, 0, 0),
        )

    @pytest.fixture
    def short_position(self, contract: OptionContract) -> OptionPosition:
        """Create a short position."""
        return OptionPosition(
            contract=contract,
            quantity=-10,
            avg_price=Decimal("5.00"),
            current_price=Decimal("6.00"),
            opened_at=datetime(2025, 1, 1, 10, 0, 0),
        )

    def test_is_long(
        self, long_position: OptionPosition, short_position: OptionPosition
    ) -> None:
        """is_long returns correct value."""
        assert long_position.is_long is True
        assert long_position.is_short is False
        assert short_position.is_long is False
        assert short_position.is_short is True

    def test_market_value_long(self, long_position: OptionPosition) -> None:
        """Market value for long position."""
        # 10 contracts * $6.00 * 100 multiplier = $6000
        assert long_position.market_value == Decimal("6000.00")

    def test_market_value_short(self, short_position: OptionPosition) -> None:
        """Market value for short position (negative)."""
        # -10 contracts * $6.00 * 100 multiplier = -$6000
        assert short_position.market_value == Decimal("-6000.00")

    def test_cost_basis_long(self, long_position: OptionPosition) -> None:
        """Cost basis for long position."""
        # 10 contracts * $5.00 * 100 multiplier = $5000
        assert long_position.cost_basis == Decimal("5000.00")

    def test_unrealized_pnl_profit(self, long_position: OptionPosition) -> None:
        """Unrealized P&L when profitable."""
        # Market value $6000 - Cost $5000 = $1000 profit
        assert long_position.unrealized_pnl == Decimal("1000.00")

    def test_unrealized_pnl_loss(self, contract: OptionContract) -> None:
        """Unrealized P&L when losing."""
        position = OptionPosition(
            contract=contract,
            quantity=10,
            avg_price=Decimal("6.00"),
            current_price=Decimal("5.00"),
            opened_at=datetime(2025, 1, 1),
        )
        # Market value $5000 - Cost $6000 = -$1000 loss
        assert position.unrealized_pnl == Decimal("-1000.00")

    def test_unrealized_pnl_pct(self, long_position: OptionPosition) -> None:
        """Unrealized P&L percentage."""
        # $1000 profit / $5000 cost = 20%
        assert long_position.unrealized_pnl_pct == Decimal("20")

    def test_unrealized_pnl_pct_zero_cost(self, contract: OptionContract) -> None:
        """Unrealized P&L pct handles zero cost basis."""
        position = OptionPosition(
            contract=contract,
            quantity=0,
            avg_price=Decimal("0"),
            current_price=Decimal("5.00"),
            opened_at=datetime(2025, 1, 1),
        )
        assert position.unrealized_pnl_pct == Decimal("0")

    def test_with_price(self, long_position: OptionPosition) -> None:
        """with_price creates new position with updated price."""
        new_position = long_position.with_price(Decimal("7.00"))
        assert new_position.current_price == Decimal("7.00")
        assert new_position.quantity == long_position.quantity
        assert new_position.avg_price == long_position.avg_price
        assert new_position.updated_at is not None
        # Original unchanged
        assert long_position.current_price == Decimal("6.00")

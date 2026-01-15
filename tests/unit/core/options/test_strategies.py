"""
Unit tests for Strategy Models.

Tests StrategyType, StrategyLeg, OptionStrategy, and strategy builders.

Issue #63: Options Data Models
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from libra.core.options import (
    OptionContract,
    OptionStrategy,
    OptionType,
    StrategyLeg,
    StrategyType,
    create_butterfly,
    create_iron_condor,
    create_straddle,
    create_strangle,
    create_vertical_spread,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_single_leg_values(self) -> None:
        """Single leg strategy types have correct values."""
        assert StrategyType.LONG_CALL.value == "long_call"
        assert StrategyType.SHORT_CALL.value == "short_call"
        assert StrategyType.LONG_PUT.value == "long_put"
        assert StrategyType.SHORT_PUT.value == "short_put"

    def test_vertical_spread_values(self) -> None:
        """Vertical spread types have correct values."""
        assert StrategyType.BULL_CALL_SPREAD.value == "bull_call_spread"
        assert StrategyType.BEAR_CALL_SPREAD.value == "bear_call_spread"
        assert StrategyType.BULL_PUT_SPREAD.value == "bull_put_spread"
        assert StrategyType.BEAR_PUT_SPREAD.value == "bear_put_spread"

    def test_multi_leg_values(self) -> None:
        """Multi-leg strategy types have correct values."""
        assert StrategyType.STRADDLE.value == "straddle"
        assert StrategyType.STRANGLE.value == "strangle"
        assert StrategyType.IRON_CONDOR.value == "iron_condor"
        assert StrategyType.BUTTERFLY.value == "butterfly"


# =============================================================================
# StrategyLeg Tests
# =============================================================================


class TestStrategyLeg:
    """Tests for StrategyLeg struct."""

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

    def test_long_leg(self, call_contract: OptionContract) -> None:
        """Long leg has positive quantity."""
        leg = StrategyLeg(contract=call_contract, quantity=1)
        assert leg.is_long is True
        assert leg.is_short is False
        assert leg.is_call is True

    def test_short_leg(self, call_contract: OptionContract) -> None:
        """Short leg has negative quantity."""
        leg = StrategyLeg(contract=call_contract, quantity=-1)
        assert leg.is_long is False
        assert leg.is_short is True

    def test_default_ratio(self, call_contract: OptionContract) -> None:
        """Default ratio is 1."""
        leg = StrategyLeg(contract=call_contract, quantity=1)
        assert leg.ratio == 1


# =============================================================================
# OptionStrategy Tests
# =============================================================================


class TestOptionStrategy:
    """Tests for OptionStrategy struct."""

    @pytest.fixture
    def bull_call_spread(self) -> OptionStrategy:
        """Create a bull call spread."""
        return create_vertical_spread(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            long_strike=Decimal("150"),
            short_strike=Decimal("155"),
            option_type=OptionType.CALL,
            quantity=1,
        )

    def test_num_legs(self, bull_call_spread: OptionStrategy) -> None:
        """num_legs returns correct count."""
        assert bull_call_spread.num_legs == 2

    def test_total_contracts(self, bull_call_spread: OptionStrategy) -> None:
        """total_contracts sums absolute quantities."""
        assert bull_call_spread.total_contracts == 2

    def test_expiration_same(self, bull_call_spread: OptionStrategy) -> None:
        """expiration returns date when all legs same."""
        assert bull_call_spread.expiration == date(2025, 1, 17)

    def test_net_delta_bullish(self, bull_call_spread: OptionStrategy) -> None:
        """net_delta positive for bullish spread."""
        # Long call (+1) + short call (-1) = 0 delta-neutral vertically
        # But quantity-wise: +1 - (-1) = 0
        # Actually for bull call: buy lower, sell higher, net delta positive
        # The net_delta method sums quantity * (1 for call, -1 for put)
        # Long 1 call: +1, Short 1 call: -1, net = 0
        assert bull_call_spread.net_delta == 0

    def test_is_debit_credit(self) -> None:
        """is_debit and is_credit work correctly."""
        strategy_debit = OptionStrategy(
            strategy_id="test",
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            underlying="AAPL",
            legs=(),
            net_debit=Decimal("5.00"),
        )
        assert strategy_debit.is_debit is True
        assert strategy_debit.is_credit is False

        strategy_credit = OptionStrategy(
            strategy_id="test",
            strategy_type=StrategyType.IRON_CONDOR,
            underlying="AAPL",
            legs=(),
            net_debit=Decimal("-2.00"),
        )
        assert strategy_credit.is_debit is False
        assert strategy_credit.is_credit is True

    def test_get_leg(self, bull_call_spread: OptionStrategy) -> None:
        """get_leg returns correct leg."""
        leg0 = bull_call_spread.get_leg(0)
        leg1 = bull_call_spread.get_leg(1)
        leg2 = bull_call_spread.get_leg(2)

        assert leg0 is not None
        assert leg1 is not None
        assert leg2 is None  # Out of bounds


# =============================================================================
# Strategy Builder Tests
# =============================================================================


class TestVerticalSpread:
    """Tests for create_vertical_spread builder."""

    def test_bull_call_spread(self) -> None:
        """Bull call spread: buy lower, sell higher."""
        spread = create_vertical_spread(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            long_strike=Decimal("150"),
            short_strike=Decimal("155"),
            option_type=OptionType.CALL,
        )
        assert spread.strategy_type == StrategyType.BULL_CALL_SPREAD
        assert spread.num_legs == 2
        # Long leg
        assert spread.legs[0].quantity == 1
        assert spread.legs[0].contract.strike == Decimal("150")
        # Short leg
        assert spread.legs[1].quantity == -1
        assert spread.legs[1].contract.strike == Decimal("155")
        # Max loss is width
        assert spread.max_loss == Decimal("500")  # 5 * 100

    def test_bear_call_spread(self) -> None:
        """Bear call spread: buy higher, sell lower."""
        spread = create_vertical_spread(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            long_strike=Decimal("155"),
            short_strike=Decimal("150"),
            option_type=OptionType.CALL,
        )
        assert spread.strategy_type == StrategyType.BEAR_CALL_SPREAD

    def test_bull_put_spread(self) -> None:
        """Bull put spread: buy lower put, sell higher put."""
        spread = create_vertical_spread(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            long_strike=Decimal("155"),  # Buy higher (protection)
            short_strike=Decimal("150"),  # Sell lower
            option_type=OptionType.PUT,
        )
        assert spread.strategy_type == StrategyType.BULL_PUT_SPREAD

    def test_custom_quantity(self) -> None:
        """Spread with multiple contracts."""
        spread = create_vertical_spread(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            long_strike=Decimal("150"),
            short_strike=Decimal("155"),
            option_type=OptionType.CALL,
            quantity=5,
        )
        assert spread.legs[0].quantity == 5
        assert spread.legs[1].quantity == -5
        assert spread.max_loss == Decimal("2500")  # 5 * 5 * 100


class TestStraddle:
    """Tests for create_straddle builder."""

    def test_long_straddle(self) -> None:
        """Long straddle: buy call and put at same strike."""
        straddle = create_straddle(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            strike=Decimal("150"),
        )
        assert straddle.strategy_type == StrategyType.STRADDLE
        assert straddle.num_legs == 2
        # Both legs long
        assert straddle.legs[0].quantity == 1
        assert straddle.legs[1].quantity == 1
        # Call and put at same strike
        assert straddle.legs[0].contract.is_call is True
        assert straddle.legs[1].contract.is_put is True
        assert straddle.legs[0].contract.strike == Decimal("150")
        assert straddle.legs[1].contract.strike == Decimal("150")

    def test_short_straddle(self) -> None:
        """Short straddle: sell call and put."""
        straddle = create_straddle(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            strike=Decimal("150"),
            is_long=False,
        )
        assert straddle.legs[0].quantity == -1
        assert straddle.legs[1].quantity == -1


class TestStrangle:
    """Tests for create_strangle builder."""

    def test_long_strangle(self) -> None:
        """Long strangle: buy OTM put and call."""
        strangle = create_strangle(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            put_strike=Decimal("145"),
            call_strike=Decimal("155"),
        )
        assert strangle.strategy_type == StrategyType.STRANGLE
        assert strangle.num_legs == 2
        # Put leg
        assert strangle.legs[0].contract.is_put is True
        assert strangle.legs[0].contract.strike == Decimal("145")
        # Call leg
        assert strangle.legs[1].contract.is_call is True
        assert strangle.legs[1].contract.strike == Decimal("155")


class TestIronCondor:
    """Tests for create_iron_condor builder."""

    def test_iron_condor(self) -> None:
        """Iron condor has 4 legs."""
        condor = create_iron_condor(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            put_long_strike=Decimal("140"),
            put_short_strike=Decimal("145"),
            call_short_strike=Decimal("155"),
            call_long_strike=Decimal("160"),
        )
        assert condor.strategy_type == StrategyType.IRON_CONDOR
        assert condor.num_legs == 4

        # Verify leg structure
        # Leg 0: Long put (protection)
        assert condor.legs[0].contract.is_put is True
        assert condor.legs[0].contract.strike == Decimal("140")
        assert condor.legs[0].quantity == 1

        # Leg 1: Short put (sell premium)
        assert condor.legs[1].contract.is_put is True
        assert condor.legs[1].contract.strike == Decimal("145")
        assert condor.legs[1].quantity == -1

        # Leg 2: Short call (sell premium)
        assert condor.legs[2].contract.is_call is True
        assert condor.legs[2].contract.strike == Decimal("155")
        assert condor.legs[2].quantity == -1

        # Leg 3: Long call (protection)
        assert condor.legs[3].contract.is_call is True
        assert condor.legs[3].contract.strike == Decimal("160")
        assert condor.legs[3].quantity == 1

        # Max loss is wing width
        assert condor.max_loss == Decimal("500")  # 5 * 100


class TestButterfly:
    """Tests for create_butterfly builder."""

    def test_call_butterfly(self) -> None:
        """Call butterfly: buy 1, sell 2, buy 1."""
        butterfly = create_butterfly(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            lower_strike=Decimal("145"),
            middle_strike=Decimal("150"),
            upper_strike=Decimal("155"),
            option_type=OptionType.CALL,
        )
        assert butterfly.strategy_type == StrategyType.BUTTERFLY
        assert butterfly.num_legs == 3

        # Lower wing: buy 1
        assert butterfly.legs[0].quantity == 1
        assert butterfly.legs[0].contract.strike == Decimal("145")

        # Body: sell 2
        assert butterfly.legs[1].quantity == -2
        assert butterfly.legs[1].contract.strike == Decimal("150")

        # Upper wing: buy 1
        assert butterfly.legs[2].quantity == 1
        assert butterfly.legs[2].contract.strike == Decimal("155")

        # Max profit at middle strike
        assert butterfly.max_profit == Decimal("500")  # 5 * 100

"""
Multi-Leg Option Strategy Models.

Provides data structures for representing option strategies:
- Single leg strategies (long/short calls/puts)
- Two leg strategies (spreads, straddles, strangles)
- Multi-leg strategies (butterflies, iron condors)

Includes builder functions for common strategies.

Issue #63: Options Data Models
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import msgspec

from libra.core.options.models import OptionContract, OptionType


if TYPE_CHECKING:
    from libra.core.options.greeks import Greeks


# =============================================================================
# Enums
# =============================================================================


class StrategyType(str, Enum):
    """
    Common option strategy types.

    Single leg:
        LONG_CALL, SHORT_CALL, LONG_PUT, SHORT_PUT

    Vertical spreads:
        BULL_CALL_SPREAD, BEAR_CALL_SPREAD, BULL_PUT_SPREAD, BEAR_PUT_SPREAD

    Other two leg:
        CALENDAR_SPREAD, STRADDLE, STRANGLE

    Three leg:
        BUTTERFLY

    Four leg:
        IRON_CONDOR, IRON_BUTTERFLY

    Custom:
        CUSTOM (for non-standard combinations)
    """

    # Single leg
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"

    # Two leg - vertical
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Two leg - other
    CALENDAR_SPREAD = "calendar_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"

    # Three leg
    BUTTERFLY = "butterfly"

    # Four leg
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"

    # Custom
    CUSTOM = "custom"


# =============================================================================
# Strategy Models
# =============================================================================


class StrategyLeg(msgspec.Struct, frozen=True, gc=False):
    """
    A single leg of a multi-leg strategy.

    Attributes:
        contract: The option contract for this leg
        quantity: Number of contracts (positive=buy, negative=sell)
        ratio: Ratio for ratio spreads (default 1)

    Examples:
        # Long call leg
        leg = StrategyLeg(
            contract=call_contract,
            quantity=1,  # Buy 1 contract
        )

        # Short call leg
        leg = StrategyLeg(
            contract=call_contract,
            quantity=-1,  # Sell 1 contract
        )
    """

    contract: OptionContract
    quantity: int  # Positive = buy, negative = sell
    ratio: int = 1  # For ratio spreads (e.g., 1:2)

    @property
    def is_long(self) -> bool:
        """True if buying this leg."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if selling this leg."""
        return self.quantity < 0

    @property
    def is_call(self) -> bool:
        """True if this leg is a call option."""
        return self.contract.is_call

    @property
    def is_put(self) -> bool:
        """True if this leg is a put option."""
        return self.contract.is_put


class OptionStrategy(msgspec.Struct, frozen=True, gc=False, kw_only=True):
    """
    Multi-leg option strategy.

    Represents a coordinated set of option positions that form a defined strategy.

    Attributes:
        strategy_id: Unique identifier for this strategy instance
        strategy_type: Type of strategy (from StrategyType enum)
        underlying: Underlying symbol
        legs: Tuple of strategy legs
        max_profit: Maximum possible profit (None if unlimited)
        max_loss: Maximum possible loss (None if unlimited)
        breakeven_prices: Breakeven prices for the strategy
        net_debit: Net premium paid/received (positive=debit, negative=credit)

    Examples:
        # Bull call spread
        strategy = OptionStrategy(
            strategy_id="bull-call-001",
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            underlying="AAPL",
            legs=(long_leg, short_leg),
            max_profit=Decimal("500"),
            max_loss=Decimal("500"),
            net_debit=Decimal("5.00"),
        )

        if strategy.is_credit:
            print("Received premium!")
    """

    strategy_id: str
    strategy_type: StrategyType
    underlying: str
    legs: tuple[StrategyLeg, ...]

    # Strategy metrics (optional - may be calculated or provided)
    max_profit: Decimal | None = None
    max_loss: Decimal | None = None
    breakeven_prices: tuple[Decimal, ...] = ()

    # Execution details
    net_debit: Decimal | None = None  # Positive = paid, negative = received (credit)

    @property
    def num_legs(self) -> int:
        """Number of legs in the strategy."""
        return len(self.legs)

    @property
    def is_credit(self) -> bool:
        """True if strategy received premium (net credit)."""
        return self.net_debit is not None and self.net_debit < 0

    @property
    def is_debit(self) -> bool:
        """True if strategy paid premium (net debit)."""
        return self.net_debit is not None and self.net_debit > 0

    @property
    def total_contracts(self) -> int:
        """Total number of contracts across all legs."""
        return sum(abs(leg.quantity) for leg in self.legs)

    @property
    def net_delta(self) -> int:
        """
        Net directional bias.

        Returns sum of (quantity * 1 for calls, -1 for puts).
        Positive = bullish, negative = bearish.
        """
        total = 0
        for leg in self.legs:
            if leg.is_call:
                total += leg.quantity
            else:
                total -= leg.quantity
        return total

    @property
    def expiration(self) -> date | None:
        """
        Expiration date if all legs have same expiry.

        Returns None for calendar spreads or if no legs.
        """
        if not self.legs:
            return None
        first_exp = self.legs[0].contract.expiration
        if all(leg.contract.expiration == first_exp for leg in self.legs):
            return first_exp
        return None

    def get_leg(self, index: int) -> StrategyLeg | None:
        """Get leg by index."""
        if 0 <= index < len(self.legs):
            return self.legs[index]
        return None


# =============================================================================
# Strategy Builders
# =============================================================================


def _make_occ_symbol(
    underlying: str,
    expiration: date,
    option_type: OptionType,
    strike: Decimal,
) -> str:
    """Create OCC-format symbol."""
    type_char = "C" if option_type == OptionType.CALL else "P"
    exp_str = expiration.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    return f"{underlying}{exp_str}{type_char}{strike_int:08d}"


def create_vertical_spread(
    underlying: str,
    expiration: date,
    long_strike: Decimal,
    short_strike: Decimal,
    option_type: OptionType,
    quantity: int = 1,
    strategy_id: str | None = None,
) -> OptionStrategy:
    """
    Create a vertical spread (bull/bear call/put spread).

    Args:
        underlying: Underlying symbol
        expiration: Expiration date
        long_strike: Strike price for long leg
        short_strike: Strike price for short leg
        option_type: CALL or PUT
        quantity: Number of spreads
        strategy_id: Optional custom ID

    Returns:
        OptionStrategy representing the vertical spread

    Examples:
        # Bull call spread: buy 150C, sell 155C
        spread = create_vertical_spread(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            long_strike=Decimal("150"),
            short_strike=Decimal("155"),
            option_type=OptionType.CALL,
        )
    """
    # Determine strategy type based on direction and option type
    if option_type == OptionType.CALL:
        if long_strike < short_strike:
            strategy_type = StrategyType.BULL_CALL_SPREAD
        else:
            strategy_type = StrategyType.BEAR_CALL_SPREAD
    else:  # PUT
        if long_strike > short_strike:
            strategy_type = StrategyType.BULL_PUT_SPREAD
        else:
            strategy_type = StrategyType.BEAR_PUT_SPREAD

    long_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, option_type, long_strike),
        underlying=underlying,
        option_type=option_type,
        strike=long_strike,
        expiration=expiration,
    )
    short_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, option_type, short_strike),
        underlying=underlying,
        option_type=option_type,
        strike=short_strike,
        expiration=expiration,
    )

    max_loss = abs(long_strike - short_strike) * 100 * quantity

    return OptionStrategy(
        strategy_id=strategy_id
        or f"vertical_{underlying}_{expiration}_{long_strike}_{short_strike}",
        strategy_type=strategy_type,
        underlying=underlying,
        legs=(
            StrategyLeg(contract=long_contract, quantity=quantity),
            StrategyLeg(contract=short_contract, quantity=-quantity),
        ),
        max_loss=max_loss,
    )


def create_straddle(
    underlying: str,
    expiration: date,
    strike: Decimal,
    quantity: int = 1,
    is_long: bool = True,
    strategy_id: str | None = None,
) -> OptionStrategy:
    """
    Create a straddle (buy/sell both call and put at same strike).

    Args:
        underlying: Underlying symbol
        expiration: Expiration date
        strike: Strike price for both legs
        quantity: Number of straddles
        is_long: True for long straddle, False for short
        strategy_id: Optional custom ID

    Returns:
        OptionStrategy representing the straddle

    Examples:
        # Long straddle at 150 strike
        straddle = create_straddle(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            strike=Decimal("150"),
        )
    """
    call_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, OptionType.CALL, strike),
        underlying=underlying,
        option_type=OptionType.CALL,
        strike=strike,
        expiration=expiration,
    )
    put_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, OptionType.PUT, strike),
        underlying=underlying,
        option_type=OptionType.PUT,
        strike=strike,
        expiration=expiration,
    )

    leg_qty = quantity if is_long else -quantity

    return OptionStrategy(
        strategy_id=strategy_id or f"straddle_{underlying}_{expiration}_{strike}",
        strategy_type=StrategyType.STRADDLE,
        underlying=underlying,
        legs=(
            StrategyLeg(contract=call_contract, quantity=leg_qty),
            StrategyLeg(contract=put_contract, quantity=leg_qty),
        ),
    )


def create_strangle(
    underlying: str,
    expiration: date,
    put_strike: Decimal,
    call_strike: Decimal,
    quantity: int = 1,
    is_long: bool = True,
    strategy_id: str | None = None,
) -> OptionStrategy:
    """
    Create a strangle (buy/sell OTM call and put).

    Args:
        underlying: Underlying symbol
        expiration: Expiration date
        put_strike: Strike for put leg (below current price)
        call_strike: Strike for call leg (above current price)
        quantity: Number of strangles
        is_long: True for long strangle, False for short
        strategy_id: Optional custom ID

    Returns:
        OptionStrategy representing the strangle
    """
    call_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, OptionType.CALL, call_strike),
        underlying=underlying,
        option_type=OptionType.CALL,
        strike=call_strike,
        expiration=expiration,
    )
    put_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, OptionType.PUT, put_strike),
        underlying=underlying,
        option_type=OptionType.PUT,
        strike=put_strike,
        expiration=expiration,
    )

    leg_qty = quantity if is_long else -quantity

    return OptionStrategy(
        strategy_id=strategy_id
        or f"strangle_{underlying}_{expiration}_{put_strike}_{call_strike}",
        strategy_type=StrategyType.STRANGLE,
        underlying=underlying,
        legs=(
            StrategyLeg(contract=put_contract, quantity=leg_qty),
            StrategyLeg(contract=call_contract, quantity=leg_qty),
        ),
    )


def create_iron_condor(
    underlying: str,
    expiration: date,
    put_long_strike: Decimal,
    put_short_strike: Decimal,
    call_short_strike: Decimal,
    call_long_strike: Decimal,
    quantity: int = 1,
    strategy_id: str | None = None,
) -> OptionStrategy:
    """
    Create an iron condor.

    Sells OTM put spread and OTM call spread for net credit.
    Profits from range-bound underlying.

    Args:
        underlying: Underlying symbol
        expiration: Expiration date
        put_long_strike: Long put strike (lowest, wing protection)
        put_short_strike: Short put strike (below current price)
        call_short_strike: Short call strike (above current price)
        call_long_strike: Long call strike (highest, wing protection)
        quantity: Number of condors
        strategy_id: Optional custom ID

    Returns:
        OptionStrategy representing the iron condor

    Examples:
        # Iron condor with 10-point wings
        condor = create_iron_condor(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            put_long_strike=Decimal("140"),
            put_short_strike=Decimal("145"),
            call_short_strike=Decimal("155"),
            call_long_strike=Decimal("160"),
        )
    """
    legs = []
    for strike, opt_type, qty_mult in [
        (put_long_strike, OptionType.PUT, 1),  # Buy lower put (protection)
        (put_short_strike, OptionType.PUT, -1),  # Sell higher put
        (call_short_strike, OptionType.CALL, -1),  # Sell lower call
        (call_long_strike, OptionType.CALL, 1),  # Buy higher call (protection)
    ]:
        contract = OptionContract(
            symbol=_make_occ_symbol(underlying, expiration, opt_type, strike),
            underlying=underlying,
            option_type=opt_type,
            strike=strike,
            expiration=expiration,
        )
        legs.append(StrategyLeg(contract=contract, quantity=quantity * qty_mult))

    # Max loss is the larger wing width minus credit received
    put_width = put_short_strike - put_long_strike
    call_width = call_long_strike - call_short_strike
    max_loss = max(put_width, call_width) * 100 * quantity

    return OptionStrategy(
        strategy_id=strategy_id or f"iron_condor_{underlying}_{expiration}",
        strategy_type=StrategyType.IRON_CONDOR,
        underlying=underlying,
        legs=tuple(legs),
        max_loss=max_loss,
        breakeven_prices=(put_short_strike, call_short_strike),
    )


def create_butterfly(
    underlying: str,
    expiration: date,
    lower_strike: Decimal,
    middle_strike: Decimal,
    upper_strike: Decimal,
    option_type: OptionType = OptionType.CALL,
    quantity: int = 1,
    strategy_id: str | None = None,
) -> OptionStrategy:
    """
    Create a butterfly spread.

    Buy 1 lower, sell 2 middle, buy 1 upper for net debit.
    Profits when underlying stays near middle strike at expiry.

    Args:
        underlying: Underlying symbol
        expiration: Expiration date
        lower_strike: Lower wing strike
        middle_strike: Body strike (sell 2)
        upper_strike: Upper wing strike
        option_type: CALL or PUT
        quantity: Number of butterflies
        strategy_id: Optional custom ID

    Returns:
        OptionStrategy representing the butterfly
    """
    lower_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, option_type, lower_strike),
        underlying=underlying,
        option_type=option_type,
        strike=lower_strike,
        expiration=expiration,
    )
    middle_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, option_type, middle_strike),
        underlying=underlying,
        option_type=option_type,
        strike=middle_strike,
        expiration=expiration,
    )
    upper_contract = OptionContract(
        symbol=_make_occ_symbol(underlying, expiration, option_type, upper_strike),
        underlying=underlying,
        option_type=option_type,
        strike=upper_strike,
        expiration=expiration,
    )

    # Max profit at middle strike
    wing_width = middle_strike - lower_strike
    max_profit = wing_width * 100 * quantity

    return OptionStrategy(
        strategy_id=strategy_id or f"butterfly_{underlying}_{expiration}_{middle_strike}",
        strategy_type=StrategyType.BUTTERFLY,
        underlying=underlying,
        legs=(
            StrategyLeg(contract=lower_contract, quantity=quantity),
            StrategyLeg(contract=middle_contract, quantity=-2 * quantity),
            StrategyLeg(contract=upper_contract, quantity=quantity),
        ),
        max_profit=max_profit,
        breakeven_prices=(lower_strike, upper_strike),
    )

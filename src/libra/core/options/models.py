"""
Core Option Data Models.

Provides fundamental data structures for options trading including:
- Option contract specifications
- Option positions with P&L tracking
- Type enums for option classification

Issue #63: Options Data Models
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum

import msgspec


# =============================================================================
# Enums
# =============================================================================


class OptionType(str, Enum):
    """
    Option type (call or put).

    - CALL: Right to buy the underlying at strike price
    - PUT: Right to sell the underlying at strike price
    """

    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """
    Option exercise style.

    - AMERICAN: Can exercise anytime before expiration
    - EUROPEAN: Can only exercise at expiration
    """

    AMERICAN = "american"
    EUROPEAN = "european"


# =============================================================================
# Option Contract
# =============================================================================


class OptionContract(msgspec.Struct, frozen=True, gc=False, kw_only=True):
    """
    Represents an option contract specification.

    Immutable struct containing all contract details for an option.
    Uses OCC symbol format for identification.

    Attributes:
        symbol: OCC symbol (e.g., AAPL250117C00150000)
        underlying: Underlying ticker symbol (e.g., AAPL)
        option_type: Call or put
        strike: Strike price
        expiration: Expiration date
        style: Exercise style (American/European)
        multiplier: Shares per contract (default 100)
        exchange: Exchange where traded (optional)

    Examples:
        contract = OptionContract(
            symbol="AAPL250117C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("150.00"),
            expiration=date(2025, 1, 17),
        )

        # Check moneyness
        if contract.is_itm(Decimal("160.00")):
            print("In the money!")

        # Get intrinsic value
        value = contract.intrinsic_value(Decimal("160.00"))  # Returns 10.00
    """

    # Identification
    symbol: str  # OCC symbol: AAPL250117C00150000
    underlying: str  # Underlying ticker: AAPL

    # Contract specifications
    option_type: OptionType
    strike: Decimal
    expiration: date
    style: OptionStyle = OptionStyle.AMERICAN

    # Contract details
    multiplier: int = 100  # Shares per contract
    exchange: str | None = None

    @property
    def is_call(self) -> bool:
        """True if this is a call option."""
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        """True if this is a put option."""
        return self.option_type == OptionType.PUT

    @property
    def days_to_expiry(self) -> int:
        """Days remaining until expiration."""
        return (self.expiration - date.today()).days

    @property
    def is_expired(self) -> bool:
        """True if option has expired."""
        return self.expiration < date.today()

    def intrinsic_value(self, underlying_price: Decimal) -> Decimal:
        """
        Calculate intrinsic value given underlying price.

        Args:
            underlying_price: Current price of underlying

        Returns:
            Intrinsic value (always >= 0)
        """
        if self.is_call:
            return max(Decimal("0"), underlying_price - self.strike)
        return max(Decimal("0"), self.strike - underlying_price)

    def is_itm(self, underlying_price: Decimal) -> bool:
        """
        Check if in-the-money.

        Args:
            underlying_price: Current underlying price

        Returns:
            True if option has positive intrinsic value
        """
        return self.intrinsic_value(underlying_price) > 0

    def is_otm(self, underlying_price: Decimal) -> bool:
        """
        Check if out-of-the-money.

        Args:
            underlying_price: Current underlying price

        Returns:
            True if option has zero intrinsic value
        """
        return self.intrinsic_value(underlying_price) == 0

    def is_atm(
        self, underlying_price: Decimal, tolerance: Decimal = Decimal("0.01")
    ) -> bool:
        """
        Check if at-the-money (within tolerance of strike).

        Args:
            underlying_price: Current underlying price
            tolerance: Percentage tolerance (default 1%)

        Returns:
            True if underlying is within tolerance of strike
        """
        if underlying_price == 0:
            return False
        diff = abs(underlying_price - self.strike) / underlying_price
        return diff <= tolerance


# =============================================================================
# Option Position
# =============================================================================


class OptionPosition(msgspec.Struct, frozen=True, gc=False, kw_only=True):
    """
    An open option position.

    Tracks a position in a single option contract with P&L calculations.

    Attributes:
        contract: The option contract
        quantity: Number of contracts (positive=long, negative=short)
        avg_price: Average entry price per contract
        current_price: Current market price per contract
        opened_at: When position was opened
        updated_at: When position was last updated

    Examples:
        position = OptionPosition(
            contract=contract,
            quantity=10,  # Long 10 contracts
            avg_price=Decimal("5.50"),
            current_price=Decimal("6.25"),
            opened_at=datetime.now(),
        )

        print(f"Unrealized P&L: ${position.unrealized_pnl}")
        print(f"P&L %: {position.unrealized_pnl_pct}%")
    """

    contract: OptionContract
    quantity: int  # Positive = long, negative = short
    avg_price: Decimal  # Average entry price per contract

    # Current market data
    current_price: Decimal = Decimal("0")

    # Timestamps
    opened_at: datetime
    updated_at: datetime | None = None

    @property
    def is_long(self) -> bool:
        """True if position is long (bought options)."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if position is short (sold/wrote options)."""
        return self.quantity < 0

    @property
    def market_value(self) -> Decimal:
        """
        Current market value of position.

        Returns:
            quantity * current_price * multiplier
        """
        return self.quantity * self.current_price * self.contract.multiplier

    @property
    def cost_basis(self) -> Decimal:
        """
        Total cost basis of position.

        Returns:
            quantity * avg_price * multiplier
        """
        return self.quantity * self.avg_price * self.contract.multiplier

    @property
    def unrealized_pnl(self) -> Decimal:
        """
        Unrealized profit/loss.

        Returns:
            market_value - cost_basis
        """
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> Decimal:
        """
        Unrealized P&L as percentage of cost basis.

        Returns:
            (unrealized_pnl / abs(cost_basis)) * 100, or 0 if cost_basis is 0
        """
        if self.cost_basis == 0:
            return Decimal("0")
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100

    def with_price(self, new_price: Decimal) -> OptionPosition:
        """
        Create new position with updated price.

        Args:
            new_price: New current price

        Returns:
            New OptionPosition with updated price and timestamp
        """
        return OptionPosition(
            contract=self.contract,
            quantity=self.quantity,
            avg_price=self.avg_price,
            current_price=new_price,
            opened_at=self.opened_at,
            updated_at=datetime.now(),
        )


# =============================================================================
# Serialization Helpers
# =============================================================================

_encoder = msgspec.json.Encoder()
_contract_decoder = msgspec.json.Decoder(OptionContract)
_position_decoder = msgspec.json.Decoder(OptionPosition)


def encode_option_contract(contract: OptionContract) -> bytes:
    """Encode OptionContract to JSON bytes."""
    return _encoder.encode(contract)


def decode_option_contract(data: bytes) -> OptionContract:
    """Decode OptionContract from JSON bytes."""
    return _contract_decoder.decode(data)


def encode_option_position(position: OptionPosition) -> bytes:
    """Encode OptionPosition to JSON bytes."""
    return _encoder.encode(position)


def decode_option_position(data: bytes) -> OptionPosition:
    """Decode OptionPosition from JSON bytes."""
    return _position_decoder.decode(data)

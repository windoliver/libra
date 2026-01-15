"""
Option Greeks Models.

Provides data structures for option price sensitivities (Greeks):
- Delta: Price sensitivity to underlying
- Gamma: Rate of delta change
- Theta: Time decay
- Vega: Volatility sensitivity
- Rho: Interest rate sensitivity
- IV: Implied volatility

Issue #63: Options Data Models
"""

from __future__ import annotations

from decimal import Decimal

import msgspec


class Greeks(msgspec.Struct, frozen=True, gc=False):
    """
    Option Greeks - price sensitivities.

    All values are per-contract. Use scale() for position-level Greeks.

    Attributes:
        delta: Price change per $1 underlying move (-1 to +1 for single option)
        gamma: Delta change per $1 underlying move (always >= 0)
        theta: Daily time decay (usually negative for long positions)
        vega: Price change per 1% IV change (always >= 0)
        rho: Price change per 1% interest rate change
        iv: Implied volatility (annualized decimal, e.g., 0.35 = 35%)
        timestamp_ns: When Greeks were calculated (nanoseconds)

    Examples:
        greeks = Greeks(
            delta=Decimal("0.65"),
            gamma=Decimal("0.02"),
            theta=Decimal("-0.05"),
            vega=Decimal("0.15"),
            rho=Decimal("0.03"),
            iv=Decimal("0.35"),
        )

        # Scale for 10 contracts
        position_greeks = greeks.scale(10)
        print(f"Position delta: {position_greeks.delta}")  # 650.0
    """

    delta: Decimal  # Price change per $1 underlying move (-1 to +1)
    gamma: Decimal  # Delta change per $1 underlying move (0+)
    theta: Decimal  # Daily time decay (usually negative)
    vega: Decimal  # Price change per 1% IV change (0+)
    rho: Decimal  # Price change per 1% rate change
    iv: Decimal  # Implied volatility (annualized, e.g., 0.35 = 35%)
    timestamp_ns: int | None = None  # When Greeks were calculated

    @property
    def delta_dollars(self) -> Decimal:
        """
        Delta in dollar terms per contract.

        Returns:
            delta * 100 (standard multiplier)
        """
        return self.delta * 100

    @property
    def iv_percent(self) -> Decimal:
        """
        IV as percentage.

        Returns:
            iv * 100 (e.g., 35.0 for 35% IV)
        """
        return self.iv * 100

    def scale(self, quantity: int, multiplier: int = 100) -> Greeks:
        """
        Scale Greeks for a position.

        Args:
            quantity: Number of contracts (positive=long, negative=short)
            multiplier: Contract multiplier (default 100)

        Returns:
            New Greeks with scaled values (IV unchanged)

        Examples:
            # Long 10 contracts
            pos_greeks = greeks.scale(10)

            # Short 5 contracts
            pos_greeks = greeks.scale(-5)
        """
        factor = Decimal(quantity * multiplier)
        return Greeks(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            theta=self.theta * factor,
            vega=self.vega * factor,
            rho=self.rho * factor,
            iv=self.iv,  # IV doesn't scale
            timestamp_ns=self.timestamp_ns,
        )

    def __add__(self, other: Greeks) -> Greeks:
        """
        Add two Greeks together (for aggregating positions).

        Note: IV is averaged when adding.
        """
        if not isinstance(other, Greeks):
            return NotImplemented
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            iv=(self.iv + other.iv) / 2,  # Average IV
            timestamp_ns=self.timestamp_ns,
        )


class GreeksSnapshot(msgspec.Struct, frozen=True, gc=False):
    """
    Greeks with market data context.

    Combines Greeks with the market conditions at time of calculation.

    Attributes:
        greeks: The option Greeks
        underlying_price: Current underlying price
        option_price: Current option mid price
        bid: Best bid price
        ask: Best ask price
        volume: Trading volume
        open_interest: Open interest (contracts outstanding)
        timestamp_ns: When snapshot was taken

    Examples:
        snapshot = GreeksSnapshot(
            greeks=greeks,
            underlying_price=Decimal("155.00"),
            option_price=Decimal("7.50"),
            bid=Decimal("7.40"),
            ask=Decimal("7.60"),
            volume=1500,
            open_interest=25000,
        )

        print(f"Spread: {snapshot.spread_pct:.2f}%")
    """

    greeks: Greeks
    underlying_price: Decimal
    option_price: Decimal
    bid: Decimal
    ask: Decimal
    volume: int
    open_interest: int
    timestamp_ns: int | None = None

    @property
    def bid_ask_spread(self) -> Decimal:
        """
        Bid-ask spread in dollars.

        Returns:
            ask - bid
        """
        return self.ask - self.bid

    @property
    def spread_pct(self) -> Decimal:
        """
        Spread as percentage of mid price.

        Returns:
            (spread / mid) * 100, or 0 if mid is 0
        """
        mid = self.mid_price
        if mid == 0:
            return Decimal("0")
        return (self.bid_ask_spread / mid) * 100

    @property
    def mid_price(self) -> Decimal:
        """
        Mid price between bid and ask.

        Returns:
            (bid + ask) / 2
        """
        return (self.bid + self.ask) / 2

    @property
    def is_liquid(self) -> bool:
        """
        Check if option appears liquid.

        Returns:
            True if spread < 10% and open_interest > 100
        """
        return self.spread_pct < 10 and self.open_interest > 100


# =============================================================================
# Factory Functions
# =============================================================================


def greeks_from_dict(data: dict) -> Greeks:
    """
    Create Greeks from a dictionary.

    Useful for converting API responses.

    Args:
        data: Dictionary with greek values

    Returns:
        Greeks instance
    """
    return Greeks(
        delta=Decimal(str(data.get("delta", 0))),
        gamma=Decimal(str(data.get("gamma", 0))),
        theta=Decimal(str(data.get("theta", 0))),
        vega=Decimal(str(data.get("vega", 0))),
        rho=Decimal(str(data.get("rho", 0))),
        iv=Decimal(str(data.get("iv", data.get("implied_volatility", 0)))),
        timestamp_ns=data.get("timestamp_ns"),
    )


def zero_greeks() -> Greeks:
    """
    Create Greeks with all zeros.

    Useful as a default or placeholder.

    Returns:
        Greeks with all values set to 0
    """
    return Greeks(
        delta=Decimal("0"),
        gamma=Decimal("0"),
        theta=Decimal("0"),
        vega=Decimal("0"),
        rho=Decimal("0"),
        iv=Decimal("0"),
    )

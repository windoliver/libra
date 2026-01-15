"""
Option Chain Models.

Provides data structures for representing option chains:
- Single entry (one strike/type)
- Single expiration (all strikes for one date)
- Full chain (all expirations for an underlying)

Issue #63: Options Data Models
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Iterator

import msgspec

from libra.core.options.greeks import GreeksSnapshot
from libra.core.options.models import OptionContract, OptionType


if TYPE_CHECKING:
    pass


class OptionChainEntry(msgspec.Struct, frozen=True, gc=False):
    """
    Single entry in an option chain.

    Represents one strike/type combination with its Greeks and market data.

    Attributes:
        contract: The option contract specification
        snapshot: Greeks and market data snapshot
    """

    contract: OptionContract
    snapshot: GreeksSnapshot


class OptionChainExpiry(msgspec.Struct, gc=False):
    """
    All options for a single expiration date.

    Contains both calls and puts at various strikes for one expiry.

    Note: Not frozen to allow efficient building during data fetch.

    Attributes:
        underlying: Underlying symbol
        expiration: Expiration date
        calls: Map of strike price to call option entry
        puts: Map of strike price to put option entry

    Examples:
        expiry = OptionChainExpiry(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            calls={Decimal("150"): call_entry, Decimal("155"): call_entry2},
            puts={Decimal("150"): put_entry, Decimal("155"): put_entry2},
        )

        # Get ATM strike
        atm = expiry.get_atm_strike(Decimal("152.50"))

        # Get straddle components
        call, put = expiry.get_straddle(atm)
    """

    underlying: str
    expiration: date
    calls: dict[Decimal, OptionChainEntry]  # strike -> entry
    puts: dict[Decimal, OptionChainEntry]  # strike -> entry

    @property
    def strikes(self) -> list[Decimal]:
        """
        All available strikes, sorted ascending.

        Returns:
            Sorted list of unique strike prices
        """
        return sorted(set(self.calls.keys()) | set(self.puts.keys()))

    @property
    def num_strikes(self) -> int:
        """Number of unique strikes."""
        return len(self.strikes)

    @property
    def days_to_expiry(self) -> int:
        """Days remaining until expiration."""
        return (self.expiration - date.today()).days

    def get_atm_strike(self, underlying_price: Decimal) -> Decimal | None:
        """
        Get strike closest to current underlying price.

        Args:
            underlying_price: Current underlying price

        Returns:
            Closest strike price, or None if no strikes available
        """
        if not self.strikes:
            return None
        return min(self.strikes, key=lambda s: abs(s - underlying_price))

    def get_call(self, strike: Decimal) -> OptionChainEntry | None:
        """
        Get call option at strike.

        Args:
            strike: Strike price

        Returns:
            OptionChainEntry for call, or None if not available
        """
        return self.calls.get(strike)

    def get_put(self, strike: Decimal) -> OptionChainEntry | None:
        """
        Get put option at strike.

        Args:
            strike: Strike price

        Returns:
            OptionChainEntry for put, or None if not available
        """
        return self.puts.get(strike)

    def get_straddle(
        self, strike: Decimal
    ) -> tuple[OptionChainEntry, OptionChainEntry] | None:
        """
        Get call and put at same strike (straddle components).

        Args:
            strike: Strike price

        Returns:
            Tuple of (call, put) entries, or None if either is missing
        """
        call = self.get_call(strike)
        put = self.get_put(strike)
        if call and put:
            return (call, put)
        return None

    def get_strangle(
        self, put_strike: Decimal, call_strike: Decimal
    ) -> tuple[OptionChainEntry, OptionChainEntry] | None:
        """
        Get put and call at different strikes (strangle components).

        Args:
            put_strike: Strike for put leg
            call_strike: Strike for call leg

        Returns:
            Tuple of (put, call) entries, or None if either is missing
        """
        put = self.get_put(put_strike)
        call = self.get_call(call_strike)
        if put and call:
            return (put, call)
        return None

    def filter_by_delta(
        self,
        target_delta: Decimal,
        option_type: OptionType,
        tolerance: Decimal = Decimal("0.05"),
    ) -> Iterator[OptionChainEntry]:
        """
        Find options with delta near target.

        Args:
            target_delta: Target delta (absolute value, e.g., 0.30)
            option_type: CALL or PUT
            tolerance: Acceptable deviation from target

        Yields:
            Matching OptionChainEntry objects
        """
        options = self.calls if option_type == OptionType.CALL else self.puts
        for entry in options.values():
            actual_delta = abs(entry.snapshot.greeks.delta)
            if abs(actual_delta - abs(target_delta)) <= tolerance:
                yield entry

    def filter_by_strike_range(
        self, min_strike: Decimal | None = None, max_strike: Decimal | None = None
    ) -> Iterator[OptionChainEntry]:
        """
        Get all options within a strike range.

        Args:
            min_strike: Minimum strike (inclusive)
            max_strike: Maximum strike (inclusive)

        Yields:
            All options (calls and puts) within range
        """
        for strike in self.strikes:
            if min_strike is not None and strike < min_strike:
                continue
            if max_strike is not None and strike > max_strike:
                continue
            if strike in self.calls:
                yield self.calls[strike]
            if strike in self.puts:
                yield self.puts[strike]

    def all_entries(self) -> Iterator[OptionChainEntry]:
        """
        Iterate over all entries (calls and puts).

        Yields:
            All OptionChainEntry objects
        """
        yield from self.calls.values()
        yield from self.puts.values()


class OptionChain(msgspec.Struct, gc=False):
    """
    Full option chain for an underlying across all expirations.

    Attributes:
        underlying: Underlying symbol
        underlying_price: Current underlying price
        expirations: Map of expiration date to chain for that date
        timestamp_ns: When chain was fetched

    Examples:
        chain = OptionChain(
            underlying="AAPL",
            underlying_price=Decimal("152.50"),
            expirations={date(2025, 1, 17): expiry1, date(2025, 2, 21): expiry2},
            timestamp_ns=time.time_ns(),
        )

        # Get nearest expiry with at least 30 DTE
        expiry = chain.nearest_expiry(min_dte=30)

        # Find 30-delta calls across all expirations
        for entry in chain.filter_by_delta(Decimal("0.30"), OptionType.CALL):
            print(f"{entry.contract.symbol}: {entry.snapshot.greeks.delta}")
    """

    underlying: str
    underlying_price: Decimal
    expirations: dict[date, OptionChainExpiry]
    timestamp_ns: int

    @property
    def expiration_dates(self) -> list[date]:
        """
        All expiration dates, sorted ascending.

        Returns:
            Sorted list of expiration dates
        """
        return sorted(self.expirations.keys())

    @property
    def num_expirations(self) -> int:
        """Number of expirations in chain."""
        return len(self.expirations)

    def get_expiry(self, expiration: date) -> OptionChainExpiry | None:
        """
        Get chain for specific expiration.

        Args:
            expiration: Expiration date

        Returns:
            OptionChainExpiry or None if not found
        """
        return self.expirations.get(expiration)

    def nearest_expiry(self, min_dte: int = 0) -> OptionChainExpiry | None:
        """
        Get nearest expiration with at least min_dte days.

        Args:
            min_dte: Minimum days to expiration

        Returns:
            OptionChainExpiry or None if none qualify
        """
        today = date.today()
        for exp in self.expiration_dates:
            if (exp - today).days >= min_dte:
                return self.expirations[exp]
        return None

    def expiry_at_dte(self, target_dte: int, tolerance: int = 7) -> OptionChainExpiry | None:
        """
        Get expiration closest to target DTE.

        Args:
            target_dte: Target days to expiration
            tolerance: Maximum days from target

        Returns:
            OptionChainExpiry closest to target, or None
        """
        today = date.today()
        best_expiry = None
        best_diff = float("inf")

        for exp in self.expiration_dates:
            dte = (exp - today).days
            diff = abs(dte - target_dte)
            if diff <= tolerance and diff < best_diff:
                best_diff = diff
                best_expiry = self.expirations[exp]

        return best_expiry

    def filter_by_delta(
        self,
        target_delta: Decimal,
        option_type: OptionType,
        tolerance: Decimal = Decimal("0.05"),
    ) -> Iterator[OptionChainEntry]:
        """
        Find options with delta near target across all expirations.

        Args:
            target_delta: Target delta (absolute value)
            option_type: CALL or PUT
            tolerance: Acceptable deviation

        Yields:
            Matching OptionChainEntry objects
        """
        for expiry in self.expirations.values():
            yield from expiry.filter_by_delta(target_delta, option_type, tolerance)

    def all_entries(self) -> Iterator[OptionChainEntry]:
        """
        Iterate over all entries across all expirations.

        Yields:
            All OptionChainEntry objects
        """
        for expiry in self.expirations.values():
            yield from expiry.all_entries()


# =============================================================================
# Builder Functions
# =============================================================================


def build_option_chain_expiry(
    underlying: str,
    expiration: date,
    entries: list[OptionChainEntry],
) -> OptionChainExpiry:
    """
    Build an OptionChainExpiry from a list of entries.

    Args:
        underlying: Underlying symbol
        expiration: Expiration date
        entries: List of option chain entries

    Returns:
        OptionChainExpiry with entries organized by type and strike
    """
    calls: dict[Decimal, OptionChainEntry] = {}
    puts: dict[Decimal, OptionChainEntry] = {}

    for entry in entries:
        if entry.contract.is_call:
            calls[entry.contract.strike] = entry
        else:
            puts[entry.contract.strike] = entry

    return OptionChainExpiry(
        underlying=underlying,
        expiration=expiration,
        calls=calls,
        puts=puts,
    )

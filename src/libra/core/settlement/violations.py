"""
Good Faith Violation (GFV) Tracker for Cash Accounts.

A Good Faith Violation occurs when you:
1. Buy a security with unsettled funds
2. Sell that security before the original funds settle

Three GFVs in 12 months = account restricted to settled funds for 90 days.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable
from uuid import uuid4

from libra.core.settlement.models import (
    GoodFaithViolation,
    GFVStatus,
)
from libra.core.settlement.tracker import SettlementTracker


@dataclass
class TradeRecord:
    """Record of a trade for GFV tracking."""

    trade_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    trade_date: date
    settlement_date: date
    funding_trade_id: str | None = None  # ID of trade that funded this buy


class GFVTracker:
    """
    Tracks Good Faith Violations for cash accounts.

    Monitors buy/sell sequences to detect potential GFVs and
    maintains violation history for account restriction tracking.

    GFV Rules:
    - Cannot sell shares bought with unsettled funds before those funds settle
    - Three violations in 12 months triggers 90-day restriction
    - During restriction, can only trade with settled funds

    Example:
        >>> tracker = GFVTracker("ACC001")
        >>> # Day 1: Sell AAPL, get $10,000 (settles T+1)
        >>> tracker.record_sell("T1", "AAPL", 100, Decimal("100"), date(2024,1,2))
        >>> # Day 1: Buy MSFT with unsettled funds
        >>> tracker.record_buy("T2", "MSFT", 50, Decimal("200"), date(2024,1,2))
        >>> # Day 1: Sell MSFT before AAPL sale settles -> GFV!
        >>> result = tracker.check_sell("MSFT", 50, date(2024,1,2))
        >>> result.would_violate
        True
    """

    # Violation thresholds
    MAX_VIOLATIONS = 3
    VIOLATION_WINDOW_DAYS = 365  # 12 months
    RESTRICTION_DAYS = 90

    def __init__(
        self,
        account_id: str,
        settlement_tracker: SettlementTracker | None = None,
        on_violation: Callable[[GoodFaithViolation], None] | None = None,
    ):
        """
        Initialize GFV tracker.

        Args:
            account_id: Account identifier
            settlement_tracker: Optional settlement tracker for integration
            on_violation: Callback when a violation occurs
        """
        self.account_id = account_id
        self.settlement_tracker = settlement_tracker
        self._on_violation = on_violation

        # Track trades by symbol
        self._buys: dict[str, list[TradeRecord]] = defaultdict(list)
        self._sells: dict[str, list[TradeRecord]] = defaultdict(list)

        # Track unsettled funding (sell proceeds used to fund buys)
        self._unsettled_funding: dict[str, str] = {}  # buy_trade_id -> sell_trade_id

        # Violation history
        self._violations: list[GoodFaithViolation] = []

        # Restriction status
        self._restricted_until: date | None = None

    @property
    def is_restricted(self) -> bool:
        """Check if account is currently restricted."""
        if self._restricted_until is None:
            return False
        return date.today() < self._restricted_until

    @property
    def violation_count(self) -> int:
        """Count violations in the last 12 months."""
        cutoff = date.today() - timedelta(days=self.VIOLATION_WINDOW_DAYS)
        return sum(1 for v in self._violations if v.violation_date >= cutoff)

    def get_status(self) -> GFVStatus:
        """Get current GFV status for the account."""
        recent_violations = [
            v
            for v in self._violations
            if v.violation_date
            >= date.today() - timedelta(days=self.VIOLATION_WINDOW_DAYS)
        ]

        return GFVStatus(
            account_id=self.account_id,
            violation_count=len(recent_violations),
            max_violations=self.MAX_VIOLATIONS,
            is_restricted=self.is_restricted,
            restriction_end_date=self._restricted_until,
            violations=tuple(recent_violations),
        )

    def record_sell(
        self,
        trade_id: str,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        settlement_date: date,
        trade_date: date | None = None,
    ) -> None:
        """
        Record a sell trade.

        Sell proceeds become available funds that may be used for subsequent buys.

        Args:
            trade_id: Unique trade identifier
            symbol: Symbol sold
            quantity: Number of shares
            price: Sell price
            settlement_date: When proceeds will settle
            trade_date: Date of trade (default: today)
        """
        trade_date = trade_date or date.today()

        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            trade_date=trade_date,
            settlement_date=settlement_date,
        )

        self._sells[symbol].append(record)

    def record_buy(
        self,
        trade_id: str,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        settlement_date: date,
        trade_date: date | None = None,
        funding_source: str | None = None,
    ) -> None:
        """
        Record a buy trade.

        If funded by unsettled proceeds, tracks the dependency for GFV detection.

        Args:
            trade_id: Unique trade identifier
            symbol: Symbol bought
            quantity: Number of shares
            price: Buy price
            settlement_date: When shares will settle
            trade_date: Date of trade (default: today)
            funding_source: Trade ID of sell that funded this (if unsettled)
        """
        trade_date = trade_date or date.today()

        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            trade_date=trade_date,
            settlement_date=settlement_date,
            funding_trade_id=funding_source,
        )

        self._buys[symbol].append(record)

        if funding_source:
            self._unsettled_funding[trade_id] = funding_source

    def check_sell(
        self,
        symbol: str,
        quantity: Decimal,
        trade_date: date | None = None,
    ) -> GFVCheckResult:
        """
        Check if a sell would cause a Good Faith Violation.

        Args:
            symbol: Symbol to sell
            quantity: Number of shares
            trade_date: Date of potential trade (default: today)

        Returns:
            GFVCheckResult with violation details
        """
        trade_date = trade_date or date.today()

        # Find buys that haven't settled yet and were funded by unsettled proceeds
        risky_buys = []
        risky_quantity = Decimal(0)

        for buy in self._buys.get(symbol, []):
            # Skip if buy has settled
            if buy.settlement_date <= trade_date:
                continue

            # Check if this buy was funded by unsettled proceeds
            funding_id = self._unsettled_funding.get(buy.trade_id)
            if funding_id:
                # Find the funding sell
                funding_sell = self._find_sell_by_id(funding_id)
                if funding_sell and funding_sell.settlement_date > trade_date:
                    # The funding hasn't settled - selling these shares is a GFV
                    risky_buys.append(buy)
                    risky_quantity += buy.quantity

        would_violate = quantity <= risky_quantity and risky_quantity > 0

        return GFVCheckResult(
            would_violate=would_violate,
            risky_quantity=risky_quantity,
            risky_buys=[b.trade_id for b in risky_buys],
            message=(
                f"Selling {quantity} shares of {symbol} would cause a GFV. "
                f"{risky_quantity} shares bought with unsettled funds."
                if would_violate
                else "No GFV risk"
            ),
        )

    def record_violation(
        self,
        symbol: str,
        buy_trade_id: str,
        sell_trade_id: str,
        amount: Decimal,
        description: str | None = None,
    ) -> GoodFaithViolation:
        """
        Record a Good Faith Violation.

        Args:
            symbol: Symbol involved
            buy_trade_id: The buy trade that was sold early
            sell_trade_id: The sell trade that caused the violation
            amount: Dollar amount of the violation
            description: Optional description

        Returns:
            The recorded violation
        """
        violation = GoodFaithViolation(
            violation_id=f"gfv_{uuid4().hex[:12]}",
            account_id=self.account_id,
            symbol=symbol,
            description=description
            or f"Sold {symbol} bought with unsettled funds",
            violation_date=date.today(),
            buy_trade_id=buy_trade_id,
            sell_trade_id=sell_trade_id,
            amount=amount,
        )

        self._violations.append(violation)

        # Check if we need to apply restriction
        if self.violation_count >= self.MAX_VIOLATIONS:
            self._apply_restriction()

        # Callback
        if self._on_violation:
            try:
                self._on_violation(violation)
            except Exception:
                pass

        return violation

    def _apply_restriction(self) -> None:
        """Apply 90-day restriction to the account."""
        self._restricted_until = date.today() + timedelta(days=self.RESTRICTION_DAYS)

    def _find_sell_by_id(self, trade_id: str) -> TradeRecord | None:
        """Find a sell trade by ID."""
        for sells in self._sells.values():
            for sell in sells:
                if sell.trade_id == trade_id:
                    return sell
        return None

    def get_unsettled_buy_funding(self, symbol: str) -> list[tuple[str, str]]:
        """
        Get list of (buy_trade_id, funding_sell_id) for unsettled buys.

        Useful for displaying which positions have GFV risk.

        Args:
            symbol: Symbol to check

        Returns:
            List of (buy_id, sell_id) tuples
        """
        result = []
        today = date.today()

        for buy in self._buys.get(symbol, []):
            if buy.settlement_date > today and buy.funding_trade_id:
                result.append((buy.trade_id, buy.funding_trade_id))

        return result

    def clear_settled(self, as_of: date | None = None) -> int:
        """
        Clear records for fully settled trades.

        Args:
            as_of: Reference date (default: today)

        Returns:
            Number of records cleared
        """
        as_of = as_of or date.today()
        cleared = 0

        for symbol in list(self._buys.keys()):
            before = len(self._buys[symbol])
            self._buys[symbol] = [
                b for b in self._buys[symbol] if b.settlement_date > as_of
            ]
            cleared += before - len(self._buys[symbol])

        for symbol in list(self._sells.keys()):
            before = len(self._sells[symbol])
            self._sells[symbol] = [
                s for s in self._sells[symbol] if s.settlement_date > as_of
            ]
            cleared += before - len(self._sells[symbol])

        # Clear funding links for settled trades
        self._unsettled_funding = {
            buy_id: sell_id
            for buy_id, sell_id in self._unsettled_funding.items()
            if any(
                b.trade_id == buy_id and b.settlement_date > as_of
                for buys in self._buys.values()
                for b in buys
            )
        }

        return cleared


@dataclass
class GFVCheckResult:
    """Result of a GFV check."""

    would_violate: bool
    risky_quantity: Decimal
    risky_buys: list[str] = field(default_factory=list)
    message: str = ""

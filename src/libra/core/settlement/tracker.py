"""
Settlement Tracker for Pending Trade Settlements.

Tracks all pending settlements and provides settlement-aware
position and cash balance calculations.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Callable
from uuid import uuid4

from libra.core.settlement.calendar import SettlementCalendar, get_settlement_calendar
from libra.core.settlement.models import (
    AccountType,
    CashBalance,
    PendingSettlement,
    SecurityType,
    SettledPosition,
    SettlementEvent,
    SettlementStatus,
    SettlementType,
)


class SettlementTracker:
    """
    Tracks pending settlements and calculates settlement-aware balances.

    Maintains the state of all pending trades and provides accurate
    buying power calculations based on settlement status.

    Example:
        >>> tracker = SettlementTracker()
        >>> # Add a buy trade
        >>> settlement = tracker.add_trade(
        ...     trade_id="T001",
        ...     symbol="AAPL",
        ...     side="buy",
        ...     quantity=Decimal("100"),
        ...     price=Decimal("150.00"),
        ... )
        >>> # Check settlement date
        >>> settlement.settlement_date  # T+1
        >>> # Get settled position breakdown
        >>> position = tracker.get_settled_position("AAPL", Decimal("100"))
        >>> position.settled_quantity  # 0 (not settled yet)
    """

    def __init__(
        self,
        calendar: SettlementCalendar | None = None,
        account_type: AccountType = AccountType.MARGIN,
        on_settlement: Callable[[SettlementEvent], None] | None = None,
    ):
        """
        Initialize settlement tracker.

        Args:
            calendar: Settlement calendar (default: NYSE)
            account_type: Account type for settlement rules
            on_settlement: Callback for settlement events
        """
        self.calendar = calendar or get_settlement_calendar()
        self.account_type = account_type
        self._on_settlement = on_settlement

        # Settlement storage
        self._pending: dict[str, PendingSettlement] = {}
        self._by_symbol: dict[str, list[str]] = defaultdict(list)
        self._by_date: dict[date, list[str]] = defaultdict(list)
        self._by_trade: dict[str, str] = {}  # trade_id -> settlement_id

    @property
    def pending_count(self) -> int:
        """Number of pending settlements."""
        return sum(
            1 for s in self._pending.values() if s.status == SettlementStatus.PENDING
        )

    def add_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        trade_date: date | None = None,
        settlement_type: SettlementType = SettlementType.REGULAR,
        security_type: SecurityType = SecurityType.STOCK,
        is_option: bool = False,
        underlying: str | None = None,
    ) -> PendingSettlement:
        """
        Add a new trade to settlement tracking.

        Calculates settlement date based on trade date and settlement type.

        Args:
            trade_id: Unique trade identifier
            symbol: Symbol traded
            side: "buy" or "sell"
            quantity: Number of shares/contracts
            price: Execution price
            trade_date: Date of trade (default: today)
            settlement_type: Type of settlement
            security_type: Type of security
            is_option: Whether this is an options trade
            underlying: Underlying symbol (for options)

        Returns:
            PendingSettlement record
        """
        trade_date = trade_date or date.today()
        settlement_date = self.calendar.get_settlement_date(
            trade_date, settlement_type, security_type
        )

        # Calculate cash impact
        cash_amount = quantity * price
        if side.lower() == "buy":
            cash_amount = -cash_amount  # Paying cash

        settlement_id = f"stl_{uuid4().hex[:12]}"

        settlement = PendingSettlement(
            settlement_id=settlement_id,
            trade_id=trade_id,
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            price=price,
            settlement_type=settlement_type,
            security_type=security_type,
            trade_date=trade_date,
            settlement_date=settlement_date,
            cash_amount=cash_amount,
            is_option=is_option,
            underlying=underlying,
            executed_at=datetime.now(),
        )

        # Store settlement
        self._pending[settlement_id] = settlement
        self._by_symbol[symbol].append(settlement_id)
        self._by_date[settlement_date].append(settlement_id)
        self._by_trade[trade_id] = settlement_id

        # Emit event
        self._emit_event(
            "settlement_pending",
            settlement,
        )

        return settlement

    def add_option_exercise(
        self,
        exercise_id: str,
        option_symbol: str,
        underlying: str,
        strike: Decimal,
        quantity: int,
        is_call: bool,
        is_assignment: bool = False,
        trade_date: date | None = None,
    ) -> PendingSettlement:
        """
        Add option exercise/assignment to settlement tracking.

        Creates a pending stock settlement from the exercise.

        Args:
            exercise_id: Unique exercise identifier
            option_symbol: Option symbol being exercised
            underlying: Underlying stock symbol
            strike: Strike price
            quantity: Number of contracts
            is_call: True for calls, False for puts
            is_assignment: True if assigned, False if exercising
            trade_date: Date of exercise (default: today)

        Returns:
            PendingSettlement for the resulting stock position
        """
        shares = Decimal(quantity * 100)  # Standard multiplier
        cash_amount = strike * shares

        if is_call:
            # Call exercise: buy stock at strike
            # Call assignment: sell stock at strike (short call assigned)
            side = "buy" if not is_assignment else "sell"
        else:
            # Put exercise: sell stock at strike
            # Put assignment: buy stock at strike (short put assigned)
            side = "sell" if not is_assignment else "buy"

        settlement_type = (
            SettlementType.ASSIGNMENT if is_assignment else SettlementType.EXERCISE
        )

        return self.add_trade(
            trade_id=exercise_id,
            symbol=underlying,
            side=side,
            quantity=shares,
            price=strike,
            trade_date=trade_date,
            settlement_type=settlement_type,
            security_type=SecurityType.STOCK,
            is_option=False,  # Result is stock, not option
            underlying=underlying,
        )

    def get_settlement(self, settlement_id: str) -> PendingSettlement | None:
        """Get a settlement by ID."""
        return self._pending.get(settlement_id)

    def get_settlement_by_trade(self, trade_id: str) -> PendingSettlement | None:
        """Get settlement by trade ID."""
        settlement_id = self._by_trade.get(trade_id)
        if settlement_id:
            return self._pending.get(settlement_id)
        return None

    def mark_settled(self, settlement_id: str) -> None:
        """
        Mark a settlement as complete.

        Args:
            settlement_id: Settlement to mark complete
        """
        if settlement_id not in self._pending:
            return

        settlement = self._pending[settlement_id]
        if settlement.status == SettlementStatus.SETTLED:
            return

        # Update status (PendingSettlement is not frozen)
        settlement.status = SettlementStatus.SETTLED
        settlement.settled_at = datetime.now()

        # Emit event
        self._emit_event("settlement_complete", settlement)

    def process_settlements(self, as_of: date | None = None) -> list[str]:
        """
        Process all settlements due by given date.

        Args:
            as_of: Process settlements due by this date (default: today)

        Returns:
            List of settlement IDs that were settled
        """
        as_of = as_of or date.today()
        settled_ids = []

        for stl_date, stl_ids in list(self._by_date.items()):
            if stl_date <= as_of:
                for stl_id in stl_ids:
                    settlement = self._pending.get(stl_id)
                    if settlement and settlement.status == SettlementStatus.PENDING:
                        self.mark_settled(stl_id)
                        settled_ids.append(stl_id)

        return settled_ids

    def get_pending_for_symbol(self, symbol: str) -> list[PendingSettlement]:
        """
        Get all pending settlements for a symbol.

        Args:
            symbol: Symbol to query

        Returns:
            List of pending settlements
        """
        return [
            self._pending[stl_id]
            for stl_id in self._by_symbol.get(symbol, [])
            if self._pending[stl_id].status == SettlementStatus.PENDING
        ]

    def get_all_pending(self) -> list[PendingSettlement]:
        """Get all pending settlements."""
        return [s for s in self._pending.values() if s.status == SettlementStatus.PENDING]

    def get_settlements_by_date(self, settlement_date: date) -> list[PendingSettlement]:
        """
        Get all settlements for a specific date.

        Args:
            settlement_date: Date to query

        Returns:
            List of settlements
        """
        return [
            self._pending[stl_id] for stl_id in self._by_date.get(settlement_date, [])
        ]

    def get_settled_position(
        self,
        symbol: str,
        total_position: Decimal,
    ) -> SettledPosition:
        """
        Calculate settled vs unsettled position breakdown.

        Used to determine shares available to sell without GFV.

        Args:
            symbol: Symbol to query
            total_position: Current total position size

        Returns:
            SettledPosition breakdown
        """
        pending = self.get_pending_for_symbol(symbol)

        pending_buys = sum(s.quantity for s in pending if s.is_buy)
        pending_sells = sum(s.quantity for s in pending if s.is_sell)

        # Unsettled = pending buys (shares not yet received)
        unsettled = pending_buys
        settled = total_position - unsettled

        return SettledPosition(
            symbol=symbol,
            total_quantity=total_position,
            settled_quantity=max(Decimal(0), settled),
            unsettled_quantity=unsettled,
            pending_buys=pending_buys,
            pending_sells=pending_sells,
        )

    def get_cash_balance(
        self,
        total_cash: Decimal,
        currency: str = "USD",
    ) -> CashBalance:
        """
        Calculate settled vs unsettled cash breakdown.

        Used for buying power calculations.

        Args:
            total_cash: Current total cash balance
            currency: Currency code

        Returns:
            CashBalance breakdown
        """
        pending_debits = Decimal(0)
        pending_credits = Decimal(0)

        for settlement in self._pending.values():
            if settlement.status != SettlementStatus.PENDING:
                continue

            if settlement.cash_amount < 0:
                pending_debits += abs(settlement.cash_amount)
            else:
                pending_credits += settlement.cash_amount

        unsettled = pending_credits  # Cash coming but not settled
        settled = total_cash - pending_credits

        return CashBalance(
            currency=currency,
            total_cash=total_cash,
            settled_cash=max(Decimal(0), settled),
            unsettled_cash=unsettled,
            pending_debits=pending_debits,
            pending_credits=pending_credits,
        )

    def get_total_pending_debits(self) -> Decimal:
        """Get total cash amount pending to be paid (buys)."""
        total = Decimal(0)
        for s in self._pending.values():
            if s.status == SettlementStatus.PENDING and s.cash_amount < 0:
                total += abs(s.cash_amount)
        return total

    def get_total_pending_credits(self) -> Decimal:
        """Get total cash amount pending to be received (sells)."""
        total = Decimal(0)
        for s in self._pending.values():
            if s.status == SettlementStatus.PENDING and s.cash_amount > 0:
                total += s.cash_amount
        return total

    def clear_settled(self) -> int:
        """
        Remove all settled transactions from tracking.

        Returns:
            Number of settlements cleared
        """
        cleared = 0
        settled_ids = [
            sid
            for sid, s in self._pending.items()
            if s.status == SettlementStatus.SETTLED
        ]

        for sid in settled_ids:
            settlement = self._pending.pop(sid, None)
            if settlement:
                # Clean up indices
                if settlement.symbol in self._by_symbol:
                    try:
                        self._by_symbol[settlement.symbol].remove(sid)
                    except ValueError:
                        pass
                if settlement.settlement_date in self._by_date:
                    try:
                        self._by_date[settlement.settlement_date].remove(sid)
                    except ValueError:
                        pass
                if settlement.trade_id in self._by_trade:
                    del self._by_trade[settlement.trade_id]
                cleared += 1

        return cleared

    def _emit_event(self, event_type: str, settlement: PendingSettlement) -> None:
        """Emit a settlement event."""
        if self._on_settlement is None:
            return

        event = SettlementEvent(
            event_type=event_type,
            settlement_id=settlement.settlement_id,
            trade_id=settlement.trade_id,
            symbol=settlement.symbol,
            status=settlement.status,
            timestamp=datetime.now(),
            cash_amount=settlement.cash_amount,
            quantity=settlement.quantity,
        )

        try:
            self._on_settlement(event)
        except Exception:
            pass  # Don't let callback errors break tracking

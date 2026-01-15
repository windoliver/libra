"""
Unit tests for Settlement Tracker.

Tests pending settlement tracking and position/cash calculations.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from libra.core.settlement import (
    SettlementTracker,
    SettlementCalendar,
    SettlementStatus,
    SettlementType,
    SecurityType,
    AccountType,
)


class TestSettlementTracker:
    """Tests for SettlementTracker class."""

    @pytest.fixture
    def tracker(self) -> SettlementTracker:
        """Create a settlement tracker."""
        return SettlementTracker()

    def test_add_buy_trade(self, tracker: SettlementTracker) -> None:
        """Add a buy trade creates pending settlement."""
        settlement = tracker.add_trade(
            trade_id="T001",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )

        assert settlement.trade_id == "T001"
        assert settlement.symbol == "AAPL"
        assert settlement.side == "buy"
        assert settlement.quantity == Decimal("100")
        assert settlement.price == Decimal("150.00")
        assert settlement.status == SettlementStatus.PENDING
        assert settlement.cash_amount == Decimal("-15000.00")  # Negative = paying

    def test_add_sell_trade(self, tracker: SettlementTracker) -> None:
        """Add a sell trade creates pending settlement with positive cash."""
        settlement = tracker.add_trade(
            trade_id="T002",
            symbol="MSFT",
            side="sell",
            quantity=Decimal("50"),
            price=Decimal("400.00"),
        )

        assert settlement.side == "sell"
        assert settlement.cash_amount == Decimal("20000.00")  # Positive = receiving

    def test_settlement_date_calculation(self, tracker: SettlementTracker) -> None:
        """Settlement date is T+1."""
        trade_date = date(2024, 1, 8)  # Monday
        settlement = tracker.add_trade(
            trade_id="T003",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            trade_date=trade_date,
        )

        assert settlement.trade_date == trade_date
        assert settlement.settlement_date == date(2024, 1, 9)  # Tuesday

    def test_crypto_same_day_settlement(self, tracker: SettlementTracker) -> None:
        """Crypto settles same day."""
        trade_date = date(2024, 1, 8)
        settlement = tracker.add_trade(
            trade_id="T004",
            symbol="BTC/USD",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("50000.00"),
            trade_date=trade_date,
            security_type=SecurityType.CRYPTO,
        )

        assert settlement.settlement_date == trade_date

    def test_get_pending_for_symbol(self, tracker: SettlementTracker) -> None:
        """Get pending settlements for a symbol."""
        tracker.add_trade("T1", "AAPL", "buy", Decimal("100"), Decimal("150"))
        tracker.add_trade("T2", "AAPL", "sell", Decimal("50"), Decimal("155"))
        tracker.add_trade("T3", "MSFT", "buy", Decimal("100"), Decimal("400"))

        aapl_pending = tracker.get_pending_for_symbol("AAPL")
        assert len(aapl_pending) == 2

        msft_pending = tracker.get_pending_for_symbol("MSFT")
        assert len(msft_pending) == 1

    def test_mark_settled(self, tracker: SettlementTracker) -> None:
        """Mark a settlement as complete."""
        settlement = tracker.add_trade(
            "T1", "AAPL", "buy", Decimal("100"), Decimal("150")
        )

        assert settlement.status == SettlementStatus.PENDING

        tracker.mark_settled(settlement.settlement_id)

        updated = tracker.get_settlement(settlement.settlement_id)
        assert updated is not None
        assert updated.status == SettlementStatus.SETTLED
        assert updated.settled_at is not None

    def test_process_settlements(self, tracker: SettlementTracker) -> None:
        """Process settlements due by date."""
        # Create trade from last week
        last_week = date.today() - timedelta(days=7)
        old_settlement = tracker.add_trade(
            "T1", "AAPL", "buy", Decimal("100"), Decimal("150"),
            trade_date=last_week,
        )

        # Create trade from today
        today_settlement = tracker.add_trade(
            "T2", "MSFT", "buy", Decimal("100"), Decimal("400"),
        )

        # Process settlements as of today
        settled_ids = tracker.process_settlements(as_of=date.today())

        # Old settlement should be settled
        assert old_settlement.settlement_id in settled_ids

        # Today's settlement might still be pending (depends on T+1)
        old = tracker.get_settlement(old_settlement.settlement_id)
        assert old is not None
        assert old.status == SettlementStatus.SETTLED

    def test_pending_count(self, tracker: SettlementTracker) -> None:
        """Count pending settlements."""
        assert tracker.pending_count == 0

        tracker.add_trade("T1", "AAPL", "buy", Decimal("100"), Decimal("150"))
        assert tracker.pending_count == 1

        tracker.add_trade("T2", "MSFT", "buy", Decimal("100"), Decimal("400"))
        assert tracker.pending_count == 2


class TestSettledPosition:
    """Tests for settled position calculations."""

    @pytest.fixture
    def tracker(self) -> SettlementTracker:
        """Create a settlement tracker."""
        return SettlementTracker()

    def test_all_settled(self, tracker: SettlementTracker) -> None:
        """Position with no pending buys is fully settled."""
        # No pending trades
        position = tracker.get_settled_position("AAPL", Decimal("100"))

        assert position.total_quantity == Decimal("100")
        assert position.settled_quantity == Decimal("100")
        assert position.unsettled_quantity == Decimal("0")
        assert position.pending_buys == Decimal("0")
        assert position.pending_sells == Decimal("0")

    def test_pending_buy(self, tracker: SettlementTracker) -> None:
        """Pending buy reduces settled quantity."""
        tracker.add_trade("T1", "AAPL", "buy", Decimal("50"), Decimal("150"))

        # Total position is 100, but 50 are pending settlement
        position = tracker.get_settled_position("AAPL", Decimal("100"))

        assert position.total_quantity == Decimal("100")
        assert position.settled_quantity == Decimal("50")
        assert position.unsettled_quantity == Decimal("50")
        assert position.pending_buys == Decimal("50")

    def test_available_to_sell(self, tracker: SettlementTracker) -> None:
        """Available to sell equals settled quantity."""
        tracker.add_trade("T1", "AAPL", "buy", Decimal("30"), Decimal("150"))

        position = tracker.get_settled_position("AAPL", Decimal("100"))

        assert position.available_to_sell == Decimal("70")


class TestCashBalance:
    """Tests for cash balance calculations."""

    @pytest.fixture
    def tracker(self) -> SettlementTracker:
        """Create a settlement tracker."""
        return SettlementTracker()

    def test_no_pending(self, tracker: SettlementTracker) -> None:
        """Cash with no pending trades is fully settled."""
        cash = tracker.get_cash_balance(Decimal("10000"))

        assert cash.total_cash == Decimal("10000")
        assert cash.settled_cash == Decimal("10000")
        assert cash.pending_debits == Decimal("0")
        assert cash.pending_credits == Decimal("0")

    def test_pending_buy_debit(self, tracker: SettlementTracker) -> None:
        """Pending buy creates pending debit."""
        tracker.add_trade("T1", "AAPL", "buy", Decimal("100"), Decimal("50"))

        cash = tracker.get_cash_balance(Decimal("10000"))

        assert cash.total_cash == Decimal("10000")
        assert cash.pending_debits == Decimal("5000")

    def test_pending_sell_credit(self, tracker: SettlementTracker) -> None:
        """Pending sell creates pending credit."""
        tracker.add_trade("T1", "AAPL", "sell", Decimal("100"), Decimal("50"))

        cash = tracker.get_cash_balance(Decimal("10000"))

        assert cash.total_cash == Decimal("10000")
        assert cash.pending_credits == Decimal("5000")
        assert cash.unsettled_cash == Decimal("5000")

    def test_available_cash(self, tracker: SettlementTracker) -> None:
        """Available cash accounts for pending debits."""
        tracker.add_trade("T1", "AAPL", "buy", Decimal("100"), Decimal("50"))

        cash = tracker.get_cash_balance(Decimal("10000"))

        # Available = settled - pending_debits
        assert cash.available_cash == Decimal("5000")


class TestOptionExercise:
    """Tests for option exercise/assignment settlements."""

    @pytest.fixture
    def tracker(self) -> SettlementTracker:
        """Create a settlement tracker."""
        return SettlementTracker()

    def test_call_exercise(self, tracker: SettlementTracker) -> None:
        """Call exercise creates buy settlement at strike."""
        settlement = tracker.add_option_exercise(
            exercise_id="EX001",
            option_symbol="AAPL250117C00150000",
            underlying="AAPL",
            strike=Decimal("150"),
            quantity=1,  # 1 contract = 100 shares
            is_call=True,
            is_assignment=False,
        )

        assert settlement.symbol == "AAPL"
        assert settlement.side == "buy"
        assert settlement.quantity == Decimal("100")
        assert settlement.price == Decimal("150")
        # Cash = -strike * shares (paying for stock)
        assert settlement.cash_amount == Decimal("-15000")
        assert settlement.settlement_type == SettlementType.EXERCISE

    def test_call_assignment(self, tracker: SettlementTracker) -> None:
        """Call assignment creates sell settlement at strike."""
        settlement = tracker.add_option_exercise(
            exercise_id="EX002",
            option_symbol="AAPL250117C00150000",
            underlying="AAPL",
            strike=Decimal("150"),
            quantity=1,
            is_call=True,
            is_assignment=True,
        )

        assert settlement.side == "sell"
        assert settlement.cash_amount == Decimal("15000")  # Receiving cash
        assert settlement.settlement_type == SettlementType.ASSIGNMENT

    def test_put_exercise(self, tracker: SettlementTracker) -> None:
        """Put exercise creates sell settlement at strike."""
        settlement = tracker.add_option_exercise(
            exercise_id="EX003",
            option_symbol="AAPL250117P00150000",
            underlying="AAPL",
            strike=Decimal("150"),
            quantity=1,
            is_call=False,
            is_assignment=False,
        )

        assert settlement.side == "sell"
        assert settlement.cash_amount == Decimal("15000")

    def test_put_assignment(self, tracker: SettlementTracker) -> None:
        """Put assignment creates buy settlement at strike."""
        settlement = tracker.add_option_exercise(
            exercise_id="EX004",
            option_symbol="AAPL250117P00150000",
            underlying="AAPL",
            strike=Decimal("150"),
            quantity=1,
            is_call=False,
            is_assignment=True,
        )

        assert settlement.side == "buy"
        assert settlement.cash_amount == Decimal("-15000")

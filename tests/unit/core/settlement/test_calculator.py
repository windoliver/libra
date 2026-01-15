"""
Unit tests for Buying Power Calculator.

Tests buying power calculations for different account types.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.core.settlement import (
    SettlementTracker,
    BuyingPowerCalculator,
    AccountType,
)


class TestCashAccountBuyingPower:
    """Tests for cash account buying power calculations."""

    @pytest.fixture
    def tracker(self) -> SettlementTracker:
        """Create a settlement tracker."""
        return SettlementTracker()

    @pytest.fixture
    def calculator(self, tracker: SettlementTracker) -> BuyingPowerCalculator:
        """Create a cash account calculator."""
        return BuyingPowerCalculator(tracker, account_type=AccountType.CASH)

    def test_full_cash_available(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Cash account with no pending has full buying power."""
        bp = calculator.calculate(cash_balance=Decimal("10000"))

        assert bp.account_type == AccountType.CASH
        assert bp.buying_power == Decimal("10000")
        assert bp.margin_available == Decimal("0")

    def test_pending_buy_reduces_power(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Pending buy reduces buying power."""
        tracker.add_trade("T1", "AAPL", "buy", Decimal("100"), Decimal("50"))

        bp = calculator.calculate(cash_balance=Decimal("10000"))

        # Buying power = settled_cash - pending_debits
        # But available_cash already accounts for this
        assert bp.buying_power < Decimal("10000")

    def test_can_sell_settled(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Can sell settled shares."""
        allowed, reason = calculator.can_sell(
            symbol="AAPL",
            quantity=Decimal("50"),
            current_position=Decimal("100"),
        )

        assert allowed is True

    def test_cannot_sell_unsettled(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Cannot sell unsettled shares in cash account."""
        # Add a pending buy
        tracker.add_trade("T1", "AAPL", "buy", Decimal("100"), Decimal("150"))

        # Try to sell the unsettled shares
        allowed, reason = calculator.can_sell(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_position=Decimal("100"),
        )

        assert allowed is False
        assert "Good Faith Violation" in reason


class TestMarginAccountBuyingPower:
    """Tests for margin account buying power calculations."""

    @pytest.fixture
    def tracker(self) -> SettlementTracker:
        """Create a settlement tracker."""
        return SettlementTracker()

    @pytest.fixture
    def calculator(self, tracker: SettlementTracker) -> BuyingPowerCalculator:
        """Create a margin account calculator."""
        return BuyingPowerCalculator(tracker, account_type=AccountType.MARGIN)

    def test_margin_leverage(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Margin account has 2x buying power."""
        bp = calculator.calculate(
            cash_balance=Decimal("10000"),
            portfolio_value=Decimal("10000"),
        )

        assert bp.account_type == AccountType.MARGIN
        # Cash + (portfolio * 0.5 margin rate) = 10000 + 5000 = 15000
        assert bp.buying_power == Decimal("15000")
        assert bp.margin_available == Decimal("5000")

    def test_margin_used_reduces_power(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Margin used reduces buying power."""
        bp = calculator.calculate(
            cash_balance=Decimal("10000"),
            portfolio_value=Decimal("10000"),
            margin_used=Decimal("3000"),
        )

        # 10000 + 5000 - 3000 = 12000
        assert bp.buying_power == Decimal("12000")

    def test_can_sell_unsettled_in_margin(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Margin account can sell unsettled shares (no GFV)."""
        tracker.add_trade("T1", "AAPL", "buy", Decimal("100"), Decimal("150"))

        allowed, reason = calculator.can_sell(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_position=Decimal("100"),
        )

        # Margin accounts don't have GFV restrictions
        assert allowed is True


class TestPDTAccountBuyingPower:
    """Tests for Pattern Day Trader buying power calculations."""

    @pytest.fixture
    def tracker(self) -> SettlementTracker:
        """Create a settlement tracker."""
        return SettlementTracker()

    @pytest.fixture
    def calculator(self, tracker: SettlementTracker) -> BuyingPowerCalculator:
        """Create a PDT account calculator."""
        return BuyingPowerCalculator(tracker, account_type=AccountType.PDT)

    def test_pdt_4x_leverage(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """PDT account has 4x day trade buying power."""
        bp = calculator.calculate(
            cash_balance=Decimal("30000"),  # Above $25k minimum
            portfolio_value=Decimal("20000"),
        )

        assert bp.account_type == AccountType.PDT
        # Day trade BP = equity * 4 = (30000 + 20000) * 4 = 200000
        assert bp.day_trade_buying_power == Decimal("200000")

    def test_pdt_below_minimum_warning(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """PDT account below $25k gets warning."""
        bp = calculator.calculate(
            cash_balance=Decimal("20000"),  # Below $25k
            portfolio_value=Decimal("0"),
        )

        assert bp.has_warnings is True
        assert any("25,000" in w for w in bp.warnings)


class TestBuyingPowerValidation:
    """Tests for trade validation."""

    @pytest.fixture
    def tracker(self) -> SettlementTracker:
        """Create a settlement tracker."""
        return SettlementTracker()

    @pytest.fixture
    def calculator(self, tracker: SettlementTracker) -> BuyingPowerCalculator:
        """Create a calculator."""
        return BuyingPowerCalculator(tracker, account_type=AccountType.CASH)

    def test_can_buy_within_power(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Can buy when within buying power."""
        allowed, reason = calculator.can_buy(
            symbol="AAPL",
            quantity=Decimal("10"),
            price=Decimal("150"),
            current_cash=Decimal("10000"),
        )

        assert allowed is True

    def test_cannot_buy_exceeds_power(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Cannot buy when exceeds buying power."""
        allowed, reason = calculator.can_buy(
            symbol="AAPL",
            quantity=Decimal("100"),
            price=Decimal("150"),  # $15,000 needed
            current_cash=Decimal("10000"),  # Only $10,000 available
        )

        assert allowed is False
        assert "Insufficient buying power" in reason

    def test_get_max_shares(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Calculate maximum shares purchasable."""
        max_shares = calculator.get_max_shares(
            price=Decimal("150"),
            current_cash=Decimal("10000"),
        )

        # $10,000 / $150 = 66 shares
        assert max_shares == 66

    def test_cannot_sell_more_than_held(
        self, tracker: SettlementTracker, calculator: BuyingPowerCalculator
    ) -> None:
        """Cannot sell more shares than held."""
        allowed, reason = calculator.can_sell(
            symbol="AAPL",
            quantity=Decimal("150"),
            current_position=Decimal("100"),
        )

        assert allowed is False
        assert "only 100 held" in reason

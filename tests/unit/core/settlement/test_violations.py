"""
Unit tests for Good Faith Violation (GFV) Tracker.

Tests GFV detection and account restriction logic.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from libra.core.settlement import GFVTracker, GFVCheckResult


class TestGFVTracker:
    """Tests for GFVTracker class."""

    @pytest.fixture
    def tracker(self) -> GFVTracker:
        """Create a GFV tracker."""
        return GFVTracker(account_id="ACC001")

    def test_initial_state(self, tracker: GFVTracker) -> None:
        """New tracker has no violations."""
        status = tracker.get_status()

        assert status.violation_count == 0
        assert status.is_restricted is False
        assert status.violations_remaining == 3

    def test_record_sell(self, tracker: GFVTracker) -> None:
        """Record a sell trade."""
        tomorrow = date.today() + timedelta(days=1)

        tracker.record_sell(
            trade_id="T1",
            symbol="AAPL",
            quantity=Decimal("100"),
            price=Decimal("150"),
            settlement_date=tomorrow,
        )

        # No violation yet
        assert tracker.violation_count == 0

    def test_record_buy_with_settled_funds(self, tracker: GFVTracker) -> None:
        """Buy with settled funds creates no GFV risk."""
        tomorrow = date.today() + timedelta(days=1)

        tracker.record_buy(
            trade_id="T1",
            symbol="MSFT",
            quantity=Decimal("50"),
            price=Decimal("400"),
            settlement_date=tomorrow,
            funding_source=None,  # No unsettled funding
        )

        # Check sell would not cause GFV
        result = tracker.check_sell("MSFT", Decimal("50"))
        assert result.would_violate is False

    def test_gfv_scenario(self, tracker: GFVTracker) -> None:
        """Detect GFV: sell bought with unsettled funds."""
        tomorrow = date.today() + timedelta(days=1)
        day_after = date.today() + timedelta(days=2)

        # Step 1: Sell AAPL (proceeds settle tomorrow)
        tracker.record_sell(
            trade_id="SELL1",
            symbol="AAPL",
            quantity=Decimal("100"),
            price=Decimal("150"),
            settlement_date=tomorrow,
        )

        # Step 2: Buy MSFT with unsettled proceeds
        tracker.record_buy(
            trade_id="BUY1",
            symbol="MSFT",
            quantity=Decimal("50"),
            price=Decimal("400"),
            settlement_date=day_after,
            funding_source="SELL1",  # Funded by unsettled AAPL sale
        )

        # Step 3: Try to sell MSFT before AAPL sale settles -> GFV!
        result = tracker.check_sell("MSFT", Decimal("50"))

        assert result.would_violate is True
        assert "GFV" in result.message

    def test_no_gfv_after_settlement(self, tracker: GFVTracker) -> None:
        """No GFV if original funds have settled."""
        yesterday = date.today() - timedelta(days=1)
        tomorrow = date.today() + timedelta(days=1)

        # Sell that already settled
        tracker.record_sell(
            trade_id="SELL1",
            symbol="AAPL",
            quantity=Decimal("100"),
            price=Decimal("150"),
            settlement_date=yesterday,  # Already settled
        )

        # Buy with those (now settled) proceeds
        tracker.record_buy(
            trade_id="BUY1",
            symbol="MSFT",
            quantity=Decimal("50"),
            price=Decimal("400"),
            settlement_date=tomorrow,
            funding_source="SELL1",
        )

        # Selling MSFT is fine - original funds settled
        result = tracker.check_sell("MSFT", Decimal("50"))

        assert result.would_violate is False

    def test_record_violation(self, tracker: GFVTracker) -> None:
        """Record a GFV increases count."""
        tracker.record_violation(
            symbol="MSFT",
            buy_trade_id="BUY1",
            sell_trade_id="SELL2",
            amount=Decimal("20000"),
        )

        status = tracker.get_status()
        assert status.violation_count == 1
        assert status.violations_remaining == 2

    def test_three_violations_restrict(self, tracker: GFVTracker) -> None:
        """Three violations triggers 90-day restriction."""
        for i in range(3):
            tracker.record_violation(
                symbol="AAPL",
                buy_trade_id=f"BUY{i}",
                sell_trade_id=f"SELL{i}",
                amount=Decimal("10000"),
            )

        status = tracker.get_status()
        assert status.violation_count == 3
        assert status.is_restricted is True
        assert status.restriction_end_date is not None

    def test_is_at_risk(self, tracker: GFVTracker) -> None:
        """Two violations means at risk."""
        tracker.record_violation("AAPL", "B1", "S1", Decimal("10000"))
        tracker.record_violation("MSFT", "B2", "S2", Decimal("10000"))

        status = tracker.get_status()
        assert status.is_at_risk is True
        assert status.violations_remaining == 1


class TestGFVCheckResult:
    """Tests for GFVCheckResult."""

    def test_no_violation(self) -> None:
        """Check result for no violation."""
        result = GFVCheckResult(
            would_violate=False,
            risky_quantity=Decimal("0"),
            message="No GFV risk",
        )

        assert result.would_violate is False

    def test_with_violation(self) -> None:
        """Check result with violation."""
        result = GFVCheckResult(
            would_violate=True,
            risky_quantity=Decimal("100"),
            risky_buys=["BUY1", "BUY2"],
            message="Would cause GFV",
        )

        assert result.would_violate is True
        assert len(result.risky_buys) == 2

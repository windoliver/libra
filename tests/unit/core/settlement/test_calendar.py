"""
Unit tests for Settlement Calendar.

Tests T+1 settlement date calculations including weekends and holidays.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from libra.core.settlement import (
    SettlementCalendar,
    SettlementType,
    SecurityType,
    get_settlement_date,
)


class TestSettlementCalendar:
    """Tests for SettlementCalendar class."""

    @pytest.fixture
    def calendar(self) -> SettlementCalendar:
        """Create a settlement calendar."""
        return SettlementCalendar()

    def test_t1_weekday(self, calendar: SettlementCalendar) -> None:
        """T+1 on a weekday settles next business day."""
        # Monday -> Tuesday
        monday = date(2024, 1, 8)  # A Monday
        settlement = calendar.get_settlement_date(monday)
        assert settlement == date(2024, 1, 9)  # Tuesday

    def test_t1_friday_settles_monday(self, calendar: SettlementCalendar) -> None:
        """T+1 on Friday settles Monday (skips weekend)."""
        friday = date(2024, 1, 5)  # A Friday
        settlement = calendar.get_settlement_date(friday)
        assert settlement == date(2024, 1, 8)  # Monday

    def test_same_day_settlement(self, calendar: SettlementCalendar) -> None:
        """Same-day settlement returns trade date."""
        trade_date = date(2024, 1, 10)
        settlement = calendar.get_settlement_date(
            trade_date, SettlementType.SAME_DAY
        )
        assert settlement == trade_date

    def test_crypto_t0(self, calendar: SettlementCalendar) -> None:
        """Crypto settles same day (T+0)."""
        trade_date = date(2024, 1, 10)
        settlement = calendar.get_settlement_date(
            trade_date,
            SettlementType.REGULAR,
            SecurityType.CRYPTO,
        )
        assert settlement == trade_date

    def test_is_business_day_weekday(self, calendar: SettlementCalendar) -> None:
        """Weekdays are business days."""
        monday = date(2024, 1, 8)
        assert calendar.is_business_day(monday) is True

    def test_is_business_day_weekend(self, calendar: SettlementCalendar) -> None:
        """Weekends are not business days."""
        saturday = date(2024, 1, 6)
        sunday = date(2024, 1, 7)
        assert calendar.is_business_day(saturday) is False
        assert calendar.is_business_day(sunday) is False

    def test_business_days_between(self, calendar: SettlementCalendar) -> None:
        """Count business days between dates."""
        monday = date(2024, 1, 8)
        friday = date(2024, 1, 12)
        # Tue, Wed, Thu, Fri = 4 days
        assert calendar.business_days_between(monday, friday) == 4

    def test_business_days_over_weekend(self, calendar: SettlementCalendar) -> None:
        """Count business days spanning a weekend."""
        friday = date(2024, 1, 5)
        next_tuesday = date(2024, 1, 9)
        # Mon, Tue = 2 days
        assert calendar.business_days_between(friday, next_tuesday) == 2

    def test_next_business_day(self, calendar: SettlementCalendar) -> None:
        """Get next business day."""
        friday = date(2024, 1, 5)
        assert calendar.get_next_business_day(friday) == date(2024, 1, 8)

    def test_previous_business_day(self, calendar: SettlementCalendar) -> None:
        """Get previous business day."""
        monday = date(2024, 1, 8)
        assert calendar.get_previous_business_day(monday) == date(2024, 1, 5)

    def test_days_until_settlement(self, calendar: SettlementCalendar) -> None:
        """Calculate days until settlement."""
        trade_date = date(2024, 1, 8)  # Monday
        # Settlement is Tuesday Jan 9
        days = calendar.days_until_settlement(
            trade_date, as_of=date(2024, 1, 8)
        )
        assert days == 1

    def test_days_until_settlement_past(self, calendar: SettlementCalendar) -> None:
        """Days until settlement can be negative if past."""
        trade_date = date(2024, 1, 5)  # Friday, settles Monday Jan 8
        days = calendar.days_until_settlement(
            trade_date, as_of=date(2024, 1, 10)  # Wednesday
        )
        assert days == -2


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_settlement_date(self) -> None:
        """get_settlement_date convenience function works."""
        trade_date = date(2024, 1, 8)
        settlement = get_settlement_date(trade_date)
        assert settlement == date(2024, 1, 9)

    def test_get_settlement_date_crypto(self) -> None:
        """get_settlement_date with crypto returns same day."""
        trade_date = date(2024, 1, 8)
        settlement = get_settlement_date(
            trade_date, security_type=SecurityType.CRYPTO
        )
        assert settlement == trade_date

"""
Settlement Calendar for T+1 Date Calculations.

Calculates settlement dates accounting for weekends and market holidays.
Uses exchange-calendars library for accurate holiday data.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

from libra.core.settlement.models import SecurityType, SettlementType, SETTLEMENT_DAYS


class SettlementCalendar:
    """
    Calculates settlement dates accounting for holidays and weekends.

    Uses exchange-calendars for accurate US market holiday data.

    Example:
        >>> calendar = SettlementCalendar()
        >>> # Trade on Friday -> settles Monday (T+1)
        >>> calendar.get_settlement_date(date(2024, 1, 5))
        date(2024, 1, 8)
        >>> # Trade before holiday -> settles after holiday
        >>> calendar.get_settlement_date(date(2024, 7, 3))  # July 4th holiday
        date(2024, 7, 5)
    """

    def __init__(self, exchange: str = "XNYS"):
        """
        Initialize settlement calendar.

        Args:
            exchange: Exchange code for calendar (default: NYSE)
                     XNYS = NYSE, XNAS = NASDAQ, etc.
        """
        self._exchange = exchange
        self._calendar = None
        self._init_calendar()

    def _init_calendar(self) -> None:
        """Lazy initialize exchange calendar."""
        try:
            import exchange_calendars as xcals

            self._calendar = xcals.get_calendar(self._exchange)
        except ImportError:
            # Fallback to basic weekend-only logic
            self._calendar = None
        except Exception:
            self._calendar = None

    def get_settlement_date(
        self,
        trade_date: date,
        settlement_type: SettlementType = SettlementType.REGULAR,
        security_type: SecurityType = SecurityType.STOCK,
    ) -> date:
        """
        Calculate settlement date for a trade.

        Args:
            trade_date: Date the trade was executed
            settlement_type: Type of settlement (regular, same-day, etc.)
            security_type: Type of security for settlement rules

        Returns:
            Settlement date (next business day for T+1)
        """
        # Same-day settlement (crypto, etc.)
        if settlement_type == SettlementType.SAME_DAY:
            return trade_date

        # Get settlement days for security type
        settlement_days = SETTLEMENT_DAYS.get(security_type, 1)

        # T+0 for crypto
        if settlement_days == 0:
            return trade_date

        # Calculate T+N settlement
        return self._add_business_days(trade_date, settlement_days)

    def _add_business_days(self, start_date: date, days: int) -> date:
        """
        Add business days to a date, skipping weekends and holidays.

        Args:
            start_date: Starting date
            days: Number of business days to add

        Returns:
            Date after adding business days
        """
        current = start_date
        days_added = 0

        while days_added < days:
            current += timedelta(days=1)
            if self.is_business_day(current):
                days_added += 1

        return current

    def is_business_day(self, d: date) -> bool:
        """
        Check if a date is a trading/settlement day.

        Args:
            d: Date to check

        Returns:
            True if date is a business day
        """
        # Weekend check
        if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Holiday check using exchange calendar
        if self._calendar is not None:
            try:
                return self._calendar.is_session(d.strftime("%Y-%m-%d"))
            except Exception:
                pass

        # Fallback: assume weekdays are business days
        return True

    def is_holiday(self, d: date) -> bool:
        """
        Check if a date is a market holiday.

        Args:
            d: Date to check

        Returns:
            True if date is a holiday (market closed)
        """
        if d.weekday() >= 5:
            return False  # Weekends aren't "holidays"

        if self._calendar is not None:
            try:
                return not self._calendar.is_session(d.strftime("%Y-%m-%d"))
            except Exception:
                pass

        return False

    def business_days_between(self, start: date, end: date) -> int:
        """
        Count business days between two dates (exclusive of start).

        Args:
            start: Start date
            end: End date

        Returns:
            Number of business days
        """
        if end <= start:
            return 0

        count = 0
        current = start + timedelta(days=1)

        while current <= end:
            if self.is_business_day(current):
                count += 1
            current += timedelta(days=1)

        return count

    def days_until_settlement(
        self,
        trade_date: date,
        settlement_type: SettlementType = SettlementType.REGULAR,
        security_type: SecurityType = SecurityType.STOCK,
        as_of: date | None = None,
    ) -> int:
        """
        Calculate days until settlement from a reference date.

        Args:
            trade_date: Date the trade was executed
            settlement_type: Type of settlement
            security_type: Type of security
            as_of: Reference date (default: today)

        Returns:
            Number of calendar days until settlement (negative if past)
        """
        settlement_date = self.get_settlement_date(
            trade_date, settlement_type, security_type
        )
        as_of = as_of or date.today()
        return (settlement_date - as_of).days

    def get_next_business_day(self, d: date) -> date:
        """
        Get the next business day after a given date.

        Args:
            d: Starting date

        Returns:
            Next business day
        """
        return self._add_business_days(d, 1)

    def get_previous_business_day(self, d: date) -> date:
        """
        Get the previous business day before a given date.

        Args:
            d: Starting date

        Returns:
            Previous business day
        """
        current = d - timedelta(days=1)
        while not self.is_business_day(current):
            current -= timedelta(days=1)
        return current

    @lru_cache(maxsize=128)
    def get_settlement_dates_for_week(
        self, week_start: date
    ) -> dict[date, date]:
        """
        Get settlement dates for all trades in a week.

        Useful for batch processing and display.

        Args:
            week_start: Monday of the week

        Returns:
            Dict mapping trade date to settlement date
        """
        result = {}
        for i in range(5):  # Monday to Friday
            trade_date = week_start + timedelta(days=i)
            if self.is_business_day(trade_date):
                result[trade_date] = self.get_settlement_date(trade_date)
        return result


# Module-level singleton for convenience
_default_calendar: SettlementCalendar | None = None


def get_settlement_calendar(exchange: str = "XNYS") -> SettlementCalendar:
    """
    Get the default settlement calendar (singleton).

    Args:
        exchange: Exchange code (only used on first call)

    Returns:
        SettlementCalendar instance
    """
    global _default_calendar
    if _default_calendar is None:
        _default_calendar = SettlementCalendar(exchange)
    return _default_calendar


def get_settlement_date(
    trade_date: date,
    settlement_type: SettlementType = SettlementType.REGULAR,
    security_type: SecurityType = SecurityType.STOCK,
) -> date:
    """
    Convenience function to get settlement date.

    Args:
        trade_date: Date the trade was executed
        settlement_type: Type of settlement
        security_type: Type of security

    Returns:
        Settlement date
    """
    return get_settlement_calendar().get_settlement_date(
        trade_date, settlement_type, security_type
    )

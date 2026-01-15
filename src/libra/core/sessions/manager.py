"""
Market Session Manager.

Provides market hours, holiday detection, and session state tracking
using exchange-calendars as the underlying data source.

Issue #62: Stock Market Session Manager
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
from exchange_calendars import ExchangeCalendar

from libra.core.sessions.types import (
    MarketStatus,
    SessionInfo,
    SessionType,
    TradingHours,
    US_AFTER_HOURS,
    US_EARLY_CLOSE,
    US_PRE_MARKET,
    US_REGULAR,
)


logger = logging.getLogger(__name__)


# Exchange codes for common markets
EXCHANGE_NYSE = "XNYS"  # New York Stock Exchange
EXCHANGE_NASDAQ = "XNAS"  # NASDAQ
EXCHANGE_CBOE = "XCBF"  # CBOE Futures
EXCHANGE_CME = "XCME"  # CME (futures)
EXCHANGE_LSE = "XLON"  # London Stock Exchange
EXCHANGE_TSE = "XTKS"  # Tokyo Stock Exchange


# US timezone
ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SessionSchedule:
    """
    Complete session schedule for a trading day.

    Includes pre-market, regular, and after-hours sessions,
    plus information about early closes and holidays.
    """

    date: date
    is_trading_day: bool
    is_holiday: bool
    holiday_name: str | None
    is_early_close: bool

    # Session times (in ET)
    pre_market_open: time | None
    pre_market_close: time | None
    regular_open: time | None
    regular_close: time | None
    after_hours_open: time | None
    after_hours_close: time | None

    @property
    def sessions(self) -> list[TradingHours]:
        """Get list of trading sessions for this day."""
        if not self.is_trading_day:
            return []

        sessions = [US_PRE_MARKET]

        if self.is_early_close:
            sessions.append(US_EARLY_CLOSE)
        else:
            sessions.append(US_REGULAR)

        sessions.append(US_AFTER_HOURS)
        return sessions


class MarketSessionManager:
    """
    Market session manager using exchange-calendars.

    Provides:
        - Market hours queries (is_open, next_open, next_close)
        - Holiday detection (is_holiday, get_holidays)
        - Session state (current session type, time to close)
        - Trading day validation (is_trading_day)

    Supports US equity markets (NYSE/NASDAQ) by default,
    with extensibility for other exchanges.

    Example:
        manager = MarketSessionManager()

        # Check if market is open
        if manager.is_market_open():
            print("Market is open for trading")

        # Get current session info
        info = manager.get_session_info()
        print(f"Status: {info.status.name}")
        print(f"Time to close: {info.time_to_close}s")

        # Check holidays
        holidays = manager.get_holidays(2025)
        for h in holidays:
            print(f"{h['date']}: {h['name']}")
    """

    def __init__(
        self,
        exchange: str = EXCHANGE_NYSE,
        include_extended_hours: bool = True,
    ) -> None:
        """
        Initialize session manager.

        Args:
            exchange: Exchange code (default: NYSE)
            include_extended_hours: Include pre/post market sessions
        """
        self._exchange_code = exchange
        self._include_extended = include_extended_hours
        self._calendar: ExchangeCalendar | None = None
        self._timezone = ET

    @property
    def calendar(self) -> ExchangeCalendar:
        """Get exchange calendar (lazy loaded)."""
        if self._calendar is None:
            self._calendar = xcals.get_calendar(self._exchange_code)
        return self._calendar

    @property
    def timezone(self) -> ZoneInfo:
        """Get exchange timezone."""
        return self._timezone

    def is_trading_day(self, d: date | None = None) -> bool:
        """
        Check if a date is a trading day.

        Args:
            d: Date to check (default: today)

        Returns:
            True if the market is open on this day
        """
        if d is None:
            d = datetime.now(self._timezone).date()

        # Check if date is within calendar range
        if d < self.calendar.first_session.date():
            return False
        if d > self.calendar.last_session.date():
            return False

        return self.calendar.is_session(d)

    def is_holiday(self, d: date | None = None) -> tuple[bool, str | None]:
        """
        Check if a date is a market holiday.

        Args:
            d: Date to check (default: today)

        Returns:
            Tuple of (is_holiday, holiday_name)
        """
        if d is None:
            d = datetime.now(self._timezone).date()

        if self.is_trading_day(d):
            return False, None

        # Check if it's a weekend
        if d.weekday() >= 5:
            return False, None

        # It's a weekday but not a trading day = holiday
        holiday_name = self._get_holiday_name(d)
        return True, holiday_name

    def _get_holiday_name(self, d: date) -> str | None:
        """Get the name of a holiday."""
        # Known US market holidays
        holidays = {
            "New Year's Day": [(1, 1)],
            "Martin Luther King Jr. Day": [(1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21)],
            "Presidents Day": [(2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21)],
            "Good Friday": [],  # Variable date
            "Memorial Day": [(5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31)],
            "Juneteenth": [(6, 19), (6, 18), (6, 20)],
            "Independence Day": [(7, 4), (7, 3), (7, 5)],
            "Labor Day": [(9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7)],
            "Thanksgiving": [(11, 22), (11, 23), (11, 24), (11, 25), (11, 26), (11, 27), (11, 28)],
            "Christmas Day": [(12, 25), (12, 24), (12, 26)],
        }

        month_day = (d.month, d.day)

        for name, dates in holidays.items():
            if month_day in dates:
                return name

        return "Market Holiday"

    def is_early_close_day(self, d: date | None = None) -> bool:
        """
        Check if a date has early market close (1:00 PM ET).

        Early close days are typically:
            - Day before Independence Day (July 3)
            - Day after Thanksgiving (Black Friday)
            - Christmas Eve (Dec 24)
            - New Year's Eve (Dec 31) - sometimes

        Args:
            d: Date to check (default: today)

        Returns:
            True if market closes early on this day
        """
        if d is None:
            d = datetime.now(self._timezone).date()

        if not self.is_trading_day(d):
            return False

        # Known early close days
        month_day = (d.month, d.day)
        early_close_dates = [
            (7, 3),  # Day before July 4
            (12, 24),  # Christmas Eve
        ]

        # Day after Thanksgiving (4th Thursday in November)
        # Check if it's the 4th Friday in November
        if d.month == 11 and d.weekday() == 4:  # Friday in November
            # Find 4th Thursday
            first_day = date(d.year, 11, 1)
            days_until_thursday = (3 - first_day.weekday()) % 7
            fourth_thursday = first_day + timedelta(days=21 + days_until_thursday)
            if d == fourth_thursday + timedelta(days=1):
                return True

        return month_day in early_close_dates

    def get_holidays(self, year: int) -> list[dict[str, date | str]]:
        """
        Get all market holidays for a year.

        Args:
            year: Year to get holidays for

        Returns:
            List of dicts with 'date' and 'name' keys
        """
        holidays = []

        # Iterate through all days in the year
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        current = start

        while current <= end:
            is_hol, name = self.is_holiday(current)
            if is_hol and name:
                holidays.append({"date": current, "name": name})
            current += timedelta(days=1)

        return holidays

    def get_regular_hours(self, d: date | None = None) -> tuple[datetime, datetime] | None:
        """
        Get regular trading hours for a date.

        Args:
            d: Date to get hours for (default: today)

        Returns:
            Tuple of (open_datetime, close_datetime) in ET, or None if closed
        """
        if d is None:
            d = datetime.now(self._timezone).date()

        if not self.is_trading_day(d):
            return None

        if self.is_early_close_day(d):
            open_time = time(9, 30)
            close_time = time(13, 0)
        else:
            open_time = time(9, 30)
            close_time = time(16, 0)

        open_dt = datetime.combine(d, open_time, tzinfo=self._timezone)
        close_dt = datetime.combine(d, close_time, tzinfo=self._timezone)

        return open_dt, close_dt

    def get_extended_hours(self, d: date | None = None) -> dict[str, tuple[datetime, datetime]]:
        """
        Get all trading sessions (including extended hours) for a date.

        Args:
            d: Date to get hours for (default: today)

        Returns:
            Dict with 'pre_market', 'regular', 'after_hours' keys
        """
        if d is None:
            d = datetime.now(self._timezone).date()

        result: dict[str, tuple[datetime, datetime]] = {}

        if not self.is_trading_day(d):
            return result

        # Pre-market: 4:00 AM - 9:30 AM ET
        result["pre_market"] = (
            datetime.combine(d, time(4, 0), tzinfo=self._timezone),
            datetime.combine(d, time(9, 30), tzinfo=self._timezone),
        )

        # Regular session
        regular = self.get_regular_hours(d)
        if regular:
            result["regular"] = regular

        # After-hours: 4:00 PM - 8:00 PM ET (no after-hours on early close days)
        if not self.is_early_close_day(d):
            result["after_hours"] = (
                datetime.combine(d, time(16, 0), tzinfo=self._timezone),
                datetime.combine(d, time(20, 0), tzinfo=self._timezone),
            )

        return result

    def get_current_session(self, dt: datetime | None = None) -> SessionType:
        """
        Get the current trading session type.

        Args:
            dt: Datetime to check (default: now)

        Returns:
            SessionType enum value
        """
        if dt is None:
            dt = datetime.now(self._timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._timezone)

        d = dt.date()
        t = dt.time()

        if not self.is_trading_day(d):
            return SessionType.CLOSED

        # Check which session we're in
        sessions = self.get_extended_hours(d)

        if "pre_market" in sessions:
            pre_open, pre_close = sessions["pre_market"]
            if pre_open.time() <= t < pre_close.time():
                return SessionType.PRE_MARKET

        if "regular" in sessions:
            reg_open, reg_close = sessions["regular"]
            if reg_open.time() <= t < reg_close.time():
                return SessionType.REGULAR

        if "after_hours" in sessions:
            ah_open, ah_close = sessions["after_hours"]
            if ah_open.time() <= t < ah_close.time():
                return SessionType.AFTER_HOURS

        return SessionType.CLOSED

    def get_market_status(self, dt: datetime | None = None) -> MarketStatus:
        """
        Get current market status.

        Args:
            dt: Datetime to check (default: now)

        Returns:
            MarketStatus enum value
        """
        if dt is None:
            dt = datetime.now(self._timezone)

        session = self.get_current_session(dt)

        if session == SessionType.CLOSED:
            is_hol, _ = self.is_holiday(dt.date())
            if is_hol:
                return MarketStatus.HOLIDAY
            return MarketStatus.CLOSED

        if session == SessionType.PRE_MARKET:
            return MarketStatus.PRE_MARKET

        if session == SessionType.AFTER_HOURS:
            return MarketStatus.AFTER_HOURS

        # Regular session
        if self.is_early_close_day(dt.date()):
            return MarketStatus.EARLY_CLOSE
        return MarketStatus.OPEN

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """
        Check if market is currently open (regular session only).

        Args:
            dt: Datetime to check (default: now)

        Returns:
            True if regular trading session is active
        """
        return self.get_current_session(dt) == SessionType.REGULAR

    def is_trading_allowed(
        self,
        dt: datetime | None = None,
        extended_hours: bool = True,
    ) -> bool:
        """
        Check if trading is allowed (including extended hours if enabled).

        Args:
            dt: Datetime to check (default: now)
            extended_hours: Include pre/post market sessions

        Returns:
            True if trading is allowed
        """
        session = self.get_current_session(dt)

        if session == SessionType.REGULAR:
            return True

        if extended_hours and session in (SessionType.PRE_MARKET, SessionType.AFTER_HOURS):
            return True

        return False

    def get_next_open(self, dt: datetime | None = None) -> datetime | None:
        """
        Get the next market open datetime.

        Args:
            dt: Starting datetime (default: now)

        Returns:
            Next market open datetime, or None if already open
        """
        if dt is None:
            dt = datetime.now(self._timezone)

        # If market is open, return None
        if self.is_market_open(dt):
            return None

        # Find next trading day
        current_date = dt.date()

        # If we haven't reached today's open yet, check today
        if self.is_trading_day(current_date):
            hours = self.get_regular_hours(current_date)
            if hours and dt < hours[0]:
                return hours[0]

        # Look for next trading day
        for i in range(1, 10):  # Look up to 10 days ahead
            next_date = current_date + timedelta(days=i)
            if self.is_trading_day(next_date):
                hours = self.get_regular_hours(next_date)
                if hours:
                    return hours[0]

        return None

    def get_next_close(self, dt: datetime | None = None) -> datetime | None:
        """
        Get the next market close datetime.

        Args:
            dt: Starting datetime (default: now)

        Returns:
            Next market close datetime, or None if closed
        """
        if dt is None:
            dt = datetime.now(self._timezone)

        # Get today's hours
        hours = self.get_regular_hours(dt.date())
        if hours and dt < hours[1]:
            return hours[1]

        return None

    def get_session_info(self, dt: datetime | None = None) -> SessionInfo:
        """
        Get comprehensive session information.

        Args:
            dt: Datetime to get info for (default: now)

        Returns:
            SessionInfo with current session details
        """
        if dt is None:
            dt = datetime.now(self._timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._timezone)

        status = self.get_market_status(dt)
        session_type = self.get_current_session(dt)

        # Get session times
        hours = self.get_regular_hours(dt.date())
        session_open = hours[0] if hours else None
        session_close = hours[1] if hours else None

        # Get next open if closed
        next_open = self.get_next_open(dt) if not self.is_market_open(dt) else None

        return SessionInfo(
            status=status,
            session_type=session_type,
            current_time=dt,
            session_open=session_open,
            session_close=session_close,
            next_session_open=next_open,
            is_early_close=self.is_early_close_day(dt.date()),
        )

    def get_schedule(self, d: date | None = None) -> SessionSchedule:
        """
        Get complete session schedule for a date.

        Args:
            d: Date to get schedule for (default: today)

        Returns:
            SessionSchedule with all session information
        """
        if d is None:
            d = datetime.now(self._timezone).date()

        is_trading = self.is_trading_day(d)
        is_hol, holiday_name = self.is_holiday(d)
        is_early = self.is_early_close_day(d)

        if is_trading:
            return SessionSchedule(
                date=d,
                is_trading_day=True,
                is_holiday=False,
                holiday_name=None,
                is_early_close=is_early,
                pre_market_open=time(4, 0),
                pre_market_close=time(9, 30),
                regular_open=time(9, 30),
                regular_close=time(13, 0) if is_early else time(16, 0),
                after_hours_open=None if is_early else time(16, 0),
                after_hours_close=None if is_early else time(20, 0),
            )

        return SessionSchedule(
            date=d,
            is_trading_day=False,
            is_holiday=is_hol,
            holiday_name=holiday_name,
            is_early_close=False,
            pre_market_open=None,
            pre_market_close=None,
            regular_open=None,
            regular_close=None,
            after_hours_open=None,
            after_hours_close=None,
        )

    def validate_order_timing(
        self,
        extended_hours: bool = False,
        dt: datetime | None = None,
    ) -> tuple[bool, str | None]:
        """
        Validate if an order can be placed at the current time.

        Args:
            extended_hours: Allow extended hours trading
            dt: Datetime to check (default: now)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if dt is None:
            dt = datetime.now(self._timezone)

        session = self.get_current_session(dt)

        if session == SessionType.CLOSED:
            is_hol, name = self.is_holiday(dt.date())
            if is_hol:
                return False, f"Market closed for holiday: {name}"
            return False, "Market is closed"

        if session == SessionType.REGULAR:
            return True, None

        if session in (SessionType.PRE_MARKET, SessionType.AFTER_HOURS):
            if extended_hours:
                return True, None
            return False, f"Extended hours trading not enabled (current: {session.name})"

        return False, "Unknown session state"

    def seconds_until_open(self, dt: datetime | None = None) -> float | None:
        """
        Get seconds until market opens.

        Args:
            dt: Starting datetime (default: now)

        Returns:
            Seconds until open, or None if already open
        """
        next_open = self.get_next_open(dt)
        if next_open is None:
            return None

        if dt is None:
            dt = datetime.now(self._timezone)

        return max(0, (next_open - dt).total_seconds())

    def seconds_until_close(self, dt: datetime | None = None) -> float | None:
        """
        Get seconds until market closes.

        Args:
            dt: Starting datetime (default: now)

        Returns:
            Seconds until close, or None if closed
        """
        next_close = self.get_next_close(dt)
        if next_close is None:
            return None

        if dt is None:
            dt = datetime.now(self._timezone)

        return max(0, (next_close - dt).total_seconds())

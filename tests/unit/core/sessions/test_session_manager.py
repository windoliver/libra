"""
Unit tests for Market Session Manager.

Issue #62: Stock Market Session Manager
"""

from __future__ import annotations

from datetime import date, datetime, time
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from libra.core.sessions.types import (
    MarketStatus,
    SessionInfo,
    SessionType,
    US_AFTER_HOURS,
    US_PRE_MARKET,
    US_REGULAR,
)
from libra.core.sessions.manager import (
    ET,
    EXCHANGE_NYSE,
    MarketSessionManager,
    SessionSchedule,
)


# ===========================================================================
# Types Tests
# ===========================================================================


class TestSessionType:
    """Tests for SessionType enum."""

    def test_session_types_exist(self) -> None:
        """All expected session types are defined."""
        assert SessionType.PRE_MARKET
        assert SessionType.REGULAR
        assert SessionType.AFTER_HOURS
        assert SessionType.CLOSED


class TestMarketStatus:
    """Tests for MarketStatus enum."""

    def test_market_statuses_exist(self) -> None:
        """All expected market statuses are defined."""
        assert MarketStatus.OPEN
        assert MarketStatus.CLOSED
        assert MarketStatus.PRE_MARKET
        assert MarketStatus.AFTER_HOURS
        assert MarketStatus.EARLY_CLOSE
        assert MarketStatus.HOLIDAY


class TestTradingHours:
    """Tests for TradingHours dataclass."""

    def test_us_regular_hours(self) -> None:
        """US regular market hours are 9:30 AM - 4:00 PM."""
        assert US_REGULAR.session_type == SessionType.REGULAR
        assert US_REGULAR.open_time == time(9, 30)
        assert US_REGULAR.close_time == time(16, 0)

    def test_us_pre_market_hours(self) -> None:
        """US pre-market hours are 4:00 AM - 9:30 AM."""
        assert US_PRE_MARKET.session_type == SessionType.PRE_MARKET
        assert US_PRE_MARKET.open_time == time(4, 0)
        assert US_PRE_MARKET.close_time == time(9, 30)

    def test_us_after_hours(self) -> None:
        """US after-hours are 4:00 PM - 8:00 PM."""
        assert US_AFTER_HOURS.session_type == SessionType.AFTER_HOURS
        assert US_AFTER_HOURS.open_time == time(16, 0)
        assert US_AFTER_HOURS.close_time == time(20, 0)

    def test_contains_time(self) -> None:
        """TradingHours.contains() checks time bounds."""
        # Within regular session
        assert US_REGULAR.contains(time(10, 0))
        assert US_REGULAR.contains(time(15, 59))

        # At boundaries
        assert US_REGULAR.contains(time(9, 30))
        assert not US_REGULAR.contains(time(16, 0))  # Close is exclusive

        # Outside session
        assert not US_REGULAR.contains(time(9, 0))
        assert not US_REGULAR.contains(time(17, 0))


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_is_trading_during_regular(self) -> None:
        """is_trading returns True during regular session."""
        info = SessionInfo(
            status=MarketStatus.OPEN,
            session_type=SessionType.REGULAR,
            current_time=datetime(2025, 1, 15, 10, 30, tzinfo=ET),
            session_open=datetime(2025, 1, 15, 9, 30, tzinfo=ET),
            session_close=datetime(2025, 1, 15, 16, 0, tzinfo=ET),
            next_session_open=None,
        )
        assert info.is_trading
        assert info.is_regular_session

    def test_is_trading_during_pre_market(self) -> None:
        """is_trading returns True during pre-market."""
        info = SessionInfo(
            status=MarketStatus.PRE_MARKET,
            session_type=SessionType.PRE_MARKET,
            current_time=datetime(2025, 1, 15, 7, 0, tzinfo=ET),
            session_open=datetime(2025, 1, 15, 9, 30, tzinfo=ET),
            session_close=datetime(2025, 1, 15, 16, 0, tzinfo=ET),
            next_session_open=None,
        )
        assert info.is_trading
        assert not info.is_regular_session

    def test_time_to_close(self) -> None:
        """time_to_close returns seconds until close."""
        info = SessionInfo(
            status=MarketStatus.OPEN,
            session_type=SessionType.REGULAR,
            current_time=datetime(2025, 1, 15, 15, 0, tzinfo=ET),
            session_open=datetime(2025, 1, 15, 9, 30, tzinfo=ET),
            session_close=datetime(2025, 1, 15, 16, 0, tzinfo=ET),
            next_session_open=None,
        )
        # 1 hour = 3600 seconds
        assert info.time_to_close == 3600.0

    def test_time_to_open_when_closed(self) -> None:
        """time_to_open returns seconds until next open."""
        info = SessionInfo(
            status=MarketStatus.CLOSED,
            session_type=SessionType.CLOSED,
            current_time=datetime(2025, 1, 15, 20, 0, tzinfo=ET),
            session_open=None,
            session_close=None,
            next_session_open=datetime(2025, 1, 16, 9, 30, tzinfo=ET),
        )
        # 13.5 hours = 48600 seconds
        assert info.time_to_open == 48600.0


# ===========================================================================
# Manager Tests
# ===========================================================================


@pytest.fixture
def mock_calendar() -> MagicMock:
    """Create a mock exchange calendar."""
    cal = MagicMock()
    cal.first_session.date.return_value = date(2020, 1, 1)
    cal.last_session.date.return_value = date(2030, 12, 31)
    return cal


@pytest.fixture
def manager(mock_calendar: MagicMock) -> MarketSessionManager:
    """Create session manager with mocked calendar."""
    with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
        mgr = MarketSessionManager()
        # Force calendar load
        _ = mgr.calendar
    return mgr


class TestMarketSessionManager:
    """Tests for MarketSessionManager."""

    def test_init_default_exchange(self) -> None:
        """Default exchange is NYSE."""
        with patch("exchange_calendars.get_calendar") as mock_get:
            mock_get.return_value = MagicMock()
            mgr = MarketSessionManager()
            _ = mgr.calendar
            mock_get.assert_called_with(EXCHANGE_NYSE)

    def test_timezone_is_eastern(self) -> None:
        """Manager uses Eastern Time."""
        with patch("exchange_calendars.get_calendar"):
            mgr = MarketSessionManager()
            assert mgr.timezone == ZoneInfo("America/New_York")

    def test_is_trading_day_regular_weekday(self, mock_calendar: MagicMock) -> None:
        """Regular weekday is a trading day."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            # Wednesday January 15, 2025
            assert mgr.is_trading_day(date(2025, 1, 15))

    def test_is_trading_day_weekend(self, mock_calendar: MagicMock) -> None:
        """Weekend is not a trading day."""
        mock_calendar.is_session.return_value = False

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            # Saturday January 18, 2025
            assert not mgr.is_trading_day(date(2025, 1, 18))

    def test_is_holiday_christmas(self, mock_calendar: MagicMock) -> None:
        """Christmas is a holiday."""
        mock_calendar.is_session.return_value = False

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            is_hol, name = mgr.is_holiday(date(2025, 12, 25))
            assert is_hol
            assert name == "Christmas Day"

    def test_is_holiday_regular_day(self, mock_calendar: MagicMock) -> None:
        """Regular trading day is not a holiday."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            is_hol, name = mgr.is_holiday(date(2025, 1, 15))
            assert not is_hol
            assert name is None

    def test_is_early_close_july_3(self, mock_calendar: MagicMock) -> None:
        """July 3 is an early close day."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            assert mgr.is_early_close_day(date(2025, 7, 3))

    def test_is_early_close_christmas_eve(self, mock_calendar: MagicMock) -> None:
        """Christmas Eve is an early close day."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            assert mgr.is_early_close_day(date(2025, 12, 24))

    def test_get_regular_hours_normal_day(self, mock_calendar: MagicMock) -> None:
        """Regular hours are 9:30 AM - 4:00 PM."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            hours = mgr.get_regular_hours(date(2025, 1, 15))

            assert hours is not None
            open_dt, close_dt = hours
            assert open_dt.time() == time(9, 30)
            assert close_dt.time() == time(16, 0)

    def test_get_regular_hours_early_close(self, mock_calendar: MagicMock) -> None:
        """Early close hours end at 1:00 PM."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            hours = mgr.get_regular_hours(date(2025, 12, 24))

            assert hours is not None
            open_dt, close_dt = hours
            assert open_dt.time() == time(9, 30)
            assert close_dt.time() == time(13, 0)

    def test_get_extended_hours(self, mock_calendar: MagicMock) -> None:
        """Extended hours includes pre and post market."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            hours = mgr.get_extended_hours(date(2025, 1, 15))

            assert "pre_market" in hours
            assert "regular" in hours
            assert "after_hours" in hours

            pre_open, pre_close = hours["pre_market"]
            assert pre_open.time() == time(4, 0)
            assert pre_close.time() == time(9, 30)

    def test_get_current_session_regular(self, mock_calendar: MagicMock) -> None:
        """Detects regular session correctly."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 10, 30, tzinfo=ET)
            assert mgr.get_current_session(dt) == SessionType.REGULAR

    def test_get_current_session_pre_market(self, mock_calendar: MagicMock) -> None:
        """Detects pre-market session correctly."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 7, 0, tzinfo=ET)
            assert mgr.get_current_session(dt) == SessionType.PRE_MARKET

    def test_get_current_session_after_hours(self, mock_calendar: MagicMock) -> None:
        """Detects after-hours session correctly."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 17, 0, tzinfo=ET)
            assert mgr.get_current_session(dt) == SessionType.AFTER_HOURS

    def test_get_current_session_closed(self, mock_calendar: MagicMock) -> None:
        """Detects closed session correctly."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 21, 0, tzinfo=ET)
            assert mgr.get_current_session(dt) == SessionType.CLOSED

    def test_is_market_open_during_regular(self, mock_calendar: MagicMock) -> None:
        """Market is open during regular session."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 10, 30, tzinfo=ET)
            assert mgr.is_market_open(dt)

    def test_is_market_open_pre_market(self, mock_calendar: MagicMock) -> None:
        """Market is not 'open' during pre-market."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 7, 0, tzinfo=ET)
            assert not mgr.is_market_open(dt)

    def test_is_trading_allowed_with_extended(self, mock_calendar: MagicMock) -> None:
        """Trading allowed during extended hours if enabled."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 7, 0, tzinfo=ET)
            assert mgr.is_trading_allowed(dt, extended_hours=True)
            assert not mgr.is_trading_allowed(dt, extended_hours=False)

    def test_validate_order_timing_regular(self, mock_calendar: MagicMock) -> None:
        """Order validation passes during regular session."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 10, 30, tzinfo=ET)
            valid, msg = mgr.validate_order_timing(dt=dt)
            assert valid
            assert msg is None

    def test_validate_order_timing_extended_disabled(
        self, mock_calendar: MagicMock
    ) -> None:
        """Order validation fails during extended hours if disabled."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 7, 0, tzinfo=ET)
            valid, msg = mgr.validate_order_timing(extended_hours=False, dt=dt)
            assert not valid
            assert "Extended hours" in msg

    def test_validate_order_timing_extended_enabled(
        self, mock_calendar: MagicMock
    ) -> None:
        """Order validation passes during extended hours if enabled."""
        mock_calendar.is_session.return_value = True

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 1, 15, 7, 0, tzinfo=ET)
            valid, msg = mgr.validate_order_timing(extended_hours=True, dt=dt)
            assert valid
            assert msg is None

    def test_validate_order_timing_holiday(self, mock_calendar: MagicMock) -> None:
        """Order validation fails on holidays."""
        mock_calendar.is_session.return_value = False

        with patch("exchange_calendars.get_calendar", return_value=mock_calendar):
            mgr = MarketSessionManager()
            dt = datetime(2025, 12, 25, 10, 30, tzinfo=ET)
            valid, msg = mgr.validate_order_timing(dt=dt)
            assert not valid
            assert "holiday" in msg.lower()


class TestSessionSchedule:
    """Tests for SessionSchedule dataclass."""

    def test_trading_day_schedule(self) -> None:
        """Trading day has all sessions."""
        schedule = SessionSchedule(
            date=date(2025, 1, 15),
            is_trading_day=True,
            is_holiday=False,
            holiday_name=None,
            is_early_close=False,
            pre_market_open=time(4, 0),
            pre_market_close=time(9, 30),
            regular_open=time(9, 30),
            regular_close=time(16, 0),
            after_hours_open=time(16, 0),
            after_hours_close=time(20, 0),
        )

        sessions = schedule.sessions
        assert len(sessions) == 3
        assert sessions[0].session_type == SessionType.PRE_MARKET
        assert sessions[1].session_type == SessionType.REGULAR
        assert sessions[2].session_type == SessionType.AFTER_HOURS

    def test_holiday_schedule(self) -> None:
        """Holiday has no sessions."""
        schedule = SessionSchedule(
            date=date(2025, 12, 25),
            is_trading_day=False,
            is_holiday=True,
            holiday_name="Christmas Day",
            is_early_close=False,
            pre_market_open=None,
            pre_market_close=None,
            regular_open=None,
            regular_close=None,
            after_hours_open=None,
            after_hours_close=None,
        )

        assert schedule.sessions == []
        assert schedule.holiday_name == "Christmas Day"

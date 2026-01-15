"""
Market Status Widget for TUI (Issue #62).

Displays real-time market session status:
- Current market status (Open/Closed/Pre-market/After-hours)
- Time until open/close countdown
- Today's trading schedule
- Upcoming holidays
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Rule, Static

from libra.core.sessions import (
    MarketSessionManager,
    MarketStatus,
    SessionType,
)


if TYPE_CHECKING:
    from libra.core.sessions import SessionSchedule


class MarketStatusBadge(Static):
    """Visual market status indicator with color coding."""

    DEFAULT_CSS = """
    MarketStatusBadge {
        width: auto;
        min-width: 20;
        height: 3;
        padding: 0 2;
        text-align: center;
        border: round $primary-darken-1;
    }

    MarketStatusBadge.open {
        background: $success-darken-2;
        color: $success;
    }

    MarketStatusBadge.closed {
        background: $surface-darken-1;
        color: $text-muted;
    }

    MarketStatusBadge.pre-market {
        background: $primary-darken-2;
        color: $primary-lighten-2;
    }

    MarketStatusBadge.after-hours {
        background: $warning-darken-2;
        color: $warning;
    }

    MarketStatusBadge.holiday {
        background: $error-darken-2;
        color: $error;
    }

    MarketStatusBadge.early-close {
        background: $warning-darken-2;
        color: $warning;
    }
    """

    def __init__(self, status: MarketStatus = MarketStatus.CLOSED, id: str | None = None) -> None:
        super().__init__(id=id)
        self._status = status

    def on_mount(self) -> None:
        self.update_status(self._status)

    def update_status(self, status: MarketStatus) -> None:
        """Update the market status display."""
        self._status = status

        # Remove old classes
        self.remove_class("open", "closed", "pre-market", "after-hours", "holiday", "early-close")

        # Map status to display
        status_display = {
            MarketStatus.OPEN: ("open", "MARKET OPEN"),
            MarketStatus.CLOSED: ("closed", "MARKET CLOSED"),
            MarketStatus.PRE_MARKET: ("pre-market", "PRE-MARKET"),
            MarketStatus.AFTER_HOURS: ("after-hours", "AFTER-HOURS"),
            MarketStatus.HOLIDAY: ("holiday", "HOLIDAY"),
            MarketStatus.EARLY_CLOSE: ("early-close", "EARLY CLOSE"),
        }

        css_class, label = status_display.get(status, ("closed", "UNKNOWN"))
        self.add_class(css_class)
        self.update(label)


class CountdownTimer(Static):
    """Countdown timer showing time until market event."""

    DEFAULT_CSS = """
    CountdownTimer {
        width: auto;
        height: 3;
        padding: 0 2;
        text-align: center;
        border: round $primary-darken-1;
    }

    CountdownTimer .label {
        color: $text-muted;
    }

    CountdownTimer .time {
        color: $text;
        text-style: bold;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._seconds: float | None = None
        self._label = ""

    def update_countdown(self, seconds: float | None, label: str) -> None:
        """Update the countdown display."""
        self._seconds = seconds
        self._label = label

        if seconds is None or seconds <= 0:
            self.update(f"[dim]{label}: --:--:--[/dim]")
            return

        # Format as HH:MM:SS
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 24:
            days = hours // 24
            hours = hours % 24
            time_str = f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            time_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"

        self.update(f"[dim]{label}:[/dim] [bold]{time_str}[/bold]")


class SessionSchedulePanel(Vertical):
    """Panel showing today's trading sessions."""

    DEFAULT_CSS = """
    SessionSchedulePanel {
        height: auto;
        border: round $primary-darken-1;
        padding: 1;
    }

    SessionSchedulePanel > Static.panel-title {
        height: 1;
        color: $text;
        text-style: bold;
    }

    SessionSchedulePanel > Horizontal.session-row {
        height: 1;
        padding: 0 1;
    }

    SessionSchedulePanel > Horizontal.session-row > Static.session-name {
        width: 15;
        color: $text-muted;
    }

    SessionSchedulePanel > Horizontal.session-row > Static.session-time {
        width: 1fr;
        color: $text;
    }

    SessionSchedulePanel > Horizontal.session-row.active > Static {
        color: $success;
        text-style: bold;
    }

    SessionSchedulePanel > Horizontal.session-row.closed > Static {
        color: $text-disabled;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._pre_market_row: Horizontal | None = None
        self._regular_row: Horizontal | None = None
        self._after_hours_row: Horizontal | None = None

    def compose(self) -> ComposeResult:
        yield Static("TODAY'S SCHEDULE", classes="panel-title")
        yield Rule()

        with Horizontal(classes="session-row", id="pre-market-row"):
            yield Static("Pre-Market", classes="session-name")
            yield Static("4:00 AM - 9:30 AM ET", classes="session-time", id="pre-market-time")

        with Horizontal(classes="session-row", id="regular-row"):
            yield Static("Regular", classes="session-name")
            yield Static("9:30 AM - 4:00 PM ET", classes="session-time", id="regular-time")

        with Horizontal(classes="session-row", id="after-hours-row"):
            yield Static("After-Hours", classes="session-name")
            yield Static("4:00 PM - 8:00 PM ET", classes="session-time", id="after-hours-time")

    def on_mount(self) -> None:
        self._pre_market_row = self.query_one("#pre-market-row", Horizontal)
        self._regular_row = self.query_one("#regular-row", Horizontal)
        self._after_hours_row = self.query_one("#after-hours-row", Horizontal)

    def update_schedule(
        self,
        schedule: SessionSchedule,
        current_session: SessionType,
    ) -> None:
        """Update the schedule display."""
        # Update pre-market
        if self._pre_market_row:
            self._pre_market_row.remove_class("active", "closed")
            if not schedule.is_trading_day:
                self._pre_market_row.add_class("closed")
            elif current_session == SessionType.PRE_MARKET:
                self._pre_market_row.add_class("active")

        # Update regular session
        if self._regular_row:
            self._regular_row.remove_class("active", "closed")
            if not schedule.is_trading_day:
                self._regular_row.add_class("closed")
            elif current_session == SessionType.REGULAR:
                self._regular_row.add_class("active")

            # Update time if early close
            try:
                regular_time = self.query_one("#regular-time", Static)
                if schedule.is_early_close:
                    regular_time.update("9:30 AM - 1:00 PM ET [yellow](Early)[/yellow]")
                else:
                    regular_time.update("9:30 AM - 4:00 PM ET")
            except Exception:
                pass

        # Update after-hours
        if self._after_hours_row:
            self._after_hours_row.remove_class("active", "closed")
            if not schedule.is_trading_day or schedule.is_early_close:
                self._after_hours_row.add_class("closed")
            elif current_session == SessionType.AFTER_HOURS:
                self._after_hours_row.add_class("active")


class HolidayTable(Vertical):
    """Table showing upcoming market holidays."""

    DEFAULT_CSS = """
    HolidayTable {
        height: auto;
        min-height: 8;
        max-height: 12;
        border: round $primary-darken-1;
    }

    HolidayTable > Static.panel-title {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }

    HolidayTable > DataTable {
        height: auto;
        max-height: 10;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("UPCOMING HOLIDAYS", classes="panel-title")
        table = DataTable(id="holiday-table")
        table.add_columns("Date", "Day", "Holiday")
        yield table

    def update_holidays(self, holidays: list[dict[str, date | str]]) -> None:
        """Update the holiday table."""
        try:
            table = self.query_one("#holiday-table", DataTable)
            table.clear()

            for holiday in holidays[:5]:  # Show next 5 holidays
                d = holiday["date"]
                if isinstance(d, date):
                    date_str = d.strftime("%b %d")
                    day_str = d.strftime("%a")
                else:
                    date_str = str(d)
                    day_str = "-"

                name = str(holiday.get("name", "Holiday"))
                # Truncate long names
                if len(name) > 20:
                    name = name[:17] + "..."

                table.add_row(date_str, day_str, name)
        except Exception:
            pass


class MarketStatusWidget(Vertical):
    """
    Complete market status widget with real-time updates.

    Shows:
    - Current market status (Open/Closed/Pre-market/After-hours/Holiday)
    - Current time in Eastern Time
    - Countdown to market open/close
    - Today's trading session schedule
    - Upcoming market holidays

    Auto-refreshes every second for real-time countdown.

    Example:
        widget = MarketStatusWidget(id="market-status")

        # Data is automatically fetched from MarketSessionManager
        # No manual updates needed - widget self-updates
    """

    DEFAULT_CSS = """
    MarketStatusWidget {
        height: auto;
        min-height: 20;
        padding: 1;
    }

    MarketStatusWidget > Horizontal.header {
        height: 5;
        margin-bottom: 1;
    }

    MarketStatusWidget > Horizontal.header > Vertical.time-info {
        width: 1fr;
    }

    MarketStatusWidget > Horizontal.header > MarketStatusBadge {
        width: 20;
    }

    MarketStatusWidget > Horizontal.countdown-row {
        height: 5;
        margin-bottom: 1;
    }

    MarketStatusWidget > Horizontal.countdown-row > CountdownTimer {
        width: 1fr;
    }

    MarketStatusWidget > Horizontal.panels {
        height: auto;
    }

    MarketStatusWidget > Horizontal.panels > SessionSchedulePanel {
        width: 1fr;
    }

    MarketStatusWidget > Horizontal.panels > HolidayTable {
        width: 1fr;
    }

    MarketStatusWidget .time-display {
        height: 1;
        color: $text;
    }

    MarketStatusWidget .date-display {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._manager = MarketSessionManager()

        # Cached widget references
        self._status_badge: MarketStatusBadge | None = None
        self._time_display: Static | None = None
        self._date_display: Static | None = None
        self._countdown_open: CountdownTimer | None = None
        self._countdown_close: CountdownTimer | None = None
        self._schedule_panel: SessionSchedulePanel | None = None
        self._holiday_table: HolidayTable | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header"):
            with Vertical(classes="time-info"):
                yield Static("MARKET STATUS", classes="panel-title")
                yield Static("--:--:-- ET", classes="time-display", id="time-display")
                yield Static("---", classes="date-display", id="date-display")
            yield MarketStatusBadge(id="status-badge")

        with Horizontal(classes="countdown-row"):
            yield CountdownTimer(id="countdown-open")
            yield CountdownTimer(id="countdown-close")

        yield Rule()

        with Horizontal(classes="panels"):
            yield SessionSchedulePanel(id="schedule-panel")
            yield HolidayTable(id="holiday-table")

    def on_mount(self) -> None:
        """Cache widget references and start auto-refresh."""
        try:
            self._status_badge = self.query_one("#status-badge", MarketStatusBadge)
            self._time_display = self.query_one("#time-display", Static)
            self._date_display = self.query_one("#date-display", Static)
            self._countdown_open = self.query_one("#countdown-open", CountdownTimer)
            self._countdown_close = self.query_one("#countdown-close", CountdownTimer)
            self._schedule_panel = self.query_one("#schedule-panel", SessionSchedulePanel)
            self._holiday_table = self.query_one("#holiday-table", HolidayTable)
        except Exception:
            pass

        # Initial update
        self._refresh_data()

        # Load holidays
        self._load_holidays()

        # Start auto-refresh (every second for countdown)
        self.set_interval(1.0, self._refresh_data)

    def _refresh_data(self) -> None:
        """Refresh market status data."""
        try:
            now = datetime.now(self._manager.timezone)

            # Update time display
            if self._time_display:
                self._time_display.update(f"{now.strftime('%H:%M:%S')} ET")

            if self._date_display:
                self._date_display.update(now.strftime("%A, %B %d, %Y"))

            # Get market status
            status = self._manager.get_market_status(now)
            if self._status_badge:
                self._status_badge.update_status(status)

            # Update countdowns
            session = self._manager.get_current_session(now)

            if session == SessionType.CLOSED:
                # Show time until market opens
                seconds_to_open = self._manager.seconds_until_open(now)
                if self._countdown_open:
                    self._countdown_open.update_countdown(seconds_to_open, "Opens in")
                if self._countdown_close:
                    self._countdown_close.update_countdown(None, "Closes in")
            else:
                # Show time until session closes
                seconds_to_close = self._manager.seconds_until_close(now)
                if self._countdown_open:
                    self._countdown_open.update_countdown(None, "Opens in")
                if self._countdown_close:
                    self._countdown_close.update_countdown(seconds_to_close, "Closes in")

            # Update schedule panel
            schedule = self._manager.get_schedule(now.date())
            if self._schedule_panel:
                self._schedule_panel.update_schedule(schedule, session)

        except Exception:
            pass

    def _load_holidays(self) -> None:
        """Load upcoming holidays."""
        try:
            now = datetime.now(self._manager.timezone)
            current_year = now.year

            # Get holidays for current and next year
            holidays = self._manager.get_holidays(current_year)
            holidays.extend(self._manager.get_holidays(current_year + 1))

            # Filter to future holidays only
            today = now.date()
            future_holidays = [h for h in holidays if h["date"] >= today]

            # Sort by date
            future_holidays.sort(key=lambda x: x["date"])

            if self._holiday_table:
                self._holiday_table.update_holidays(future_holidays)
        except Exception:
            pass


def create_demo_market_status() -> dict:
    """Create demo market status data for testing."""
    manager = MarketSessionManager()
    now = datetime.now(manager.timezone)

    return {
        "status": manager.get_market_status(now),
        "session": manager.get_current_session(now),
        "schedule": manager.get_schedule(now.date()),
        "info": manager.get_session_info(now),
    }

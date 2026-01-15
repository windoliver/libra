"""
Session types and data structures.

Issue #62: Stock Market Session Manager
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum, auto


class SessionType(Enum):
    """
    Trading session type.

    US Equity Markets:
        - PRE_MARKET: 4:00 AM - 9:30 AM ET
        - REGULAR: 9:30 AM - 4:00 PM ET
        - AFTER_HOURS: 4:00 PM - 8:00 PM ET

    Options Markets (CBOE):
        - REGULAR: 9:30 AM - 4:00 PM ET (most options)
        - EXTENDED: 9:30 AM - 4:15 PM ET (index options like SPX)
    """

    PRE_MARKET = auto()
    REGULAR = auto()
    AFTER_HOURS = auto()
    CLOSED = auto()


class MarketStatus(Enum):
    """
    Current market status.

    Used for real-time market state tracking.
    """

    OPEN = auto()  # Regular session is open
    CLOSED = auto()  # Market is closed
    PRE_MARKET = auto()  # Pre-market session
    AFTER_HOURS = auto()  # After-hours session
    EARLY_CLOSE = auto()  # Holiday early close (1:00 PM ET)
    HOLIDAY = auto()  # Market closed for holiday


@dataclass(frozen=True)
class TradingHours:
    """
    Trading hours for a specific session.

    All times are in the exchange's local timezone (ET for US markets).
    """

    session_type: SessionType
    open_time: time
    close_time: time

    def contains(self, t: time) -> bool:
        """Check if a time falls within this session."""
        return self.open_time <= t < self.close_time


# Standard US equity market hours (Eastern Time)
US_PRE_MARKET = TradingHours(
    session_type=SessionType.PRE_MARKET,
    open_time=time(4, 0),
    close_time=time(9, 30),
)

US_REGULAR = TradingHours(
    session_type=SessionType.REGULAR,
    open_time=time(9, 30),
    close_time=time(16, 0),
)

US_AFTER_HOURS = TradingHours(
    session_type=SessionType.AFTER_HOURS,
    open_time=time(16, 0),
    close_time=time(20, 0),
)

# Early close hours (day before holidays)
US_EARLY_CLOSE = TradingHours(
    session_type=SessionType.REGULAR,
    open_time=time(9, 30),
    close_time=time(13, 0),
)

# Index options extended hours (SPX, VIX, etc.)
OPTIONS_EXTENDED = TradingHours(
    session_type=SessionType.REGULAR,
    open_time=time(9, 30),
    close_time=time(16, 15),
)


@dataclass(frozen=True)
class SessionInfo:
    """
    Information about the current trading session.

    Provides details about the current session state, including
    time remaining and next session information.
    """

    status: MarketStatus
    session_type: SessionType
    current_time: datetime
    session_open: datetime | None
    session_close: datetime | None
    next_session_open: datetime | None
    is_early_close: bool = False

    @property
    def is_trading(self) -> bool:
        """Check if trading is currently allowed."""
        return self.status in (
            MarketStatus.OPEN,
            MarketStatus.PRE_MARKET,
            MarketStatus.AFTER_HOURS,
            MarketStatus.EARLY_CLOSE,
        )

    @property
    def is_regular_session(self) -> bool:
        """Check if regular session is active."""
        return self.status in (MarketStatus.OPEN, MarketStatus.EARLY_CLOSE)

    @property
    def time_to_close(self) -> float | None:
        """Seconds until session close, or None if closed."""
        if self.session_close is None:
            return None
        delta = self.session_close - self.current_time
        return max(0, delta.total_seconds())

    @property
    def time_to_open(self) -> float | None:
        """Seconds until next session open, or None if open."""
        if self.next_session_open is None:
            return None
        if self.is_trading:
            return None
        delta = self.next_session_open - self.current_time
        return max(0, delta.total_seconds())

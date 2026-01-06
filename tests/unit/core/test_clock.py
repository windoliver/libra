"""Unit tests for Clock component."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from libra.core.clock import Clock, ClockType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def live_clock() -> Clock:
    """Create a live clock."""
    return Clock(clock_type=ClockType.LIVE)


@pytest.fixture
def backtest_clock() -> Clock:
    """Create a backtest clock."""
    initial_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return Clock(clock_type=ClockType.BACKTEST, initial_time=initial_time)


# =============================================================================
# Clock Type Tests
# =============================================================================


class TestClockType:
    """Tests for clock type."""

    def test_live_clock_type(self, live_clock: Clock) -> None:
        """Test live clock type."""
        assert live_clock.clock_type == ClockType.LIVE
        assert live_clock.is_live is True
        assert live_clock.is_backtest is False

    def test_backtest_clock_type(self, backtest_clock: Clock) -> None:
        """Test backtest clock type."""
        assert backtest_clock.clock_type == ClockType.BACKTEST
        assert backtest_clock.is_backtest is True
        assert backtest_clock.is_live is False


# =============================================================================
# Time Access Tests
# =============================================================================


class TestTimeAccess:
    """Tests for time access methods."""

    def test_utc_now_live(self, live_clock: Clock) -> None:
        """Test utc_now returns current time in live mode."""
        now = live_clock.utc_now()
        assert isinstance(now, datetime)
        assert now.tzinfo == timezone.utc

    def test_utc_now_backtest(self, backtest_clock: Clock) -> None:
        """Test utc_now returns simulated time in backtest mode."""
        now = backtest_clock.utc_now()
        assert now.year == 2024
        assert now.month == 1
        assert now.day == 1
        assert now.hour == 12

    def test_timestamp_ns(self, live_clock: Clock) -> None:
        """Test timestamp_ns returns nanoseconds."""
        ts = live_clock.timestamp_ns()
        assert isinstance(ts, int)
        assert ts > 0

    def test_timestamp_sec(self, live_clock: Clock) -> None:
        """Test timestamp_sec returns seconds."""
        ts = live_clock.timestamp_sec()
        assert isinstance(ts, float)
        assert ts > 0

    def test_timestamp_ms(self, live_clock: Clock) -> None:
        """Test timestamp_ms returns milliseconds."""
        ts = live_clock.timestamp_ms()
        assert isinstance(ts, int)
        assert ts > 0


# =============================================================================
# Backtest Time Control Tests
# =============================================================================


class TestBacktestTimeControl:
    """Tests for backtest time control."""

    def test_set_time(self, backtest_clock: Clock) -> None:
        """Test setting backtest time."""
        new_time = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        backtest_clock.set_time(new_time)

        assert backtest_clock.utc_now() == new_time

    def test_advance_time(self, backtest_clock: Clock) -> None:
        """Test advancing backtest time."""
        initial = backtest_clock.utc_now()
        backtest_clock.advance_time(timedelta(hours=2))

        expected = initial + timedelta(hours=2)
        assert backtest_clock.utc_now() == expected

    def test_set_time_live_fails(self, live_clock: Clock) -> None:
        """Test that set_time fails in live mode."""
        with pytest.raises(RuntimeError, match="BACKTEST mode"):
            live_clock.set_time(datetime.now(timezone.utc))

    def test_advance_time_live_fails(self, live_clock: Clock) -> None:
        """Test that advance_time fails in live mode."""
        with pytest.raises(RuntimeError, match="BACKTEST mode"):
            live_clock.advance_time(timedelta(hours=1))


# =============================================================================
# Timer Tests
# =============================================================================


class TestTimers:
    """Tests for timer functionality."""

    def test_set_timer(self, live_clock: Clock) -> None:
        """Test setting a timer."""
        callback = lambda: None  # noqa: E731

        async def async_callback() -> None:
            pass

        live_clock.set_timer("test_timer", timedelta(seconds=10), async_callback)

        assert "test_timer" in live_clock.timer_names()

    def test_cancel_timer(self, live_clock: Clock) -> None:
        """Test canceling a timer."""

        async def callback() -> None:
            pass

        live_clock.set_timer("test_timer", timedelta(seconds=10), callback)
        assert live_clock.cancel_timer("test_timer") is True
        assert "test_timer" not in live_clock.timer_names()

    def test_cancel_nonexistent_timer(self, live_clock: Clock) -> None:
        """Test canceling a timer that doesn't exist."""
        assert live_clock.cancel_timer("nonexistent") is False

    def test_cancel_all_timers(self, live_clock: Clock) -> None:
        """Test canceling all timers."""

        async def callback() -> None:
            pass

        live_clock.set_timer("timer1", timedelta(seconds=10), callback)
        live_clock.set_timer("timer2", timedelta(seconds=20), callback)

        count = live_clock.cancel_all_timers()

        assert count == 2
        assert len(live_clock.timer_names()) == 0

    def test_replace_timer(self, live_clock: Clock) -> None:
        """Test that setting timer with same name replaces it."""

        async def callback1() -> None:
            pass

        async def callback2() -> None:
            pass

        live_clock.set_timer("test", timedelta(seconds=10), callback1)
        live_clock.set_timer("test", timedelta(seconds=20), callback2)

        assert len(live_clock.timer_names()) == 1


# =============================================================================
# Alert Tests
# =============================================================================


class TestAlerts:
    """Tests for alert functionality."""

    def test_set_alert(self, live_clock: Clock) -> None:
        """Test setting an alert."""

        async def callback() -> None:
            pass

        alert_time = live_clock.utc_now() + timedelta(minutes=5)
        live_clock.set_alert("test_alert", alert_time, callback)

        assert "test_alert" in live_clock.alert_names()

    def test_cancel_alert(self, live_clock: Clock) -> None:
        """Test canceling an alert."""

        async def callback() -> None:
            pass

        alert_time = live_clock.utc_now() + timedelta(minutes=5)
        live_clock.set_alert("test_alert", alert_time, callback)

        assert live_clock.cancel_alert("test_alert") is True
        assert "test_alert" not in live_clock.alert_names()

    def test_cancel_nonexistent_alert(self, live_clock: Clock) -> None:
        """Test canceling an alert that doesn't exist."""
        assert live_clock.cancel_alert("nonexistent") is False

    def test_cancel_all_alerts(self, live_clock: Clock) -> None:
        """Test canceling all alerts."""

        async def callback() -> None:
            pass

        now = live_clock.utc_now()
        live_clock.set_alert("alert1", now + timedelta(minutes=5), callback)
        live_clock.set_alert("alert2", now + timedelta(minutes=10), callback)

        count = live_clock.cancel_all_alerts()

        assert count == 2
        assert len(live_clock.alert_names()) == 0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestClockLifecycle:
    """Tests for clock lifecycle."""

    def test_start_stop(self, live_clock: Clock) -> None:
        """Test starting and stopping the clock."""
        live_clock.start()
        live_clock.stop()

        # Should be safe to call multiple times
        live_clock.start()
        live_clock.start()
        live_clock.stop()
        live_clock.stop()


# =============================================================================
# Stats Tests
# =============================================================================


class TestClockStats:
    """Tests for clock statistics."""

    def test_stats(self, live_clock: Clock) -> None:
        """Test clock statistics."""

        async def callback() -> None:
            pass

        live_clock.set_timer("timer1", timedelta(seconds=10), callback)
        live_clock.set_alert("alert1", live_clock.utc_now() + timedelta(minutes=5), callback)

        stats = live_clock.stats()

        assert stats["type"] == "live"
        assert stats["timers"] == 1
        assert stats["alerts"] == 1
        assert "current_time" in stats


# =============================================================================
# Integration Tests
# =============================================================================


class TestClockIntegration:
    """Integration tests for clock."""

    @pytest.mark.asyncio
    async def test_timer_triggers(self, live_clock: Clock) -> None:
        """Test that timer actually triggers."""
        triggered = []

        async def callback() -> None:
            triggered.append(True)

        live_clock.set_timer("fast_timer", timedelta(milliseconds=50), callback, repeat=False)
        live_clock.start()

        # Wait for timer to trigger
        await asyncio.sleep(0.1)

        live_clock.stop()
        assert len(triggered) == 1

    @pytest.mark.asyncio
    async def test_alert_triggers(self, live_clock: Clock) -> None:
        """Test that alert actually triggers."""
        triggered = []

        async def callback() -> None:
            triggered.append(True)

        alert_time = live_clock.utc_now() + timedelta(milliseconds=50)
        live_clock.set_alert("fast_alert", alert_time, callback)
        live_clock.start()

        # Wait for alert to trigger
        await asyncio.sleep(0.1)

        live_clock.stop()
        assert len(triggered) == 1

    @pytest.mark.asyncio
    async def test_repeating_timer(self, live_clock: Clock) -> None:
        """Test that repeating timer triggers multiple times."""
        triggered = []

        async def callback() -> None:
            triggered.append(True)

        live_clock.set_timer("repeating", timedelta(milliseconds=30), callback, repeat=True)
        live_clock.start()

        # Wait for multiple triggers
        await asyncio.sleep(0.1)

        live_clock.stop()
        assert len(triggered) >= 2  # Should trigger at least twice

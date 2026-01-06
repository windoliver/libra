"""
Clock: Time and scheduling utilities for actors and strategies.

Provides:
- Current time access (UTC, nanoseconds)
- Timer scheduling
- Time alerts
- Backtest time simulation support

Design references:
- NautilusTrader clock pattern
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ClockType(Enum):
    """Type of clock."""

    LIVE = "live"  # Real-time clock
    BACKTEST = "backtest"  # Simulated clock for backtesting


@dataclass
class Timer:
    """Timer configuration."""

    name: str
    interval: timedelta
    callback: Callable[[], Coroutine[Any, Any, None]]
    next_trigger: datetime
    repeat: bool = True
    task: asyncio.Task[None] | None = None


@dataclass
class Alert:
    """Time alert configuration."""

    name: str
    trigger_time: datetime
    callback: Callable[[], Coroutine[Any, Any, None]]
    task: asyncio.Task[None] | None = None


class Clock:
    """
    Clock for time access and scheduling.

    Supports:
    - Live mode: Real-time clock
    - Backtest mode: Simulated time (manually advanced)

    Example:
        clock = Clock()

        # Get current time
        now = clock.utc_now()
        ts_ns = clock.timestamp_ns()

        # Schedule a timer
        async def on_timer():
            print("Timer triggered!")

        clock.set_timer("my_timer", timedelta(seconds=10), on_timer)

        # Set a time alert
        async def on_alert():
            print("Alert!")

        clock.set_alert("my_alert", clock.utc_now() + timedelta(minutes=5), on_alert)
    """

    def __init__(
        self,
        clock_type: ClockType = ClockType.LIVE,
        initial_time: datetime | None = None,
    ) -> None:
        """
        Initialize clock.

        Args:
            clock_type: Type of clock (LIVE or BACKTEST)
            initial_time: Initial time for backtest mode
        """
        self._type = clock_type
        self._timers: dict[str, Timer] = {}
        self._alerts: dict[str, Alert] = {}
        self._running = False

        # Backtest time (only used in BACKTEST mode)
        if clock_type == ClockType.BACKTEST:
            self._backtest_time = initial_time or datetime.now(timezone.utc)
        else:
            self._backtest_time = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def clock_type(self) -> ClockType:
        """Clock type (LIVE or BACKTEST)."""
        return self._type

    @property
    def is_live(self) -> bool:
        """Check if using live clock."""
        return self._type == ClockType.LIVE

    @property
    def is_backtest(self) -> bool:
        """Check if using backtest clock."""
        return self._type == ClockType.BACKTEST

    # =========================================================================
    # Time Access
    # =========================================================================

    def utc_now(self) -> datetime:
        """
        Get current UTC time.

        Returns:
            Current datetime in UTC
        """
        if self._type == ClockType.BACKTEST:
            return self._backtest_time or datetime.now(timezone.utc)
        return datetime.now(timezone.utc)

    def timestamp_ns(self) -> int:
        """
        Get current timestamp in nanoseconds.

        Returns:
            Nanoseconds since Unix epoch
        """
        if self._type == ClockType.BACKTEST:
            dt = self._backtest_time or datetime.now(timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
        return time.time_ns()

    def timestamp_sec(self) -> float:
        """
        Get current timestamp in seconds.

        Returns:
            Seconds since Unix epoch
        """
        if self._type == ClockType.BACKTEST:
            dt = self._backtest_time or datetime.now(timezone.utc)
            return dt.timestamp()
        return time.time()

    def timestamp_ms(self) -> int:
        """
        Get current timestamp in milliseconds.

        Returns:
            Milliseconds since Unix epoch
        """
        return self.timestamp_ns() // 1_000_000

    # =========================================================================
    # Backtest Time Control
    # =========================================================================

    def set_time(self, dt: datetime) -> None:
        """
        Set backtest time (only valid in BACKTEST mode).

        Args:
            dt: Time to set

        Raises:
            RuntimeError: If not in backtest mode
        """
        if self._type != ClockType.BACKTEST:
            raise RuntimeError("set_time() only valid in BACKTEST mode")

        self._backtest_time = dt
        self._check_alerts()

    def advance_time(self, delta: timedelta) -> None:
        """
        Advance backtest time (only valid in BACKTEST mode).

        Args:
            delta: Time to advance

        Raises:
            RuntimeError: If not in backtest mode
        """
        if self._type != ClockType.BACKTEST:
            raise RuntimeError("advance_time() only valid in BACKTEST mode")

        if self._backtest_time:
            self._backtest_time += delta
            self._check_alerts()

    def _check_alerts(self) -> None:
        """Check and trigger alerts in backtest mode."""
        if self._backtest_time is None:
            return

        triggered: list[str] = []
        for name, alert in self._alerts.items():
            if alert.trigger_time <= self._backtest_time:
                triggered.append(name)
                # In backtest, run callback synchronously
                asyncio.create_task(alert.callback())

        for name in triggered:
            self._alerts.pop(name, None)

    # =========================================================================
    # Timer Management
    # =========================================================================

    def set_timer(
        self,
        name: str,
        interval: timedelta,
        callback: Callable[[], Coroutine[Any, Any, None]],
        repeat: bool = True,
    ) -> None:
        """
        Set a timer.

        Args:
            name: Timer name (unique identifier)
            interval: Time between triggers
            callback: Async function to call on trigger
            repeat: If True, timer repeats; if False, one-shot
        """
        # Cancel existing timer with same name
        if name in self._timers:
            self.cancel_timer(name)

        next_trigger = self.utc_now() + interval
        timer = Timer(
            name=name,
            interval=interval,
            callback=callback,
            next_trigger=next_trigger,
            repeat=repeat,
        )

        self._timers[name] = timer

        if self._type == ClockType.LIVE and self._running:
            self._start_timer_task(timer)

        logger.debug("Timer '%s' set for %s", name, interval)

    def cancel_timer(self, name: str) -> bool:
        """
        Cancel a timer.

        Args:
            name: Timer name

        Returns:
            True if timer was found and canceled
        """
        timer = self._timers.pop(name, None)
        if timer:
            if timer.task and not timer.task.done():
                timer.task.cancel()
            logger.debug("Timer '%s' canceled", name)
            return True
        return False

    def cancel_all_timers(self) -> int:
        """
        Cancel all timers.

        Returns:
            Number of timers canceled
        """
        count = len(self._timers)
        for name in list(self._timers.keys()):
            self.cancel_timer(name)
        return count

    def _start_timer_task(self, timer: Timer) -> None:
        """Start the async task for a timer."""

        async def timer_loop() -> None:
            while timer.name in self._timers:
                # Calculate sleep duration
                now = self.utc_now()
                sleep_seconds = (timer.next_trigger - now).total_seconds()

                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

                # Trigger callback
                try:
                    await timer.callback()
                except Exception:
                    logger.exception("Error in timer '%s' callback", timer.name)

                if not timer.repeat:
                    self._timers.pop(timer.name, None)
                    break

                # Schedule next trigger
                timer.next_trigger = self.utc_now() + timer.interval

        timer.task = asyncio.create_task(timer_loop())

    # =========================================================================
    # Alert Management
    # =========================================================================

    def set_alert(
        self,
        name: str,
        trigger_time: datetime,
        callback: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Set a time alert.

        Args:
            name: Alert name (unique identifier)
            trigger_time: When to trigger the alert
            callback: Async function to call on trigger
        """
        # Cancel existing alert with same name
        if name in self._alerts:
            self.cancel_alert(name)

        alert = Alert(
            name=name,
            trigger_time=trigger_time,
            callback=callback,
        )

        self._alerts[name] = alert

        if self._type == ClockType.LIVE and self._running:
            self._start_alert_task(alert)

        logger.debug("Alert '%s' set for %s", name, trigger_time)

    def cancel_alert(self, name: str) -> bool:
        """
        Cancel an alert.

        Args:
            name: Alert name

        Returns:
            True if alert was found and canceled
        """
        alert = self._alerts.pop(name, None)
        if alert:
            if alert.task and not alert.task.done():
                alert.task.cancel()
            logger.debug("Alert '%s' canceled", name)
            return True
        return False

    def cancel_all_alerts(self) -> int:
        """
        Cancel all alerts.

        Returns:
            Number of alerts canceled
        """
        count = len(self._alerts)
        for name in list(self._alerts.keys()):
            self.cancel_alert(name)
        return count

    def _start_alert_task(self, alert: Alert) -> None:
        """Start the async task for an alert."""

        async def alert_task() -> None:
            now = self.utc_now()
            sleep_seconds = (alert.trigger_time - now).total_seconds()

            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

            # Trigger callback
            try:
                await alert.callback()
            except Exception:
                logger.exception("Error in alert '%s' callback", alert.name)
            finally:
                self._alerts.pop(alert.name, None)

        alert.task = asyncio.create_task(alert_task())

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the clock (activates timers and alerts in live mode)."""
        if self._running:
            return

        self._running = True

        if self._type == ClockType.LIVE:
            # Start all pending timers
            for timer in self._timers.values():
                self._start_timer_task(timer)

            # Start all pending alerts
            for alert in self._alerts.values():
                self._start_alert_task(alert)

        logger.debug("Clock started in %s mode", self._type.value)

    def stop(self) -> None:
        """Stop the clock (cancels all timers and alerts)."""
        if not self._running:
            return

        self._running = False
        self.cancel_all_timers()
        self.cancel_all_alerts()
        logger.debug("Clock stopped")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def timer_names(self) -> list[str]:
        """Get list of active timer names."""
        return list(self._timers.keys())

    def alert_names(self) -> list[str]:
        """Get list of pending alert names."""
        return list(self._alerts.keys())

    def stats(self) -> dict[str, Any]:
        """Get clock statistics."""
        return {
            "type": self._type.value,
            "running": self._running,
            "timers": len(self._timers),
            "alerts": len(self._alerts),
            "current_time": self.utc_now().isoformat(),
        }

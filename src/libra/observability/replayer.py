"""
Event Replay: Replay recorded event streams for debugging.

Provides:
- Replay recordings at variable speed
- Deterministic replay for reproducibility
- Time-based seeking
- Event filtering during replay

See: https://github.com/windoliver/libra/issues/25
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from libra.core.events import Event

from libra.core.events import Event as EventClass, EventType
from libra.observability.recorder import EventRecorder, Recording

logger = logging.getLogger(__name__)


class ReplayMode(str, Enum):
    """Replay modes."""

    REALTIME = "realtime"  # Replay at original speed
    FAST = "fast"  # Replay as fast as possible
    STEP = "step"  # Step through events manually


@dataclass
class ReplayConfig:
    """
    Configuration for event replay.

    Attributes:
        speed: Replay speed multiplier (1.0 = realtime)
        mode: Replay mode
        start_time: Start from this time in recording
        end_time: End at this time in recording
        event_types: Only replay these event types
        exclude_types: Skip these event types
        deterministic: Use deterministic timing
    """

    speed: float = 1.0
    mode: ReplayMode = ReplayMode.REALTIME
    start_time: float | None = None
    end_time: float | None = None
    event_types: list[str] | None = None
    exclude_types: list[str] | None = None
    deterministic: bool = False


@dataclass
class ReplayStats:
    """Statistics from replay."""

    total_events: int = 0
    replayed_events: int = 0
    skipped_events: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    errors: int = 0

    @property
    def duration_seconds(self) -> float:
        """Replay duration in seconds."""
        if self.end_time == 0:
            return 0.0
        return self.end_time - self.start_time

    @property
    def events_per_second(self) -> float:
        """Events replayed per second."""
        if self.duration_seconds == 0:
            return 0.0
        return self.replayed_events / self.duration_seconds


class EventReplayer:
    """
    Replays recorded event streams.

    Features:
    - Variable speed replay
    - Time-based seeking
    - Event filtering
    - Async iterator interface

    Example:
        replayer = EventReplayer(recorder)

        # Replay at 2x speed
        config = ReplayConfig(speed=2.0)
        async for event in replayer.replay(recording_id, config):
            await message_bus.publish(event)

        # Replay specific event types
        config = ReplayConfig(event_types=["ORDER_FILLED", "SIGNAL"])
        async for event in replayer.replay(recording_id, config):
            process_event(event)
    """

    def __init__(self, recorder: EventRecorder | None = None) -> None:
        self._recorder = recorder or EventRecorder()
        self._current_replay: str | None = None
        self._paused = False
        self._stopped = False

    async def replay(
        self,
        recording_id: str,
        config: ReplayConfig | None = None,
    ) -> AsyncIterator[Event]:
        """
        Replay a recording as async iterator.

        Args:
            recording_id: Recording to replay
            config: Replay configuration

        Yields:
            Events from the recording
        """
        config = config or ReplayConfig()

        # Load recording
        recording = self._recorder.load_recording(recording_id)
        if recording is None:
            raise ValueError(f"Recording not found: {recording_id}")

        self._current_replay = recording_id
        self._paused = False
        self._stopped = False

        events = recording.events
        if not events:
            return

        # Filter events by time
        if config.start_time:
            events = [e for e in events if e["timestamp_ns"] / 1e9 >= config.start_time]
        if config.end_time:
            events = [e for e in events if e["timestamp_ns"] / 1e9 <= config.end_time]

        # Filter by event types
        if config.event_types:
            events = [e for e in events if e["event_type"] in config.event_types]
        if config.exclude_types:
            events = [e for e in events if e["event_type"] not in config.exclude_types]

        # Sort by timestamp
        events = sorted(events, key=lambda e: e["timestamp_ns"])

        logger.info(
            f"Starting replay: {recording_id} "
            f"({len(events)} events, speed={config.speed}x)"
        )

        last_event_time: float | None = None
        replay_start_time = time.time()

        for event_data in events:
            if self._stopped:
                break

            # Handle pause
            while self._paused and not self._stopped:
                await asyncio.sleep(0.1)

            if self._stopped:
                break

            # Calculate delay based on replay mode
            if config.mode == ReplayMode.REALTIME:
                event_time = event_data["timestamp_ns"] / 1e9

                if last_event_time is not None:
                    delay = (event_time - last_event_time) / config.speed
                    if delay > 0:
                        await asyncio.sleep(delay)

                last_event_time = event_time

            elif config.mode == ReplayMode.STEP:
                # In step mode, external code controls advancement
                # Just yield immediately
                pass

            # Create Event from recorded data
            try:
                event = self._create_event(event_data)
                yield event
            except Exception as e:
                logger.error(f"Error creating event: {e}")

        self._current_replay = None
        logger.info(f"Replay complete: {recording_id}")

    def _create_event(self, event_data: dict[str, Any]) -> Event:
        """Create Event from recorded data."""
        event_type = EventType[event_data["event_type"]]

        return EventClass(
            priority=event_data["priority"],
            sequence=event_data["sequence"],
            event_type=event_type,
            timestamp_ns=event_data["timestamp_ns"],
            source=event_data["source"],
            payload=event_data["payload"],
            trace_id=event_data["trace_id"],
            span_id=event_data["span_id"],
        )

    async def replay_to_bus(
        self,
        recording_id: str,
        bus: Any,  # MessageBus
        config: ReplayConfig | None = None,
    ) -> ReplayStats:
        """
        Replay recording directly to message bus.

        Args:
            recording_id: Recording to replay
            bus: MessageBus to publish events to
            config: Replay configuration

        Returns:
            Replay statistics
        """
        stats = ReplayStats(start_time=time.time())

        recording = self._recorder.load_recording(recording_id)
        if recording:
            stats.total_events = len(recording.events)

        try:
            async for event in self.replay(recording_id, config):
                try:
                    await bus.publish(event)
                    stats.replayed_events += 1
                except Exception as e:
                    logger.error(f"Error publishing event: {e}")
                    stats.errors += 1
        finally:
            stats.end_time = time.time()
            stats.skipped_events = stats.total_events - stats.replayed_events

        return stats

    def pause(self) -> None:
        """Pause current replay."""
        self._paused = True
        logger.info("Replay paused")

    def resume(self) -> None:
        """Resume paused replay."""
        self._paused = False
        logger.info("Replay resumed")

    def stop(self) -> None:
        """Stop current replay."""
        self._stopped = True
        logger.info("Replay stopped")

    @property
    def is_replaying(self) -> bool:
        """Check if replay is active."""
        return self._current_replay is not None

    @property
    def is_paused(self) -> bool:
        """Check if replay is paused."""
        return self._paused

    @property
    def current_replay(self) -> str | None:
        """Get current replay ID."""
        return self._current_replay


async def quick_replay(
    recording_id: str,
    recorder: EventRecorder | None = None,
    speed: float = 1.0,
) -> list[Event]:
    """
    Quick replay utility for testing.

    Replays all events and returns them as list.

    Args:
        recording_id: Recording to replay
        recorder: Event recorder
        speed: Replay speed

    Returns:
        List of events
    """
    replayer = EventReplayer(recorder)
    config = ReplayConfig(speed=speed, mode=ReplayMode.FAST)

    events = []
    async for event in replayer.replay(recording_id, config):
        events.append(event)

    return events

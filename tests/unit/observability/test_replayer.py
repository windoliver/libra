"""
Tests for event replay (Issue #25).
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path

from libra.observability.replayer import (
    EventReplayer,
    ReplayConfig,
    ReplayMode,
    ReplayStats,
    quick_replay,
)
from libra.observability.recorder import EventRecorder
from libra.core.events import Event, EventType


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def recorder(temp_storage):
    """Create a recorder with temp storage."""
    return EventRecorder(storage_path=temp_storage, compression=False)


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    events = []
    for i in range(10):
        event = Event.create(
            event_type=EventType.TICK,
            source="test",
            payload={"symbol": "BTC/USDT", "price": 50000 + i},
        )
        events.append(event)
        time.sleep(0.001)  # Small delay for timestamps
    return events


@pytest.fixture
def recording_with_events(recorder, sample_events):
    """Create a recording with sample events."""
    recording_id = recorder.start_recording("Test Recording")
    for event in sample_events:
        recorder.record(event)
    recorder.stop_recording(recording_id)
    return recording_id


class TestReplayConfig:
    """Tests for ReplayConfig."""

    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = ReplayConfig()

        assert config.speed == 1.0
        assert config.mode == ReplayMode.REALTIME
        assert config.deterministic is False

    def test_config_custom(self):
        """Config accepts custom values."""
        config = ReplayConfig(
            speed=2.0,
            mode=ReplayMode.FAST,
            event_types=["TICK", "ORDER_FILLED"],
            exclude_types=["HEARTBEAT"],
        )

        assert config.speed == 2.0
        assert config.mode == ReplayMode.FAST
        assert config.event_types == ["TICK", "ORDER_FILLED"]
        assert config.exclude_types == ["HEARTBEAT"]


class TestReplayStats:
    """Tests for ReplayStats."""

    def test_stats_init(self):
        """Stats initializes correctly."""
        stats = ReplayStats()

        assert stats.total_events == 0
        assert stats.replayed_events == 0
        assert stats.errors == 0

    def test_stats_duration(self):
        """Stats calculates duration."""
        stats = ReplayStats()
        stats.start_time = time.time()
        stats.end_time = stats.start_time + 5.5

        assert stats.duration_seconds == pytest.approx(5.5)

    def test_stats_events_per_second(self):
        """Stats calculates throughput."""
        stats = ReplayStats()
        stats.start_time = time.time()
        stats.end_time = stats.start_time + 2.0
        stats.replayed_events = 100

        assert stats.events_per_second == pytest.approx(50.0)


class TestEventReplayer:
    """Tests for EventReplayer."""

    def test_replayer_init(self, recorder):
        """Replayer initializes correctly."""
        replayer = EventReplayer(recorder)

        assert not replayer.is_replaying
        assert replayer.current_replay is None

    @pytest.mark.asyncio
    async def test_replayer_fast_mode(self, recorder, recording_with_events):
        """Replayer replays in fast mode."""
        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST)

        events = []
        async for event in replayer.replay(recording_with_events, config):
            events.append(event)

        assert len(events) == 10
        assert all(e.event_type == EventType.TICK for e in events)

    @pytest.mark.asyncio
    async def test_replayer_event_order(self, recorder, recording_with_events):
        """Replayer maintains event order."""
        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST)

        prices = []
        async for event in replayer.replay(recording_with_events, config):
            prices.append(event.payload["price"])

        # Prices should be in ascending order
        assert prices == sorted(prices)

    @pytest.mark.asyncio
    async def test_replayer_time_filter(self, recorder, sample_events):
        """Replayer filters by time."""
        # Record events with timestamps
        recording_id = recorder.start_recording("Test")
        for event in sample_events:
            recorder.record(event)
        recorder.stop_recording(recording_id)

        # Get timestamps
        recording = recorder.load_recording(recording_id)
        timestamps = [e["timestamp_ns"] / 1e9 for e in recording.events]
        mid_time = timestamps[5]

        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST, start_time=mid_time)

        count = 0
        async for _ in replayer.replay(recording_id, config):
            count += 1

        assert count <= 5  # Should only get events after midpoint

    @pytest.mark.asyncio
    async def test_replayer_type_filter(self, recorder):
        """Replayer filters by event type."""
        # Create recording with mixed event types
        recording_id = recorder.start_recording("Test")
        recorder.record(Event.create(event_type=EventType.TICK, source="test"))
        recorder.record(Event.create(event_type=EventType.ORDER_FILLED, source="test"))
        recorder.record(Event.create(event_type=EventType.TICK, source="test"))
        recorder.stop_recording(recording_id)

        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST, event_types=["TICK"])

        count = 0
        async for event in replayer.replay(recording_id, config):
            assert event.event_type == EventType.TICK
            count += 1

        assert count == 2

    @pytest.mark.asyncio
    async def test_replayer_exclude_types(self, recorder):
        """Replayer excludes event types."""
        recording_id = recorder.start_recording("Test")
        recorder.record(Event.create(event_type=EventType.TICK, source="test"))
        recorder.record(Event.create(event_type=EventType.ORDER_FILLED, source="test"))
        recorder.record(Event.create(event_type=EventType.TICK, source="test"))
        recorder.stop_recording(recording_id)

        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST, exclude_types=["ORDER_FILLED"])

        count = 0
        async for event in replayer.replay(recording_id, config):
            assert event.event_type != EventType.ORDER_FILLED
            count += 1

        assert count == 2

    @pytest.mark.asyncio
    async def test_replayer_not_found(self, recorder):
        """Replayer raises error for missing recording."""
        replayer = EventReplayer(recorder)

        with pytest.raises(ValueError, match="Recording not found"):
            async for _ in replayer.replay("nonexistent"):
                pass

    @pytest.mark.asyncio
    async def test_replayer_pause_resume(self, recorder, recording_with_events):
        """Replayer pauses and resumes."""
        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST)

        events = []

        async def collect_events():
            async for event in replayer.replay(recording_with_events, config):
                events.append(event)
                if len(events) == 5:
                    replayer.pause()
                    assert replayer.is_paused
                    await asyncio.sleep(0.01)
                    replayer.resume()
                    assert not replayer.is_paused

        await collect_events()
        assert len(events) == 10

    @pytest.mark.asyncio
    async def test_replayer_stop(self, recorder, recording_with_events):
        """Replayer stops early."""
        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST)

        events = []
        async for event in replayer.replay(recording_with_events, config):
            events.append(event)
            if len(events) == 3:
                replayer.stop()

        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_replayer_is_replaying(self, recorder, recording_with_events):
        """Replayer tracks replaying state."""
        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST)

        assert not replayer.is_replaying

        async for _ in replayer.replay(recording_with_events, config):
            assert replayer.is_replaying
            break

    @pytest.mark.asyncio
    async def test_quick_replay(self, recorder, recording_with_events):
        """Quick replay helper works."""
        events = await quick_replay(recording_with_events, recorder)

        assert len(events) == 10
        assert all(e.event_type == EventType.TICK for e in events)


class TestReplayToBus:
    """Tests for replay_to_bus method."""

    @pytest.mark.asyncio
    async def test_replay_to_bus(self, recorder, recording_with_events):
        """Replayer publishes to bus."""

        # Mock bus
        class MockBus:
            def __init__(self):
                self.events = []

            async def publish(self, event):
                self.events.append(event)

        bus = MockBus()
        replayer = EventReplayer(recorder)
        config = ReplayConfig(mode=ReplayMode.FAST)

        stats = await replayer.replay_to_bus(recording_with_events, bus, config)

        assert stats.replayed_events == 10
        assert stats.errors == 0
        assert len(bus.events) == 10

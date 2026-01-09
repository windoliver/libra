"""
Tests for event recording (Issue #25).
"""

import pytest
import tempfile
import time
from pathlib import Path

from libra.observability.recorder import (
    EventRecorder,
    Recording,
    RecordingMetadata,
)
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
def sample_event():
    """Create a sample event for testing."""
    return Event.create(
        event_type=EventType.TICK,
        source="test",
        payload={"symbol": "BTC/USDT", "price": 50000.0},
    )


class TestRecordingMetadata:
    """Tests for RecordingMetadata."""

    def test_metadata_init(self):
        """Metadata initializes correctly."""
        meta = RecordingMetadata(
            recording_id="rec123",
            name="Test Recording",
            start_time=time.time(),
            source="test",
        )

        assert meta.recording_id == "rec123"
        assert meta.name == "Test Recording"
        assert meta.event_count == 0

    def test_metadata_duration(self):
        """Metadata calculates duration."""
        start = time.time()
        meta = RecordingMetadata(
            recording_id="rec123",
            name="Test",
            start_time=start,
        )

        # No end time -> no duration
        assert meta.duration_seconds is None

        meta.end_time = start + 10.5
        assert meta.duration_seconds == pytest.approx(10.5)

    def test_metadata_datetime(self):
        """Metadata provides datetime properties."""
        start = time.time()
        meta = RecordingMetadata(
            recording_id="rec123",
            name="Test",
            start_time=start,
            end_time=start + 60,
        )

        assert meta.start_datetime is not None
        assert meta.end_datetime is not None

    def test_metadata_to_dict(self):
        """Metadata exports to dictionary."""
        meta = RecordingMetadata(
            recording_id="rec123",
            name="Test",
            start_time=time.time(),
            source="gateway",
            tags=["live", "btc"],
        )

        data = meta.to_dict()

        assert data["recording_id"] == "rec123"
        assert data["name"] == "Test"
        assert data["source"] == "gateway"
        assert data["tags"] == ["live", "btc"]


class TestRecording:
    """Tests for Recording."""

    def test_recording_init(self):
        """Recording initializes correctly."""
        meta = RecordingMetadata(
            recording_id="rec123",
            name="Test",
            start_time=time.time(),
        )
        recording = Recording(metadata=meta)

        assert recording.metadata.recording_id == "rec123"
        assert len(recording.events) == 0

    def test_recording_add_event(self):
        """Recording adds events."""
        meta = RecordingMetadata(
            recording_id="rec123",
            name="Test",
            start_time=time.time(),
        )
        recording = Recording(metadata=meta)

        recording.add_event({"event_type": "TICK", "payload": {}})
        recording.add_event({"event_type": "ORDER_FILLED", "payload": {}})

        assert len(recording.events) == 2
        assert recording.metadata.event_count == 2

    def test_recording_to_dict(self):
        """Recording exports to dictionary."""
        meta = RecordingMetadata(
            recording_id="rec123",
            name="Test",
            start_time=time.time(),
        )
        recording = Recording(metadata=meta)
        recording.add_event({"event_type": "TICK"})

        data = recording.to_dict()

        assert "metadata" in data
        assert "events" in data
        assert len(data["events"]) == 1


class TestEventRecorder:
    """Tests for EventRecorder."""

    def test_recorder_init(self, temp_storage):
        """Recorder initializes correctly."""
        recorder = EventRecorder(storage_path=temp_storage)

        assert recorder._storage_path == temp_storage
        assert not recorder.is_recording

    def test_recorder_start_recording(self, recorder):
        """Recorder starts recording."""
        recording_id = recorder.start_recording("Test Session", source="test")

        assert recording_id is not None
        assert len(recording_id) == 16
        assert recorder.is_recording

    def test_recorder_record_event(self, recorder, sample_event):
        """Recorder records events."""
        recording_id = recorder.start_recording("Test")
        recorder.record(sample_event, recording_id)
        recorder.record(sample_event)  # All active recordings

        meta = recorder.stop_recording(recording_id)

        assert meta.event_count == 2

    def test_recorder_stop_recording(self, recorder, sample_event):
        """Recorder stops and saves recording."""
        recording_id = recorder.start_recording("Test")
        recorder.record(sample_event)
        meta = recorder.stop_recording(recording_id)

        assert not recorder.is_recording
        assert meta.end_time is not None
        assert meta.duration_seconds is not None

    def test_recorder_multiple_recordings(self, recorder, sample_event):
        """Recorder handles multiple concurrent recordings."""
        rec1 = recorder.start_recording("Session 1")
        rec2 = recorder.start_recording("Session 2")

        # Record to specific recording
        recorder.record(sample_event, rec1)

        # Record to all
        recorder.record(sample_event)

        meta1 = recorder.stop_recording(rec1)
        meta2 = recorder.stop_recording(rec2)

        assert meta1.event_count == 2  # specific + all
        assert meta2.event_count == 1  # only "all"

    def test_recorder_save_and_load(self, recorder, sample_event):
        """Recorder saves and loads recordings."""
        recording_id = recorder.start_recording("Test", source="test_source")
        recorder.record(sample_event)
        recorder.record(sample_event)
        recorder.stop_recording(recording_id)

        # Load it back
        loaded = recorder.load_recording(recording_id)

        assert loaded is not None
        assert loaded.metadata.name == "Test"
        assert loaded.metadata.source == "test_source"
        assert len(loaded.events) == 2

    def test_recorder_load_nonexistent(self, recorder):
        """Recorder returns None for missing recording."""
        result = recorder.load_recording("nonexistent")
        assert result is None

    def test_recorder_list_recordings(self, recorder, sample_event):
        """Recorder lists all recordings."""
        rec1 = recorder.start_recording("Session 1")
        recorder.record(sample_event)
        recorder.stop_recording(rec1)

        rec2 = recorder.start_recording("Session 2")
        recorder.record(sample_event)
        recorder.stop_recording(rec2)

        recordings = recorder.list_recordings()

        assert len(recordings) == 2
        # Should be sorted by start time (newest first)
        assert recordings[0].name == "Session 2"

    def test_recorder_delete_recording(self, recorder, sample_event):
        """Recorder deletes recordings."""
        recording_id = recorder.start_recording("Test")
        recorder.record(sample_event)
        recorder.stop_recording(recording_id)

        # Delete it
        deleted = recorder.delete_recording(recording_id)
        assert deleted

        # Should be gone
        assert recorder.load_recording(recording_id) is None

        # Delete nonexistent returns False
        assert not recorder.delete_recording("nonexistent")

    def test_recorder_get_active_recordings(self, recorder):
        """Recorder returns active recordings metadata."""
        rec1 = recorder.start_recording("Session 1")
        rec2 = recorder.start_recording("Session 2")

        active = recorder.get_active_recordings()

        assert len(active) == 2

        # Stop one
        recorder.stop_recording(rec1)
        active = recorder.get_active_recordings()

        assert len(active) == 1
        assert active[0].name == "Session 2"

        recorder.stop_recording(rec2)

    def test_recorder_with_compression(self, temp_storage, sample_event):
        """Recorder uses compression."""
        recorder = EventRecorder(storage_path=temp_storage, compression=True)

        recording_id = recorder.start_recording("Compressed")
        recorder.record(sample_event)
        recorder.stop_recording(recording_id)

        # Check file exists with .gz extension
        compressed_file = temp_storage / f"{recording_id}.json.gz"
        assert compressed_file.exists()

        # Should still load correctly
        loaded = recorder.load_recording(recording_id)
        assert loaded is not None
        assert len(loaded.events) == 1

    def test_recorder_tags_and_metadata(self, recorder, sample_event):
        """Recorder stores tags and custom metadata."""
        recording_id = recorder.start_recording(
            "Test",
            tags=["live", "btc", "strategy-a"],
            metadata={"strategy": "sma_cross", "timeframe": "1h"},
        )
        recorder.record(sample_event)
        recorder.stop_recording(recording_id)

        loaded = recorder.load_recording(recording_id)

        assert loaded.metadata.tags == ["live", "btc", "strategy-a"]
        assert loaded.metadata.metadata["strategy"] == "sma_cross"

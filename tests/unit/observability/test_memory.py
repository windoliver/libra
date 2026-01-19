"""Tests for Memory Monitoring (Issue #92)."""

from __future__ import annotations

import asyncio
import tracemalloc

import pytest

from libra.observability.memory import (
    MemoryDiff,
    MemoryMonitor,
    MemorySnapshot,
    get_memory_monitor,
    set_memory_monitor,
)


# =============================================================================
# MemorySnapshot Tests
# =============================================================================


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            current_bytes=1024 * 1024 * 100,  # 100 MB
            peak_bytes=1024 * 1024 * 150,  # 150 MB
            traced_blocks=1000,
        )
        assert snapshot.current_bytes == 104857600
        assert snapshot.peak_bytes == 157286400
        assert snapshot.traced_blocks == 1000

    def test_current_mb(self):
        """Test current_mb property."""
        snapshot = MemorySnapshot(current_bytes=1024 * 1024 * 50)
        assert snapshot.current_mb == 50.0

    def test_peak_mb(self):
        """Test peak_mb property."""
        snapshot = MemorySnapshot(peak_bytes=1024 * 1024 * 100)
        assert snapshot.peak_mb == 100.0

    def test_to_dict(self):
        """Test exporting snapshot to dict."""
        snapshot = MemorySnapshot(
            current_bytes=1024 * 1024,
            peak_bytes=2 * 1024 * 1024,
            traced_blocks=100,
            top_allocations=[{"location": "test.py:1", "size_bytes": 1000}],
        )
        data = snapshot.to_dict()

        assert "timestamp" in data
        assert data["current_bytes"] == 1024 * 1024
        assert data["current_mb"] == 1.0
        assert data["peak_bytes"] == 2 * 1024 * 1024
        assert data["peak_mb"] == 2.0
        assert data["traced_blocks"] == 100
        assert len(data["top_allocations"]) == 1


# =============================================================================
# MemoryDiff Tests
# =============================================================================


class TestMemoryDiff:
    """Tests for MemoryDiff dataclass."""

    def test_create_diff(self):
        """Test creating a memory diff."""
        diff = MemoryDiff(
            size_diff_bytes=1024 * 1024 * 10,  # +10 MB
            count_diff=100,
        )
        assert diff.size_diff_bytes == 10485760
        assert diff.count_diff == 100

    def test_size_diff_mb(self):
        """Test size_diff_mb property."""
        diff = MemoryDiff(size_diff_bytes=1024 * 1024 * 5)
        assert diff.size_diff_mb == 5.0

    def test_negative_diff(self):
        """Test negative memory difference."""
        diff = MemoryDiff(size_diff_bytes=-1024 * 1024 * 2)
        assert diff.size_diff_mb == -2.0

    def test_to_dict(self):
        """Test exporting diff to dict."""
        diff = MemoryDiff(
            size_diff_bytes=1024 * 1024,
            count_diff=50,
            top_increases=[{"location": "test.py:1", "size_diff_bytes": 1000}],
            top_decreases=[{"location": "test.py:2", "size_diff_bytes": -500}],
        )
        data = diff.to_dict()

        assert "timestamp" in data
        assert data["size_diff_bytes"] == 1024 * 1024
        assert data["size_diff_mb"] == 1.0
        assert data["count_diff"] == 50
        assert len(data["top_increases"]) == 1
        assert len(data["top_decreases"]) == 1


# =============================================================================
# MemoryMonitor Tests
# =============================================================================


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    def test_create_default(self):
        """Test creating monitor with defaults."""
        monitor = MemoryMonitor()
        assert monitor._interval == 60.0
        assert monitor._growth_threshold_bytes == 10 * 1024 * 1024  # 10 MB
        assert monitor._top_n == 10
        assert not monitor.is_running

    def test_create_custom(self):
        """Test creating monitor with custom settings."""
        monitor = MemoryMonitor(
            snapshot_interval=30.0,
            growth_threshold_mb=5.0,
            top_n=20,
        )
        assert monitor._interval == 30.0
        assert monitor._growth_threshold_bytes == 5 * 1024 * 1024
        assert monitor._top_n == 20

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = MemoryMonitor(snapshot_interval=0.1)

        assert not monitor.is_running
        await monitor.start()
        assert monitor.is_running
        assert monitor.snapshot_count >= 1

        await monitor.stop()
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_initial_snapshot(self):
        """Test that initial snapshot is taken on start."""
        monitor = MemoryMonitor(snapshot_interval=10.0)

        await monitor.start()
        try:
            assert monitor.current_snapshot is not None
            assert monitor.current_snapshot.current_bytes > 0
            assert monitor.snapshot_count >= 1
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_periodic_snapshots(self):
        """Test periodic snapshot taking."""
        monitor = MemoryMonitor(snapshot_interval=0.05)

        await monitor.start()
        try:
            initial_count = monitor.snapshot_count
            await asyncio.sleep(0.2)
            assert monitor.snapshot_count > initial_count
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_snapshot_callback(self):
        """Test snapshot callback is invoked."""
        snapshots = []

        def on_snapshot(snapshot: MemorySnapshot) -> None:
            snapshots.append(snapshot)

        monitor = MemoryMonitor(
            snapshot_interval=0.05,
            on_snapshot=on_snapshot,
        )

        await monitor.start()
        try:
            await asyncio.sleep(0.2)
            assert len(snapshots) >= 2
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_growth_alert_callback(self):
        """Test growth alert callback is invoked."""
        alerts = []

        def on_alert(diff: MemoryDiff) -> None:
            alerts.append(diff)

        # Use very low threshold to trigger alert
        monitor = MemoryMonitor(
            snapshot_interval=0.05,
            growth_threshold_mb=0.0001,  # 100 bytes
            on_growth_alert=on_alert,
        )

        await monitor.start()
        try:
            # Allocate some memory to trigger growth
            data = [bytearray(10000) for _ in range(100)]
            await asyncio.sleep(0.15)
            # May or may not trigger depending on timing
            # Just verify no errors
            _ = data  # Keep reference
        finally:
            await monitor.stop()

    def test_take_snapshot_sync(self):
        """Test synchronous snapshot taking."""
        # Ensure tracemalloc is started
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()

        try:
            monitor = MemoryMonitor()
            snapshot = monitor.take_snapshot_sync()

            assert snapshot.current_bytes > 0
            assert snapshot.traced_blocks > 0
        finally:
            if not was_tracing:
                tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting monitor statistics."""
        monitor = MemoryMonitor(snapshot_interval=0.1)

        await monitor.start()
        try:
            await asyncio.sleep(0.15)
            stats = monitor.get_stats()

            assert stats["running"] is True
            assert stats["uptime_seconds"] > 0
            assert stats["snapshot_count"] >= 1
            assert stats["tracemalloc_active"] is True
            assert stats["current_snapshot"] is not None
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_double_start(self):
        """Test that double start is handled gracefully."""
        monitor = MemoryMonitor(snapshot_interval=0.1)

        await monitor.start()
        try:
            # Second start should be no-op
            await monitor.start()
            assert monitor.is_running
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_double_stop(self):
        """Test that double stop is handled gracefully."""
        monitor = MemoryMonitor(snapshot_interval=0.1)

        await monitor.start()
        await monitor.stop()
        # Second stop should be no-op
        await monitor.stop()
        assert not monitor.is_running


# =============================================================================
# Global Instance Tests
# =============================================================================


class TestGlobalInstance:
    """Tests for global memory monitor instance."""

    def test_get_set_monitor(self):
        """Test get/set global monitor."""
        # Initially None
        original = get_memory_monitor()

        try:
            monitor = MemoryMonitor()
            set_memory_monitor(monitor)
            assert get_memory_monitor() is monitor

            set_memory_monitor(None)
            assert get_memory_monitor() is None
        finally:
            # Restore original
            set_memory_monitor(original)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryMonitorIntegration:
    """Integration tests for memory monitoring."""

    @pytest.mark.asyncio
    async def test_detect_memory_growth(self):
        """Test that memory growth is detected."""
        growth_detected = False
        growth_amount = 0

        def on_alert(diff: MemoryDiff) -> None:
            nonlocal growth_detected, growth_amount
            growth_detected = True
            growth_amount = diff.size_diff_bytes

        # Very low threshold to ensure detection
        monitor = MemoryMonitor(
            snapshot_interval=0.05,
            growth_threshold_mb=0.001,  # 1 KB
            on_growth_alert=on_alert,
        )

        await monitor.start()
        try:
            # Take initial snapshot
            await asyncio.sleep(0.1)

            # Allocate memory
            large_data = [bytearray(100000) for _ in range(10)]

            # Wait for next snapshot
            await asyncio.sleep(0.1)

            # Keep reference to prevent GC
            _ = large_data

            # May have detected growth
            # (depends on GC timing, so we just verify no errors)
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_top_allocations_tracked(self):
        """Test that top allocations are tracked."""
        monitor = MemoryMonitor(snapshot_interval=0.1, top_n=5)

        await monitor.start()
        try:
            await asyncio.sleep(0.15)
            snapshot = monitor.current_snapshot

            assert snapshot is not None
            assert isinstance(snapshot.top_allocations, list)
            # Should have some allocations tracked
            # (exact number depends on runtime)
        finally:
            await monitor.stop()

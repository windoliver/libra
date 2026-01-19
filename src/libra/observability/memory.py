"""
Memory Monitoring: Tracemalloc periodic snapshots for leak detection.

Provides:
- Periodic memory snapshots using tracemalloc
- Memory growth detection and alerting
- Top allocation hotspot tracking
- Metrics export for observability

See: https://github.com/windoliver/libra/issues/92
"""

from __future__ import annotations

import asyncio
import logging
import tracemalloc
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Snapshot
# =============================================================================


@dataclass
class MemorySnapshot:
    """Point-in-time memory snapshot.

    Attributes:
        timestamp: When snapshot was taken
        current_bytes: Current memory usage
        peak_bytes: Peak memory since start
        traced_blocks: Number of traced memory blocks
        top_allocations: Top memory allocators
    """

    timestamp: float = field(default_factory=time.time)
    current_bytes: int = 0
    peak_bytes: int = 0
    traced_blocks: int = 0
    top_allocations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def current_mb(self) -> float:
        """Current memory in MB."""
        return self.current_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        """Peak memory in MB."""
        return self.peak_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "timestamp": self.timestamp,
            "current_bytes": self.current_bytes,
            "current_mb": self.current_mb,
            "peak_bytes": self.peak_bytes,
            "peak_mb": self.peak_mb,
            "traced_blocks": self.traced_blocks,
            "top_allocations": self.top_allocations,
        }


@dataclass
class MemoryDiff:
    """Memory difference between two snapshots.

    Attributes:
        timestamp: When diff was computed
        size_diff_bytes: Total memory change
        count_diff: Allocation count change
        top_increases: Top memory increases by location
        top_decreases: Top memory decreases by location
    """

    timestamp: float = field(default_factory=time.time)
    size_diff_bytes: int = 0
    count_diff: int = 0
    top_increases: list[dict[str, Any]] = field(default_factory=list)
    top_decreases: list[dict[str, Any]] = field(default_factory=list)

    @property
    def size_diff_mb(self) -> float:
        """Size difference in MB."""
        return self.size_diff_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "timestamp": self.timestamp,
            "size_diff_bytes": self.size_diff_bytes,
            "size_diff_mb": self.size_diff_mb,
            "count_diff": self.count_diff,
            "top_increases": self.top_increases,
            "top_decreases": self.top_decreases,
        }


# =============================================================================
# Memory Monitor
# =============================================================================


class MemoryMonitor:
    """Monitor memory usage with periodic tracemalloc snapshots.

    Features:
    - Periodic memory snapshots at configurable interval
    - Memory growth detection with alerting
    - Top allocation tracking for leak identification
    - Callback support for metrics export

    Example:
        monitor = MemoryMonitor(snapshot_interval=60, growth_threshold_mb=10)
        await monitor.start()

        # Check current memory
        snapshot = monitor.current_snapshot
        print(f"Memory: {snapshot.current_mb:.2f} MB")

        # Stop monitoring
        await monitor.stop()
    """

    def __init__(
        self,
        snapshot_interval: float = 60.0,
        growth_threshold_mb: float = 10.0,
        top_n: int = 10,
        on_snapshot: Callable[[MemorySnapshot], None] | None = None,
        on_growth_alert: Callable[[MemoryDiff], None] | None = None,
    ) -> None:
        """Initialize memory monitor.

        Args:
            snapshot_interval: Seconds between snapshots (default 60)
            growth_threshold_mb: Alert if memory grows more than this (default 10 MB)
            top_n: Number of top allocators to track (default 10)
            on_snapshot: Callback for each snapshot
            on_growth_alert: Callback when growth exceeds threshold
        """
        self._interval = snapshot_interval
        self._growth_threshold_bytes = int(growth_threshold_mb * 1024 * 1024)
        self._top_n = top_n
        self._on_snapshot = on_snapshot
        self._on_growth_alert = on_growth_alert

        self._previous_snapshot: tracemalloc.Snapshot | None = None
        self._current: MemorySnapshot | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._started = False

        # Statistics
        self._snapshot_count = 0
        self._alert_count = 0
        self._start_time: float = 0

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def current_snapshot(self) -> MemorySnapshot | None:
        """Get most recent snapshot."""
        return self._current

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots taken."""
        return self._snapshot_count

    @property
    def alert_count(self) -> int:
        """Number of growth alerts triggered."""
        return self._alert_count

    async def start(self) -> None:
        """Start memory monitoring."""
        if self._running:
            logger.warning("MemoryMonitor already running")
            return

        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started = True
            logger.info("Started tracemalloc")

        self._running = True
        self._start_time = time.time()

        # Take initial snapshot
        await self._take_snapshot()

        # Start periodic task
        self._task = asyncio.create_task(self._periodic_snapshot())
        logger.info(
            "MemoryMonitor started (interval=%ds, threshold=%.1f MB)",
            self._interval,
            self._growth_threshold_bytes / (1024 * 1024),
        )

    async def stop(self) -> None:
        """Stop memory monitoring."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Only stop tracemalloc if we started it
        if self._started and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._started = False
            logger.info("Stopped tracemalloc")

        logger.info(
            "MemoryMonitor stopped (snapshots=%d, alerts=%d)",
            self._snapshot_count,
            self._alert_count,
        )

    async def _periodic_snapshot(self) -> None:
        """Background task for periodic snapshots."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if self._running:
                    await self._take_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error taking memory snapshot: %s", e)

    async def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot and check for growth."""
        snapshot = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()

        # Get top allocators
        top_stats = snapshot.statistics("lineno")[:self._top_n]
        top_allocations = [
            {
                "location": str(stat.traceback),
                "size_bytes": stat.size,
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count,
            }
            for stat in top_stats
        ]

        # Create snapshot record
        self._current = MemorySnapshot(
            current_bytes=current,
            peak_bytes=peak,
            traced_blocks=len(snapshot.traces),
            top_allocations=top_allocations,
        )
        self._snapshot_count += 1

        # Check for memory growth
        if self._previous_snapshot:
            diff = self._compute_diff(snapshot)
            if diff and diff.size_diff_bytes > self._growth_threshold_bytes:
                self._alert_count += 1
                self._log_growth_alert(diff)
                if self._on_growth_alert:
                    try:
                        self._on_growth_alert(diff)
                    except Exception as e:
                        logger.exception("Error in growth alert callback: %s", e)

        # Invoke snapshot callback
        if self._on_snapshot:
            try:
                self._on_snapshot(self._current)
            except Exception as e:
                logger.exception("Error in snapshot callback: %s", e)

        self._previous_snapshot = snapshot
        return self._current

    def _compute_diff(self, snapshot: tracemalloc.Snapshot) -> MemoryDiff | None:
        """Compute difference from previous snapshot."""
        if not self._previous_snapshot:
            return None

        diff_stats = snapshot.compare_to(self._previous_snapshot, "lineno")

        # Separate increases and decreases
        increases = []
        decreases = []
        total_size_diff = 0
        total_count_diff = 0

        for stat in diff_stats:
            total_size_diff += stat.size_diff
            total_count_diff += stat.count_diff

            entry = {
                "location": str(stat.traceback),
                "size_diff_bytes": stat.size_diff,
                "size_diff_mb": stat.size_diff / (1024 * 1024),
                "count_diff": stat.count_diff,
            }

            if stat.size_diff > 0:
                increases.append(entry)
            elif stat.size_diff < 0:
                decreases.append(entry)

        # Sort by absolute size diff
        increases.sort(key=lambda x: x["size_diff_bytes"], reverse=True)
        decreases.sort(key=lambda x: abs(x["size_diff_bytes"]), reverse=True)

        return MemoryDiff(
            size_diff_bytes=total_size_diff,
            count_diff=total_count_diff,
            top_increases=increases[: self._top_n],
            top_decreases=decreases[: self._top_n],
        )

    def _log_growth_alert(self, diff: MemoryDiff) -> None:
        """Log memory growth alert."""
        logger.warning(
            "Memory growth detected: +%.2f MB (%+d allocations)",
            diff.size_diff_mb,
            diff.count_diff,
        )
        for entry in diff.top_increases[:5]:
            if entry["size_diff_bytes"] > 100_000:  # Log entries > 100KB
                logger.warning(
                    "  +%.2f MB: %s",
                    entry["size_diff_mb"],
                    entry["location"][:100],
                )

    def take_snapshot_sync(self) -> MemorySnapshot:
        """Take a snapshot synchronously (for testing)."""
        snapshot = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()

        top_stats = snapshot.statistics("lineno")[:self._top_n]
        top_allocations = [
            {
                "location": str(stat.traceback),
                "size_bytes": stat.size,
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count,
            }
            for stat in top_stats
        ]

        return MemorySnapshot(
            current_bytes=current,
            peak_bytes=peak,
            traced_blocks=len(snapshot.traces),
            top_allocations=top_allocations,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "snapshot_count": self._snapshot_count,
            "alert_count": self._alert_count,
            "interval_seconds": self._interval,
            "growth_threshold_mb": self._growth_threshold_bytes / (1024 * 1024),
            "tracemalloc_active": tracemalloc.is_tracing(),
            "current_snapshot": self._current.to_dict() if self._current else None,
        }


# =============================================================================
# Global Instance
# =============================================================================

_monitor: MemoryMonitor | None = None


def get_memory_monitor() -> MemoryMonitor | None:
    """Get global memory monitor instance."""
    return _monitor


def set_memory_monitor(monitor: MemoryMonitor | None) -> None:
    """Set global memory monitor instance."""
    global _monitor
    _monitor = monitor

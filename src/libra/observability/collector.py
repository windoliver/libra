"""
Metrics Collection: Counters, Gauges, and Histograms.

Provides Prometheus-style metrics collection:
- Counter: Monotonically increasing values
- Gauge: Point-in-time values
- Histogram: Distribution of values with percentiles

Performance optimized for high-frequency trading:
- Lock-free atomic operations where possible
- Pre-allocated histogram buckets
- Efficient percentile calculation

See: https://github.com/windoliver/libra/issues/25
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


# =============================================================================
# Counter
# =============================================================================


class Counter:
    """
    Monotonically increasing counter.

    Thread-safe via lock. Use for:
    - Events published/dispatched/dropped
    - Handler errors
    - Orders executed

    Example:
        events_published = Counter("events_published", "Total events published")
        events_published.inc()
        events_published.inc(10)
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()
        self._labels: dict[tuple[str, ...], float] = defaultdict(float)

    def inc(self, amount: float = 1.0) -> None:
        """Increment counter by amount."""
        with self._lock:
            self._value += amount

    def inc_labeled(self, labels: dict[str, str], amount: float = 1.0) -> None:
        """Increment counter for specific labels."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._labels[key] += amount

    @property
    def value(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value

    def get_labeled(self, labels: dict[str, str]) -> float:
        """Get value for specific labels."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._labels.get(key, 0.0)

    def reset(self) -> None:
        """Reset counter (for testing only)."""
        with self._lock:
            self._value = 0.0
            self._labels.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        with self._lock:
            return {
                "name": self.name,
                "type": MetricType.COUNTER.value,
                "value": self._value,
                "labels": {str(k): v for k, v in self._labels.items()},
            }


# =============================================================================
# Gauge
# =============================================================================


class Gauge:
    """
    Point-in-time value that can go up or down.

    Thread-safe via lock. Use for:
    - Queue sizes
    - Active handlers
    - Memory usage
    - Current positions

    Example:
        queue_size = Gauge("queue_size", "Current queue size")
        queue_size.set(100)
        queue_size.inc(5)
        queue_size.dec(3)
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()
        self._labels: dict[tuple[str, ...], float] = defaultdict(float)

    def set(self, value: float) -> None:
        """Set gauge to specific value."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= amount

    def set_labeled(self, labels: dict[str, str], value: float) -> None:
        """Set gauge for specific labels."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._labels[key] = value

    @property
    def value(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value

    def get_labeled(self, labels: dict[str, str]) -> float:
        """Get value for specific labels."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._labels.get(key, 0.0)

    def reset(self) -> None:
        """Reset gauge (for testing only)."""
        with self._lock:
            self._value = 0.0
            self._labels.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        with self._lock:
            return {
                "name": self.name,
                "type": MetricType.GAUGE.value,
                "value": self._value,
                "labels": {str(k): v for k, v in self._labels.items()},
            }


# =============================================================================
# Histogram
# =============================================================================


@dataclass
class HistogramBuckets:
    """Pre-defined bucket boundaries for histograms."""

    # Default buckets for latencies (in seconds)
    LATENCY_SECONDS: list[float] = field(
        default_factory=lambda: [
            0.001,  # 1ms
            0.005,  # 5ms
            0.01,  # 10ms
            0.025,  # 25ms
            0.05,  # 50ms
            0.1,  # 100ms
            0.25,  # 250ms
            0.5,  # 500ms
            1.0,  # 1s
            2.5,  # 2.5s
            5.0,  # 5s
            10.0,  # 10s
        ]
    )

    # Buckets for microsecond latencies (trading)
    LATENCY_MICROS: list[float] = field(
        default_factory=lambda: [
            0.00001,  # 10us
            0.00005,  # 50us
            0.0001,  # 100us
            0.0005,  # 500us
            0.001,  # 1ms
            0.005,  # 5ms
            0.01,  # 10ms
            0.05,  # 50ms
            0.1,  # 100ms
        ]
    )

    # Buckets for sizes (bytes, counts)
    SIZE: list[float] = field(
        default_factory=lambda: [
            10,
            50,
            100,
            500,
            1000,
            5000,
            10000,
            50000,
            100000,
        ]
    )


class Histogram:
    """
    Distribution of values with percentile calculation.

    Uses pre-defined buckets for efficiency.

    Example:
        latency = Histogram("handler_latency", buckets=[0.001, 0.01, 0.1, 1.0])
        latency.observe(0.025)  # 25ms

        print(latency.percentile(0.95))  # p95 latency
        print(latency.percentile(0.99))  # p99 latency
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: list[float] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._buckets = sorted(buckets or HistogramBuckets().LATENCY_SECONDS)
        self._bucket_counts: list[int] = [0] * (len(self._buckets) + 1)  # +1 for +Inf
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()
        # Keep recent values for percentile calculation
        self._values: list[float] = []
        self._max_values = 10000  # Rolling window

    def observe(self, value: float) -> None:
        """Record a value."""
        with self._lock:
            self._sum += value
            self._count += 1

            # Update bucket counts
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._bucket_counts[i] += 1
                    break
            else:
                self._bucket_counts[-1] += 1  # +Inf bucket

            # Keep value for percentile
            self._values.append(value)
            if len(self._values) > self._max_values:
                self._values = self._values[-self._max_values :]

    def percentile(self, p: float) -> float | None:
        """
        Calculate percentile (0.0 to 1.0).

        Returns None if no values recorded.
        """
        with self._lock:
            if not self._values:
                return None

            sorted_values = sorted(self._values)
            idx = int(len(sorted_values) * p)
            idx = min(idx, len(sorted_values) - 1)
            return sorted_values[idx]

    @property
    def p50(self) -> float | None:
        """Median (50th percentile)."""
        return self.percentile(0.5)

    @property
    def p95(self) -> float | None:
        """95th percentile."""
        return self.percentile(0.95)

    @property
    def p99(self) -> float | None:
        """99th percentile."""
        return self.percentile(0.99)

    @property
    def mean(self) -> float | None:
        """Average value."""
        with self._lock:
            if self._count == 0:
                return None
            return self._sum / self._count

    @property
    def count(self) -> int:
        """Total observations."""
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        """Sum of all values."""
        with self._lock:
            return self._sum

    def reset(self) -> None:
        """Reset histogram (for testing only)."""
        with self._lock:
            self._bucket_counts = [0] * len(self._bucket_counts)
            self._sum = 0.0
            self._count = 0
            self._values.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        with self._lock:
            return {
                "name": self.name,
                "type": MetricType.HISTOGRAM.value,
                "count": self._count,
                "sum": self._sum,
                "mean": self._sum / self._count if self._count > 0 else None,
                "p50": self.p50,
                "p95": self.p95,
                "p99": self.p99,
                "buckets": dict(zip(self._buckets + [float("inf")], self._bucket_counts)),
            }


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """
    Central metrics registry and collector.

    Collects all metrics from the trading system:
    - Event bus metrics (published, dispatched, dropped)
    - Handler latencies
    - Queue sizes
    - Error counts

    Example:
        collector = MetricsCollector()

        # Register metrics
        collector.register_counter("events_published", "Total events published")
        collector.register_histogram("handler_latency", "Handler execution time")

        # Record values
        collector.inc_counter("events_published")
        collector.observe_histogram("handler_latency", 0.025)

        # Export all metrics
        metrics = collector.collect()
    """

    def __init__(self, prefix: str = "libra") -> None:
        self.prefix = prefix
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Pre-register standard metrics
        self._register_standard_metrics()

    def _register_standard_metrics(self) -> None:
        """Register standard trading system metrics."""
        # Event bus counters
        self.register_counter("events_published", "Total events published")
        self.register_counter("events_dispatched", "Total events dispatched to handlers")
        self.register_counter("events_dropped", "Events dropped due to queue overflow")
        self.register_counter("handler_errors", "Handler execution errors")

        # Event bus gauges
        self.register_gauge("queue_size", "Current event queue size")
        self.register_gauge("active_handlers", "Number of active handlers")
        self.register_gauge("pending_events", "Events pending dispatch")

        # Latency histograms
        self.register_histogram(
            "event_dispatch_latency",
            "Time from publish to dispatch",
            buckets=HistogramBuckets().LATENCY_MICROS,
        )
        self.register_histogram(
            "handler_execution_latency",
            "Handler execution time",
            buckets=HistogramBuckets().LATENCY_SECONDS,
        )

        # Trading metrics
        self.register_counter("orders_submitted", "Total orders submitted")
        self.register_counter("orders_filled", "Total orders filled")
        self.register_counter("orders_rejected", "Total orders rejected")
        self.register_histogram(
            "order_fill_latency",
            "Time from submission to fill",
            buckets=HistogramBuckets().LATENCY_SECONDS,
        )

    def register_counter(self, name: str, description: str = "") -> Counter:
        """Register a new counter metric."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._counters:
                self._counters[full_name] = Counter(full_name, description)
            return self._counters[full_name]

    def register_gauge(self, name: str, description: str = "") -> Gauge:
        """Register a new gauge metric."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._gauges:
                self._gauges[full_name] = Gauge(full_name, description)
            return self._gauges[full_name]

    def register_histogram(
        self,
        name: str,
        description: str = "",
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Register a new histogram metric."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._histograms:
                self._histograms[full_name] = Histogram(full_name, description, buckets)
            return self._histograms[full_name]

    def inc_counter(self, name: str, amount: float = 1.0) -> None:
        """Increment a counter."""
        full_name = f"{self.prefix}_{name}"
        if full_name in self._counters:
            self._counters[full_name].inc(amount)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        full_name = f"{self.prefix}_{name}"
        if full_name in self._gauges:
            self._gauges[full_name].set(value)

    def observe_histogram(self, name: str, value: float) -> None:
        """Record a histogram observation."""
        full_name = f"{self.prefix}_{name}"
        if full_name in self._histograms:
            self._histograms[full_name].observe(value)

    def get_counter(self, name: str) -> Counter | None:
        """Get a counter by name."""
        full_name = f"{self.prefix}_{name}"
        return self._counters.get(full_name)

    def get_gauge(self, name: str) -> Gauge | None:
        """Get a gauge by name."""
        full_name = f"{self.prefix}_{name}"
        return self._gauges.get(full_name)

    def get_histogram(self, name: str) -> Histogram | None:
        """Get a histogram by name."""
        full_name = f"{self.prefix}_{name}"
        return self._histograms.get(full_name)

    def collect(self) -> dict[str, Any]:
        """
        Collect all metrics.

        Returns dict with all metric values for export.
        """
        with self._lock:
            return {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self._start_time,
                "counters": {name: c.to_dict() for name, c in self._counters.items()},
                "gauges": {name: g.to_dict() for name, g in self._gauges.items()},
                "histograms": {name: h.to_dict() for name, h in self._histograms.items()},
            }

    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns string ready for /metrics endpoint.
        """
        lines = []

        # Counters
        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.value}")

        # Gauges
        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.value}")

        # Histograms
        for name, histogram in self._histograms.items():
            lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count {histogram.count}")
            lines.append(f"{name}_sum {histogram.sum}")

        return "\n".join(lines)

    def reset_all(self) -> None:
        """Reset all metrics (for testing only)."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for gauge in self._gauges.values():
                gauge.reset()
            for histogram in self._histograms.values():
                histogram.reset()


# =============================================================================
# Global Collector Instance
# =============================================================================

_default_collector: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    """Get the default metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def set_collector(collector: MetricsCollector) -> None:
    """Set the default metrics collector."""
    global _default_collector
    _default_collector = collector

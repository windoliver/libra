"""
Tests for metrics collection (Issue #25).
"""

import pytest
import time

from libra.observability.collector import (
    Counter,
    Gauge,
    Histogram,
    HistogramBuckets,
    MetricsCollector,
    MetricType,
    get_collector,
    set_collector,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_init(self):
        """Counter initializes with zero value."""
        counter = Counter("test_counter", "Test description")
        assert counter.name == "test_counter"
        assert counter.description == "Test description"
        assert counter.value == 0.0

    def test_counter_inc(self):
        """Counter increments correctly."""
        counter = Counter("test")
        counter.inc()
        assert counter.value == 1.0
        counter.inc(5)
        assert counter.value == 6.0

    def test_counter_inc_labeled(self):
        """Counter with labels increments correctly."""
        counter = Counter("test")
        counter.inc_labeled({"method": "GET"})
        counter.inc_labeled({"method": "POST"}, 3)
        counter.inc_labeled({"method": "GET"}, 2)

        assert counter.get_labeled({"method": "GET"}) == 3.0
        assert counter.get_labeled({"method": "POST"}) == 3.0
        assert counter.get_labeled({"method": "DELETE"}) == 0.0

    def test_counter_reset(self):
        """Counter resets to zero."""
        counter = Counter("test")
        counter.inc(10)
        counter.inc_labeled({"type": "error"}, 5)
        counter.reset()

        assert counter.value == 0.0
        assert counter.get_labeled({"type": "error"}) == 0.0

    def test_counter_to_dict(self):
        """Counter exports to dictionary."""
        counter = Counter("test_counter", "Test")
        counter.inc(42)
        data = counter.to_dict()

        assert data["name"] == "test_counter"
        assert data["type"] == "counter"
        assert data["value"] == 42.0


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_init(self):
        """Gauge initializes with zero value."""
        gauge = Gauge("test_gauge", "Test description")
        assert gauge.name == "test_gauge"
        assert gauge.value == 0.0

    def test_gauge_set(self):
        """Gauge sets value correctly."""
        gauge = Gauge("test")
        gauge.set(100)
        assert gauge.value == 100.0
        gauge.set(50)
        assert gauge.value == 50.0

    def test_gauge_inc_dec(self):
        """Gauge increments and decrements."""
        gauge = Gauge("test")
        gauge.set(50)
        gauge.inc(10)
        assert gauge.value == 60.0
        gauge.dec(5)
        assert gauge.value == 55.0

    def test_gauge_labeled(self):
        """Gauge with labels works correctly."""
        gauge = Gauge("queue_size")
        gauge.set_labeled({"queue": "priority"}, 100)
        gauge.set_labeled({"queue": "normal"}, 50)

        assert gauge.get_labeled({"queue": "priority"}) == 100.0
        assert gauge.get_labeled({"queue": "normal"}) == 50.0

    def test_gauge_to_dict(self):
        """Gauge exports to dictionary."""
        gauge = Gauge("test_gauge")
        gauge.set(42.5)
        data = gauge.to_dict()

        assert data["name"] == "test_gauge"
        assert data["type"] == "gauge"
        assert data["value"] == 42.5


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_init(self):
        """Histogram initializes correctly."""
        hist = Histogram("latency", "Request latency")
        assert hist.name == "latency"
        assert hist.count == 0
        assert hist.sum == 0.0

    def test_histogram_observe(self):
        """Histogram records observations."""
        hist = Histogram("latency", buckets=[0.1, 0.5, 1.0])
        hist.observe(0.05)
        hist.observe(0.3)
        hist.observe(0.8)
        hist.observe(2.0)

        assert hist.count == 4
        assert hist.sum == pytest.approx(3.15)

    def test_histogram_percentiles(self):
        """Histogram calculates percentiles."""
        hist = Histogram("latency")

        # Add 100 values from 0 to 99
        for i in range(100):
            hist.observe(i)

        # Check percentiles
        assert hist.p50 == pytest.approx(50, abs=1)
        assert hist.p95 == pytest.approx(95, abs=1)
        assert hist.p99 == pytest.approx(99, abs=1)

    def test_histogram_mean(self):
        """Histogram calculates mean."""
        hist = Histogram("latency")
        hist.observe(10)
        hist.observe(20)
        hist.observe(30)

        assert hist.mean == pytest.approx(20.0)

    def test_histogram_empty_percentile(self):
        """Empty histogram returns None for percentiles."""
        hist = Histogram("latency")
        assert hist.percentile(0.5) is None
        assert hist.p50 is None

    def test_histogram_buckets(self):
        """Histogram uses custom buckets."""
        buckets = [0.001, 0.01, 0.1]
        hist = Histogram("latency", buckets=buckets)
        hist.observe(0.0005)  # <= 0.001
        hist.observe(0.005)  # <= 0.01
        hist.observe(0.05)  # <= 0.1
        hist.observe(0.5)  # > 0.1 (inf bucket)

        data = hist.to_dict()
        assert data["count"] == 4

    def test_histogram_to_dict(self):
        """Histogram exports to dictionary."""
        hist = Histogram("latency")
        hist.observe(0.5)
        data = hist.to_dict()

        assert data["name"] == "latency"
        assert data["type"] == "histogram"
        assert data["count"] == 1
        assert "p50" in data
        assert "p95" in data
        assert "p99" in data


class TestHistogramBuckets:
    """Tests for HistogramBuckets."""

    def test_default_buckets(self):
        """Default buckets are available."""
        buckets = HistogramBuckets()
        assert len(buckets.LATENCY_SECONDS) > 0
        assert len(buckets.LATENCY_MICROS) > 0
        assert len(buckets.SIZE) > 0

    def test_buckets_sorted(self):
        """Buckets are sorted."""
        buckets = HistogramBuckets()
        assert buckets.LATENCY_SECONDS == sorted(buckets.LATENCY_SECONDS)
        assert buckets.LATENCY_MICROS == sorted(buckets.LATENCY_MICROS)


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_collector_init(self):
        """Collector initializes with standard metrics."""
        collector = MetricsCollector(prefix="test")
        assert collector.prefix == "test"

        # Check standard metrics are registered
        assert collector.get_counter("events_published") is not None
        assert collector.get_gauge("queue_size") is not None
        assert collector.get_histogram("handler_execution_latency") is not None

    def test_collector_register_counter(self):
        """Collector registers counters."""
        collector = MetricsCollector(prefix="test")
        counter = collector.register_counter("custom_counter", "Custom")

        assert counter.name == "test_custom_counter"
        assert collector.get_counter("custom_counter") is counter

    def test_collector_register_gauge(self):
        """Collector registers gauges."""
        collector = MetricsCollector(prefix="test")
        gauge = collector.register_gauge("custom_gauge", "Custom")

        assert gauge.name == "test_custom_gauge"
        assert collector.get_gauge("custom_gauge") is gauge

    def test_collector_register_histogram(self):
        """Collector registers histograms."""
        collector = MetricsCollector(prefix="test")
        hist = collector.register_histogram("custom_hist", "Custom")

        assert hist.name == "test_custom_hist"
        assert collector.get_histogram("custom_hist") is hist

    def test_collector_shortcuts(self):
        """Collector shortcut methods work."""
        collector = MetricsCollector(prefix="test")

        collector.inc_counter("events_published", 5)
        assert collector.get_counter("events_published").value == 5

        collector.set_gauge("queue_size", 100)
        assert collector.get_gauge("queue_size").value == 100

        collector.observe_histogram("handler_execution_latency", 0.5)
        assert collector.get_histogram("handler_execution_latency").count == 1

    def test_collector_collect(self):
        """Collector exports all metrics."""
        collector = MetricsCollector(prefix="test")
        collector.inc_counter("events_published", 10)
        collector.set_gauge("queue_size", 50)

        data = collector.collect()

        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data

    def test_collector_prometheus_format(self):
        """Collector exports Prometheus format."""
        collector = MetricsCollector(prefix="test")
        collector.inc_counter("events_published", 10)

        output = collector.to_prometheus()

        assert "# HELP" in output
        assert "# TYPE" in output
        assert "test_events_published" in output

    def test_collector_reset_all(self):
        """Collector resets all metrics."""
        collector = MetricsCollector(prefix="test")
        collector.inc_counter("events_published", 10)
        collector.set_gauge("queue_size", 50)
        collector.observe_histogram("handler_execution_latency", 0.5)

        collector.reset_all()

        assert collector.get_counter("events_published").value == 0
        assert collector.get_gauge("queue_size").value == 0
        assert collector.get_histogram("handler_execution_latency").count == 0


class TestGlobalCollector:
    """Tests for global collector functions."""

    def test_get_collector(self):
        """get_collector returns default instance."""
        collector = get_collector()
        assert collector is not None
        assert isinstance(collector, MetricsCollector)

    def test_set_collector(self):
        """set_collector changes default instance."""
        original = get_collector()
        new_collector = MetricsCollector(prefix="custom")

        set_collector(new_collector)
        assert get_collector() is new_collector

        # Restore original
        set_collector(original)

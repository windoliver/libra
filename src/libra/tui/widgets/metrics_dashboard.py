"""
Metrics Dashboard Widget for Observability (Issue #25).

Displays system metrics in real-time:
- Counters (events published, handled, errors)
- Gauges (queue sizes, active handlers)
- Histograms (latency percentiles)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Rule, Static


@dataclass
class MetricData:
    """Data for a single metric."""

    name: str
    value: float
    metric_type: str  # "counter", "gauge", "histogram"
    labels: dict[str, str] = field(default_factory=dict)
    description: str = ""
    # Histogram-specific
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    count: int = 0


@dataclass
class MetricsDashboardData:
    """Data for the metrics dashboard."""

    counters: dict[str, MetricData] = field(default_factory=dict)
    gauges: dict[str, MetricData] = field(default_factory=dict)
    histograms: dict[str, MetricData] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    timestamp: float = 0.0


class MetricCard(Static):
    """Single metric display card."""

    DEFAULT_CSS = """
    MetricCard {
        width: 1fr;
        height: 5;
        border: round $primary-darken-2;
        padding: 0 1;
        margin: 0 1 1 0;
    }

    MetricCard > .metric-name {
        color: $text-muted;
        text-style: bold;
    }

    MetricCard > .metric-value {
        color: $success;
        text-style: bold;
    }

    MetricCard > .metric-type {
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(
        self,
        name: str,
        value: str,
        metric_type: str,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._name = name
        self._value = value
        self._type = metric_type

    def compose(self) -> ComposeResult:
        yield Static(self._name, classes="metric-name")
        yield Static(self._value, classes="metric-value")
        yield Static(self._type, classes="metric-type")

    def update_value(self, value: str) -> None:
        """Update the metric value."""
        self._value = value
        try:
            self.query_one(".metric-value", Static).update(value)
        except Exception:
            pass


class CounterPanel(Vertical):
    """Panel showing counter metrics. PERFORMANCE: Uses differential updates."""

    DEFAULT_CSS = """
    CounterPanel {
        height: auto;
        min-height: 8;
        border: round $primary-darken-1;
        padding: 0 1;
    }

    CounterPanel > Static.panel-title {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
    }

    CounterPanel > DataTable {
        height: auto;
        max-height: 12;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cached_table: DataTable | None = None
        self._row_keys: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Static("COUNTERS", classes="panel-title")
        table = DataTable(id="counter-table")
        table.add_columns("Metric", "Value", "Rate/s")
        yield table

    def on_mount(self) -> None:
        """Cache table reference."""
        self._cached_table = self.query_one("#counter-table", DataTable)

    def update_counters(self, counters: dict[str, MetricData]) -> None:
        """Update counter display using differential updates."""
        table = self._cached_table
        if not table:
            return

        for name, metric in counters.items():
            # Format name (remove prefix)
            short_name = name.split("_", 1)[-1] if "_" in name else name
            value_str = f"{metric.value:,.0f}"
            rate_str = f"{metric.value / max(1, metric.count):,.1f}" if metric.count > 0 else "-"

            # PERFORMANCE: Differential update with proper error handling
            if name in self._row_keys:
                row_key = self._row_keys[name]
                try:
                    table.update_cell(row_key, "Value", value_str)
                    table.update_cell(row_key, "Rate/s", rate_str)
                except Exception:
                    del self._row_keys[name]
                    self._add_counter_row(name, short_name, value_str, rate_str)
            else:
                self._add_counter_row(name, short_name, value_str, rate_str)

    def _add_counter_row(self, name: str, short_name: str, value_str: str, rate_str: str) -> None:
        """Add a counter row, handling duplicates gracefully."""
        table = self._cached_table
        if not table:
            return
        try:
            row_key = table.add_row(short_name, value_str, rate_str, key=name)
            self._row_keys[name] = row_key
        except Exception:
            for rk in table.rows:
                if rk.value == name:
                    self._row_keys[name] = rk
                    try:
                        table.update_cell(rk, "Value", value_str)
                        table.update_cell(rk, "Rate/s", rate_str)
                    except Exception:
                        pass
                    break


class GaugePanel(Vertical):
    """Panel showing gauge metrics. PERFORMANCE: Uses differential updates."""

    DEFAULT_CSS = """
    GaugePanel {
        height: auto;
        min-height: 8;
        border: round $primary-darken-1;
        padding: 0 1;
    }

    GaugePanel > Static.panel-title {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
    }

    GaugePanel > DataTable {
        height: auto;
        max-height: 12;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cached_table: DataTable | None = None
        self._row_keys: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Static("GAUGES", classes="panel-title")
        table = DataTable(id="gauge-table")
        table.add_columns("Metric", "Value")
        yield table

    def on_mount(self) -> None:
        """Cache table reference."""
        self._cached_table = self.query_one("#gauge-table", DataTable)

    def update_gauges(self, gauges: dict[str, MetricData]) -> None:
        """Update gauge display using differential updates."""
        table = self._cached_table
        if not table:
            return

        for name, metric in gauges.items():
            short_name = name.split("_", 1)[-1] if "_" in name else name

            # Color code based on value
            value = metric.value
            if "queue" in name.lower() or "pending" in name.lower():
                # Queue size coloring
                if value > 1000:
                    color = "red"
                elif value > 100:
                    color = "yellow"
                else:
                    color = "green"
                value_str = f"[{color}]{value:,.0f}[/{color}]"
            else:
                value_str = f"{value:,.2f}"

            # PERFORMANCE: Differential update with proper error handling
            if name in self._row_keys:
                row_key = self._row_keys[name]
                try:
                    table.update_cell(row_key, "Value", value_str)
                except Exception:
                    del self._row_keys[name]
                    self._add_gauge_row(name, short_name, value_str)
            else:
                self._add_gauge_row(name, short_name, value_str)

    def _add_gauge_row(self, name: str, short_name: str, value_str: str) -> None:
        """Add a gauge row, handling duplicates gracefully."""
        table = self._cached_table
        if not table:
            return
        try:
            row_key = table.add_row(short_name, value_str, key=name)
            self._row_keys[name] = row_key
        except Exception:
            for rk in table.rows:
                if rk.value == name:
                    self._row_keys[name] = rk
                    try:
                        table.update_cell(rk, "Value", value_str)
                    except Exception:
                        pass
                    break


class HistogramPanel(Vertical):
    """Panel showing histogram metrics with percentiles. PERFORMANCE: Uses differential updates."""

    DEFAULT_CSS = """
    HistogramPanel {
        height: auto;
        min-height: 10;
        border: round $primary-darken-1;
        padding: 0 1;
    }

    HistogramPanel > Static.panel-title {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
    }

    HistogramPanel > DataTable {
        height: auto;
        max-height: 15;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cached_table: DataTable | None = None
        self._row_keys: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Static("LATENCIES (ms)", classes="panel-title")
        table = DataTable(id="histogram-table")
        table.add_columns("Metric", "Count", "P50", "P95", "P99")
        yield table

    def on_mount(self) -> None:
        """Cache table reference."""
        self._cached_table = self.query_one("#histogram-table", DataTable)

    def update_histograms(self, histograms: dict[str, MetricData]) -> None:
        """Update histogram display using differential updates."""
        table = self._cached_table
        if not table:
            return

        for name, metric in histograms.items():
            short_name = name.split("_", 1)[-1] if "_" in name else name
            count_str = f"{metric.count:,}"

            # Format percentiles (convert to ms)
            p50_str = f"{metric.p50 * 1000:.2f}" if metric.p50 else "-"
            p95_str = f"{metric.p95 * 1000:.2f}" if metric.p95 else "-"
            p99_str = f"{metric.p99 * 1000:.2f}" if metric.p99 else "-"

            # Color P99 if high
            if metric.p99 and metric.p99 * 1000 > 100:  # >100ms
                p99_str = f"[red]{p99_str}[/red]"
            elif metric.p99 and metric.p99 * 1000 > 50:  # >50ms
                p99_str = f"[yellow]{p99_str}[/yellow]"

            # PERFORMANCE: Differential update with proper error handling
            if name in self._row_keys:
                row_key = self._row_keys[name]
                try:
                    table.update_cell(row_key, "Count", count_str)
                    table.update_cell(row_key, "P50", p50_str)
                    table.update_cell(row_key, "P95", p95_str)
                    table.update_cell(row_key, "P99", p99_str)
                except Exception:
                    del self._row_keys[name]
                    self._add_histogram_row(name, short_name, count_str, p50_str, p95_str, p99_str)
            else:
                self._add_histogram_row(name, short_name, count_str, p50_str, p95_str, p99_str)

    def _add_histogram_row(self, name: str, short_name: str, count_str: str, p50_str: str, p95_str: str, p99_str: str) -> None:
        """Add a histogram row, handling duplicates gracefully."""
        table = self._cached_table
        if not table:
            return
        try:
            row_key = table.add_row(short_name, count_str, p50_str, p95_str, p99_str, key=name)
            self._row_keys[name] = row_key
        except Exception:
            for rk in table.rows:
                if rk.value == name:
                    self._row_keys[name] = rk
                    try:
                        table.update_cell(rk, "Count", count_str)
                        table.update_cell(rk, "P50", p50_str)
                        table.update_cell(rk, "P95", p95_str)
                        table.update_cell(rk, "P99", p99_str)
                    except Exception:
                        pass
                    break


class MetricsDashboard(Vertical):
    """
    Complete metrics dashboard for system observability.

    Shows:
    - Counter metrics (events, errors, trades)
    - Gauge metrics (queue sizes, handler counts)
    - Histogram metrics (latency percentiles)
    - System uptime

    PERFORMANCE: Uses cached panel references.

    Example:
        dashboard = MetricsDashboard(id="metrics-dashboard")

        # Update with data from MetricsCollector
        data = collector.collect()
        dashboard.update_from_collector(data)
    """

    DEFAULT_CSS = """
    MetricsDashboard {
        height: auto;
        padding: 1;
    }

    MetricsDashboard > .header {
        height: 2;
        background: $primary-darken-2;
        padding: 0 1;
        margin-bottom: 1;
    }

    MetricsDashboard > .header > Static {
        width: 1fr;
    }

    MetricsDashboard > Horizontal {
        height: auto;
    }

    MetricsDashboard > Horizontal > * {
        width: 1fr;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._uptime = 0.0
        # PERFORMANCE: Cache panel references
        self._cached_uptime_display: Static | None = None
        self._cached_counter_panel: CounterPanel | None = None
        self._cached_gauge_panel: GaugePanel | None = None
        self._cached_histogram_panel: HistogramPanel | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header"):
            yield Static("SYSTEM METRICS")
            yield Static("Uptime: 0s", id="uptime-display")

        with Horizontal():
            yield CounterPanel(id="counter-panel")
            yield GaugePanel(id="gauge-panel")

        yield Rule()
        yield HistogramPanel(id="histogram-panel")

    def on_mount(self) -> None:
        """Cache panel references for performance."""
        try:
            self._cached_uptime_display = self.query_one("#uptime-display", Static)
        except Exception:
            pass
        try:
            self._cached_counter_panel = self.query_one("#counter-panel", CounterPanel)
        except Exception:
            pass
        try:
            self._cached_gauge_panel = self.query_one("#gauge-panel", GaugePanel)
        except Exception:
            pass
        try:
            self._cached_histogram_panel = self.query_one("#histogram-panel", HistogramPanel)
        except Exception:
            pass

    def update_from_collector(self, data: dict[str, Any]) -> None:
        """
        Update dashboard from MetricsCollector.collect() output.
        PERFORMANCE: Uses cached panel references.

        Args:
            data: Output from collector.collect()
        """
        # Update uptime
        uptime = data.get("uptime_seconds", 0)
        self._uptime = uptime
        if self._cached_uptime_display:
            uptime_str = self._format_uptime(uptime)
            self._cached_uptime_display.update(f"Uptime: {uptime_str}")

        # Convert collector data to MetricData
        counters = {}
        for name, counter_data in data.get("counters", {}).items():
            counters[name] = MetricData(
                name=name,
                value=counter_data.get("value", 0),
                metric_type="counter",
            )

        gauges = {}
        for name, gauge_data in data.get("gauges", {}).items():
            gauges[name] = MetricData(
                name=name,
                value=gauge_data.get("value", 0),
                metric_type="gauge",
            )

        histograms = {}
        for name, hist_data in data.get("histograms", {}).items():
            histograms[name] = MetricData(
                name=name,
                value=hist_data.get("mean", 0) or 0,
                metric_type="histogram",
                p50=hist_data.get("p50"),
                p95=hist_data.get("p95"),
                p99=hist_data.get("p99"),
                count=hist_data.get("count", 0),
            )

        # Update panels using cached references
        if self._cached_counter_panel:
            self._cached_counter_panel.update_counters(counters)

        if self._cached_gauge_panel:
            self._cached_gauge_panel.update_gauges(gauges)

        if self._cached_histogram_panel:
            self._cached_histogram_panel.update_histograms(histograms)

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime for display."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"


def create_demo_metrics_data() -> dict[str, Any]:
    """Create demo data for testing."""
    import random
    import time

    return {
        "timestamp": time.time(),
        "uptime_seconds": random.uniform(3600, 86400),
        "counters": {
            "libra_events_published": {"value": random.randint(10000, 100000)},
            "libra_events_dispatched": {"value": random.randint(9000, 99000)},
            "libra_events_dropped": {"value": random.randint(0, 100)},
            "libra_handler_errors": {"value": random.randint(0, 50)},
            "libra_orders_submitted": {"value": random.randint(100, 1000)},
            "libra_orders_filled": {"value": random.randint(90, 950)},
        },
        "gauges": {
            "libra_queue_size": {"value": random.randint(0, 500)},
            "libra_active_handlers": {"value": random.randint(5, 20)},
            "libra_pending_events": {"value": random.randint(0, 100)},
        },
        "histograms": {
            "libra_event_dispatch_latency": {
                "count": random.randint(1000, 10000),
                "p50": random.uniform(0.0001, 0.001),
                "p95": random.uniform(0.001, 0.01),
                "p99": random.uniform(0.01, 0.1),
            },
            "libra_handler_execution_latency": {
                "count": random.randint(500, 5000),
                "p50": random.uniform(0.001, 0.01),
                "p95": random.uniform(0.01, 0.05),
                "p99": random.uniform(0.05, 0.2),
            },
            "libra_order_fill_latency": {
                "count": random.randint(100, 500),
                "p50": random.uniform(0.01, 0.1),
                "p95": random.uniform(0.1, 0.5),
                "p99": random.uniform(0.5, 2.0),
            },
        },
    }

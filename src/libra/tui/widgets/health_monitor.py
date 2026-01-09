"""
Health Monitor Widget for Observability (Issue #25).

Displays system and component health status:
- Overall system health
- Individual component health checks
- System resources (CPU, memory, disk)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, ProgressBar, Rule, Static


@dataclass
class ComponentHealthData:
    """Health data for a single component."""

    name: str
    status: str  # healthy, degraded, unhealthy, unknown
    message: str = ""
    latency_ms: float = 0.0
    last_check: float = 0.0
    age_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthData:
    """Overall system health data."""

    status: str  # healthy, degraded, unhealthy, unknown
    timestamp: float = 0.0
    components: dict[str, ComponentHealthData] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)


class HealthStatusBadge(Static):
    """Visual health status indicator."""

    DEFAULT_CSS = """
    HealthStatusBadge {
        width: auto;
        height: 3;
        padding: 0 2;
        text-align: center;
        border: round $primary-darken-1;
    }

    HealthStatusBadge.healthy {
        background: $success-darken-2;
        color: $success;
    }

    HealthStatusBadge.degraded {
        background: $warning-darken-2;
        color: $warning;
    }

    HealthStatusBadge.unhealthy {
        background: $error-darken-2;
        color: $error;
    }

    HealthStatusBadge.unknown {
        background: $surface;
        color: $text-muted;
    }
    """

    def __init__(self, status: str = "unknown", id: str | None = None) -> None:
        super().__init__(id=id)
        self._status = status

    def on_mount(self) -> None:
        self.update_status(self._status)

    def update_status(self, status: str) -> None:
        """Update the health status display."""
        self._status = status

        # Remove old classes
        self.remove_class("healthy", "degraded", "unhealthy", "unknown")
        self.add_class(status)

        # Update display
        status_icons = {
            "healthy": "✓ HEALTHY",
            "degraded": "⚠ DEGRADED",
            "unhealthy": "✗ UNHEALTHY",
            "unknown": "? UNKNOWN",
        }
        self.update(status_icons.get(status, "? UNKNOWN"))


class ResourceGauge(Vertical):
    """Resource usage gauge (CPU, memory, disk)."""

    DEFAULT_CSS = """
    ResourceGauge {
        width: 1fr;
        height: 4;
        padding: 0 1;
        margin: 0 1;
    }

    ResourceGauge > Static.label {
        height: 1;
        color: $text-muted;
    }

    ResourceGauge > Horizontal {
        height: 2;
    }

    ResourceGauge > Horizontal > ProgressBar {
        width: 1fr;
    }

    ResourceGauge > Horizontal > Static.value {
        width: 6;
        text-align: right;
    }
    """

    def __init__(
        self,
        label: str,
        value: float = 0.0,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="label")
        with Horizontal():
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="gauge-bar")
            yield Static(f"{self._value:.0f}%", classes="value", id="gauge-value")

    def on_mount(self) -> None:
        self.update_value(self._value)

    def update_value(self, value: float) -> None:
        """Update the gauge value (0-100)."""
        self._value = min(100, max(0, value))
        try:
            bar = self.query_one("#gauge-bar", ProgressBar)
            bar.update(progress=self._value)

            value_label = self.query_one("#gauge-value", Static)
            # Color code based on value
            if self._value >= 90:
                color = "red"
            elif self._value >= 70:
                color = "yellow"
            else:
                color = "green"
            value_label.update(f"[{color}]{self._value:.0f}%[/{color}]")
        except Exception:
            pass


class SystemResourcesPanel(Vertical):
    """Panel showing system resource usage."""

    DEFAULT_CSS = """
    SystemResourcesPanel {
        height: auto;
        border: round $primary-darken-1;
        padding: 1;
    }

    SystemResourcesPanel > Static.panel-title {
        height: 1;
        color: $text;
        text-style: bold;
    }

    SystemResourcesPanel > Horizontal {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("SYSTEM RESOURCES", classes="panel-title")
        yield Rule()
        with Horizontal():
            yield ResourceGauge("CPU", id="cpu-gauge")
            yield ResourceGauge("Memory", id="memory-gauge")
            yield ResourceGauge("Disk", id="disk-gauge")

    def update_resources(
        self,
        cpu: float = 0.0,
        memory: float = 0.0,
        disk: float = 0.0,
    ) -> None:
        """Update resource gauges."""
        try:
            self.query_one("#cpu-gauge", ResourceGauge).update_value(cpu)
            self.query_one("#memory-gauge", ResourceGauge).update_value(memory)
            self.query_one("#disk-gauge", ResourceGauge).update_value(disk)
        except Exception:
            pass


class ComponentHealthTable(Vertical):
    """Table showing health status of all components."""

    DEFAULT_CSS = """
    ComponentHealthTable {
        height: auto;
        min-height: 10;
        border: round $primary-darken-1;
    }

    ComponentHealthTable > Static.panel-title {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }

    ComponentHealthTable > DataTable {
        height: auto;
        max-height: 20;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("COMPONENT HEALTH", classes="panel-title")
        table = DataTable(id="component-table")
        table.add_columns("Component", "Status", "Message", "Latency", "Age")
        yield table

    def update_components(self, components: dict[str, ComponentHealthData]) -> None:
        """Update the component table."""
        try:
            table = self.query_one("#component-table", DataTable)
            table.clear()

            for name, health in components.items():
                # Format status with color
                status_colors = {
                    "healthy": "green",
                    "degraded": "yellow",
                    "unhealthy": "red",
                    "unknown": "dim",
                }
                color = status_colors.get(health.status, "dim")
                status_str = f"[{color}]{health.status.upper()}[/{color}]"

                # Format latency
                latency_str = f"{health.latency_ms:.1f}ms" if health.latency_ms > 0 else "-"

                # Format age
                if health.age_seconds < 60:
                    age_str = f"{health.age_seconds:.0f}s"
                elif health.age_seconds < 3600:
                    age_str = f"{health.age_seconds / 60:.0f}m"
                else:
                    age_str = f"{health.age_seconds / 3600:.1f}h"

                # Truncate message
                message = health.message[:30] + "..." if len(health.message) > 33 else health.message

                table.add_row(name, status_str, message, latency_str, age_str)
        except Exception:
            pass


class HealthMonitorWidget(Vertical):
    """
    Complete health monitoring widget.

    Shows:
    - Overall system health status
    - System resource usage (CPU, memory, disk)
    - Individual component health checks
    - Health summary counts

    Example:
        monitor = HealthMonitorWidget(id="health-monitor")

        # Update with data from HealthMonitor
        data = await health_monitor.check_all()
        monitor.update_health(data)
    """

    DEFAULT_CSS = """
    HealthMonitorWidget {
        height: auto;
        padding: 1;
    }

    HealthMonitorWidget > Horizontal.header {
        height: 5;
        margin-bottom: 1;
    }

    HealthMonitorWidget > Horizontal.header > Vertical {
        width: 1fr;
    }

    HealthMonitorWidget > Horizontal.header > HealthStatusBadge {
        width: 20;
    }

    HealthMonitorWidget .summary-item {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._health_data: SystemHealthData | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header"):
            with Vertical():
                yield Static("SYSTEM HEALTH", classes="panel-title")
                yield Static("Last check: -", id="last-check")
            yield HealthStatusBadge(status="unknown", id="health-badge")

        yield SystemResourcesPanel(id="resources-panel")
        yield Rule()

        # Health summary
        with Horizontal():
            yield Static("✓ Healthy: 0", classes="summary-item", id="summary-healthy")
            yield Static("⚠ Degraded: 0", classes="summary-item", id="summary-degraded")
            yield Static("✗ Unhealthy: 0", classes="summary-item", id="summary-unhealthy")

        yield Rule()
        yield ComponentHealthTable(id="component-table-panel")

    def update_health(self, data: dict[str, Any]) -> None:
        """
        Update from HealthMonitor.check_all() output.

        Args:
            data: Output from monitor.check_all()
        """
        # Update overall status
        status = data.get("status", "unknown")
        try:
            self.query_one("#health-badge", HealthStatusBadge).update_status(status)
        except Exception:
            pass

        # Update last check time
        timestamp = data.get("timestamp", 0)
        if timestamp > 0:
            time_str = datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).strftime("%H:%M:%S")
            try:
                self.query_one("#last-check", Static).update(f"Last check: {time_str}")
            except Exception:
                pass

        # Update summary
        summary = data.get("summary", {})
        try:
            self.query_one("#summary-healthy", Static).update(
                f"[green]✓ Healthy: {summary.get('healthy', 0)}[/green]"
            )
            self.query_one("#summary-degraded", Static).update(
                f"[yellow]⚠ Degraded: {summary.get('degraded', 0)}[/yellow]"
            )
            self.query_one("#summary-unhealthy", Static).update(
                f"[red]✗ Unhealthy: {summary.get('unhealthy', 0)}[/red]"
            )
        except Exception:
            pass

        # Update components
        components = {}
        for name, comp_data in data.get("components", {}).items():
            components[name] = ComponentHealthData(
                name=name,
                status=comp_data.get("status", "unknown"),
                message=comp_data.get("message", ""),
                latency_ms=comp_data.get("latency_ms", 0),
                last_check=comp_data.get("last_check", 0),
                age_seconds=comp_data.get("age_seconds", 0),
                details=comp_data.get("details", {}),
            )

        try:
            self.query_one("#component-table-panel", ComponentHealthTable).update_components(components)
        except Exception:
            pass

        # Update resources from system components
        cpu_data = data.get("components", {}).get("system.cpu", {})
        mem_data = data.get("components", {}).get("system.memory", {})
        disk_data = data.get("components", {}).get("system.disk", {})

        try:
            self.query_one("#resources-panel", SystemResourcesPanel).update_resources(
                cpu=cpu_data.get("details", {}).get("percent", 0),
                memory=mem_data.get("details", {}).get("percent", 0),
                disk=disk_data.get("details", {}).get("percent", 0),
            )
        except Exception:
            pass


def create_demo_health_data() -> dict[str, Any]:
    """Create demo health data for testing."""
    import random
    import time

    # Simulate different health scenarios
    scenarios = ["healthy", "healthy", "healthy", "degraded", "unhealthy"]
    overall = random.choice(scenarios)

    components = {
        "system.cpu": {
            "status": "healthy" if random.uniform(0, 100) < 70 else "degraded",
            "message": f"CPU usage: {random.uniform(10, 80):.1f}%",
            "latency_ms": random.uniform(0.1, 5),
            "last_check": time.time(),
            "age_seconds": random.uniform(0, 30),
            "details": {"percent": random.uniform(10, 80)},
        },
        "system.memory": {
            "status": "healthy" if random.uniform(0, 100) < 80 else "degraded",
            "message": f"Memory usage: {random.uniform(30, 70):.1f}%",
            "latency_ms": random.uniform(0.1, 3),
            "last_check": time.time(),
            "age_seconds": random.uniform(0, 30),
            "details": {
                "percent": random.uniform(30, 70),
                "total_gb": 16.0,
                "available_gb": random.uniform(4, 12),
            },
        },
        "system.disk": {
            "status": "healthy",
            "message": f"Disk usage: {random.uniform(40, 60):.1f}%",
            "latency_ms": random.uniform(0.1, 10),
            "last_check": time.time(),
            "age_seconds": random.uniform(0, 60),
            "details": {
                "percent": random.uniform(40, 60),
                "total_gb": 500.0,
                "free_gb": random.uniform(200, 300),
            },
        },
        "gateway.binance": {
            "status": random.choice(["healthy", "healthy", "degraded"]),
            "message": "Connected, latency OK" if random.random() > 0.2 else "High latency",
            "latency_ms": random.uniform(10, 200),
            "last_check": time.time(),
            "age_seconds": random.uniform(0, 5),
            "details": {},
        },
        "message_bus": {
            "status": "healthy",
            "message": "Running, queue healthy",
            "latency_ms": random.uniform(0.1, 2),
            "last_check": time.time(),
            "age_seconds": random.uniform(0, 10),
            "details": {
                "queue_size": random.randint(0, 100),
                "handlers": random.randint(5, 15),
            },
        },
        "risk_engine": {
            "status": random.choice(["healthy", "healthy", "healthy", "degraded"]),
            "message": "Active, limits OK" if random.random() > 0.1 else "Near daily limit",
            "latency_ms": random.uniform(0.5, 5),
            "last_check": time.time(),
            "age_seconds": random.uniform(0, 15),
            "details": {},
        },
    }

    # Count by status
    summary = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
    for comp in components.values():
        status = comp.get("status", "unknown")
        summary[status] = summary.get(status, 0) + 1

    return {
        "status": overall,
        "timestamp": time.time(),
        "components": components,
        "summary": summary,
    }

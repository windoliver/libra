"""
Audit Log Dashboard for TUI.

Displays audit events, order trails, and risk events in real-time.
Part of Issue #16: Audit Logging System.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Label, RichLog, Static


# Severity to color mapping
SEVERITY_COLORS: dict[str, str] = {
    "debug": "dim",
    "info": "white",
    "warning": "yellow",
    "error": "red",
    "critical": "red bold",
}

# Event type to color mapping
EVENT_TYPE_COLORS: dict[str, str] = {
    "order.created": "cyan",
    "order.submitted": "cyan",
    "order.accepted": "green",
    "order.filled": "green bold",
    "order.rejected": "red",
    "order.cancelled": "yellow",
    "risk.check_passed": "green",
    "risk.check_failed": "red",
    "risk.limit_breach": "red bold",
    "risk.circuit_breaker": "red bold",
    "agent.decision": "magenta",
    "agent.signal": "magenta",
    "system.error": "red",
    "system.start": "green",
}


class AuditStatsPanel(Static):
    """Panel showing audit statistics."""

    DEFAULT_CSS = """
    AuditStatsPanel {
        height: 3;
        background: $surface-darken-1;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stats = {
            "events": 0,
            "orders": 0,
            "risk": 0,
            "warnings": 0,
            "errors": 0,
        }

    def compose(self) -> ComposeResult:
        yield Static(id="stats-content")

    def on_mount(self) -> None:
        self._update_display()

    def update_stats(
        self,
        events: int = 0,
        orders: int = 0,
        risk: int = 0,
        warnings: int = 0,
        errors: int = 0,
    ) -> None:
        self._stats["events"] = events
        self._stats["orders"] = orders
        self._stats["risk"] = risk
        self._stats["warnings"] = warnings
        self._stats["errors"] = errors
        self._update_display()

    def increment(self, key: str, amount: int = 1) -> None:
        if key in self._stats:
            self._stats[key] += amount
            self._update_display()

    def _update_display(self) -> None:
        try:
            content = self.query_one("#stats-content", Static)
            text = Text()
            text.append("Events: ", style="bold")
            text.append(f"{self._stats['events']}  ", style="cyan")
            text.append("Orders: ", style="bold")
            text.append(f"{self._stats['orders']}  ", style="cyan")
            text.append("Risk: ", style="bold")
            text.append(f"{self._stats['risk']}  ", style="cyan")
            text.append("Warnings: ", style="bold")
            text.append(f"{self._stats['warnings']}  ", style="yellow")
            text.append("Errors: ", style="bold")
            text.append(f"{self._stats['errors']}", style="red")
            content.update(text)
        except Exception:
            pass


class AuditEventLogPanel(Static):
    """Panel showing live audit event log."""

    DEFAULT_CSS = """
    AuditEventLogPanel {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._log: RichLog | None = None

    def compose(self) -> ComposeResult:
        yield Label("[bold]Live Audit Log[/bold]")
        yield RichLog(
            highlight=True,
            markup=True,
            max_lines=500,
            wrap=False,
            auto_scroll=True,
            id="audit-log",
        )

    def on_mount(self) -> None:
        self._log = self.query_one("#audit-log", RichLog)
        self._log.can_focus = False

    def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "info",
        timestamp: datetime | None = None,
    ) -> None:
        if not self._log:
            return

        ts = timestamp or datetime.now(timezone.utc)
        ts_str = ts.strftime("%H:%M:%S")

        sev_color = SEVERITY_COLORS.get(severity, "white")
        evt_color = EVENT_TYPE_COLORS.get(event_type, sev_color)
        evt_short = event_type.split(".")[-1].upper()

        self._log.write(
            f"[dim]{ts_str}[/dim] [{evt_color}]{evt_short:12}[/{evt_color}] "
            f"[{sev_color}]{message}[/{sev_color}]"
        )


class AuditOrdersPanel(Static):
    """Panel showing order audit trails."""

    DEFAULT_CSS = """
    AuditOrdersPanel {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._orders: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Label("[bold]Order Audit Trails[/bold]")
        yield Static(id="orders-content")

    def on_mount(self) -> None:
        self._update_display()

    def add_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: str,
        status: str,
        risk_passed: bool,
        strategy: str,
        timestamp: datetime,
    ) -> None:
        self._orders.append({
            "time": timestamp.strftime("%H:%M:%S"),
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": status,
            "risk_passed": risk_passed,
            "strategy": strategy,
        })
        self._update_display()

    def _update_display(self) -> None:
        try:
            content = self.query_one("#orders-content", Static)

            if not self._orders:
                content.update("No order audit data")
                return

            table = Table(box=None, expand=True)
            table.add_column("Time", style="dim")
            table.add_column("Order ID", style="cyan")
            table.add_column("Symbol")
            table.add_column("Side")
            table.add_column("Qty", justify="right")
            table.add_column("Status")
            table.add_column("Risk")
            table.add_column("Strategy")

            for order in self._orders[-10:]:  # Last 10 orders
                side_style = "green" if order["side"].lower() == "buy" else "red"
                status_style = {
                    "filled": "green",
                    "rejected": "red",
                    "cancelled": "yellow",
                }.get(order["status"].lower(), "white")
                risk_text = Text("PASS", style="green") if order["risk_passed"] else Text("FAIL", style="red")

                table.add_row(
                    order["time"],
                    order["order_id"],
                    order["symbol"],
                    Text(order["side"].upper(), style=side_style),
                    order["quantity"],
                    Text(order["status"].upper(), style=status_style),
                    risk_text,
                    order["strategy"] or "-",
                )

            content.update(table)
        except Exception:
            pass


class AuditRiskPanel(Static):
    """Panel showing risk audit events."""

    DEFAULT_CSS = """
    AuditRiskPanel {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._risk_events: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Label("[bold]Risk Audit Events[/bold]")
        yield Static(id="risk-content")

    def on_mount(self) -> None:
        self._update_display()

    def add_risk_event(
        self,
        check_name: str,
        passed: bool,
        current_value: str,
        limit_value: str,
        utilization_pct: float,
        order_id: str,
        timestamp: datetime,
    ) -> None:
        self._risk_events.append({
            "time": timestamp.strftime("%H:%M:%S"),
            "check_name": check_name,
            "passed": passed,
            "current": current_value,
            "limit": limit_value,
            "utilization": utilization_pct,
            "order_id": order_id,
        })
        self._update_display()

    def _update_display(self) -> None:
        try:
            content = self.query_one("#risk-content", Static)

            if not self._risk_events:
                content.update("No risk audit data")
                return

            table = Table(box=None, expand=True)
            table.add_column("Time", style="dim")
            table.add_column("Check")
            table.add_column("Result")
            table.add_column("Current", justify="right")
            table.add_column("Limit", justify="right")
            table.add_column("Util %", justify="right")
            table.add_column("Order")

            for event in self._risk_events[-10:]:  # Last 10 events
                result_text = Text("PASS", style="green") if event["passed"] else Text("FAIL", style="red")

                util = event["utilization"]
                if util >= 100:
                    util_style = "red bold"
                elif util >= 80:
                    util_style = "yellow"
                else:
                    util_style = "green"

                table.add_row(
                    event["time"],
                    event["check_name"],
                    result_text,
                    event["current"],
                    event["limit"],
                    Text(f"{util:.1f}%", style=util_style),
                    event["order_id"] or "-",
                )

            content.update(table)
        except Exception:
            pass


class AuditDashboard(Container):
    """
    Audit logging dashboard widget.

    Displays:
    - Statistics summary
    - Real-time audit event log
    - Order audit trails
    - Risk events
    """

    DEFAULT_CSS = """
    AuditDashboard {
        height: 100%;
        width: 100%;
    }

    AuditDashboard > Vertical {
        height: 100%;
    }

    AuditDashboard Horizontal {
        height: 1fr;
    }

    AuditDashboard Horizontal > * {
        width: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stats_panel: AuditStatsPanel | None = None
        self._log_panel: AuditEventLogPanel | None = None
        self._orders_panel: AuditOrdersPanel | None = None
        self._risk_panel: AuditRiskPanel | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield AuditStatsPanel(id="audit-stats")
            yield AuditEventLogPanel(id="audit-event-log")
            with Horizontal():
                yield AuditOrdersPanel(id="audit-orders")
                yield AuditRiskPanel(id="audit-risk")

    def on_mount(self) -> None:
        """Cache panel references and load demo data."""
        self._stats_panel = self.query_one("#audit-stats", AuditStatsPanel)
        self._log_panel = self.query_one("#audit-event-log", AuditEventLogPanel)
        self._orders_panel = self.query_one("#audit-orders", AuditOrdersPanel)
        self._risk_panel = self.query_one("#audit-risk", AuditRiskPanel)

        # Load demo data
        self._load_demo_data()

    def _load_demo_data(self) -> None:
        """Load demo audit data."""
        now = datetime.now(timezone.utc)

        # Demo events
        demo_events = [
            ("system.start", "LIBRA Audit System initialized", "info", now - timedelta(minutes=30)),
            ("order.created", "Order ORD-001 created: BUY 0.5 BTC/USDT", "info", now - timedelta(minutes=25)),
            ("risk.check_passed", "Position limit check passed for ORD-001", "info", now - timedelta(minutes=25)),
            ("order.submitted", "Order ORD-001 submitted to exchange", "info", now - timedelta(minutes=24)),
            ("order.filled", "Order ORD-001 filled @ 42,500.00", "info", now - timedelta(minutes=24)),
            ("order.created", "Order ORD-002 created: BUY 5.0 ETH/USDT", "info", now - timedelta(minutes=20)),
            ("risk.check_passed", "Notional limit check passed for ORD-002", "info", now - timedelta(minutes=20)),
            ("order.filled", "Order ORD-002 filled @ 2,205.50", "info", now - timedelta(minutes=19)),
            ("order.created", "Order ORD-003 created: SELL 100 SOL/USDT", "warning", now - timedelta(minutes=10)),
            ("risk.check_failed", "Notional limit exceeded for ORD-003 (119%)", "warning", now - timedelta(minutes=10)),
            ("order.rejected", "Order ORD-003 rejected: risk limit exceeded", "error", now - timedelta(minutes=10)),
            ("agent.decision", "Momentum agent: BUY signal BTC/USDT (conf: 0.85)", "info", now - timedelta(minutes=5)),
        ]

        event_count = 0
        warning_count = 0
        error_count = 0

        for event_type, message, severity, timestamp in demo_events:
            if self._log_panel:
                self._log_panel.log_event(event_type, message, severity, timestamp)
            event_count += 1
            if severity == "warning":
                warning_count += 1
            elif severity == "error":
                error_count += 1

        # Demo orders
        demo_orders = [
            ("ORD-001", "BTC/USDT", "buy", "0.5", "filled", True, "momentum_v1", now - timedelta(minutes=24)),
            ("ORD-002", "ETH/USDT", "buy", "5.0", "filled", True, "mean_reversion", now - timedelta(minutes=19)),
            ("ORD-003", "SOL/USDT", "sell", "100.0", "rejected", False, "momentum_v1", now - timedelta(minutes=10)),
        ]

        for order_id, symbol, side, qty, status, risk, strategy, ts in demo_orders:
            if self._orders_panel:
                self._orders_panel.add_order(order_id, symbol, side, qty, status, risk, strategy, ts)

        # Demo risk events
        demo_risk = [
            ("position_limit", True, "0.5", "2.0", 25.0, "ORD-001", now - timedelta(minutes=25)),
            ("notional_limit", True, "11027.50", "50000.00", 22.05, "ORD-002", now - timedelta(minutes=20)),
            ("notional_limit", False, "59500.00", "50000.00", 119.0, "ORD-003", now - timedelta(minutes=10)),
        ]

        for check, passed, current, limit, util, order_id, ts in demo_risk:
            if self._risk_panel:
                self._risk_panel.add_risk_event(check, passed, current, limit, util, order_id, ts)

        # Update stats
        if self._stats_panel:
            self._stats_panel.update_stats(
                events=event_count,
                orders=len(demo_orders),
                risk=len(demo_risk),
                warnings=warning_count,
                errors=error_count,
            )


def create_demo_audit_dashboard() -> AuditDashboard:
    """Create audit dashboard with demo data."""
    return AuditDashboard()

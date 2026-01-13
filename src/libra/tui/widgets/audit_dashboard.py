"""
Audit Log Dashboard for TUI.

Displays audit events, order trails, and risk events in real-time.
Part of Issue #16: Audit Logging System.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import DataTable, Static


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
        height: 5;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    AuditStatsPanel .header {
        text-style: bold;
        color: $text;
    }

    AuditStatsPanel .stats-row {
        height: 2;
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
        yield Static("AUDIT STATISTICS", classes="header")
        yield Static(id="stats-content", classes="stats-row")

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

    def _update_display(self) -> None:
        try:
            content = self.query_one("#stats-content", Static)
            text = Text()
            text.append("Events: ", style="dim")
            text.append(f"{self._stats['events']}  ", style="cyan")
            text.append("Orders: ", style="dim")
            text.append(f"{self._stats['orders']}  ", style="cyan")
            text.append("Risk: ", style="dim")
            text.append(f"{self._stats['risk']}  ", style="cyan")
            text.append("Warnings: ", style="dim")
            text.append(f"{self._stats['warnings']}  ", style="yellow")
            text.append("Errors: ", style="dim")
            text.append(f"{self._stats['errors']}", style="red")
            content.update(text)
        except Exception:
            pass


class AuditEventLogPanel(Static):
    """Panel showing live audit event log."""

    DEFAULT_CSS = """
    AuditEventLogPanel {
        height: 100%;
        border: round $primary-darken-2;
        background: $surface;
    }

    AuditEventLogPanel DataTable {
        height: 1fr;
    }

    AuditEventLogPanel .header {
        background: $primary-darken-3;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: list[tuple[str, str, str, datetime]] = []

    def compose(self) -> ComposeResult:
        yield Static("LIVE AUDIT LOG", classes="header")
        table = DataTable(id="events-table")
        table.cursor_type = "row"
        yield table

    def on_mount(self) -> None:
        table = self.query_one("#events-table", DataTable)
        table.add_columns("Time", "Type", "Message")
        self._update_display()

    def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "info",
        timestamp: datetime | None = None,
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc)
        self._events.append((event_type, message, severity, ts))
        self._update_display()

    def _update_display(self) -> None:
        try:
            table = self.query_one("#events-table", DataTable)
            table.clear()

            for event_type, message, severity, ts in self._events[-15:]:
                ts_str = ts.strftime("%H:%M:%S")
                evt_short = event_type.split(".")[-1].upper()

                # Color based on severity
                if severity == "error" or severity == "critical":
                    msg_display = f"[red]{message}[/red]"
                    type_display = f"[red]{evt_short}[/red]"
                elif severity == "warning":
                    msg_display = f"[yellow]{message}[/yellow]"
                    type_display = f"[yellow]{evt_short}[/yellow]"
                else:
                    evt_color = EVENT_TYPE_COLORS.get(event_type, "white")
                    msg_display = message
                    type_display = f"[{evt_color}]{evt_short}[/{evt_color}]"

                table.add_row(ts_str, type_display, msg_display)
        except Exception:
            pass


class AuditOrdersPanel(Static):
    """Panel showing order audit trails."""

    DEFAULT_CSS = """
    AuditOrdersPanel {
        height: 100%;
        border: round $primary-darken-2;
        background: $surface;
    }

    AuditOrdersPanel DataTable {
        height: 1fr;
    }

    AuditOrdersPanel .header {
        background: $primary-darken-3;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._orders: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Static("ORDER AUDIT TRAILS", classes="header")
        table = DataTable(id="orders-table")
        table.cursor_type = "row"
        yield table

    def on_mount(self) -> None:
        table = self.query_one("#orders-table", DataTable)
        table.add_columns("Time", "ID", "Symbol", "Side", "Qty", "Status", "Risk")
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
            table = self.query_one("#orders-table", DataTable)
            table.clear()

            for order in self._orders[-8:]:
                side_style = "green" if order["side"].lower() == "buy" else "red"
                status_lower = order["status"].lower()
                if status_lower == "filled":
                    status_style = "green"
                elif status_lower == "rejected":
                    status_style = "red"
                elif status_lower == "cancelled":
                    status_style = "yellow"
                else:
                    status_style = "white"

                risk_text = "[green]PASS[/green]" if order["risk_passed"] else "[red]FAIL[/red]"

                table.add_row(
                    order["time"],
                    order["order_id"],
                    order["symbol"],
                    f"[{side_style}]{order['side'].upper()}[/{side_style}]",
                    order["quantity"],
                    f"[{status_style}]{order['status'].upper()}[/{status_style}]",
                    risk_text,
                )
        except Exception:
            pass


class AuditRiskPanel(Static):
    """Panel showing risk audit events."""

    DEFAULT_CSS = """
    AuditRiskPanel {
        height: 100%;
        border: round $primary-darken-2;
        background: $surface;
    }

    AuditRiskPanel DataTable {
        height: 1fr;
    }

    AuditRiskPanel .header {
        background: $primary-darken-3;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._risk_events: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Static("RISK AUDIT EVENTS", classes="header")
        table = DataTable(id="risk-table")
        table.cursor_type = "row"
        yield table

    def on_mount(self) -> None:
        table = self.query_one("#risk-table", DataTable)
        table.add_columns("Time", "Check", "Result", "Current", "Limit", "Util %")
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
            table = self.query_one("#risk-table", DataTable)
            table.clear()

            for event in self._risk_events[-8:]:
                result_text = "[green]PASS[/green]" if event["passed"] else "[red]FAIL[/red]"

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
                    f"[{util_style}]{util:.1f}%[/{util_style}]",
                )
        except Exception:
            pass


class AuditDashboard(Container):
    """
    Audit logging dashboard widget.

    Displays audit statistics, live event log, order trails, and risk events.
    """

    DEFAULT_CSS = """
    AuditDashboard {
        height: 1fr;
        min-height: 25;
        padding: 1;
    }

    AuditDashboard #stats-row {
        height: 5;
        margin-bottom: 1;
    }

    AuditDashboard #log-row {
        height: 1fr;
        min-height: 10;
        margin-bottom: 1;
    }

    AuditDashboard #log-row AuditEventLogPanel {
        width: 100%;
    }

    AuditDashboard #details-row {
        height: 1fr;
        min-height: 10;
    }

    AuditDashboard #orders-container {
        width: 1fr;
        height: 100%;
        margin-right: 1;
    }

    AuditDashboard #risk-container {
        width: 1fr;
        height: 100%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stats_panel: AuditStatsPanel | None = None
        self._log_panel: AuditEventLogPanel | None = None
        self._orders_panel: AuditOrdersPanel | None = None
        self._risk_panel: AuditRiskPanel | None = None

    def compose(self) -> ComposeResult:
        # Stats row at top
        with Horizontal(id="stats-row"):
            yield AuditStatsPanel(id="audit-stats")

        # Event log row
        with Horizontal(id="log-row"):
            yield AuditEventLogPanel(id="audit-log")

        # Orders and Risk side by side
        with Horizontal(id="details-row"):
            with Container(id="orders-container"):
                yield AuditOrdersPanel(id="audit-orders")
            with Container(id="risk-container"):
                yield AuditRiskPanel(id="audit-risk")

    def on_mount(self) -> None:
        """Cache panel references and load demo data."""
        self._stats_panel = self.query_one("#audit-stats", AuditStatsPanel)
        self._log_panel = self.query_one("#audit-log", AuditEventLogPanel)
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
            ("order.created", "Order ORD-001: BUY 0.5 BTC/USDT", "info", now - timedelta(minutes=25)),
            ("risk.check_passed", "Position limit passed for ORD-001", "info", now - timedelta(minutes=25)),
            ("order.submitted", "ORD-001 submitted to exchange", "info", now - timedelta(minutes=24)),
            ("order.filled", "ORD-001 filled @ 42,500", "info", now - timedelta(minutes=24)),
            ("order.created", "Order ORD-002: BUY 5.0 ETH/USDT", "info", now - timedelta(minutes=20)),
            ("risk.check_passed", "Notional limit passed ORD-002", "info", now - timedelta(minutes=20)),
            ("order.filled", "ORD-002 filled @ 2,205.50", "info", now - timedelta(minutes=19)),
            ("order.created", "Order ORD-003: SELL 100 SOL", "warning", now - timedelta(minutes=10)),
            ("risk.check_failed", "Notional limit exceeded (119%)", "warning", now - timedelta(minutes=10)),
            ("order.rejected", "ORD-003 rejected: risk limit", "error", now - timedelta(minutes=10)),
            ("agent.decision", "Momentum: BUY BTC (0.85)", "info", now - timedelta(minutes=5)),
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
            ("ORD-001", "BTC/USDT", "buy", "0.5", "filled", True, "momentum", now - timedelta(minutes=24)),
            ("ORD-002", "ETH/USDT", "buy", "5.0", "filled", True, "reversion", now - timedelta(minutes=19)),
            ("ORD-003", "SOL/USDT", "sell", "100", "rejected", False, "momentum", now - timedelta(minutes=10)),
        ]

        for order_id, symbol, side, qty, status, risk, strategy, ts in demo_orders:
            if self._orders_panel:
                self._orders_panel.add_order(order_id, symbol, side, qty, status, risk, strategy, ts)

        # Demo risk events
        demo_risk = [
            ("position_limit", True, "0.5", "2.0", 25.0, "ORD-001", now - timedelta(minutes=25)),
            ("notional_limit", True, "11027", "50000", 22.05, "ORD-002", now - timedelta(minutes=20)),
            ("notional_limit", False, "59500", "50000", 119.0, "ORD-003", now - timedelta(minutes=10)),
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

"""
Audit Log Dashboard for TUI.

Displays audit events, order trails, and risk events in real-time.
Part of Issue #16: Audit Logging System.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Label, RichLog, Static, TabbedContent, TabPane


# Severity to color mapping
SEVERITY_COLORS: dict[str, str] = {
    "debug": "dim",
    "info": "white",
    "warning": "yellow",
    "error": "red",
    "critical": "red bold reverse",
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
    "risk.circuit_breaker": "red bold reverse",
    "agent.decision": "magenta",
    "agent.signal": "magenta",
    "system.error": "red",
}


class AuditDashboard(Vertical):
    """
    Audit logging dashboard widget.

    Displays:
    - Real-time audit event log
    - Order audit trails table
    - Risk events table
    - Statistics summary

    Features:
    - Color-coded by severity and event type
    - Filterable by event type
    - Searchable message text
    - Auto-refresh with live data
    """

    DEFAULT_CSS = """
    AuditDashboard {
        height: 100%;
        padding: 0;
    }

    AuditDashboard > TabbedContent {
        height: 100%;
    }

    AuditDashboard .stats-row {
        height: 3;
        padding: 0 1;
        background: $surface-darken-1;
    }

    AuditDashboard .stats-row > Static {
        width: 1fr;
        content-align: center middle;
    }

    AuditDashboard .audit-log {
        height: 1fr;
        border: round $primary-darken-1;
        background: $surface;
        padding: 0 1;
    }

    AuditDashboard DataTable {
        height: 1fr;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._event_log: RichLog | None = None
        self._orders_table: DataTable | None = None
        self._risk_table: DataTable | None = None
        self._stats: dict[str, int] = {
            "total_events": 0,
            "orders": 0,
            "risk_events": 0,
            "warnings": 0,
            "errors": 0,
        }

    def compose(self) -> ComposeResult:
        """Create dashboard layout."""
        # Statistics row
        with Horizontal(classes="stats-row"):
            yield Static("Events: 0", id="stat-events")
            yield Static("Orders: 0", id="stat-orders")
            yield Static("Risk: 0", id="stat-risk")
            yield Static("Warnings: 0", id="stat-warnings")
            yield Static("Errors: 0", id="stat-errors")

        # Tabbed content
        with TabbedContent():
            with TabPane("Live Log", id="tab-log"):
                yield RichLog(
                    highlight=True,
                    markup=True,
                    max_lines=1000,
                    wrap=False,
                    auto_scroll=True,
                    classes="audit-log",
                    id="audit-event-log",
                )

            with TabPane("Orders", id="tab-orders"):
                yield DataTable(id="orders-audit-table")

            with TabPane("Risk Events", id="tab-risk"):
                yield DataTable(id="risk-audit-table")

    def on_mount(self) -> None:
        """Initialize tables after mount."""
        # Cache references
        self._event_log = self.query_one("#audit-event-log", RichLog)
        self._orders_table = self.query_one("#orders-audit-table", DataTable)
        self._risk_table = self.query_one("#risk-audit-table", DataTable)

        # Configure event log
        self._event_log.can_focus = False

        # Setup orders table
        self._orders_table.add_columns(
            "Time", "Order ID", "Symbol", "Side", "Qty", "Status", "Risk", "Strategy"
        )
        self._orders_table.cursor_type = "row"

        # Setup risk table
        self._risk_table.add_columns(
            "Time", "Check", "Passed", "Current", "Limit", "Util %", "Order ID"
        )
        self._risk_table.cursor_type = "row"

        # Load demo data
        self.set_demo_data()

    def log_audit_event(
        self,
        event_type: str,
        message: str,
        severity: str = "info",
        timestamp: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an audit event to the live log.

        Args:
            event_type: Type of event (e.g., "order.created")
            message: Human-readable message
            severity: Severity level
            timestamp: Event timestamp
            details: Additional details
        """
        if not self._event_log:
            return

        ts = timestamp or datetime.now(timezone.utc)
        ts_str = ts.strftime("%H:%M:%S")

        # Get colors
        sev_color = SEVERITY_COLORS.get(severity, "white")
        evt_color = EVENT_TYPE_COLORS.get(event_type, sev_color)

        # Format event type for display
        evt_short = event_type.split(".")[-1].upper()

        self._event_log.write(
            f"[dim]{ts_str}[/dim] [{evt_color}]{evt_short:12}[/{evt_color}] "
            f"[{sev_color}]{message}[/{sev_color}]"
        )

        # Update stats
        self._stats["total_events"] += 1
        if severity == "warning":
            self._stats["warnings"] += 1
        elif severity in ("error", "critical"):
            self._stats["errors"] += 1

        self._update_stats_display()

    def add_order_audit(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal | str,
        status: str,
        risk_passed: bool = True,
        strategy: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        """
        Add an order audit trail to the table.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: buy/sell
            quantity: Order quantity
            status: Order status
            risk_passed: Whether risk check passed
            strategy: Strategy name
            timestamp: Order timestamp
        """
        if not self._orders_table:
            return

        ts = timestamp or datetime.now(timezone.utc)
        ts_str = ts.strftime("%H:%M:%S")

        # Color based on status
        status_colors = {
            "filled": "green",
            "rejected": "red",
            "cancelled": "yellow",
            "pending": "dim",
            "submitted": "cyan",
        }
        status_color = status_colors.get(status.lower(), "white")

        side_color = "green" if side.lower() == "buy" else "red"
        risk_display = "[green]PASS[/green]" if risk_passed else "[red]FAIL[/red]"

        self._orders_table.add_row(
            ts_str,
            order_id,
            symbol,
            f"[{side_color}]{side.upper()}[/{side_color}]",
            str(quantity),
            f"[{status_color}]{status.upper()}[/{status_color}]",
            risk_display,
            strategy or "-",
        )

        self._stats["orders"] += 1
        self._update_stats_display()

    def add_risk_audit(
        self,
        check_name: str,
        passed: bool,
        current_value: Decimal | str,
        limit_value: Decimal | str,
        utilization_pct: float,
        order_id: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        """
        Add a risk audit event to the table.

        Args:
            check_name: Name of risk check
            passed: Whether check passed
            current_value: Current value
            limit_value: Limit value
            utilization_pct: Utilization percentage
            order_id: Related order ID
            timestamp: Event timestamp
        """
        if not self._risk_table:
            return

        ts = timestamp or datetime.now(timezone.utc)
        ts_str = ts.strftime("%H:%M:%S")

        passed_display = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"

        # Color utilization based on level
        if utilization_pct >= 100:
            util_color = "red bold"
        elif utilization_pct >= 80:
            util_color = "yellow"
        else:
            util_color = "green"

        self._risk_table.add_row(
            ts_str,
            check_name,
            passed_display,
            str(current_value),
            str(limit_value),
            f"[{util_color}]{utilization_pct:.1f}%[/{util_color}]",
            order_id or "-",
        )

        self._stats["risk_events"] += 1
        if not passed:
            self._stats["warnings"] += 1
        self._update_stats_display()

    def _update_stats_display(self) -> None:
        """Update statistics display."""
        try:
            self.query_one("#stat-events", Static).update(
                f"Events: {self._stats['total_events']}"
            )
            self.query_one("#stat-orders", Static).update(
                f"Orders: {self._stats['orders']}"
            )
            self.query_one("#stat-risk", Static).update(
                f"Risk: {self._stats['risk_events']}"
            )
            self.query_one("#stat-warnings", Static).update(
                f"[yellow]Warnings: {self._stats['warnings']}[/yellow]"
            )
            self.query_one("#stat-errors", Static).update(
                f"[red]Errors: {self._stats['errors']}[/red]"
            )
        except Exception:
            pass  # Widget may not be mounted yet

    def clear_logs(self) -> None:
        """Clear all logs and reset statistics."""
        if self._event_log:
            self._event_log.clear()
        if self._orders_table:
            self._orders_table.clear()
        if self._risk_table:
            self._risk_table.clear()

        self._stats = {
            "total_events": 0,
            "orders": 0,
            "risk_events": 0,
            "warnings": 0,
            "errors": 0,
        }
        self._update_stats_display()

    def set_demo_data(self) -> None:
        """Populate with demo audit data."""
        now = datetime.now(timezone.utc)

        # Demo audit events
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
            ("agent.decision", "Momentum agent: BUY signal for BTC/USDT (confidence: 0.85)", "info", now - timedelta(minutes=5)),
        ]

        for event_type, message, severity, timestamp in demo_events:
            self.log_audit_event(event_type, message, severity, timestamp)

        # Demo order audits
        demo_orders = [
            ("ORD-001", "BTC/USDT", "buy", "0.5", "filled", True, "momentum_v1", now - timedelta(minutes=24)),
            ("ORD-002", "ETH/USDT", "buy", "5.0", "filled", True, "mean_reversion", now - timedelta(minutes=19)),
            ("ORD-003", "SOL/USDT", "sell", "100.0", "rejected", False, "momentum_v1", now - timedelta(minutes=10)),
        ]

        for order_id, symbol, side, qty, status, risk, strategy, ts in demo_orders:
            self.add_order_audit(order_id, symbol, side, qty, status, risk, strategy, ts)

        # Demo risk events
        demo_risk = [
            ("position_limit", True, "0.5", "2.0", 25.0, "ORD-001", now - timedelta(minutes=25)),
            ("notional_limit", True, "11027.50", "50000.00", 22.05, "ORD-002", now - timedelta(minutes=20)),
            ("notional_limit", False, "59500.00", "50000.00", 119.0, "ORD-003", now - timedelta(minutes=10)),
        ]

        for check, passed, current, limit, util, order_id, ts in demo_risk:
            self.add_risk_audit(check, passed, current, limit, util, order_id, ts)


def create_demo_audit_dashboard() -> AuditDashboard:
    """Create audit dashboard with demo data."""
    dashboard = AuditDashboard()
    return dashboard

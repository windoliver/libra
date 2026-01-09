"""
Order History Widget.

Displays order history from the gateway using CCXTOrderFetcher (Issue #27).

Features:
- Shows open and closed orders
- Real-time updates from gateway
- Color-coded status display
- Filtering by status/symbol
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static

if TYPE_CHECKING:
    from textual.widgets.data_table import RowKey

    from libra.gateways.protocol import OrderResult


# =============================================================================
# Filter Options
# =============================================================================


class OrderStatusFilter(Enum):
    """Filter options for order status."""

    ALL = "All Orders"
    OPEN = "Open"
    FILLED = "Filled"
    CANCELLED = "Cancelled"


# =============================================================================
# Order History Table Widget
# =============================================================================


class OrderHistoryTable(Container):
    """
    Order history table showing orders from gateway.

    Uses gateway.get_order_history() which delegates to CCXTOrderFetcher.

    Features:
    - Click column headers to sort
    - Filter by status, symbol
    - Color-coded status values
    - Real-time refresh
    """

    DEFAULT_CSS = """
    OrderHistoryTable {
        height: 100%;
        width: 100%;
        layout: vertical;
    }

    OrderHistoryTable .table-header {
        height: 3;
        padding: 0 1;
        background: $surface-darken-1;
    }

    OrderHistoryTable .table-title {
        width: auto;
        padding-right: 2;
        text-style: bold;
    }

    OrderHistoryTable .filter-controls {
        width: 1fr;
    }

    OrderHistoryTable .filter-select {
        width: 15;
        margin-right: 1;
    }

    OrderHistoryTable .filter-input {
        width: 20;
        margin-right: 1;
    }

    OrderHistoryTable .table-container {
        height: 1fr;
    }

    OrderHistoryTable DataTable {
        height: 100%;
    }

    OrderHistoryTable .table-footer {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("c", "clear_filters", "Clear Filters"),
    ]

    # Reactive state
    filter_status: reactive[OrderStatusFilter] = reactive(OrderStatusFilter.ALL)
    filter_symbol: reactive[str] = reactive("")

    class OrderSelected(Message):
        """Message sent when an order is selected."""

        def __init__(self, order: OrderResult) -> None:
            super().__init__()
            self.order = order

    def __init__(
        self,
        orders: list[OrderResult] | None = None,
        title: str = "ORDER HISTORY",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._orders: list[OrderResult] = orders or []
        self._filtered_orders: list[OrderResult] = []
        self._title = title
        self._row_to_order: dict[RowKey, OrderResult] = {}

    def compose(self) -> ComposeResult:
        # Header with title and filters
        with Horizontal(classes="table-header"):
            yield Static(self._title, classes="table-title")
            with Horizontal(classes="filter-controls"):
                yield Select(
                    [(f.value, f) for f in OrderStatusFilter],
                    value=OrderStatusFilter.ALL,
                    id="filter-status",
                    classes="filter-select",
                )
                yield Input(
                    placeholder="Filter symbol...",
                    id="filter-symbol",
                    classes="filter-input",
                )
                yield Button("Refresh", id="refresh-btn", variant="primary")

        # Table
        with Container(classes="table-container"):
            yield DataTable(id="orders-table", cursor_type="row", zebra_stripes=True)

        # Footer with stats
        yield Static(self._get_footer_text(), id="table-footer", classes="table-footer")

    def on_mount(self) -> None:
        """Set up the table on mount."""
        table = self.query_one("#orders-table", DataTable)

        # Add columns
        table.add_column("Time", key="time")
        table.add_column("Symbol", key="symbol")
        table.add_column("Side", key="side")
        table.add_column("Type", key="type")
        table.add_column("Status", key="status")
        table.add_column("Amount", key="amount")
        table.add_column("Filled", key="filled")
        table.add_column("Price", key="price")
        table.add_column("Avg Fill", key="avg")

        # Populate table
        self._apply_filters()
        self._populate_table()

    def _get_footer_text(self) -> str:
        """Generate footer statistics text."""
        total = len(self._orders)
        filtered = len(self._filtered_orders)
        if total == 0:
            return "No orders"

        open_count = sum(1 for o in self._filtered_orders if o.status.value in ("open", "partially_filled"))
        filled_count = sum(1 for o in self._filtered_orders if o.status.value == "filled")
        cancelled_count = sum(1 for o in self._filtered_orders if o.status.value in ("cancelled", "rejected"))

        return (
            f"Showing {filtered}/{total} orders  |  "
            f"[cyan]Open: {open_count}[/cyan]  "
            f"[green]Filled: {filled_count}[/green]  "
            f"[yellow]Cancelled: {cancelled_count}[/yellow]"
        )

    def _apply_filters(self) -> None:
        """Apply current filters to orders."""
        self._filtered_orders = self._orders.copy()

        # Filter by status
        if self.filter_status == OrderStatusFilter.OPEN:
            self._filtered_orders = [
                o for o in self._filtered_orders
                if o.status.value in ("open", "partially_filled", "pending")
            ]
        elif self.filter_status == OrderStatusFilter.FILLED:
            self._filtered_orders = [
                o for o in self._filtered_orders
                if o.status.value == "filled"
            ]
        elif self.filter_status == OrderStatusFilter.CANCELLED:
            self._filtered_orders = [
                o for o in self._filtered_orders
                if o.status.value in ("cancelled", "rejected", "expired")
            ]

        # Filter by symbol
        if self.filter_symbol:
            symbol_filter = self.filter_symbol.upper()
            self._filtered_orders = [
                o for o in self._filtered_orders
                if symbol_filter in o.symbol.upper()
            ]

        # Sort by time descending
        self._filtered_orders.sort(key=lambda o: o.timestamp_ns, reverse=True)

    def _populate_table(self) -> None:
        """Populate table with filtered orders."""
        table = self.query_one("#orders-table", DataTable)
        table.clear()
        self._row_to_order.clear()

        for order in self._filtered_orders:
            # Format time
            time_str = datetime.fromtimestamp(order.timestamp_ns / 1e9).strftime("%Y-%m-%d %H:%M")

            # Format side with color
            side_str = (
                f"[green]{order.side.value.upper()}[/green]"
                if order.side.value == "buy"
                else f"[red]{order.side.value.upper()}[/red]"
            )

            # Format status with color
            status_colors = {
                "open": "cyan",
                "pending": "cyan",
                "partially_filled": "yellow",
                "filled": "green",
                "cancelled": "dim",
                "rejected": "red",
                "expired": "dim",
            }
            status_color = status_colors.get(order.status.value, "white")
            status_str = f"[{status_color}]{order.status.value.upper()}[/{status_color}]"

            # Format prices
            price_str = f"${order.price:,.2f}" if order.price else "MARKET"
            avg_str = f"${order.average_price:,.2f}" if order.average_price else "-"

            # Format amounts
            filled_pct = (order.filled_amount / order.amount * 100) if order.amount > 0 else 0
            filled_str = f"{order.filled_amount:,.4f} ({filled_pct:.0f}%)"

            row_key = table.add_row(
                time_str,
                order.symbol,
                side_str,
                order.order_type.value.upper(),
                status_str,
                f"{order.amount:,.4f}",
                filled_str,
                price_str,
                avg_str,
                key=order.order_id,
            )
            self._row_to_order[row_key] = order

        # Update footer
        try:
            footer = self.query_one("#table-footer", Static)
            footer.update(self._get_footer_text())
        except Exception:
            pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key in self._row_to_order:
            order = self._row_to_order[event.row_key]
            self.post_message(self.OrderSelected(order))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter select change."""
        if event.select.id == "filter-status" and isinstance(event.value, OrderStatusFilter):
            self.filter_status = event.value
            self._apply_filters()
            self._populate_table()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input change."""
        if event.input.id == "filter-symbol":
            self.filter_symbol = event.value
            self._apply_filters()
            self._populate_table()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "refresh-btn":
            self.post_message(self.RefreshRequested())

    class RefreshRequested(Message):
        """Message sent when refresh is requested."""

        pass

    def action_refresh(self) -> None:
        """Refresh the table."""
        self.post_message(self.RefreshRequested())

    def action_clear_filters(self) -> None:
        """Clear all filters."""
        self.filter_status = OrderStatusFilter.ALL
        self.filter_symbol = ""

        try:
            status_select = self.query_one("#filter-status", Select)
            status_select.value = OrderStatusFilter.ALL
        except Exception:
            pass

        try:
            symbol_input = self.query_one("#filter-symbol", Input)
            symbol_input.value = ""
        except Exception:
            pass

        self._apply_filters()
        self._populate_table()

    def update_orders(self, orders: list[OrderResult]) -> None:
        """Update table with new orders."""
        self._orders = orders
        self._apply_filters()
        self._populate_table()


# =============================================================================
# Fill History Table (Individual Trades/Fills)
# =============================================================================


class FillHistoryTable(Container):
    """
    Fill/trade history table showing individual executions.

    Uses gateway.get_trades() which delegates to CCXTTradeFetcher.
    """

    DEFAULT_CSS = """
    FillHistoryTable {
        height: 100%;
        width: 100%;
        layout: vertical;
    }

    FillHistoryTable .table-header {
        height: 3;
        padding: 0 1;
        background: $surface-darken-1;
    }

    FillHistoryTable .table-title {
        width: auto;
        padding-right: 2;
        text-style: bold;
    }

    FillHistoryTable .filter-input {
        width: 20;
        margin-right: 1;
    }

    FillHistoryTable .table-container {
        height: 1fr;
    }

    FillHistoryTable DataTable {
        height: 100%;
    }

    FillHistoryTable .table-footer {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        trades: list | None = None,
        title: str = "FILL HISTORY",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._trades = trades or []
        self._title = title

    def compose(self) -> ComposeResult:
        # Header
        with Horizontal(classes="table-header"):
            yield Static(self._title, classes="table-title")
            yield Input(
                placeholder="Filter symbol...",
                id="filter-symbol",
                classes="filter-input",
            )
            yield Button("Refresh", id="refresh-btn", variant="primary")

        # Table
        with Container(classes="table-container"):
            yield DataTable(id="fills-table", cursor_type="row", zebra_stripes=True)

        # Footer
        yield Static(self._get_footer_text(), id="table-footer", classes="table-footer")

    def on_mount(self) -> None:
        """Set up the table on mount."""
        table = self.query_one("#fills-table", DataTable)

        # Add columns
        table.add_column("Time", key="time")
        table.add_column("Trade ID", key="trade_id")
        table.add_column("Order ID", key="order_id")
        table.add_column("Symbol", key="symbol")
        table.add_column("Side", key="side")
        table.add_column("Amount", key="amount")
        table.add_column("Price", key="price")
        table.add_column("Cost", key="cost")
        table.add_column("Fee", key="fee")

        self._populate_table()

    def _get_footer_text(self) -> str:
        """Generate footer text."""
        total = len(self._trades)
        if total == 0:
            return "No fills"

        total_cost = sum(getattr(t, "cost", Decimal("0")) for t in self._trades)
        total_fees = sum(getattr(t, "fee", Decimal("0")) or Decimal("0") for t in self._trades)

        return f"Total: {total} fills  |  Volume: ${total_cost:,.2f}  |  Fees: ${total_fees:,.2f}"

    def _populate_table(self) -> None:
        """Populate table with trades."""
        table = self.query_one("#fills-table", DataTable)
        table.clear()

        for trade in self._trades:
            # Format time
            time_str = datetime.fromtimestamp(trade.timestamp_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S")

            # Format side with color
            side_str = (
                f"[green]{trade.side.upper()}[/green]"
                if trade.side == "buy"
                else f"[red]{trade.side.upper()}[/red]"
            )

            # Format fee
            fee_str = f"${trade.fee:,.4f}" if trade.fee else "-"
            if trade.fee_currency and trade.fee:
                fee_str = f"{trade.fee:,.4f} {trade.fee_currency}"

            table.add_row(
                time_str,
                trade.trade_id[:12] + "..." if len(trade.trade_id) > 15 else trade.trade_id,
                trade.order_id[:12] + "..." if len(trade.order_id) > 15 else trade.order_id,
                trade.symbol,
                side_str,
                f"{trade.amount:,.6f}",
                f"${trade.price:,.2f}",
                f"${trade.cost:,.2f}",
                fee_str,
                key=trade.trade_id,
            )

        # Update footer
        try:
            footer = self.query_one("#table-footer", Static)
            footer.update(self._get_footer_text())
        except Exception:
            pass

    class RefreshRequested(Message):
        """Message sent when refresh is requested."""

        pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "refresh-btn":
            self.post_message(self.RefreshRequested())

    def action_refresh(self) -> None:
        """Refresh the table."""
        self.post_message(self.RefreshRequested())

    def update_trades(self, trades: list) -> None:
        """Update table with new trades."""
        self._trades = trades
        self._populate_table()

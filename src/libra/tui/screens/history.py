"""
History Screen.

Displays order history and fill/trade history using Gateway fetchers (Issue #27).

Features:
- Order History table (open, filled, cancelled orders)
- Fill History table (individual trade executions)
- Real-time refresh from gateway
- Filtering and sorting
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static, TabbedContent, TabPane

from libra.tui.widgets.order_history import FillHistoryTable, OrderHistoryTable

if TYPE_CHECKING:
    from libra.gateways.protocol import Gateway


class HistoryScreen(Screen):
    """
    Screen showing order and fill history.

    Uses:
    - gateway.get_order_history() -> CCXTOrderFetcher
    - gateway.get_trades() -> CCXTTradeFetcher
    """

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("o", "show_orders", "Orders"),
        Binding("f", "show_fills", "Fills"),
    ]

    DEFAULT_CSS = """
    HistoryScreen {
        layout: vertical;
    }

    HistoryScreen .screen-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        background: $primary-darken-2;
    }

    HistoryScreen .content-area {
        height: 1fr;
        padding: 1;
    }

    HistoryScreen .button-row {
        height: 3;
        align: center middle;
        padding: 0 1;
    }

    HistoryScreen TabbedContent {
        height: 100%;
    }

    HistoryScreen TabPane {
        padding: 1;
    }

    HistoryScreen .loading-message {
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    """

    def __init__(
        self,
        gateway: Gateway | None = None,
        name: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self._gateway = gateway

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("ORDER & TRADE HISTORY", classes="screen-title")

        with Container(classes="content-area"):
            with TabbedContent(id="history-tabs"):
                with TabPane("Orders", id="orders-tab"):
                    yield OrderHistoryTable(id="order-history-table")

                with TabPane("Fills", id="fills-tab"):
                    yield FillHistoryTable(id="fill-history-table")

        with Horizontal(classes="button-row"):
            yield Button("Refresh", id="refresh-btn", variant="primary")
            yield Button("Back", id="back-btn", variant="default")

        yield Footer()

    async def on_mount(self) -> None:
        """Load data on mount."""
        await self._load_data()

    async def _load_data(self) -> None:
        """Load order and fill history from gateway."""
        if not self._gateway:
            self.notify("No gateway connected", severity="warning")
            return

        if not self._gateway.is_connected:
            self.notify("Gateway not connected", severity="warning")
            return

        # Load order history
        try:
            orders = await self._gateway.get_order_history(limit=100)
            order_table = self.query_one("#order-history-table", OrderHistoryTable)
            order_table.update_orders(orders)
            self.notify(f"Loaded {len(orders)} orders")
        except Exception as e:
            self.notify(f"Failed to load orders: {e}", severity="error")

        # Load fill history
        try:
            # get_trades returns TradeRecord from fetcher
            trades = await self._gateway.get_trades(limit=100)
            fill_table = self.query_one("#fill-history-table", FillHistoryTable)
            fill_table.update_trades(trades)
            self.notify(f"Loaded {len(trades)} fills")
        except Exception as e:
            self.notify(f"Failed to load fills: {e}", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh-btn":
            self.action_refresh()
        elif event.button.id == "back-btn":
            self.action_pop_screen()

    def on_order_history_table_refresh_requested(
        self, event: OrderHistoryTable.RefreshRequested
    ) -> None:
        """Handle refresh request from order table."""
        self.action_refresh()

    def on_fill_history_table_refresh_requested(
        self, event: FillHistoryTable.RefreshRequested
    ) -> None:
        """Handle refresh request from fill table."""
        self.action_refresh()

    def action_refresh(self) -> None:
        """Refresh data from gateway."""
        self.run_worker(self._load_data())

    def action_show_orders(self) -> None:
        """Switch to orders tab."""
        tabs = self.query_one("#history-tabs", TabbedContent)
        tabs.active = "orders-tab"

    def action_show_fills(self) -> None:
        """Switch to fills tab."""
        tabs = self.query_one("#history-tabs", TabbedContent)
        tabs.active = "fills-tab"

    def action_pop_screen(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()

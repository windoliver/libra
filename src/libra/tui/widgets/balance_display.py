"""
Balance display widget showing account balances.

Professional styling with border title and zebra stripes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable


if TYPE_CHECKING:
    from libra.gateways.protocol import Balance


class BalanceDisplay(Vertical, can_focus=False):
    """
    Displays account balances in a styled table.

    Features:
    - Border title "BALANCES"
    - Zebra stripes for readability
    - Non-focusable to prevent stealing input focus

    Columns: Currency, Total, Available, Locked, %Used
    """

    DEFAULT_CSS = """
    BalanceDisplay {
        width: 1fr;
        height: 100%;
        border: round $primary-darken-1;
        border-title-color: $text;
        border-title-style: bold;
        background: $surface;
        margin: 0 1 0 0;
    }

    BalanceDisplay > DataTable {
        height: 1fr;
        background: $surface;
    }
    """

    BORDER_TITLE = "BALANCES"

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        table = DataTable(id="balance-table")
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.can_focus = False
        yield table

    def on_mount(self) -> None:
        """Initialize the data table."""
        table = self.query_one(DataTable)
        table.add_columns("Currency", "Total", "Available", "Locked", "%Used")

    def update_balances(self, balances: dict[str, Balance]) -> None:
        """
        Update the balance display with new data.

        Args:
            balances: Dictionary mapping currency to Balance objects.
        """
        table = self.query_one(DataTable)
        table.clear()

        for currency, balance in sorted(balances.items()):
            used_pct = (
                (balance.locked / balance.total * 100)
                if balance.total > 0
                else Decimal("0")
            )

            # Color code usage percentage
            pct_str = f"{used_pct:.1f}%"
            if used_pct > 80:
                pct_str = f"[red]{pct_str}[/red]"
            elif used_pct > 50:
                pct_str = f"[yellow]{pct_str}[/yellow]"

            table.add_row(
                currency,
                f"{balance.total:,.2f}",
                f"{balance.available:,.2f}",
                f"{balance.locked:,.2f}",
                pct_str,
            )

    def set_demo_data(self) -> None:
        """Set demo data for standalone mode."""
        table = self.query_one(DataTable)
        table.clear()
        table.add_row("USDT", "10,000.00", "8,500.00", "1,500.00", "[green]15.0%[/green]")
        table.add_row("BTC", "0.50000", "0.50000", "0.00000", "[green]0.0%[/green]")
        table.add_row("ETH", "5.00000", "3.00000", "2.00000", "[yellow]40.0%[/yellow]")

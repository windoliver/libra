"""
Position display widget showing open positions with P&L.

Professional styling with border title, zebra stripes, and color-coded P&L.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable


if TYPE_CHECKING:
    from libra.gateways.protocol import Position


class PositionDisplay(Vertical, can_focus=False):
    """
    Displays open positions in a styled table.

    Features:
    - Border title "POSITIONS"
    - Zebra stripes for readability
    - Color-coded: green for LONG/profit, red for SHORT/loss
    - Non-focusable to prevent stealing input focus

    Columns: Symbol, Side, Size, Entry, Current, P&L, P&L%
    """

    DEFAULT_CSS = """
    PositionDisplay {
        width: 2fr;
        height: 100%;
        border: round $primary-darken-1;
        border-title-color: $text;
        border-title-style: bold;
        background: $surface;
    }

    PositionDisplay > DataTable {
        height: 1fr;
        background: $surface;
    }
    """

    BORDER_TITLE = "POSITIONS"

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        table = DataTable(id="position-table")
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.can_focus = False
        yield table

    def on_mount(self) -> None:
        """Initialize the data table."""
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Side", "Size", "Entry", "Current", "P&L", "P&L%")

    def update_positions(self, positions: list[Position]) -> None:
        """
        Update the position display with new data.

        Args:
            positions: List of Position objects.
        """
        table = self.query_one(DataTable)
        table.clear()

        for pos in positions:
            pnl = pos.unrealized_pnl
            pnl_pct = pos.pnl_percent

            # Color code P&L
            if pnl >= 0:
                pnl_str = f"[green]+{pnl:,.2f}[/green]"
                pnl_pct_str = f"[green]+{pnl_pct:.2f}%[/green]"
            else:
                pnl_str = f"[red]{pnl:,.2f}[/red]"
                pnl_pct_str = f"[red]{pnl_pct:.2f}%[/red]"

            # Color code side
            side = pos.side.value
            if side == "LONG":
                side_str = "[green]LONG[/green]"
            elif side == "SHORT":
                side_str = "[red]SHORT[/red]"
            else:
                side_str = side

            table.add_row(
                pos.symbol,
                side_str,
                f"{pos.amount:,.4f}",
                f"{pos.entry_price:,.2f}",
                f"{pos.current_price:,.2f}",
                pnl_str,
                pnl_pct_str,
            )

    def set_demo_data(self) -> None:
        """Set demo data for standalone mode."""
        table = self.query_one(DataTable)
        table.clear()
        table.add_row(
            "BTC/USDT",
            "[green]LONG[/green]",
            "0.1000",
            "50,000.00",
            "51,250.00",
            "[green]+125.00[/green]",
            "[green]+2.50%[/green]",
        )
        table.add_row(
            "ETH/USDT",
            "[red]SHORT[/red]",
            "2.0000",
            "3,000.00",
            "3,045.00",
            "[red]-90.00[/red]",
            "[red]-1.50%[/red]",
        )
        table.add_row(
            "SOL/USDT",
            "[green]LONG[/green]",
            "10.000",
            "135.00",
            "142.50",
            "[green]+75.00[/green]",
            "[green]+5.56%[/green]",
        )

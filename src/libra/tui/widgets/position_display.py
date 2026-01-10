"""
Position display widget showing open positions with P&L.

Professional styling with border title, zebra stripes, and color-coded P&L.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    - PERFORMANCE: Differential updates instead of clear+rebuild

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
        # PERFORMANCE: Cache table reference and track row keys
        self._cached_table: DataTable | None = None
        self._row_keys: dict[str, Any] = {}  # symbol -> row_key

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        table = DataTable(id="position-table")
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.can_focus = False
        yield table

    def on_mount(self) -> None:
        """Initialize the data table and cache reference."""
        self._cached_table = self.query_one(DataTable)
        self._cached_table.add_columns("Symbol", "Side", "Size", "Entry", "Current", "P&L", "P&L%")

    def update_positions(self, positions: list[Position]) -> None:
        """
        Update the position display with new data using differential updates.

        Args:
            positions: List of Position objects.
        """
        table = self._cached_table
        if not table:
            return

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

            # PERFORMANCE: Use differential update instead of clear+rebuild
            self._update_position_row(
                pos.symbol,
                side_str,
                f"{pos.amount:,.4f}",
                f"{pos.entry_price:,.2f}",
                f"{pos.current_price:,.2f}",
                pnl_str,
                pnl_pct_str,
            )

    def _update_position_row(
        self,
        symbol: str,
        side: str,
        size: str,
        entry: str,
        current: str,
        pnl: str,
        pnl_pct: str,
    ) -> None:
        """
        PERFORMANCE: Update a single position row differentially.

        If row exists, update cells. If not, add new row.
        """
        table = self._cached_table
        if not table:
            return

        if symbol in self._row_keys:
            # Update existing row cells
            row_key = self._row_keys[symbol]
            try:
                table.update_cell(row_key, "Side", side)
                table.update_cell(row_key, "Size", size)
                table.update_cell(row_key, "Entry", entry)
                table.update_cell(row_key, "Current", current)
                table.update_cell(row_key, "P&L", pnl)
                table.update_cell(row_key, "P&L%", pnl_pct)
            except Exception:
                # Row was removed from table, clear stale tracking
                del self._row_keys[symbol]
                self._add_position_row(symbol, side, size, entry, current, pnl, pnl_pct)
        else:
            self._add_position_row(symbol, side, size, entry, current, pnl, pnl_pct)

    def _add_position_row(
        self,
        symbol: str,
        side: str,
        size: str,
        entry: str,
        current: str,
        pnl: str,
        pnl_pct: str,
    ) -> None:
        """Add a new position row, handling duplicates gracefully."""
        table = self._cached_table
        if not table:
            return
        try:
            row_key = table.add_row(symbol, side, size, entry, current, pnl, pnl_pct, key=symbol)
            self._row_keys[symbol] = row_key
        except Exception:
            # Key already exists - sync our tracking and try update
            for rk in table.rows:
                if rk.value == symbol:
                    self._row_keys[symbol] = rk
                    try:
                        table.update_cell(rk, "Side", side)
                        table.update_cell(rk, "Size", size)
                        table.update_cell(rk, "Entry", entry)
                        table.update_cell(rk, "Current", current)
                        table.update_cell(rk, "P&L", pnl)
                        table.update_cell(rk, "P&L%", pnl_pct)
                    except Exception:
                        pass
                    break

    def set_demo_data(self) -> None:
        """Set demo data for standalone mode."""
        table = self._cached_table
        if not table:
            return
        table.clear()
        self._row_keys.clear()
        self._update_position_row(
            "BTC/USDT",
            "[green]LONG[/green]",
            "0.1000",
            "50,000.00",
            "51,250.00",
            "[green]+125.00[/green]",
            "[green]+2.50%[/green]",
        )
        self._update_position_row(
            "ETH/USDT",
            "[red]SHORT[/red]",
            "2.0000",
            "3,000.00",
            "3,045.00",
            "[red]-90.00[/red]",
            "[red]-1.50%[/red]",
        )
        self._update_position_row(
            "SOL/USDT",
            "[green]LONG[/green]",
            "10.000",
            "135.00",
            "142.50",
            "[green]+75.00[/green]",
            "[green]+5.56%[/green]",
        )

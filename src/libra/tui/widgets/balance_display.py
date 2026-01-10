"""
Balance display widget showing account balances.

Professional styling with border title and zebra stripes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

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
    - PERFORMANCE: Differential updates instead of clear+rebuild

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

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        # PERFORMANCE: Cache table reference and track row keys
        self._cached_table: DataTable | None = None
        self._row_keys: dict[str, Any] = {}  # currency -> row_key

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        table = DataTable(id="balance-table")
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.can_focus = False
        yield table

    def on_mount(self) -> None:
        """Initialize the data table and cache reference."""
        self._cached_table = self.query_one(DataTable)
        self._cached_table.add_columns("Currency", "Total", "Available", "Locked", "%Used")

    def update_balances(self, balances: dict[str, Balance]) -> None:
        """
        Update the balance display with new data using differential updates.

        Args:
            balances: Dictionary mapping currency to Balance objects.
        """
        table = self._cached_table
        if not table:
            return

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

            # PERFORMANCE: Use differential update instead of clear+rebuild
            self.update_balance_row(
                currency,
                f"{balance.total:,.2f}",
                f"{balance.available:,.2f}",
                f"{balance.locked:,.2f}",
                pct_str,
            )

    def update_balance_row(
        self,
        currency: str,
        total: str,
        available: str,
        locked: str,
        pct_used: str,
    ) -> None:
        """
        PERFORMANCE: Update a single balance row differentially.

        If row exists, update cells. If not, add new row.
        This avoids expensive clear+rebuild pattern.
        """
        table = self._cached_table
        if not table:
            return

        if currency in self._row_keys:
            # Update existing row cells
            row_key = self._row_keys[currency]
            try:
                table.update_cell(row_key, "Total", total)
                table.update_cell(row_key, "Available", available)
                table.update_cell(row_key, "Locked", locked)
                table.update_cell(row_key, "%Used", pct_used)
            except Exception:
                # Row was removed from table, clear stale tracking
                del self._row_keys[currency]
                self._add_balance_row(currency, total, available, locked, pct_used)
        else:
            self._add_balance_row(currency, total, available, locked, pct_used)

    def _add_balance_row(
        self,
        currency: str,
        total: str,
        available: str,
        locked: str,
        pct_used: str,
    ) -> None:
        """Add a new balance row, handling duplicates gracefully."""
        table = self._cached_table
        if not table:
            return
        try:
            row_key = table.add_row(currency, total, available, locked, pct_used, key=currency)
            self._row_keys[currency] = row_key
        except Exception:
            # Key already exists - sync our tracking and try update
            # Find the row key by iterating (fallback)
            for rk in table.rows:
                if rk.value == currency:
                    self._row_keys[currency] = rk
                    try:
                        table.update_cell(rk, "Total", total)
                        table.update_cell(rk, "Available", available)
                        table.update_cell(rk, "Locked", locked)
                        table.update_cell(rk, "%Used", pct_used)
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
        self.update_balance_row("USDT", "10,000.00", "8,500.00", "1,500.00", "[green]15.0%[/green]")
        self.update_balance_row("BTC", "0.50000", "0.50000", "0.00000", "[green]0.0%[/green]")
        self.update_balance_row("ETH", "5.00000", "3.00000", "2.00000", "[yellow]40.0%[/yellow]")

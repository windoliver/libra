"""
Options Chain Widget.

Displays options chain data with Greeks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Select, Static


if TYPE_CHECKING:
    from libra.gateways.openbb.fetchers import OptionContract


def _format_price(value: Decimal | None) -> str:
    """Format price for display."""
    if value is None:
        return "--"
    return f"${float(value):.2f}"


def _format_greek(value: Decimal | None) -> str:
    """Format Greek value for display."""
    if value is None:
        return "--"
    return f"{float(value):.4f}"


def _format_iv(value: Decimal | None) -> str:
    """Format implied volatility for display."""
    if value is None:
        return "--"
    return f"{float(value)*100:.1f}%"


def _format_volume(value: int | None) -> str:
    """Format volume for display."""
    if value is None:
        return "--"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return str(value)


@dataclass
class OptionsChainData:
    """Container for options chain data."""

    symbol: str
    contracts: list[OptionContract] = field(default_factory=list)
    expirations: list[date] = field(default_factory=list)
    selected_expiration: date | None = None

    @property
    def calls(self) -> list[OptionContract]:
        """Get call options."""
        return [c for c in self.contracts if c.option_type == "call"]

    @property
    def puts(self) -> list[OptionContract]:
        """Get put options."""
        return [c for c in self.contracts if c.option_type == "put"]


class OptionsChainWidget(Container):
    """
    Options chain widget displaying calls and puts with Greeks.

    Features:
    - Expiration date selector
    - Calls/Puts split view
    - Greeks display (Delta, Gamma, Theta, Vega)
    - ITM/OTM highlighting
    """

    DEFAULT_CSS = """
    OptionsChainWidget {
        height: 100%;
    }

    OptionsChainWidget .chain-header {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }

    OptionsChainWidget .chain-title {
        width: 1fr;
        text-style: bold;
    }

    OptionsChainWidget .expiration-select {
        width: 20;
        margin-right: 1;
    }

    OptionsChainWidget .type-select {
        width: 12;
    }

    OptionsChainWidget .chain-tables {
        height: 1fr;
        layout: horizontal;
    }

    OptionsChainWidget .calls-container {
        width: 1fr;
        border-right: solid $primary-background;
    }

    OptionsChainWidget .puts-container {
        width: 1fr;
    }

    OptionsChainWidget .table-header {
        height: 2;
        padding: 0 1;
        background: $surface;
        text-style: bold;
        content-align: center middle;
    }

    OptionsChainWidget .calls-header {
        color: $success;
    }

    OptionsChainWidget .puts-header {
        color: $error;
    }

    OptionsChainWidget DataTable {
        height: 1fr;
    }

    OptionsChainWidget .itm {
        background: $primary-background-lighten-2;
    }

    OptionsChainWidget .no-data {
        height: 100%;
        align: center middle;
        color: $text-muted;
    }
    """

    # Reactive data
    data: reactive[OptionsChainData | None] = reactive(None, init=False)
    selected_expiration: reactive[date | None] = reactive(None, init=False)
    option_type_filter: reactive[str] = reactive("all", init=False)

    class DataRequested(Message):
        """Message requesting options data load."""

        def __init__(
            self, symbol: str, expiration: date | None, provider: str
        ) -> None:
            self.symbol = symbol
            self.expiration = expiration
            self.provider = provider
            super().__init__()

    def __init__(
        self,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with Horizontal(classes="chain-header"):
            yield Label(
                "[bold]Options Chain[/bold]",
                id="chain-title",
                classes="chain-title",
            )
            yield Select(
                options=[("Select Expiration", None)],
                value=None,
                id="expiration-select",
                classes="expiration-select",
            )
            yield Select(
                options=[
                    ("All", "all"),
                    ("Calls", "call"),
                    ("Puts", "put"),
                ],
                value="all",
                id="type-select",
                classes="type-select",
            )

        with Horizontal(classes="chain-tables"):
            with Container(classes="calls-container"):
                yield Static("CALLS", classes="table-header calls-header")
                yield DataTable(id="calls-table")

            with Container(classes="puts-container"):
                yield Static("PUTS", classes="table-header puts-header")
                yield DataTable(id="puts-table")

    def on_mount(self) -> None:
        """Set up tables after mount."""
        self._setup_tables()

    def _setup_tables(self) -> None:
        """Set up data tables."""
        columns = [
            "Strike",
            "Bid",
            "Ask",
            "Last",
            "IV",
            "Delta",
            "Gamma",
            "Theta",
            "Vol",
            "OI",
        ]

        try:
            calls_table = self.query_one("#calls-table", DataTable)
            calls_table.add_columns(*columns)
            calls_table.zebra_stripes = True
        except Exception:
            pass

        try:
            puts_table = self.query_one("#puts-table", DataTable)
            puts_table.add_columns(*columns)
            puts_table.zebra_stripes = True
        except Exception:
            pass

    @on(Select.Changed, "#expiration-select")
    def _on_expiration_changed(self, event: Select.Changed) -> None:
        """Handle expiration selection change."""
        if event.value and self.data:
            self.selected_expiration = event.value
            self.post_message(
                self.DataRequested(
                    self.data.symbol,
                    self.selected_expiration,
                    "cboe",
                )
            )

    @on(Select.Changed, "#type-select")
    def _on_type_changed(self, event: Select.Changed) -> None:
        """Handle option type filter change."""
        if event.value:
            self.option_type_filter = str(event.value)
            self._update_tables()

    def watch_data(self, data: OptionsChainData | None) -> None:
        """Update display when data changes."""
        if data:
            self._update_expirations(data)
            self._update_tables()
            self._update_header(data)

    def _update_header(self, data: OptionsChainData) -> None:
        """Update header with symbol."""
        try:
            header = self.query_one("#chain-title", Label)
            header.update(f"[bold]Options Chain - {data.symbol}[/bold]")
        except Exception:
            pass

    def _update_expirations(self, data: OptionsChainData) -> None:
        """Update expiration dropdown."""
        try:
            select = self.query_one("#expiration-select", Select)

            if data.expirations:
                options = [
                    (exp.strftime("%Y-%m-%d"), exp) for exp in data.expirations
                ]
                select.set_options(options)
                if data.selected_expiration:
                    select.value = data.selected_expiration
                elif data.expirations:
                    select.value = data.expirations[0]
            else:
                select.set_options([("No expirations", None)])
        except Exception:
            pass

    def _update_tables(self) -> None:
        """Update calls and puts tables."""
        if not self.data:
            return

        # Update calls table
        try:
            calls_table = self.query_one("#calls-table", DataTable)
            calls_table.clear()

            if self.option_type_filter in ("all", "call"):
                calls = sorted(self.data.calls, key=lambda c: float(c.strike))
                for contract in calls:
                    calls_table.add_row(
                        _format_price(contract.strike),
                        _format_price(contract.bid),
                        _format_price(contract.ask),
                        _format_price(contract.last),
                        _format_iv(contract.implied_volatility),
                        _format_greek(contract.delta),
                        _format_greek(contract.gamma),
                        _format_greek(contract.theta),
                        _format_volume(contract.volume),
                        _format_volume(contract.open_interest),
                    )
        except Exception:
            pass

        # Update puts table
        try:
            puts_table = self.query_one("#puts-table", DataTable)
            puts_table.clear()

            if self.option_type_filter in ("all", "put"):
                puts = sorted(self.data.puts, key=lambda c: float(c.strike))
                for contract in puts:
                    puts_table.add_row(
                        _format_price(contract.strike),
                        _format_price(contract.bid),
                        _format_price(contract.ask),
                        _format_price(contract.last),
                        _format_iv(contract.implied_volatility),
                        _format_greek(contract.delta),
                        _format_greek(contract.gamma),
                        _format_greek(contract.theta),
                        _format_volume(contract.volume),
                        _format_volume(contract.open_interest),
                    )
        except Exception:
            pass

    def set_data(self, data: OptionsChainData) -> None:
        """Set options chain data."""
        self.data = data

    def clear(self) -> None:
        """Clear the display."""
        self.data = None
        try:
            calls_table = self.query_one("#calls-table", DataTable)
            calls_table.clear()
        except Exception:
            pass
        try:
            puts_table = self.query_one("#puts-table", DataTable)
            puts_table.clear()
        except Exception:
            pass

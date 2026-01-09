"""
Trade History Table Widget.

Displays trade history with sorting and filtering using DataTable.

Features:
- Sortable columns (click header to sort)
- Filtering by side (LONG/SHORT), symbol, P&L
- Color-coded P&L display
- High-performance Line API for 1000s of trades

Design inspired by:
- Interactive Brokers trade history
- QuantConnect trade log
- TradingView strategy tester
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Collapsible, DataTable, Input, Select, Static

if TYPE_CHECKING:
    from textual.widgets.data_table import RowKey


# =============================================================================
# Data Models
# =============================================================================


class TradeSide(Enum):
    """Trade side enumeration."""

    LONG = auto()
    SHORT = auto()


@dataclass
class TradeRecord:
    """Single trade record for display."""

    trade_id: str
    symbol: str
    side: TradeSide
    entry_time: datetime
    entry_price: Decimal
    exit_time: datetime | None = None
    exit_price: Decimal | None = None
    quantity: Decimal = Decimal("1")
    pnl: Decimal = Decimal("0")
    pnl_pct: float = 0.0
    commission: Decimal = Decimal("0")
    notes: str = ""

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None

    @property
    def duration(self) -> str:
        """Get trade duration as string."""
        if not self.exit_time:
            return "Open"
        delta = self.exit_time - self.entry_time
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes = remainder // 60
        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"


class SortColumn(Enum):
    """Sortable columns."""

    TIME = "time"
    SYMBOL = "symbol"
    SIDE = "side"
    ENTRY = "entry"
    EXIT = "exit"
    PNL = "pnl"
    PNL_PCT = "pnl_pct"
    DURATION = "duration"


class SortDirection(Enum):
    """Sort direction."""

    ASC = "asc"
    DESC = "desc"


class FilterSide(Enum):
    """Filter options for side."""

    ALL = "All"
    LONG = "Long Only"
    SHORT = "Short Only"
    WINNERS = "Winners"
    LOSERS = "Losers"


# =============================================================================
# Trade History Table Widget
# =============================================================================


class TradeHistoryTable(Container):
    """
    Trade history table with sorting and filtering.

    Features:
    - Click column headers to sort
    - Filter by side, symbol, P&L
    - Color-coded P&L values
    - Export capability
    """

    DEFAULT_CSS = """
    TradeHistoryTable {
        height: 100%;
        width: 100%;
        layout: vertical;
    }

    TradeHistoryTable .table-header {
        height: 3;
        padding: 0 1;
        background: $surface-darken-1;
    }

    TradeHistoryTable .table-title {
        width: auto;
        padding-right: 2;
        text-style: bold;
    }

    TradeHistoryTable .filter-controls {
        width: 1fr;
    }

    TradeHistoryTable .filter-select {
        width: 15;
        margin-right: 1;
    }

    TradeHistoryTable .filter-input {
        width: 20;
        margin-right: 1;
    }

    TradeHistoryTable .table-container {
        height: 1fr;
    }

    TradeHistoryTable DataTable {
        height: 100%;
    }

    TradeHistoryTable .table-footer {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }

    TradeHistoryTable .positive {
        color: $success;
    }

    TradeHistoryTable .negative {
        color: $error;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("e", "export", "Export"),
        Binding("c", "clear_filters", "Clear Filters"),
    ]

    # Reactive state
    sort_column: reactive[SortColumn] = reactive(SortColumn.TIME)
    sort_direction: reactive[SortDirection] = reactive(SortDirection.DESC)
    filter_side: reactive[FilterSide] = reactive(FilterSide.ALL)
    filter_symbol: reactive[str] = reactive("")

    class TradeSelected(Message):
        """Message sent when a trade is selected."""

        def __init__(self, trade: TradeRecord) -> None:
            super().__init__()
            self.trade = trade

    def __init__(
        self,
        trades: list[TradeRecord] | None = None,
        title: str = "TRADE HISTORY",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._trades = trades or []
        self._filtered_trades: list[TradeRecord] = []
        self._title = title
        self._row_to_trade: dict[RowKey, TradeRecord] = {}

    def compose(self) -> ComposeResult:
        # Header with title and filters
        with Horizontal(classes="table-header"):
            yield Static(self._title, classes="table-title")
            with Horizontal(classes="filter-controls"):
                yield Select(
                    [(f.value, f) for f in FilterSide],
                    value=FilterSide.ALL,
                    id="filter-side",
                    classes="filter-select",
                )
                yield Input(
                    placeholder="Filter symbol...",
                    id="filter-symbol",
                    classes="filter-input",
                )
                yield Button("Clear", id="clear-filters", variant="default")

        # Table
        with Container(classes="table-container"):
            yield DataTable(id="trades-table", cursor_type="row", zebra_stripes=True)

        # Footer with stats
        yield Static(self._get_footer_text(), id="table-footer", classes="table-footer")

    def on_mount(self) -> None:
        """Set up the table on mount."""
        table = self.query_one("#trades-table", DataTable)

        # Add columns with sort indicators
        table.add_column("Time ▼", key="time")
        table.add_column("Symbol", key="symbol")
        table.add_column("Side", key="side")
        table.add_column("Entry", key="entry")
        table.add_column("Exit", key="exit")
        table.add_column("P&L", key="pnl")
        table.add_column("P&L %", key="pnl_pct")
        table.add_column("Duration", key="duration")

        # Populate table
        self._apply_filters()
        self._populate_table()

    def _get_footer_text(self) -> str:
        """Generate footer statistics text."""
        total = len(self._trades)
        filtered = len(self._filtered_trades)
        if total == 0:
            return "No trades"

        winners = sum(1 for t in self._filtered_trades if t.pnl > 0)
        losers = sum(1 for t in self._filtered_trades if t.pnl < 0)
        total_pnl = sum(t.pnl for t in self._filtered_trades)

        pnl_class = "positive" if total_pnl >= 0 else "negative"
        pnl_sign = "+" if total_pnl >= 0 else ""

        return (
            f"Showing {filtered}/{total} trades  |  "
            f"[green]W: {winners}[/green]  [red]L: {losers}[/red]  |  "
            f"Total P&L: [{pnl_class}]{pnl_sign}${total_pnl:,.2f}[/{pnl_class}]"
        )

    def _apply_filters(self) -> None:
        """Apply current filters to trades."""
        self._filtered_trades = self._trades.copy()

        # Filter by side
        if self.filter_side == FilterSide.LONG:
            self._filtered_trades = [t for t in self._filtered_trades if t.side == TradeSide.LONG]
        elif self.filter_side == FilterSide.SHORT:
            self._filtered_trades = [t for t in self._filtered_trades if t.side == TradeSide.SHORT]
        elif self.filter_side == FilterSide.WINNERS:
            self._filtered_trades = [t for t in self._filtered_trades if t.pnl > 0]
        elif self.filter_side == FilterSide.LOSERS:
            self._filtered_trades = [t for t in self._filtered_trades if t.pnl < 0]

        # Filter by symbol
        if self.filter_symbol:
            symbol_filter = self.filter_symbol.upper()
            self._filtered_trades = [
                t for t in self._filtered_trades if symbol_filter in t.symbol.upper()
            ]

        # Apply sorting
        self._sort_trades()

    def _sort_trades(self) -> None:
        """Sort filtered trades by current column and direction."""
        reverse = self.sort_direction == SortDirection.DESC

        key_funcs = {
            SortColumn.TIME: lambda t: t.entry_time,
            SortColumn.SYMBOL: lambda t: t.symbol,
            SortColumn.SIDE: lambda t: t.side.name,
            SortColumn.ENTRY: lambda t: t.entry_price,
            SortColumn.EXIT: lambda t: t.exit_price or Decimal("0"),
            SortColumn.PNL: lambda t: t.pnl,
            SortColumn.PNL_PCT: lambda t: t.pnl_pct,
            SortColumn.DURATION: lambda t: (
                (t.exit_time - t.entry_time).total_seconds() if t.exit_time else 0
            ),
        }

        key_func = key_funcs.get(self.sort_column, key_funcs[SortColumn.TIME])
        self._filtered_trades.sort(key=key_func, reverse=reverse)

    def _populate_table(self) -> None:
        """Populate table with filtered and sorted trades."""
        table = self.query_one("#trades-table", DataTable)
        table.clear()
        self._row_to_trade.clear()

        for trade in self._filtered_trades:
            # Format values
            time_str = trade.entry_time.strftime("%Y-%m-%d %H:%M")
            side_str = f"[green]LONG[/green]" if trade.side == TradeSide.LONG else "[red]SHORT[/red]"
            entry_str = f"${trade.entry_price:,.2f}"
            exit_str = f"${trade.exit_price:,.2f}" if trade.exit_price else "-"

            # P&L with color
            pnl_sign = "+" if trade.pnl >= 0 else ""
            pnl_class = "green" if trade.pnl >= 0 else "red"
            pnl_str = f"[{pnl_class}]{pnl_sign}${trade.pnl:,.2f}[/{pnl_class}]"

            pnl_pct_sign = "+" if trade.pnl_pct >= 0 else ""
            pnl_pct_str = f"[{pnl_class}]{pnl_pct_sign}{trade.pnl_pct:.2f}%[/{pnl_class}]"

            row_key = table.add_row(
                time_str,
                trade.symbol,
                side_str,
                entry_str,
                exit_str,
                pnl_str,
                pnl_pct_str,
                trade.duration,
                key=trade.trade_id,
            )
            self._row_to_trade[row_key] = trade

        # Update footer
        try:
            footer = self.query_one("#table-footer", Static)
            footer.update(self._get_footer_text())
        except Exception:
            pass

    def _update_column_headers(self) -> None:
        """Update column headers with sort indicators."""
        table = self.query_one("#trades-table", DataTable)

        # Map columns to sort columns
        column_map = {
            "time": SortColumn.TIME,
            "symbol": SortColumn.SYMBOL,
            "side": SortColumn.SIDE,
            "entry": SortColumn.ENTRY,
            "exit": SortColumn.EXIT,
            "pnl": SortColumn.PNL,
            "pnl_pct": SortColumn.PNL_PCT,
            "duration": SortColumn.DURATION,
        }

        labels = {
            "time": "Time",
            "symbol": "Symbol",
            "side": "Side",
            "entry": "Entry",
            "exit": "Exit",
            "pnl": "P&L",
            "pnl_pct": "P&L %",
            "duration": "Duration",
        }

        for col_key, sort_col in column_map.items():
            label = labels[col_key]
            if sort_col == self.sort_column:
                indicator = "▼" if self.sort_direction == SortDirection.DESC else "▲"
                label = f"{label} {indicator}"

            # Update column label using the columns property
            for i, column in enumerate(table.columns.values()):
                if column.key and column.key.value == col_key:
                    # DataTable doesn't have direct label update, so we track state
                    break

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Handle column header click for sorting."""
        column_key = str(event.column_key)

        column_map = {
            "time": SortColumn.TIME,
            "symbol": SortColumn.SYMBOL,
            "side": SortColumn.SIDE,
            "entry": SortColumn.ENTRY,
            "exit": SortColumn.EXIT,
            "pnl": SortColumn.PNL,
            "pnl_pct": SortColumn.PNL_PCT,
            "duration": SortColumn.DURATION,
        }

        if column_key in column_map:
            new_sort = column_map[column_key]
            if self.sort_column == new_sort:
                # Toggle direction
                self.sort_direction = (
                    SortDirection.ASC
                    if self.sort_direction == SortDirection.DESC
                    else SortDirection.DESC
                )
            else:
                # New column, default to descending
                self.sort_column = new_sort
                self.sort_direction = SortDirection.DESC

            self._apply_filters()
            self._populate_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key in self._row_to_trade:
            trade = self._row_to_trade[event.row_key]
            self.post_message(self.TradeSelected(trade))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter select change."""
        if event.select.id == "filter-side" and isinstance(event.value, FilterSide):
            self.filter_side = event.value
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
        if event.button.id == "clear-filters":
            self.action_clear_filters()

    def action_refresh(self) -> None:
        """Refresh the table."""
        self._apply_filters()
        self._populate_table()

    def action_clear_filters(self) -> None:
        """Clear all filters."""
        self.filter_side = FilterSide.ALL
        self.filter_symbol = ""

        try:
            side_select = self.query_one("#filter-side", Select)
            side_select.value = FilterSide.ALL
        except Exception:
            pass

        try:
            symbol_input = self.query_one("#filter-symbol", Input)
            symbol_input.value = ""
        except Exception:
            pass

        self._apply_filters()
        self._populate_table()

    def action_export(self) -> None:
        """Export trades (placeholder for future implementation)."""
        self.notify("Export feature coming soon!", title="Export")

    def update_trades(self, trades: list[TradeRecord]) -> None:
        """Update table with new trades."""
        self._trades = trades
        self._apply_filters()
        self._populate_table()


# =============================================================================
# Trade Detail Panel
# =============================================================================


class TradeDetailPanel(Container):
    """
    Panel showing detailed information for a selected trade.
    """

    DEFAULT_CSS = """
    TradeDetailPanel {
        height: auto;
        min-height: 8;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    TradeDetailPanel .detail-title {
        text-style: bold;
        margin-bottom: 1;
    }

    TradeDetailPanel .detail-row {
        height: 1;
    }

    TradeDetailPanel .detail-label {
        width: 15;
        color: $text-muted;
    }

    TradeDetailPanel .detail-value {
        width: 1fr;
    }

    TradeDetailPanel .positive {
        color: $success;
    }

    TradeDetailPanel .negative {
        color: $error;
    }
    """

    def __init__(
        self,
        trade: TradeRecord | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._trade = trade

    def compose(self) -> ComposeResult:
        if not self._trade:
            yield Static("[dim]Select a trade to view details[/dim]")
            return

        t = self._trade

        yield Static(f"Trade Details: {t.trade_id}", classes="detail-title")

        # Trade info
        side_str = "[green]LONG[/green]" if t.side == TradeSide.LONG else "[red]SHORT[/red]"
        with Horizontal(classes="detail-row"):
            yield Static("Symbol:", classes="detail-label")
            yield Static(f"{t.symbol} ({side_str})", classes="detail-value")

        with Horizontal(classes="detail-row"):
            yield Static("Entry:", classes="detail-label")
            yield Static(
                f"${t.entry_price:,.2f} @ {t.entry_time.strftime('%Y-%m-%d %H:%M')}",
                classes="detail-value",
            )

        if t.exit_time and t.exit_price:
            with Horizontal(classes="detail-row"):
                yield Static("Exit:", classes="detail-label")
                yield Static(
                    f"${t.exit_price:,.2f} @ {t.exit_time.strftime('%Y-%m-%d %H:%M')}",
                    classes="detail-value",
                )

        with Horizontal(classes="detail-row"):
            yield Static("Quantity:", classes="detail-label")
            yield Static(f"{t.quantity:,.4f}", classes="detail-value")

        with Horizontal(classes="detail-row"):
            yield Static("Duration:", classes="detail-label")
            yield Static(t.duration, classes="detail-value")

        # P&L
        pnl_class = "positive" if t.pnl >= 0 else "negative"
        pnl_sign = "+" if t.pnl >= 0 else ""
        with Horizontal(classes="detail-row"):
            yield Static("P&L:", classes="detail-label")
            yield Static(
                f"[{pnl_class}]{pnl_sign}${t.pnl:,.2f} ({pnl_sign}{t.pnl_pct:.2f}%)[/{pnl_class}]",
                classes="detail-value",
            )

        if t.commission > 0:
            with Horizontal(classes="detail-row"):
                yield Static("Commission:", classes="detail-label")
                yield Static(f"${t.commission:,.2f}", classes="detail-value")

        if t.notes:
            with Horizontal(classes="detail-row"):
                yield Static("Notes:", classes="detail-label")
                yield Static(t.notes, classes="detail-value")

    def update_trade(self, trade: TradeRecord | None) -> None:
        """Update displayed trade."""
        self._trade = trade
        self.remove_children()
        for widget in self.compose():
            self.mount(widget)


# =============================================================================
# Collapsible Trade Details
# =============================================================================


class CollapsibleTradeDetails(Container):
    """
    Collapsible panel for trade details.

    Click to expand and see full trade information.
    """

    DEFAULT_CSS = """
    CollapsibleTradeDetails {
        height: auto;
        width: 100%;
        margin-top: 1;
    }

    CollapsibleTradeDetails Collapsible {
        background: $surface;
        border: round $primary-darken-2;
    }

    CollapsibleTradeDetails .trade-summary {
        height: auto;
        padding: 0 1;
    }

    CollapsibleTradeDetails .trade-detail-grid {
        height: auto;
        padding: 1;
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
    }

    CollapsibleTradeDetails .detail-item {
        height: 2;
        padding: 0 1;
        background: $surface-darken-1;
        border: round $primary-darken-3;
    }

    CollapsibleTradeDetails .detail-label {
        color: $text-muted;
    }

    CollapsibleTradeDetails .detail-value {
        text-style: bold;
    }

    CollapsibleTradeDetails .positive {
        color: $success;
    }

    CollapsibleTradeDetails .negative {
        color: $error;
    }

    CollapsibleTradeDetails .no-trade {
        height: 3;
        content-align: center middle;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        trade: TradeRecord | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._trade = trade

    def compose(self) -> ComposeResult:
        if not self._trade:
            yield Static("[dim]Select a trade to view details[/dim]", classes="no-trade")
            return

        t = self._trade
        pnl_class = "positive" if t.pnl >= 0 else "negative"
        pnl_sign = "+" if t.pnl >= 0 else ""
        side_color = "green" if t.side == TradeSide.LONG else "red"

        # Collapsible title shows summary
        title = (
            f"📊 {t.trade_id} | {t.symbol} | "
            f"[{side_color}]{t.side.name}[/{side_color}] | "
            f"[{pnl_class}]{pnl_sign}${t.pnl:,.2f}[/{pnl_class}]"
        )

        with Collapsible(title=title, collapsed=False):
            with Vertical(classes="trade-detail-grid"):
                # Row 1: Entry info
                with Vertical(classes="detail-item"):
                    yield Static("Entry Price", classes="detail-label")
                    yield Static(f"${t.entry_price:,.2f}", classes="detail-value")

                with Vertical(classes="detail-item"):
                    yield Static("Entry Time", classes="detail-label")
                    yield Static(t.entry_time.strftime("%Y-%m-%d %H:%M"), classes="detail-value")

                # Row 2: Exit info
                with Vertical(classes="detail-item"):
                    yield Static("Exit Price", classes="detail-label")
                    exit_str = f"${t.exit_price:,.2f}" if t.exit_price else "Open"
                    yield Static(exit_str, classes="detail-value")

                with Vertical(classes="detail-item"):
                    yield Static("Exit Time", classes="detail-label")
                    exit_time_str = t.exit_time.strftime("%Y-%m-%d %H:%M") if t.exit_time else "Open"
                    yield Static(exit_time_str, classes="detail-value")

                # Row 3: P&L and Duration
                with Vertical(classes="detail-item"):
                    yield Static("P&L", classes="detail-label")
                    yield Static(
                        f"[{pnl_class}]{pnl_sign}${t.pnl:,.2f} ({pnl_sign}{t.pnl_pct:.2f}%)[/{pnl_class}]",
                        classes="detail-value",
                    )

                with Vertical(classes="detail-item"):
                    yield Static("Duration", classes="detail-label")
                    yield Static(t.duration, classes="detail-value")

                # Row 4: Quantity and Commission
                with Vertical(classes="detail-item"):
                    yield Static("Quantity", classes="detail-label")
                    yield Static(f"{t.quantity:,.4f}", classes="detail-value")

                with Vertical(classes="detail-item"):
                    yield Static("Commission", classes="detail-label")
                    yield Static(f"${t.commission:,.2f}", classes="detail-value")

    def update_trade(self, trade: TradeRecord | None) -> None:
        """Update displayed trade with new data."""
        self._trade = trade
        self.remove_children()
        for widget in self.compose():
            self.mount(widget)


# =============================================================================
# Demo Data Generator
# =============================================================================


def create_demo_trades(count: int = 50) -> list[TradeRecord]:
    """Create demo trade records for testing."""
    import random
    from datetime import timedelta

    trades: list[TradeRecord] = []
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "AMD"]
    base_time = datetime.now() - timedelta(days=90)

    for i in range(count):
        symbol = random.choice(symbols)
        side = random.choice([TradeSide.LONG, TradeSide.SHORT])
        entry_time = base_time + timedelta(days=random.randint(0, 85), hours=random.randint(9, 16))
        duration_hours = random.randint(1, 72)
        exit_time = entry_time + timedelta(hours=duration_hours)

        entry_price = Decimal(str(round(random.uniform(100, 500), 2)))
        # Random P&L percentage (-10% to +15%)
        pnl_pct = random.gauss(0.5, 5)  # Slight positive bias
        if side == TradeSide.SHORT:
            exit_price = entry_price * Decimal(str(1 - pnl_pct / 100))
        else:
            exit_price = entry_price * Decimal(str(1 + pnl_pct / 100))

        quantity = Decimal(str(random.randint(10, 100)))
        pnl = (exit_price - entry_price) * quantity
        if side == TradeSide.SHORT:
            pnl = -pnl

        trades.append(
            TradeRecord(
                trade_id=f"TRD-{i + 1:04d}",
                symbol=symbol,
                side=side,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price.quantize(Decimal("0.01")),
                quantity=quantity,
                pnl=pnl.quantize(Decimal("0.01")),
                pnl_pct=pnl_pct,
                commission=Decimal(str(round(random.uniform(1, 10), 2))),
            )
        )

    # Sort by entry time descending
    trades.sort(key=lambda t: t.entry_time, reverse=True)
    return trades

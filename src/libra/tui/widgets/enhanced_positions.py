"""
Enhanced Positions Table Widget.

Advanced positions display with:
- Collapsible position details
- Risk metrics per position
- Close position actions
- Real-time P&L updates
- Sorting and filtering

Design inspired by:
- Bloomberg Terminal position views
- Interactive Brokers TWS
- Professional trading platforms

Layout:
    +-- POSITIONS ------------------------------------------------+
    | Symbol    Side   Size      Entry     Current    P&L    [X]  |
    +------------------------------------------------------------|
    | ▼ BTC/USDT LONG  0.5    $42,500    $43,200  +$350 (+1.6%)   |
    |   ├─ Unrealized: +$350.00                                   |
    |   ├─ Liquidation: $38,250                                   |
    |   ├─ Leverage: 3x                                           |
    |   └─ Duration: 2h 15m                                       |
    | ▶ ETH/USDT SHORT 2.0    $2,850     $2,820   +$60 (+1.1%)    |
    +------------------------------------------------------------|
    |                                     Total: +$410 (+1.4%)    |
    +-------------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Collapsible, DataTable, Static


if TYPE_CHECKING:
    pass


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PositionData:
    """Complete position data for display."""

    position_id: str
    symbol: str
    side: str  # LONG, SHORT
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")
    leverage: int = 1
    liquidation_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    opened_at: datetime = field(default_factory=datetime.now)
    strategy_id: str | None = None
    strategy_name: str | None = None

    @property
    def pnl_pct(self) -> float:
        """Calculate P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        notional = self.size * self.entry_price
        if notional == 0:
            return 0.0
        return float((self.unrealized_pnl / notional) * 100)

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value."""
        return self.size * self.current_price

    @property
    def duration(self) -> timedelta:
        """Calculate position duration."""
        return datetime.now() - self.opened_at

    @property
    def duration_str(self) -> str:
        """Format duration as string."""
        delta = self.duration
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h"
        return f"{hours}h {minutes}m"


# =============================================================================
# Position Row Widget
# =============================================================================


class PositionRow(Collapsible):
    """
    Expandable position row with details.

    Shows summary in collapsed state, full details when expanded.
    """

    DEFAULT_CSS = """
    PositionRow {
        height: auto;
        padding: 0 1;
        background: $surface;
        border-bottom: solid $primary-darken-3;
    }

    PositionRow .position-header {
        height: 2;
        layout: horizontal;
    }

    PositionRow .pos-symbol {
        width: 12;
        text-style: bold;
    }

    PositionRow .pos-side {
        width: 8;
    }

    PositionRow .pos-side.long {
        color: $success;
    }

    PositionRow .pos-side.short {
        color: $error;
    }

    PositionRow .pos-size {
        width: 10;
        text-align: right;
    }

    PositionRow .pos-entry {
        width: 12;
        text-align: right;
        color: $text-muted;
    }

    PositionRow .pos-current {
        width: 12;
        text-align: right;
    }

    PositionRow .pos-pnl {
        width: 18;
        text-align: right;
    }

    PositionRow .pos-pnl.positive {
        color: $success;
    }

    PositionRow .pos-pnl.negative {
        color: $error;
    }

    PositionRow .pos-close-btn {
        width: 5;
        min-width: 5;
        height: 1;
        margin-left: 1;
    }

    PositionRow .position-details {
        padding: 0 2;
        color: $text-muted;
    }

    PositionRow .detail-row {
        height: 1;
    }
    """

    class CloseRequested(Message):
        """Message sent when close button is clicked."""

        def __init__(self, position: PositionData) -> None:
            self.position = position
            super().__init__()

    class Selected(Message):
        """Message sent when position is selected."""

        def __init__(self, position: PositionData) -> None:
            self.position = position
            super().__init__()

    def __init__(
        self,
        position: PositionData,
        id: str | None = None,
    ) -> None:
        # Create collapsible title
        title = self._create_title(position)
        super().__init__(title=title, id=id, collapsed=True)
        self.position = position

    def _create_title(self, pos: PositionData) -> str:
        """Create the collapsible title with position summary."""
        side_color = "green" if pos.side == "LONG" else "red"
        pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
        pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
        pnl_pct_sign = "+" if pos.pnl_pct >= 0 else ""

        return (
            f"[bold]{pos.symbol}[/bold]  "
            f"[{side_color}]{pos.side}[/]  "
            f"{pos.size}  "
            f"[dim]@${pos.entry_price:,.2f}[/dim]  "
            f"${pos.current_price:,.2f}  "
            f"[{pnl_color}]{pnl_sign}${pos.unrealized_pnl:,.2f} "
            f"({pnl_pct_sign}{pos.pnl_pct:.1f}%)[/]"
        )

    def compose(self) -> ComposeResult:
        with Vertical(classes="position-details"):
            # Unrealized P&L
            pnl_color = "green" if self.position.unrealized_pnl >= 0 else "red"
            pnl_sign = "+" if self.position.unrealized_pnl >= 0 else ""
            yield Static(
                f"├─ Unrealized: [{pnl_color}]{pnl_sign}${self.position.unrealized_pnl:,.2f}[/]",
                classes="detail-row",
            )

            # Realized P&L
            rpnl_color = "green" if self.position.realized_pnl >= 0 else "red"
            rpnl_sign = "+" if self.position.realized_pnl >= 0 else ""
            yield Static(
                f"├─ Realized: [{rpnl_color}]{rpnl_sign}${self.position.realized_pnl:,.2f}[/]",
                classes="detail-row",
            )

            # Notional value
            yield Static(
                f"├─ Notional: ${self.position.notional_value:,.2f}",
                classes="detail-row",
            )

            # Leverage
            yield Static(
                f"├─ Leverage: {self.position.leverage}x",
                classes="detail-row",
            )

            # Liquidation price
            if self.position.liquidation_price:
                yield Static(
                    f"├─ Liquidation: ${self.position.liquidation_price:,.2f}",
                    classes="detail-row",
                )

            # Stop loss / Take profit
            if self.position.stop_loss:
                yield Static(
                    f"├─ Stop Loss: ${self.position.stop_loss:,.2f}",
                    classes="detail-row",
                )
            if self.position.take_profit:
                yield Static(
                    f"├─ Take Profit: ${self.position.take_profit:,.2f}",
                    classes="detail-row",
                )

            # Duration
            yield Static(
                f"├─ Duration: {self.position.duration_str}",
                classes="detail-row",
            )

            # Strategy
            if self.position.strategy_name:
                yield Static(
                    f"├─ Strategy: {self.position.strategy_name}",
                    classes="detail-row",
                )

            # Close button
            with Horizontal(classes="detail-row"):
                yield Static("└─ ", classes="detail-row")
                yield Button(
                    "Close Position",
                    variant="error",
                    id=f"close-{self.position.position_id}",
                    classes="pos-close-btn",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle close button press."""
        if event.button.id and event.button.id.startswith("close-"):
            self.post_message(self.CloseRequested(self.position))
            event.stop()

    def update_position(self, position: PositionData) -> None:
        """Update position data and refresh display."""
        self.position = position
        self.title = self._create_title(position)


# =============================================================================
# Enhanced Positions Table
# =============================================================================


class EnhancedPositionsTable(Container):
    """
    Advanced positions table with expandable details.

    Features:
    - Collapsible position rows
    - Per-position risk metrics
    - Close position actions
    - Total P&L summary
    - Real-time updates
    """

    DEFAULT_CSS = """
    EnhancedPositionsTable {
        height: auto;
        min-height: 10;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    EnhancedPositionsTable .table-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    EnhancedPositionsTable .table-header {
        height: 1;
        color: $text-muted;
        border-bottom: solid $primary-darken-3;
        margin-bottom: 1;
    }

    EnhancedPositionsTable .positions-container {
        height: auto;
    }

    EnhancedPositionsTable .no-positions {
        height: 3;
        content-align: center middle;
        color: $text-muted;
    }

    EnhancedPositionsTable .total-row {
        height: 2;
        margin-top: 1;
        border-top: solid $primary-darken-3;
        padding-top: 1;
        text-align: right;
    }

    EnhancedPositionsTable .total-row.positive {
        color: $success;
    }

    EnhancedPositionsTable .total-row.negative {
        color: $error;
    }
    """

    positions: reactive[list[PositionData]] = reactive(list, init=False)

    class PositionCloseRequested(Message):
        """Bubbles up when a position close is requested."""

        def __init__(self, position: PositionData) -> None:
            self.position = position
            super().__init__()

    class PositionSelected(Message):
        """Bubbles up when a position is selected."""

        def __init__(self, position: PositionData) -> None:
            self.position = position
            super().__init__()

    def __init__(
        self,
        positions: list[PositionData] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.positions = positions or []
        self._position_rows: dict[str, PositionRow] = {}

    def compose(self) -> ComposeResult:
        yield Static("POSITIONS", classes="table-title")
        yield Static(
            "Symbol        Side    Size        Entry        Current         P&L",
            classes="table-header",
        )

        with Vertical(id="positions-container", classes="positions-container"):
            if not self.positions:
                yield Static("[dim]No open positions[/dim]", classes="no-positions")
            else:
                for pos in self.positions:
                    row = PositionRow(pos, id=f"pos-{pos.position_id}")
                    self._position_rows[pos.position_id] = row
                    yield row

        yield Static(
            self._format_total(),
            classes=f"total-row {self._total_class()}",
            id="total-row",
        )

    def _calculate_total_pnl(self) -> Decimal:
        """Calculate total P&L across all positions."""
        return sum((p.unrealized_pnl for p in self.positions), Decimal("0"))

    def _calculate_total_pnl_pct(self) -> float:
        """Calculate total P&L percentage."""
        total_notional = sum(
            (p.size * p.entry_price for p in self.positions), Decimal("0")
        )
        if total_notional == 0:
            return 0.0
        total_pnl = self._calculate_total_pnl()
        return float((total_pnl / total_notional) * 100)

    def _format_total(self) -> str:
        """Format total P&L display."""
        total = self._calculate_total_pnl()
        pct = self._calculate_total_pnl_pct()
        sign = "+" if total >= 0 else ""
        pct_sign = "+" if pct >= 0 else ""
        return f"Total: {sign}${total:,.2f} ({pct_sign}{pct:.1f}%)"

    def _total_class(self) -> str:
        """Get CSS class for total row."""
        return "positive" if self._calculate_total_pnl() >= 0 else "negative"

    def on_position_row_close_requested(self, event: PositionRow.CloseRequested) -> None:
        """Handle close request from position row."""
        self.post_message(self.PositionCloseRequested(event.position))

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        """Handle position expansion (selection)."""
        if isinstance(event.collapsible, PositionRow):
            self.post_message(self.PositionSelected(event.collapsible.position))

    def update_positions(self, positions: list[PositionData]) -> None:
        """Update all positions."""
        self.positions = positions
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Rebuild position rows."""
        try:
            container = self.query_one("#positions-container", Vertical)
            container.remove_children()
            self._position_rows.clear()

            if not self.positions:
                container.mount(Static("[dim]No open positions[/dim]", classes="no-positions"))
            else:
                for pos in self.positions:
                    row = PositionRow(pos, id=f"pos-{pos.position_id}")
                    self._position_rows[pos.position_id] = row
                    container.mount(row)

            # Update total
            total_row = self.query_one("#total-row", Static)
            total_row.update(self._format_total())
            total_row.remove_class("positive", "negative")
            total_row.add_class(self._total_class())
        except Exception:
            pass

    def update_position(self, position: PositionData) -> None:
        """Update a single position."""
        # Update in list
        for i, pos in enumerate(self.positions):
            if pos.position_id == position.position_id:
                self.positions[i] = position
                break

        # Update row if exists
        if position.position_id in self._position_rows:
            self._position_rows[position.position_id].update_position(position)

        # Update total
        try:
            total_row = self.query_one("#total-row", Static)
            total_row.update(self._format_total())
            total_row.remove_class("positive", "negative")
            total_row.add_class(self._total_class())
        except Exception:
            pass

    def add_position(self, position: PositionData) -> None:
        """Add a new position."""
        self.positions.append(position)
        self._rebuild_rows()

    def remove_position(self, position_id: str) -> None:
        """Remove a position."""
        self.positions = [p for p in self.positions if p.position_id != position_id]
        self._rebuild_rows()


# =============================================================================
# Compact Positions Widget (Alternative using DataTable)
# =============================================================================


class SortablePositionsTable(Container):
    """
    Sortable positions table using DataTable.

    Features:
    - Click column headers to sort (ascending/descending)
    - Visual sort indicators (▲/▼)
    - CVD-friendly P&L colors
    - Row selection emits position ID
    """

    DEFAULT_CSS = """
    SortablePositionsTable {
        height: auto;
        min-height: 8;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    SortablePositionsTable .table-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    SortablePositionsTable .sort-hint {
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }

    SortablePositionsTable DataTable {
        height: auto;
        max-height: 20;
    }
    """

    # Column definitions: (key, label, sort_key_func)
    COLUMNS = [
        ("symbol", "Symbol", lambda p: p.symbol),
        ("side", "Side", lambda p: p.side),
        ("size", "Size", lambda p: float(p.size)),
        ("entry", "Entry", lambda p: float(p.entry_price)),
        ("current", "Current", lambda p: float(p.current_price)),
        ("pnl", "P&L ⇅", lambda p: float(p.unrealized_pnl)),
        ("pnl_pct", "P&L %", lambda p: p.pnl_pct),
    ]

    class RowSelected(Message):
        """Emitted when a position row is selected."""

        def __init__(self, position_id: str, position: PositionData) -> None:
            self.position_id = position_id
            self.position = position
            super().__init__()

    class ColumnSorted(Message):
        """Emitted when a column is sorted."""

        def __init__(self, column: str, reverse: bool) -> None:
            self.column = column
            self.reverse = reverse
            super().__init__()

    def __init__(
        self,
        positions: list[PositionData] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._positions = positions or []
        self._position_map: dict[int, str] = {}  # row_key -> position_id
        self._sort_column: str = "pnl"  # Default sort by P&L
        self._sort_reverse: bool = True  # Default descending (best P&L first)

    def compose(self) -> ComposeResult:
        yield Static("POSITIONS", classes="table-title")
        yield Static("[dim]Click column headers to sort[/dim]", classes="sort-hint")
        yield DataTable(id="positions-dt")

    def on_mount(self) -> None:
        """Setup table on mount."""
        table = self.query_one("#positions-dt", DataTable)

        # Add columns with keys for sorting
        for key, label, _ in self.COLUMNS:
            table.add_column(label, key=key)

        table.cursor_type = "row"
        self._populate_table()

    def _get_sorted_positions(self) -> list[PositionData]:
        """Get positions sorted by current sort column."""
        # Find sort key function
        sort_func = None
        for key, _, func in self.COLUMNS:
            if key == self._sort_column:
                sort_func = func
                break

        if sort_func is None:
            return self._positions

        return sorted(self._positions, key=sort_func, reverse=self._sort_reverse)

    def _populate_table(self) -> None:
        """Populate table with sorted positions."""
        try:
            table = self.query_one("#positions-dt", DataTable)
            table.clear()
            self._position_map.clear()

            sorted_positions = self._get_sorted_positions()

            for pos in sorted_positions:
                # Format P&L with color
                pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
                pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                pnl_pct_sign = "+" if pos.pnl_pct >= 0 else ""

                # Side with color
                side_color = "green" if pos.side == "LONG" else "red"

                row_key = table.add_row(
                    pos.symbol,
                    f"[{side_color}]{pos.side}[/{side_color}]",
                    str(pos.size),
                    f"${pos.entry_price:,.2f}",
                    f"${pos.current_price:,.2f}",
                    f"[{pnl_color}]{pnl_sign}${pos.unrealized_pnl:,.2f}[/{pnl_color}]",
                    f"[{pnl_color}]{pnl_pct_sign}{pos.pnl_pct:.2f}%[/{pnl_color}]",
                )
                self._position_map[row_key.value] = pos.position_id
        except Exception:
            pass

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Handle column header click for sorting."""
        column_key = str(event.column_key)

        # Toggle sort direction if same column, otherwise sort ascending
        if column_key == self._sort_column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column_key
            # Default to descending for numeric columns, ascending for text
            self._sort_reverse = column_key in ("pnl", "pnl_pct", "size", "entry", "current")

        self._populate_table()
        self.post_message(self.ColumnSorted(column_key, self._sort_reverse))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key.value in self._position_map:
            position_id = self._position_map[event.row_key.value]
            # Find the position
            for pos in self._positions:
                if pos.position_id == position_id:
                    self.post_message(self.RowSelected(position_id, pos))
                    break

    def update_positions(self, positions: list[PositionData]) -> None:
        """Update positions and refresh table."""
        self._positions = positions
        self._populate_table()

    def sort_by(self, column: str, reverse: bool = False) -> None:
        """Programmatically sort by column."""
        self._sort_column = column
        self._sort_reverse = reverse
        self._populate_table()


class CompactPositionsTable(Container):
    """
    Compact positions display using DataTable.

    Simpler alternative without sorting for space-constrained layouts.
    """

    DEFAULT_CSS = """
    CompactPositionsTable {
        height: auto;
        min-height: 5;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    CompactPositionsTable .table-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    CompactPositionsTable DataTable {
        height: auto;
        max-height: 15;
    }
    """

    class RowSelected(Message):
        """Emitted when a position row is selected."""

        def __init__(self, position_id: str) -> None:
            self.position_id = position_id
            super().__init__()

    def __init__(
        self,
        positions: list[PositionData] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._positions = positions or []
        self._position_map: dict[int, str] = {}  # row_key -> position_id

    def compose(self) -> ComposeResult:
        yield Static("POSITIONS", classes="table-title")
        yield DataTable(id="positions-dt")

    def on_mount(self) -> None:
        """Setup table on mount."""
        table = self.query_one("#positions-dt", DataTable)
        table.add_columns("Symbol", "Side", "Size", "Entry", "Current", "P&L")
        table.cursor_type = "row"
        self._populate_table()

    def _populate_table(self) -> None:
        """Populate table with positions."""
        try:
            table = self.query_one("#positions-dt", DataTable)
            table.clear()
            self._position_map.clear()

            for pos in self._positions:
                pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                row_key = table.add_row(
                    pos.symbol,
                    pos.side,
                    str(pos.size),
                    f"${pos.entry_price:,.2f}",
                    f"${pos.current_price:,.2f}",
                    f"{pnl_sign}${pos.unrealized_pnl:,.2f}",
                )
                self._position_map[row_key.value] = pos.position_id
        except Exception:
            pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key.value in self._position_map:
            position_id = self._position_map[event.row_key.value]
            self.post_message(self.RowSelected(position_id))

    def update_positions(self, positions: list[PositionData]) -> None:
        """Update positions and refresh table."""
        self._positions = positions
        self._populate_table()


# =============================================================================
# Demo Data Generator
# =============================================================================


def create_demo_positions() -> list[PositionData]:
    """Create demo positions for testing."""
    now = datetime.now()
    return [
        PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500.00"),
            current_price=Decimal("43200.00"),
            unrealized_pnl=Decimal("350.00"),
            leverage=3,
            liquidation_price=Decimal("38250.00"),
            stop_loss=Decimal("41000.00"),
            take_profit=Decimal("46000.00"),
            opened_at=now - timedelta(hours=2, minutes=15),
            strategy_name="Momentum Alpha",
        ),
        PositionData(
            position_id="pos-2",
            symbol="ETH/USDT",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("2850.00"),
            current_price=Decimal("2820.00"),
            unrealized_pnl=Decimal("60.00"),
            leverage=2,
            liquidation_price=Decimal("3135.00"),
            opened_at=now - timedelta(hours=5, minutes=30),
            strategy_name="Mean Reversion",
        ),
        PositionData(
            position_id="pos-3",
            symbol="SOL/USDT",
            side="LONG",
            size=Decimal("25.0"),
            entry_price=Decimal("108.50"),
            current_price=Decimal("105.20"),
            unrealized_pnl=Decimal("-82.50"),
            leverage=1,
            opened_at=now - timedelta(days=1, hours=3),
        ),
    ]

"""
Strategy Card Widget.

Displays a strategy summary in a compact card format.

Features:
- Status icon with state color
- Strategy name
- Mini sparkline for P&L trend
- Current P&L badge
- Selection highlighting

Layout:
    +-----------------------------------------------+
    | [icon] Strategy Name        ▁▂▃▅▆ +$1,250.00 |
    +-----------------------------------------------+
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Label, Sparkline, Static


if TYPE_CHECKING:
    pass


# =============================================================================
# Strategy Status Configuration
# =============================================================================

STRATEGY_STATUS_CONFIG: dict[str, dict[str, str]] = {
    "RUNNING": {"icon": "[green]●[/green]", "label": "RUNNING"},
    "STOPPED": {"icon": "[dim]○[/dim]", "label": "STOPPED"},
    "PAUSED": {"icon": "[yellow]◐[/yellow]", "label": "PAUSED"},
    "STARTING": {"icon": "[yellow]◔[/yellow]", "label": "STARTING"},
    "STOPPING": {"icon": "[yellow]◕[/yellow]", "label": "STOPPING"},
    "ERROR": {"icon": "[red]✗[/red]", "label": "ERROR"},
    "DEGRADED": {"icon": "[yellow]![/yellow]", "label": "DEGRADED"},
}


# =============================================================================
# Strategy Card Widget
# =============================================================================


class StrategyCard(Horizontal):
    """
    Compact strategy summary card.

    Displays:
    - Status icon (color-coded by state)
    - Strategy name
    - Mini P&L sparkline
    - Current P&L value with color

    Emits:
    - StrategyCard.Selected when clicked or Enter pressed
    """

    DEFAULT_CSS = """
    StrategyCard {
        height: 3;
        padding: 0 1;
        border: round $primary-darken-2;
        margin-bottom: 1;
        background: $surface;
    }

    StrategyCard:hover {
        background: $surface-lighten-1;
    }

    StrategyCard:focus {
        border: round $primary;
        background: $surface-lighten-1;
    }

    StrategyCard.-selected {
        border: round $secondary;
        background: $secondary-darken-2;
    }

    StrategyCard .status-icon {
        width: 3;
        height: 1;
        content-align: center middle;
    }

    StrategyCard .strategy-name {
        width: 1fr;
        height: 1;
        text-style: bold;
    }

    StrategyCard .pnl-sparkline {
        width: 12;
        height: 1;
        margin: 0 1;
    }

    StrategyCard .pnl-value {
        width: 15;
        height: 1;
        text-align: right;
    }

    StrategyCard .pnl-value.positive {
        color: $success;
    }

    StrategyCard .pnl-value.negative {
        color: $error;
    }

    StrategyCard .pnl-value.neutral {
        color: $text-muted;
    }
    """

    BINDINGS: ClassVar[list] = []

    # Reactive attributes
    strategy_id: reactive[str] = reactive("")
    strategy_name: reactive[str] = reactive("Unnamed Strategy")
    status: reactive[str] = reactive("STOPPED")
    pnl: reactive[Decimal] = reactive(Decimal("0"))
    pnl_history: reactive[list[float]] = reactive(list, init=False)
    selected: reactive[bool] = reactive(False)

    class Selected(Message):
        """Message sent when strategy card is selected."""

        def __init__(self, strategy_id: str, strategy_name: str) -> None:
            self.strategy_id = strategy_id
            self.strategy_name = strategy_name
            super().__init__()

    def __init__(
        self,
        strategy_id: str,
        name: str,
        status: str = "STOPPED",
        pnl: Decimal = Decimal("0"),
        pnl_history: list[float] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id or f"strategy-card-{strategy_id}")
        self.strategy_id = strategy_id
        self.strategy_name = name
        self.status = status
        self.pnl = pnl
        self.pnl_history = pnl_history or [0.0] * 10
        self.can_focus = True

    def compose(self) -> ComposeResult:
        yield Static(self._get_status_icon(), classes="status-icon", id="status-icon")
        yield Label(self.strategy_name, classes="strategy-name", id="strategy-name")
        yield Sparkline(self.pnl_history, classes="pnl-sparkline", id="pnl-sparkline")
        yield Static(self._format_pnl(), classes=f"pnl-value {self._pnl_class()}", id="pnl-value")

    def _get_status_icon(self) -> str:
        """Get the status icon markup."""
        config = STRATEGY_STATUS_CONFIG.get(self.status, STRATEGY_STATUS_CONFIG["STOPPED"])
        return config["icon"]

    def _format_pnl(self) -> str:
        """Format P&L value with sign."""
        if self.pnl > 0:
            return f"+${self.pnl:,.2f}"
        elif self.pnl < 0:
            return f"-${abs(self.pnl):,.2f}"
        else:
            return "$0.00"

    def _pnl_class(self) -> str:
        """Get CSS class for P&L coloring."""
        if self.pnl > 0:
            return "positive"
        elif self.pnl < 0:
            return "negative"
        return "neutral"

    def watch_status(self, value: str) -> None:
        """Update status icon when status changes."""
        try:
            icon = self.query_one("#status-icon", Static)
            icon.update(self._get_status_icon())
        except Exception:
            pass

    def watch_strategy_name(self, value: str) -> None:
        """Update name label when name changes."""
        try:
            label = self.query_one("#strategy-name", Label)
            label.update(value)
        except Exception:
            pass

    def watch_pnl(self, value: Decimal) -> None:
        """Update P&L display when value changes."""
        try:
            pnl_widget = self.query_one("#pnl-value", Static)
            pnl_widget.update(self._format_pnl())
            # Update CSS class
            pnl_widget.remove_class("positive", "negative", "neutral")
            pnl_widget.add_class(self._pnl_class())
        except Exception:
            pass

    def watch_pnl_history(self, value: list[float]) -> None:
        """Update sparkline when history changes."""
        try:
            sparkline = self.query_one("#pnl-sparkline", Sparkline)
            sparkline.data = value
        except Exception:
            pass

    def watch_selected(self, value: bool) -> None:
        """Toggle selected class."""
        if value:
            self.add_class("-selected")
        else:
            self.remove_class("-selected")

    def on_click(self) -> None:
        """Handle click event."""
        self.post_message(self.Selected(self.strategy_id, self.strategy_name))

    def action_select(self) -> None:
        """Action to select this card."""
        self.post_message(self.Selected(self.strategy_id, self.strategy_name))

    def update_data(
        self,
        status: str | None = None,
        pnl: Decimal | None = None,
        pnl_history: list[float] | None = None,
    ) -> None:
        """Update multiple fields at once."""
        if status is not None:
            self.status = status
        if pnl is not None:
            self.pnl = pnl
        if pnl_history is not None:
            self.pnl_history = pnl_history

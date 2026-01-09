"""
Position Detail Modal Screen.

Modal dialog displaying detailed position information.

Features:
- Complete position metrics
- Risk parameters display
- Close position action
- Modify SL/TP action
- Position history sparkline
- Keyboard shortcuts

Design inspired by:
- Bloomberg Terminal position detail
- Interactive Brokers TWS
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from libra.tui.widgets.enhanced_positions import PositionData


# =============================================================================
# Position Action Result
# =============================================================================


@dataclass
class PositionActionResult:
    """Result returned from position detail modal."""

    action: str = "none"  # none, close, modify_sl, modify_tp
    position_id: str = ""
    new_stop_loss: Decimal | None = None
    new_take_profit: Decimal | None = None


# =============================================================================
# Metric Card Widget
# =============================================================================


class MetricCard(Static):
    """Small metric display card."""

    DEFAULT_CSS = """
    MetricCard {
        height: 3;
        padding: 0 1;
        border: round $primary-darken-3;
        background: $surface;
    }

    MetricCard .metric-label {
        color: $text-muted;
        text-style: bold;
    }

    MetricCard .metric-value {
        text-align: center;
    }

    MetricCard .positive {
        color: $success;
    }

    MetricCard .negative {
        color: $error;
    }
    """

    def __init__(
        self,
        label: str,
        value: str,
        value_class: str = "",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._label = label
        self._value = value
        self._value_class = value_class

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="metric-label")
        yield Static(self._value, classes=f"metric-value {self._value_class}")


# =============================================================================
# Position Detail Modal
# =============================================================================


class PositionDetailModal(ModalScreen[PositionActionResult]):
    """
    Modal screen displaying detailed position information.

    Returns:
        PositionActionResult indicating what action to take (if any).
    """

    DEFAULT_CSS = """
    PositionDetailModal {
        align: center middle;
    }

    PositionDetailModal > Container {
        width: 80;
        height: auto;
        max-height: 85%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    PositionDetailModal .modal-title {
        height: 2;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $text;
    }

    PositionDetailModal .position-header {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }

    PositionDetailModal .symbol-display {
        width: 20;
        text-style: bold;
        content-align: left middle;
    }

    PositionDetailModal .side-display {
        width: 10;
        content-align: center middle;
    }

    PositionDetailModal .side-display.long {
        color: $success;
    }

    PositionDetailModal .side-display.short {
        color: $error;
    }

    PositionDetailModal .pnl-display {
        width: 1fr;
        text-align: right;
        content-align: right middle;
    }

    PositionDetailModal .pnl-display.positive {
        color: $success;
    }

    PositionDetailModal .pnl-display.negative {
        color: $error;
    }

    PositionDetailModal .section-title {
        height: 1;
        text-style: bold;
        color: $primary;
        margin: 1 0;
        border-bottom: solid $primary-darken-3;
    }

    PositionDetailModal .metrics-grid {
        height: auto;
        grid-size: 3;
        grid-gutter: 1;
        margin-bottom: 1;
    }

    PositionDetailModal .risk-section {
        height: auto;
        margin-bottom: 1;
    }

    PositionDetailModal .risk-row {
        height: 1;
        layout: horizontal;
    }

    PositionDetailModal .risk-label {
        width: 20;
        color: $text-muted;
    }

    PositionDetailModal .risk-value {
        width: 1fr;
    }

    PositionDetailModal .risk-value.danger {
        color: $error;
    }

    PositionDetailModal .risk-value.warning {
        color: $warning;
    }

    PositionDetailModal .strategy-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-1;
        border: round $primary-darken-3;
    }

    PositionDetailModal .modal-actions {
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 1;
    }

    PositionDetailModal .modal-actions Button {
        margin: 0 1;
        min-width: 16;
    }

    PositionDetailModal .close-btn {
        background: $error;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "dismiss_modal", "Close", priority=True),
        Binding("c", "close_position", "Close Position"),
        Binding("s", "modify_stop_loss", "Modify SL"),
        Binding("t", "modify_take_profit", "Modify TP"),
    ]

    def __init__(
        self,
        position: PositionData,
    ) -> None:
        super().__init__()
        self._position = position

    def compose(self) -> ComposeResult:
        pos = self._position
        pnl_class = "positive" if pos.unrealized_pnl >= 0 else "negative"
        pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
        side_class = "long" if pos.side == "LONG" else "short"

        with Container():
            yield Static("POSITION DETAILS", classes="modal-title")

            # Header with symbol, side, P&L
            with Horizontal(classes="position-header"):
                yield Static(f"{pos.symbol}", classes="symbol-display")
                yield Static(f"{pos.side}", classes=f"side-display {side_class}")
                yield Static(
                    f"{pnl_sign}${pos.unrealized_pnl:,.2f} ({pnl_sign}{pos.pnl_pct:.2f}%)",
                    classes=f"pnl-display {pnl_class}",
                )

            # Core metrics grid
            yield Static("POSITION METRICS", classes="section-title")
            with Grid(classes="metrics-grid"):
                yield self._metric_card("Size", f"{pos.size}")
                yield self._metric_card("Entry Price", f"${pos.entry_price:,.2f}")
                yield self._metric_card("Current Price", f"${pos.current_price:,.2f}")
                yield self._metric_card("Notional", f"${pos.notional_value:,.2f}")
                yield self._metric_card("Leverage", f"{pos.leverage}x")
                yield self._metric_card("Duration", pos.duration_str)

            # P&L breakdown
            yield Static("P&L BREAKDOWN", classes="section-title")
            with Grid(classes="metrics-grid"):
                yield self._metric_card(
                    "Unrealized",
                    f"{pnl_sign}${pos.unrealized_pnl:,.2f}",
                    pnl_class,
                )
                rpnl_sign = "+" if pos.realized_pnl >= 0 else ""
                rpnl_class = "positive" if pos.realized_pnl >= 0 else "negative"
                yield self._metric_card(
                    "Realized",
                    f"{rpnl_sign}${pos.realized_pnl:,.2f}",
                    rpnl_class,
                )
                total_pnl = pos.unrealized_pnl + pos.realized_pnl
                total_sign = "+" if total_pnl >= 0 else ""
                total_class = "positive" if total_pnl >= 0 else "negative"
                yield self._metric_card(
                    "Total",
                    f"{total_sign}${total_pnl:,.2f}",
                    total_class,
                )

            # Risk parameters
            yield Static("RISK PARAMETERS", classes="section-title")
            with Vertical(classes="risk-section"):
                # Liquidation price
                if pos.liquidation_price:
                    liq_dist = self._liquidation_distance(pos)
                    liq_class = "danger" if liq_dist < 10 else ("warning" if liq_dist < 20 else "")
                    yield self._risk_row(
                        "Liquidation Price:",
                        f"${pos.liquidation_price:,.2f} ({liq_dist:.1f}% away)",
                        liq_class,
                    )
                else:
                    yield self._risk_row("Liquidation Price:", "N/A (spot)")

                # Stop loss
                if pos.stop_loss:
                    sl_dist = self._stop_loss_distance(pos)
                    yield self._risk_row(
                        "Stop Loss:",
                        f"${pos.stop_loss:,.2f} ({sl_dist:.1f}% away)",
                    )
                else:
                    yield self._risk_row("Stop Loss:", "[dim]Not set[/dim]")

                # Take profit
                if pos.take_profit:
                    tp_dist = self._take_profit_distance(pos)
                    yield self._risk_row(
                        "Take Profit:",
                        f"${pos.take_profit:,.2f} ({tp_dist:.1f}% away)",
                    )
                else:
                    yield self._risk_row("Take Profit:", "[dim]Not set[/dim]")

            # Strategy info
            if pos.strategy_name:
                yield Static("STRATEGY", classes="section-title")
                with Vertical(classes="strategy-section"):
                    yield Static(f"Name: {pos.strategy_name}")
                    if pos.strategy_id:
                        yield Static(f"ID: {pos.strategy_id}")

            # Action buttons
            with Horizontal(classes="modal-actions"):
                yield Button("Close", variant="default", id="btn-dismiss")
                yield Button("Modify SL/TP", variant="primary", id="btn-modify")
                yield Button("Close Position", variant="error", id="btn-close-position")

    def _metric_card(
        self,
        label: str,
        value: str,
        value_class: str = "",
    ) -> MetricCard:
        """Create a metric card."""
        return MetricCard(label=label, value=value, value_class=value_class)

    def _risk_row(
        self,
        label: str,
        value: str,
        value_class: str = "",
    ) -> Horizontal:
        """Create a risk parameter row."""
        container = Horizontal(classes="risk-row")
        container.compose_add_child(Static(label, classes="risk-label"))
        container.compose_add_child(Static(value, classes=f"risk-value {value_class}"))
        return container

    def _liquidation_distance(self, pos: PositionData) -> float:
        """Calculate distance to liquidation as percentage."""
        if not pos.liquidation_price or pos.current_price == 0:
            return 100.0
        return abs(float((pos.current_price - pos.liquidation_price) / pos.current_price * 100))

    def _stop_loss_distance(self, pos: PositionData) -> float:
        """Calculate distance to stop loss as percentage."""
        if not pos.stop_loss or pos.current_price == 0:
            return 0.0
        return abs(float((pos.current_price - pos.stop_loss) / pos.current_price * 100))

    def _take_profit_distance(self, pos: PositionData) -> float:
        """Calculate distance to take profit as percentage."""
        if not pos.take_profit or pos.current_price == 0:
            return 0.0
        return abs(float((pos.take_profit - pos.current_price) / pos.current_price * 100))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-dismiss":
            self.action_dismiss_modal()
        elif event.button.id == "btn-modify":
            self.action_modify_stop_loss()
        elif event.button.id == "btn-close-position":
            self.action_close_position()

    def action_dismiss_modal(self) -> None:
        """Dismiss without action."""
        self.dismiss(PositionActionResult(
            action="none",
            position_id=self._position.position_id,
        ))

    def action_close_position(self) -> None:
        """Request to close the position."""
        self.dismiss(PositionActionResult(
            action="close",
            position_id=self._position.position_id,
        ))

    def action_modify_stop_loss(self) -> None:
        """Request to modify stop loss."""
        self.dismiss(PositionActionResult(
            action="modify_sl",
            position_id=self._position.position_id,
        ))

    def action_modify_take_profit(self) -> None:
        """Request to modify take profit."""
        self.dismiss(PositionActionResult(
            action="modify_tp",
            position_id=self._position.position_id,
        ))


# =============================================================================
# Close Position Confirmation Modal
# =============================================================================


class ClosePositionModal(ModalScreen[bool]):
    """
    Confirmation modal for closing a position.

    Returns True if confirmed, False if cancelled.
    """

    DEFAULT_CSS = """
    ClosePositionModal {
        align: center middle;
    }

    ClosePositionModal > Container {
        width: 60;
        height: auto;
        background: $surface;
        border: round $error;
        padding: 1 2;
    }

    ClosePositionModal .modal-title {
        height: 1;
        text-style: bold;
        text-align: center;
        color: $error;
        margin-bottom: 1;
    }

    ClosePositionModal .position-summary {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: round $primary-darken-3;
        margin: 1 0;
    }

    ClosePositionModal .summary-row {
        height: 1;
        layout: horizontal;
    }

    ClosePositionModal .summary-label {
        width: 15;
        color: $text-muted;
    }

    ClosePositionModal .summary-value {
        width: 1fr;
    }

    ClosePositionModal .pnl-positive {
        color: $success;
    }

    ClosePositionModal .pnl-negative {
        color: $error;
    }

    ClosePositionModal .warning-text {
        height: 2;
        text-align: center;
        color: $warning;
        margin: 1 0;
    }

    ClosePositionModal .modal-actions {
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 1;
    }

    ClosePositionModal .modal-actions Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    def __init__(
        self,
        position: PositionData,
    ) -> None:
        super().__init__()
        self._position = position

    def compose(self) -> ComposeResult:
        pos = self._position
        pnl_class = "pnl-positive" if pos.unrealized_pnl >= 0 else "pnl-negative"
        pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""

        with Container():
            yield Static("CLOSE POSITION", classes="modal-title")

            with Vertical(classes="position-summary"):
                yield self._summary_row("Symbol:", pos.symbol)
                yield self._summary_row("Side:", pos.side)
                yield self._summary_row("Size:", str(pos.size))
                yield self._summary_row("Entry:", f"${pos.entry_price:,.2f}")
                yield self._summary_row("Current:", f"${pos.current_price:,.2f}")
                yield self._summary_row(
                    "P&L:",
                    f"{pnl_sign}${pos.unrealized_pnl:,.2f} ({pnl_sign}{pos.pnl_pct:.1f}%)",
                    pnl_class,
                )

            yield Static(
                "This action will close your position at market price.\nThis cannot be undone.",
                classes="warning-text",
            )

            with Horizontal(classes="modal-actions"):
                yield Button("Cancel (N)", variant="default", id="btn-cancel")
                yield Button("Close Position (Y)", variant="error", id="btn-confirm")

    def _summary_row(
        self,
        label: str,
        value: str,
        value_class: str = "",
    ) -> Horizontal:
        """Create a summary row."""
        container = Horizontal(classes="summary-row")
        container.compose_add_child(Static(label, classes="summary-label"))
        container.compose_add_child(Static(value, classes=f"summary-value {value_class}"))
        return container

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-confirm":
            self.action_confirm()
        else:
            self.action_cancel()

    def action_confirm(self) -> None:
        """Confirm closing the position."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel the close."""
        self.dismiss(False)

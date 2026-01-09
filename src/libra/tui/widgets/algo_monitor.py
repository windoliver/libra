"""
Algorithm Monitor Widget.

Real-time execution algorithm monitoring for Issue #36.

Displays:
- Active algorithm executions with progress
- Algorithm state (RUNNING, PAUSED, COMPLETED, CANCELLED, FAILED)
- Execution metrics (fill rate, avg price, slippage)
- Controls for pause/resume/cancel

Layout:
    +-- ALGORITHM EXECUTIONS -----------------------+
    | [No active executions]                        |
    |                                               |
    | twap-001 BTC/USDT BUY                        |
    |   [########--] 80% | 8/10 slices             |
    |   State: RUNNING | Filled: 80.00 / 100.00    |
    |   Avg: $51,250.00 | Slip: +0.02%             |
    |   [Pause] [Cancel]                            |
    |                                               |
    | vwap-002 ETH/USDT SELL                       |
    |   [####------] 40% | 5/12 slices             |
    |   State: RUNNING | Filled: 2.50 / 5.00       |
    |   Avg: $3,045.00 | Slip: -0.01%              |
    |   [Pause] [Cancel]                            |
    +-----------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label, ProgressBar, Static


if TYPE_CHECKING:
    from libra.execution.algorithm import ExecutionProgress


# =============================================================================
# Algorithm Execution Data
# =============================================================================


@dataclass
class AlgorithmExecutionData:
    """Data for a single algorithm execution."""

    execution_id: str
    algorithm: str  # "twap", "vwap", "iceberg", "pov"
    symbol: str
    side: str  # "BUY" or "SELL"
    total_quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    state: str = "RUNNING"  # PENDING, RUNNING, PAUSED, COMPLETED, CANCELLED, FAILED
    total_slices: int = 10
    completed_slices: int = 0
    average_price: Decimal | None = None
    slippage_pct: float = 0.0
    error_message: str = ""


# =============================================================================
# Execution Card Widget
# =============================================================================


class ExecutionCard(Container):
    """
    Card displaying a single algorithm execution.

    Shows progress, metrics, and control buttons.
    """

    DEFAULT_CSS = """
    ExecutionCard {
        height: auto;
        width: 100%;
        background: $surface-darken-1;
        border: round $primary-darken-2;
        padding: 1;
        margin-bottom: 1;
    }

    ExecutionCard.completed {
        border: round $success-darken-1;
    }

    ExecutionCard.failed {
        border: round $error-darken-1;
    }

    ExecutionCard.cancelled {
        border: round $warning-darken-1;
    }

    ExecutionCard.paused {
        border: round $warning;
    }

    ExecutionCard .exec-header {
        height: 1;
        margin-bottom: 0;
    }

    ExecutionCard .exec-id {
        color: $primary;
        text-style: bold;
        width: auto;
    }

    ExecutionCard .exec-symbol {
        color: $text;
        margin-left: 1;
    }

    ExecutionCard .exec-side-buy {
        color: $success;
        text-style: bold;
        margin-left: 1;
    }

    ExecutionCard .exec-side-sell {
        color: $error;
        text-style: bold;
        margin-left: 1;
    }

    ExecutionCard .progress-row {
        height: 1;
        margin-top: 0;
    }

    ExecutionCard ProgressBar {
        width: 50%;
    }

    ExecutionCard .slice-info {
        color: $text-muted;
        margin-left: 1;
    }

    ExecutionCard .metrics-row {
        height: 1;
        color: $text-muted;
    }

    ExecutionCard .state-running {
        color: $success;
    }

    ExecutionCard .state-paused {
        color: $warning;
    }

    ExecutionCard .state-completed {
        color: $success;
    }

    ExecutionCard .state-cancelled {
        color: $warning;
    }

    ExecutionCard .state-failed {
        color: $error;
    }

    ExecutionCard .button-row {
        height: 3;
        margin-top: 1;
    }

    ExecutionCard Button {
        min-width: 10;
        margin-right: 1;
    }
    """

    class PauseRequested(Message):
        """Message sent when pause is requested."""

        def __init__(self, execution_id: str) -> None:
            super().__init__()
            self.execution_id = execution_id

    class ResumeRequested(Message):
        """Message sent when resume is requested."""

        def __init__(self, execution_id: str) -> None:
            super().__init__()
            self.execution_id = execution_id

    class CancelRequested(Message):
        """Message sent when cancel is requested."""

        def __init__(self, execution_id: str) -> None:
            super().__init__()
            self.execution_id = execution_id

    def __init__(self, data: AlgorithmExecutionData, id: str | None = None) -> None:
        super().__init__(id=id or f"exec-{data.execution_id}")
        self._data = data

    def compose(self) -> ComposeResult:
        # Header: algorithm-id symbol side
        with Horizontal(classes="exec-header"):
            algo_display = self._data.algorithm.upper()
            yield Static(f"{algo_display}-{self._data.execution_id[:6]}", classes="exec-id")
            yield Static(self._data.symbol, classes="exec-symbol")
            side_class = "exec-side-buy" if self._data.side == "BUY" else "exec-side-sell"
            yield Static(self._data.side, classes=side_class)

        # Progress bar and slice info
        with Horizontal(classes="progress-row"):
            pct = self._calc_progress_pct()
            yield ProgressBar(total=100, show_eta=False, show_percentage=True, id="progress")
            yield Static(
                f"| {self._data.completed_slices}/{self._data.total_slices} slices",
                classes="slice-info",
            )

        # State and fill info
        state_class = f"state-{self._data.state.lower()}"
        yield Static(
            f"State: [{state_class}]{self._data.state}[/{state_class}] | "
            f"Filled: {self._data.filled_quantity} / {self._data.total_quantity}",
            classes="metrics-row",
        )

        # Price and slippage info
        avg_price = f"${self._data.average_price:,.2f}" if self._data.average_price else "N/A"
        slip_sign = "+" if self._data.slippage_pct >= 0 else ""
        yield Static(
            f"Avg: {avg_price} | Slip: {slip_sign}{self._data.slippage_pct:.2f}%",
            classes="metrics-row",
        )

        # Error message if failed
        if self._data.state == "FAILED" and self._data.error_message:
            yield Static(f"[red]Error: {self._data.error_message}[/red]")

        # Control buttons (only show for active executions)
        if self._data.state in ("RUNNING", "PAUSED", "PENDING"):
            with Horizontal(classes="button-row"):
                if self._data.state == "RUNNING":
                    yield Button("Pause", id="pause-btn", variant="warning")
                elif self._data.state == "PAUSED":
                    yield Button("Resume", id="resume-btn", variant="success")
                yield Button("Cancel", id="cancel-btn", variant="error")

    def on_mount(self) -> None:
        """Initialize progress bar."""
        self._update_progress()
        self._update_card_class()

    def _calc_progress_pct(self) -> float:
        """Calculate progress percentage."""
        if self._data.total_quantity > 0:
            return float(self._data.filled_quantity / self._data.total_quantity * 100)
        return 0.0

    def _update_progress(self) -> None:
        """Update progress bar."""
        try:
            bar = self.query_one("#progress", ProgressBar)
            bar.update(progress=self._calc_progress_pct())
        except Exception:
            pass

    def _update_card_class(self) -> None:
        """Update card class based on state."""
        self.remove_class("completed", "failed", "cancelled", "paused")
        state_lower = self._data.state.lower()
        if state_lower in ("completed", "failed", "cancelled", "paused"):
            self.add_class(state_lower)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "pause-btn":
            self.post_message(self.PauseRequested(self._data.execution_id))
        elif event.button.id == "resume-btn":
            self.post_message(self.ResumeRequested(self._data.execution_id))
        elif event.button.id == "cancel-btn":
            self.post_message(self.CancelRequested(self._data.execution_id))

    def update_data(self, data: AlgorithmExecutionData) -> None:
        """Update the execution data and refresh display."""
        self._data = data
        self._update_progress()
        self._update_card_class()
        # Re-render would require more complex logic
        # For now, the widget should be replaced when data changes significantly


# =============================================================================
# Algorithm Monitor Widget
# =============================================================================


class AlgorithmMonitor(Container):
    """
    Real-time algorithm execution monitor.

    Displays all active algorithm executions with progress,
    metrics, and control buttons.

    Usage:
        monitor = AlgorithmMonitor()

        # Add/update an execution
        monitor.update_execution(AlgorithmExecutionData(
            execution_id="abc123",
            algorithm="twap",
            symbol="BTC/USDT",
            side="BUY",
            total_quantity=Decimal("100"),
        ))

        # Remove an execution
        monitor.remove_execution("abc123")

        # Clear all
        monitor.clear_executions()
    """

    DEFAULT_CSS = """
    AlgorithmMonitor {
        height: auto;
        min-height: 8;
        max-height: 50%;
        width: 100%;
        background: $surface;
        border: round $primary-darken-1;
        padding: 1;
    }

    AlgorithmMonitor .monitor-title {
        text-style: bold;
        color: $text;
        background: $primary-darken-2;
        text-align: center;
        padding: 0 1;
        margin-bottom: 1;
    }

    AlgorithmMonitor .empty-message {
        color: $text-muted;
        text-style: italic;
        text-align: center;
        padding: 2;
    }

    AlgorithmMonitor .stats-row {
        height: 1;
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }

    AlgorithmMonitor #executions-container {
        height: auto;
        max-height: 100%;
    }
    """

    def __init__(self, id: str = "algo-monitor") -> None:
        super().__init__(id=id)
        self._executions: dict[str, AlgorithmExecutionData] = {}

    def compose(self) -> ComposeResult:
        yield Static("ALGORITHM EXECUTIONS", classes="monitor-title")
        yield Static("Active: 0 | Completed: 0 | Failed: 0", id="stats-row", classes="stats-row")
        yield ScrollableContainer(id="executions-container")

    def on_mount(self) -> None:
        """Initialize the monitor."""
        self._update_empty_state()
        self._update_stats()

    def _update_empty_state(self) -> None:
        """Show/hide empty message based on execution count."""
        try:
            container = self.query_one("#executions-container", ScrollableContainer)
            if not self._executions:
                # Clear and show empty message
                container.remove_children()
                container.mount(Static("[dim]No active executions[/dim]", classes="empty-message"))
            else:
                # Remove empty message if present
                try:
                    empty_msg = container.query_one(".empty-message")
                    empty_msg.remove()
                except Exception:
                    pass
        except Exception:
            pass

    def _update_stats(self) -> None:
        """Update the stats row."""
        try:
            active = sum(1 for e in self._executions.values() if e.state in ("RUNNING", "PAUSED", "PENDING"))
            completed = sum(1 for e in self._executions.values() if e.state == "COMPLETED")
            failed = sum(1 for e in self._executions.values() if e.state in ("FAILED", "CANCELLED"))

            stats = self.query_one("#stats-row", Static)
            stats.update(f"Active: {active} | Completed: {completed} | Failed: {failed}")
        except Exception:
            pass

    def update_execution(self, data: AlgorithmExecutionData) -> None:
        """
        Add or update an algorithm execution.

        Args:
            data: Execution data to add/update
        """
        execution_id = data.execution_id
        is_new = execution_id not in self._executions
        self._executions[execution_id] = data

        try:
            container = self.query_one("#executions-container", ScrollableContainer)

            # Remove empty message if this is first execution
            if is_new and len(self._executions) == 1:
                try:
                    empty_msg = container.query_one(".empty-message")
                    empty_msg.remove()
                except Exception:
                    pass

            if is_new:
                # Create new card
                card = ExecutionCard(data)
                container.mount(card)
            else:
                # Update existing card
                try:
                    card = self.query_one(f"#exec-{execution_id}", ExecutionCard)
                    card.update_data(data)
                except Exception:
                    # Card not found, create new one
                    card = ExecutionCard(data)
                    container.mount(card)

        except Exception:
            pass

        self._update_stats()

    def remove_execution(self, execution_id: str) -> None:
        """
        Remove an execution from the monitor.

        Args:
            execution_id: ID of execution to remove
        """
        if execution_id in self._executions:
            del self._executions[execution_id]

            try:
                card = self.query_one(f"#exec-{execution_id}", ExecutionCard)
                card.remove()
            except Exception:
                pass

            self._update_empty_state()
            self._update_stats()

    def clear_executions(self) -> None:
        """Remove all executions."""
        self._executions.clear()

        try:
            container = self.query_one("#executions-container", ScrollableContainer)
            container.remove_children()
        except Exception:
            pass

        self._update_empty_state()
        self._update_stats()

    def update_from_progress(
        self,
        progress: ExecutionProgress,
        algorithm: str = "unknown",
        symbol: str = "UNKNOWN",
        side: str = "BUY",
        avg_price: Decimal | None = None,
        slippage_pct: float = 0.0,
    ) -> None:
        """
        Update from an ExecutionProgress object.

        Args:
            progress: Progress object from execution engine
            algorithm: Algorithm name (e.g., "twap", "vwap")
            symbol: Trading symbol
            side: Order side ("BUY" or "SELL")
            avg_price: Average execution price
            slippage_pct: Slippage percentage
        """
        data = AlgorithmExecutionData(
            execution_id=progress.parent_order_id,
            algorithm=algorithm,
            symbol=symbol,
            side=side.upper(),
            total_quantity=progress.total_quantity,
            filled_quantity=progress.executed_quantity,
            state=progress.state.value if hasattr(progress.state, "value") else str(progress.state),
            total_slices=progress.num_children_spawned,
            completed_slices=progress.num_children_filled,
            average_price=avg_price or progress.avg_fill_price,
            slippage_pct=slippage_pct,
        )
        self.update_execution(data)

    def get_execution(self, execution_id: str) -> AlgorithmExecutionData | None:
        """Get execution data by ID."""
        return self._executions.get(execution_id)

    def get_active_executions(self) -> list[AlgorithmExecutionData]:
        """Get all active executions."""
        return [e for e in self._executions.values() if e.state in ("RUNNING", "PAUSED", "PENDING")]

    # =========================================================================
    # Event Handlers (Forward to parent)
    # =========================================================================

    def on_execution_card_pause_requested(self, event: ExecutionCard.PauseRequested) -> None:
        """Forward pause request to parent."""
        # Parent app should handle this and call execution engine
        pass

    def on_execution_card_resume_requested(self, event: ExecutionCard.ResumeRequested) -> None:
        """Forward resume request to parent."""
        pass

    def on_execution_card_cancel_requested(self, event: ExecutionCard.CancelRequested) -> None:
        """Forward cancel request to parent."""
        pass

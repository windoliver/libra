"""
Strategy Management Screen.

Main screen for managing trading strategies with:
- Strategy list/tree view (left panel)
- Strategy details with metrics, parameters, positions (right panel)
- Actions: start, stop, pause, edit, delete strategies

Design inspired by:
- Bloomberg Terminal (information-dense, professional)
- Lazygit (vim-style navigation, keyboard-first)
- K9s (panel layout, context-sensitive actions)

Layout:
    +-- STRATEGY MANAGEMENT -----------------------------------------+
    |                                                                 |
    | +-- STRATEGIES --+  +-- STRATEGY DETAILS -------------------+  |
    | | [filter____]   |  | SMA_Cross_BTC           [RUNNING]     |  |
    | |                |  |                                        |  |
    | | > SMA_Cross    |  | +-- PERFORMANCE ----------------------+  |
    | |   RSI_Revert   |  | | +$1,250.00  [sparkline]             |  |
    | |   MACD_Mom     |  | | Sharpe: 2.1  DD: 12%  WR: 65%       |  |
    | |                |  | +------------------------------------+  |
    | |                |  |                                        |  |
    | | [n]New         |  | +-- PARAMETERS -----------------------+  |
    | |                |  | | fast_period: 10                      |  |
    | +----------------+  | | slow_period: 30                      |  |
    |                     | +------------------------------------+  |
    |                     |                                        |  |
    |                     | [s]Start [x]Stop [e]Edit [d]Delete     |  |
    |                     +----------------------------------------+  |
    +-----------------------------------------------------------------+
"""

from __future__ import annotations

import random
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Input, Label, Rule, Sparkline, Static

from libra.tui.screens.strategy_edit_modal import (
    ConfirmationModal,
    StrategyCreateModal,
    StrategyEditModal,
    StrategyEditResult,
)
from libra.tui.widgets.parameter_editor import (
    ParameterDefinition,
    ParameterEditor,
    ParameterType,
    create_sma_cross_parameters,
)
from libra.tui.widgets.strategy_card import StrategyCard
from libra.tui.widgets.strategy_metrics import StrategyMetricsPanel
from libra.tui.widgets.strategy_tree import PositionInfo, StrategyInfo, StrategyTree


if TYPE_CHECKING:
    from libra.core.message_bus import MessageBus


# =============================================================================
# Strategy Status Indicator
# =============================================================================


class StrategyStatusIndicator(Static):
    """Large status indicator for the detail panel header."""

    DEFAULT_CSS = """
    StrategyStatusIndicator {
        height: 1;
        padding: 0 1;
        text-style: bold;
    }

    StrategyStatusIndicator.running {
        color: $success;
    }

    StrategyStatusIndicator.stopped {
        color: $text-muted;
    }

    StrategyStatusIndicator.paused {
        color: $warning;
    }

    StrategyStatusIndicator.error {
        color: $error;
    }
    """

    status: reactive[str] = reactive("STOPPED")

    def __init__(self, status: str = "STOPPED", id: str | None = None) -> None:
        super().__init__(id=id)
        self.status = status

    def render(self) -> str:
        icons = {
            "RUNNING": "[green]● RUNNING[/green]",
            "STOPPED": "[dim]○ STOPPED[/dim]",
            "PAUSED": "[yellow]◐ PAUSED[/yellow]",
            "STARTING": "[yellow]◔ STARTING...[/yellow]",
            "STOPPING": "[yellow]◕ STOPPING...[/yellow]",
            "ERROR": "[red]✗ ERROR[/red]",
        }
        return icons.get(self.status, "[dim]? UNKNOWN[/dim]")

    def watch_status(self, value: str) -> None:
        """Update CSS class when status changes."""
        self.remove_class("running", "stopped", "paused", "error")
        if value == "RUNNING":
            self.add_class("running")
        elif value in ("PAUSED", "STARTING", "STOPPING"):
            self.add_class("paused")
        elif value == "ERROR":
            self.add_class("error")
        else:
            self.add_class("stopped")


# =============================================================================
# Strategy Detail Header
# =============================================================================


class StrategyDetailHeader(Horizontal):
    """Header for the strategy detail panel."""

    DEFAULT_CSS = """
    StrategyDetailHeader {
        height: 3;
        padding: 0 1;
        background: $surface-darken-1;
        border: round $primary-darken-2;
    }

    StrategyDetailHeader .strategy-name {
        width: 1fr;
        height: 1;
        text-style: bold;
        content-align: left middle;
    }

    StrategyDetailHeader StrategyStatusIndicator {
        width: auto;
    }
    """

    name: reactive[str] = reactive("No Strategy Selected")
    status: reactive[str] = reactive("STOPPED")

    def __init__(self, name: str = "No Strategy Selected", status: str = "STOPPED", id: str | None = None) -> None:
        super().__init__(id=id)
        self.name = name
        self.status = status

    def compose(self) -> ComposeResult:
        yield Label(self.name, classes="strategy-name", id="detail-name")
        yield StrategyStatusIndicator(self.status, id="detail-status")

    def watch_name(self, value: str) -> None:
        try:
            self.query_one("#detail-name", Label).update(value)
        except Exception:
            pass

    def watch_status(self, value: str) -> None:
        try:
            self.query_one("#detail-status", StrategyStatusIndicator).status = value
        except Exception:
            pass


# =============================================================================
# Strategy Control Bar
# =============================================================================


class StrategyControlBar(Horizontal):
    """Action buttons for strategy control."""

    DEFAULT_CSS = """
    StrategyControlBar {
        height: 3;
        padding: 0 1;
        align: center middle;
    }

    StrategyControlBar Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    def compose(self) -> ComposeResult:
        yield Button("Start (s)", variant="success", id="btn-start")
        yield Button("Stop (x)", variant="error", id="btn-stop")
        yield Button("Pause (p)", variant="warning", id="btn-pause")
        yield Button("Edit (e)", variant="primary", id="btn-edit")
        yield Button("Delete (d)", variant="default", id="btn-delete")

    def set_status(self, status: str) -> None:
        """Update button states based on strategy status."""
        try:
            start_btn = self.query_one("#btn-start", Button)
            stop_btn = self.query_one("#btn-stop", Button)
            pause_btn = self.query_one("#btn-pause", Button)

            if status == "RUNNING":
                start_btn.disabled = True
                stop_btn.disabled = False
                pause_btn.disabled = False
            elif status == "PAUSED":
                start_btn.disabled = False
                stop_btn.disabled = False
                pause_btn.disabled = True
            else:  # STOPPED or other
                start_btn.disabled = False
                stop_btn.disabled = True
                pause_btn.disabled = True
        except Exception:
            pass


# =============================================================================
# Strategy List Panel
# =============================================================================


class StrategyListPanel(Container):
    """Left panel containing the strategy list and filter."""

    DEFAULT_CSS = """
    StrategyListPanel {
        width: 40;
        height: 100%;
        border-right: solid $primary-darken-2;
        padding: 1;
    }

    StrategyListPanel .panel-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    StrategyListPanel #strategy-filter {
        margin-bottom: 1;
    }

    StrategyListPanel .strategy-list {
        height: 1fr;
    }

    StrategyListPanel .panel-actions {
        height: 3;
        margin-top: 1;
    }
    """

    def __init__(self, strategies: list[StrategyInfo] | None = None, id: str | None = None) -> None:
        super().__init__(id=id)
        self._strategies = strategies or []

    def compose(self) -> ComposeResult:
        yield Static("STRATEGIES", classes="panel-title")
        yield Input(placeholder="Filter strategies...", id="strategy-filter")
        yield StrategyTree(
            label="Strategies",
            strategies=self._strategies,
            id="strategy-tree",
        )
        with Horizontal(classes="panel-actions"):
            yield Button("+ New", variant="primary", id="btn-new-strategy")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter strategies by name."""
        if event.input.id == "strategy-filter":
            filter_text = event.value.lower()
            # TODO: Implement filtering in StrategyTree
            pass


# =============================================================================
# Strategy Detail Panel
# =============================================================================


class StrategyDetailPanel(VerticalScroll):
    """Right panel showing strategy details."""

    DEFAULT_CSS = """
    StrategyDetailPanel {
        width: 1fr;
        height: 100%;
        padding: 1;
    }

    StrategyDetailPanel .no-selection {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }

    StrategyDetailPanel .detail-content {
        height: auto;
    }

    StrategyDetailPanel Collapsible {
        margin-bottom: 1;
    }
    """

    strategy_id: reactive[str] = reactive("")
    strategy_name: reactive[str] = reactive("")
    strategy_status: reactive[str] = reactive("STOPPED")

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._current_strategy: StrategyInfo | None = None
        self._parameters: list[ParameterDefinition] = []

    def compose(self) -> ComposeResult:
        yield Static("Select a strategy to view details", classes="no-selection", id="no-selection")

        with Container(classes="detail-content", id="detail-content"):
            yield StrategyDetailHeader(id="detail-header")

            with Collapsible(title="Performance", collapsed=False, id="perf-section"):
                yield StrategyMetricsPanel(id="metrics-panel")

            with Collapsible(title="Parameters", collapsed=False, id="params-section"):
                yield ParameterEditor(
                    parameters=create_sma_cross_parameters(),
                    title="",
                    id="params-editor",
                )

            with Collapsible(title="Positions", collapsed=True, id="positions-section"):
                yield Static("[dim]No open positions[/dim]", id="positions-content")

            with Collapsible(title="Recent Signals", collapsed=True, id="signals-section"):
                yield Static("[dim]No recent signals[/dim]", id="signals-content")

            yield StrategyControlBar(id="control-bar")

    def on_mount(self) -> None:
        """Hide detail content initially."""
        try:
            self.query_one("#detail-content").display = False
        except Exception:
            pass

    def show_strategy(self, strategy: StrategyInfo, parameters: list[ParameterDefinition] | None = None) -> None:
        """Display strategy details."""
        self._current_strategy = strategy
        self._parameters = parameters or create_sma_cross_parameters()

        try:
            # Hide no-selection message
            self.query_one("#no-selection").display = False

            # Show detail content
            self.query_one("#detail-content").display = True

            # Update header
            header = self.query_one("#detail-header", StrategyDetailHeader)
            header.name = strategy.name
            header.status = strategy.status

            # Update metrics
            metrics = self.query_one("#metrics-panel", StrategyMetricsPanel)
            metrics.update_all_metrics(
                total_pnl=strategy.total_pnl,
                total_trades=strategy.total_trades,
                win_rate=strategy.win_rate,
                sharpe_ratio=random.uniform(0.5, 3.0),
                max_drawdown=random.uniform(5, 30),
                profit_factor=random.uniform(1.0, 2.5),
            )

            # Update positions
            positions_content = self.query_one("#positions-content", Static)
            if strategy.positions:
                pos_text = "\n".join(
                    f"{sym}: {pos.display_text}"
                    for sym, pos in strategy.positions.items()
                )
                positions_content.update(pos_text)
            else:
                positions_content.update("[dim]No open positions[/dim]")

            # Update control bar
            control_bar = self.query_one("#control-bar", StrategyControlBar)
            control_bar.set_status(strategy.status)

        except Exception as e:
            pass

    def clear_selection(self) -> None:
        """Clear the detail view."""
        self._current_strategy = None
        try:
            self.query_one("#no-selection").display = True
            self.query_one("#detail-content").display = False
        except Exception:
            pass

    def get_current_strategy(self) -> StrategyInfo | None:
        """Get the currently displayed strategy."""
        return self._current_strategy


# =============================================================================
# Strategy Management Screen
# =============================================================================


class StrategyManagementScreen(Screen):
    """
    Main strategy management screen.

    Provides a split-panel view for browsing and managing trading strategies.
    """

    DEFAULT_CSS = """
    StrategyManagementScreen {
        layout: horizontal;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        # Navigation
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("/", "focus_filter", "Filter"),
        # Actions
        Binding("n", "new_strategy", "New"),
        Binding("e", "edit_strategy", "Edit"),
        Binding("d", "delete_strategy", "Delete"),
        Binding("s", "start_strategy", "Start"),
        Binding("x", "stop_strategy", "Stop"),
        Binding("p", "pause_strategy", "Pause"),
        Binding("r", "refresh", "Refresh"),
        # Section toggles
        Binding("1", "toggle_section('perf-section')", show=False),
        Binding("2", "toggle_section('params-section')", show=False),
        Binding("3", "toggle_section('positions-section')", show=False),
        Binding("4", "toggle_section('signals-section')", show=False),
    ]

    def __init__(
        self,
        bus: "MessageBus | None" = None,
        name: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.bus = bus
        self._strategies: dict[str, StrategyInfo] = {}
        self._selected_strategy_id: str | None = None

        # Initialize with demo strategies
        self._init_demo_strategies()

    def _init_demo_strategies(self) -> None:
        """Initialize with demo strategies for testing."""
        demo_strategies = [
            StrategyInfo(
                strategy_id="sma_cross_btc",
                name="SMA_Cross_BTC",
                status="RUNNING",
                symbols=["BTC/USDT", "ETH/USDT"],
                positions={
                    "BTC/USDT": PositionInfo(
                        symbol="BTC/USDT",
                        side="LONG",
                        size=Decimal("0.1"),
                        entry_price=Decimal("42500"),
                        current_price=Decimal("43200"),
                        unrealized_pnl=Decimal("70"),
                    ),
                },
                total_pnl=Decimal("1250.50"),
                total_trades=42,
                win_rate=65.2,
            ),
            StrategyInfo(
                strategy_id="rsi_mean_revert",
                name="RSI_Mean_Revert",
                status="RUNNING",
                symbols=["ETH/USDT", "SOL/USDT"],
                positions={},
                total_pnl=Decimal("340.25"),
                total_trades=28,
                win_rate=57.1,
            ),
            StrategyInfo(
                strategy_id="macd_momentum",
                name="MACD_Momentum",
                status="STOPPED",
                symbols=["BTC/USDT"],
                positions={},
                total_pnl=Decimal("-120.00"),
                total_trades=15,
                win_rate=40.0,
            ),
            StrategyInfo(
                strategy_id="funding_arb",
                name="Funding_Arb",
                status="PAUSED",
                symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                positions={},
                total_pnl=Decimal("0"),
                total_trades=0,
                win_rate=0.0,
            ),
        ]

        for strategy in demo_strategies:
            self._strategies[strategy.strategy_id] = strategy

    def compose(self) -> ComposeResult:
        yield StrategyListPanel(
            strategies=list(self._strategies.values()),
            id="list-panel",
        )
        yield StrategyDetailPanel(id="detail-panel")

    def on_strategy_tree_strategy_selected(self, event: StrategyTree.StrategySelected) -> None:
        """Handle strategy selection from tree."""
        self._select_strategy(event.strategy_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-new-strategy":
            self.action_new_strategy()
        elif button_id == "btn-start":
            self.action_start_strategy()
        elif button_id == "btn-stop":
            self.action_stop_strategy()
        elif button_id == "btn-pause":
            self.action_pause_strategy()
        elif button_id == "btn-edit":
            self.action_edit_strategy()
        elif button_id == "btn-delete":
            self.action_delete_strategy()

    def _select_strategy(self, strategy_id: str) -> None:
        """Select and display a strategy."""
        if strategy_id in self._strategies:
            self._selected_strategy_id = strategy_id
            strategy = self._strategies[strategy_id]

            detail_panel = self.query_one("#detail-panel", StrategyDetailPanel)
            detail_panel.show_strategy(strategy)

    def action_new_strategy(self) -> None:
        """Open the new strategy modal."""
        self.app.push_screen(
            StrategyCreateModal(),
            callback=self._on_strategy_created,
        )

    def _on_strategy_created(self, result: StrategyEditResult) -> None:
        """Handle new strategy creation."""
        if result.saved and result.parameters:
            # Create new strategy info
            new_strategy = StrategyInfo(
                strategy_id=result.strategy_id,
                name=result.strategy_name,
                status="STOPPED",
                symbols=[result.parameters.get("symbol", "BTC/USDT")],
                positions={},
                total_pnl=Decimal("0"),
                total_trades=0,
                win_rate=0.0,
            )

            self._strategies[new_strategy.strategy_id] = new_strategy

            # Update tree
            try:
                tree = self.query_one("#strategy-tree", StrategyTree)
                tree.add_strategy(new_strategy)
            except Exception:
                pass

            self.notify(f"Created strategy: {result.strategy_name}")

    def action_edit_strategy(self) -> None:
        """Open edit modal for selected strategy."""
        if not self._selected_strategy_id:
            self.notify("No strategy selected", severity="warning")
            return

        strategy = self._strategies.get(self._selected_strategy_id)
        if not strategy:
            return

        self.app.push_screen(
            StrategyEditModal(
                strategy_id=strategy.strategy_id,
                strategy_name=strategy.name,
                parameters=create_sma_cross_parameters(),
            ),
            callback=self._on_strategy_edited,
        )

    def _on_strategy_edited(self, result: StrategyEditResult) -> None:
        """Handle strategy edit."""
        if result.saved and result.parameters:
            # Update strategy
            if result.strategy_id in self._strategies:
                strategy = self._strategies[result.strategy_id]
                # Update parameters would go here

                # Refresh tree
                try:
                    tree = self.query_one("#strategy-tree", StrategyTree)
                    tree.update_strategy(strategy)
                except Exception:
                    pass

                self.notify(f"Updated strategy: {result.strategy_name}")

    def action_delete_strategy(self) -> None:
        """Delete selected strategy with confirmation."""
        if not self._selected_strategy_id:
            self.notify("No strategy selected", severity="warning")
            return

        strategy = self._strategies.get(self._selected_strategy_id)
        if not strategy:
            return

        self.app.push_screen(
            ConfirmationModal(
                title="Delete Strategy",
                message=f"Are you sure you want to delete '{strategy.name}'?",
                confirm_label="Delete",
                cancel_label="Cancel",
            ),
            callback=self._on_delete_confirmed,
        )

    def _on_delete_confirmed(self, confirmed: bool) -> None:
        """Handle delete confirmation."""
        if confirmed and self._selected_strategy_id:
            strategy = self._strategies.pop(self._selected_strategy_id, None)
            if strategy:
                # Update tree
                try:
                    tree = self.query_one("#strategy-tree", StrategyTree)
                    tree.remove_strategy(strategy.strategy_id)
                except Exception:
                    pass

                # Clear detail panel
                detail_panel = self.query_one("#detail-panel", StrategyDetailPanel)
                detail_panel.clear_selection()

                self._selected_strategy_id = None
                self.notify(f"Deleted strategy: {strategy.name}")

    def action_start_strategy(self) -> None:
        """Start the selected strategy."""
        self._set_strategy_status("RUNNING")

    def action_stop_strategy(self) -> None:
        """Stop the selected strategy."""
        self._set_strategy_status("STOPPED")

    def action_pause_strategy(self) -> None:
        """Pause the selected strategy."""
        self._set_strategy_status("PAUSED")

    def _set_strategy_status(self, status: str) -> None:
        """Update strategy status."""
        if not self._selected_strategy_id:
            self.notify("No strategy selected", severity="warning")
            return

        if self._selected_strategy_id in self._strategies:
            strategy = self._strategies[self._selected_strategy_id]
            strategy.status = status

            # Update tree
            try:
                tree = self.query_one("#strategy-tree", StrategyTree)
                tree.update_strategy(strategy)
            except Exception:
                pass

            # Update detail panel
            try:
                detail_panel = self.query_one("#detail-panel", StrategyDetailPanel)
                header = detail_panel.query_one("#detail-header", StrategyDetailHeader)
                header.status = status

                control_bar = detail_panel.query_one("#control-bar", StrategyControlBar)
                control_bar.set_status(status)
            except Exception:
                pass

            self.notify(f"Strategy {strategy.name}: {status}")

    def action_focus_filter(self) -> None:
        """Focus the filter input."""
        try:
            filter_input = self.query_one("#strategy-filter", Input)
            filter_input.focus()
        except Exception:
            pass

    def action_toggle_section(self, section_id: str) -> None:
        """Toggle a collapsible section."""
        try:
            section = self.query_one(f"#{section_id}", Collapsible)
            section.collapsed = not section.collapsed
        except Exception:
            pass

    def action_refresh(self) -> None:
        """Refresh strategy data."""
        # In a real implementation, this would fetch from the kernel
        self.notify("Refreshed strategy data")

    def action_cursor_down(self) -> None:
        """Move cursor down in tree."""
        try:
            tree = self.query_one("#strategy-tree", StrategyTree)
            tree.action_cursor_down()
        except Exception:
            pass

    def action_cursor_up(self) -> None:
        """Move cursor up in tree."""
        try:
            tree = self.query_one("#strategy-tree", StrategyTree)
            tree.action_cursor_up()
        except Exception:
            pass

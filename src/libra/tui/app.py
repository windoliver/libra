"""
LIBRA Trading Terminal - Professional TUI Application.

A Textual-based terminal UI inspired by:
- Bloomberg Terminal (command-driven, information-dense)
- Lazygit/Lazydocker (vim-style navigation, keyboard-first)
- Dolphie (real-time monitoring, panel layout)
- Harlequin (professional SQL IDE polish)

Ergonomic Features (based on 2025 best practices):
- Vim-style navigation (j/k/h/l/g/G)
- ? key for instant help overlay
- Command palette with fuzzy search (Ctrl+P)
- User-configurable keymaps
- Alert throttling to prevent fatigue
- CVD-friendly icons alongside colors
"""

from __future__ import annotations

import random
import time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Iterable

from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Digits,
    Footer,
    Header,
    Rule,
    Static,
    TabbedContent,
    TabPane,
)

from libra.tui.demo_execution_client import DemoExecutionClient
from libra.tui.demo_trader import DemoTrader, TradingState as DemoTradingState
from libra.tui.screens import HelpScreen, HistoryScreen, OrderEntryResult, OrderEntryScreen
from libra.tui.screens.strategy_management import (
    StrategyDetailPanel,
    StrategyListPanel,
)
from libra.tui.widgets import (
    AlgorithmExecutionData,
    AlgorithmMonitor,
    BacktestResultsDashboard,
    BalanceDisplay,
    CommandInput,
    EnhancedPositionsTable,
    LogViewer,
    PortfolioDashboard,
    PositionData,
    PositionDisplay,
    PositionInfo,
    RiskDashboard,
    StatusBar,
    StrategyInfo,
    StrategyTree,
    create_demo_backtest_results,
    create_demo_positions,
    # Observability widgets (Issue #25)
    MetricsDashboard,
    TraceViewer,
    HealthMonitorWidget,
    create_demo_metrics_data,
    create_demo_trace_data,
    create_demo_health_data,
)
from libra.tui.screens.position_detail import (
    ClosePositionModal,
    PositionActionResult,
    PositionDetailModal,
)


if TYPE_CHECKING:
    from libra.core.events import Event
    from libra.core.message_bus import MessageBus
    from libra.gateways.protocol import Gateway


# CVD-friendly icons for profit/loss (colorblind accessible)
ICON_UP = "▲"
ICON_DOWN = "▼"
ICON_NEUTRAL = "●"


class PnLDisplay(Static):
    """Large P&L display using Digits widget with CVD-friendly icon."""

    DEFAULT_CSS = """
    PnLDisplay {
        height: auto;
        width: 100%;
        padding: 0 1;
        background: $surface;
        border: round $primary-darken-1;
    }

    PnLDisplay > Static {
        height: 1;
        color: $text-muted;
        text-style: bold;
    }

    PnLDisplay > Digits {
        width: 100%;
        color: $success;
    }

    PnLDisplay > Digits.negative {
        color: $error;
    }
    """

    def __init__(self, id: str = "pnl-display") -> None:
        super().__init__(id=id)
        self._value = Decimal("0.00")

    def compose(self) -> ComposeResult:
        yield Static("TOTAL P&L")
        yield Digits("+$0.00", id="pnl-digits")

    def update_pnl(self, value: Decimal) -> None:
        """Update the P&L display with CVD-friendly icon."""
        self._value = value
        digits = self.query_one("#pnl-digits", Digits)

        # Add icon for colorblind accessibility
        if value > 0:
            icon = ICON_UP
            sign = "+"
        elif value < 0:
            icon = ICON_DOWN
            sign = ""
        else:
            icon = ICON_NEUTRAL
            sign = ""

        digits.update(f"{icon}{sign}${value:,.2f}")

        digits.remove_class("negative")
        if value < 0:
            digits.add_class("negative")


class LibraApp(App):
    """
    LIBRA Trading Terminal - Ergonomic Professional Dashboard.

    Keyboard Shortcuts (Vim-style):
    ─────────────────────────────────────────────────────────
    NAVIGATION
      j / ↓        Move down in list
      k / ↑        Move up in list
      h / ←        Previous tab
      l / →        Next tab
      g            Go to top
      G            Go to bottom
      1-4          Switch to tab directly

    COMMANDS
      / or :       Focus command input
      Ctrl+P       Command palette (fuzzy search)
      Ctrl+C       Copy log to clipboard
      ?            Show help overlay

    ACTIONS
      q            Quit application
      F1           Help
      F2           Toggle dark/light theme
      ESC          Cancel / Close modal
    ─────────────────────────────────────────────────────────
    """

    TITLE = "LIBRA Trading Terminal"
    SUB_TITLE = "Professional Trading Dashboard"
    CSS_PATH = Path(__file__).parent / "styles.tcss"

    # Keyboard-first design
    AUTO_FOCUS = None
    ENABLE_COMMAND_PALETTE = True

    # Binding IDs for user-configurable keymaps
    BINDINGS: ClassVar[list[Binding]] = [
        # Core actions
        Binding("q", "quit", "Quit", priority=True, id="app.quit"),
        Binding("question_mark", "show_help_overlay", "?Help", id="app.help_overlay"),
        Binding("f1", "show_help_overlay", "Help", id="app.help"),
        Binding("f2", "toggle_dark", "Theme", id="app.theme"),
        Binding("ctrl+p", "command_palette", "Commands", id="app.palette"),
        Binding("ctrl+c", "copy_log", "Copy", priority=True, id="app.copy"),
        Binding("escape", "unfocus_or_quit", "Cancel", priority=True, id="app.escape"),
        # Order entry
        Binding("o", "open_order_entry", "Order", id="app.order"),
        Binding("n", "open_order_entry", "New", show=False, id="app.new_order"),
        # History screen (Issue #27)
        Binding("H", "open_history", "History", id="app.history"),
        # Command input focus
        Binding("slash", "focus_command", "/Cmd", id="cmd.focus_slash"),
        Binding("colon", "focus_command", ":Cmd", show=False, id="cmd.focus_colon"),
        # Vim-style navigation
        Binding("j", "cursor_down", "↓", show=False, id="nav.down"),
        Binding("k", "cursor_up", "↑", show=False, id="nav.up"),
        Binding("h", "previous_tab", "←Tab", show=False, id="nav.prev_tab"),
        Binding("l", "next_tab", "→Tab", show=False, id="nav.next_tab"),
        Binding("g", "cursor_top", "Top", show=False, id="nav.top"),
        Binding("G", "cursor_bottom", "Bottom", show=False, id="nav.bottom"),
        # Tab switching by number
        Binding("1", "switch_tab('dashboard')", show=False, id="tab.1"),
        Binding("2", "switch_tab('positions')", show=False, id="tab.2"),
        Binding("3", "switch_tab('orders')", show=False, id="tab.3"),
        Binding("4", "switch_tab('risk')", show=False, id="tab.4"),
        Binding("5", "switch_tab('strategies')", show=False, id="tab.5"),
        Binding("6", "switch_tab('portfolio')", show=False, id="tab.6"),
        Binding("7", "switch_tab('backtest')", show=False, id="tab.7"),
        Binding("8", "switch_tab('observability')", show=False, id="tab.8"),
        Binding("9", "switch_tab('settings')", show=False, id="tab.9"),
    ]

    def __init__(
        self,
        bus: MessageBus | None = None,
        gateway: Gateway | None = None,
        demo_mode: bool = True,
    ) -> None:
        super().__init__()
        self.bus = bus
        self.gateway = gateway
        self.demo_mode = demo_mode

        # Demo trading engine
        self.demo_trader = DemoTrader()

        # Demo execution client and engine (Issue #36)
        self._demo_execution_client: DemoExecutionClient | None = None
        self._execution_engine = None  # Lazy import to avoid circular deps

        # Legacy price references (for backward compatibility)
        self._btc_price = Decimal("51250.00")
        self._eth_price = Decimal("3045.00")
        self._sol_price = Decimal("142.50")
        self._total_pnl = Decimal("347.50")
        self._tick_count = 0

        # Alert throttling (prevent alarm fatigue)
        self._last_alert_time: float = 0
        self._alert_count: int = 0
        self._max_alerts_per_minute: int = 10

    # =========================================================================
    # System Commands (Command Palette)
    # =========================================================================

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand("Show Help", "Display keyboard shortcuts (?)", self.action_show_help_overlay)
        yield SystemCommand("Clear Log", "Clear the event log", self._clear_log)
        yield SystemCommand("Show Status", "Display system status", self._show_status)
        yield SystemCommand("Toggle Theme", "Switch dark/light mode", self.action_toggle_dark)
        yield SystemCommand("Copy Log", "Copy log to clipboard", self.action_copy_log)

    # =========================================================================
    # Layout
    # =========================================================================

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusBar()

        # Tabbed content for different views
        with TabbedContent(initial="dashboard", id="main-tabs"):
            with TabPane("Dashboard", id="dashboard"):
                with Horizontal(id="main-panels"):
                    yield BalanceDisplay()
                    yield PositionDisplay()

            with TabPane("Positions", id="positions"):
                with VerticalScroll():
                    yield Static("Position Details", classes="panel-title")
                    yield Rule()
                    yield PositionDisplay(id="positions-detail")

            with TabPane("Orders", id="orders"):
                with VerticalScroll():
                    # Algorithm executions monitor (Issue #36)
                    yield AlgorithmMonitor(id="algo-monitor")
                    yield Rule()
                    yield Static("Order History", classes="panel-title")
                    yield Rule()
                    yield Static("[dim]No pending orders[/dim]", id="orders-placeholder")

            with TabPane("Risk", id="risk"):
                with VerticalScroll():
                    yield RiskDashboard(
                        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                        id="risk-dashboard",
                    )

            with TabPane("Strategies", id="strategies"):
                with Horizontal(id="strategy-panels"):
                    yield StrategyListPanel(
                        strategies=self._get_demo_strategies(),
                        id="strategy-list-panel",
                    )
                    yield StrategyDetailPanel(id="strategy-detail-panel")

            with TabPane("Portfolio", id="portfolio"):
                with VerticalScroll():
                    yield PortfolioDashboard(id="portfolio-dashboard")
                    yield EnhancedPositionsTable(
                        positions=create_demo_positions(),
                        id="enhanced-positions",
                    )

            with TabPane("Backtest", id="backtest"):
                with VerticalScroll():
                    yield BacktestResultsDashboard(
                        data=create_demo_backtest_results(),
                        id="backtest-dashboard",
                    )

            # Observability tab (Issue #25)
            with TabPane("Observe", id="observability"):
                with VerticalScroll():
                    yield HealthMonitorWidget(id="health-monitor")
                    yield Rule()
                    yield MetricsDashboard(id="metrics-dashboard")

            with TabPane("Settings", id="settings"):
                with VerticalScroll():
                    yield Static("Settings", classes="panel-title")
                    yield Rule()
                    yield Static("[dim]Settings coming soon...[/dim]")

        yield LogViewer()
        yield CommandInput()
        yield Footer()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def on_mount(self) -> None:
        """Initialize application."""
        # Load user keymaps if available
        self._load_user_keymap()

        if self.demo_mode or (self.gateway is None and self.bus is None):
            self._setup_demo_mode()
        else:
            self._setup_live_mode()

    def _load_user_keymap(self) -> None:
        """Load user-configurable keymaps from config file."""
        config_path = Path.home() / ".config" / "libra" / "keymaps.toml"
        if not config_path.exists():
            return

        try:
            import tomllib
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            if "keymaps" in config:
                self.set_keymap(config["keymaps"])
                self.query_one(LogViewer).log_message(
                    f"Loaded keymaps from {config_path}", "debug"
                )
        except Exception as e:
            self.query_one(LogViewer).log_message(
                f"Failed to load keymaps: {e}", "warning"
            )

    def _setup_demo_mode(self) -> None:
        """Initialize demo with simulated data."""
        self.query_one(StatusBar).set_status(connected=True, gateway_name="Paper Trading (Demo)")
        self._sync_balance_from_demo()
        self._update_positions()

        log = self.query_one(LogViewer)
        log.log_message("LIBRA Trading Terminal v0.1.0", "success")
        log.log_message("Demo Mode: Realistic trading simulation", "info")
        log.log_message("Press ? for help, o to place orders", "info")

        # Setup demo trader callbacks
        self.demo_trader.on_trade = self._on_demo_trade
        self.demo_trader.on_risk_event = self._on_demo_risk_event

        # Initialize execution engine with demo client (Issue #36)
        self._setup_execution_engine()

        self._throttled_notify(
            "Demo Mode Active - Trade with simulated funds",
            title="Welcome to LIBRA",
            severity="information",
            timeout=3,
        )

        # Timers for simulation
        self.set_interval(0.5, self._simulate_tick)  # Faster price updates
        self.set_interval(2.0, self._update_positions)
        self.set_interval(1.0, self._update_risk_dashboard)
        self.set_interval(5.0, self._maybe_auto_trade)  # Auto trades every 5s
        self.set_interval(2.0, self._update_observability)  # Observability updates (Issue #25)

    def _setup_live_mode(self) -> None:
        """Setup for live trading mode."""
        if self.gateway:
            self.query_one(StatusBar).set_status(
                connected=self.gateway.is_connected,
                gateway_name=self.gateway.name,
            )

        if self.bus:
            self._subscribe_to_events()

        self.set_interval(2.0, self._refresh_account_data)
        self.set_interval(1.0, self._refresh_status)

    def _setup_execution_engine(self) -> None:
        """Initialize the execution engine for algorithm-based orders (Issue #36)."""
        try:
            from libra.execution import create_execution_engine

            # Create demo execution client wrapping demo trader
            self._demo_execution_client = DemoExecutionClient(self.demo_trader)

            # Create execution engine
            self._execution_engine = create_execution_engine(
                execution_client=self._demo_execution_client,
                enable_risk_checks=False,  # Demo mode doesn't need risk checks
            )

            self.query_one(LogViewer).log_message(
                "Execution engine initialized (TWAP, VWAP, Iceberg, POV available)",
                "debug",
            )
        except Exception as e:
            self.query_one(LogViewer).log_message(
                f"Failed to initialize execution engine: {e}",
                "warning",
            )
            self._execution_engine = None

    def _execute_with_algorithm(self, result: OrderEntryResult, log: LogViewer) -> None:
        """Execute an order using an execution algorithm (Issue #36)."""
        import asyncio
        from libra.gateways.protocol import Order, OrderSide, OrderType

        # Build the order
        side = OrderSide.BUY if result.side == "BUY" else OrderSide.SELL
        order_type = OrderType.MARKET if result.order_type == "MARKET" else OrderType.LIMIT

        order = Order(
            symbol=result.symbol,
            side=side,
            order_type=order_type,
            amount=result.quantity,
            price=result.price,
            exec_algorithm=result.exec_algorithm,
            exec_algorithm_params=result.exec_algorithm_params,
        )

        algo_name = result.exec_algorithm.upper() if result.exec_algorithm else "UNKNOWN"
        log.log_message(
            f"Starting {algo_name} execution: {result.side} {result.quantity} {result.symbol}",
            "info",
        )

        # Update algorithm monitor widget if present
        try:
            monitor = self.query_one("#algo-monitor", AlgorithmMonitor)
            monitor.update_execution(AlgorithmExecutionData(
                execution_id=order.client_order_id or f"{result.symbol}-algo",
                algorithm=result.exec_algorithm or "unknown",
                symbol=result.symbol,
                side=result.side,
                total_quantity=result.quantity,
                state="RUNNING",
            ))
        except Exception:
            pass

        async def run_algo():
            try:
                progress = await self._execution_engine.submit_order(order)

                # Log completion
                if progress and hasattr(progress, 'state'):
                    state = progress.state.value if hasattr(progress.state, 'value') else str(progress.state)
                    log.log_message(
                        f"{algo_name} execution {state}: "
                        f"Filled {progress.executed_quantity}/{progress.total_quantity}",
                        "success" if state == "completed" else "warning",
                    )

                    # Update monitor
                    try:
                        monitor = self.query_one("#algo-monitor", AlgorithmMonitor)
                        monitor.update_execution(AlgorithmExecutionData(
                            execution_id=order.client_order_id or f"{result.symbol}-algo",
                            algorithm=result.exec_algorithm or "unknown",
                            symbol=result.symbol,
                            side=result.side,
                            total_quantity=progress.total_quantity,
                            filled_quantity=progress.executed_quantity,
                            state=state.upper(),
                            total_slices=progress.num_children_spawned,
                            completed_slices=progress.num_children_filled,
                            average_price=progress.avg_fill_price,
                        ))
                    except Exception:
                        pass

                    self._throttled_notify(
                        f"{algo_name} {result.side} {result.quantity} {result.symbol}",
                        title=f"Algo {state.title()}",
                        severity="information",
                        timeout=3,
                    )
            except Exception as e:
                log.log_message(f"{algo_name} execution failed: {e}", "error")
                self._throttled_notify(
                    f"{algo_name} failed: {e}",
                    title="Algorithm Error",
                    severity="error",
                    timeout=5,
                )

        # Run the algorithm in the background
        asyncio.create_task(run_algo())

    # =========================================================================
    # Alert Throttling (Prevent Alarm Fatigue)
    # =========================================================================

    def _throttled_notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float = 5,
    ) -> bool:
        """
        Show notification with throttling to prevent alarm fatigue.

        Returns True if notification was shown, False if throttled.
        """
        current_time = time.time()

        # Reset counter every minute
        if current_time - self._last_alert_time > 60:
            self._alert_count = 0

        # Check if we've exceeded the limit
        if self._alert_count >= self._max_alerts_per_minute:
            return False

        self._alert_count += 1
        self._last_alert_time = current_time

        self.notify(message, title=title, severity=severity, timeout=timeout)
        return True

    # =========================================================================
    # Vim-style Navigation Actions
    # =========================================================================

    def action_cursor_down(self) -> None:
        """Move cursor down (j key - vim style)."""
        try:
            table = self.query_one(DataTable)
            table.action_cursor_down()
        except Exception:
            pass

    def action_cursor_up(self) -> None:
        """Move cursor up (k key - vim style)."""
        try:
            table = self.query_one(DataTable)
            table.action_cursor_up()
        except Exception:
            pass

    def action_cursor_top(self) -> None:
        """Move cursor to top (g key - vim style)."""
        try:
            table = self.query_one(DataTable)
            table.action_scroll_top()
        except Exception:
            pass

    def action_cursor_bottom(self) -> None:
        """Move cursor to bottom (G key - vim style)."""
        try:
            table = self.query_one(DataTable)
            table.action_scroll_bottom()
        except Exception:
            pass

    def action_previous_tab(self) -> None:
        """Switch to previous tab (h key - vim style)."""
        try:
            tabs = self.query_one(TabbedContent)
            tabs.action_previous_tab()
        except Exception:
            pass

    def action_next_tab(self) -> None:
        """Switch to next tab (l key - vim style)."""
        try:
            tabs = self.query_one(TabbedContent)
            tabs.action_next_tab()
        except Exception:
            pass

    def action_switch_tab(self, tab: str) -> None:
        """Switch to a specific tab by ID."""
        try:
            self.query_one(TabbedContent).active = tab
        except Exception:
            pass

    # =========================================================================
    # Core Actions
    # =========================================================================

    def action_show_help_overlay(self) -> None:
        """Show the help overlay modal (? key)."""
        self.push_screen(HelpScreen())

    def action_open_history(self) -> None:
        """Open the order/trade history screen (Shift+H key, Issue #27)."""
        self.push_screen(HistoryScreen(gateway=self.gateway))

    def action_open_order_entry(self) -> None:
        """Open the order entry modal (o key)."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        current_prices = {
            "BTC/USDT": self._btc_price,
            "ETH/USDT": self._eth_price,
            "SOL/USDT": self._sol_price,
        }

        def handle_order_result(result: OrderEntryResult | None) -> None:
            """Handle the result from order entry screen."""
            if result is None or not result.submitted:
                self.query_one(LogViewer).log_message("Order cancelled", "debug")
                return

            log = self.query_one(LogViewer)

            # Execute via DemoTrader if in demo mode
            if self.demo_mode:
                # Check if execution algorithm is specified (Issue #36)
                if result.exec_algorithm and self._execution_engine is not None:
                    # Route through ExecutionEngine for algorithm execution
                    self._execute_with_algorithm(result, log)
                else:
                    # Direct execution through DemoTrader
                    success, msg = self.demo_trader.execute_order(
                        symbol=result.symbol,
                        side=result.side,
                        quantity=result.quantity,
                        price=result.price,
                    )

                    if success:
                        icon = ICON_UP if result.side == "BUY" else ICON_DOWN
                        log.log_message(f"{icon} {msg}", "success")
                        self._throttled_notify(
                            f"{result.side} {result.quantity} {result.symbol}",
                            title="Order Executed",
                            severity="information",
                            timeout=3,
                        )
                    else:
                        log.log_message(f"ORDER REJECTED: {msg}", "error")
                        self._throttled_notify(msg, title="Order Rejected", severity="error", timeout=5)
            else:
                # Live mode - just log (would send to gateway)
                log.log_message(
                    f"{ICON_UP if result.side == 'BUY' else ICON_DOWN} ORDER {result.side} "
                    f"{result.quantity} {result.symbol} @ "
                    f"{'MARKET' if result.order_type == 'MARKET' else f'${result.price:,.2f}'}",
                    "success",
                )
                self._throttled_notify(
                    f"{result.side} {result.quantity} {result.symbol}",
                    title="Order Submitted",
                    severity="information",
                    timeout=3,
                )

        self.push_screen(
            OrderEntryScreen(
                symbols=symbols,
                current_prices=current_prices,
            ),
            handle_order_result,
        )

    def action_focus_command(self) -> None:
        """Focus the command input (/ or :)."""
        self.call_after_refresh(lambda: self.query_one(CommandInput).focus())

    def action_unfocus_or_quit(self) -> None:
        """Unfocus current widget or close modal (ESC)."""
        if self.screen.is_modal:
            self.screen.dismiss()
        elif self.focused:
            self.focused.blur()

    def action_copy_log(self) -> None:
        """Copy log to clipboard with feedback."""
        log_viewer = self.query_one(LogViewer)
        text = log_viewer.get_log_text(max_lines=50)

        try:
            self.copy_to_clipboard(text)
            self._throttled_notify("Log copied!", severity="information", timeout=2)
            return
        except Exception:
            pass

        try:
            import pyperclip
            pyperclip.copy(text)
            self._throttled_notify("Log copied!", severity="information", timeout=2)
        except Exception as e:
            self._throttled_notify(f"Copy failed: {e}", severity="error")

    async def action_quit(self) -> None:
        """Graceful quit with cleanup."""
        log = self.query_one(LogViewer)
        log.log_message("Shutting down...", "warning")

        if self.bus:
            await self.bus.stop(drain=True)

        if self.gateway and self.gateway.is_connected:
            await self.gateway.disconnect()

        self.exit()

    # =========================================================================
    # Strategy Management Event Handlers
    # =========================================================================

    def on_strategy_tree_strategy_selected(
        self, event: StrategyTree.StrategySelected
    ) -> None:
        """Handle strategy selection from tree."""
        strategy_id = event.strategy_id
        strategies = {s.strategy_id: s for s in self._get_demo_strategies()}

        if strategy_id in strategies:
            strategy = strategies[strategy_id]
            try:
                detail_panel = self.query_one("#strategy-detail-panel", StrategyDetailPanel)
                detail_panel.show_strategy(strategy)
            except Exception as e:
                self.query_one(LogViewer).log_message(f"Error showing strategy: {e}", "error")

    # =========================================================================
    # Position Management Event Handlers
    # =========================================================================

    def on_enhanced_positions_table_position_selected(
        self, event: EnhancedPositionsTable.PositionSelected
    ) -> None:
        """Handle position selection from enhanced positions table."""
        position = event.position

        def handle_position_action(result: PositionActionResult | None) -> None:
            if result is None or result.action == "none":
                return
            if result.action == "close":
                self._request_close_position(position)
            elif result.action in ("modify_sl", "modify_tp"):
                self.query_one(LogViewer).log_message(
                    f"Modify SL/TP requested for {position.symbol}", "info"
                )

        self.push_screen(PositionDetailModal(position), handle_position_action)

    def on_enhanced_positions_table_position_close_requested(
        self, event: EnhancedPositionsTable.PositionCloseRequested
    ) -> None:
        """Handle close position request from enhanced positions table."""
        self._request_close_position(event.position)

    def _request_close_position(self, position: PositionData) -> None:
        """Request to close a position with confirmation."""

        def handle_confirmation(confirmed: bool | None) -> None:
            if not confirmed:
                return
            log = self.query_one(LogViewer)
            log.log_message(
                f"Closing position: {position.symbol} {position.side} {position.size}",
                "warning",
            )
            # In demo mode, we'd actually close via demo_trader
            if self.demo_mode:
                side = "SELL" if position.side == "LONG" else "BUY"
                success, msg = self.demo_trader.execute_order(
                    symbol=position.symbol,
                    side=side,
                    quantity=position.size,
                )
                if success:
                    log.log_message(f"Position closed: {msg}", "success")
                    # Refresh positions table
                    self._refresh_enhanced_positions()
                else:
                    log.log_message(f"Failed to close position: {msg}", "error")

        self.push_screen(ClosePositionModal(position), handle_confirmation)

    def _refresh_enhanced_positions(self) -> None:
        """Refresh the enhanced positions table."""
        try:
            positions_table = self.query_one("#enhanced-positions", EnhancedPositionsTable)
            positions_table.update_positions(create_demo_positions())
        except Exception:
            pass

    # =========================================================================
    # Demo Simulation
    # =========================================================================

    def _get_demo_strategies(self) -> list[StrategyInfo]:
        """Generate demo strategies for the strategy management tab."""
        return [
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

    def _update_positions(self) -> None:
        """Update position tables with CVD-friendly icons."""
        for selector in [PositionDisplay, "#positions-detail"]:
            try:
                if isinstance(selector, str):
                    table = self.query_one(selector, PositionDisplay).query_one(DataTable)
                else:
                    table = self.query_one(selector).query_one(DataTable)
                table.clear()
                self._add_position_rows(table)
            except Exception:
                pass

    def _add_position_rows(self, table: DataTable) -> None:
        """Add position rows with CVD-friendly icons."""
        positions = [
            ("BTC/USDT", "LONG", "0.1000", Decimal("50000"), self._btc_price, Decimal("0.1")),
            ("ETH/USDT", "SHORT", "2.0000", Decimal("3000"), self._eth_price, Decimal("2.0")),
            ("SOL/USDT", "LONG", "10.000", Decimal("135"), self._sol_price, Decimal("10.0")),
        ]

        total_pnl = Decimal("0")
        for symbol, side, size, entry, current, amount in positions:
            if side == "LONG":
                pnl = (current - entry) * amount
                pnl_pct = ((current / entry) - 1) * 100
            else:
                pnl = (entry - current) * amount
                pnl_pct = ((entry / current) - 1) * 100

            total_pnl += pnl

            # CVD-friendly: use icons alongside colors
            side_icon = ICON_UP if side == "LONG" else ICON_DOWN
            side_color = "green" if side == "LONG" else "red"

            pnl_icon = ICON_UP if pnl >= 0 else ICON_DOWN
            pnl_color = "green" if pnl >= 0 else "red"
            pnl_sign = "+" if pnl >= 0 else ""

            table.add_row(
                symbol,
                f"[{side_color}]{side_icon} {side}[/{side_color}]",
                size,
                f"{entry:,.2f}",
                f"{current:,.2f}",
                f"[{pnl_color}]{pnl_icon} {pnl_sign}{pnl:,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pnl_sign}{pnl_pct:.2f}%[/{pnl_color}]",
            )

        self._total_pnl = total_pnl

    # =========================================================================
    # Demo Trading Integration
    # =========================================================================

    def _sync_balance_from_demo(self) -> None:
        """Sync balance display from demo trader."""
        try:
            stats = self.demo_trader.get_stats()
            balance_display = self.query_one(BalanceDisplay)
            table = balance_display.query_one(DataTable)
            table.clear()

            # Show USDT balance
            equity = stats["equity"]
            available = stats["balance"]
            in_use = stats["total_exposure"]
            pct_used = float(in_use / equity * 100) if equity > 0 else 0

            pct_color = "green" if pct_used < 50 else "yellow" if pct_used < 80 else "red"
            table.add_row(
                "USDT",
                f"{equity:,.2f}",
                f"{available:,.2f}",
                f"{in_use:,.2f}",
                f"[{pct_color}]{pct_used:.1f}%[/{pct_color}]",
            )
        except Exception:
            pass

    def _on_demo_trade(self, message: str) -> None:
        """Callback when demo trader executes a trade."""
        log = self.query_one(LogViewer)
        icon = ICON_UP if "BUY" in message else ICON_DOWN
        log.log_message(f"{icon} {message}", "success")

    def _on_demo_risk_event(self, state: str, reason: str) -> None:
        """Callback when demo trader triggers risk event."""
        log = self.query_one(LogViewer)
        log.log_message(f"⚠ RISK STATE: {state} - {reason}", "warning")
        self._throttled_notify(reason, title=f"Risk: {state}", severity="warning", timeout=5)

    def _simulate_tick(self) -> None:
        """Simulate price updates using DemoTrader."""
        self._tick_count += 1

        # Update prices via demo trader
        price_updates = self.demo_trader.tick_prices()

        # Sync to legacy price vars for compatibility
        self._btc_price = price_updates.get("BTC/USDT", self._btc_price)
        self._eth_price = price_updates.get("ETH/USDT", self._eth_price)
        self._sol_price = price_updates.get("SOL/USDT", self._sol_price)

        # Log price tick occasionally
        if self._tick_count % 4 == 0:
            log = self.query_one(LogViewer)
            symbols = [("BTC", self._btc_price), ("ETH", self._eth_price), ("SOL", self._sol_price)]
            sym, price = symbols[self._tick_count % 3]
            log.log_message(f"TICK {sym}/USDT @ ${price:,.2f}", "debug")

        # Update P&L display
        stats = self.demo_trader.get_stats()
        self._total_pnl = stats["realized_pnl"] + stats["unrealized_pnl"]

        try:
            pnl_display = self.query_one(PnLDisplay)
            pnl_display.update_pnl(self._total_pnl)
        except Exception:
            pass

    def _maybe_auto_trade(self) -> None:
        """Occasionally execute auto trades for demo."""
        if self.demo_trader.trading_state != DemoTradingState.ACTIVE:
            return

        log = self.query_one(LogViewer)

        # Pick random symbol
        symbol = random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"])  # noqa: S311

        # Smart side selection: if we have a position, 50% chance to close it
        position = self.demo_trader.positions.get(symbol)
        if position and position.side != "FLAT" and random.random() > 0.5:  # noqa: S311
            # Close existing position
            side = "SELL" if position.side == "LONG" else "BUY"
            quantity = position.quantity
        else:
            # Open new position (BUY only to avoid short-selling complexity)
            side = "BUY"
            qty_map = {"BTC/USDT": "0.02", "ETH/USDT": "0.1", "SOL/USDT": "1.0"}
            quantity = Decimal(qty_map[symbol]) * Decimal(str(random.uniform(0.5, 1.5)))  # noqa: S311

        # Execute
        success, msg = self.demo_trader.execute_order(symbol, side, quantity)

        if success:
            icon = ICON_UP if side == "BUY" else ICON_DOWN
            log.log_message(f"{icon} AUTO {msg}", "info")
        else:
            log.log_message(f"{ICON_NEUTRAL} AUTO blocked: {msg}", "debug")

    def _update_risk_dashboard(self) -> None:
        """Update risk dashboard from DemoTrader state."""
        try:
            dashboard = self.query_one("#risk-dashboard", RiskDashboard)
            stats = self.demo_trader.get_stats()

            # Trading state
            dashboard.set_trading_state(stats["trading_state"])

            # Drawdown
            dashboard.set_drawdown(stats["drawdown_pct"], self.demo_trader.max_drawdown_pct)

            # Daily P&L
            daily_pnl = float(stats["daily_pnl"])
            daily_limit = float(self.demo_trader.daily_loss_limit)
            dashboard.set_daily_pnl(daily_pnl, daily_limit)

            # Order rate
            dashboard.set_order_rate(
                self.demo_trader.orders_this_second,
                self.demo_trader.order_rate_limit,
            )
            self.demo_trader.orders_this_second = 0  # Reset per-second counter

            # Position exposures
            for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
                current, limit = self.demo_trader.get_position_exposure(symbol)
                dashboard.set_exposure(symbol, current, limit)

            # Circuit breaker
            dashboard.set_circuit_breaker(stats["circuit_breaker"])

            # Also sync balance
            self._sync_balance_from_demo()
        except Exception:
            pass

    def _update_observability(self) -> None:
        """Update observability widgets with demo data (Issue #25)."""
        try:
            # Update health monitor
            health_monitor = self.query_one("#health-monitor", HealthMonitorWidget)
            health_data = create_demo_health_data()
            health_monitor.update_health(health_data)
        except Exception:
            pass

        try:
            # Update metrics dashboard
            metrics_dashboard = self.query_one("#metrics-dashboard", MetricsDashboard)
            metrics_data = create_demo_metrics_data()
            metrics_dashboard.update_from_collector(metrics_data)
        except Exception:
            pass

    # =========================================================================
    # Live Mode Event Handling
    # =========================================================================

    def _subscribe_to_events(self) -> None:
        if not self.bus:
            return

        from libra.core.events import EventType

        handlers = {
            EventType.GATEWAY_CONNECTED: self._on_gateway_event,
            EventType.GATEWAY_DISCONNECTED: self._on_gateway_event,
            EventType.TICK: self._on_market_event,
            EventType.ORDER_FILLED: self._on_order_event,
            EventType.RISK_LIMIT_BREACH: self._on_risk_event,
        }

        for event_type, handler in handlers.items():
            self.bus.subscribe(event_type, handler)

    async def _on_gateway_event(self, event: Event) -> None:
        from libra.core.events import EventType

        connected = event.event_type == EventType.GATEWAY_CONNECTED
        gateway_name = event.payload.get("gateway", "Unknown")
        self.query_one(StatusBar).set_status(connected=connected, gateway_name=gateway_name)
        self.query_one(LogViewer).log_event(event)
        self._throttled_notify(
            f"{'Connected to' if connected else 'Disconnected from'} {gateway_name}",
            severity="information" if connected else "error",
        )

    async def _on_market_event(self, event: Event) -> None:
        self.query_one(LogViewer).log_event(event)

    async def _on_order_event(self, event: Event) -> None:
        self.query_one(LogViewer).log_event(event)
        await self._refresh_account_data()
        self._throttled_notify(
            f"Order filled: {event.payload.get('symbol', '')}",
            title="Order Update",
            severity="information",
        )

    async def _on_risk_event(self, event: Event) -> None:
        self.query_one(LogViewer).log_event(event)
        self._throttled_notify(
            event.payload.get("message", "Risk event"),
            title="Risk Alert",
            severity="warning",
            timeout=10,
        )

    async def _refresh_account_data(self) -> None:
        if not self.gateway or not self.gateway.is_connected:
            return

        try:
            balances = await self.gateway.get_balances()
            self.query_one(BalanceDisplay).update_balances(balances)

            positions = await self.gateway.get_positions()
            self.query_one(PositionDisplay).update_positions(positions)
        except Exception as e:
            self.query_one(LogViewer).log_message(f"Error: {e}", "error")

    async def _refresh_status(self) -> None:
        if self.gateway:
            self.query_one(StatusBar).set_status(
                connected=self.gateway.is_connected,
                gateway_name=self.gateway.name,
            )

    # =========================================================================
    # Command Handlers
    # =========================================================================

    def on_command_input_builtin_command(self, message: CommandInput.BuiltinCommand) -> None:
        cmd = message.command
        log = self.query_one(LogViewer)

        # Log executed command for transparency (Lazygit pattern)
        log.log_message(f"→ {cmd}", "debug")

        if cmd in ("/quit", "/exit"):
            self.exit()
        elif cmd == "/clear":
            self._clear_log()
        elif cmd == "/theme":
            self.dark = not self.dark
        elif cmd == "/help":
            self.action_show_help_overlay()
        elif cmd == "/status":
            self._show_status()

    def on_command_input_command_submitted(self, message: CommandInput.CommandSubmitted) -> None:
        log = self.query_one(LogViewer)
        log.log_message(f"→ {message.command}", "debug")
        self._throttled_notify(
            f"Unknown: {message.command}. Press ? for help",
            severity="warning",
            timeout=3,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _clear_log(self) -> None:
        self.query_one(LogViewer).clear_log()
        self._throttled_notify("Log cleared", timeout=2)

    def _show_status(self) -> None:
        log = self.query_one(LogViewer)
        status = self.query_one(StatusBar)
        log.log_message("─" * 40, "debug")
        log.log_message("SYSTEM STATUS", "success")
        log.log_message(f"  Gateway:   {status.gateway_name}", "info")
        log.log_message(f"  Connected: {'Yes' if status.connected else 'No'}", "info")
        log.log_message(f"  Mode:      {'DEMO' if self.demo_mode else 'LIVE'}", "warning")
        if self.demo_mode:
            pnl_icon = ICON_UP if self._total_pnl > 0 else ICON_DOWN
            log.log_message(f"  BTC:       ${self._btc_price:,.2f}", "info")
            log.log_message(f"  ETH:       ${self._eth_price:,.2f}", "info")
            log.log_message(
                f"  P&L:       {pnl_icon} ${self._total_pnl:+,.2f}",
                "success" if self._total_pnl > 0 else "error"
            )
        log.log_message("─" * 40, "debug")


def run_app(
    bus: MessageBus | None = None,
    gateway: Gateway | None = None,
    demo_mode: bool = True,
) -> None:
    """Run the LIBRA Trading Terminal."""
    app = LibraApp(bus=bus, gateway=gateway, demo_mode=demo_mode)
    app.run()


if __name__ == "__main__":
    run_app()

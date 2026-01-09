"""
LIBRA TUI Screens.

Modal screens and overlays for the trading terminal.
"""

from libra.tui.screens.help import HelpScreen
from libra.tui.screens.order_entry import OrderEntryResult, OrderEntryScreen
from libra.tui.screens.strategy_edit_modal import (
    ConfirmationModal,
    StrategyCreateModal,
    StrategyEditModal,
    StrategyEditResult,
)
from libra.tui.screens.strategy_management import StrategyManagementScreen
from libra.tui.screens.position_detail import (
    ClosePositionModal,
    PositionActionResult,
    PositionDetailModal,
)
from libra.tui.screens.backtest_modal import (
    BacktestConfig,
    BacktestConfigModal,
    BacktestResult,
    BacktestResultsModal,
    run_demo_backtest,
)

# History Screen (Issue #27)
from libra.tui.screens.history import HistoryScreen


__all__ = [
    # Help
    "HelpScreen",
    # Order Entry
    "OrderEntryResult",
    "OrderEntryScreen",
    # Strategy Management
    "ConfirmationModal",
    "StrategyCreateModal",
    "StrategyEditModal",
    "StrategyEditResult",
    "StrategyManagementScreen",
    # Position Detail
    "ClosePositionModal",
    "PositionActionResult",
    "PositionDetailModal",
    # Backtest
    "BacktestConfig",
    "BacktestConfigModal",
    "BacktestResult",
    "BacktestResultsModal",
    "run_demo_backtest",
    # History (Issue #27)
    "HistoryScreen",
]

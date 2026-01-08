"""
LIBRA TUI Widgets.

Reusable terminal UI components for the trading dashboard.

All widgets follow Textual best practices:
- Use DEFAULT_CSS for encapsulated styling
- Non-focusable containers to prevent focus stealing
- Proper message passing for events
"""

from libra.tui.widgets.balance_display import BalanceDisplay
from libra.tui.widgets.command_input import CommandInput
from libra.tui.widgets.log_viewer import LogViewer
from libra.tui.widgets.parameter_editor import (
    ParameterDefinition,
    ParameterEditor,
    ParameterRow,
    ParameterType,
    create_sma_cross_parameters,
)
from libra.tui.widgets.position_display import PositionDisplay
from libra.tui.widgets.risk_dashboard import (
    CircuitBreakerIndicator,
    DrawdownGauge,
    ExposureBar,
    OrderRateIndicator,
    RiskDashboard,
    TradingStateIndicator,
)
from libra.tui.widgets.status_bar import StatusBar
from libra.tui.widgets.strategy_card import StrategyCard
from libra.tui.widgets.strategy_metrics import (
    DrawdownGauge as StrategyDrawdownGauge,
    MetricCard,
    StrategyMetricsPanel,
)
from libra.tui.widgets.strategy_tree import (
    PositionInfo,
    StrategyInfo,
    StrategyListView,
    StrategyTree,
)


__all__ = [
    # Balance & Position
    "BalanceDisplay",
    "PositionDisplay",
    # Command
    "CommandInput",
    # Logging
    "LogViewer",
    # Risk Dashboard
    "CircuitBreakerIndicator",
    "DrawdownGauge",
    "ExposureBar",
    "OrderRateIndicator",
    "RiskDashboard",
    "TradingStateIndicator",
    # Status
    "StatusBar",
    # Strategy Management
    "MetricCard",
    "ParameterDefinition",
    "ParameterEditor",
    "ParameterRow",
    "ParameterType",
    "PositionInfo",
    "StrategyCard",
    "StrategyDrawdownGauge",
    "StrategyInfo",
    "StrategyListView",
    "StrategyMetricsPanel",
    "StrategyTree",
    "create_sma_cross_parameters",
]

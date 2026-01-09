"""
LIBRA TUI Widgets.

Reusable terminal UI components for the trading dashboard.

All widgets follow Textual best practices:
- Use DEFAULT_CSS for encapsulated styling
- Non-focusable containers to prevent focus stealing
- Proper message passing for events
"""

# Algorithm Monitor (Issue #36)
from libra.tui.widgets.algo_monitor import (
    AlgorithmExecutionData,
    AlgorithmMonitor,
    ExecutionCard,
)
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
from libra.tui.widgets.enhanced_positions import (
    CompactPositionsTable,
    EnhancedPositionsTable,
    PositionData,
    PositionRow,
    SortablePositionsTable,
    create_demo_positions,
)
from libra.tui.widgets.portfolio_dashboard import (
    AllocationBar,
    AssetAllocationTable,
    AssetHolding,
    DailyPnLCard,
    PeriodReturns,
    PortfolioDashboard,
    PortfolioData,
    TotalValueCard,
)

# Backtest Results Dashboard (Issue #40)
from libra.tui.widgets.equity_chart import (
    DrawdownChart,
    EquityCurveChart,
    EquityCurveData,
    EquityPoint,
    EquitySummary,
    TradeMarker,
    create_demo_equity_data,
)
from libra.tui.widgets.backtest_metrics import (
    BacktestMetrics,
    BacktestMetricsPanel,
    MetricThresholds,
    ReturnDisplay,
    RiskMetricsPanel,
    TradeStatsPanel,
    create_demo_metrics,
)
from libra.tui.widgets.trade_history import (
    CollapsibleTradeDetails,
    FilterSide,
    SortColumn,
    SortDirection,
    TradeDetailPanel,
    TradeHistoryTable,
    TradeRecord,
    TradeSide,
    create_demo_trades,
)
from libra.tui.widgets.backtest_dashboard import (
    BacktestResultsData,
    BacktestResultsDashboard,
    BacktestResultsScreen,
    convert_from_backtest_result,
    create_demo_backtest_results,
)

# Order/Fill History (Issue #27)
from libra.tui.widgets.order_history import (
    FillHistoryTable,
    OrderHistoryTable,
    OrderStatusFilter,
)

# Strategy Signal Log (Issue #43)
from libra.tui.widgets.signal_log import (
    Signal,
    SignalType,
    StrategySignalLog,
    create_demo_signals,
)


__all__ = [
    # Algorithm Monitor (Issue #36)
    "AlgorithmExecutionData",
    "AlgorithmMonitor",
    "ExecutionCard",
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
    # Enhanced Positions
    "CompactPositionsTable",
    "EnhancedPositionsTable",
    "PositionData",
    "PositionRow",
    "create_demo_positions",
    "SortablePositionsTable",
    # Portfolio Dashboard
    "AllocationBar",
    "AssetAllocationTable",
    "AssetHolding",
    "DailyPnLCard",
    "PeriodReturns",
    "PortfolioDashboard",
    "PortfolioData",
    "TotalValueCard",
    # Equity Chart (Issue #40)
    "DrawdownChart",
    "EquityCurveChart",
    "EquityCurveData",
    "EquityPoint",
    "EquitySummary",
    "TradeMarker",
    "create_demo_equity_data",
    # Backtest Metrics (Issue #40)
    "BacktestMetrics",
    "BacktestMetricsPanel",
    "MetricThresholds",
    "ReturnDisplay",
    "RiskMetricsPanel",
    "TradeStatsPanel",
    "create_demo_metrics",
    # Trade History (Issue #40)
    "CollapsibleTradeDetails",
    "FilterSide",
    "SortColumn",
    "SortDirection",
    "TradeDetailPanel",
    "TradeHistoryTable",
    "TradeRecord",
    "TradeSide",
    "create_demo_trades",
    # Backtest Dashboard (Issue #40)
    "BacktestResultsData",
    "BacktestResultsDashboard",
    "BacktestResultsScreen",
    "convert_from_backtest_result",
    "create_demo_backtest_results",
    # Order/Fill History (Issue #27)
    "FillHistoryTable",
    "OrderHistoryTable",
    "OrderStatusFilter",
    # Strategy Signal Log (Issue #43)
    "Signal",
    "SignalType",
    "StrategySignalLog",
    "create_demo_signals",
]

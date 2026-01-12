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

# Observability Widgets (Issue #25)
from libra.tui.widgets.metrics_dashboard import (
    CounterPanel,
    GaugePanel,
    HistogramPanel,
    MetricCard,
    MetricData,
    MetricsDashboard,
    MetricsDashboardData,
    create_demo_metrics_data,
)
from libra.tui.widgets.trace_viewer import (
    SpanData,
    SpanDetailPanel,
    SpanTreePanel,
    TraceData,
    TraceListPanel,
    TraceViewer,
    create_demo_trace_data,
)
from libra.tui.widgets.health_monitor import (
    ComponentHealthData,
    ComponentHealthTable,
    HealthMonitorWidget,
    HealthStatusBadge,
    ResourceGauge,
    SystemHealthData,
    SystemResourcesPanel,
    create_demo_health_data,
)

# Prediction Market Dashboard (Issue #39)
from libra.tui.widgets.prediction_market_dashboard import (
    MarketDisplayData,
    MarketSearchInput,
    MarketsTable,
    PositionDisplayData,
    PositionsTable,
    PredictionMarketDashboard,
    PredictionMarketDashboardData,
    ProviderStatusBar,
    create_demo_prediction_markets,
)

# Whale Alerts Dashboard (Issue #38)
from libra.tui.widgets.whale_alerts import (
    AlertDetailPanel,
    AlertSummaryCard,
    SourceStatusBar as WhaleSourceStatusBar,
    WhaleAlertData,
    WhaleAlertsDashboard,
    WhaleAlertsDashboardData,
    WhaleAlertsTable,
    create_demo_whale_alerts,
)

# Funding Rate Dashboard (Issue #13)
from libra.tui.widgets.funding_rate_dashboard import (
    ArbitragePositionDisplayData,
    ArbitragePositionsTable,
    ExchangeStatusBar as FundingExchangeStatusBar,
    FundingRateDashboard,
    FundingRateDashboardData,
    FundingRateDisplayData,
    FundingRatesTable,
    FundingRateSummary,
    OpportunityCard,
    create_demo_funding_dashboard_data,
)

# Audit Dashboard (Issue #16)
from libra.tui.widgets.audit_dashboard import (
    AuditDashboard,
    create_demo_audit_dashboard,
)

# Risk Analytics Dashboard (Issue #15)
from libra.tui.widgets.risk_analytics_dashboard import (
    ConcentrationPanel,
    CorrelationDisplayData,
    CorrelationPanel,
    MarginDisplayData,
    MarginPositionsTable,
    MarginUtilizationPanel,
    RiskAnalyticsDashboard,
    RiskAnalyticsDashboardData,
    StressTestDisplayData,
    StressTestPanel,
    VaRDisplayData,
    VaRPanel,
    create_demo_risk_analytics_data,
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
    # Observability Widgets (Issue #25)
    "CounterPanel",
    "GaugePanel",
    "HistogramPanel",
    "MetricCard",
    "MetricData",
    "MetricsDashboard",
    "MetricsDashboardData",
    "create_demo_metrics_data",
    "SpanData",
    "SpanDetailPanel",
    "SpanTreePanel",
    "TraceData",
    "TraceListPanel",
    "TraceViewer",
    "create_demo_trace_data",
    "ComponentHealthData",
    "ComponentHealthTable",
    "HealthMonitorWidget",
    "HealthStatusBadge",
    "ResourceGauge",
    "SystemHealthData",
    "SystemResourcesPanel",
    "create_demo_health_data",
    # Prediction Market Dashboard (Issue #39)
    "MarketDisplayData",
    "MarketSearchInput",
    "MarketsTable",
    "PositionDisplayData",
    "PositionsTable",
    "PredictionMarketDashboard",
    "PredictionMarketDashboardData",
    "ProviderStatusBar",
    "create_demo_prediction_markets",
    # Whale Alerts Dashboard (Issue #38)
    "AlertDetailPanel",
    "AlertSummaryCard",
    "WhaleSourceStatusBar",
    "WhaleAlertData",
    "WhaleAlertsDashboard",
    "WhaleAlertsDashboardData",
    "WhaleAlertsTable",
    "create_demo_whale_alerts",
    # Funding Rate Dashboard (Issue #13)
    "ArbitragePositionDisplayData",
    "ArbitragePositionsTable",
    "FundingExchangeStatusBar",
    "FundingRateDashboard",
    "FundingRateDashboardData",
    "FundingRateDisplayData",
    "FundingRatesTable",
    "FundingRateSummary",
    "OpportunityCard",
    "create_demo_funding_dashboard_data",
    # Audit Dashboard (Issue #16)
    "AuditDashboard",
    "create_demo_audit_dashboard",
    # Risk Analytics Dashboard (Issue #15)
    "ConcentrationPanel",
    "CorrelationDisplayData",
    "CorrelationPanel",
    "MarginDisplayData",
    "MarginPositionsTable",
    "MarginUtilizationPanel",
    "RiskAnalyticsDashboard",
    "RiskAnalyticsDashboardData",
    "StressTestDisplayData",
    "StressTestPanel",
    "VaRDisplayData",
    "VaRPanel",
    "create_demo_risk_analytics_data",
]

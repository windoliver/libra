"""
Backtest Results Dashboard Widget.

Combines equity chart, metrics panel, and trade history into a comprehensive
backtest results view.

Features:
- Equity curve with drawdown visualization
- Performance metrics with threshold coloring
- Trade history table with sorting/filtering
- Tabbed interface for different views
- Data export capabilities

Design inspired by:
- QuantConnect backtest results
- TradingView strategy tester
- Interactive Brokers portfolio analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static, TabbedContent, TabPane

from libra.tui.widgets.backtest_metrics import (
    BacktestMetrics,
    BacktestMetricsPanel,
    create_demo_metrics,
)
from libra.tui.widgets.equity_chart import (
    DrawdownChart,
    EquityCurveChart,
    EquityCurveData,
    EquityPoint,
    TradeMarker,
    create_demo_equity_data,
)
from libra.tui.widgets.trade_history import (
    CollapsibleTradeDetails,
    TradeHistoryTable,
    TradeRecord,
    TradeSide,
    create_demo_trades,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class BacktestResultsData:
    """Complete backtest results data for the dashboard."""

    # Strategy info
    strategy_name: str = "Strategy"
    symbol: str = "BTCUSD"
    timeframe: str = "1h"
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Equity data
    equity_data: EquityCurveData = field(default_factory=EquityCurveData)

    # Metrics
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)

    # Trades
    trades: list[TradeRecord] = field(default_factory=list)

    @property
    def backtest_period(self) -> str:
        """Get backtest period string."""
        if self.start_date and self.end_date:
            return f"{self.start_date:%Y-%m-%d} to {self.end_date:%Y-%m-%d}"
        return "N/A"


def convert_from_backtest_result(result: object) -> BacktestResultsData:
    """
    Convert from libra.backtest.result.BacktestResult to dashboard data.

    This bridges the existing backtest engine output to the dashboard display.
    """
    # Import here to avoid circular imports
    try:
        from libra.backtest.result import BacktestResult as EngineResult
    except ImportError:
        # Return empty data if backtest module not available
        return BacktestResultsData()

    if not isinstance(result, EngineResult):
        return BacktestResultsData()

    # Convert equity curve
    equity_points: list[EquityPoint] = []
    for ep in result.equity_curve:
        equity_points.append(
            EquityPoint(
                timestamp=datetime.fromtimestamp(ep.timestamp_ns / 1e9),
                equity=Decimal(str(ep.equity)),
                cash=Decimal(str(ep.cash)),
                position_value=Decimal(str(ep.position_value)),
                drawdown=Decimal(str(ep.drawdown)),
                drawdown_pct=ep.drawdown_pct,
            )
        )

    # Convert trade markers
    trade_markers: list[TradeMarker] = []
    dashboard_trades: list[TradeRecord] = []

    for tr in result.trades:
        # Trade marker for chart
        trade_markers.append(
            TradeMarker(
                timestamp=datetime.fromtimestamp(tr.entry_time_ns / 1e9),
                price=Decimal(str(tr.entry_price)),
                side=f"{tr.side.name}_ENTRY",
                pnl=Decimal(str(tr.pnl)),
            )
        )

        # Trade record for table
        dashboard_trades.append(
            TradeRecord(
                trade_id=tr.trade_id,
                symbol=tr.symbol,
                side=TradeSide.LONG if tr.side.name == "BUY" else TradeSide.SHORT,
                entry_time=datetime.fromtimestamp(tr.entry_time_ns / 1e9),
                entry_price=Decimal(str(tr.entry_price)),
                exit_time=datetime.fromtimestamp(tr.exit_time_ns / 1e9) if tr.exit_time_ns else None,
                exit_price=Decimal(str(tr.exit_price)) if tr.exit_price else None,
                quantity=Decimal(str(tr.quantity)),
                pnl=Decimal(str(tr.pnl)),
                pnl_pct=tr.pnl_pct,
            )
        )

    # Create equity curve data
    equity_data = EquityCurveData(
        points=equity_points,
        trades=trade_markers,
        initial_capital=Decimal(str(result.summary.initial_capital)),
        final_equity=Decimal(str(result.summary.final_equity)),
        max_equity=Decimal(str(max(ep.equity for ep in equity_points))) if equity_points else Decimal("10000"),
        min_equity=Decimal(str(min(ep.equity for ep in equity_points))) if equity_points else Decimal("10000"),
        max_drawdown_pct=result.summary.max_drawdown_pct,
    )

    # Create metrics
    metrics = BacktestMetrics(
        total_return=result.summary.total_return,
        total_return_pct=result.summary.total_return_pct,
        sharpe_ratio=result.summary.sharpe_ratio,
        sortino_ratio=result.summary.sortino_ratio if hasattr(result.summary, "sortino_ratio") else 0.0,
        calmar_ratio=result.summary.calmar_ratio if hasattr(result.summary, "calmar_ratio") else 0.0,
        max_drawdown=result.summary.max_drawdown,
        max_drawdown_pct=result.summary.max_drawdown_pct,
        total_trades=result.summary.total_trades,
        winning_trades=result.summary.winning_trades,
        losing_trades=result.summary.losing_trades,
        win_rate=result.summary.win_rate,
        profit_factor=result.summary.profit_factor,
        avg_win=result.summary.avg_win if hasattr(result.summary, "avg_win") else 0.0,
        avg_loss=result.summary.avg_loss if hasattr(result.summary, "avg_loss") else 0.0,
        expectancy=result.summary.expectancy if hasattr(result.summary, "expectancy") else 0.0,
    )

    return BacktestResultsData(
        strategy_name=result.summary.strategy_name if hasattr(result.summary, "strategy_name") else "Strategy",
        symbol=result.summary.symbol if hasattr(result.summary, "symbol") else "UNKNOWN",
        start_date=equity_points[0].timestamp if equity_points else None,
        end_date=equity_points[-1].timestamp if equity_points else None,
        equity_data=equity_data,
        metrics=metrics,
        trades=dashboard_trades,
    )


# =============================================================================
# Backtest Results Dashboard
# =============================================================================


class BacktestResultsDashboard(Container):
    """
    Comprehensive backtest results dashboard.

    Combines:
    - Header with strategy info and key metrics
    - Equity curve chart with drawdown
    - Performance metrics panel
    - Trade history table with filtering
    """

    DEFAULT_CSS = """
    BacktestResultsDashboard {
        height: auto;
        min-height: 50;
        width: 100%;
        layout: vertical;
    }

    BacktestResultsDashboard .dashboard-header {
        height: 3;
        padding: 0 1;
        background: $primary-darken-2;
        layout: horizontal;
    }

    BacktestResultsDashboard TabbedContent {
        height: auto;
        min-height: 35;
    }

    BacktestResultsDashboard TabPane {
        height: auto;
        min-height: 30;
    }

    BacktestResultsDashboard TabPane > Vertical {
        height: auto;
    }

    BacktestResultsDashboard .header-title {
        width: auto;
        text-style: bold;
        padding-right: 2;
    }

    BacktestResultsDashboard .header-info {
        width: 1fr;
    }

    BacktestResultsDashboard .header-actions {
        width: auto;
        dock: right;
    }

    BacktestResultsDashboard .main-content {
        height: auto;
        min-height: 35;
        layout: horizontal;
    }

    BacktestResultsDashboard .left-panel {
        width: 60%;
        height: auto;
        min-height: 30;
    }

    BacktestResultsDashboard .right-panel {
        width: 40%;
        height: auto;
        min-height: 30;
        border-left: solid $primary-darken-3;
        padding-left: 1;
    }

    BacktestResultsDashboard .chart-section {
        height: 18;
        margin-bottom: 1;
    }

    BacktestResultsDashboard .trades-section {
        height: 15;
    }

    BacktestResultsDashboard .metrics-section {
        height: auto;
        padding: 0 1;
    }

    BacktestResultsDashboard .section-title {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
        text-style: bold;
        margin-bottom: 1;
    }

    BacktestResultsDashboard .positive {
        color: $success;
    }

    BacktestResultsDashboard .negative {
        color: $error;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("e", "export", "Export Results"),
        Binding("escape", "close", "Close"),
    ]

    class ExportRequested(Message):
        """Message sent when export is requested."""

        pass

    class CloseRequested(Message):
        """Message sent when close is requested."""

        pass

    def __init__(
        self,
        data: BacktestResultsData | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._data = data or BacktestResultsData()

    def compose(self) -> ComposeResult:
        d = self._data

        # Header
        with Horizontal(classes="dashboard-header"):
            yield Static(f"📊 {d.strategy_name}", classes="header-title")

            # Key info
            return_class = "positive" if d.metrics.total_return_pct >= 0 else "negative"
            return_sign = "+" if d.metrics.total_return_pct >= 0 else ""
            yield Static(
                f"{d.symbol} | {d.timeframe} | {d.backtest_period} | "
                f"Return: [{return_class}]{return_sign}{d.metrics.total_return_pct:.1f}%[/{return_class}] | "
                f"Sharpe: {d.metrics.sharpe_ratio:.2f} | "
                f"Trades: {d.metrics.total_trades}",
                classes="header-info",
            )

            with Horizontal(classes="header-actions"):
                yield Button("Export", id="export-btn", variant="default")
                yield Button("Close", id="close-btn", variant="default")

        # Main content
        with Horizontal(classes="main-content"):
            # Left panel - Charts and trades
            with Vertical(classes="left-panel"):
                with TabbedContent():
                    with TabPane("Overview", id="tab-overview"):
                        with Vertical():
                            with Container(classes="chart-section"):
                                yield EquityCurveChart(
                                    data=d.equity_data,
                                    title="EQUITY CURVE",
                                    id="equity-chart",
                                )
                            with Container(classes="trades-section"):
                                yield TradeHistoryTable(
                                    trades=d.trades,
                                    title="TRADE HISTORY",
                                    id="trade-table",
                                )

                    with TabPane("Charts", id="tab-charts"):
                        with Vertical():
                            with Container(classes="chart-section"):
                                yield EquityCurveChart(
                                    data=d.equity_data,
                                    title="EQUITY CURVE",
                                    id="equity-chart-full",
                                )
                            with Container(classes="chart-section"):
                                drawdown_data = [
                                    (p.timestamp, p.drawdown_pct) for p in d.equity_data.points
                                ]
                                yield DrawdownChart(
                                    drawdown_data=drawdown_data,
                                    max_drawdown=d.equity_data.max_drawdown_pct,
                                    id="drawdown-chart",
                                )

                    with TabPane("Trades", id="tab-trades"):
                        with Vertical():
                            yield TradeHistoryTable(
                                trades=d.trades,
                                title="FULL TRADE HISTORY",
                                id="trade-table-full",
                            )
                            yield CollapsibleTradeDetails(id="trade-detail")

            # Right panel - Metrics
            with Vertical(classes="right-panel"):
                yield Static("PERFORMANCE METRICS", classes="section-title")
                with Container(classes="metrics-section"):
                    yield BacktestMetricsPanel(
                        metrics=d.metrics,
                        id="metrics-panel",
                    )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "export-btn":
            self.action_export()
        elif event.button.id == "close-btn":
            self.action_close()

    def on_trade_history_table_trade_selected(
        self, event: TradeHistoryTable.TradeSelected
    ) -> None:
        """Handle trade selection."""
        try:
            detail_panel = self.query_one("#trade-detail", CollapsibleTradeDetails)
            detail_panel.update_trade(event.trade)
        except Exception:
            pass

    def action_refresh(self) -> None:
        """Refresh all components."""
        # Re-render charts
        try:
            equity_chart = self.query_one("#equity-chart", EquityCurveChart)
            equity_chart.update_data(self._data.equity_data)
        except Exception:
            pass

        try:
            trade_table = self.query_one("#trade-table", TradeHistoryTable)
            trade_table.update_trades(self._data.trades)
        except Exception:
            pass

    def action_export(self) -> None:
        """Request export."""
        self.post_message(self.ExportRequested())
        self.notify("Export feature - coming soon!", title="Export")

    def action_close(self) -> None:
        """Request close."""
        self.post_message(self.CloseRequested())

    def update_data(self, data: BacktestResultsData) -> None:
        """Update dashboard with new data."""
        self._data = data

        # Update equity chart
        try:
            equity_chart = self.query_one("#equity-chart", EquityCurveChart)
            equity_chart.update_data(data.equity_data)
        except Exception:
            pass

        # Update metrics
        try:
            metrics_panel = self.query_one("#metrics-panel", BacktestMetricsPanel)
            metrics_panel.update_metrics(data.metrics)
        except Exception:
            pass

        # Update trades
        try:
            trade_table = self.query_one("#trade-table", TradeHistoryTable)
            trade_table.update_trades(data.trades)
        except Exception:
            pass


# =============================================================================
# Demo Data Generator
# =============================================================================


def create_demo_backtest_results() -> BacktestResultsData:
    """Create demo backtest results for testing."""
    from datetime import timedelta

    equity_data = create_demo_equity_data(days=90, initial_capital=10000.0)
    metrics = create_demo_metrics()
    trades = create_demo_trades(count=50)

    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()

    return BacktestResultsData(
        strategy_name="Demo Strategy",
        symbol="BTCUSD",
        timeframe="1h",
        start_date=start_date,
        end_date=end_date,
        equity_data=equity_data,
        metrics=metrics,
        trades=trades,
    )


# =============================================================================
# Standalone Demo Screen
# =============================================================================


class BacktestResultsScreen(Container):
    """
    Standalone screen for backtest results.

    Can be used as a modal or full screen.
    """

    DEFAULT_CSS = """
    BacktestResultsScreen {
        height: 100%;
        width: 100%;
        background: $surface;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "close", "Close"),
    ]

    def __init__(
        self,
        data: BacktestResultsData | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._data = data or create_demo_backtest_results()

    def compose(self) -> ComposeResult:
        yield BacktestResultsDashboard(data=self._data, id="dashboard")

    def on_backtest_results_dashboard_close_requested(
        self, event: BacktestResultsDashboard.CloseRequested
    ) -> None:
        """Handle close request."""
        self.action_close()

    def action_close(self) -> None:
        """Close the screen."""
        # This would typically be handled by parent
        pass

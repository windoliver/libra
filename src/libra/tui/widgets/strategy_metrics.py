"""
Strategy Metrics Panel Widget.

Displays key performance indicators (KPIs) for a strategy.

Metrics displayed:
- Sharpe Ratio (risk-adjusted returns)
- Max Drawdown (worst peak-to-trough decline)
- Win Rate (percentage of profitable trades)
- Profit Factor (gross profit / gross loss)
- Total Trades
- Average Trade (mean P&L per trade)

Design inspired by:
- TradingView Strategy Tester
- QuantConnect metrics panels
- Professional quant dashboards

Layout:
    +-- PERFORMANCE METRICS ---------------------------+
    | Sharpe Ratio      Max Drawdown                   |
    | [2.15]            [12.5%]                        |
    |                                                  |
    | Win Rate          Profit Factor                  |
    | [65.2%]           [1.85]                         |
    |                                                  |
    | Total Trades      Avg Trade                      |
    | [142]             [$45.20]                       |
    +-------------------------------------------------+
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Grid
from textual.reactive import reactive
from textual.widgets import Label, ProgressBar, Sparkline, Static


if TYPE_CHECKING:
    pass


# =============================================================================
# Metric Card Widget
# =============================================================================


class MetricCard(Container):
    """
    Single metric display card.

    Shows a labeled value with optional threshold-based coloring.
    """

    DEFAULT_CSS = """
    MetricCard {
        height: 4;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    MetricCard .metric-label {
        height: 1;
        color: $text-muted;
    }

    MetricCard .metric-value {
        height: 2;
        text-style: bold;
        content-align: left middle;
    }

    MetricCard .metric-value.good {
        color: $success;
    }

    MetricCard .metric-value.warning {
        color: $warning;
    }

    MetricCard .metric-value.bad {
        color: $error;
    }

    MetricCard .metric-value.neutral {
        color: $text;
    }
    """

    value: reactive[str] = reactive("--")
    threshold_good: reactive[float | None] = reactive(None)
    threshold_bad: reactive[float | None] = reactive(None)
    invert_threshold: reactive[bool] = reactive(False)  # True = lower is better
    numeric_value: reactive[float] = reactive(0.0)

    def __init__(
        self,
        label: str,
        value: str = "--",
        numeric_value: float = 0.0,
        threshold_good: float | None = None,
        threshold_bad: float | None = None,
        invert_threshold: bool = False,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._label = label
        self.value = value
        self.numeric_value = numeric_value
        self.threshold_good = threshold_good
        self.threshold_bad = threshold_bad
        self.invert_threshold = invert_threshold

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="metric-label")
        yield Static(self.value, classes=f"metric-value {self._get_value_class()}", id="metric-value")

    def _get_value_class(self) -> str:
        """Determine CSS class based on thresholds."""
        if self.threshold_good is None and self.threshold_bad is None:
            return "neutral"

        val = self.numeric_value

        if self.invert_threshold:
            # Lower is better (e.g., drawdown)
            if self.threshold_good is not None and val <= self.threshold_good:
                return "good"
            if self.threshold_bad is not None and val >= self.threshold_bad:
                return "bad"
        else:
            # Higher is better (e.g., Sharpe)
            if self.threshold_good is not None and val >= self.threshold_good:
                return "good"
            if self.threshold_bad is not None and val <= self.threshold_bad:
                return "bad"

        return "warning"

    def watch_value(self, new_value: str) -> None:
        """Update displayed value."""
        try:
            value_widget = self.query_one("#metric-value", Static)
            value_widget.update(new_value)
            # Update class
            value_widget.remove_class("good", "warning", "bad", "neutral")
            value_widget.add_class(self._get_value_class())
        except Exception:
            pass

    def update_metric(self, value: str, numeric_value: float) -> None:
        """Update both display value and numeric value."""
        self.numeric_value = numeric_value
        self.value = value


# =============================================================================
# Drawdown Gauge Widget
# =============================================================================


class DrawdownGauge(Container):
    """
    Visual gauge for drawdown with progress bar and sparkline history.
    """

    DEFAULT_CSS = """
    DrawdownGauge {
        height: 5;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    DrawdownGauge .gauge-label {
        height: 1;
        color: $text-muted;
    }

    DrawdownGauge .gauge-row {
        height: 1;
        layout: horizontal;
    }

    DrawdownGauge ProgressBar {
        width: 1fr;
        padding: 0;
    }

    DrawdownGauge .gauge-value {
        width: 12;
        text-align: right;
    }

    DrawdownGauge .gauge-value.safe {
        color: $success;
    }

    DrawdownGauge .gauge-value.warning {
        color: $warning;
    }

    DrawdownGauge .gauge-value.danger {
        color: $error;
    }

    DrawdownGauge Sparkline {
        height: 1;
        margin-top: 1;
    }
    """

    current: reactive[float] = reactive(0.0)
    maximum: reactive[float] = reactive(50.0)
    history: reactive[list[float]] = reactive(list, init=False)

    def __init__(
        self,
        current: float = 0.0,
        maximum: float = 50.0,
        history: list[float] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.current = current
        self.maximum = maximum
        self.history = history or [0.0] * 20

    def compose(self) -> ComposeResult:
        yield Static("Drawdown", classes="gauge-label")
        with Container(classes="gauge-row"):
            yield ProgressBar(total=self.maximum, show_eta=False, id="dd-bar")
            yield Static(
                self._format_value(),
                classes=f"gauge-value {self._get_value_class()}",
                id="dd-value",
            )
        yield Sparkline(self.history, id="dd-sparkline")

    def on_mount(self) -> None:
        """Initialize progress bar value."""
        bar = self.query_one("#dd-bar", ProgressBar)
        bar.update(progress=self.current)

    def _format_value(self) -> str:
        """Format the current/max display."""
        return f"{self.current:.1f}% / {self.maximum:.0f}%"

    def _get_value_class(self) -> str:
        """Get CSS class based on drawdown severity."""
        ratio = self.current / self.maximum if self.maximum > 0 else 0
        if ratio < 0.5:
            return "safe"
        elif ratio < 0.75:
            return "warning"
        return "danger"

    def watch_current(self, value: float) -> None:
        """Update gauge when current drawdown changes."""
        try:
            bar = self.query_one("#dd-bar", ProgressBar)
            bar.update(progress=value)

            value_widget = self.query_one("#dd-value", Static)
            value_widget.update(self._format_value())
            value_widget.remove_class("safe", "warning", "danger")
            value_widget.add_class(self._get_value_class())
        except Exception:
            pass

    def watch_history(self, value: list[float]) -> None:
        """Update sparkline when history changes."""
        try:
            sparkline = self.query_one("#dd-sparkline", Sparkline)
            sparkline.data = value
        except Exception:
            pass


# =============================================================================
# Strategy Metrics Panel
# =============================================================================


class StrategyMetricsPanel(Container):
    """
    Comprehensive strategy performance metrics display.

    Displays key KPIs in a grid layout with threshold-based coloring.
    """

    DEFAULT_CSS = """
    StrategyMetricsPanel {
        height: auto;
        padding: 1;
    }

    StrategyMetricsPanel .metrics-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    StrategyMetricsPanel .metrics-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        height: auto;
    }

    StrategyMetricsPanel .metrics-row {
        layout: horizontal;
        height: auto;
    }

    StrategyMetricsPanel .pnl-section {
        height: auto;
        margin-bottom: 1;
    }

    StrategyMetricsPanel .large-pnl {
        height: 3;
        text-style: bold;
        content-align: center middle;
        border: round $primary-darken-2;
        background: $surface;
    }

    StrategyMetricsPanel .large-pnl.positive {
        color: $success;
    }

    StrategyMetricsPanel .large-pnl.negative {
        color: $error;
    }

    StrategyMetricsPanel .pnl-sparkline-container {
        height: 3;
        border: round $primary-darken-2;
        background: $surface;
        padding: 0 1;
    }

    StrategyMetricsPanel .pnl-sparkline-container Sparkline {
        height: 2;
    }
    """

    # Reactive metrics
    total_pnl: reactive[Decimal] = reactive(Decimal("0"))
    sharpe_ratio: reactive[float] = reactive(0.0)
    max_drawdown: reactive[float] = reactive(0.0)
    win_rate: reactive[float] = reactive(0.0)
    profit_factor: reactive[float] = reactive(0.0)
    total_trades: reactive[int] = reactive(0)
    avg_trade: reactive[Decimal] = reactive(Decimal("0"))
    pnl_history: reactive[list[float]] = reactive(list, init=False)

    def __init__(
        self,
        total_pnl: Decimal = Decimal("0"),
        sharpe_ratio: float = 0.0,
        max_drawdown: float = 0.0,
        win_rate: float = 0.0,
        profit_factor: float = 0.0,
        total_trades: int = 0,
        avg_trade: Decimal = Decimal("0"),
        pnl_history: list[float] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.total_pnl = total_pnl
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.total_trades = total_trades
        self.avg_trade = avg_trade
        self.pnl_history = pnl_history or [0.0] * 30

    def compose(self) -> ComposeResult:
        yield Static("PERFORMANCE METRICS", classes="metrics-title")

        # Large P&L display with sparkline
        with Container(classes="pnl-section metrics-row"):
            yield Static(
                self._format_large_pnl(),
                classes=f"large-pnl {self._pnl_class()}",
                id="large-pnl",
            )
            with Container(classes="pnl-sparkline-container"):
                yield Static("P&L Trend", classes="metric-label")
                yield Sparkline(self.pnl_history, id="pnl-sparkline")

        # Metrics grid
        with Grid(classes="metrics-grid"):
            yield MetricCard(
                "Sharpe Ratio",
                f"{self.sharpe_ratio:.2f}",
                self.sharpe_ratio,
                threshold_good=2.0,
                threshold_bad=0.5,
                id="metric-sharpe",
            )
            yield MetricCard(
                "Max Drawdown",
                f"{self.max_drawdown:.1f}%",
                self.max_drawdown,
                threshold_good=10.0,
                threshold_bad=30.0,
                invert_threshold=True,
                id="metric-drawdown",
            )
            yield MetricCard(
                "Win Rate",
                f"{self.win_rate:.1f}%",
                self.win_rate,
                threshold_good=55.0,
                threshold_bad=40.0,
                id="metric-winrate",
            )
            yield MetricCard(
                "Profit Factor",
                f"{self.profit_factor:.2f}",
                self.profit_factor,
                threshold_good=1.5,
                threshold_bad=1.0,
                id="metric-pf",
            )
            yield MetricCard(
                "Total Trades",
                str(self.total_trades),
                float(self.total_trades),
                id="metric-trades",
            )
            yield MetricCard(
                "Avg Trade",
                self._format_avg_trade(),
                float(self.avg_trade),
                threshold_good=0.0,
                threshold_bad=-50.0,
                id="metric-avgtrade",
            )

    def _format_large_pnl(self) -> str:
        """Format large P&L display."""
        pnl = self.total_pnl
        if pnl >= 0:
            return f"+${pnl:,.2f}"
        return f"-${abs(pnl):,.2f}"

    def _pnl_class(self) -> str:
        """Get CSS class for P&L."""
        if self.total_pnl > 0:
            return "positive"
        elif self.total_pnl < 0:
            return "negative"
        return ""

    def _format_avg_trade(self) -> str:
        """Format average trade value."""
        avg = self.avg_trade
        if avg >= 0:
            return f"+${avg:,.2f}"
        return f"-${abs(avg):,.2f}"

    def watch_total_pnl(self, value: Decimal) -> None:
        """Update P&L display."""
        try:
            pnl_widget = self.query_one("#large-pnl", Static)
            pnl_widget.update(self._format_large_pnl())
            pnl_widget.remove_class("positive", "negative")
            if value > 0:
                pnl_widget.add_class("positive")
            elif value < 0:
                pnl_widget.add_class("negative")
        except Exception:
            pass

    def watch_pnl_history(self, value: list[float]) -> None:
        """Update sparkline."""
        try:
            sparkline = self.query_one("#pnl-sparkline", Sparkline)
            sparkline.data = value
        except Exception:
            pass

    def watch_sharpe_ratio(self, value: float) -> None:
        """Update Sharpe metric."""
        try:
            card = self.query_one("#metric-sharpe", MetricCard)
            card.update_metric(f"{value:.2f}", value)
        except Exception:
            pass

    def watch_max_drawdown(self, value: float) -> None:
        """Update drawdown metric."""
        try:
            card = self.query_one("#metric-drawdown", MetricCard)
            card.update_metric(f"{value:.1f}%", value)
        except Exception:
            pass

    def watch_win_rate(self, value: float) -> None:
        """Update win rate metric."""
        try:
            card = self.query_one("#metric-winrate", MetricCard)
            card.update_metric(f"{value:.1f}%", value)
        except Exception:
            pass

    def watch_profit_factor(self, value: float) -> None:
        """Update profit factor metric."""
        try:
            card = self.query_one("#metric-pf", MetricCard)
            card.update_metric(f"{value:.2f}", value)
        except Exception:
            pass

    def watch_total_trades(self, value: int) -> None:
        """Update total trades metric."""
        try:
            card = self.query_one("#metric-trades", MetricCard)
            card.update_metric(str(value), float(value))
        except Exception:
            pass

    def watch_avg_trade(self, value: Decimal) -> None:
        """Update average trade metric."""
        try:
            card = self.query_one("#metric-avgtrade", MetricCard)
            card.update_metric(self._format_avg_trade(), float(value))
        except Exception:
            pass

    def update_all_metrics(
        self,
        total_pnl: Decimal | None = None,
        sharpe_ratio: float | None = None,
        max_drawdown: float | None = None,
        win_rate: float | None = None,
        profit_factor: float | None = None,
        total_trades: int | None = None,
        avg_trade: Decimal | None = None,
        pnl_history: list[float] | None = None,
    ) -> None:
        """Batch update all metrics."""
        if total_pnl is not None:
            self.total_pnl = total_pnl
        if sharpe_ratio is not None:
            self.sharpe_ratio = sharpe_ratio
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
        if win_rate is not None:
            self.win_rate = win_rate
        if profit_factor is not None:
            self.profit_factor = profit_factor
        if total_trades is not None:
            self.total_trades = total_trades
        if avg_trade is not None:
            self.avg_trade = avg_trade
        if pnl_history is not None:
            self.pnl_history = pnl_history

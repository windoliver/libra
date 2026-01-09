"""
Backtest Metrics Panel Widget.

Displays key performance metrics from backtest results.

Features:
- Large return display with Digits
- Grid of KPI cards
- Color-coded thresholds (green/yellow/red)
- Tooltips explaining each metric

Metrics displayed:
- Total Return (absolute and %)
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown (% and duration)
- Win Rate, Profit Factor
- Total Trades, Avg Win/Loss
- Expectancy

Design inspired by:
- QuantConnect Results
- TradingView Strategy Tester
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal
from textual.reactive import reactive
from textual.widgets import Digits, Static

if TYPE_CHECKING:
    pass


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class BacktestMetrics:
    """Complete backtest metrics for display."""

    # Returns
    total_return: Decimal = Decimal("0")
    total_return_pct: float = 0.0
    cagr: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Profit
    profit_factor: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    avg_trade: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")

    # Risk/Reward
    avg_win_loss_ratio: float = 0.0
    expectancy: Decimal = Decimal("0")

    # Volume
    total_volume: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")

    # Execution
    bars_processed: int = 0
    duration_days: float = 0.0


# =============================================================================
# Metric Thresholds
# =============================================================================


class MetricThresholds:
    """Threshold definitions for metric coloring."""

    SHARPE = {"good": 1.5, "warning": 1.0}  # > good = green, > warning = yellow, else red
    SORTINO = {"good": 2.0, "warning": 1.0}
    CALMAR = {"good": 2.0, "warning": 1.0}
    MAX_DRAWDOWN = {"good": 10, "warning": 20}  # < good = green, < warning = yellow, else red (inverted)
    WIN_RATE = {"good": 55, "warning": 45}
    PROFIT_FACTOR = {"good": 1.5, "warning": 1.0}

    @staticmethod
    def get_class(value: float, good: float, warning: float, inverted: bool = False) -> str:
        """Get CSS class based on threshold."""
        if inverted:
            if value < good:
                return "good"
            elif value < warning:
                return "warning"
            return "bad"
        else:
            if value >= good:
                return "good"
            elif value >= warning:
                return "warning"
            return "bad"


# =============================================================================
# Metric Card Widget
# =============================================================================


class MetricCard(Static):
    """Single metric display card with threshold coloring."""

    DEFAULT_CSS = """
    MetricCard {
        height: 4;
        padding: 0 1;
        border: round $primary-darken-3;
        background: $surface;
    }

    MetricCard .metric-label {
        color: $text-muted;
        height: 1;
    }

    MetricCard .metric-value {
        text-style: bold;
        height: 2;
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

    def __init__(
        self,
        label: str,
        value: str,
        value_class: str = "neutral",
        tooltip: str = "",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._label = label
        self._value = value
        self._value_class = value_class
        self._tooltip = tooltip

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="metric-label")
        yield Static(self._value, classes=f"metric-value {self._value_class}")

    def update_value(self, value: str, value_class: str = "neutral") -> None:
        """Update the metric value and class."""
        self._value = value
        self._value_class = value_class
        try:
            value_widget = self.query_one(".metric-value", Static)
            value_widget.update(value)
            value_widget.remove_class("good", "warning", "bad", "neutral")
            value_widget.add_class(value_class)
        except Exception:
            pass


# =============================================================================
# Large Return Display
# =============================================================================


class ReturnDisplay(Container):
    """Large return display using Digits widget."""

    DEFAULT_CSS = """
    ReturnDisplay {
        height: 9;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
        layout: vertical;
        margin-bottom: 1;
    }

    ReturnDisplay .return-label {
        height: 1;
        text-style: bold;
        color: $text-muted;
    }

    ReturnDisplay .return-row {
        height: 5;
        layout: horizontal;
    }

    ReturnDisplay Digits {
        width: 1fr;
        height: 100%;
    }

    ReturnDisplay .return-pct {
        width: auto;
        padding: 0 1;
        content-align: right middle;
        text-style: bold;
    }

    ReturnDisplay .return-pct.positive {
        color: $success;
    }

    ReturnDisplay .return-pct.negative {
        color: $error;
    }

    ReturnDisplay .cagr-label {
        height: 1;
        color: $text-muted;
    }
    """

    total_return: reactive[Decimal] = reactive(Decimal("0"))
    total_return_pct: reactive[float] = reactive(0.0)
    cagr: reactive[float] = reactive(0.0)

    def __init__(
        self,
        total_return: Decimal = Decimal("0"),
        total_return_pct: float = 0.0,
        cagr: float = 0.0,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.total_return = total_return
        self.total_return_pct = total_return_pct
        self.cagr = cagr

    def compose(self) -> ComposeResult:
        yield Static("TOTAL RETURN", classes="return-label")
        with Horizontal(classes="return-row"):
            sign = "+" if self.total_return >= 0 else ""
            yield Digits(f"{sign}${self.total_return:,.2f}", id="return-digits")
            pct_class = "positive" if self.total_return_pct >= 0 else "negative"
            pct_sign = "+" if self.total_return_pct >= 0 else ""
            yield Static(
                f"({pct_sign}{self.total_return_pct:.2f}%)",
                classes=f"return-pct {pct_class}",
                id="return-pct",
            )
        cagr_sign = "+" if self.cagr >= 0 else ""
        yield Static(
            f"CAGR: {cagr_sign}{self.cagr:.2f}%",
            classes="cagr-label",
            id="cagr-label",
        )

    def update_data(
        self,
        total_return: Decimal,
        total_return_pct: float,
        cagr: float,
    ) -> None:
        """Update return values."""
        self.total_return = total_return
        self.total_return_pct = total_return_pct
        self.cagr = cagr

        try:
            sign = "+" if total_return >= 0 else ""
            digits = self.query_one("#return-digits", Digits)
            digits.update(f"{sign}${total_return:,.2f}")

            pct_class = "positive" if total_return_pct >= 0 else "negative"
            pct_sign = "+" if total_return_pct >= 0 else ""
            pct_widget = self.query_one("#return-pct", Static)
            pct_widget.update(f"({pct_sign}{total_return_pct:.2f}%)")
            pct_widget.remove_class("positive", "negative")
            pct_widget.add_class(pct_class)

            cagr_sign = "+" if cagr >= 0 else ""
            cagr_widget = self.query_one("#cagr-label", Static)
            cagr_widget.update(f"CAGR: {cagr_sign}{cagr:.2f}%")
        except Exception:
            pass


# =============================================================================
# Risk Metrics Panel
# =============================================================================


class RiskMetricsPanel(Container):
    """Panel showing risk-adjusted metrics."""

    DEFAULT_CSS = """
    RiskMetricsPanel {
        height: auto;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
        margin-bottom: 1;
    }

    RiskMetricsPanel .panel-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    RiskMetricsPanel .metrics-grid {
        height: auto;
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
    }
    """

    def __init__(
        self,
        sharpe: float = 0.0,
        sortino: float = 0.0,
        calmar: float = 0.0,
        max_dd_pct: float = 0.0,
        max_dd_days: float = 0.0,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._sharpe = sharpe
        self._sortino = sortino
        self._calmar = calmar
        self._max_dd_pct = max_dd_pct
        self._max_dd_days = max_dd_days

    def compose(self) -> ComposeResult:
        yield Static("RISK METRICS", classes="panel-title")
        with Grid(classes="metrics-grid"):
            yield MetricCard(
                "Sharpe Ratio",
                f"{self._sharpe:.2f}",
                MetricThresholds.get_class(self._sharpe, 1.5, 1.0),
                id="metric-sharpe",
            )
            yield MetricCard(
                "Sortino Ratio",
                f"{self._sortino:.2f}",
                MetricThresholds.get_class(self._sortino, 2.0, 1.0),
                id="metric-sortino",
            )
            yield MetricCard(
                "Calmar Ratio",
                f"{self._calmar:.2f}",
                MetricThresholds.get_class(self._calmar, 2.0, 1.0),
                id="metric-calmar",
            )
            yield MetricCard(
                "Max Drawdown",
                f"{self._max_dd_pct:.1f}%",
                MetricThresholds.get_class(self._max_dd_pct, 10, 20, inverted=True),
                id="metric-maxdd",
            )
            yield MetricCard(
                "DD Duration",
                f"{self._max_dd_days:.0f} days",
                "neutral",
                id="metric-dddays",
            )

    def update_metrics(
        self,
        sharpe: float,
        sortino: float,
        calmar: float,
        max_dd_pct: float,
        max_dd_days: float,
    ) -> None:
        """Update risk metrics."""
        try:
            self.query_one("#metric-sharpe", MetricCard).update_value(
                f"{sharpe:.2f}",
                MetricThresholds.get_class(sharpe, 1.5, 1.0),
            )
            self.query_one("#metric-sortino", MetricCard).update_value(
                f"{sortino:.2f}",
                MetricThresholds.get_class(sortino, 2.0, 1.0),
            )
            self.query_one("#metric-calmar", MetricCard).update_value(
                f"{calmar:.2f}",
                MetricThresholds.get_class(calmar, 2.0, 1.0),
            )
            self.query_one("#metric-maxdd", MetricCard).update_value(
                f"{max_dd_pct:.1f}%",
                MetricThresholds.get_class(max_dd_pct, 10, 20, inverted=True),
            )
            self.query_one("#metric-dddays", MetricCard).update_value(
                f"{max_dd_days:.0f} days",
                "neutral",
            )
        except Exception:
            pass


# =============================================================================
# Trade Stats Panel
# =============================================================================


class TradeStatsPanel(Container):
    """Panel showing trade statistics."""

    DEFAULT_CSS = """
    TradeStatsPanel {
        height: auto;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    TradeStatsPanel .panel-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    TradeStatsPanel .metrics-grid {
        height: auto;
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
    }
    """

    def __init__(
        self,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        win_rate: float = 0.0,
        profit_factor: float = 0.0,
        avg_win: Decimal = Decimal("0"),
        avg_loss: Decimal = Decimal("0"),
        expectancy: Decimal = Decimal("0"),
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._total_trades = total_trades
        self._winning_trades = winning_trades
        self._losing_trades = losing_trades
        self._win_rate = win_rate
        self._profit_factor = profit_factor
        self._avg_win = avg_win
        self._avg_loss = avg_loss
        self._expectancy = expectancy

    def compose(self) -> ComposeResult:
        yield Static("TRADE STATISTICS", classes="panel-title")
        with Grid(classes="metrics-grid"):
            yield MetricCard(
                "Total Trades",
                str(self._total_trades),
                "neutral",
                id="metric-total",
            )
            yield MetricCard(
                "Winning",
                str(self._winning_trades),
                "good",
                id="metric-winning",
            )
            yield MetricCard(
                "Losing",
                str(self._losing_trades),
                "bad" if self._losing_trades > self._winning_trades else "neutral",
                id="metric-losing",
            )
            yield MetricCard(
                "Win Rate",
                f"{self._win_rate:.1f}%",
                MetricThresholds.get_class(self._win_rate, 55, 45),
                id="metric-winrate",
            )
            yield MetricCard(
                "Profit Factor",
                f"{self._profit_factor:.2f}",
                MetricThresholds.get_class(self._profit_factor, 1.5, 1.0),
                id="metric-pf",
            )
            yield MetricCard(
                "Expectancy",
                f"${self._expectancy:,.2f}",
                "good" if self._expectancy > 0 else "bad",
                id="metric-expectancy",
            )
            yield MetricCard(
                "Avg Win",
                f"${self._avg_win:,.2f}",
                "good",
                id="metric-avgwin",
            )
            yield MetricCard(
                "Avg Loss",
                f"${self._avg_loss:,.2f}",
                "bad",
                id="metric-avgloss",
            )

    def update_metrics(
        self,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        win_rate: float,
        profit_factor: float,
        avg_win: Decimal,
        avg_loss: Decimal,
        expectancy: Decimal,
    ) -> None:
        """Update trade statistics."""
        try:
            self.query_one("#metric-total", MetricCard).update_value(str(total_trades), "neutral")
            self.query_one("#metric-winning", MetricCard).update_value(str(winning_trades), "good")
            self.query_one("#metric-losing", MetricCard).update_value(
                str(losing_trades),
                "bad" if losing_trades > winning_trades else "neutral",
            )
            self.query_one("#metric-winrate", MetricCard).update_value(
                f"{win_rate:.1f}%",
                MetricThresholds.get_class(win_rate, 55, 45),
            )
            self.query_one("#metric-pf", MetricCard).update_value(
                f"{profit_factor:.2f}",
                MetricThresholds.get_class(profit_factor, 1.5, 1.0),
            )
            self.query_one("#metric-expectancy", MetricCard).update_value(
                f"${expectancy:,.2f}",
                "good" if expectancy > 0 else "bad",
            )
            self.query_one("#metric-avgwin", MetricCard).update_value(f"${avg_win:,.2f}", "good")
            self.query_one("#metric-avgloss", MetricCard).update_value(f"${avg_loss:,.2f}", "bad")
        except Exception:
            pass


# =============================================================================
# Combined Backtest Metrics Panel
# =============================================================================


class BacktestMetricsPanel(Container):
    """
    Complete backtest metrics display panel.

    Combines return display, risk metrics, and trade stats.
    """

    DEFAULT_CSS = """
    BacktestMetricsPanel {
        height: auto;
        width: 100%;
    }

    BacktestMetricsPanel .metrics-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }

    BacktestMetricsPanel .metrics-row > * {
        margin-right: 1;
    }

    BacktestMetricsPanel .metrics-row > *:last-child {
        margin-right: 0;
    }

    BacktestMetricsPanel ReturnDisplay {
        width: 1fr;
    }
    """

    def __init__(
        self,
        metrics: BacktestMetrics | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._metrics = metrics or BacktestMetrics()

    def compose(self) -> ComposeResult:
        m = self._metrics

        # Top row: Return display
        yield ReturnDisplay(
            total_return=m.total_return,
            total_return_pct=m.total_return_pct,
            cagr=m.cagr,
            id="return-display",
        )

        # Risk metrics
        yield RiskMetricsPanel(
            sharpe=m.sharpe_ratio,
            sortino=m.sortino_ratio,
            calmar=m.calmar_ratio,
            max_dd_pct=m.max_drawdown_pct,
            max_dd_days=m.max_drawdown_duration_days,
            id="risk-metrics",
        )

        # Trade stats
        yield TradeStatsPanel(
            total_trades=m.total_trades,
            winning_trades=m.winning_trades,
            losing_trades=m.losing_trades,
            win_rate=m.win_rate,
            profit_factor=m.profit_factor,
            avg_win=m.avg_win,
            avg_loss=m.avg_loss,
            expectancy=m.expectancy,
            id="trade-stats",
        )

    def update_metrics(self, metrics: BacktestMetrics) -> None:
        """Update all metrics."""
        self._metrics = metrics

        try:
            self.query_one("#return-display", ReturnDisplay).update_data(
                metrics.total_return,
                metrics.total_return_pct,
                metrics.cagr,
            )
            self.query_one("#risk-metrics", RiskMetricsPanel).update_metrics(
                metrics.sharpe_ratio,
                metrics.sortino_ratio,
                metrics.calmar_ratio,
                metrics.max_drawdown_pct,
                metrics.max_drawdown_duration_days,
            )
            self.query_one("#trade-stats", TradeStatsPanel).update_metrics(
                metrics.total_trades,
                metrics.winning_trades,
                metrics.losing_trades,
                metrics.win_rate,
                metrics.profit_factor,
                metrics.avg_win,
                metrics.avg_loss,
                metrics.expectancy,
            )
        except Exception:
            pass


# =============================================================================
# Demo Data Generator
# =============================================================================


def create_demo_metrics() -> BacktestMetrics:
    """Create demo backtest metrics for testing."""
    import random

    total_trades = random.randint(50, 200)
    win_rate = random.uniform(45, 65)
    winning_trades = int(total_trades * win_rate / 100)
    losing_trades = total_trades - winning_trades

    avg_win = Decimal(str(random.uniform(100, 300)))
    avg_loss = Decimal(str(random.uniform(50, 200)))

    gross_profit = avg_win * winning_trades
    gross_loss = avg_loss * losing_trades
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0

    total_return = gross_profit - gross_loss
    initial_capital = Decimal("10000")
    total_return_pct = float(total_return / initial_capital * 100)

    return BacktestMetrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        cagr=total_return_pct / 3,  # Assume 3 years
        sharpe_ratio=random.uniform(0.5, 2.5),
        sortino_ratio=random.uniform(0.8, 3.0),
        calmar_ratio=random.uniform(0.5, 3.0),
        max_drawdown=Decimal(str(random.uniform(500, 2000))),
        max_drawdown_pct=random.uniform(5, 25),
        max_drawdown_duration_days=random.uniform(10, 60),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_trade=Decimal(str(float(total_return) / total_trades)) if total_trades > 0 else Decimal("0"),
        largest_win=Decimal(str(float(avg_win) * random.uniform(2, 5))),
        largest_loss=Decimal(str(float(avg_loss) * random.uniform(2, 4))),
        avg_win_loss_ratio=float(avg_win / avg_loss) if avg_loss > 0 else 0,
        expectancy=Decimal(str(float(total_return) / total_trades)) if total_trades > 0 else Decimal("0"),
        duration_days=random.uniform(180, 365),
    )

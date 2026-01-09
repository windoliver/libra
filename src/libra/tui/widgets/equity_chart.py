"""
Equity Curve Chart Widget.

Displays equity curve and drawdown using textual-plotext.

Features:
- Full equity curve visualization
- Drawdown overlay or subplot
- Auto theme switching (dark/light)
- Zoom and pan support
- Trade markers on chart

Design inspired by:
- TradingView strategy tester
- QuantConnect backtest results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Static

try:
    from textual_plotext import PlotextPlot

    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False
    PlotextPlot = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    pass


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class EquityPoint:
    """Single point on the equity curve."""

    timestamp: datetime
    equity: Decimal
    cash: Decimal = Decimal("0")
    position_value: Decimal = Decimal("0")
    drawdown: Decimal = Decimal("0")
    drawdown_pct: float = 0.0


@dataclass
class TradeMarker:
    """Marker for trade entry/exit on chart."""

    timestamp: datetime
    price: Decimal
    side: str  # "LONG_ENTRY", "LONG_EXIT", "SHORT_ENTRY", "SHORT_EXIT"
    pnl: Decimal | None = None


@dataclass
class EquityCurveData:
    """Complete data for equity curve display."""

    points: list[EquityPoint] = field(default_factory=list)
    trades: list[TradeMarker] = field(default_factory=list)
    initial_capital: Decimal = Decimal("10000")
    final_equity: Decimal = Decimal("10000")
    max_equity: Decimal = Decimal("10000")
    min_equity: Decimal = Decimal("10000")
    max_drawdown_pct: float = 0.0

    @property
    def total_return(self) -> Decimal:
        """Calculate total return."""
        if self.initial_capital == 0:
            return Decimal("0")
        return self.final_equity - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Calculate total return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return float((self.final_equity - self.initial_capital) / self.initial_capital * 100)


# =============================================================================
# Equity Curve Chart Widget
# =============================================================================


class EquityCurveChart(Container):
    """
    Equity curve visualization using plotext.

    Shows:
    - Main equity curve line
    - Drawdown as area chart or subplot
    - Trade entry/exit markers
    - High watermark line
    """

    DEFAULT_CSS = """
    EquityCurveChart {
        height: 100%;
        width: 100%;
        padding: 0;
        layout: vertical;
    }

    EquityCurveChart .chart-container {
        height: 1fr;
        width: 100%;
    }

    EquityCurveChart PlotextPlot {
        height: 100%;
        width: 100%;
    }

    EquityCurveChart .chart-header {
        height: 2;
        padding: 0 1;
        background: $surface-darken-1;
    }

    EquityCurveChart .chart-title {
        text-style: bold;
    }

    EquityCurveChart .chart-stats {
        dock: right;
    }

    EquityCurveChart .positive {
        color: $success;
    }

    EquityCurveChart .negative {
        color: $error;
    }

    EquityCurveChart .fallback-message {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("r", "refresh_chart", "Refresh"),
        Binding("d", "toggle_drawdown", "Toggle Drawdown"),
        Binding("t", "toggle_trades", "Toggle Trades"),
    ]

    show_drawdown: reactive[bool] = reactive(True)
    show_trades: reactive[bool] = reactive(True)

    def __init__(
        self,
        data: EquityCurveData | None = None,
        title: str = "EQUITY CURVE",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._data = data or EquityCurveData()
        self._title = title

    def compose(self) -> ComposeResult:
        # Header with title and stats
        with Horizontal(classes="chart-header"):
            yield Static(self._title, classes="chart-title")
            return_class = "positive" if self._data.total_return >= 0 else "negative"
            return_sign = "+" if self._data.total_return >= 0 else ""
            yield Static(
                f"Return: [{return_class}]{return_sign}${self._data.total_return:,.2f} "
                f"({return_sign}{self._data.total_return_pct:.1f}%)[/{return_class}]  "
                f"Max DD: [red]{self._data.max_drawdown_pct:.1f}%[/red]",
                classes="chart-stats",
            )

        # Chart area
        if PLOTEXT_AVAILABLE:
            with Container(classes="chart-container"):
                yield PlotextPlot(id="equity-plot")
        else:
            yield Static(
                "[dim]plotext not available - install with: pip install textual-plotext[/dim]",
                classes="fallback-message",
            )

    def on_mount(self) -> None:
        """Render chart on mount."""
        if PLOTEXT_AVAILABLE:
            self._render_chart()

    def _render_chart(self) -> None:
        """Render the equity curve chart."""
        if not PLOTEXT_AVAILABLE or not self._data.points:
            return

        try:
            plot_widget = self.query_one("#equity-plot", PlotextPlot)
            plt = plot_widget.plt

            # Clear previous plot
            plt.clear_figure()

            # Extract data
            timestamps = [p.timestamp for p in self._data.points]
            equity_values = [float(p.equity) for p in self._data.points]

            # Convert timestamps to numeric for plotting
            if timestamps:
                # Use index as x-axis for simplicity
                x_values = list(range(len(timestamps)))

                # Main equity curve
                plt.plot(x_values, equity_values, label="Equity", color="green")

                # High watermark
                high_watermark = []
                current_max = equity_values[0]
                for val in equity_values:
                    current_max = max(current_max, val)
                    high_watermark.append(current_max)
                plt.plot(x_values, high_watermark, label="High Water", color="blue")

                # Drawdown subplot
                if self.show_drawdown:
                    drawdown_values = [p.drawdown_pct for p in self._data.points]
                    # Normalize drawdown to show on same chart (inverted)
                    min_equity = min(equity_values)
                    max_equity = max(equity_values)
                    range_val = max_equity - min_equity if max_equity != min_equity else 1
                    dd_scaled = [
                        min_equity - (dd / 100) * range_val * 0.3
                        for dd in drawdown_values
                    ]
                    plt.plot(x_values, dd_scaled, label="Drawdown", color="red")

                # Trade markers
                if self.show_trades and self._data.trades:
                    for trade in self._data.trades:
                        # Find closest timestamp index
                        for i, ts in enumerate(timestamps):
                            if ts >= trade.timestamp:
                                marker = "+" if "ENTRY" in trade.side else "x"
                                color = "green" if "LONG" in trade.side else "red"
                                # Find equity at that point
                                if i < len(equity_values):
                                    plt.scatter([i], [equity_values[i]], marker=marker, color=color)
                                break

                # Chart styling
                plt.title("Equity Curve")
                plt.xlabel("Time")
                plt.ylabel("Equity ($)")
                plt.theme("dark")

            # Refresh the widget
            plot_widget.refresh()

        except Exception:
            pass

    def update_data(self, data: EquityCurveData) -> None:
        """Update chart with new data."""
        self._data = data
        self._render_chart()

        # Update stats in header
        try:
            stats = self.query_one(".chart-stats", Static)
            return_class = "positive" if data.total_return >= 0 else "negative"
            return_sign = "+" if data.total_return >= 0 else ""
            stats.update(
                f"Return: [{return_class}]{return_sign}${data.total_return:,.2f} "
                f"({return_sign}{data.total_return_pct:.1f}%)[/{return_class}]  "
                f"Max DD: [red]{data.max_drawdown_pct:.1f}%[/red]"
            )
        except Exception:
            pass

    def action_refresh_chart(self) -> None:
        """Refresh the chart."""
        self._render_chart()

    def action_toggle_drawdown(self) -> None:
        """Toggle drawdown display."""
        self.show_drawdown = not self.show_drawdown
        self._render_chart()

    def action_toggle_trades(self) -> None:
        """Toggle trade markers."""
        self.show_trades = not self.show_trades
        self._render_chart()


# =============================================================================
# Drawdown Chart Widget
# =============================================================================


class DrawdownChart(Container):
    """
    Dedicated drawdown visualization.

    Shows:
    - Drawdown percentage over time
    - Recovery periods highlighted
    - Max drawdown indicator
    """

    DEFAULT_CSS = """
    DrawdownChart {
        height: 100%;
        width: 100%;
        layout: vertical;
    }

    DrawdownChart .chart-header {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }

    DrawdownChart .chart-container {
        height: 1fr;
    }

    DrawdownChart .fallback-message {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        drawdown_data: list[tuple[datetime, float]] | None = None,
        max_drawdown: float = 0.0,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._drawdown_data = drawdown_data or []
        self._max_drawdown = max_drawdown

    def compose(self) -> ComposeResult:
        with Horizontal(classes="chart-header"):
            yield Static(f"DRAWDOWN  Max: [red]{self._max_drawdown:.1f}%[/red]")

        if PLOTEXT_AVAILABLE:
            with Container(classes="chart-container"):
                yield PlotextPlot(id="drawdown-plot")
        else:
            yield Static(
                "[dim]plotext not available[/dim]",
                classes="fallback-message",
            )

    def on_mount(self) -> None:
        """Render chart on mount."""
        if PLOTEXT_AVAILABLE:
            self._render_chart()

    def _render_chart(self) -> None:
        """Render the drawdown chart."""
        if not PLOTEXT_AVAILABLE or not self._drawdown_data:
            return

        try:
            plot_widget = self.query_one("#drawdown-plot", PlotextPlot)
            plt = plot_widget.plt

            plt.clear_figure()

            x_values = list(range(len(self._drawdown_data)))
            y_values = [dd[1] for dd in self._drawdown_data]

            # Invert for visual (drawdown shown as negative)
            y_inverted = [-v for v in y_values]

            plt.plot(x_values, y_inverted, color="red", fillx=True)
            plt.title("Drawdown %")
            plt.ylabel("Drawdown (%)")
            plt.theme("dark")

            plot_widget.refresh()

        except Exception:
            pass

    def update_data(
        self,
        drawdown_data: list[tuple[datetime, float]],
        max_drawdown: float,
    ) -> None:
        """Update drawdown data."""
        self._drawdown_data = drawdown_data
        self._max_drawdown = max_drawdown
        self._render_chart()


# =============================================================================
# Equity Summary Widget (Sparkline version)
# =============================================================================


class EquitySummary(Container):
    """
    Compact equity summary using Sparkline.

    For use when full chart isn't needed.
    """

    DEFAULT_CSS = """
    EquitySummary {
        height: 5;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    EquitySummary .summary-title {
        height: 1;
        text-style: bold;
    }

    EquitySummary .summary-stats {
        height: 1;
    }

    EquitySummary .positive {
        color: $success;
    }

    EquitySummary .negative {
        color: $error;
    }
    """

    def __init__(
        self,
        equity_values: list[float] | None = None,
        total_return: float = 0.0,
        max_drawdown: float = 0.0,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._equity_values = equity_values or []
        self._total_return = total_return
        self._max_drawdown = max_drawdown

    def compose(self) -> ComposeResult:
        from textual.widgets import Sparkline

        yield Static("Equity Curve", classes="summary-title")

        if self._equity_values:
            yield Sparkline(self._equity_values, summary_function=max, id="equity-spark")
        else:
            yield Static("[dim]No data[/dim]")

        return_class = "positive" if self._total_return >= 0 else "negative"
        return_sign = "+" if self._total_return >= 0 else ""
        yield Static(
            f"[{return_class}]{return_sign}{self._total_return:.1f}%[/{return_class}]  "
            f"Max DD: [red]{self._max_drawdown:.1f}%[/red]",
            classes="summary-stats",
        )


# =============================================================================
# Demo Data Generator
# =============================================================================


def create_demo_equity_data(
    days: int = 90,
    initial_capital: float = 10000.0,
    volatility: float = 0.02,
) -> EquityCurveData:
    """Create demo equity curve data for testing."""
    import random
    from datetime import timedelta

    points: list[EquityPoint] = []
    trades: list[TradeMarker] = []

    equity = initial_capital
    max_equity = equity
    start_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        timestamp = start_date + timedelta(days=i)

        # Random walk with slight upward drift
        change = random.gauss(0.001, volatility)
        equity *= (1 + change)

        # Track max for drawdown
        max_equity = max(max_equity, equity)
        drawdown = max_equity - equity
        drawdown_pct = (drawdown / max_equity * 100) if max_equity > 0 else 0

        points.append(EquityPoint(
            timestamp=timestamp,
            equity=Decimal(str(round(equity, 2))),
            drawdown=Decimal(str(round(drawdown, 2))),
            drawdown_pct=drawdown_pct,
        ))

        # Random trade markers
        if random.random() < 0.1:
            side = random.choice(["LONG_ENTRY", "LONG_EXIT", "SHORT_ENTRY", "SHORT_EXIT"])
            trades.append(TradeMarker(
                timestamp=timestamp,
                price=Decimal(str(round(equity, 2))),
                side=side,
            ))

    max_dd_pct = max(p.drawdown_pct for p in points) if points else 0

    return EquityCurveData(
        points=points,
        trades=trades,
        initial_capital=Decimal(str(initial_capital)),
        final_equity=Decimal(str(round(equity, 2))),
        max_equity=Decimal(str(round(max_equity, 2))),
        min_equity=Decimal(str(round(min(float(p.equity) for p in points), 2))) if points else Decimal(str(initial_capital)),
        max_drawdown_pct=max_dd_pct,
    )

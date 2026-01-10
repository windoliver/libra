"""
Price Chart Widget.

Displays historical price data as a candlestick or line chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Label, LoadingIndicator, Select, Static

try:
    from textual_plotext import PlotextPlot
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False


if TYPE_CHECKING:
    from libra.gateways.fetcher import Bar


@dataclass
class PriceChartData:
    """Data for price chart."""

    symbol: str
    interval: str
    bars: list[Bar] = field(default_factory=list)
    provider: str = "yfinance"

    @property
    def dates(self) -> list[datetime]:
        """Get dates from bars."""
        return [b.datetime for b in self.bars]

    @property
    def opens(self) -> list[float]:
        """Get open prices."""
        return [float(b.open) for b in self.bars]

    @property
    def highs(self) -> list[float]:
        """Get high prices."""
        return [float(b.high) for b in self.bars]

    @property
    def lows(self) -> list[float]:
        """Get low prices."""
        return [float(b.low) for b in self.bars]

    @property
    def closes(self) -> list[float]:
        """Get close prices."""
        return [float(b.close) for b in self.bars]

    @property
    def volumes(self) -> list[float]:
        """Get volumes."""
        return [float(b.volume) for b in self.bars]

    @property
    def last_price(self) -> Decimal | None:
        """Get last closing price."""
        return self.bars[-1].close if self.bars else None

    @property
    def price_change(self) -> Decimal | None:
        """Get price change from first to last bar."""
        if len(self.bars) < 2:
            return None
        return self.bars[-1].close - self.bars[0].close

    @property
    def price_change_pct(self) -> float | None:
        """Get price change percentage."""
        if len(self.bars) < 2 or self.bars[0].close == 0:
            return None
        return float((self.bars[-1].close - self.bars[0].close) / self.bars[0].close * 100)


class PriceChartWidget(Container):
    """
    Price chart widget for displaying historical OHLCV data.

    Features:
    - Candlestick or line chart display
    - Multiple timeframe support
    - Volume overlay
    - Price statistics panel
    """

    DEFAULT_CSS = """
    PriceChartWidget {
        height: 1fr;
        padding: 1;
    }

    PriceChartWidget .chart-header {
        height: 3;
        layout: horizontal;
    }

    PriceChartWidget .chart-title {
        width: 1fr;
        text-style: bold;
    }

    PriceChartWidget .interval-select {
        width: 12;
        margin-right: 1;
    }

    PriceChartWidget .chart-type-select {
        width: 14;
    }

    PriceChartWidget .chart-area {
        height: 1fr;
        min-height: 10;
    }

    PriceChartWidget #price-plot {
        height: 1fr;
    }

    PriceChartWidget .stats-row {
        height: 3;
        layout: horizontal;
        padding-top: 1;
    }

    PriceChartWidget .stat-item {
        width: 1fr;
        text-align: center;
    }

    PriceChartWidget .stat-label {
        color: $text-muted;
    }

    PriceChartWidget .stat-value {
        text-style: bold;
    }

    PriceChartWidget .price-up {
        color: $success;
    }

    PriceChartWidget .price-down {
        color: $error;
    }

    PriceChartWidget .loading-container {
        height: 100%;
        align: center middle;
    }

    PriceChartWidget .no-data {
        height: 100%;
        align: center middle;
        color: $text-muted;
    }
    """

    # Reactive data
    chart_data: reactive[PriceChartData | None] = reactive(None, init=False)
    loading: reactive[bool] = reactive(False, init=False)
    interval: reactive[str] = reactive("1d", init=False)
    chart_type: reactive[str] = reactive("line", init=False)

    class DataRequested(Message):
        """Message requesting data load."""

        def __init__(self, symbol: str, interval: str, provider: str) -> None:
            self.symbol = symbol
            self.interval = interval
            self.provider = provider
            super().__init__()

    def __init__(
        self,
        symbol: str = "",
        interval: str = "1d",
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._symbol = symbol
        self._interval = interval

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with Horizontal(classes="chart-header"):
            yield Label(
                f"[bold]{self._symbol or 'Select Symbol'}[/bold]",
                id="chart-title",
                classes="chart-title",
            )
            yield Select(
                options=[
                    ("1m", "1m"),
                    ("5m", "5m"),
                    ("15m", "15m"),
                    ("1h", "1h"),
                    ("4h", "4h"),
                    ("1d", "1d"),
                    ("1W", "1W"),
                    ("1M", "1M"),
                ],
                value=self._interval,
                id="interval-select",
                classes="interval-select",
            )
            yield Select(
                options=[
                    ("Line", "line"),
                    ("Candle", "candle"),
                ],
                value="line",
                id="chart-type-select",
                classes="chart-type-select",
            )

        with Container(classes="chart-area"):
            if PLOTEXT_AVAILABLE:
                yield PlotextPlot(id="price-plot")
            else:
                yield Static(
                    "[dim]Chart display requires textual-plotext.[/dim]\n"
                    "[dim]Install with: pip install textual-plotext[/dim]",
                    classes="no-data",
                )

        with Horizontal(classes="stats-row"):
            yield Static(
                "[dim]Open[/dim]\n--",
                id="stat-open",
                classes="stat-item",
            )
            yield Static(
                "[dim]High[/dim]\n--",
                id="stat-high",
                classes="stat-item",
            )
            yield Static(
                "[dim]Low[/dim]\n--",
                id="stat-low",
                classes="stat-item",
            )
            yield Static(
                "[dim]Close[/dim]\n--",
                id="stat-close",
                classes="stat-item",
            )
            yield Static(
                "[dim]Change[/dim]\n--",
                id="stat-change",
                classes="stat-item",
            )
            yield Static(
                "[dim]Volume[/dim]\n--",
                id="stat-volume",
                classes="stat-item",
            )

    @on(Select.Changed, "#interval-select")
    def _on_interval_changed(self, event: Select.Changed) -> None:
        """Handle interval change."""
        if event.value:
            self.interval = str(event.value)
            if self.chart_data:
                self.post_message(
                    self.DataRequested(
                        self.chart_data.symbol,
                        self.interval,
                        self.chart_data.provider,
                    )
                )

    @on(Select.Changed, "#chart-type-select")
    def _on_chart_type_changed(self, event: Select.Changed) -> None:
        """Handle chart type change."""
        if event.value:
            self.chart_type = str(event.value)
            self._update_chart()

    def watch_chart_data(self, data: PriceChartData | None) -> None:
        """Update chart when data changes."""
        self._update_chart()
        self._update_stats()
        self._update_title()

    def _update_title(self) -> None:
        """Update chart title."""
        try:
            title = self.query_one("#chart-title", Label)
            if self.chart_data:
                symbol = self.chart_data.symbol
                change_pct = self.chart_data.price_change_pct
                if change_pct is not None:
                    color = "green" if change_pct >= 0 else "red"
                    sign = "+" if change_pct >= 0 else ""
                    title.update(
                        f"[bold]{symbol}[/bold] "
                        f"[{color}]{sign}{change_pct:.2f}%[/{color}]"
                    )
                else:
                    title.update(f"[bold]{symbol}[/bold]")
            else:
                title.update("[bold]Select Symbol[/bold]")
        except Exception:
            pass

    def _update_chart(self) -> None:
        """Update the chart display."""
        if not PLOTEXT_AVAILABLE:
            return

        try:
            plot = self.query_one("#price-plot", PlotextPlot)
            plot.plt.clear_figure()

            if not self.chart_data or not self.chart_data.bars:
                plot.plt.title("No data")
                plot.refresh()
                return

            data = self.chart_data
            x_values = list(range(len(data.bars)))

            if self.chart_type == "candle":
                # Candlestick chart using bars
                # plotext candlestick requires string dates
                date_strs = [str(i) for i in x_values]
                plot.plt.candlestick(
                    date_strs,
                    {"Open": data.opens, "Close": data.closes, "High": data.highs, "Low": data.lows},
                )
            else:
                # Line chart
                plot.plt.plot(x_values, data.closes, label="Close")

            plot.plt.title(f"{data.symbol} - {data.interval}")
            plot.plt.xlabel("Time")
            plot.plt.ylabel("Price")
            plot.refresh()

        except Exception as e:
            # Log plotting errors
            self.log.error(f"Chart error: {e}")

    def _update_stats(self) -> None:
        """Update statistics display."""
        try:
            if not self.chart_data or not self.chart_data.bars:
                return

            data = self.chart_data
            last_bar = data.bars[-1]

            # Update each stat
            self._update_stat("stat-open", "Open", f"${float(last_bar.open):,.2f}")
            self._update_stat("stat-high", "High", f"${float(last_bar.high):,.2f}")
            self._update_stat("stat-low", "Low", f"${float(last_bar.low):,.2f}")
            self._update_stat("stat-close", "Close", f"${float(last_bar.close):,.2f}")

            # Volume
            volume = float(last_bar.volume)
            if volume >= 1_000_000_000:
                vol_str = f"{volume/1_000_000_000:.1f}B"
            elif volume >= 1_000_000:
                vol_str = f"{volume/1_000_000:.1f}M"
            elif volume >= 1_000:
                vol_str = f"{volume/1_000:.1f}K"
            else:
                vol_str = f"{volume:.0f}"
            self._update_stat("stat-volume", "Volume", vol_str)

            # Change
            change_pct = data.price_change_pct
            if change_pct is not None:
                color = "green" if change_pct >= 0 else "red"
                sign = "+" if change_pct >= 0 else ""
                change_str = f"[{color}]{sign}{change_pct:.2f}%[/{color}]"
            else:
                change_str = "--"
            self._update_stat("stat-change", "Change", change_str)

        except Exception:
            pass

    def _update_stat(self, stat_id: str, label: str, value: str) -> None:
        """Update a single stat display."""
        try:
            stat = self.query_one(f"#{stat_id}", Static)
            stat.update(f"[dim]{label}[/dim]\n{value}")
        except Exception:
            pass

    def set_data(self, data: PriceChartData) -> None:
        """Set chart data."""
        self.chart_data = data

    def set_symbol(self, symbol: str, provider: str = "yfinance") -> None:
        """Request data for a new symbol."""
        self._symbol = symbol
        self.post_message(self.DataRequested(symbol, self.interval, provider))

    def clear(self) -> None:
        """Clear the chart."""
        self.chart_data = None

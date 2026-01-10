"""
Economic Data Widget.

Displays FRED economic data series with charts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable, Input, Label, Select, Static

try:
    from textual_plotext import PlotextPlot
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False


if TYPE_CHECKING:
    from libra.gateways.openbb.fetchers import EconomicDataPoint


# Common economic series
POPULAR_SERIES = [
    ("GDP", "GDP", "Gross Domestic Product"),
    ("UNRATE", "Unemployment", "Unemployment Rate"),
    ("CPIAUCSL", "CPI", "Consumer Price Index"),
    ("FEDFUNDS", "Fed Funds", "Federal Funds Rate"),
    ("DGS10", "10Y Treasury", "10-Year Treasury Yield"),
    ("SP500", "S&P 500", "S&P 500 Index"),
    ("VIXCLS", "VIX", "CBOE Volatility Index"),
    ("M2SL", "M2 Supply", "M2 Money Supply"),
    ("UMCSENT", "Consumer Sentiment", "U of Michigan Consumer Sentiment"),
    ("INDPRO", "Industrial Production", "Industrial Production Index"),
    ("HOUST", "Housing Starts", "New Housing Units Started"),
    ("PAYEMS", "Nonfarm Payrolls", "Total Nonfarm Payrolls"),
]


@dataclass
class EconomicChartData:
    """Container for economic data."""

    series_id: str
    series_name: str
    data_points: list[EconomicDataPoint] = field(default_factory=list)
    transform: str | None = None

    @property
    def dates(self) -> list[date]:
        """Get dates from data points."""
        return [p.date for p in self.data_points]

    @property
    def values(self) -> list[float]:
        """Get values from data points."""
        return [float(p.value) for p in self.data_points]

    @property
    def latest_value(self) -> Decimal | None:
        """Get latest value."""
        return self.data_points[-1].value if self.data_points else None

    @property
    def latest_date(self) -> date | None:
        """Get latest date."""
        return self.data_points[-1].date if self.data_points else None


class EconomicDataWidget(Container):
    """
    Economic data widget for displaying FRED series.

    Features:
    - Popular series quick select
    - Custom series ID input
    - Time series chart
    - Data transformation options (percent change, etc.)
    """

    DEFAULT_CSS = """
    EconomicDataWidget {
        height: 100%;
    }

    EconomicDataWidget .economic-header {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }

    EconomicDataWidget .series-title {
        width: 1fr;
        text-style: bold;
    }

    EconomicDataWidget .series-input {
        width: 15;
        margin-right: 1;
    }

    EconomicDataWidget .series-select {
        width: 20;
        margin-right: 1;
    }

    EconomicDataWidget .transform-select {
        width: 15;
    }

    EconomicDataWidget .chart-container {
        height: 1fr;
        min-height: 12;
    }

    EconomicDataWidget .stats-row {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }

    EconomicDataWidget .stat-item {
        width: 1fr;
        text-align: center;
    }

    EconomicDataWidget .quick-series {
        height: auto;
        padding: 1;
        layout: horizontal;
    }

    EconomicDataWidget .quick-series-btn {
        margin-right: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    EconomicDataWidget .quick-series-btn:hover {
        background: $primary;
        color: $text;
    }

    EconomicDataWidget .no-data {
        height: 100%;
        align: center middle;
        color: $text-muted;
    }
    """

    # Reactive data
    data: reactive[EconomicChartData | None] = reactive(None, init=False)
    transform: reactive[str | None] = reactive(None, init=False)

    class DataRequested(Message):
        """Message requesting economic data load."""

        def __init__(
            self, series_id: str, transform: str | None, provider: str
        ) -> None:
            self.series_id = series_id
            self.transform = transform
            self.provider = provider
            super().__init__()

    def __init__(
        self,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with Horizontal(classes="economic-header"):
            yield Label(
                "[bold]Economic Data (FRED)[/bold]",
                id="series-title",
                classes="series-title",
            )
            yield Input(
                placeholder="Series ID",
                id="series-input",
                classes="series-input",
            )
            yield Select(
                options=[(name, sid) for sid, name, _ in POPULAR_SERIES],
                value="GDP",
                id="series-select",
                classes="series-select",
            )
            yield Select(
                options=[
                    ("Raw", None),
                    ("% Change", "pch"),
                    ("YoY %", "pc1"),
                    ("Log", "log"),
                ],
                value=None,
                id="transform-select",
                classes="transform-select",
            )

        with Container(classes="chart-container"):
            if PLOTEXT_AVAILABLE:
                yield PlotextPlot(id="economic-plot")
            else:
                yield Static(
                    "[dim]Chart display requires textual-plotext.[/dim]",
                    classes="no-data",
                )

        with Horizontal(classes="stats-row"):
            yield Static(
                "[dim]Latest Value[/dim]\n--",
                id="stat-value",
                classes="stat-item",
            )
            yield Static(
                "[dim]Latest Date[/dim]\n--",
                id="stat-date",
                classes="stat-item",
            )
            yield Static(
                "[dim]Data Points[/dim]\n--",
                id="stat-count",
                classes="stat-item",
            )
            yield Static(
                "[dim]Min[/dim]\n--",
                id="stat-min",
                classes="stat-item",
            )
            yield Static(
                "[dim]Max[/dim]\n--",
                id="stat-max",
                classes="stat-item",
            )

    @on(Select.Changed, "#series-select")
    def _on_series_changed(self, event: Select.Changed) -> None:
        """Handle series selection change."""
        if event.value:
            series_id = str(event.value)
            self.post_message(
                self.DataRequested(series_id, self.transform, "fred")
            )

    @on(Input.Submitted, "#series-input")
    def _on_series_input(self, event: Input.Submitted) -> None:
        """Handle custom series ID input."""
        if event.value:
            series_id = event.value.upper().strip()
            self.post_message(
                self.DataRequested(series_id, self.transform, "fred")
            )

    @on(Select.Changed, "#transform-select")
    def _on_transform_changed(self, event: Select.Changed) -> None:
        """Handle transform change."""
        self.transform = event.value if event.value else None
        if self.data:
            self.post_message(
                self.DataRequested(self.data.series_id, self.transform, "fred")
            )

    def watch_data(self, data: EconomicChartData | None) -> None:
        """Update display when data changes."""
        if data:
            self._update_chart(data)
            self._update_stats(data)
            self._update_header(data)

    def _update_header(self, data: EconomicChartData) -> None:
        """Update header with series name."""
        try:
            header = self.query_one("#series-title", Label)
            header.update(f"[bold]{data.series_name}[/bold] ({data.series_id})")
        except Exception:
            pass

    def _update_chart(self, data: EconomicChartData) -> None:
        """Update the chart display."""
        if not PLOTEXT_AVAILABLE:
            return

        try:
            plot = self.query_one("#economic-plot", PlotextPlot)
            plot.plt.clear_figure()

            if not data.data_points:
                plot.plt.title("No data")
                plot.refresh()
                return

            x_values = list(range(len(data.data_points)))
            y_values = data.values

            plot.plt.plot(x_values, y_values, label=data.series_id)
            plot.plt.fill(x_values, y_values)

            title = data.series_name
            if data.transform:
                title += f" ({data.transform})"
            plot.plt.title(title)
            plot.plt.xlabel("Time")
            plot.plt.ylabel("Value")
            plot.refresh()

        except Exception:
            pass

    def _update_stats(self, data: EconomicChartData) -> None:
        """Update statistics display."""
        try:
            if not data.data_points:
                return

            # Latest value
            latest = data.latest_value
            if latest:
                self._update_stat(
                    "stat-value", "Latest Value", f"{float(latest):,.2f}"
                )

            # Latest date
            latest_date = data.latest_date
            if latest_date:
                self._update_stat(
                    "stat-date", "Latest Date", latest_date.strftime("%Y-%m-%d")
                )

            # Count
            self._update_stat(
                "stat-count", "Data Points", str(len(data.data_points))
            )

            # Min/Max
            values = data.values
            if values:
                self._update_stat("stat-min", "Min", f"{min(values):,.2f}")
                self._update_stat("stat-max", "Max", f"{max(values):,.2f}")

        except Exception:
            pass

    def _update_stat(self, stat_id: str, label: str, value: str) -> None:
        """Update a single stat display."""
        try:
            stat = self.query_one(f"#{stat_id}", Static)
            stat.update(f"[dim]{label}[/dim]\n{value}")
        except Exception:
            pass

    def set_data(self, data: EconomicChartData) -> None:
        """Set economic data."""
        self.data = data

    def load_series(self, series_id: str, transform: str | None = None) -> None:
        """Request loading a specific series."""
        self.post_message(self.DataRequested(series_id, transform, "fred"))

    def clear(self) -> None:
        """Clear the display."""
        self.data = None

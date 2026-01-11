"""
Whale Alerts TUI Widget.

Displays real-time whale activity signals in the trading dashboard.

Features:
- Live signal feed with auto-scroll
- Color-coded by direction (bullish/bearish)
- Signal filtering by type and strength
- Historical signal browser

See: https://github.com/windoliver/libra/issues/38
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable, Select, Static


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class WhaleAlertData:
    """Display data for a whale alert."""

    signal_id: str
    signal_type: str
    symbol: str
    direction: str  # "bullish", "bearish", "neutral"
    strength: float
    value_usd: Decimal
    source: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

    @property
    def strength_pct(self) -> str:
        """Strength as percentage string."""
        return f"{self.strength * 100:.0f}%"

    @property
    def value_formatted(self) -> str:
        """Format USD value with K/M suffix."""
        value = float(self.value_usd)
        if value >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        elif value >= 1_000:
            return f"${value / 1_000:.0f}K"
        return f"${value:.0f}"

    @property
    def time_formatted(self) -> str:
        """Format timestamp as HH:MM:SS."""
        return self.timestamp.strftime("%H:%M:%S")

    @property
    def direction_icon(self) -> str:
        """Get icon for direction."""
        if self.direction == "bullish":
            return "▲"
        elif self.direction == "bearish":
            return "▼"
        return "●"

    @property
    def type_short(self) -> str:
        """Short signal type name."""
        type_map = {
            # Crypto order flow
            "order_imbalance": "IMBAL",
            "large_wall": "WALL",
            "ladder_wall": "LADDER",
            "volume_spike": "VOL",
            "large_trade": "TRADE",
            # Crypto on-chain
            "exchange_inflow": "IN",
            "exchange_outflow": "OUT",
            "whale_transfer": "XFER",
            "dormant_activation": "DORM",
            "mint_burn": "MINT",
            # Prediction markets
            "pm_large_bet": "PM-BET",
            "pm_position_change": "PM-POS",
            "pm_market_move": "PM-MOV",
            "pm_smart_money": "PM-SM",
            # Stocks/equities
            "options_unusual": "OPT",
            "options_sweep": "SWEEP",
            "dark_pool": "DARK",
            "block_trade": "BLOCK",
            "insider_filing": "INSDR",
            "inst_13f": "13F",
        }
        return type_map.get(self.signal_type, self.signal_type[:5].upper())


@dataclass
class WhaleAlertsDashboardData:
    """Data for the whale alerts dashboard."""

    alerts: list[WhaleAlertData] = field(default_factory=list)
    total_signals: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    total_value_usd: Decimal = Decimal("0")
    connected_sources: list[str] = field(default_factory=list)


# =============================================================================
# Widgets
# =============================================================================


class AlertSummaryCard(Static):
    """Summary card showing signal statistics."""

    DEFAULT_CSS = """
    AlertSummaryCard {
        height: 5;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    AlertSummaryCard .title {
        text-style: bold;
        color: $text;
    }

    AlertSummaryCard .value {
        text-style: bold;
    }

    AlertSummaryCard .bullish {
        color: $success;
    }

    AlertSummaryCard .bearish {
        color: $error;
    }

    AlertSummaryCard .neutral {
        color: $warning;
    }
    """

    def __init__(
        self,
        title: str = "Signals",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._title = title
        self._total = 0
        self._bullish = 0
        self._bearish = 0
        self._value = Decimal("0")

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="title")
        yield Static("", id="summary-content")

    def update_stats(
        self,
        total: int,
        bullish: int,
        bearish: int,
        value: Decimal,
    ) -> None:
        """Update summary statistics."""
        self._total = total
        self._bullish = bullish
        self._bearish = bearish
        self._value = value

        # Format value
        v = float(value)
        if v >= 1_000_000:
            value_str = f"${v / 1_000_000:.1f}M"
        elif v >= 1_000:
            value_str = f"${v / 1_000:.0f}K"
        else:
            value_str = f"${v:.0f}"

        content = (
            f"Total: [bold]{total}[/bold] | "
            f"[green]▲ {bullish}[/green] | "
            f"[red]▼ {bearish}[/red] | "
            f"Value: [bold]{value_str}[/bold]"
        )

        try:
            self.query_one("#summary-content", Static).update(content)
        except Exception:
            pass


class WhaleAlertsTable(DataTable):
    """Table displaying whale alerts."""

    DEFAULT_CSS = """
    WhaleAlertsTable {
        height: 100%;
        border: round $primary-darken-2;
    }

    WhaleAlertsTable > .datatable--header {
        background: $primary-darken-1;
        text-style: bold;
    }
    """

    COLUMNS: ClassVar[list[tuple[str, int]]] = [
        ("Time", 10),
        ("Type", 8),
        ("Symbol", 12),
        ("Dir", 5),
        ("Str", 6),
        ("Value", 10),
        ("Source", 10),
    ]

    class AlertSelected(Message):
        """Emitted when an alert is selected."""

        def __init__(self, alert: WhaleAlertData) -> None:
            super().__init__()
            self.alert = alert

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id, cursor_type="row")
        self._alerts: dict[str, WhaleAlertData] = {}

    def on_mount(self) -> None:
        """Set up table columns."""
        for name, width in self.COLUMNS:
            self.add_column(name, width=width)

    def update_alerts(self, alerts: list[WhaleAlertData]) -> None:
        """Update table with new alerts."""
        self.clear()
        self._alerts.clear()

        for alert in alerts:
            self._add_alert_row(alert)

    def add_alert(self, alert: WhaleAlertData) -> None:
        """Add a single alert to the top of the table."""
        if alert.signal_id in self._alerts:
            return

        self._alerts[alert.signal_id] = alert

        # Get direction styling
        if alert.direction == "bullish":
            dir_style = "[green]▲ BUY[/green]"
        elif alert.direction == "bearish":
            dir_style = "[red]▼ SELL[/red]"
        else:
            dir_style = "[yellow]● HOLD[/yellow]"

        # Strength color
        if alert.strength >= 0.7:
            str_style = f"[bold green]{alert.strength_pct}[/bold green]"
        elif alert.strength >= 0.5:
            str_style = f"[yellow]{alert.strength_pct}[/yellow]"
        else:
            str_style = f"[dim]{alert.strength_pct}[/dim]"

        self.add_row(
            alert.time_formatted,
            alert.type_short,
            alert.symbol,
            dir_style,
            str_style,
            alert.value_formatted,
            alert.source[:8],
            key=alert.signal_id,
        )

    def _add_alert_row(self, alert: WhaleAlertData) -> None:
        """Add a single alert row."""
        self._alerts[alert.signal_id] = alert

        # Get direction styling
        if alert.direction == "bullish":
            dir_style = "[green]▲ BUY[/green]"
        elif alert.direction == "bearish":
            dir_style = "[red]▼ SELL[/red]"
        else:
            dir_style = "[yellow]● HOLD[/yellow]"

        # Strength color
        if alert.strength >= 0.7:
            str_style = f"[bold green]{alert.strength_pct}[/bold green]"
        elif alert.strength >= 0.5:
            str_style = f"[yellow]{alert.strength_pct}[/yellow]"
        else:
            str_style = f"[dim]{alert.strength_pct}[/dim]"

        self.add_row(
            alert.time_formatted,
            alert.type_short,
            alert.symbol,
            dir_style,
            str_style,
            alert.value_formatted,
            alert.source[:8],
            key=alert.signal_id,
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key and event.row_key.value in self._alerts:
            alert = self._alerts[event.row_key.value]
            self.post_message(self.AlertSelected(alert))


class AlertDetailPanel(Static):
    """Panel showing details of selected alert."""

    DEFAULT_CSS = """
    AlertDetailPanel {
        height: 10;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    AlertDetailPanel .detail-title {
        text-style: bold;
        margin-bottom: 1;
    }

    AlertDetailPanel .detail-row {
        height: 1;
    }

    AlertDetailPanel .label {
        color: $text-muted;
        width: 12;
    }

    AlertDetailPanel .value {
        color: $text;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._alert: WhaleAlertData | None = None

    def compose(self) -> ComposeResult:
        yield Static("Signal Details", classes="detail-title")
        yield Static("[dim]Select a signal to view details[/dim]", id="detail-content")

    def show_alert(self, alert: WhaleAlertData) -> None:
        """Display alert details."""
        self._alert = alert

        # Build detail content
        lines = [
            f"[bold]{alert.signal_type.upper()}[/bold] - {alert.symbol}",
            "",
            f"Direction:  {alert.direction_icon} {alert.direction.upper()}",
            f"Strength:   {alert.strength_pct}",
            f"Value:      {alert.value_formatted}",
            f"Source:     {alert.source}",
            f"Time:       {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        # Add metadata
        if alert.metadata:
            lines.append("")
            for key, value in list(alert.metadata.items())[:3]:
                lines.append(f"{key}: {value}")

        try:
            self.query_one("#detail-content", Static).update("\n".join(lines))
        except Exception:
            pass


class SourceStatusBar(Static):
    """Status bar showing connected data sources."""

    DEFAULT_CSS = """
    SourceStatusBar {
        height: 3;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    SourceStatusBar .source-label {
        margin-right: 2;
    }

    SourceStatusBar .connected {
        color: $success;
    }

    SourceStatusBar .disconnected {
        color: $error;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._sources: dict[str, bool] = {
            "crypto": True,
            "onchain": False,
            "pred_mkts": False,
            "stocks": False,
        }

    def compose(self) -> ComposeResult:
        yield Static("Sources: ", id="sources-content")

    def update_sources(self, connected: list[str]) -> None:
        """Update connected sources."""
        self._sources = {
            "crypto": any(s in connected for s in ["order_book", "order_flow", "trades"]),
            "onchain": any(s in connected for s in ["whale_alert", "dune", "nansen"]),
            "pred_mkts": any(s in connected for s in ["polymarket", "kalshi", "manifold"]),
            "stocks": any(s in connected for s in ["unusual_whales", "sec_edgar", "finra", "options_flow"]),
        }
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the sources display."""
        parts = []
        for source, is_connected in self._sources.items():
            icon = "●" if is_connected else "○"
            color = "green" if is_connected else "dim"
            name = source.replace("_", " ").title()
            parts.append(f"[{color}]{icon} {name}[/{color}]")

        content = "Sources: " + "  ".join(parts)

        try:
            self.query_one("#sources-content", Static).update(content)
        except Exception:
            pass


class WhaleAlertsDashboard(Container):
    """
    Complete whale alerts dashboard widget.

    Displays real-time whale activity signals with filtering and details.
    """

    DEFAULT_CSS = """
    WhaleAlertsDashboard {
        height: 1fr;
        min-height: 25;
        padding: 1;
    }

    WhaleAlertsDashboard #alerts-header {
        height: 5;
        margin-bottom: 1;
    }

    WhaleAlertsDashboard #alerts-main {
        height: 1fr;
        min-height: 15;
    }

    WhaleAlertsDashboard #alerts-table-container {
        width: 2fr;
        height: 1fr;
        min-height: 12;
        margin-right: 1;
    }

    WhaleAlertsDashboard #alerts-table {
        height: 1fr;
        min-height: 10;
    }

    WhaleAlertsDashboard #alerts-sidebar {
        width: 1fr;
        height: 1fr;
    }

    WhaleAlertsDashboard #filter-row {
        height: 3;
        margin-bottom: 1;
    }

    WhaleAlertsDashboard #filter-row Select {
        width: 20;
        margin-right: 1;
    }
    """

    # Reactive data
    data: reactive[WhaleAlertsDashboardData] = reactive(
        WhaleAlertsDashboardData, init=False
    )

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._data = WhaleAlertsDashboardData()

        # Cached widget refs
        self._table: WhaleAlertsTable | None = None
        self._summary: AlertSummaryCard | None = None
        self._detail: AlertDetailPanel | None = None
        self._sources: SourceStatusBar | None = None

    def compose(self) -> ComposeResult:
        # Header with summary and filters
        with Horizontal(id="alerts-header"):
            yield AlertSummaryCard(title="Whale Signals", id="alerts-summary")

        # Filter row
        with Horizontal(id="filter-row"):
            yield Select(
                [
                    ("All Types", "all"),
                    ("Crypto Flow", "order_flow"),
                    ("On-Chain", "onchain"),
                    ("Pred Markets", "prediction"),
                    ("Options", "options"),
                    ("Stocks", "stocks"),
                ],
                value="all",
                id="type-filter",
            )
            yield Select(
                [
                    ("All Directions", "all"),
                    ("Bullish", "bullish"),
                    ("Bearish", "bearish"),
                ],
                value="all",
                id="direction-filter",
            )

        # Main content
        with Horizontal(id="alerts-main"):
            with VerticalScroll(id="alerts-table-container"):
                yield WhaleAlertsTable(id="alerts-table")

            with Vertical(id="alerts-sidebar"):
                yield AlertDetailPanel(id="alert-detail")
                yield SourceStatusBar(id="source-status")

    def on_mount(self) -> None:
        """Cache widget references."""
        try:
            self._table = self.query_one("#alerts-table", WhaleAlertsTable)
            self._summary = self.query_one("#alerts-summary", AlertSummaryCard)
            self._detail = self.query_one("#alert-detail", AlertDetailPanel)
            self._sources = self.query_one("#source-status", SourceStatusBar)
        except Exception:
            pass

    def update_data(self, data: WhaleAlertsDashboardData) -> None:
        """Update dashboard with new data."""
        self._data = data

        # Ensure widgets are cached (may not be cached if called before on_mount)
        if not self._table:
            try:
                self._table = self.query_one("#alerts-table", WhaleAlertsTable)
            except Exception:
                pass
        if not self._summary:
            try:
                self._summary = self.query_one("#alerts-summary", AlertSummaryCard)
            except Exception:
                pass
        if not self._sources:
            try:
                self._sources = self.query_one("#source-status", SourceStatusBar)
            except Exception:
                pass

        # Update summary
        if self._summary:
            self._summary.update_stats(
                total=data.total_signals,
                bullish=data.bullish_count,
                bearish=data.bearish_count,
                value=data.total_value_usd,
            )

        # Update table
        if self._table:
            self._table.update_alerts(data.alerts)

        # Update sources
        if self._sources:
            self._sources.update_sources(data.connected_sources)

    def add_alert(self, alert: WhaleAlertData) -> None:
        """Add a single new alert."""
        self._data.alerts.insert(0, alert)
        self._data.total_signals += 1

        if alert.direction == "bullish":
            self._data.bullish_count += 1
        elif alert.direction == "bearish":
            self._data.bearish_count += 1

        self._data.total_value_usd += alert.value_usd

        # Update widgets
        if self._table:
            self._table.add_alert(alert)

        if self._summary:
            self._summary.update_stats(
                total=self._data.total_signals,
                bullish=self._data.bullish_count,
                bearish=self._data.bearish_count,
                value=self._data.total_value_usd,
            )

    def on_whale_alerts_table_alert_selected(
        self,
        event: WhaleAlertsTable.AlertSelected,
    ) -> None:
        """Handle alert selection from table."""
        if self._detail:
            self._detail.show_alert(event.alert)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter changes."""
        # Re-filter and update table
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Apply current filters to the alerts table."""
        try:
            type_filter = self.query_one("#type-filter", Select).value
            dir_filter = self.query_one("#direction-filter", Select).value
        except Exception:
            return

        filtered = self._data.alerts

        # Apply type filter
        if type_filter == "order_flow":
            filtered = [
                a for a in filtered
                if a.signal_type in (
                    "order_imbalance", "large_wall", "ladder_wall",
                    "large_trade", "volume_spike"
                )
            ]
        elif type_filter == "onchain":
            filtered = [
                a for a in filtered
                if a.signal_type in (
                    "exchange_inflow", "exchange_outflow", "whale_transfer",
                    "dormant_activation", "mint_burn"
                )
            ]
        elif type_filter == "prediction":
            filtered = [
                a for a in filtered
                if a.signal_type in (
                    "pm_large_bet", "pm_position_change",
                    "pm_market_move", "pm_smart_money"
                )
            ]
        elif type_filter == "options":
            filtered = [
                a for a in filtered
                if a.signal_type in ("options_unusual", "options_sweep")
            ]
        elif type_filter == "stocks":
            filtered = [
                a for a in filtered
                if a.signal_type in (
                    "dark_pool", "block_trade",
                    "insider_filing", "inst_13f"
                )
            ]

        # Apply direction filter
        if dir_filter != "all":
            filtered = [a for a in filtered if a.direction == dir_filter]

        # Update table
        if self._table:
            self._table.update_alerts(filtered)


# =============================================================================
# Demo Data
# =============================================================================


def create_demo_whale_alerts() -> WhaleAlertsDashboardData:
    """Create demo whale alerts data."""
    from datetime import timedelta
    import random

    now = datetime.now()
    alerts: list[WhaleAlertData] = []

    # Generate demo alerts - now includes all asset classes
    demo_signals = [
        # Crypto order flow
        ("order_imbalance", "BTC/USDT", "bullish", 0.75, 2500000, "order_book"),
        ("large_wall", "ETH/USDT", "bearish", 0.85, 1800000, "order_book"),
        ("volume_spike", "SOL/USDT", "bullish", 0.65, 850000, "order_book"),
        # Crypto on-chain
        ("exchange_inflow", "BTC", "bearish", 0.90, 15000000, "whale_alert"),
        ("whale_transfer", "ETH", "neutral", 0.70, 5200000, "whale_alert"),
        ("exchange_outflow", "BTC", "bullish", 0.80, 8500000, "whale_alert"),
        # Prediction markets
        ("pm_large_bet", "BTC $100k by Dec?", "bullish", 0.88, 150000, "polymarket"),
        ("pm_market_move", "US Election 2024", "bullish", 0.75, 500000, "polymarket"),
        ("pm_smart_money", "Fed Rate Cut Mar", "bearish", 0.92, 75000, "kalshi"),
        # Stocks - options
        ("options_sweep", "NVDA", "bullish", 0.92, 2500000, "unusual_whales"),
        ("options_unusual", "TSLA", "bearish", 0.78, 850000, "options_flow"),
        # Stocks - other
        ("dark_pool", "AAPL", "neutral", 0.65, 15000000, "finra"),
        ("insider_filing", "META", "bullish", 0.88, 5000000, "sec_edgar"),
        ("inst_13f", "MSFT", "bullish", 0.72, 500000000, "sec_edgar"),
        ("block_trade", "AMD", "bearish", 0.55, 8500000, "options_flow"),
    ]

    for i, (sig_type, symbol, direction, strength, value, source) in enumerate(demo_signals):
        alerts.append(WhaleAlertData(
            signal_id=f"sig_{i}_{random.randint(1000, 9999)}",
            signal_type=sig_type,
            symbol=symbol,
            direction=direction,
            strength=strength,
            value_usd=Decimal(str(value)),
            source=source,
            timestamp=now - timedelta(minutes=i * 3 + random.randint(0, 2)),
            metadata={
                "imbalance_ratio": "0.42" if sig_type == "order_imbalance" else None,
                "wall_price": "42,500" if "wall" in sig_type else None,
                "asset_class": (
                    "crypto" if sig_type.startswith(("order", "large", "volume", "exchange", "whale", "ladder"))
                    else "prediction_market" if sig_type.startswith("pm_")
                    else "stock" if sig_type in ("dark_pool", "block_trade", "insider_filing", "inst_13f")
                    else "options"
                ),
            },
        ))

    # Calculate totals
    bullish = sum(1 for a in alerts if a.direction == "bullish")
    bearish = sum(1 for a in alerts if a.direction == "bearish")
    total_value = Decimal("0")
    for a in alerts:
        total_value += a.value_usd

    return WhaleAlertsDashboardData(
        alerts=alerts,
        total_signals=len(alerts),
        bullish_count=bullish,
        bearish_count=bearish,
        total_value_usd=total_value,
        connected_sources=["order_book", "whale_alert", "polymarket", "unusual_whales"],
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "WhaleAlertData",
    "WhaleAlertsDashboardData",
    "AlertSummaryCard",
    "WhaleAlertsTable",
    "AlertDetailPanel",
    "SourceStatusBar",
    "WhaleAlertsDashboard",
    "create_demo_whale_alerts",
]

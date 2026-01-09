"""
Portfolio Dashboard Widget.

Comprehensive portfolio overview with:
- Total value display with Digits widget
- Asset allocation visualization
- P&L summary with Sparkline for trend
- Period returns (24h, 7d, 30d, YTD)
- Real-time updates via MessageBus

Design inspired by:
- Bloomberg Portfolio Analytics (multi-portfolio view)
- cointop (cryptocurrency tracking)
- btop++ (gauge bars, real-time updates)

Layout:
    +-- PORTFOLIO OVERVIEW ------------------------------------------+
    |                                                                 |
    |  TOTAL VALUE          DAILY P&L           ALLOCATION           |
    |  ▲ $125,430.50       +$1,250.00 (+1.0%)   [==BTC====ETH==SOL=] |
    |  ▁▂▃▅▆▇█▇▆▅▆▇█                                                  |
    |                                                                 |
    |  24h: +$1,250  7d: +$3,450  30d: +$8,200  YTD: +$25,430        |
    +-----------------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Digits, ProgressBar, Sparkline, Static


if TYPE_CHECKING:
    from libra.core.message_bus import MessageBus


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class AssetHolding:
    """Single asset holding in portfolio."""

    symbol: str
    amount: Decimal
    value_usd: Decimal
    pct_of_portfolio: float
    pnl_24h: Decimal = Decimal("0")
    pnl_24h_pct: float = 0.0
    color: str = "white"


@dataclass
class PortfolioData:
    """Complete portfolio data."""

    total_value: Decimal = Decimal("0")
    available_balance: Decimal = Decimal("0")
    total_exposure: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    daily_pnl_pct: float = 0.0
    weekly_pnl: Decimal = Decimal("0")
    monthly_pnl: Decimal = Decimal("0")
    ytd_pnl: Decimal = Decimal("0")
    holdings: list[AssetHolding] = field(default_factory=list)
    equity_history: list[float] = field(default_factory=list)


# =============================================================================
# Asset Colors
# =============================================================================

ASSET_COLORS: dict[str, str] = {
    "BTC": "orange1",
    "ETH": "dodger_blue1",
    "SOL": "medium_purple",
    "USDT": "green",
    "USDC": "blue",
    "XRP": "cyan",
    "ADA": "blue1",
    "DOGE": "yellow",
    "DOT": "magenta",
    "LINK": "blue",
}


def get_asset_color(symbol: str) -> str:
    """Get color for asset symbol."""
    base = symbol.split("/")[0] if "/" in symbol else symbol
    return ASSET_COLORS.get(base, "white")


# =============================================================================
# Total Value Card
# =============================================================================


class TotalValueCard(Container):
    """
    Large total portfolio value display with trend sparkline.

    Shows:
    - Total value in large Digits
    - Trend indicator (▲/▼)
    - Mini equity curve sparkline
    """

    DEFAULT_CSS = """
    TotalValueCard {
        width: 1fr;
        height: auto;
        min-height: 7;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    TotalValueCard .card-title {
        height: 1;
        color: $text-muted;
        text-style: bold;
    }

    TotalValueCard .value-row {
        height: 3;
        layout: horizontal;
    }

    TotalValueCard .trend-icon {
        width: 3;
        height: 3;
        content-align: center middle;
    }

    TotalValueCard .trend-icon.up {
        color: $success;
    }

    TotalValueCard .trend-icon.down {
        color: $error;
    }

    TotalValueCard Digits {
        width: 1fr;
    }

    TotalValueCard Sparkline {
        height: 2;
        margin-top: 1;
    }
    """

    value: reactive[Decimal] = reactive(Decimal("0"))
    trend: reactive[str] = reactive("up")  # "up", "down", "flat"
    history: reactive[list[float]] = reactive(list, init=False)

    def __init__(
        self,
        value: Decimal = Decimal("0"),
        history: list[float] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.value = value
        self.history = history or [0.0] * 30

    def compose(self) -> ComposeResult:
        yield Static("TOTAL VALUE", classes="card-title")
        with Horizontal(classes="value-row"):
            yield Static("▲", classes="trend-icon up", id="trend-icon")
            yield Digits(self._format_value(), id="total-digits")
        yield Sparkline(self.history, summary_function=max, id="equity-sparkline")

    def _format_value(self) -> str:
        """Format value for Digits display."""
        return f"${self.value:,.2f}"

    def watch_value(self, _new_value: Decimal) -> None:
        """Update display when value changes."""
        try:
            digits = self.query_one("#total-digits", Digits)
            digits.update(self._format_value())
        except Exception:
            pass

    def watch_trend(self, new_trend: str) -> None:
        """Update trend indicator."""
        try:
            icon = self.query_one("#trend-icon", Static)
            icon.remove_class("up", "down")
            if new_trend == "up":
                icon.update("▲")
                icon.add_class("up")
            elif new_trend == "down":
                icon.update("▼")
                icon.add_class("down")
            else:
                icon.update("●")
        except Exception:
            pass

    def watch_history(self, new_history: list[float]) -> None:
        """Update sparkline."""
        try:
            sparkline = self.query_one("#equity-sparkline", Sparkline)
            sparkline.data = new_history

            # Update trend based on recent history
            if len(new_history) >= 2:
                if new_history[-1] > new_history[-2]:
                    self.trend = "up"
                elif new_history[-1] < new_history[-2]:
                    self.trend = "down"
                else:
                    self.trend = "flat"
        except Exception:
            pass

    def update_data(
        self,
        value: Decimal | None = None,
        history: list[float] | None = None,
    ) -> None:
        """Batch update value and history."""
        if value is not None:
            self.value = value
        if history is not None:
            self.history = history


# =============================================================================
# Daily P&L Card
# =============================================================================


class DailyPnLCard(Container):
    """
    Daily P&L display with absolute and percentage values.

    Color-coded: green for profit, red for loss.
    """

    DEFAULT_CSS = """
    DailyPnLCard {
        width: 1fr;
        height: auto;
        min-height: 7;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    DailyPnLCard .card-title {
        height: 1;
        color: $text-muted;
        text-style: bold;
    }

    DailyPnLCard .pnl-value {
        height: 2;
        text-style: bold;
        content-align: left middle;
    }

    DailyPnLCard .pnl-value.positive {
        color: $success;
    }

    DailyPnLCard .pnl-value.negative {
        color: $error;
    }

    DailyPnLCard .pnl-percent {
        height: 1;
    }

    DailyPnLCard .pnl-percent.positive {
        color: $success;
    }

    DailyPnLCard .pnl-percent.negative {
        color: $error;
    }

    DailyPnLCard .pnl-breakdown {
        height: 2;
        margin-top: 1;
        color: $text-muted;
    }
    """

    pnl: reactive[Decimal] = reactive(Decimal("0"))
    pnl_pct: reactive[float] = reactive(0.0)
    unrealized: reactive[Decimal] = reactive(Decimal("0"))
    realized: reactive[Decimal] = reactive(Decimal("0"))

    def __init__(
        self,
        pnl: Decimal = Decimal("0"),
        pnl_pct: float = 0.0,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.pnl = pnl
        self.pnl_pct = pnl_pct

    def compose(self) -> ComposeResult:
        yield Static("DAILY P&L", classes="card-title")
        yield Static(
            self._format_pnl(),
            classes=f"pnl-value {self._pnl_class()}",
            id="pnl-value",
        )
        yield Static(
            self._format_pct(),
            classes=f"pnl-percent {self._pnl_class()}",
            id="pnl-percent",
        )
        yield Static(
            self._format_breakdown(),
            classes="pnl-breakdown",
            id="pnl-breakdown",
        )

    def _format_pnl(self) -> str:
        """Format P&L value."""
        sign = "+" if self.pnl >= 0 else ""
        return f"{sign}${self.pnl:,.2f}"

    def _format_pct(self) -> str:
        """Format percentage."""
        sign = "+" if self.pnl_pct >= 0 else ""
        return f"({sign}{self.pnl_pct:.2f}%)"

    def _format_breakdown(self) -> str:
        """Format unrealized/realized breakdown."""
        return f"Unrealized: ${self.unrealized:,.2f}  Realized: ${self.realized:,.2f}"

    def _pnl_class(self) -> str:
        """Get CSS class based on P&L."""
        return "positive" if self.pnl >= 0 else "negative"

    def watch_pnl(self, _value: Decimal) -> None:
        """Update P&L display."""
        try:
            pnl_widget = self.query_one("#pnl-value", Static)
            pnl_widget.update(self._format_pnl())
            pnl_widget.remove_class("positive", "negative")
            pnl_widget.add_class(self._pnl_class())
        except Exception:
            pass

    def watch_pnl_pct(self, _value: float) -> None:
        """Update percentage display."""
        try:
            pct_widget = self.query_one("#pnl-percent", Static)
            pct_widget.update(self._format_pct())
            pct_widget.remove_class("positive", "negative")
            pct_widget.add_class(self._pnl_class())
        except Exception:
            pass

    def update_data(
        self,
        pnl: Decimal | None = None,
        pnl_pct: float | None = None,
        unrealized: Decimal | None = None,
        realized: Decimal | None = None,
    ) -> None:
        """Batch update P&L data."""
        if pnl is not None:
            self.pnl = pnl
        if pnl_pct is not None:
            self.pnl_pct = pnl_pct
        if unrealized is not None:
            self.unrealized = unrealized
            try:
                self.query_one("#pnl-breakdown", Static).update(self._format_breakdown())
            except Exception:
                pass
        if realized is not None:
            self.realized = realized
            try:
                self.query_one("#pnl-breakdown", Static).update(self._format_breakdown())
            except Exception:
                pass


# =============================================================================
# Allocation Bar
# =============================================================================


class AllocationBar(Static):
    """
    Horizontal stacked bar showing asset allocation.

    Each asset is a colored segment proportional to its portfolio weight.
    """

    DEFAULT_CSS = """
    AllocationBar {
        height: 3;
        width: 1fr;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }
    """

    allocations: reactive[list[tuple[str, float]]] = reactive(list, init=False)

    def __init__(
        self,
        allocations: list[tuple[str, float]] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.allocations = allocations or []

    def render(self) -> str:
        """Render allocation as colored segments."""
        if not self.allocations:
            return "ALLOCATION\n[dim]No holdings[/dim]"

        width = max(self.size.width - 4, 20)  # Leave margin
        segments = []

        for symbol, pct in self.allocations:
            chars = max(int(width * pct / 100), 1)
            color = get_asset_color(symbol)
            segments.append(f"[{color}]{'█' * chars}[/{color}]")

        bar = "".join(segments)

        # Legend
        legend_parts = [f"[{get_asset_color(s)}]{s}[/] {p:.0f}%" for s, p in self.allocations[:4]]
        legend = "  ".join(legend_parts)

        return f"ALLOCATION\n{bar}\n{legend}"

    def update_allocations(self, allocations: list[tuple[str, float]]) -> None:
        """Update allocation data."""
        self.allocations = allocations
        self.refresh()


# =============================================================================
# Asset Allocation Table
# =============================================================================


class AssetRow(Horizontal):
    """Single asset row with progress bar."""

    DEFAULT_CSS = """
    AssetRow {
        height: 2;
        padding: 0 1;
    }

    AssetRow .asset-symbol {
        width: 10;
        text-style: bold;
    }

    AssetRow .asset-pct {
        width: 6;
        text-align: right;
    }

    AssetRow ProgressBar {
        width: 1fr;
        padding: 0 1;
    }

    AssetRow .asset-value {
        width: 15;
        text-align: right;
    }

    AssetRow .asset-pnl {
        width: 12;
        text-align: right;
    }

    AssetRow .asset-pnl.positive {
        color: $success;
    }

    AssetRow .asset-pnl.negative {
        color: $error;
    }
    """

    def __init__(self, holding: AssetHolding, id: str | None = None) -> None:
        super().__init__(id=id)
        self.holding = holding

    def compose(self) -> ComposeResult:
        color = get_asset_color(self.holding.symbol)
        yield Static(f"[{color}]{self.holding.symbol}[/{color}]", classes="asset-symbol")
        yield Static(f"{self.holding.pct_of_portfolio:.1f}%", classes="asset-pct")
        yield ProgressBar(
            total=100,
            show_eta=False,
            show_percentage=False,
            id="progress",
        )
        yield Static(f"${self.holding.value_usd:,.2f}", classes="asset-value")

        pnl_class = "positive" if self.holding.pnl_24h >= 0 else "negative"
        pnl_sign = "+" if self.holding.pnl_24h >= 0 else ""
        yield Static(
            f"{pnl_sign}${self.holding.pnl_24h:,.2f}",
            classes=f"asset-pnl {pnl_class}",
        )

    def on_mount(self) -> None:
        """Set progress after mount."""
        try:
            progress_bar = self.query_one("#progress", ProgressBar)
            progress_bar.progress = self.holding.pct_of_portfolio
        except Exception:
            pass


class AssetAllocationTable(Container):
    """Table showing asset breakdown with progress bars."""

    DEFAULT_CSS = """
    AssetAllocationTable {
        height: auto;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    AssetAllocationTable .table-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    AssetAllocationTable .table-header {
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    holdings: reactive[list[AssetHolding]] = reactive(list, init=False)

    def __init__(
        self,
        holdings: list[AssetHolding] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.holdings = holdings or []

    def compose(self) -> ComposeResult:
        yield Static("ASSET BREAKDOWN", classes="table-title")
        yield Static(
            "Symbol      %      Allocation                Value           24h P&L",
            classes="table-header",
        )
        with Vertical(id="asset-rows"):
            for holding in self.holdings:
                yield AssetRow(holding, id=f"asset-{holding.symbol.replace('/', '-')}")

    def update_holdings(self, holdings: list[AssetHolding]) -> None:
        """Update holdings and rebuild rows."""
        self.holdings = holdings

        try:
            rows_container = self.query_one("#asset-rows", Vertical)
            rows_container.remove_children()

            for holding in holdings:
                rows_container.mount(
                    AssetRow(holding, id=f"asset-{holding.symbol.replace('/', '-')}")
                )
        except Exception:
            pass


# =============================================================================
# Period Returns
# =============================================================================


class PeriodReturns(Horizontal):
    """Display returns for different time periods."""

    DEFAULT_CSS = """
    PeriodReturns {
        height: 3;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    PeriodReturns .period-card {
        width: 1fr;
        height: 100%;
        padding: 0 1;
        content-align: center middle;
    }

    PeriodReturns .period-label {
        color: $text-muted;
    }

    PeriodReturns .period-value {
        text-style: bold;
    }

    PeriodReturns .period-value.positive {
        color: $success;
    }

    PeriodReturns .period-value.negative {
        color: $error;
    }
    """

    daily: reactive[Decimal] = reactive(Decimal("0"))
    weekly: reactive[Decimal] = reactive(Decimal("0"))
    monthly: reactive[Decimal] = reactive(Decimal("0"))
    ytd: reactive[Decimal] = reactive(Decimal("0"))

    def compose(self) -> ComposeResult:
        for period, attr in [("24h", "daily"), ("7d", "weekly"), ("30d", "monthly"), ("YTD", "ytd")]:
            value = getattr(self, attr)
            yield self._create_period_card(period, value, f"period-{attr}")

    def _create_period_card(self, label: str, value: Decimal, id: str) -> Static:
        """Create a period display card."""
        sign = "+" if value >= 0 else ""
        color_class = "positive" if value >= 0 else "negative"
        return Static(
            f"[dim]{label}[/dim]\n[{color_class}]{sign}${value:,.2f}[/{color_class}]",
            classes="period-card",
            id=id,
        )

    def _update_period(self, period_id: str, value: Decimal) -> None:
        """Update a single period display."""
        try:
            label_map = {"period-daily": "24h", "period-weekly": "7d", "period-monthly": "30d", "period-ytd": "YTD"}
            label = label_map.get(period_id, "")
            sign = "+" if value >= 0 else ""
            color_class = "positive" if value >= 0 else "negative"

            card = self.query_one(f"#{period_id}", Static)
            card.update(f"[dim]{label}[/dim]\n[{color_class}]{sign}${value:,.2f}[/{color_class}]")
        except Exception:
            pass

    def watch_daily(self, value: Decimal) -> None:
        self._update_period("period-daily", value)

    def watch_weekly(self, value: Decimal) -> None:
        self._update_period("period-weekly", value)

    def watch_monthly(self, value: Decimal) -> None:
        self._update_period("period-monthly", value)

    def watch_ytd(self, value: Decimal) -> None:
        self._update_period("period-ytd", value)

    def update_returns(
        self,
        daily: Decimal | None = None,
        weekly: Decimal | None = None,
        monthly: Decimal | None = None,
        ytd: Decimal | None = None,
    ) -> None:
        """Batch update period returns."""
        if daily is not None:
            self.daily = daily
        if weekly is not None:
            self.weekly = weekly
        if monthly is not None:
            self.monthly = monthly
        if ytd is not None:
            self.ytd = ytd


# =============================================================================
# Portfolio Dashboard (Main Widget)
# =============================================================================


class PortfolioDashboard(Container):
    """
    Comprehensive portfolio overview dashboard.

    Combines:
    - Total value with trend sparkline
    - Daily P&L with breakdown
    - Asset allocation bar
    - Asset breakdown table
    - Period returns
    """

    DEFAULT_CSS = """
    PortfolioDashboard {
        height: auto;
        padding: 1;
    }

    PortfolioDashboard .dashboard-title {
        height: 1;
        text-style: bold;
        background: $primary-darken-2;
        padding: 0 1;
        margin-bottom: 1;
    }

    PortfolioDashboard .top-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }

    PortfolioDashboard .top-row > * {
        margin-right: 1;
    }

    PortfolioDashboard .top-row > *:last-child {
        margin-right: 0;
    }
    """

    class Updated(Message):
        """Message sent when portfolio data is updated."""

        def __init__(self, data: PortfolioData) -> None:
            self.data = data
            super().__init__()

    def __init__(
        self,
        data: PortfolioData | None = None,
        bus: "MessageBus | None" = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._data = data or PortfolioData()
        self.bus = bus

    def compose(self) -> ComposeResult:
        yield Static("PORTFOLIO OVERVIEW", classes="dashboard-title")

        with Horizontal(classes="top-row"):
            yield TotalValueCard(
                value=self._data.total_value,
                history=self._data.equity_history,
                id="total-value-card",
            )
            yield DailyPnLCard(
                pnl=self._data.daily_pnl,
                pnl_pct=self._data.daily_pnl_pct,
                id="daily-pnl-card",
            )
            yield AllocationBar(
                allocations=[(h.symbol, h.pct_of_portfolio) for h in self._data.holdings],
                id="allocation-bar",
            )

        yield AssetAllocationTable(
            holdings=self._data.holdings,
            id="asset-table",
        )

        yield PeriodReturns(id="period-returns")

    def on_mount(self) -> None:
        """Setup real-time updates."""
        # Initial data sync
        self._sync_all_widgets()

        # If no bus, use demo mode with periodic updates
        if self.bus is None:
            self.set_interval(2.0, self._simulate_update)

    def _sync_all_widgets(self) -> None:
        """Sync all child widgets with current data."""
        try:
            # Total value
            total_card = self.query_one("#total-value-card", TotalValueCard)
            total_card.update_data(
                value=self._data.total_value,
                history=self._data.equity_history,
            )

            # Daily P&L
            pnl_card = self.query_one("#daily-pnl-card", DailyPnLCard)
            pnl_card.update_data(
                pnl=self._data.daily_pnl,
                pnl_pct=self._data.daily_pnl_pct,
                unrealized=self._data.unrealized_pnl,
                realized=self._data.realized_pnl,
            )

            # Allocation bar
            alloc_bar = self.query_one("#allocation-bar", AllocationBar)
            alloc_bar.update_allocations(
                [(h.symbol, h.pct_of_portfolio) for h in self._data.holdings]
            )

            # Asset table
            asset_table = self.query_one("#asset-table", AssetAllocationTable)
            asset_table.update_holdings(self._data.holdings)

            # Period returns
            period_returns = self.query_one("#period-returns", PeriodReturns)
            period_returns.update_returns(
                daily=self._data.daily_pnl,
                weekly=self._data.weekly_pnl,
                monthly=self._data.monthly_pnl,
                ytd=self._data.ytd_pnl,
            )
        except Exception:
            pass

    def update_portfolio(self, data: PortfolioData) -> None:
        """Update portfolio with new data."""
        self._data = data
        self._sync_all_widgets()
        self.post_message(self.Updated(data))

    def _simulate_update(self) -> None:
        """Simulate portfolio updates for demo mode."""
        import random

        # Simulate price movement
        change = Decimal(str(random.uniform(-500, 700)))
        self._data.total_value += change
        self._data.daily_pnl += change
        self._data.daily_pnl_pct = float(self._data.daily_pnl / self._data.total_value * 100)

        # Update history
        if self._data.equity_history:
            self._data.equity_history.append(float(self._data.total_value))
            if len(self._data.equity_history) > 50:
                self._data.equity_history.pop(0)

        self._sync_all_widgets()


# =============================================================================
# Demo Data Generator
# =============================================================================


def create_demo_portfolio() -> PortfolioData:
    """Create demo portfolio data for testing."""
    holdings = [
        AssetHolding(
            symbol="BTC",
            amount=Decimal("1.5"),
            value_usd=Decimal("64500.00"),
            pct_of_portfolio=45.0,
            pnl_24h=Decimal("1250.00"),
            pnl_24h_pct=1.98,
            color="orange1",
        ),
        AssetHolding(
            symbol="ETH",
            amount=Decimal("15.0"),
            value_usd=Decimal("43000.00"),
            pct_of_portfolio=30.0,
            pnl_24h=Decimal("-320.00"),
            pnl_24h_pct=-0.74,
            color="dodger_blue1",
        ),
        AssetHolding(
            symbol="SOL",
            amount=Decimal("200.0"),
            value_usd=Decimal("21500.00"),
            pct_of_portfolio=15.0,
            pnl_24h=Decimal("450.00"),
            pnl_24h_pct=2.14,
            color="medium_purple",
        ),
        AssetHolding(
            symbol="USDT",
            amount=Decimal("14350.00"),
            value_usd=Decimal("14350.00"),
            pct_of_portfolio=10.0,
            pnl_24h=Decimal("0"),
            pnl_24h_pct=0.0,
            color="green",
        ),
    ]

    # Generate equity history
    import random
    base = 140000
    history = [base + random.uniform(-2000, 2000) for _ in range(30)]
    history.append(143350.0)  # Current value

    return PortfolioData(
        total_value=Decimal("143350.00"),
        available_balance=Decimal("14350.00"),
        total_exposure=Decimal("129000.00"),
        unrealized_pnl=Decimal("1380.00"),
        realized_pnl=Decimal("0"),
        daily_pnl=Decimal("1380.00"),
        daily_pnl_pct=0.97,
        weekly_pnl=Decimal("3450.00"),
        monthly_pnl=Decimal("8200.00"),
        ytd_pnl=Decimal("25430.00"),
        holdings=holdings,
        equity_history=history,
    )

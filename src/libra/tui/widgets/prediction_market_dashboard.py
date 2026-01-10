"""
Prediction Market Dashboard Widget.

Displays prediction market data from multiple platforms:
- Polymarket (crypto, USDC-based)
- Kalshi (regulated, USD-based)
- Metaculus (reputation-based forecasting)
- Manifold Markets (play-money)

Features:
- Real-time market probabilities
- Cross-platform price comparison
- User positions and P&L
- Market search and filtering
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option

# Check if prediction market gateway is available
try:
    from libra.gateways.prediction_market import (
        PredictionMarketGateway,
        PredictionMarket,
        PredictionQuote,
        MarketStatus,
    )
    PREDICTION_MARKET_AVAILABLE = True
except ImportError:
    PREDICTION_MARKET_AVAILABLE = False


if TYPE_CHECKING:
    from libra.gateways.prediction_market import PredictionMarketGateway


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class MarketDisplayData:
    """Display data for a prediction market."""

    market_id: str
    platform: str
    title: str
    yes_price: Decimal
    no_price: Decimal
    volume: Decimal
    status: str = "open"
    category: str = ""

    @property
    def yes_percent(self) -> float:
        """Yes probability as percentage."""
        return float(self.yes_price) * 100

    @property
    def no_percent(self) -> float:
        """No probability as percentage."""
        return float(self.no_price) * 100


@dataclass
class PositionDisplayData:
    """Display data for a user position."""

    market_id: str
    platform: str
    title: str
    outcome: str  # "yes" or "no"
    size: Decimal
    avg_price: Decimal
    current_price: Decimal
    pnl: Decimal

    @property
    def pnl_percent(self) -> float:
        """P&L as percentage."""
        if self.avg_price == 0:
            return 0.0
        return float((self.current_price - self.avg_price) / self.avg_price) * 100


@dataclass
class PredictionMarketDashboardData:
    """Aggregate data for the dashboard."""

    markets: list[MarketDisplayData] = field(default_factory=list)
    positions: list[PositionDisplayData] = field(default_factory=list)
    total_position_value: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    connected_providers: list[str] = field(default_factory=list)


def create_demo_prediction_markets() -> PredictionMarketDashboardData:
    """Create demo data for testing."""
    markets = [
        MarketDisplayData(
            market_id="poly_btc100k",
            platform="polymarket",
            title="Will Bitcoin exceed $100,000 in 2025?",
            yes_price=Decimal("0.72"),
            no_price=Decimal("0.28"),
            volume=Decimal("2500000"),
            category="crypto",
        ),
        MarketDisplayData(
            market_id="kalshi_fed",
            platform="kalshi",
            title="Will Fed cut rates in March 2025?",
            yes_price=Decimal("0.45"),
            no_price=Decimal("0.55"),
            volume=Decimal("1800000"),
            category="economics",
        ),
        MarketDisplayData(
            market_id="poly_eth5k",
            platform="polymarket",
            title="Will ETH exceed $5,000 by Q2 2025?",
            yes_price=Decimal("0.38"),
            no_price=Decimal("0.62"),
            volume=Decimal("890000"),
            category="crypto",
        ),
        MarketDisplayData(
            market_id="meta_agi",
            platform="metaculus",
            title="AGI achieved by 2030?",
            yes_price=Decimal("0.15"),
            no_price=Decimal("0.85"),
            volume=Decimal("0"),  # Metaculus is reputation-based
            category="ai",
        ),
        MarketDisplayData(
            market_id="manifold_ai",
            platform="manifold",
            title="GPT-5 released in 2025?",
            yes_price=Decimal("0.68"),
            no_price=Decimal("0.32"),
            volume=Decimal("45000"),
            category="ai",
        ),
        MarketDisplayData(
            market_id="kalshi_sp500",
            platform="kalshi",
            title="S&P 500 closes above 5500 in Q1?",
            yes_price=Decimal("0.62"),
            no_price=Decimal("0.38"),
            volume=Decimal("1200000"),
            category="markets",
        ),
    ]

    positions = [
        PositionDisplayData(
            market_id="poly_btc100k",
            platform="polymarket",
            title="Bitcoin $100k 2025",
            outcome="yes",
            size=Decimal("500"),
            avg_price=Decimal("0.60"),
            current_price=Decimal("0.72"),
            pnl=Decimal("60"),
        ),
        PositionDisplayData(
            market_id="kalshi_fed",
            platform="kalshi",
            title="Fed rate cut March",
            outcome="no",
            size=Decimal("200"),
            avg_price=Decimal("0.50"),
            current_price=Decimal("0.55"),
            pnl=Decimal("10"),
        ),
    ]

    return PredictionMarketDashboardData(
        markets=markets,
        positions=positions,
        total_position_value=Decimal("700"),
        total_pnl=Decimal("70"),
        connected_providers=["polymarket", "kalshi", "metaculus", "manifold"],
    )


# =============================================================================
# Widgets
# =============================================================================


class MarketSearchInput(Container):
    """Search input for prediction markets."""

    DEFAULT_CSS = """
    MarketSearchInput {
        height: auto;
        width: 100%;
        padding: 0 1;
    }

    MarketSearchInput Horizontal {
        height: 3;
        width: 100%;
    }

    MarketSearchInput Input {
        width: 1fr;
    }

    MarketSearchInput Select {
        width: 20;
    }

    MarketSearchInput Button {
        width: 10;
    }
    """

    class SearchSubmitted(Message):
        """Emitted when search is submitted."""

        def __init__(self, query: str, platform: str, category: str) -> None:
            self.query = query
            self.platform = platform
            self.category = category
            super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Input(placeholder="Search markets...", id="market-search-input")
            yield Select(
                [
                    ("All Platforms", "all"),
                    ("Polymarket", "polymarket"),
                    ("Kalshi", "kalshi"),
                    ("Metaculus", "metaculus"),
                    ("Manifold", "manifold"),
                ],
                value="all",
                id="platform-select",
                allow_blank=False,
            )
            yield Select(
                [
                    ("All Categories", "all"),
                    ("Crypto", "crypto"),
                    ("Politics", "politics"),
                    ("Economics", "economics"),
                    ("AI", "ai"),
                    ("Sports", "sports"),
                ],
                value="all",
                id="category-select",
                allow_blank=False,
            )
            yield Button("Search", id="search-btn", variant="primary")

    @on(Button.Pressed, "#search-btn")
    def on_search(self) -> None:
        """Handle search button press."""
        search_input = self.query_one("#market-search-input", Input)
        platform_select = self.query_one("#platform-select", Select)
        category_select = self.query_one("#category-select", Select)

        self.post_message(
            self.SearchSubmitted(
                query=search_input.value,
                platform=str(platform_select.value),
                category=str(category_select.value),
            )
        )


class MarketsTable(Container):
    """Table displaying prediction markets."""

    DEFAULT_CSS = """
    MarketsTable {
        height: 1fr;
        min-height: 10;
        border: round $primary-darken-1;
        padding: 0;
    }

    MarketsTable DataTable {
        height: 1fr;
    }

    MarketsTable .table-title {
        height: 1;
        padding: 0 1;
        background: $surface;
        text-style: bold;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._table: DataTable | None = None

    def compose(self) -> ComposeResult:
        yield Static("📊 ACTIVE MARKETS", classes="table-title")
        yield DataTable(id="markets-datatable", zebra_stripes=True)

    def on_mount(self) -> None:
        """Initialize table columns."""
        self._table = self.query_one("#markets-datatable", DataTable)
        self._table.add_columns(
            "Platform",
            "Market",
            "Yes %",
            "No %",
            "Volume",
            "Category",
        )
        self._table.cursor_type = "row"

    def update_markets(self, markets: list[MarketDisplayData]) -> None:
        """Update table with market data."""
        if not self._table:
            return

        self._table.clear()

        for market in markets:
            # Color-code probabilities
            yes_str = f"[green]{market.yes_percent:.1f}%[/]" if market.yes_percent > 50 else f"{market.yes_percent:.1f}%"
            no_str = f"[red]{market.no_percent:.1f}%[/]" if market.no_percent > 50 else f"{market.no_percent:.1f}%"

            # Format volume
            if market.volume >= 1_000_000:
                vol_str = f"${market.volume / 1_000_000:.1f}M"
            elif market.volume >= 1_000:
                vol_str = f"${market.volume / 1_000:.1f}K"
            elif market.volume > 0:
                vol_str = f"${market.volume:.0f}"
            else:
                vol_str = "—"

            # Platform icon
            platform_icons = {
                "polymarket": "🔮",
                "kalshi": "📈",
                "metaculus": "🔬",
                "manifold": "🎲",
            }
            platform_str = f"{platform_icons.get(market.platform, '•')} {market.platform.title()}"

            self._table.add_row(
                platform_str,
                market.title[:45] + "..." if len(market.title) > 45 else market.title,
                yes_str,
                no_str,
                vol_str,
                market.category.title() if market.category else "—",
                key=market.market_id,
            )


class PositionsTable(Container):
    """Table displaying user positions."""

    DEFAULT_CSS = """
    PositionsTable {
        height: auto;
        min-height: 8;
        max-height: 15;
        border: round $primary-darken-1;
        padding: 0;
    }

    PositionsTable DataTable {
        height: auto;
        max-height: 12;
    }

    PositionsTable .table-title {
        height: 1;
        padding: 0 1;
        background: $surface;
        text-style: bold;
    }

    PositionsTable .positions-summary {
        height: 1;
        padding: 0 1;
        background: $surface;
        dock: bottom;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._table: DataTable | None = None
        self._summary: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static("💼 YOUR POSITIONS", classes="table-title")
        yield DataTable(id="positions-datatable", zebra_stripes=True)
        yield Static("Total: $0.00 | P&L: $0.00", id="positions-summary", classes="positions-summary")

    def on_mount(self) -> None:
        """Initialize table columns."""
        self._table = self.query_one("#positions-datatable", DataTable)
        self._summary = self.query_one("#positions-summary", Static)
        self._table.add_columns(
            "Platform",
            "Market",
            "Side",
            "Size",
            "Avg Price",
            "Current",
            "P&L",
        )
        self._table.cursor_type = "row"

    def update_positions(
        self,
        positions: list[PositionDisplayData],
        total_value: Decimal,
        total_pnl: Decimal,
    ) -> None:
        """Update table with position data."""
        if not self._table or not self._summary:
            return

        self._table.clear()

        for pos in positions:
            # Side indicator
            side_str = f"[green]YES ▲[/]" if pos.outcome.lower() == "yes" else f"[red]NO ▼[/]"

            # P&L coloring
            if pos.pnl > 0:
                pnl_str = f"[green]+${pos.pnl:.2f}[/]"
            elif pos.pnl < 0:
                pnl_str = f"[red]-${abs(pos.pnl):.2f}[/]"
            else:
                pnl_str = f"${pos.pnl:.2f}"

            self._table.add_row(
                pos.platform.title(),
                pos.title[:30] + "..." if len(pos.title) > 30 else pos.title,
                side_str,
                f"${pos.size:.2f}",
                f"{float(pos.avg_price) * 100:.1f}%",
                f"{float(pos.current_price) * 100:.1f}%",
                pnl_str,
                key=pos.market_id,
            )

        # Update summary
        pnl_color = "green" if total_pnl >= 0 else "red"
        pnl_sign = "+" if total_pnl >= 0 else ""
        self._summary.update(
            f"Total: ${total_value:.2f} | P&L: [{pnl_color}]{pnl_sign}${total_pnl:.2f}[/]"
        )


class ProviderStatusBar(Container):
    """Status bar showing connected providers."""

    DEFAULT_CSS = """
    ProviderStatusBar {
        height: 1;
        dock: bottom;
        padding: 0 1;
        background: $surface;
    }

    ProviderStatusBar .provider-item {
        margin-right: 2;
    }

    ProviderStatusBar .connected {
        color: $success;
    }

    ProviderStatusBar .disconnected {
        color: $text-muted;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._status_label: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static("Providers: Loading...", id="provider-status")

    def on_mount(self) -> None:
        self._status_label = self.query_one("#provider-status", Static)

    def update_providers(self, connected: list[str]) -> None:
        """Update provider connection status."""
        if not self._status_label:
            return

        all_providers = ["polymarket", "kalshi", "metaculus", "manifold"]
        status_parts = []

        for provider in all_providers:
            if provider in connected:
                status_parts.append(f"[green]● {provider.title()}[/]")
            else:
                status_parts.append(f"[dim]○ {provider.title()}[/]")

        self._status_label.update("Providers: " + "  ".join(status_parts))


# =============================================================================
# Main Dashboard
# =============================================================================


class PredictionMarketDashboard(Container):
    """
    Main dashboard for prediction markets.

    Provides a unified view of prediction markets across multiple platforms
    with search, filtering, and position tracking.

    Usage:
        dashboard = PredictionMarketDashboard()
        # Optionally set gateway for live data
        dashboard.gateway = prediction_market_gateway
    """

    DEFAULT_CSS = """
    PredictionMarketDashboard {
        height: 1fr;
        min-height: 25;
        padding: 0;
    }

    PredictionMarketDashboard .dashboard-title {
        height: 1;
        padding: 0 1;
        background: $primary-darken-2;
        text-style: bold;
        color: $text;
    }

    PredictionMarketDashboard .main-content {
        height: 1fr;
        padding: 1;
    }

    PredictionMarketDashboard .markets-section {
        height: 2fr;
        min-height: 12;
    }

    PredictionMarketDashboard .positions-section {
        height: 1fr;
        min-height: 8;
        margin-top: 1;
    }

    PredictionMarketDashboard .loading-overlay {
        height: 100%;
        align: center middle;
        background: $surface 50%;
        layer: loading;
    }
    """

    data: reactive[PredictionMarketDashboardData] = reactive(
        create_demo_prediction_markets, init=False
    )

    def __init__(
        self,
        gateway: PredictionMarketGateway | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._gateway = gateway
        self._markets_table: MarketsTable | None = None
        self._positions_table: PositionsTable | None = None
        self._provider_status: ProviderStatusBar | None = None
        self._loading: LoadingIndicator | None = None

    @property
    def gateway(self) -> PredictionMarketGateway | None:
        """Get the gateway."""
        return self._gateway

    @gateway.setter
    def gateway(self, value: PredictionMarketGateway | None) -> None:
        """Set the gateway and refresh data."""
        self._gateway = value
        if value and value.is_connected:
            self._refresh_data()

    def compose(self) -> ComposeResult:
        yield Static("🔮 PREDICTION MARKETS", classes="dashboard-title")
        yield MarketSearchInput(id="market-search")

        with Vertical(classes="main-content"):
            with Vertical(classes="markets-section"):
                yield MarketsTable(id="markets-table")

            with Vertical(classes="positions-section"):
                yield PositionsTable(id="positions-table")

        yield ProviderStatusBar(id="provider-status-bar")
        yield LoadingIndicator(id="loading-indicator")

    def on_mount(self) -> None:
        """Cache widget references."""
        self._markets_table = self.query_one("#markets-table", MarketsTable)
        self._positions_table = self.query_one("#positions-table", PositionsTable)
        self._provider_status = self.query_one("#provider-status-bar", ProviderStatusBar)
        self._loading = self.query_one("#loading-indicator", LoadingIndicator)
        self._loading.display = False

        # Initial update with demo data
        self._update_display()

    def watch_data(self, new_data: PredictionMarketDashboardData) -> None:
        """React to data changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update all display elements."""
        if self._markets_table:
            self._markets_table.update_markets(self.data.markets)

        if self._positions_table:
            self._positions_table.update_positions(
                self.data.positions,
                self.data.total_position_value,
                self.data.total_pnl,
            )

        if self._provider_status:
            self._provider_status.update_providers(self.data.connected_providers)

    @on(MarketSearchInput.SearchSubmitted)
    def on_search(self, event: MarketSearchInput.SearchSubmitted) -> None:
        """Handle search request."""
        if self._gateway and self._gateway.is_connected:
            self._search_markets(event.query, event.platform, event.category)
        else:
            # Filter demo data
            self._filter_demo_data(event.query, event.platform, event.category)

    def _filter_demo_data(self, query: str, platform: str, category: str) -> None:
        """Filter demo data based on search criteria."""
        demo_data = create_demo_prediction_markets()
        filtered_markets = []

        for market in demo_data.markets:
            # Platform filter
            if platform != "all" and market.platform != platform:
                continue
            # Category filter
            if category != "all" and market.category != category:
                continue
            # Query filter
            if query and query.lower() not in market.title.lower():
                continue
            filtered_markets.append(market)

        self.data = PredictionMarketDashboardData(
            markets=filtered_markets,
            positions=demo_data.positions,
            total_position_value=demo_data.total_position_value,
            total_pnl=demo_data.total_pnl,
            connected_providers=demo_data.connected_providers,
        )

    @work(exclusive=True)
    async def _search_markets(self, query: str, platform: str, category: str) -> None:
        """Search markets using gateway (async worker)."""
        if not self._gateway:
            return

        if self._loading:
            self._loading.display = True

        try:
            provider = None if platform == "all" else platform
            cat = None if category == "all" else category

            markets = await self._gateway.get_markets(
                provider=provider,
                category=cat,
                search=query if query else None,
                limit=50,
            )

            # Convert to display data
            display_markets = []
            for m in markets:
                yes_price = m.best_yes_price or Decimal("0.5")
                no_price = m.best_no_price or Decimal("0.5")
                display_markets.append(
                    MarketDisplayData(
                        market_id=m.market_id,
                        platform=m.platform,
                        title=m.title,
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=m.volume,
                        category=m.category or "",
                    )
                )

            self.data = PredictionMarketDashboardData(
                markets=display_markets,
                positions=self.data.positions,
                total_position_value=self.data.total_position_value,
                total_pnl=self.data.total_pnl,
                connected_providers=self._gateway.available_providers,
            )

        except Exception as e:
            self.notify(f"Search failed: {e}", severity="error")

        finally:
            if self._loading:
                self._loading.display = False

    @work(exclusive=True)
    async def _refresh_data(self) -> None:
        """Refresh all data from gateway."""
        if not self._gateway or not self._gateway.is_connected:
            return

        if self._loading:
            self._loading.display = True

        try:
            # Fetch markets
            markets = await self._gateway.get_markets(limit=20)

            display_markets = []
            for m in markets:
                yes_price = m.best_yes_price or Decimal("0.5")
                no_price = m.best_no_price or Decimal("0.5")
                display_markets.append(
                    MarketDisplayData(
                        market_id=m.market_id,
                        platform=m.platform,
                        title=m.title,
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=m.volume,
                        category=m.category or "",
                    )
                )

            self.data = PredictionMarketDashboardData(
                markets=display_markets,
                positions=[],
                total_position_value=Decimal("0"),
                total_pnl=Decimal("0"),
                connected_providers=self._gateway.available_providers,
            )

        except Exception as e:
            self.notify(f"Refresh failed: {e}", severity="error")

        finally:
            if self._loading:
                self._loading.display = False

    def update_from_data(self, data: PredictionMarketDashboardData) -> None:
        """Update dashboard from external data."""
        self.data = data

    def simulate_price_changes(self) -> None:
        """Simulate price changes for demo mode."""
        import random

        new_markets = []
        for market in self.data.markets:
            # Random walk on prices
            delta = Decimal(str(random.uniform(-0.02, 0.02)))
            new_yes = max(Decimal("0.01"), min(Decimal("0.99"), market.yes_price + delta))
            new_no = Decimal("1") - new_yes

            new_markets.append(
                MarketDisplayData(
                    market_id=market.market_id,
                    platform=market.platform,
                    title=market.title,
                    yes_price=new_yes,
                    no_price=new_no,
                    volume=market.volume,
                    category=market.category,
                )
            )

        # Update positions P&L
        new_positions = []
        total_pnl = Decimal("0")
        total_value = Decimal("0")

        for pos in self.data.positions:
            # Find corresponding market
            market = next((m for m in new_markets if m.market_id == pos.market_id), None)
            if market:
                new_price = market.yes_price if pos.outcome == "yes" else market.no_price
                pnl = (new_price - pos.avg_price) * pos.size
                new_positions.append(
                    PositionDisplayData(
                        market_id=pos.market_id,
                        platform=pos.platform,
                        title=pos.title,
                        outcome=pos.outcome,
                        size=pos.size,
                        avg_price=pos.avg_price,
                        current_price=new_price,
                        pnl=pnl,
                    )
                )
                total_pnl += pnl
                total_value += pos.size

        self.data = PredictionMarketDashboardData(
            markets=new_markets,
            positions=new_positions,
            total_position_value=total_value,
            total_pnl=total_pnl,
            connected_providers=self.data.connected_providers,
        )

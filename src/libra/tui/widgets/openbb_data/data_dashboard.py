"""
OpenBB Data Dashboard Widget.

Main dashboard combining all OpenBB data widgets into a unified interface.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Label, LoadingIndicator, Static, TabbedContent, TabPane

from libra.tui.widgets.openbb_data.symbol_search import SymbolSearchInput, SymbolSelection
from libra.tui.widgets.openbb_data.price_chart import PriceChartWidget, PriceChartData
from libra.tui.widgets.openbb_data.fundamentals_panel import FundamentalsPanel, FundamentalsData
from libra.tui.widgets.openbb_data.options_chain import OptionsChainWidget, OptionsChainData
from libra.tui.widgets.openbb_data.economic_chart import EconomicDataWidget, EconomicChartData


# Check if OpenBB is available
try:
    import openbb
    OPENBB_AVAILABLE = True
except ImportError as e:
    OPENBB_AVAILABLE = False



if TYPE_CHECKING:
    from libra.gateways.openbb import OpenBBGateway


class OpenBBDataDashboard(Container):
    """
    Main dashboard for OpenBB data visualization.

    Combines:
    - Symbol search with provider selection
    - Price chart with multiple timeframes
    - Company fundamentals panel
    - Options chain with Greeks
    - Economic data (FRED) charts

    Usage:
        dashboard = OpenBBDataDashboard()
        # Set gateway for data fetching
        dashboard.gateway = openbb_gateway
    """

    DEFAULT_CSS = """
    OpenBBDataDashboard {
        height: 1fr;
        min-height: 20;
        padding: 0;
    }

    OpenBBDataDashboard .dashboard-header {
        height: auto;
        padding: 0 1;
        background: $surface;
    }

    OpenBBDataDashboard .dashboard-content {
        height: 1fr;
        min-height: 15;
    }

    OpenBBDataDashboard TabbedContent {
        height: 1fr;
    }

    OpenBBDataDashboard TabPane {
        height: 1fr;
    }

    OpenBBDataDashboard .loading-overlay {
        height: 100%;
        align: center middle;
        background: $surface 50%;
        layer: loading;
    }

    OpenBBDataDashboard .status-bar {
        height: 1;
        dock: bottom;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    OpenBBDataDashboard .provider-status {
        color: $success;
    }

    OpenBBDataDashboard .error-status {
        color: $error;
    }

    OpenBBDataDashboard .install-banner {
        height: 100%;
        align: center middle;
        padding: 2 4;
    }

    OpenBBDataDashboard .install-box {
        width: 60;
        height: auto;
        padding: 2 3;
        border: round $primary;
        background: $surface;
    }

    OpenBBDataDashboard .install-title {
        text-align: center;
        text-style: bold;
        color: $warning;
        padding-bottom: 1;
    }

    OpenBBDataDashboard .install-text {
        text-align: center;
        color: $text;
        padding-bottom: 1;
    }

    OpenBBDataDashboard .install-cmd {
        text-align: center;
        color: $accent;
        text-style: bold;
        background: $surface-darken-1;
        padding: 1;
    }
    """

    # Gateway for data fetching
    gateway: OpenBBGateway | None = None

    # State
    loading: reactive[bool] = reactive(False, init=False)
    current_symbol: reactive[str] = reactive("", init=False)
    current_provider: reactive[str] = reactive("yfinance", init=False)
    status_message: reactive[str] = reactive("Ready", init=False)

    def __init__(
        self,
        gateway: OpenBBGateway | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.gateway = gateway

    def compose(self) -> ComposeResult:
        """Compose the dashboard."""
        # Show installation banner if OpenBB is not available
        if not OPENBB_AVAILABLE:
            with Container(classes="install-banner"):
                with Container(classes="install-box"):
                    yield Static(
                        "OpenBB Not Installed",
                        classes="install-title",
                    )
                    yield Static(
                        "The Data tab requires OpenBB Platform for market data.\n"
                        "OpenBB provides access to 30+ data providers including\n"
                        "Yahoo Finance, Polygon, FRED, and more.",
                        classes="install-text",
                    )
                    yield Static(
                        "pip install openbb openbb-yfinance",
                        classes="install-cmd",
                    )
                    yield Static(
                        "\nOptional: For charts, also install:\n",
                        classes="install-text",
                    )
                    yield Static(
                        "pip install textual-plotext",
                        classes="install-cmd",
                    )
            return

        # Header with symbol search
        with Container(classes="dashboard-header"):
            yield SymbolSearchInput(
                initial_symbol="AAPL",
                initial_provider="yfinance",
                id="symbol-search",
            )

        # Main content area with tabs
        with Container(classes="dashboard-content"):
            with TabbedContent(initial="prices", id="data-tabs"):
                with TabPane("Prices", id="prices"):
                    yield PriceChartWidget(id="price-chart")

                with TabPane("Fundamentals", id="fundamentals"):
                    yield FundamentalsPanel(id="fundamentals-panel")

                with TabPane("Options", id="options"):
                    yield OptionsChainWidget(id="options-chain")

                with TabPane("Economic", id="economic"):
                    yield EconomicDataWidget(id="economic-data")

        # Status bar
        yield Static(
            "[dim]Ready[/dim] | Provider: yfinance",
            id="status-bar",
            classes="status-bar",
        )

    def on_mount(self) -> None:
        """Initialize after mount."""
        # Load initial data for AAPL
        if self.gateway:
            self._load_equity_data("AAPL", "yfinance")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    @on(SymbolSearchInput.SymbolSelected)
    def _on_symbol_selected(self, event: SymbolSearchInput.SymbolSelected) -> None:
        """Handle symbol selection from search."""
        selection = event.selection
        self.current_symbol = selection.symbol
        self.current_provider = selection.provider

        if selection.data_type == "equity":
            self._load_equity_data(selection.symbol, selection.provider)
        elif selection.data_type == "crypto":
            self._load_crypto_data(selection.symbol, selection.provider)
        elif selection.data_type == "options":
            self._load_options_data(selection.symbol, selection.provider)
        elif selection.data_type == "economic":
            self._load_economic_data(selection.symbol, selection.provider)

    @on(PriceChartWidget.DataRequested)
    def _on_price_data_requested(self, event: PriceChartWidget.DataRequested) -> None:
        """Handle price chart data request."""
        self._load_price_data(event.symbol, event.interval, event.provider)

    @on(FundamentalsPanel.DataRequested)
    def _on_fundamentals_requested(
        self, event: FundamentalsPanel.DataRequested
    ) -> None:
        """Handle fundamentals data request."""
        self._load_fundamentals_data(
            event.symbol, event.statement, event.period, event.provider
        )

    @on(OptionsChainWidget.DataRequested)
    def _on_options_requested(self, event: OptionsChainWidget.DataRequested) -> None:
        """Handle options data request."""
        self._load_options_chain(event.symbol, event.expiration, event.provider)

    @on(EconomicDataWidget.DataRequested)
    def _on_economic_requested(self, event: EconomicDataWidget.DataRequested) -> None:
        """Handle economic data request."""
        self._load_economic_series(event.series_id, event.transform, event.provider)

    # =========================================================================
    # Data Loading Methods
    # =========================================================================

    @work(exclusive=True, group="equity_data")
    async def _load_equity_data(self, symbol: str, provider: str) -> None:
        """Load all equity data for a symbol."""
        self._set_loading(True, f"Loading {symbol}...")

        try:
            if not self.gateway:
                self._set_status("No gateway configured", error=True)
                return

            # Load price data directly (not via nested worker)
            bars = await self.gateway.get_equity_historical(
                symbol=symbol,
                interval="1d",
                provider=provider,
            )

            chart_data = PriceChartData(
                symbol=symbol,
                interval="1d",
                bars=bars,
                provider=provider,
            )

            try:
                chart = self.query_one("#price-chart", PriceChartWidget)
                chart.set_data(chart_data)
                # Explicitly trigger chart update since reactive watcher may not fire
                # when called from within an async worker
                chart._update_chart()
                chart._update_stats()
                chart._update_title()
            except Exception as e:
                self.log.error(f"Error setting chart data: {e}")

            self._set_status(f"Loaded {symbol} ({len(bars)} bars)", error=False)

        except Exception as e:
            self._set_status(f"Error: {e}", error=True)
        finally:
            self._set_loading(False)

    @work(exclusive=True, group="crypto_data")
    async def _load_crypto_data(self, symbol: str, provider: str) -> None:
        """Load crypto data."""
        self._set_loading(True, f"Loading {symbol}...")

        try:
            if not self.gateway:
                self._set_status("No gateway configured", error=True)
                self._clear_chart_on_error(symbol)
                return

            bars = await self.gateway.get_crypto_historical(
                symbol=symbol,
                interval="1d",
                provider=provider,
            )

            if not bars:
                self._set_status(f"No data for {symbol}", error=True)
                self._clear_chart_on_error(symbol)
                return

            chart_data = PriceChartData(
                symbol=symbol,
                interval="1d",
                bars=bars,
                provider=provider,
            )

            try:
                chart = self.query_one("#price-chart", PriceChartWidget)
                chart.set_data(chart_data)
                chart._update_chart()
                chart._update_stats()
                chart._update_title()
            except Exception:
                pass

            self._set_status(f"Loaded {symbol} ({len(bars)} bars)", error=False)

        except Exception as e:
            self._set_status(f"Error: {e}", error=True)
            self._clear_chart_on_error(symbol)
        finally:
            self._set_loading(False)

    @work(exclusive=True, group="price_data")
    async def _load_price_data(
        self, symbol: str, interval: str, provider: str
    ) -> None:
        """Load price chart data."""
        try:
            if not self.gateway:
                return

            bars = await self.gateway.get_equity_historical(
                symbol=symbol,
                interval=interval,
                provider=provider,
            )

            chart_data = PriceChartData(
                symbol=symbol,
                interval=interval,
                bars=bars,
                provider=provider,
            )

            try:
                chart = self.query_one("#price-chart", PriceChartWidget)
                chart.set_data(chart_data)
            except Exception:
                pass

        except Exception as e:
            self._set_status(f"Price error: {e}", error=True)

    @work(exclusive=True, group="fundamentals_data")
    async def _load_fundamentals_full(self, symbol: str, provider: str) -> None:
        """Load all fundamental data."""
        try:
            if not self.gateway:
                return

            # Load profile
            profile = await self.gateway.get_company_profile(
                symbol=symbol, provider=provider
            )

            # Load income statement
            income = await self.gateway.get_income_statement(
                symbol=symbol, period="annual", limit=4, provider=provider
            )

            # Load balance sheet
            balance = await self.gateway.get_balance_sheet(
                symbol=symbol, period="annual", limit=4, provider=provider
            )

            # Load cash flow
            cash_flow = await self.gateway.get_cash_flow(
                symbol=symbol, period="annual", limit=4, provider=provider
            )

            # Load ratios
            ratios = await self.gateway.get_ratios(
                symbol=symbol, period="annual", limit=1, provider=provider
            )

            data = FundamentalsData(
                symbol=symbol,
                profile=profile,
                income=income,
                balance=balance,
                cash_flow=cash_flow,
                ratios=ratios,
            )

            try:
                panel = self.query_one("#fundamentals-panel", FundamentalsPanel)
                panel.set_data(data)
            except Exception:
                pass

        except Exception as e:
            self._set_status(f"Fundamentals error: {e}", error=True)

    @work(exclusive=True, group="fundamentals_data")
    async def _load_fundamentals_data(
        self, symbol: str, statement: str, period: str, provider: str
    ) -> None:
        """Load specific fundamental data."""
        try:
            if not self.gateway:
                return

            records = await self.gateway.get_fundamentals(
                symbol=symbol,
                statement=statement,
                period=period,
                limit=8,
                provider=provider,
            )

            # Update appropriate field in existing data
            try:
                panel = self.query_one("#fundamentals-panel", FundamentalsPanel)
                if panel.data:
                    if statement == "income":
                        panel.data.income = records
                    elif statement == "balance":
                        panel.data.balance = records
                    elif statement == "cash":
                        panel.data.cash_flow = records
                    elif statement == "ratios":
                        panel.data.ratios = records
                    panel.set_data(panel.data)
            except Exception:
                pass

        except Exception as e:
            self._set_status(f"Fundamentals error: {e}", error=True)

    @work(exclusive=True, group="options_data")
    async def _load_options_data(self, symbol: str, provider: str) -> None:
        """Load options chain data."""
        try:
            if not self.gateway:
                return

            contracts = await self.gateway.get_options_chain(
                symbol=symbol, provider=provider
            )

            expirations = sorted(set(c.expiration for c in contracts))

            data = OptionsChainData(
                symbol=symbol,
                contracts=contracts,
                expirations=expirations,
                selected_expiration=expirations[0] if expirations else None,
            )

            try:
                widget = self.query_one("#options-chain", OptionsChainWidget)
                widget.set_data(data)
            except Exception:
                pass

        except Exception as e:
            self._set_status(f"Options error: {e}", error=True)

    @work(exclusive=True, group="options_data")
    async def _load_options_chain(
        self, symbol: str, expiration: date | None, provider: str
    ) -> None:
        """Load options chain for specific expiration."""
        try:
            if not self.gateway:
                return

            contracts = await self.gateway.get_options_chain(
                symbol=symbol,
                expiration=expiration,
                provider=provider,
            )

            expirations = sorted(set(c.expiration for c in contracts))

            data = OptionsChainData(
                symbol=symbol,
                contracts=contracts,
                expirations=expirations,
                selected_expiration=expiration,
            )

            try:
                widget = self.query_one("#options-chain", OptionsChainWidget)
                widget.set_data(data)
            except Exception:
                pass

        except Exception as e:
            self._set_status(f"Options error: {e}", error=True)

    @work(exclusive=True, group="economic_data")
    async def _load_economic_data(self, series_id: str, provider: str) -> None:
        """Load economic data series."""
        # Call the internal implementation directly (not via worker)
        await self._load_economic_series_impl(series_id, None, provider)

    @work(exclusive=True, group="economic_data")
    async def _load_economic_series(
        self, series_id: str, transform: str | None, provider: str
    ) -> None:
        """Load FRED economic series (worker wrapper)."""
        await self._load_economic_series_impl(series_id, transform, provider)

    async def _load_economic_series_impl(
        self, series_id: str, transform: str | None, provider: str
    ) -> None:
        """Load FRED economic series (implementation)."""
        try:
            if not self.gateway:
                return

            points = await self.gateway.get_economic_series(
                series_id=series_id,
                transform=transform,
                provider=provider,
            )

            # Get series name
            series_names = {s[0]: s[1] for s in [
                ("GDP", "Gross Domestic Product"),
                ("UNRATE", "Unemployment Rate"),
                ("CPIAUCSL", "Consumer Price Index"),
                ("FEDFUNDS", "Federal Funds Rate"),
                ("DGS10", "10-Year Treasury"),
                ("SP500", "S&P 500"),
                ("VIXCLS", "VIX"),
                ("M2SL", "M2 Money Supply"),
            ]}
            series_name = series_names.get(series_id, series_id)

            data = EconomicChartData(
                series_id=series_id,
                series_name=series_name,
                data_points=points,
                transform=transform,
            )

            try:
                widget = self.query_one("#economic-data", EconomicDataWidget)
                widget.set_data(data)
            except Exception:
                pass

        except Exception as e:
            self._set_status(f"Economic error: {e}", error=True)

    # =========================================================================
    # UI Helpers
    # =========================================================================

    def _clear_chart_on_error(self, symbol: str) -> None:
        """Clear chart and show error state for a symbol."""
        try:
            chart = self.query_one("#price-chart", PriceChartWidget)
            # Set empty data with the symbol so the title updates
            empty_data = PriceChartData(
                symbol=symbol,
                interval="1d",
                bars=[],
                provider=self.current_provider,
            )
            chart.set_data(empty_data)
            chart._update_title()
        except Exception:
            pass

    def _set_loading(self, loading: bool, message: str = "") -> None:
        """Set loading state."""
        self.loading = loading
        if message:
            self._set_status(message, error=False)

    def _set_status(self, message: str, error: bool = False) -> None:
        """Update status bar."""
        self.status_message = message
        try:
            status = self.query_one("#status-bar", Static)
            color = "red" if error else "dim"
            provider_text = f" | Provider: {self.current_provider}"
            status.update(f"[{color}]{message}[/{color}]{provider_text}")
        except Exception:
            pass

    def watch_loading(self, loading: bool) -> None:
        """Handle loading state changes."""
        # Could add loading overlay here if needed
        pass

    def set_gateway(self, gateway: OpenBBGateway) -> None:
        """Set the OpenBB gateway for data fetching."""
        self.gateway = gateway

    def refresh_data(self) -> None:
        """Refresh current data."""
        if self.current_symbol:
            self._load_equity_data(self.current_symbol, self.current_provider)

"""
Symbol Search Input Widget.

Provides a search input for finding and selecting stock/crypto symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Label, Select, Static


if TYPE_CHECKING:
    pass


# Provider options for different data types
EQUITY_PROVIDERS = [
    ("Yahoo Finance", "yfinance"),
    ("FMP", "fmp"),
    ("Polygon", "polygon"),
    ("Alpha Vantage", "alpha_vantage"),
    ("Tiingo", "tiingo"),
]

CRYPTO_PROVIDERS = [
    ("Yahoo Finance", "yfinance"),
    ("FMP", "fmp"),
    ("Polygon", "polygon"),
    ("Tiingo", "tiingo"),
]

FUNDAMENTALS_PROVIDERS = [
    ("FMP", "fmp"),
    ("Polygon", "polygon"),
    ("Yahoo Finance", "yfinance"),
]

OPTIONS_PROVIDERS = [
    ("CBOE", "cboe"),
    ("Tradier", "tradier"),
    ("Yahoo Finance", "yfinance"),
]

ECONOMIC_PROVIDERS = [
    ("FRED", "fred"),
]

# Common economic series
ECONOMIC_SERIES = [
    ("GDP", "GDP"),
    ("Unemployment Rate", "UNRATE"),
    ("CPI (Inflation)", "CPIAUCSL"),
    ("Fed Funds Rate", "FEDFUNDS"),
    ("10Y Treasury", "DGS10"),
    ("S&P 500", "SP500"),
    ("VIX", "VIXCLS"),
    ("M2 Money Supply", "M2SL"),
]


@dataclass
class SymbolSelection:
    """Represents a symbol selection with provider."""

    symbol: str
    provider: str
    data_type: str  # "equity", "crypto", "options", "economic"


class SymbolSearchInput(Static):
    """
    Symbol search input with provider selection.

    Allows users to enter a symbol and select a data provider.

    Messages:
        SymbolSelected: Sent when a symbol is confirmed
    """

    DEFAULT_CSS = """
    SymbolSearchInput {
        height: auto;
        padding: 0 1;
    }

    SymbolSearchInput .search-row {
        height: 3;
        layout: horizontal;
    }

    SymbolSearchInput .symbol-input {
        width: 1fr;
        margin-right: 1;
    }

    SymbolSearchInput .provider-select {
        width: 20;
        margin-right: 1;
    }

    SymbolSearchInput .data-type-select {
        width: 15;
    }

    SymbolSearchInput .search-label {
        width: 8;
        content-align: center middle;
    }

    SymbolSearchInput .quick-symbols {
        height: 2;
        padding-top: 1;
    }

    SymbolSearchInput .quick-symbol {
        margin-right: 1;
        color: $text-muted;
    }

    SymbolSearchInput .quick-symbol:hover {
        color: $accent;
        text-style: bold;
    }
    """

    # Current selection
    symbol: reactive[str] = reactive("", init=False)
    provider: reactive[str] = reactive("yfinance", init=False)
    data_type: reactive[str] = reactive("equity", init=False)

    class SymbolSelected(Message):
        """Message sent when a symbol is selected."""

        def __init__(self, selection: SymbolSelection) -> None:
            self.selection = selection
            super().__init__()

    class SymbolChanged(Message):
        """Message sent when symbol input changes."""

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            super().__init__()

    def __init__(
        self,
        initial_symbol: str = "",
        initial_provider: str = "yfinance",
        data_type: str = "equity",
        show_quick_symbols: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._initial_symbol = initial_symbol
        self._initial_provider = initial_provider
        self._data_type = data_type
        self._show_quick_symbols = show_quick_symbols

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with Horizontal(classes="search-row"):
            yield Label("Symbol:", classes="search-label")
            yield Input(
                placeholder="Enter symbol (e.g., AAPL, BTC-USD)",
                value=self._initial_symbol,
                id="symbol-input",
                classes="symbol-input",
            )
            yield Select(
                options=self._get_data_type_options(),
                value=self._data_type,
                id="data-type-select",
                classes="data-type-select",
            )
            yield Select(
                options=self._get_provider_options(self._data_type),
                value=self._initial_provider,
                id="provider-select",
                classes="provider-select",
            )

        if self._show_quick_symbols:
            with Horizontal(classes="quick-symbols"):
                yield Label("Quick:", classes="search-label")
                for sym in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "ETH-USD"]:
                    # Use quotes around link value to avoid markup parsing issues with hyphens
                    yield Label(
                        f'[link="{sym}"]{sym}[/link]',
                        classes="quick-symbol",
                        id=f"quick-{sym}",
                    )

    def _get_data_type_options(self) -> list[tuple[str, str]]:
        """Get data type options."""
        return [
            ("Equity", "equity"),
            ("Crypto", "crypto"),
            ("Options", "options"),
            ("Economic", "economic"),
        ]

    def _get_provider_options(self, data_type: str) -> list[tuple[str, str]]:
        """Get provider options for data type."""
        if data_type == "equity":
            return EQUITY_PROVIDERS
        elif data_type == "crypto":
            return CRYPTO_PROVIDERS
        elif data_type == "options":
            return OPTIONS_PROVIDERS
        elif data_type == "economic":
            return ECONOMIC_PROVIDERS
        return EQUITY_PROVIDERS

    @on(Input.Changed, "#symbol-input")
    def _on_symbol_changed(self, event: Input.Changed) -> None:
        """Handle symbol input change."""
        self.symbol = event.value.upper().strip()
        self.post_message(self.SymbolChanged(self.symbol))

    @on(Input.Submitted, "#symbol-input")
    def _on_symbol_submitted(self, event: Input.Submitted) -> None:
        """Handle symbol submission (Enter key)."""
        symbol = event.value.upper().strip()
        if symbol:
            self._emit_selection(symbol)

    @on(Select.Changed, "#provider-select")
    def _on_provider_changed(self, event: Select.Changed) -> None:
        """Handle provider selection change."""
        if event.value:
            self.provider = str(event.value)

    @on(Select.Changed, "#data-type-select")
    def _on_data_type_changed(self, event: Select.Changed) -> None:
        """Handle data type change."""
        if event.value:
            self.data_type = str(event.value)
            # Update provider options
            try:
                provider_select = self.query_one("#provider-select", Select)
                provider_select.set_options(self._get_provider_options(self.data_type))
                # Set default provider for data type
                defaults = {
                    "equity": "yfinance",
                    "crypto": "yfinance",
                    "options": "cboe",
                    "economic": "fred",
                }
                self.provider = defaults.get(self.data_type, "yfinance")
                provider_select.value = self.provider
            except Exception:
                pass

    def _on_click(self, event) -> None:
        """Handle click on quick symbols."""
        # Check if clicked on a quick symbol label
        if hasattr(event, "widget") and event.widget:
            widget_id = getattr(event.widget, "id", "")
            if widget_id and widget_id.startswith("quick-"):
                symbol = widget_id.replace("quick-", "")
                self._set_symbol(symbol)

    def _set_symbol(self, symbol: str) -> None:
        """Set symbol and trigger search."""
        try:
            input_widget = self.query_one("#symbol-input", Input)
            input_widget.value = symbol
            self.symbol = symbol
            self._emit_selection(symbol)
        except Exception:
            pass

    def _emit_selection(self, symbol: str) -> None:
        """Emit symbol selected message."""
        # Normalize crypto symbols: BTC -> BTC-USD, ETH -> ETH-USD
        normalized_symbol = self._normalize_symbol(symbol)

        selection = SymbolSelection(
            symbol=normalized_symbol,
            provider=self.provider,
            data_type=self.data_type,
        )
        self.post_message(self.SymbolSelected(selection))

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol based on data type."""
        symbol = symbol.upper().strip()

        if self.data_type == "crypto":
            # Common crypto symbols that need -USD suffix
            crypto_bases = {
                "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "MATIC",
                "AVAX", "LINK", "UNI", "ATOM", "LTC", "BCH", "XLM", "ALGO",
                "VET", "FIL", "TRX", "ETC", "XMR", "AAVE", "MKR", "COMP",
            }
            # If it's a bare crypto symbol, add -USD suffix
            if symbol in crypto_bases:
                return f"{symbol}-USD"
            # If it's already in format like BTCUSD, convert to BTC-USD
            for base in crypto_bases:
                if symbol == f"{base}USD":
                    return f"{base}-USD"

        return symbol

    def get_selection(self) -> SymbolSelection:
        """Get current selection."""
        return SymbolSelection(
            symbol=self.symbol,
            provider=self.provider,
            data_type=self.data_type,
        )

    def set_symbol(self, symbol: str) -> None:
        """Set the symbol programmatically."""
        self._set_symbol(symbol)

    def set_provider(self, provider: str) -> None:
        """Set the provider programmatically."""
        self.provider = provider
        try:
            provider_select = self.query_one("#provider-select", Select)
            provider_select.value = provider
        except Exception:
            pass

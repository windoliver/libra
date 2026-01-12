"""
Funding Rate Dashboard Widget.

Displays funding rate arbitrage data in real-time:
- Current funding rates across exchanges
- Best arbitrage opportunities
- Active positions and P&L
- Historical rate trends

See: https://github.com/windoliver/libra/issues/13
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable, Static


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class FundingRateDisplayData:
    """Display data for a funding rate."""

    symbol: str
    exchange: str
    funding_rate: Decimal
    annualized_rate: Decimal
    mark_price: Decimal
    index_price: Decimal
    basis_bps: Decimal
    next_funding_time: datetime
    predicted_rate: Decimal | None = None

    @property
    def rate_formatted(self) -> str:
        """Format funding rate as percentage."""
        return f"{float(self.funding_rate) * 100:.4f}%"

    @property
    def apr_formatted(self) -> str:
        """Format annualized rate as percentage."""
        return f"{float(self.annualized_rate) * 100:.2f}%"

    @property
    def basis_formatted(self) -> str:
        """Format basis in bps."""
        return f"{float(self.basis_bps):.1f}"

    @property
    def time_to_funding(self) -> str:
        """Format time to next funding."""
        now = datetime.now()
        delta = self.next_funding_time - now
        if delta.total_seconds() < 0:
            return "NOW"
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"

    @property
    def direction_icon(self) -> str:
        """Get icon indicating rate direction."""
        if self.funding_rate > 0:
            return "▲"  # Longs pay shorts
        elif self.funding_rate < 0:
            return "▼"  # Shorts pay longs
        return "●"

    @property
    def is_opportunity(self) -> bool:
        """Check if this is a viable opportunity."""
        return abs(self.funding_rate) >= Decimal("0.0001")


@dataclass
class ArbitragePositionDisplayData:
    """Display data for an arbitrage position."""

    symbol: str
    direction: str  # "long_spot_short_perp" or "short_spot_long_perp"
    size_usd: Decimal
    entry_funding_rate: Decimal
    cumulative_funding: Decimal
    total_pnl: Decimal
    funding_payments: int
    holding_hours: float

    @property
    def direction_short(self) -> str:
        """Short direction label."""
        if self.direction == "long_spot_short_perp":
            return "L-Spot/S-Perp"
        return "S-Spot/L-Perp"

    @property
    def pnl_formatted(self) -> str:
        """Format P&L with sign."""
        pnl = float(self.total_pnl)
        if pnl >= 0:
            return f"+${pnl:,.2f}"
        return f"-${abs(pnl):,.2f}"

    @property
    def funding_formatted(self) -> str:
        """Format cumulative funding."""
        funding = float(self.cumulative_funding)
        if funding >= 0:
            return f"+${funding:,.2f}"
        return f"-${abs(funding):,.2f}"

    @property
    def size_formatted(self) -> str:
        """Format position size."""
        size = float(self.size_usd)
        if size >= 1_000_000:
            return f"${size / 1_000_000:.1f}M"
        elif size >= 1_000:
            return f"${size / 1_000:.0f}K"
        return f"${size:.0f}"

    @property
    def holding_formatted(self) -> str:
        """Format holding time."""
        if self.holding_hours < 1:
            return f"{int(self.holding_hours * 60)}m"
        elif self.holding_hours < 24:
            return f"{self.holding_hours:.1f}h"
        return f"{self.holding_hours / 24:.1f}d"


@dataclass
class FundingRateDashboardData:
    """Data for the funding rate dashboard."""

    rates: list[FundingRateDisplayData] = field(default_factory=list)
    positions: list[ArbitragePositionDisplayData] = field(default_factory=list)
    total_funding_collected: Decimal = Decimal("0")
    total_positions_pnl: Decimal = Decimal("0")
    active_positions: int = 0
    connected_exchanges: list[str] = field(default_factory=list)


# =============================================================================
# Widgets
# =============================================================================


class FundingRateSummary(Static):
    """Summary card showing funding rate statistics."""

    DEFAULT_CSS = """
    FundingRateSummary {
        height: 5;
        padding: 0 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    FundingRateSummary .title {
        text-style: bold;
        color: $text;
    }

    FundingRateSummary .value {
        text-style: bold;
    }

    FundingRateSummary .positive {
        color: $success;
    }

    FundingRateSummary .negative {
        color: $error;
    }

    FundingRateSummary .neutral {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        title: str,
        value: str,
        status: str = "neutral",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._title = title
        self._value = value
        self._status = status

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="title")
        yield Static(self._value, classes=f"value {self._status}")

    def update_value(self, value: str, status: str = "neutral") -> None:
        """Update the displayed value."""
        self._value = value
        self._status = status
        try:
            value_widget = self.query_one(".value", Static)
            value_widget.update(value)
            value_widget.set_classes(f"value {status}")
        except Exception:
            pass


class FundingRatesTable(Static):
    """Table displaying current funding rates."""

    DEFAULT_CSS = """
    FundingRatesTable {
        height: 100%;
        border: round $primary-darken-2;
        background: $surface;
    }

    FundingRatesTable DataTable {
        height: 100%;
    }

    FundingRatesTable .header {
        background: $primary-darken-3;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Funding Rates", classes="header")
        table = DataTable(id="rates-table")
        table.cursor_type = "row"
        yield table

    def on_mount(self) -> None:
        """Set up the table columns."""
        table = self.query_one("#rates-table", DataTable)
        table.add_columns(
            "Symbol",
            "Exch",
            "Rate",
            "APR",
            "Basis",
            "Next",
            "Dir",
        )

    def update_rates(self, rates: list[FundingRateDisplayData]) -> None:
        """Update the funding rates display."""
        table = self.query_one("#rates-table", DataTable)
        table.clear()

        # Sort by absolute rate descending
        sorted_rates = sorted(rates, key=lambda r: abs(r.funding_rate), reverse=True)

        for rate in sorted_rates:
            # Color based on rate direction
            if rate.funding_rate > 0:
                rate_style = "[green]"
            elif rate.funding_rate < 0:
                rate_style = "[red]"
            else:
                rate_style = ""

            table.add_row(
                rate.symbol.split("/")[0],  # Just the base asset
                rate.exchange[:4].upper(),
                f"{rate_style}{rate.rate_formatted}[/]",
                rate.apr_formatted,
                rate.basis_formatted,
                rate.time_to_funding,
                rate.direction_icon,
            )


class ArbitragePositionsTable(Static):
    """Table displaying active arbitrage positions."""

    DEFAULT_CSS = """
    ArbitragePositionsTable {
        height: 100%;
        border: round $primary-darken-2;
        background: $surface;
    }

    ArbitragePositionsTable DataTable {
        height: 100%;
    }

    ArbitragePositionsTable .header {
        background: $primary-darken-3;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Active Positions", classes="header")
        table = DataTable(id="positions-table")
        table.cursor_type = "row"
        yield table

    def on_mount(self) -> None:
        """Set up the table columns."""
        table = self.query_one("#positions-table", DataTable)
        table.add_columns(
            "Symbol",
            "Direction",
            "Size",
            "Funding",
            "P&L",
            "Payments",
            "Time",
        )

    def update_positions(self, positions: list[ArbitragePositionDisplayData]) -> None:
        """Update the positions display."""
        table = self.query_one("#positions-table", DataTable)
        table.clear()

        for pos in positions:
            # Color P&L
            pnl_val = float(pos.total_pnl)
            if pnl_val > 0:
                pnl_style = "[green]"
            elif pnl_val < 0:
                pnl_style = "[red]"
            else:
                pnl_style = ""

            table.add_row(
                pos.symbol.split("/")[0],
                pos.direction_short,
                pos.size_formatted,
                pos.funding_formatted,
                f"{pnl_style}{pos.pnl_formatted}[/]",
                str(pos.funding_payments),
                pos.holding_formatted,
            )


class OpportunityCard(Static):
    """Card highlighting best arbitrage opportunity."""

    DEFAULT_CSS = """
    OpportunityCard {
        height: 7;
        padding: 1;
        border: round $success;
        background: $surface;
    }

    OpportunityCard .title {
        text-style: bold;
        color: $success;
    }

    OpportunityCard .symbol {
        text-style: bold;
        color: $text;
    }

    OpportunityCard .rate {
        color: $success;
        text-style: bold;
    }

    OpportunityCard .exchange {
        color: $text-muted;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._opportunity: FundingRateDisplayData | None = None

    def compose(self) -> ComposeResult:
        yield Static("Best Opportunity", classes="title")
        yield Static("--", id="opp-symbol", classes="symbol")
        yield Static("Rate: --", id="opp-rate", classes="rate")
        yield Static("Exchange: --", id="opp-exchange", classes="exchange")

    def update_opportunity(self, opportunity: FundingRateDisplayData | None) -> None:
        """Update the displayed opportunity."""
        self._opportunity = opportunity
        try:
            if opportunity:
                symbol = opportunity.symbol.split("/")[0]
                action = "SHORT perp" if opportunity.funding_rate > 0 else "LONG perp"
                self.query_one("#opp-symbol", Static).update(f"{symbol} - {action}")
                self.query_one("#opp-rate", Static).update(
                    f"Rate: {opportunity.rate_formatted} ({opportunity.apr_formatted} APR)"
                )
                self.query_one("#opp-exchange", Static).update(
                    f"Exchange: {opportunity.exchange.upper()}"
                )
            else:
                self.query_one("#opp-symbol", Static).update("No opportunities")
                self.query_one("#opp-rate", Static).update("Rate: --")
                self.query_one("#opp-exchange", Static).update("Exchange: --")
        except Exception:
            pass


class ExchangeStatusBar(Static):
    """Status bar showing connected exchanges."""

    DEFAULT_CSS = """
    ExchangeStatusBar {
        height: 1;
        background: $primary-darken-3;
        padding: 0 1;
    }

    ExchangeStatusBar .label {
        color: $text-muted;
    }

    ExchangeStatusBar .exchange {
        color: $success;
        margin: 0 1;
    }
    """

    def __init__(self, exchanges: list[str] | None = None, id: str | None = None) -> None:
        super().__init__(id=id)
        self._exchanges = exchanges or []

    def compose(self) -> ComposeResult:
        yield Static("Exchanges: ", classes="label")
        for exchange in self._exchanges:
            yield Static(exchange.upper(), classes="exchange")

    def update_exchanges(self, exchanges: list[str]) -> None:
        """Update connected exchanges."""
        self._exchanges = exchanges
        self.refresh(recompose=True)


class FundingRateDashboard(Container):
    """
    Main funding rate arbitrage dashboard.

    Displays funding rates, opportunities, and positions.
    """

    DEFAULT_CSS = """
    FundingRateDashboard {
        height: 1fr;
        min-height: 25;
        padding: 1;
    }

    FundingRateDashboard #summary-row {
        height: 5;
        margin-bottom: 1;
    }

    FundingRateDashboard #summary-row > FundingRateSummary {
        width: 1fr;
        margin-right: 1;
    }

    FundingRateDashboard #summary-row > OpportunityCard {
        width: 2fr;
    }

    FundingRateDashboard #tables-row {
        height: 1fr;
        min-height: 15;
    }

    FundingRateDashboard #rates-container {
        width: 1fr;
        height: 1fr;
        min-height: 12;
        margin-right: 1;
    }

    FundingRateDashboard #positions-container {
        width: 1fr;
        height: 1fr;
        min-height: 12;
    }
    """

    class OpportunitySelected(Message):
        """Fired when an opportunity is selected."""

        def __init__(self, rate_data: FundingRateDisplayData) -> None:
            super().__init__()
            self.rate_data = rate_data

    # Reactive data
    data: reactive[FundingRateDashboardData] = reactive(
        FundingRateDashboardData, init=False
    )

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._data = FundingRateDashboardData()
        # Cached widget refs
        self._rates_table: FundingRatesTable | None = None
        self._positions_table: ArbitragePositionsTable | None = None
        self._funding_summary: FundingRateSummary | None = None
        self._pnl_summary: FundingRateSummary | None = None
        self._positions_summary: FundingRateSummary | None = None
        self._opportunity_card: OpportunityCard | None = None

    def compose(self) -> ComposeResult:
        # Summary row
        with Horizontal(id="summary-row"):
            yield FundingRateSummary(
                "Total Funding",
                "$0.00",
                "neutral",
                id="total-funding",
            )
            yield FundingRateSummary(
                "Total P&L",
                "$0.00",
                "neutral",
                id="total-pnl",
            )
            yield FundingRateSummary(
                "Active Positions",
                "0",
                "neutral",
                id="active-positions",
            )
            yield OpportunityCard(id="best-opportunity")

        # Tables row
        with Horizontal(id="tables-row"):
            # Rates table
            with Container(id="rates-container"):
                yield FundingRatesTable(id="rates-table-widget")

            # Positions table
            with Container(id="positions-container"):
                yield ArbitragePositionsTable(id="positions-table-widget")

    def watch_data(self, data: FundingRateDashboardData) -> None:
        """React to data changes."""
        self._update_display(data)

    def _update_display(self, data: FundingRateDashboardData) -> None:
        """Update all dashboard components."""
        try:
            # Update summary cards
            total_funding = float(data.total_funding_collected)
            funding_status = "positive" if total_funding > 0 else "negative" if total_funding < 0 else "neutral"
            self.query_one("#total-funding", FundingRateSummary).update_value(
                f"${total_funding:,.2f}", funding_status
            )

            total_pnl = float(data.total_positions_pnl)
            pnl_status = "positive" if total_pnl > 0 else "negative" if total_pnl < 0 else "neutral"
            self.query_one("#total-pnl", FundingRateSummary).update_value(
                f"${total_pnl:,.2f}", pnl_status
            )

            self.query_one("#active-positions", FundingRateSummary).update_value(
                str(data.active_positions), "neutral"
            )

            # Update best opportunity
            best_opp = None
            if data.rates:
                # Find best opportunity (highest absolute rate with reasonable basis)
                opportunities = [r for r in data.rates if r.is_opportunity]
                if opportunities:
                    best_opp = max(opportunities, key=lambda r: abs(r.funding_rate))
            self.query_one("#best-opportunity", OpportunityCard).update_opportunity(best_opp)

            # Update rates table
            self.query_one("#rates-table-widget", FundingRatesTable).update_rates(data.rates)

            # Update positions table
            self.query_one("#positions-table-widget", ArbitragePositionsTable).update_positions(
                data.positions
            )

        except Exception:
            pass

    def update_data(self, data: FundingRateDashboardData) -> None:
        """Update dashboard with new data."""
        self._data = data

        # Ensure widgets are cached (may not be cached if called before on_mount)
        if not self._rates_table:
            try:
                self._rates_table = self.query_one("#rates-table-widget", FundingRatesTable)
            except Exception:
                pass
        if not self._positions_table:
            try:
                self._positions_table = self.query_one("#positions-table-widget", ArbitragePositionsTable)
            except Exception:
                pass
        if not self._funding_summary:
            try:
                self._funding_summary = self.query_one("#total-funding", FundingRateSummary)
            except Exception:
                pass
        if not self._pnl_summary:
            try:
                self._pnl_summary = self.query_one("#total-pnl", FundingRateSummary)
            except Exception:
                pass
        if not self._positions_summary:
            try:
                self._positions_summary = self.query_one("#active-positions", FundingRateSummary)
            except Exception:
                pass
        if not self._opportunity_card:
            try:
                self._opportunity_card = self.query_one("#best-opportunity", OpportunityCard)
            except Exception:
                pass

        # Update summary cards
        total_funding = float(data.total_funding_collected)
        funding_status = "positive" if total_funding > 0 else "negative" if total_funding < 0 else "neutral"
        if self._funding_summary:
            self._funding_summary.update_value(f"${total_funding:,.2f}", funding_status)

        total_pnl = float(data.total_positions_pnl)
        pnl_status = "positive" if total_pnl > 0 else "negative" if total_pnl < 0 else "neutral"
        if self._pnl_summary:
            self._pnl_summary.update_value(f"${total_pnl:,.2f}", pnl_status)

        if self._positions_summary:
            self._positions_summary.update_value(str(data.active_positions), "neutral")

        # Update best opportunity
        best_opp = None
        if data.rates:
            opportunities = [r for r in data.rates if r.is_opportunity]
            if opportunities:
                best_opp = max(opportunities, key=lambda r: abs(r.funding_rate))
        if self._opportunity_card:
            self._opportunity_card.update_opportunity(best_opp)

        # Update tables
        if self._rates_table:
            self._rates_table.update_rates(data.rates)

        if self._positions_table:
            self._positions_table.update_positions(data.positions)


# =============================================================================
# Demo Data
# =============================================================================


def create_demo_funding_dashboard_data() -> FundingRateDashboardData:
    """Create demo data for the funding rate dashboard."""
    from datetime import timedelta

    now = datetime.now()
    next_funding = now + timedelta(hours=4)

    rates = [
        FundingRateDisplayData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0003"),
            annualized_rate=Decimal("0.3285"),
            mark_price=Decimal("95000.00"),
            index_price=Decimal("94950.00"),
            basis_bps=Decimal("5.26"),
            next_funding_time=next_funding,
            predicted_rate=Decimal("0.00025"),
        ),
        FundingRateDisplayData(
            symbol="ETH/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0002"),
            annualized_rate=Decimal("0.219"),
            mark_price=Decimal("3400.00"),
            index_price=Decimal("3395.00"),
            basis_bps=Decimal("14.73"),
            next_funding_time=next_funding,
        ),
        FundingRateDisplayData(
            symbol="SOL/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("-0.0001"),
            annualized_rate=Decimal("-0.1095"),
            mark_price=Decimal("180.00"),
            index_price=Decimal("180.50"),
            basis_bps=Decimal("-27.70"),
            next_funding_time=next_funding,
        ),
        FundingRateDisplayData(
            symbol="BTC/USDT:USDT",
            exchange="bybit",
            funding_rate=Decimal("0.00035"),
            annualized_rate=Decimal("0.38325"),
            mark_price=Decimal("95010.00"),
            index_price=Decimal("94950.00"),
            basis_bps=Decimal("6.32"),
            next_funding_time=next_funding,
        ),
    ]

    positions = [
        ArbitragePositionDisplayData(
            symbol="BTC/USDT:USDT",
            direction="long_spot_short_perp",
            size_usd=Decimal("50000"),
            entry_funding_rate=Decimal("0.00032"),
            cumulative_funding=Decimal("85.50"),
            total_pnl=Decimal("72.30"),
            funding_payments=3,
            holding_hours=24.5,
        ),
        ArbitragePositionDisplayData(
            symbol="ETH/USDT:USDT",
            direction="long_spot_short_perp",
            size_usd=Decimal("25000"),
            entry_funding_rate=Decimal("0.00025"),
            cumulative_funding=Decimal("31.25"),
            total_pnl=Decimal("28.10"),
            funding_payments=2,
            holding_hours=16.0,
        ),
    ]

    return FundingRateDashboardData(
        rates=rates,
        positions=positions,
        total_funding_collected=Decimal("116.75"),
        total_positions_pnl=Decimal("100.40"),
        active_positions=2,
        connected_exchanges=["binance", "bybit"],
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data models
    "FundingRateDisplayData",
    "ArbitragePositionDisplayData",
    "FundingRateDashboardData",
    # Widgets
    "FundingRateSummary",
    "FundingRatesTable",
    "ArbitragePositionsTable",
    "OpportunityCard",
    "ExchangeStatusBar",
    "FundingRateDashboard",
    # Demo
    "create_demo_funding_dashboard_data",
]

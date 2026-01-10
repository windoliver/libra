"""
Fundamentals Panel Widget.

Displays company fundamental data including income statement, balance sheet,
key ratios, and company profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Select, Static, TabbedContent, TabPane


if TYPE_CHECKING:
    from libra.gateways.openbb.fetchers import CompanyProfile, FundamentalRecord


def _format_number(value: Decimal | None, prefix: str = "$", suffix: str = "") -> str:
    """Format a number for display."""
    if value is None:
        return "--"
    v = float(value)
    if abs(v) >= 1_000_000_000_000:
        return f"{prefix}{v/1_000_000_000_000:.1f}T{suffix}"
    elif abs(v) >= 1_000_000_000:
        return f"{prefix}{v/1_000_000_000:.1f}B{suffix}"
    elif abs(v) >= 1_000_000:
        return f"{prefix}{v/1_000_000:.1f}M{suffix}"
    elif abs(v) >= 1_000:
        return f"{prefix}{v/1_000:.1f}K{suffix}"
    else:
        return f"{prefix}{v:.2f}{suffix}"


def _format_ratio(value: Decimal | None, suffix: str = "") -> str:
    """Format a ratio for display."""
    if value is None:
        return "--"
    return f"{float(value):.2f}{suffix}"


def _format_percent(value: Decimal | None) -> str:
    """Format a percentage for display."""
    if value is None:
        return "--"
    v = float(value)
    color = "green" if v >= 0 else "red"
    return f"[{color}]{v:.1f}%[/{color}]"


@dataclass
class FundamentalsData:
    """Container for fundamental data."""

    symbol: str
    profile: CompanyProfile | None = None
    income: list[FundamentalRecord] = field(default_factory=list)
    balance: list[FundamentalRecord] = field(default_factory=list)
    cash_flow: list[FundamentalRecord] = field(default_factory=list)
    ratios: list[FundamentalRecord] = field(default_factory=list)


class FundamentalsPanel(Container):
    """
    Panel displaying company fundamental data.

    Features:
    - Company profile overview
    - Income statement table
    - Balance sheet table
    - Key financial ratios
    - Cash flow statement
    """

    DEFAULT_CSS = """
    FundamentalsPanel {
        height: 100%;
    }

    FundamentalsPanel .fundamentals-header {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }

    FundamentalsPanel .company-name {
        width: 1fr;
        text-style: bold;
    }

    FundamentalsPanel .period-select {
        width: 15;
    }

    FundamentalsPanel .profile-grid {
        height: auto;
        padding: 1;
        layout: grid;
        grid-size: 4;
        grid-gutter: 1;
    }

    FundamentalsPanel .profile-card {
        height: auto;
        padding: 1;
        background: $surface;
        border: solid $primary-background;
    }

    FundamentalsPanel .profile-label {
        color: $text-muted;
    }

    FundamentalsPanel .profile-value {
        text-style: bold;
    }

    FundamentalsPanel .data-table-container {
        height: 1fr;
        padding: 1;
    }

    FundamentalsPanel DataTable {
        height: 100%;
    }

    FundamentalsPanel .no-data {
        height: 100%;
        align: center middle;
        color: $text-muted;
    }
    """

    # Reactive data
    data: reactive[FundamentalsData | None] = reactive(None, init=False)
    period: reactive[str] = reactive("annual", init=False)

    class DataRequested(Message):
        """Message requesting fundamental data load."""

        def __init__(
            self, symbol: str, statement: str, period: str, provider: str
        ) -> None:
            self.symbol = symbol
            self.statement = statement
            self.period = period
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
        with Horizontal(classes="fundamentals-header"):
            yield Label(
                "[bold]Company Fundamentals[/bold]",
                id="company-name",
                classes="company-name",
            )
            yield Select(
                options=[
                    ("Annual", "annual"),
                    ("Quarterly", "quarter"),
                    ("TTM", "ttm"),
                ],
                value="annual",
                id="period-select",
                classes="period-select",
            )

        with TabbedContent(initial="profile"):
            with TabPane("Profile", id="profile"):
                yield self._create_profile_view()

            with TabPane("Income", id="income"):
                yield self._create_table_view("income-table")

            with TabPane("Balance", id="balance"):
                yield self._create_table_view("balance-table")

            with TabPane("Cash Flow", id="cashflow"):
                yield self._create_table_view("cashflow-table")

            with TabPane("Ratios", id="ratios"):
                yield self._create_ratios_view()

    def _create_profile_view(self) -> Container:
        """Create the profile view."""
        container = ScrollableContainer(classes="profile-container")
        return container

    def _create_table_view(self, table_id: str) -> Container:
        """Create a table view for financial statements."""
        container = Container(classes="data-table-container")
        return container

    def _create_ratios_view(self) -> Container:
        """Create the ratios view."""
        container = ScrollableContainer(classes="ratios-container")
        return container

    def on_mount(self) -> None:
        """Set up tables after mount."""
        self._setup_tables()

    def _setup_tables(self) -> None:
        """Set up data tables."""
        # Income table
        try:
            income_container = self.query_one("#income .data-table-container", Container)
            income_table = DataTable(id="income-table")
            income_table.add_columns("Metric", "Period 1", "Period 2", "Period 3", "Period 4")
            income_container.mount(income_table)
        except Exception:
            pass

        # Balance table
        try:
            balance_container = self.query_one("#balance .data-table-container", Container)
            balance_table = DataTable(id="balance-table")
            balance_table.add_columns("Metric", "Period 1", "Period 2", "Period 3", "Period 4")
            balance_container.mount(balance_table)
        except Exception:
            pass

        # Cash flow table
        try:
            cashflow_container = self.query_one("#cashflow .data-table-container", Container)
            cashflow_table = DataTable(id="cashflow-table")
            cashflow_table.add_columns("Metric", "Period 1", "Period 2", "Period 3", "Period 4")
            cashflow_container.mount(cashflow_table)
        except Exception:
            pass

    @on(Select.Changed, "#period-select")
    def _on_period_changed(self, event: Select.Changed) -> None:
        """Handle period change."""
        if event.value:
            self.period = str(event.value)
            if self.data:
                # Request new data with updated period
                self.post_message(
                    self.DataRequested(
                        self.data.symbol,
                        "income",
                        self.period,
                        "fmp",
                    )
                )

    def watch_data(self, data: FundamentalsData | None) -> None:
        """Update display when data changes."""
        if data:
            self._update_profile(data)
            self._update_income_table(data)
            self._update_balance_table(data)
            self._update_cashflow_table(data)
            self._update_ratios(data)
            self._update_header(data)

    def _update_header(self, data: FundamentalsData) -> None:
        """Update header with company name."""
        try:
            header = self.query_one("#company-name", Label)
            if data.profile:
                header.update(f"[bold]{data.profile.name}[/bold] ({data.symbol})")
            else:
                header.update(f"[bold]{data.symbol}[/bold]")
        except Exception:
            pass

    def _update_profile(self, data: FundamentalsData) -> None:
        """Update profile view."""
        try:
            container = self.query_one("#profile .profile-container", ScrollableContainer)
            container.remove_children()

            if not data.profile:
                container.mount(Static("[dim]No profile data available[/dim]", classes="no-data"))
                return

            profile = data.profile

            # Create profile cards
            cards = Horizontal(classes="profile-grid")

            profile_items = [
                ("Market Cap", _format_number(profile.market_cap)),
                ("P/E Ratio", _format_ratio(profile.pe_ratio)),
                ("Forward P/E", _format_ratio(profile.forward_pe)),
                ("PEG Ratio", _format_ratio(profile.peg_ratio)),
                ("Price/Book", _format_ratio(profile.price_to_book)),
                ("Price/Sales", _format_ratio(profile.price_to_sales)),
                ("Dividend Yield", _format_percent(profile.dividend_yield) if profile.dividend_yield else "--"),
                ("Beta", _format_ratio(profile.beta)),
                ("52W High", _format_number(profile.week_52_high)),
                ("52W Low", _format_number(profile.week_52_low)),
                ("Avg Volume", _format_number(profile.avg_volume, prefix="")),
                ("Employees", f"{profile.employees:,}" if profile.employees else "--"),
            ]

            for label, value in profile_items:
                card = Vertical(
                    Static(f"[dim]{label}[/dim]", classes="profile-label"),
                    Static(value, classes="profile-value"),
                    classes="profile-card",
                )
                cards.mount(card)

            container.mount(cards)

            # Add description if available
            if profile.description:
                desc_text = profile.description[:500] + "..." if len(profile.description) > 500 else profile.description
                container.mount(Static(f"\n[dim]{desc_text}[/dim]"))

        except Exception:
            pass

    def _update_income_table(self, data: FundamentalsData) -> None:
        """Update income statement table."""
        try:
            table = self.query_one("#income-table", DataTable)
            table.clear()

            if not data.income:
                return

            # Get periods
            periods = data.income[:4]  # Max 4 periods

            # Update column headers
            table.clear(columns=True)
            table.add_column("Metric")
            for record in periods:
                table.add_column(record.period[:12] if record.period else "Period")

            # Add rows
            metrics = [
                ("Revenue", "revenue"),
                ("Cost of Revenue", "cost_of_revenue"),
                ("Gross Profit", "gross_profit"),
                ("Operating Expenses", "operating_expenses"),
                ("Operating Income", "operating_income"),
                ("EBITDA", "ebitda"),
                ("Net Income", "net_income"),
                ("EPS", "eps"),
                ("EPS Diluted", "eps_diluted"),
            ]

            for label, attr in metrics:
                row = [label]
                for record in periods:
                    value = getattr(record, attr, None)
                    if attr in ("eps", "eps_diluted"):
                        row.append(_format_ratio(value, prefix="$"))
                    else:
                        row.append(_format_number(value))
                table.add_row(*row)

        except Exception:
            pass

    def _update_balance_table(self, data: FundamentalsData) -> None:
        """Update balance sheet table."""
        try:
            table = self.query_one("#balance-table", DataTable)
            table.clear()

            if not data.balance:
                return

            periods = data.balance[:4]

            table.clear(columns=True)
            table.add_column("Metric")
            for record in periods:
                table.add_column(record.period[:12] if record.period else "Period")

            metrics = [
                ("Total Assets", "total_assets"),
                ("Total Liabilities", "total_liabilities"),
                ("Total Equity", "total_equity"),
                ("Cash & Equivalents", "cash_and_equivalents"),
                ("Total Debt", "total_debt"),
            ]

            for label, attr in metrics:
                row = [label]
                for record in periods:
                    value = getattr(record, attr, None)
                    row.append(_format_number(value))
                table.add_row(*row)

        except Exception:
            pass

    def _update_cashflow_table(self, data: FundamentalsData) -> None:
        """Update cash flow table."""
        try:
            table = self.query_one("#cashflow-table", DataTable)
            table.clear()

            if not data.cash_flow:
                return

            periods = data.cash_flow[:4]

            table.clear(columns=True)
            table.add_column("Metric")
            for record in periods:
                table.add_column(record.period[:12] if record.period else "Period")

            metrics = [
                ("Operating Cash Flow", "operating_cash_flow"),
                ("Capital Expenditure", "capital_expenditure"),
                ("Free Cash Flow", "free_cash_flow"),
            ]

            for label, attr in metrics:
                row = [label]
                for record in periods:
                    value = getattr(record, attr, None)
                    row.append(_format_number(value))
                table.add_row(*row)

        except Exception:
            pass

    def _update_ratios(self, data: FundamentalsData) -> None:
        """Update ratios view."""
        try:
            container = self.query_one("#ratios .ratios-container", ScrollableContainer)
            container.remove_children()

            if not data.ratios:
                container.mount(Static("[dim]No ratio data available[/dim]", classes="no-data"))
                return

            # Use most recent ratios
            ratios = data.ratios[0] if data.ratios else None
            if not ratios:
                return

            cards = Horizontal(classes="profile-grid")

            ratio_items = [
                ("P/E Ratio", _format_ratio(ratios.pe_ratio)),
                ("P/B Ratio", _format_ratio(ratios.pb_ratio)),
                ("P/S Ratio", _format_ratio(ratios.ps_ratio)),
                ("ROE", _format_percent(ratios.roe)),
                ("ROA", _format_percent(ratios.roa)),
                ("Debt/Equity", _format_ratio(ratios.debt_to_equity)),
                ("Current Ratio", _format_ratio(ratios.current_ratio)),
                ("Quick Ratio", _format_ratio(ratios.quick_ratio)),
                ("Gross Margin", _format_percent(ratios.gross_margin)),
                ("Operating Margin", _format_percent(ratios.operating_margin)),
                ("Net Margin", _format_percent(ratios.net_margin)),
            ]

            for label, value in ratio_items:
                card = Vertical(
                    Static(f"[dim]{label}[/dim]", classes="profile-label"),
                    Static(value, classes="profile-value"),
                    classes="profile-card",
                )
                cards.mount(card)

            container.mount(cards)

        except Exception:
            pass

    def set_data(self, data: FundamentalsData) -> None:
        """Set fundamental data."""
        self.data = data

    def clear(self) -> None:
        """Clear the display."""
        self.data = None

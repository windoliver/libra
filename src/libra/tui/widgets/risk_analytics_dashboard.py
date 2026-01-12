"""
Risk Analytics Dashboard Widget.

Comprehensive TUI dashboard for advanced risk management:
- VaR display with multiple methodologies
- Stress test results visualization
- Correlation matrix heatmap
- Margin utilization monitoring

Issue #15: Advanced Risk Management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label, ProgressBar, Static


@dataclass
class VaRDisplayData:
    """VaR data for display."""

    method: str  # "Historical", "Parametric", "Monte Carlo"
    confidence_level: float
    time_horizon_days: int
    var_amount: Decimal
    var_pct: float
    cvar_amount: Decimal
    cvar_pct: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StressTestDisplayData:
    """Stress test result for display."""

    scenario_name: str
    scenario_type: str  # "Historical", "Hypothetical"
    total_pnl: Decimal
    total_pnl_pct: float
    is_severe: bool
    breaches_limits: bool


@dataclass
class CorrelationDisplayData:
    """Correlation data for display."""

    symbol1: str
    symbol2: str
    correlation: float
    regime: str  # "Low", "Normal", "High", "Crisis"


@dataclass
class MarginDisplayData:
    """Margin data for display."""

    symbol: str
    side: str
    leverage: float
    margin_ratio: float
    distance_to_liquidation: float
    alert_level: str


@dataclass
class RiskAnalyticsDashboardData:
    """Complete data for risk analytics dashboard."""

    # VaR data
    var_results: list[VaRDisplayData] = field(default_factory=list)
    portfolio_value: Decimal = Decimal("0")

    # Stress test data
    stress_results: list[StressTestDisplayData] = field(default_factory=list)
    worst_case_pnl: Decimal = Decimal("0")
    expected_loss: Decimal = Decimal("0")

    # Correlation data
    correlations: list[CorrelationDisplayData] = field(default_factory=list)
    avg_correlation: float = 0.0
    correlation_regime: str = "Normal"
    diversification_ratio: float = 1.0

    # Concentration
    hhi: float = 0.0
    effective_positions: float = 0.0
    is_concentrated: bool = False

    # Margin data
    margin_positions: list[MarginDisplayData] = field(default_factory=list)
    total_margin: Decimal = Decimal("0")
    available_margin: Decimal = Decimal("0")
    margin_utilization: float = 0.0
    margin_level: float = 0.0
    margin_alert_level: str = "Healthy"
    effective_leverage: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)


class VaRPanel(Static):
    """Panel displaying VaR metrics."""

    DEFAULT_CSS = """
    VaRPanel {
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    VaRPanel .panel-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    VaRPanel .var-value {
        color: $error;
        text-style: bold;
    }

    VaRPanel .cvar-value {
        color: $warning;
    }
    """

    def __init__(
        self,
        var_data: list[VaRDisplayData] | None = None,
        portfolio_value: Decimal = Decimal("0"),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._var_data = var_data or []
        self._portfolio_value = portfolio_value

    def compose(self) -> ComposeResult:
        yield Label("Value at Risk (VaR)", classes="panel-title")
        yield Static(id="var-content")

    def on_mount(self) -> None:
        self._update_display()

    def update_data(
        self,
        var_data: list[VaRDisplayData],
        portfolio_value: Decimal,
    ) -> None:
        self._var_data = var_data
        self._portfolio_value = portfolio_value
        self._update_display()

    def _update_display(self) -> None:
        content = self.query_one("#var-content", Static)

        if not self._var_data:
            content.update("No VaR data available")
            return

        table = Table(box=None, expand=True)
        table.add_column("Method", style="cyan")
        table.add_column("Conf.", justify="right")
        table.add_column("VaR", justify="right", style="red")
        table.add_column("%", justify="right")
        table.add_column("CVaR", justify="right", style="yellow")

        for var in self._var_data:
            table.add_row(
                var.method,
                f"{var.confidence_level:.0%}",
                f"${var.var_amount:,.0f}",
                f"{var.var_pct:.2f}%",
                f"${var.cvar_amount:,.0f}",
            )

        content.update(table)


class StressTestPanel(Static):
    """Panel displaying stress test results."""

    DEFAULT_CSS = """
    StressTestPanel {
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    StressTestPanel .panel-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    StressTestPanel .severe {
        color: $error;
        text-style: bold;
    }

    StressTestPanel .warning {
        color: $warning;
    }
    """

    def __init__(
        self,
        stress_data: list[StressTestDisplayData] | None = None,
        worst_case: Decimal = Decimal("0"),
        expected_loss: Decimal = Decimal("0"),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._stress_data = stress_data or []
        self._worst_case = worst_case
        self._expected_loss = expected_loss

    def compose(self) -> ComposeResult:
        yield Label("Stress Test Scenarios", classes="panel-title")
        yield Static(id="stress-content")

    def on_mount(self) -> None:
        self._update_display()

    def update_data(
        self,
        stress_data: list[StressTestDisplayData],
        worst_case: Decimal,
        expected_loss: Decimal,
    ) -> None:
        self._stress_data = stress_data
        self._worst_case = worst_case
        self._expected_loss = expected_loss
        self._update_display()

    def _update_display(self) -> None:
        content = self.query_one("#stress-content", Static)

        if not self._stress_data:
            content.update("No stress test data available")
            return

        table = Table(box=None, expand=True)
        table.add_column("Scenario", style="cyan", no_wrap=True)
        table.add_column("Type", style="dim")
        table.add_column("PnL", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Status")

        # Show top 5 worst scenarios
        sorted_data = sorted(self._stress_data, key=lambda x: x.total_pnl)[:5]

        for stress in sorted_data:
            # Truncate scenario name
            name = stress.scenario_name[:20] + "..." if len(stress.scenario_name) > 20 else stress.scenario_name

            pnl_style = "red" if stress.total_pnl < 0 else "green"
            pnl_text = Text(f"${stress.total_pnl:,.0f}", style=pnl_style)

            pct_text = Text(f"{stress.total_pnl_pct:.1f}%", style=pnl_style)

            if stress.breaches_limits:
                status = Text("BREACH", style="red bold")
            elif stress.is_severe:
                status = Text("SEVERE", style="yellow")
            else:
                status = Text("OK", style="green")

            table.add_row(name, stress.scenario_type[:4], pnl_text, pct_text, status)

        # Add summary row
        table.add_row("", "", "", "", "")
        table.add_row(
            "Worst Case:",
            "",
            Text(f"${self._worst_case:,.0f}", style="red bold"),
            "",
            "",
        )
        table.add_row(
            "Expected Loss:",
            "",
            Text(f"${self._expected_loss:,.0f}", style="yellow"),
            "",
            "",
        )

        content.update(table)


class CorrelationPanel(Static):
    """Panel displaying correlation metrics."""

    DEFAULT_CSS = """
    CorrelationPanel {
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    CorrelationPanel .panel-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    CorrelationPanel .regime-low {
        color: $success;
    }

    CorrelationPanel .regime-normal {
        color: $text;
    }

    CorrelationPanel .regime-high {
        color: $warning;
    }

    CorrelationPanel .regime-crisis {
        color: $error;
        text-style: bold;
    }
    """

    def __init__(
        self,
        correlations: list[CorrelationDisplayData] | None = None,
        avg_correlation: float = 0.0,
        regime: str = "Normal",
        diversification_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._correlations = correlations or []
        self._avg_correlation = avg_correlation
        self._regime = regime
        self._div_ratio = diversification_ratio

    def compose(self) -> ComposeResult:
        yield Label("Correlation Analysis", classes="panel-title")
        yield Static(id="corr-summary")
        yield Static(id="corr-content")

    def on_mount(self) -> None:
        self._update_display()

    def update_data(
        self,
        correlations: list[CorrelationDisplayData],
        avg_correlation: float,
        regime: str,
        diversification_ratio: float,
    ) -> None:
        self._correlations = correlations
        self._avg_correlation = avg_correlation
        self._regime = regime
        self._div_ratio = diversification_ratio
        self._update_display()

    def _update_display(self) -> None:
        summary = self.query_one("#corr-summary", Static)
        content = self.query_one("#corr-content", Static)

        # Regime styling
        regime_style = {
            "Low": "green",
            "Normal": "white",
            "High": "yellow",
            "Crisis": "red bold",
        }.get(self._regime, "white")

        summary_text = Text()
        summary_text.append("Regime: ")
        summary_text.append(self._regime, style=regime_style)
        summary_text.append(f"  Avg: {self._avg_correlation:.2f}")
        summary_text.append(f"  Div Ratio: {self._div_ratio:.2f}")
        summary.update(summary_text)

        if not self._correlations:
            content.update("No correlation data")
            return

        table = Table(box=None, expand=True)
        table.add_column("Pair", style="cyan")
        table.add_column("Corr", justify="right")

        # Show top correlations
        sorted_corrs = sorted(
            self._correlations, key=lambda x: abs(x.correlation), reverse=True
        )[:6]

        for corr in sorted_corrs:
            corr_style = "red" if corr.correlation > 0.7 else (
                "yellow" if corr.correlation > 0.5 else "green"
            )
            table.add_row(
                f"{corr.symbol1}/{corr.symbol2}",
                Text(f"{corr.correlation:.2f}", style=corr_style),
            )

        content.update(table)


class ConcentrationPanel(Static):
    """Panel displaying concentration metrics."""

    DEFAULT_CSS = """
    ConcentrationPanel {
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    ConcentrationPanel .panel-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    ConcentrationPanel .concentrated {
        color: $error;
    }

    ConcentrationPanel .diversified {
        color: $success;
    }
    """

    def __init__(
        self,
        hhi: float = 0.0,
        effective_positions: float = 0.0,
        is_concentrated: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._hhi = hhi
        self._effective_positions = effective_positions
        self._is_concentrated = is_concentrated

    def compose(self) -> ComposeResult:
        yield Label("Concentration", classes="panel-title")
        yield Static(id="concentration-content")
        yield ProgressBar(id="hhi-bar", total=100, show_eta=False)

    def on_mount(self) -> None:
        self._update_display()

    def update_data(
        self,
        hhi: float,
        effective_positions: float,
        is_concentrated: bool,
    ) -> None:
        self._hhi = hhi
        self._effective_positions = effective_positions
        self._is_concentrated = is_concentrated
        self._update_display()

    def _update_display(self) -> None:
        content = self.query_one("#concentration-content", Static)
        bar = self.query_one("#hhi-bar", ProgressBar)

        status_style = "red" if self._is_concentrated else "green"
        status_text = "CONCENTRATED" if self._is_concentrated else "Diversified"

        text = Text()
        text.append(f"HHI: {self._hhi:.3f}  ")
        text.append(f"Effective Positions: {self._effective_positions:.1f}  ")
        text.append(status_text, style=status_style)

        content.update(text)
        bar.update(progress=self._hhi * 100)


class MarginUtilizationPanel(Static):
    """Panel displaying margin utilization."""

    DEFAULT_CSS = """
    MarginUtilizationPanel {
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    MarginUtilizationPanel .panel-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    MarginUtilizationPanel .healthy {
        color: $success;
    }

    MarginUtilizationPanel .caution {
        color: $warning-lighten-1;
    }

    MarginUtilizationPanel .warning {
        color: $warning;
    }

    MarginUtilizationPanel .critical {
        color: $error;
        text-style: bold;
    }
    """

    def __init__(
        self,
        total_margin: Decimal = Decimal("0"),
        available_margin: Decimal = Decimal("0"),
        margin_utilization: float = 0.0,
        margin_level: float = 0.0,
        alert_level: str = "Healthy",
        effective_leverage: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._total_margin = total_margin
        self._available_margin = available_margin
        self._utilization = margin_utilization
        self._margin_level = margin_level
        self._alert_level = alert_level
        self._leverage = effective_leverage

    def compose(self) -> ComposeResult:
        yield Label("Margin Utilization", classes="panel-title")
        yield Static(id="margin-summary")
        yield ProgressBar(id="margin-bar", total=100, show_eta=False)
        yield Static(id="margin-details")

    def on_mount(self) -> None:
        self._update_display()

    def update_data(
        self,
        total_margin: Decimal,
        available_margin: Decimal,
        margin_utilization: float,
        margin_level: float,
        alert_level: str,
        effective_leverage: float,
    ) -> None:
        self._total_margin = total_margin
        self._available_margin = available_margin
        self._utilization = margin_utilization
        self._margin_level = margin_level
        self._alert_level = alert_level
        self._leverage = effective_leverage
        self._update_display()

    def _update_display(self) -> None:
        summary = self.query_one("#margin-summary", Static)
        bar = self.query_one("#margin-bar", ProgressBar)
        details = self.query_one("#margin-details", Static)

        # Alert level styling
        alert_style = {
            "Healthy": "green",
            "Caution": "yellow",
            "Warning": "orange1",
            "Critical": "red bold",
            "Liquidation": "red bold blink",
        }.get(self._alert_level, "white")

        summary_text = Text()
        summary_text.append("Status: ")
        summary_text.append(self._alert_level.upper(), style=alert_style)
        summary_text.append(f"  Utilization: {self._utilization:.1%}")
        summary.update(summary_text)

        bar.update(progress=min(self._utilization * 100, 100))

        details_text = Text()
        details_text.append(f"Total: ${self._total_margin:,.0f}  ")
        details_text.append(f"Available: ${self._available_margin:,.0f}  ")
        details_text.append(f"Leverage: {self._leverage:.1f}x")
        details.update(details_text)


class MarginPositionsTable(Static):
    """Table showing margin positions at risk."""

    DEFAULT_CSS = """
    MarginPositionsTable {
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    MarginPositionsTable .panel-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        positions: list[MarginDisplayData] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._positions = positions or []

    def compose(self) -> ComposeResult:
        yield Label("Positions at Risk", classes="panel-title")
        yield Static(id="positions-content")

    def on_mount(self) -> None:
        self._update_display()

    def update_data(self, positions: list[MarginDisplayData]) -> None:
        self._positions = positions
        self._update_display()

    def _update_display(self) -> None:
        content = self.query_one("#positions-content", Static)

        if not self._positions:
            content.update("No positions at risk")
            return

        table = Table(box=None, expand=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Side")
        table.add_column("Lev", justify="right")
        table.add_column("Dist to Liq", justify="right")
        table.add_column("Status")

        # Sort by distance to liquidation
        sorted_positions = sorted(
            self._positions, key=lambda p: p.distance_to_liquidation
        )[:5]

        for pos in sorted_positions:
            side_style = "green" if pos.side == "Long" else "red"

            if pos.distance_to_liquidation < 5:
                dist_style = "red bold"
            elif pos.distance_to_liquidation < 10:
                dist_style = "yellow"
            else:
                dist_style = "green"

            alert_style = {
                "Healthy": "green",
                "Caution": "yellow",
                "Warning": "orange1",
                "Critical": "red bold",
            }.get(pos.alert_level, "white")

            table.add_row(
                pos.symbol,
                Text(pos.side, style=side_style),
                f"{pos.leverage:.1f}x",
                Text(f"{pos.distance_to_liquidation:.1f}%", style=dist_style),
                Text(pos.alert_level, style=alert_style),
            )

        content.update(table)


class RiskAnalyticsDashboard(Container):
    """
    Comprehensive Risk Analytics Dashboard.

    Displays VaR, stress tests, correlations, and margin monitoring
    in a unified dashboard view.
    """

    DEFAULT_CSS = """
    RiskAnalyticsDashboard {
        layout: grid;
        grid-size: 2 3;
        grid-gutter: 1;
        padding: 1;
    }

    RiskAnalyticsDashboard VaRPanel {
        column-span: 1;
    }

    RiskAnalyticsDashboard StressTestPanel {
        column-span: 1;
    }

    RiskAnalyticsDashboard CorrelationPanel {
        column-span: 1;
    }

    RiskAnalyticsDashboard ConcentrationPanel {
        column-span: 1;
    }

    RiskAnalyticsDashboard MarginUtilizationPanel {
        column-span: 1;
    }

    RiskAnalyticsDashboard MarginPositionsTable {
        column-span: 1;
    }
    """

    data = reactive[RiskAnalyticsDashboardData | None](None, init=False)

    def __init__(
        self,
        data: RiskAnalyticsDashboardData | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._initial_data = data
        self._var_panel: VaRPanel | None = None
        self._stress_panel: StressTestPanel | None = None
        self._corr_panel: CorrelationPanel | None = None
        self._conc_panel: ConcentrationPanel | None = None
        self._margin_panel: MarginUtilizationPanel | None = None
        self._positions_table: MarginPositionsTable | None = None

    def compose(self) -> ComposeResult:
        yield VaRPanel(id="var-panel")
        yield StressTestPanel(id="stress-panel")
        yield CorrelationPanel(id="corr-panel")
        yield ConcentrationPanel(id="conc-panel")
        yield MarginUtilizationPanel(id="margin-panel")
        yield MarginPositionsTable(id="positions-table")

    def on_mount(self) -> None:
        """Cache widget references and set initial data."""
        self._var_panel = self.query_one("#var-panel", VaRPanel)
        self._stress_panel = self.query_one("#stress-panel", StressTestPanel)
        self._corr_panel = self.query_one("#corr-panel", CorrelationPanel)
        self._conc_panel = self.query_one("#conc-panel", ConcentrationPanel)
        self._margin_panel = self.query_one("#margin-panel", MarginUtilizationPanel)
        self._positions_table = self.query_one("#positions-table", MarginPositionsTable)

        if self._initial_data:
            self.data = self._initial_data

    def watch_data(self, data: RiskAnalyticsDashboardData | None) -> None:
        """React to data changes."""
        if data is None:
            return
        self._update_panels(data)

    def update_data(self, data: RiskAnalyticsDashboardData) -> None:
        """Update dashboard with new data."""
        self.data = data

    def _update_panels(self, data: RiskAnalyticsDashboardData) -> None:
        """Update all panels with new data."""
        if self._var_panel:
            self._var_panel.update_data(data.var_results, data.portfolio_value)

        if self._stress_panel:
            self._stress_panel.update_data(
                data.stress_results, data.worst_case_pnl, data.expected_loss
            )

        if self._corr_panel:
            self._corr_panel.update_data(
                data.correlations,
                data.avg_correlation,
                data.correlation_regime,
                data.diversification_ratio,
            )

        if self._conc_panel:
            self._conc_panel.update_data(
                data.hhi, data.effective_positions, data.is_concentrated
            )

        if self._margin_panel:
            self._margin_panel.update_data(
                data.total_margin,
                data.available_margin,
                data.margin_utilization,
                data.margin_level,
                data.margin_alert_level,
                data.effective_leverage,
            )

        if self._positions_table:
            self._positions_table.update_data(data.margin_positions)


def create_demo_risk_analytics_data() -> RiskAnalyticsDashboardData:
    """Create demo data for testing."""
    return RiskAnalyticsDashboardData(
        var_results=[
            VaRDisplayData(
                method="Historical",
                confidence_level=0.95,
                time_horizon_days=1,
                var_amount=Decimal("5234"),
                var_pct=5.23,
                cvar_amount=Decimal("7891"),
                cvar_pct=7.89,
            ),
            VaRDisplayData(
                method="Parametric",
                confidence_level=0.95,
                time_horizon_days=1,
                var_amount=Decimal("4987"),
                var_pct=4.99,
                cvar_amount=Decimal("6543"),
                cvar_pct=6.54,
            ),
            VaRDisplayData(
                method="Monte Carlo",
                confidence_level=0.99,
                time_horizon_days=1,
                var_amount=Decimal("8765"),
                var_pct=8.77,
                cvar_amount=Decimal("12345"),
                cvar_pct=12.35,
            ),
        ],
        portfolio_value=Decimal("100000"),
        stress_results=[
            StressTestDisplayData(
                scenario_name="COVID-19 Crash (March 2020)",
                scenario_type="Historical",
                total_pnl=Decimal("-35000"),
                total_pnl_pct=-35.0,
                is_severe=True,
                breaches_limits=True,
            ),
            StressTestDisplayData(
                scenario_name="FTX Collapse (November 2022)",
                scenario_type="Historical",
                total_pnl=Decimal("-28000"),
                total_pnl_pct=-28.0,
                is_severe=True,
                breaches_limits=True,
            ),
            StressTestDisplayData(
                scenario_name="Crypto Winter",
                scenario_type="Hypothetical",
                total_pnl=Decimal("-55000"),
                total_pnl_pct=-55.0,
                is_severe=True,
                breaches_limits=True,
            ),
            StressTestDisplayData(
                scenario_name="Flash Crash",
                scenario_type="Hypothetical",
                total_pnl=Decimal("-15000"),
                total_pnl_pct=-15.0,
                is_severe=False,
                breaches_limits=False,
            ),
        ],
        worst_case_pnl=Decimal("-55000"),
        expected_loss=Decimal("2500"),
        correlations=[
            CorrelationDisplayData("BTC", "ETH", 0.85, "High"),
            CorrelationDisplayData("BTC", "SOL", 0.72, "High"),
            CorrelationDisplayData("ETH", "SOL", 0.68, "Normal"),
            CorrelationDisplayData("BTC", "LINK", 0.55, "Normal"),
        ],
        avg_correlation=0.65,
        correlation_regime="High",
        diversification_ratio=1.45,
        hhi=0.35,
        effective_positions=2.86,
        is_concentrated=True,
        margin_positions=[
            MarginDisplayData(
                symbol="BTCUSDT",
                side="Long",
                leverage=5.0,
                margin_ratio=0.25,
                distance_to_liquidation=18.5,
                alert_level="Healthy",
            ),
            MarginDisplayData(
                symbol="ETHUSDT",
                side="Long",
                leverage=10.0,
                margin_ratio=0.45,
                distance_to_liquidation=8.2,
                alert_level="Warning",
            ),
            MarginDisplayData(
                symbol="SOLUSDT",
                side="Short",
                leverage=3.0,
                margin_ratio=0.15,
                distance_to_liquidation=32.0,
                alert_level="Healthy",
            ),
        ],
        total_margin=Decimal("50000"),
        available_margin=Decimal("25000"),
        margin_utilization=0.50,
        margin_level=2.0,
        margin_alert_level="Healthy",
        effective_leverage=3.5,
    )

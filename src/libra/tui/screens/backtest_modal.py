"""
Backtest Modal Screen.

Modal dialog for configuring and running strategy backtests.

Features:
- Date range selection
- Initial capital configuration
- Symbol selection
- Progress display during backtest
- Results summary with key metrics
- Equity curve sparkline

Design inspired by:
- TradingView Strategy Tester
- QuantConnect backtest interface
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ProgressBar, Select, Sparkline, Static

if TYPE_CHECKING:
    pass


# =============================================================================
# Backtest Configuration
# =============================================================================


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    strategy_id: str
    strategy_name: str
    symbol: str = "BTC/USDT"
    start_date: str = ""  # YYYY-MM-DD
    end_date: str = ""  # YYYY-MM-DD
    initial_capital: Decimal = Decimal("10000")
    commission_pct: float = 0.1  # 0.1%
    slippage_pct: float = 0.05  # 0.05%

    def __post_init__(self) -> None:
        if not self.start_date:
            self.start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    success: bool = False
    error_message: str = ""
    # Performance metrics
    total_return: Decimal = Decimal("0")
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    # Equity curve
    equity_curve: list[float] = field(default_factory=list)
    # Config used
    config: BacktestConfig | None = None


# =============================================================================
# Backtest Config Modal
# =============================================================================


class BacktestConfigModal(ModalScreen[BacktestConfig | None]):
    """
    Modal for configuring backtest parameters.

    Returns BacktestConfig if user clicks Run, None if cancelled.
    """

    DEFAULT_CSS = """
    BacktestConfigModal {
        align: center middle;
    }

    BacktestConfigModal > Container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    BacktestConfigModal .modal-title {
        height: 2;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    BacktestConfigModal .form-row {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }

    BacktestConfigModal .form-label {
        width: 20;
        height: 1;
        content-align: left middle;
        color: $text-muted;
    }

    BacktestConfigModal .form-input {
        width: 1fr;
    }

    BacktestConfigModal .form-hint {
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }

    BacktestConfigModal .modal-actions {
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 1;
    }

    BacktestConfigModal .modal-actions Button {
        margin: 0 1;
        min-width: 14;
    }

    BacktestConfigModal Select {
        width: 1fr;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+enter", "run_backtest", "Run", show=False),
    ]

    # Available symbols for backtest
    SYMBOLS = [
        ("BTC/USDT", "BTC/USDT"),
        ("ETH/USDT", "ETH/USDT"),
        ("SOL/USDT", "SOL/USDT"),
        ("BNB/USDT", "BNB/USDT"),
    ]

    # Preset date ranges
    DATE_PRESETS = [
        ("Last 30 days", 30),
        ("Last 90 days", 90),
        ("Last 180 days", 180),
        ("Last 1 year", 365),
        ("Custom", 0),
    ]

    def __init__(
        self,
        strategy_id: str,
        strategy_name: str,
    ) -> None:
        super().__init__()
        self._strategy_id = strategy_id
        self._strategy_name = strategy_name

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(f"BACKTEST: {self._strategy_name}", classes="modal-title")

            # Symbol selection
            with Horizontal(classes="form-row"):
                yield Label("Symbol:", classes="form-label")
                yield Select(
                    self.SYMBOLS,
                    value="BTC/USDT",
                    id="symbol-select",
                    classes="form-input",
                )

            # Date range preset
            with Horizontal(classes="form-row"):
                yield Label("Date Range:", classes="form-label")
                yield Select(
                    [(label, str(days)) for label, days in self.DATE_PRESETS],
                    value="90",
                    id="date-preset",
                    classes="form-input",
                )

            # Custom start date
            with Horizontal(classes="form-row"):
                yield Label("Start Date:", classes="form-label")
                start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                yield Input(value=start_date, placeholder="YYYY-MM-DD", id="start-date", classes="form-input")

            # Custom end date
            with Horizontal(classes="form-row"):
                yield Label("End Date:", classes="form-label")
                end_date = datetime.now().strftime("%Y-%m-%d")
                yield Input(value=end_date, placeholder="YYYY-MM-DD", id="end-date", classes="form-input")

            # Initial capital
            with Horizontal(classes="form-row"):
                yield Label("Initial Capital:", classes="form-label")
                yield Input(value="10000", placeholder="10000", id="initial-capital", classes="form-input")

            # Commission
            with Horizontal(classes="form-row"):
                yield Label("Commission (%):", classes="form-label")
                yield Input(value="0.1", placeholder="0.1", id="commission", classes="form-input")

            yield Static(
                "[dim]Backtest will simulate strategy execution on historical data[/dim]",
                classes="form-hint",
            )

            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Run Backtest", variant="primary", id="btn-run")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle date preset selection."""
        if event.select.id == "date-preset" and event.value != "0":
            try:
                days = int(event.value)
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")

                self.query_one("#start-date", Input).value = start_date
                self.query_one("#end-date", Input).value = end_date
            except (ValueError, Exception):
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-run":
            self.action_run_backtest()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def action_run_backtest(self) -> None:
        """Gather config and dismiss with it."""
        try:
            symbol = self.query_one("#symbol-select", Select).value
            start_date = self.query_one("#start-date", Input).value
            end_date = self.query_one("#end-date", Input).value
            initial_capital = Decimal(self.query_one("#initial-capital", Input).value)
            commission = float(self.query_one("#commission", Input).value)

            config = BacktestConfig(
                strategy_id=self._strategy_id,
                strategy_name=self._strategy_name,
                symbol=str(symbol) if symbol else "BTC/USDT",
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                commission_pct=commission,
            )
            self.dismiss(config)
        except Exception as e:
            self.notify(f"Invalid configuration: {e}", severity="error")

    def action_cancel(self) -> None:
        """Cancel and dismiss."""
        self.dismiss(None)


# =============================================================================
# Backtest Progress Modal
# =============================================================================


class BacktestProgressModal(ModalScreen[None]):
    """
    Modal showing backtest progress.

    This is a non-dismissible modal that shows while backtest runs.
    """

    DEFAULT_CSS = """
    BacktestProgressModal {
        align: center middle;
    }

    BacktestProgressModal > Container {
        width: 50;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    BacktestProgressModal .modal-title {
        height: 1;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    BacktestProgressModal .progress-status {
        height: 1;
        text-align: center;
        margin-bottom: 1;
    }

    BacktestProgressModal ProgressBar {
        margin: 1 0;
    }

    BacktestProgressModal .progress-detail {
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(self, strategy_name: str) -> None:
        super().__init__()
        self._strategy_name = strategy_name

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("RUNNING BACKTEST", classes="modal-title")
            yield Static(f"Strategy: {self._strategy_name}", classes="progress-status")
            yield ProgressBar(total=100, show_eta=False, id="backtest-progress")
            yield Static("Initializing...", classes="progress-detail", id="progress-detail")

    def update_progress(self, progress: float, detail: str) -> None:
        """Update progress bar and detail text."""
        try:
            bar = self.query_one("#backtest-progress", ProgressBar)
            bar.progress = progress

            detail_label = self.query_one("#progress-detail", Static)
            detail_label.update(detail)
        except Exception:
            pass


# =============================================================================
# Backtest Results Modal
# =============================================================================


class BacktestResultsModal(ModalScreen[bool]):
    """
    Modal displaying backtest results.

    Returns True if user wants to save/apply results, False otherwise.
    """

    DEFAULT_CSS = """
    BacktestResultsModal {
        align: center middle;
    }

    BacktestResultsModal > Container {
        width: 80;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    BacktestResultsModal .modal-title {
        height: 2;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    BacktestResultsModal .results-summary {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }

    BacktestResultsModal .summary-card {
        width: 1fr;
        height: 3;
        padding: 0 1;
        border: round $primary-darken-3;
        content-align: center middle;
    }

    BacktestResultsModal .summary-card.positive {
        border: round $success-darken-1;
    }

    BacktestResultsModal .summary-card.negative {
        border: round $error-darken-1;
    }

    BacktestResultsModal .metrics-grid {
        height: auto;
        grid-size: 3;
        grid-gutter: 1;
        margin-bottom: 1;
    }

    BacktestResultsModal .metric-box {
        height: 3;
        padding: 0 1;
        border: round $primary-darken-3;
    }

    BacktestResultsModal .metric-label {
        color: $text-muted;
    }

    BacktestResultsModal .metric-value {
        text-style: bold;
    }

    BacktestResultsModal .metric-value.good {
        color: $success;
    }

    BacktestResultsModal .metric-value.bad {
        color: $error;
    }

    BacktestResultsModal .metric-value.warning {
        color: $warning;
    }

    BacktestResultsModal .equity-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: round $primary-darken-3;
    }

    BacktestResultsModal .section-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    BacktestResultsModal Sparkline {
        height: 4;
        width: 100%;
    }

    BacktestResultsModal .trade-stats {
        height: auto;
        margin-bottom: 1;
    }

    BacktestResultsModal .modal-actions {
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 1;
    }

    BacktestResultsModal .modal-actions Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "close", "Close", priority=True),
    ]

    def __init__(self, result: BacktestResult) -> None:
        super().__init__()
        self._result = result

    def compose(self) -> ComposeResult:
        r = self._result
        config = r.config

        strategy_name = config.strategy_name if config else "Unknown"
        return_class = "positive" if r.total_return >= 0 else "negative"
        return_sign = "+" if r.total_return >= 0 else ""

        with Container():
            yield Static(f"BACKTEST RESULTS: {strategy_name}", classes="modal-title")

            # Summary row
            with Horizontal(classes="results-summary"):
                yield Static(
                    f"[dim]Return[/dim]\n[bold {return_class}]{return_sign}${r.total_return:,.2f} ({return_sign}{r.total_return_pct:.1f}%)[/]",
                    classes=f"summary-card {return_class}",
                )
                yield Static(
                    f"[dim]Sharpe[/dim]\n[bold]{r.sharpe_ratio:.2f}[/bold]",
                    classes="summary-card",
                )
                yield Static(
                    f"[dim]Max DD[/dim]\n[bold red]{r.max_drawdown_pct:.1f}%[/bold red]",
                    classes="summary-card",
                )

            # Metrics grid
            with Grid(classes="metrics-grid"):
                yield self._metric_box("Win Rate", f"{r.win_rate:.1f}%", self._win_rate_class(r.win_rate))
                yield self._metric_box("Profit Factor", f"{r.profit_factor:.2f}", self._pf_class(r.profit_factor))
                yield self._metric_box("Total Trades", str(r.total_trades), "")
                yield self._metric_box("Winning", str(r.winning_trades), "good")
                yield self._metric_box("Losing", str(r.losing_trades), "bad" if r.losing_trades > r.winning_trades else "")
                yield self._metric_box("Avg Win", f"${r.avg_win:,.2f}", "good")
                yield self._metric_box("Avg Loss", f"${r.avg_loss:,.2f}", "bad")

                # Config info
                if config:
                    yield self._metric_box("Period", f"{config.start_date} to {config.end_date}", "")
                    yield self._metric_box("Symbol", config.symbol, "")

            # Equity curve
            if r.equity_curve:
                with Vertical(classes="equity-section"):
                    yield Static("EQUITY CURVE", classes="section-title")
                    yield Sparkline(r.equity_curve, summary_function=max, id="equity-sparkline")

            # Actions
            with Horizontal(classes="modal-actions"):
                yield Button("Close", variant="default", id="btn-close")
                yield Button("Export Results", variant="primary", id="btn-export")

    def _metric_box(self, label: str, value: str, value_class: str) -> Static:
        """Create a metric display box."""
        class_str = f"metric-value {value_class}" if value_class else "metric-value"
        return Static(
            f"[dim]{label}[/dim]\n[{class_str}]{value}[/]",
            classes="metric-box",
        )

    def _win_rate_class(self, win_rate: float) -> str:
        """Get CSS class for win rate."""
        if win_rate >= 55:
            return "good"
        elif win_rate >= 45:
            return "warning"
        return "bad"

    def _pf_class(self, pf: float) -> str:
        """Get CSS class for profit factor."""
        if pf >= 1.5:
            return "good"
        elif pf >= 1.0:
            return "warning"
        return "bad"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-close":
            self.dismiss(False)
        elif event.button.id == "btn-export":
            self.notify("Results exported to backtest_results.json")
            self.dismiss(True)

    def action_close(self) -> None:
        """Close the modal."""
        self.dismiss(False)


# =============================================================================
# Demo Backtest Runner
# =============================================================================


def run_demo_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run a simulated backtest for demo purposes.

    In a real implementation, this would call the backtesting engine.
    """
    import random

    # Simulate backtest metrics
    initial = float(config.initial_capital)
    total_return_pct = random.uniform(-20, 50)
    total_return = Decimal(str(initial * total_return_pct / 100))

    total_trades = random.randint(20, 100)
    win_rate = random.uniform(40, 70)
    winning_trades = int(total_trades * win_rate / 100)
    losing_trades = total_trades - winning_trades

    # Generate equity curve
    equity = [initial]
    for _ in range(50):
        change = random.uniform(-0.02, 0.025) * equity[-1]
        equity.append(equity[-1] + change)
    # Ensure final equity matches return
    equity[-1] = initial * (1 + total_return_pct / 100)

    return BacktestResult(
        success=True,
        total_return=total_return,
        total_return_pct=total_return_pct,
        sharpe_ratio=random.uniform(0.5, 2.5),
        max_drawdown_pct=random.uniform(5, 25),
        win_rate=win_rate,
        profit_factor=random.uniform(0.8, 2.0),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        avg_win=Decimal(str(random.uniform(50, 200))),
        avg_loss=Decimal(str(random.uniform(30, 150))),
        equity_curve=equity,
        config=config,
    )

"""
Risk Dashboard Widget.

Real-time risk metrics display inspired by:
- btop++ (color-coded panels, keyboard hints)
- htop (gauge bars, real-time updates)
- cointop (sparkline charts)

Features:
- Trading state indicator (ACTIVE/REDUCING/HALTED)
- Drawdown gauge with historical sparkline
- Daily P&L progress bar
- Position exposure bars per symbol
- Order rate indicator
- Circuit breaker status

Layout:
    +-- RISK STATUS -----------------------------+
    | Trading State: [green]ACTIVE[/green]       |
    |                                            |
    | Drawdown:  [####------] 25% / 50% max      |
    | History:   [sparkline]                     |
    |                                            |
    | Daily P&L: [########--] -$2,000 / $10,000  |
    |                                            |
    | Positions:                                 |
    |   BTC/USDT [######----] 60% / 100%        |
    |   ETH/USDT [###-------] 30% / 75%         |
    |                                            |
    | Order Rate: [ooooo-----] 5/10 per sec     |
    |                                            |
    | Circuit Breaker: [green]CLOSED[/green]     |
    +--------------------------------------------+
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Label, ProgressBar, Sparkline, Static


if TYPE_CHECKING:
    from libra.risk.manager import RiskManager


# =============================================================================
# Trading State Indicator
# =============================================================================


class TradingStateIndicator(Static):
    """
    Displays current trading state with color-coded status.

    States:
        ACTIVE   - Green, normal trading
        REDUCING - Yellow, only closing positions
        HALTED   - Red, no trading allowed
    """

    DEFAULT_CSS = """
    TradingStateIndicator {
        height: 1;
        width: 100%;
    }

    TradingStateIndicator .state-active {
        color: $success;
        text-style: bold;
    }

    TradingStateIndicator .state-reducing {
        color: $warning;
        text-style: bold;
    }

    TradingStateIndicator .state-halted {
        color: $error;
        text-style: bold;
    }
    """

    state: reactive[str] = reactive("ACTIVE")

    def __init__(self, id: str = "trading-state") -> None:
        super().__init__(id=id)

    def render(self) -> str:
        """Render the trading state."""
        icons = {
            "ACTIVE": "[green]ACTIVE[/green]",
            "REDUCING": "[yellow]REDUCING[/yellow]",
            "HALTED": "[red]HALTED[/red]",
        }
        icon = icons.get(self.state, "[dim]UNKNOWN[/dim]")
        return f"Trading State: {icon}"

    def set_state(self, state: str) -> None:
        """Update the trading state."""
        self.state = state.upper()


# =============================================================================
# Drawdown Gauge
# =============================================================================


class DrawdownGauge(Container):
    """
    Drawdown visualization with progress bar and sparkline history.

    Shows current drawdown as percentage of maximum allowed,
    plus historical trend via sparkline.
    """

    DEFAULT_CSS = """
    DrawdownGauge {
        height: auto;
        width: 100%;
        padding: 0;
    }

    DrawdownGauge .gauge-label {
        height: 1;
        color: $text-muted;
    }

    DrawdownGauge .gauge-row {
        height: 1;
        width: 100%;
    }

    DrawdownGauge ProgressBar {
        width: 100%;
        padding: 0;
    }

    DrawdownGauge .gauge-value {
        color: $text;
        text-align: right;
    }

    DrawdownGauge Sparkline {
        height: 1;
        width: 100%;
        margin-top: 0;
    }
    """

    current: reactive[float] = reactive(0.0)
    maximum: reactive[float] = reactive(50.0)  # 50% max drawdown

    def __init__(
        self,
        current: float = 0.0,
        maximum: float = 50.0,
        id: str = "drawdown-gauge",
    ) -> None:
        super().__init__(id=id)
        self.current = current
        self.maximum = maximum
        self._history: list[float] = [0.0] * 20  # Last 20 data points

    def compose(self) -> ComposeResult:
        yield Static("Drawdown", classes="gauge-label")
        yield ProgressBar(total=100, show_eta=False, show_percentage=True, id="drawdown-bar")
        yield Static("", id="drawdown-value", classes="gauge-value")
        yield Sparkline(self._history, summary_function=max, id="drawdown-sparkline")

    def on_mount(self) -> None:
        """Initialize gauge on mount."""
        self._update_display()

    def watch_current(self, value: float) -> None:
        """React to current value changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the gauge display."""
        try:
            # Update progress bar (percentage of max)
            pct = min(abs(self.current) / self.maximum * 100, 100) if self.maximum > 0 else 0
            bar = self.query_one("#drawdown-bar", ProgressBar)
            bar.update(progress=pct)

            # Update value label
            value_label = self.query_one("#drawdown-value", Static)
            color = "green" if self.current < self.maximum * 0.5 else "yellow" if self.current < self.maximum * 0.8 else "red"
            value_label.update(f"[{color}]{self.current:.1f}%[/{color}] / {self.maximum:.0f}% max")
        except Exception:
            pass

    def update_value(self, current: float, maximum: float | None = None) -> None:
        """Update the drawdown value."""
        self.current = current
        if maximum is not None:
            self.maximum = maximum

        # Add to history
        self._history.append(current)
        if len(self._history) > 20:
            self._history.pop(0)

        try:
            sparkline = self.query_one("#drawdown-sparkline", Sparkline)
            sparkline.data = self._history
        except Exception:
            pass


# =============================================================================
# Exposure Bar
# =============================================================================


class ExposureBar(Container):
    """
    Position exposure bar for a single symbol.

    Shows current position size as percentage of limit.
    """

    DEFAULT_CSS = """
    ExposureBar {
        height: 2;
        width: 100%;
    }

    ExposureBar .exposure-label {
        height: 1;
        width: 12;
    }

    ExposureBar ProgressBar {
        width: 1fr;
    }

    ExposureBar .exposure-value {
        width: 20;
        text-align: right;
    }
    """

    def __init__(
        self,
        symbol: str,
        current: float = 0.0,
        limit: float = 100.0,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id or f"exposure-{symbol.replace('/', '-')}")
        self._symbol = symbol
        self._symbol_id = symbol.replace("/", "-")  # Sanitized for use in IDs
        self._current = current
        self._limit = limit

    def compose(self) -> ComposeResult:
        yield Static(f"  {self._symbol}", classes="exposure-label")
        yield ProgressBar(total=100, show_eta=False, show_percentage=False, id=f"bar-{self._symbol_id}")
        yield Static("", id=f"value-{self._symbol_id}", classes="exposure-value")

    def on_mount(self) -> None:
        """Initialize on mount."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the exposure display."""
        try:
            pct = min(self._current / self._limit * 100, 100) if self._limit > 0 else 0
            bar = self.query_one(f"#bar-{self._symbol_id}", ProgressBar)
            bar.update(progress=pct)

            value = self.query_one(f"#value-{self._symbol_id}", Static)
            color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
            value.update(f"[{color}]{self._current:.0f}%[/{color}] / {self._limit:.0f}%")
        except Exception:
            pass

    def update_value(self, current: float, limit: float | None = None) -> None:
        """Update the exposure values."""
        self._current = current
        if limit is not None:
            self._limit = limit
        self._update_display()


# =============================================================================
# Order Rate Indicator
# =============================================================================


class OrderRateIndicator(Static):
    """
    Order rate indicator showing current rate vs limit.

    Uses dot visualization: filled dots = used, empty dots = available
    """

    DEFAULT_CSS = """
    OrderRateIndicator {
        height: 1;
        width: 100%;
    }
    """

    current: reactive[int] = reactive(0)
    limit: reactive[int] = reactive(10)

    def __init__(self, current: int = 0, limit: int = 10, id: str = "order-rate") -> None:
        super().__init__(id=id)
        self.current = current
        self.limit = limit

    def render(self) -> str:
        """Render the order rate indicator."""
        filled = min(self.current, self.limit)
        empty = self.limit - filled

        dots = "[green]" + "" * filled + "[/green]" + "[dim]" + "" * empty + "[/dim]"
        return f"Order Rate: {dots} {self.current}/{self.limit} per sec"

    def update_rate(self, current: int, limit: int | None = None) -> None:
        """Update the rate values."""
        self.current = current
        if limit is not None:
            self.limit = limit


# =============================================================================
# Circuit Breaker Indicator
# =============================================================================


class CircuitBreakerIndicator(Static):
    """
    Circuit breaker status indicator.

    States:
        CLOSED    - Green, normal operation
        OPEN      - Red, blocking all orders
        HALF_OPEN - Yellow, testing with limited orders
    """

    DEFAULT_CSS = """
    CircuitBreakerIndicator {
        height: 1;
        width: 100%;
    }
    """

    state: reactive[str] = reactive("CLOSED")
    reason: reactive[str] = reactive("")

    def __init__(self, id: str = "circuit-breaker") -> None:
        super().__init__(id=id)

    def render(self) -> str:
        """Render the circuit breaker status."""
        icons = {
            "CLOSED": "[green]CLOSED[/green]",
            "OPEN": "[red]OPEN[/red]",
            "HALF_OPEN": "[yellow]HALF_OPEN[/yellow]",
        }
        icon = icons.get(self.state, "[dim]UNKNOWN[/dim]")

        if self.reason and self.state != "CLOSED":
            return f"Circuit Breaker: {icon} ({self.reason})"
        return f"Circuit Breaker: {icon}"

    def update_status(self, state: str, reason: str = "") -> None:
        """Update the circuit breaker status."""
        self.state = state.upper()
        self.reason = reason


# =============================================================================
# Risk Dashboard (Main Widget)
# =============================================================================


class RiskDashboard(Container):
    """
    Real-time risk metrics dashboard.

    Combines all risk indicators into a single panel.

    Features:
        - Trading state indicator
        - Drawdown gauge with sparkline
        - Daily P&L progress
        - Position exposure bars
        - Order rate indicator
        - Circuit breaker status

    Usage:
        dashboard = RiskDashboard()
        dashboard.update_from_risk_manager(risk_manager)

        # Or update individual metrics
        dashboard.set_trading_state("ACTIVE")
        dashboard.set_drawdown(2.5, 10.0)
        dashboard.set_daily_pnl(-2000, 10000)
    """

    DEFAULT_CSS = """
    RiskDashboard {
        height: auto;
        width: 100%;
        background: $surface;
        border: round $primary-darken-1;
        padding: 1;
    }

    RiskDashboard .dashboard-title {
        text-style: bold;
        color: $text;
        background: $primary-darken-2;
        text-align: center;
        padding: 0 1;
        margin-bottom: 1;
    }

    RiskDashboard .section-title {
        color: $text-muted;
        margin-top: 1;
        margin-bottom: 0;
    }

    RiskDashboard .divider {
        color: $primary-darken-2;
        height: 1;
    }
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        id: str = "risk-dashboard",
    ) -> None:
        """
        Initialize the risk dashboard.

        Args:
            symbols: List of symbols to track for exposure bars
        """
        super().__init__(id=id)
        self._symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def compose(self) -> ComposeResult:
        yield Static("RISK STATUS", classes="dashboard-title")

        # Trading state
        yield TradingStateIndicator()

        yield Static("", classes="divider")

        # Drawdown gauge
        yield DrawdownGauge()

        yield Static("", classes="divider")

        # Daily P&L
        yield Static("Daily P&L", classes="section-title")
        yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="daily-pnl-bar")
        yield Static("", id="daily-pnl-value")

        yield Static("", classes="divider")

        # Position exposure
        yield Static("Position Exposure", classes="section-title")
        for symbol in self._symbols:
            yield ExposureBar(symbol=symbol)

        yield Static("", classes="divider")

        # Order rate
        yield OrderRateIndicator()

        yield Static("", classes="divider")

        # Circuit breaker
        yield CircuitBreakerIndicator()

    # =========================================================================
    # Update Methods
    # =========================================================================

    def set_trading_state(self, state: str) -> None:
        """Update the trading state display."""
        try:
            self.query_one(TradingStateIndicator).set_state(state)
        except Exception:
            pass

    def set_drawdown(self, current: float, maximum: float | None = None) -> None:
        """Update the drawdown display."""
        try:
            gauge = self.query_one(DrawdownGauge)
            gauge.update_value(current, maximum)
        except Exception:
            pass

    def set_daily_pnl(self, current: float, limit: float) -> None:
        """Update the daily P&L display."""
        try:
            # Calculate percentage of limit used
            pct = min(abs(current) / limit * 100, 100) if limit > 0 else 0

            bar = self.query_one("#daily-pnl-bar", ProgressBar)
            bar.update(progress=pct)

            value = self.query_one("#daily-pnl-value", Static)
            color = "green" if current >= 0 else "yellow" if abs(current) < limit * 0.5 else "red"
            sign = "+" if current >= 0 else ""
            value.update(f"[{color}]{sign}${current:,.0f}[/{color}] / ${limit:,.0f} limit")
        except Exception:
            pass

    def set_exposure(self, symbol: str, current: float, limit: float | None = None) -> None:
        """Update a symbol's exposure display."""
        try:
            bar_id = f"exposure-{symbol.replace('/', '-')}"
            exposure_bar = self.query_one(f"#{bar_id}", ExposureBar)
            exposure_bar.update_value(current, limit)
        except Exception:
            pass

    def set_order_rate(self, current: int, limit: int | None = None) -> None:
        """Update the order rate display."""
        try:
            self.query_one(OrderRateIndicator).update_rate(current, limit)
        except Exception:
            pass

    def set_circuit_breaker(self, state: str, reason: str = "") -> None:
        """Update the circuit breaker display."""
        try:
            self.query_one(CircuitBreakerIndicator).update_status(state, reason)
        except Exception:
            pass

    def update_from_risk_manager(self, risk_manager: RiskManager) -> None:
        """
        Update all displays from a RiskManager instance.

        Args:
            risk_manager: The RiskManager to read state from
        """
        try:
            # Trading state
            self.set_trading_state(risk_manager.trading_state.value)

            # Drawdown
            drawdown_pct = float(risk_manager.current_drawdown) * 100
            max_drawdown = float(risk_manager.limits.max_total_drawdown_pct) * 100
            self.set_drawdown(abs(drawdown_pct), abs(max_drawdown))

            # Circuit breaker
            cb_status = risk_manager.circuit_breaker_status
            self.set_circuit_breaker(
                cb_status.get("state", "CLOSED"),
                cb_status.get("trip_reason", ""),
            )

            # Stats
            stats = risk_manager.get_stats()

            # Daily P&L (convert from percentage to dollar amount if needed)
            daily_pnl_pct = float(stats.get("daily_pnl", "0%").rstrip("%"))
            self.set_daily_pnl(daily_pnl_pct * 100, 10000)  # Simulated dollar conversion

        except Exception:
            pass

    def update_from_dict(self, data: dict[str, Any]) -> None:
        """
        Update displays from a dictionary of values.

        Args:
            data: Dictionary with keys like 'trading_state', 'drawdown', etc.
        """
        if "trading_state" in data:
            self.set_trading_state(data["trading_state"])

        if "drawdown" in data:
            dd = data["drawdown"]
            self.set_drawdown(dd.get("current", 0), dd.get("maximum", 50))

        if "daily_pnl" in data:
            pnl = data["daily_pnl"]
            self.set_daily_pnl(pnl.get("current", 0), pnl.get("limit", 10000))

        if "exposures" in data:
            for symbol, exposure in data["exposures"].items():
                self.set_exposure(symbol, exposure.get("current", 0), exposure.get("limit", 100))

        if "order_rate" in data:
            rate = data["order_rate"]
            self.set_order_rate(rate.get("current", 0), rate.get("limit", 10))

        if "circuit_breaker" in data:
            cb = data["circuit_breaker"]
            self.set_circuit_breaker(cb.get("state", "CLOSED"), cb.get("reason", ""))

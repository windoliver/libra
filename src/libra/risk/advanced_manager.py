"""
Advanced Risk Manager.

Integrates all advanced risk management components:
- VaR Calculator
- Stress Testing Engine
- Correlation Monitor
- Margin Monitor

Provides a unified interface for comprehensive portfolio risk assessment.

Issue #15: Advanced Risk Management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np

from libra.risk.correlation import (
    ConcentrationMetrics,
    CorrelationMatrix,
    CorrelationMonitor,
    CorrelationRegime,
    DiversificationMetrics,
    create_correlation_monitor,
)
from libra.risk.margin import (
    MarginAlert,
    MarginConfig,
    MarginMonitor,
    MarginPosition,
    PortfolioMargin,
    create_margin_monitor,
)
from libra.risk.stress_testing import (
    StressScenario,
    StressTestEngine,
    StressTestResult,
    create_stress_test_engine,
)
from libra.risk.var import (
    PositionVaR,
    VaRCalculator,
    VaRConfig,
    VaRMethod,
    VaRResult,
    create_var_calculator,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AdvancedRiskConfig:
    """Configuration for advanced risk management."""

    # VaR settings
    var_confidence_level: float = 0.95
    var_time_horizon_days: int = 1
    var_lookback_days: int = 252
    var_method: VaRMethod = VaRMethod.HISTORICAL

    # Stress testing settings
    stress_loss_limit_pct: float = 20.0

    # Correlation settings
    correlation_window_size: int = 60

    # Margin settings
    margin_max_leverage: Decimal = Decimal("10")
    margin_warning_threshold: float = 1.25

    # Update frequency (seconds)
    update_interval: float = 60.0


@dataclass
class AdvancedRiskReport:
    """Comprehensive risk report from advanced risk manager."""

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # VaR metrics
    var_result: VaRResult | None = None
    position_vars: list[PositionVaR] = field(default_factory=list)

    # Stress testing
    stress_results: list[StressTestResult] = field(default_factory=list)
    worst_case_scenario: StressTestResult | None = None
    expected_loss: Decimal = Decimal("0")

    # Correlation
    correlation_matrix: CorrelationMatrix | None = None
    correlation_regime: CorrelationRegime = CorrelationRegime.NORMAL
    concentration: ConcentrationMetrics | None = None
    diversification: DiversificationMetrics | None = None

    # Margin
    portfolio_margin: PortfolioMargin | None = None
    margin_alerts: list[MarginAlert] = field(default_factory=list)

    # Summary metrics
    portfolio_value: Decimal = Decimal("0")
    var_pct: float = 0.0
    worst_case_pct: float = 0.0
    effective_leverage: float = 0.0
    is_healthy: bool = True
    risk_warnings: list[str] = field(default_factory=list)


class AdvancedRiskManager:
    """
    Advanced Risk Manager.

    Coordinates all advanced risk management components to provide
    comprehensive portfolio risk assessment.

    Example:
        manager = AdvancedRiskManager(config=AdvancedRiskConfig())

        # Update with portfolio data
        report = manager.update(
            position_returns=returns_dict,
            position_values=values_dict,
            margin_positions=margin_list,
        )

        # Check risk status
        if not report.is_healthy:
            for warning in report.risk_warnings:
                print(f"Warning: {warning}")

        # Get VaR
        print(f"95% VaR: ${report.var_result.var:,.2f}")
    """

    def __init__(
        self,
        config: AdvancedRiskConfig | None = None,
        account_equity: Decimal | None = None,
    ) -> None:
        """
        Initialize advanced risk manager.

        Args:
            config: Configuration settings
            account_equity: Initial account equity for margin calculations
        """
        self.config = config or AdvancedRiskConfig()
        self._account_equity = account_equity or Decimal("0")

        # Initialize components
        self._var_calculator = create_var_calculator(
            confidence_level=self.config.var_confidence_level,
            time_horizon_days=self.config.var_time_horizon_days,
            lookback_days=self.config.var_lookback_days,
        )

        self._stress_engine = create_stress_test_engine(
            loss_limit_pct=self.config.stress_loss_limit_pct,
        )

        self._correlation_monitor = create_correlation_monitor(
            window_size=self.config.correlation_window_size,
        )

        self._margin_monitor = create_margin_monitor(
            account_equity=self._account_equity,
            max_leverage=self.config.margin_max_leverage,
            warning_threshold=self.config.margin_warning_threshold,
        )

        # State
        self._last_report: AdvancedRiskReport | None = None
        self._returns_history: dict[str, list[float]] = {}

    def update_account_equity(self, equity: Decimal) -> None:
        """Update account equity."""
        self._account_equity = equity
        self._margin_monitor.update_account_equity(equity)

    def add_return_observation(
        self,
        symbol: str,
        return_value: float,
    ) -> None:
        """
        Add a single return observation.

        Args:
            symbol: Asset symbol
            return_value: Period return (e.g., daily return)
        """
        if symbol not in self._returns_history:
            self._returns_history[symbol] = []
        self._returns_history[symbol].append(return_value)

        # Trim to lookback window
        max_len = self.config.var_lookback_days
        if len(self._returns_history[symbol]) > max_len:
            self._returns_history[symbol] = self._returns_history[symbol][-max_len:]

    def update(
        self,
        position_returns: dict[str, np.ndarray] | None = None,
        position_values: dict[str, Decimal] | None = None,
        margin_positions: list[MarginPosition] | None = None,
    ) -> AdvancedRiskReport:
        """
        Perform comprehensive risk update.

        Args:
            position_returns: Historical returns per position
            position_values: Current position values
            margin_positions: Margin position details

        Returns:
            AdvancedRiskReport with all risk metrics
        """
        report = AdvancedRiskReport()

        # Use provided returns or history
        if position_returns is None:
            position_returns = {
                s: np.array(r) for s, r in self._returns_history.items()
                if len(r) >= 30  # Minimum observations
            }

        if position_values is None:
            position_values = {}

        # Calculate portfolio value
        report.portfolio_value = sum(position_values.values(), Decimal("0"))

        # 1. VaR Calculation
        if position_returns and position_values:
            try:
                portfolio_var, position_vars = self._var_calculator.calculate_portfolio_var(
                    position_returns,
                    position_values,
                    method=self.config.var_method,
                )
                report.var_result = portfolio_var
                report.position_vars = position_vars
                report.var_pct = portfolio_var.var_pct
            except Exception as e:
                logger.warning("VaR calculation failed: %s", e)
                report.risk_warnings.append(f"VaR calculation failed: {e}")

        # 2. Stress Testing
        if position_values:
            try:
                report.stress_results = self._stress_engine.run_all_scenarios(
                    position_values
                )
                if report.stress_results:
                    report.worst_case_scenario = report.stress_results[0]
                    report.worst_case_pct = report.worst_case_scenario.total_pnl_pct

                report.expected_loss = self._stress_engine.expected_loss(
                    position_values
                )
            except Exception as e:
                logger.warning("Stress testing failed: %s", e)
                report.risk_warnings.append(f"Stress testing failed: {e}")

        # 3. Correlation Analysis
        if position_returns and len(position_returns) >= 2:
            try:
                # Update correlation monitor
                self._correlation_monitor.update(position_returns)

                report.correlation_matrix = self._correlation_monitor.current_matrix
                report.correlation_regime = self._correlation_monitor.regime

                # Calculate concentration
                if position_values:
                    report.concentration = self._correlation_monitor.calculate_concentration(
                        position_values
                    )

                    # Calculate diversification
                    report.diversification = self._correlation_monitor.calculate_diversification(
                        position_returns, position_values
                    )
            except Exception as e:
                logger.warning("Correlation analysis failed: %s", e)
                report.risk_warnings.append(f"Correlation analysis failed: {e}")

        # 4. Margin Monitoring
        if margin_positions:
            try:
                for pos in margin_positions:
                    self._margin_monitor.update_position(pos)

                report.portfolio_margin = self._margin_monitor.get_portfolio_margin()
                report.margin_alerts = self._margin_monitor.check_portfolio_alerts()
                report.effective_leverage = report.portfolio_margin.effective_leverage
            except Exception as e:
                logger.warning("Margin monitoring failed: %s", e)
                report.risk_warnings.append(f"Margin monitoring failed: {e}")

        # 5. Health Assessment
        report.is_healthy = self._assess_health(report)

        self._last_report = report
        return report

    def _assess_health(self, report: AdvancedRiskReport) -> bool:
        """
        Assess overall portfolio health.

        Returns:
            True if healthy, False if there are concerns
        """
        is_healthy = True

        # Check VaR
        if report.var_pct > 10:  # VaR > 10% of portfolio
            report.risk_warnings.append(
                f"High VaR: {report.var_pct:.1f}% of portfolio at risk"
            )
            is_healthy = False

        # Check worst-case stress
        if report.worst_case_pct < -30:  # Worst case > 30% loss
            report.risk_warnings.append(
                f"Severe stress exposure: {abs(report.worst_case_pct):.1f}% potential loss"
            )
            is_healthy = False

        # Check correlation regime
        if report.correlation_regime == CorrelationRegime.CRISIS:
            report.risk_warnings.append(
                "Crisis correlation regime: diversification benefits reduced"
            )
            is_healthy = False

        # Check concentration
        if report.concentration and report.concentration.is_concentrated:
            report.risk_warnings.append(
                f"Portfolio concentrated: HHI={report.concentration.hhi:.3f}"
            )
            # Concentration is a warning but not unhealthy by itself

        # Check margin
        if report.portfolio_margin:
            if report.portfolio_margin.margin_level < 1.25:
                report.risk_warnings.append(
                    f"Low margin level: {report.portfolio_margin.margin_level:.2f}"
                )
                is_healthy = False

            if report.effective_leverage > float(self.config.margin_max_leverage):
                report.risk_warnings.append(
                    f"Excessive leverage: {report.effective_leverage:.1f}x"
                )
                is_healthy = False

        # Check for critical margin alerts
        critical_alerts = [
            a for a in report.margin_alerts
            if a.severity.value in ("critical", "liquidation")
        ]
        if critical_alerts:
            for alert in critical_alerts:
                report.risk_warnings.append(f"CRITICAL: {alert.description}")
            is_healthy = False

        return is_healthy

    def get_var(
        self,
        method: VaRMethod | None = None,
        confidence_level: float | None = None,
    ) -> VaRResult | None:
        """
        Get current VaR estimate.

        Args:
            method: Override VaR method
            confidence_level: Override confidence level

        Returns:
            VaRResult or None if not available
        """
        if not self._returns_history:
            return None

        # Build return arrays
        position_returns = {
            s: np.array(r) for s, r in self._returns_history.items()
            if len(r) >= 30
        }

        if not position_returns:
            return None

        # Need position values - use last report or default
        if self._last_report and self._last_report.position_vars:
            position_values = {
                pv.symbol: pv.position_value
                for pv in self._last_report.position_vars
            }
        else:
            # Default to equal weights
            position_values = {
                s: Decimal("10000") for s in position_returns
            }

        method = method or self.config.var_method
        conf = confidence_level or self.config.var_confidence_level

        portfolio_var, _ = self._var_calculator.calculate_portfolio_var(
            position_returns, position_values, method, conf
        )
        return portfolio_var

    def run_stress_scenario(
        self,
        scenario: StressScenario,
        positions: dict[str, Decimal] | None = None,
    ) -> StressTestResult:
        """
        Run a specific stress scenario.

        Args:
            scenario: Stress scenario to run
            positions: Positions to stress (uses last if None)

        Returns:
            StressTestResult
        """
        if positions is None and self._last_report:
            positions = {
                pv.symbol: pv.position_value
                for pv in self._last_report.position_vars
            }

        if not positions:
            raise ValueError("No positions provided")

        return self._stress_engine.run_scenario(scenario, positions)

    def get_margin_status(self) -> dict[str, Any]:
        """Get current margin status."""
        return self._margin_monitor.get_margin_summary()

    def get_correlation_state(self) -> dict[str, Any]:
        """Get current correlation state."""
        state = self._correlation_monitor.get_state()
        return {
            "regime": state.regime.value,
            "average_correlation": state.average_correlation,
            "matrix": state.current_matrix.matrix.tolist() if state.current_matrix else None,
            "symbols": state.current_matrix.symbols if state.current_matrix else [],
            "recent_alerts": [
                {
                    "type": a.alert_type,
                    "description": a.description,
                    "severity": a.severity,
                }
                for a in state.recent_alerts
            ],
        }

    def get_last_report(self) -> AdvancedRiskReport | None:
        """Get last risk report."""
        return self._last_report

    @property
    def var_calculator(self) -> VaRCalculator:
        """Access VaR calculator directly."""
        return self._var_calculator

    @property
    def stress_engine(self) -> StressTestEngine:
        """Access stress test engine directly."""
        return self._stress_engine

    @property
    def correlation_monitor(self) -> CorrelationMonitor:
        """Access correlation monitor directly."""
        return self._correlation_monitor

    @property
    def margin_monitor(self) -> MarginMonitor:
        """Access margin monitor directly."""
        return self._margin_monitor


def create_advanced_risk_manager(
    var_confidence: float = 0.95,
    stress_limit_pct: float = 20.0,
    max_leverage: Decimal = Decimal("10"),
    account_equity: Decimal = Decimal("0"),
) -> AdvancedRiskManager:
    """
    Factory function to create advanced risk manager.

    Args:
        var_confidence: VaR confidence level
        stress_limit_pct: Stress test loss limit percentage
        max_leverage: Maximum allowed leverage
        account_equity: Initial account equity

    Returns:
        Configured AdvancedRiskManager
    """
    config = AdvancedRiskConfig(
        var_confidence_level=var_confidence,
        stress_loss_limit_pct=stress_limit_pct,
        margin_max_leverage=max_leverage,
    )
    return AdvancedRiskManager(config=config, account_equity=account_equity)

"""
Unit tests for Advanced Risk Management (Issue #15).

Tests for VaR Calculator, Stress Testing, Correlation Monitor, and Margin Monitor.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from libra.risk.var import (
    VaRCalculator,
    VaRConfig,
    VaRMethod,
    VaRResult,
    create_var_calculator,
)
from libra.risk.stress_testing import (
    AssetClass,
    MarketShock,
    ScenarioType,
    StressScenario,
    StressTestEngine,
    create_stress_test_engine,
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS,
)
from libra.risk.correlation import (
    CorrelationMatrix,
    CorrelationMethod,
    CorrelationMonitor,
    CorrelationRegime,
    create_correlation_monitor,
)
from libra.risk.margin import (
    MarginAlertLevel,
    MarginConfig,
    MarginMode,
    MarginMonitor,
    MarginPosition,
    PositionSide,
    create_margin_monitor,
)


# ============================================================================
# VaR Calculator Tests
# ============================================================================


class TestVaRCalculator:
    """Tests for VaR Calculator."""

    @pytest.fixture
    def returns(self) -> np.ndarray:
        """Generate synthetic return data."""
        np.random.seed(42)
        return np.random.normal(0, 0.02, 252)  # 1 year of daily returns

    @pytest.fixture
    def calculator(self) -> VaRCalculator:
        """Create VaR calculator."""
        return create_var_calculator(confidence_level=0.95)

    def test_historical_var_calculation(
        self, calculator: VaRCalculator, returns: np.ndarray
    ) -> None:
        """Test historical VaR calculation."""
        portfolio_value = Decimal("100000")
        result = calculator.calculate_historical_var(returns, portfolio_value)

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.HISTORICAL
        assert result.confidence_level == 0.95
        assert result.var > 0
        assert result.cvar >= result.var  # CVaR should be >= VaR
        assert result.var_pct > 0
        assert result.cvar_pct >= result.var_pct

    def test_parametric_var_calculation(
        self, calculator: VaRCalculator, returns: np.ndarray
    ) -> None:
        """Test parametric VaR calculation."""
        portfolio_value = Decimal("100000")
        result = calculator.calculate_parametric_var(returns, portfolio_value)

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.PARAMETRIC
        assert result.var > 0
        assert result.cvar >= result.var

    def test_monte_carlo_var_calculation(
        self, calculator: VaRCalculator, returns: np.ndarray
    ) -> None:
        """Test Monte Carlo VaR calculation."""
        portfolio_value = Decimal("100000")
        result = calculator.calculate_monte_carlo_var(
            returns, portfolio_value, num_simulations=1000
        )

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.MONTE_CARLO
        assert result.num_simulations == 1000
        assert result.var > 0

    def test_var_with_different_confidence_levels(
        self, calculator: VaRCalculator, returns: np.ndarray
    ) -> None:
        """Test that higher confidence levels produce higher VaR."""
        portfolio_value = Decimal("100000")

        var_95 = calculator.calculate_historical_var(
            returns, portfolio_value, confidence_level=0.95
        )
        var_99 = calculator.calculate_historical_var(
            returns, portfolio_value, confidence_level=0.99
        )

        assert var_99.var > var_95.var

    def test_var_time_scaling(self, returns: np.ndarray) -> None:
        """Test that 10-day VaR is scaled appropriately."""
        config = VaRConfig(
            confidence_level=0.95,
            time_horizon_days=10,
            sqrt_time_scaling=True,
        )
        calculator = VaRCalculator(config)
        portfolio_value = Decimal("100000")

        result_10d = calculator.calculate_historical_var(returns, portfolio_value)

        # 10-day VaR should be roughly sqrt(10) times 1-day VaR
        config_1d = VaRConfig(confidence_level=0.95, time_horizon_days=1)
        calc_1d = VaRCalculator(config_1d)
        result_1d = calc_1d.calculate_historical_var(returns, portfolio_value)

        expected_ratio = np.sqrt(10)
        actual_ratio = float(result_10d.var / result_1d.var)
        assert abs(actual_ratio - expected_ratio) < 0.1

    def test_portfolio_var(self, calculator: VaRCalculator) -> None:
        """Test portfolio VaR calculation with correlation."""
        np.random.seed(42)

        # Generate correlated returns
        returns_btc = np.random.normal(0, 0.03, 100)
        returns_eth = returns_btc * 0.8 + np.random.normal(0, 0.02, 100)

        position_returns = {"BTC": returns_btc, "ETH": returns_eth}
        position_values = {"BTC": Decimal("60000"), "ETH": Decimal("40000")}

        portfolio_var, position_vars = calculator.calculate_portfolio_var(
            position_returns, position_values
        )

        assert portfolio_var.var > 0
        assert len(position_vars) == 2
        # Portfolio VaR should be less than sum of individual VaRs (diversification)
        sum_individual_var = sum(pv.var for pv in position_vars)
        assert portfolio_var.var < sum_individual_var

    def test_component_var(self, calculator: VaRCalculator) -> None:
        """Test component VaR calculation."""
        np.random.seed(42)

        returns_btc = np.random.normal(0, 0.03, 100)
        returns_eth = returns_btc * 0.5 + np.random.normal(0, 0.02, 100)

        position_returns = {"BTC": returns_btc, "ETH": returns_eth}
        position_values = {"BTC": Decimal("70000"), "ETH": Decimal("30000")}

        component_vars = calculator.calculate_component_var(
            position_returns, position_values
        )

        assert "BTC" in component_vars
        assert "ETH" in component_vars
        assert component_vars["BTC"] > 0
        assert component_vars["ETH"] > 0

    def test_minimum_observations_check(self, calculator: VaRCalculator) -> None:
        """Test that VaR calculation requires minimum observations."""
        returns = np.array([0.01, 0.02, -0.01])  # Too few
        portfolio_value = Decimal("100000")

        with pytest.raises(ValueError, match="Need at least"):
            calculator.calculate_historical_var(returns, portfolio_value)

    def test_var_backtest(self, calculator: VaRCalculator) -> None:
        """Test VaR backtesting."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)
        portfolio_values = np.full(500, 100000.0)

        results = calculator.backtest_var(
            returns, portfolio_values, window_size=252
        )

        assert "exception_rate" in results
        assert "kupiec_pvalue" in results
        assert "model_adequate" in results
        # Exception rate should be close to expected (1 - confidence)
        assert 0 < results["exception_rate"] < 0.15


# ============================================================================
# Stress Testing Tests
# ============================================================================


class TestStressTestEngine:
    """Tests for Stress Test Engine."""

    @pytest.fixture
    def engine(self) -> StressTestEngine:
        """Create stress test engine."""
        return create_stress_test_engine(loss_limit_pct=20.0)

    @pytest.fixture
    def positions(self) -> dict[str, Decimal]:
        """Sample portfolio positions."""
        return {
            "BTC": Decimal("50000"),
            "ETH": Decimal("30000"),
            "SOL": Decimal("20000"),
        }

    def test_historical_scenarios_exist(self) -> None:
        """Test that historical scenarios are defined."""
        assert len(HISTORICAL_SCENARIOS) > 0
        assert "covid_crash_2020" in HISTORICAL_SCENARIOS
        assert "ftx_collapse_2022" in HISTORICAL_SCENARIOS

    def test_hypothetical_scenarios_exist(self) -> None:
        """Test that hypothetical scenarios are defined."""
        assert len(HYPOTHETICAL_SCENARIOS) > 0
        assert "crypto_winter" in HYPOTHETICAL_SCENARIOS
        assert "flash_crash" in HYPOTHETICAL_SCENARIOS

    def test_run_single_scenario(
        self, engine: StressTestEngine, positions: dict[str, Decimal]
    ) -> None:
        """Test running a single scenario."""
        scenario = HISTORICAL_SCENARIOS["ftx_collapse_2022"]
        result = engine.run_scenario(scenario, positions)

        assert result.scenario.name == scenario.name
        assert result.total_pnl < 0  # Should be a loss
        assert result.total_pnl_pct < 0
        assert len(result.position_impacts) == 3

    def test_run_all_historical(
        self, engine: StressTestEngine, positions: dict[str, Decimal]
    ) -> None:
        """Test running all historical scenarios."""
        results = engine.run_all_historical(positions)

        assert len(results) > 0
        # Results should be sorted by PnL (worst first)
        for i in range(len(results) - 1):
            assert results[i].total_pnl <= results[i + 1].total_pnl

    def test_run_custom_scenario(
        self, engine: StressTestEngine, positions: dict[str, Decimal]
    ) -> None:
        """Test running a custom scenario."""
        shocks = {"BTC": -0.25, "ETH": -0.30, "SOL": -0.35}
        result = engine.run_custom_scenario("Custom Test", shocks, positions)

        assert result.scenario.name == "Custom Test"
        assert result.total_pnl < 0

        # Verify all positions were impacted with reasonable shocks
        assert len(result.position_impacts) == 3
        for impact in result.position_impacts:
            # Each position should have a negative shock applied
            assert impact.shock_applied < 0
            # PnL should be negative (loss)
            assert impact.pnl < 0

    def test_sensitivity_analysis(
        self, engine: StressTestEngine, positions: dict[str, Decimal]
    ) -> None:
        """Test sensitivity analysis."""
        result = engine.sensitivity_analysis(positions, "BTC")

        assert result.factor == "BTC"
        assert len(result.sensitivities) > 0
        assert result.delta != 0  # Portfolio should be sensitive to BTC

    def test_reverse_stress_test(
        self, engine: StressTestEngine, positions: dict[str, Decimal]
    ) -> None:
        """Test reverse stress testing."""
        result = engine.reverse_stress_test(
            positions, target_loss_pct=30.0, factor="BTC"
        )

        assert result.target_loss_pct == 30.0
        if result.scenario_found:
            assert result.breaking_scenario is not None
            # Verify the found scenario produces approximately the target loss
            test_result = engine.run_scenario(result.breaking_scenario, positions)
            assert abs(test_result.total_pnl_pct + 30.0) < 5.0

    def test_expected_loss(
        self, engine: StressTestEngine, positions: dict[str, Decimal]
    ) -> None:
        """Test expected loss calculation."""
        expected_loss = engine.expected_loss(positions)

        assert expected_loss >= 0

    def test_worst_case_scenario(
        self, engine: StressTestEngine, positions: dict[str, Decimal]
    ) -> None:
        """Test finding worst case scenario."""
        worst = engine.get_worst_case(positions)

        assert worst is not None
        assert worst.total_pnl < 0

    def test_limit_breach_detection(
        self, engine: StressTestEngine, positions: dict[str, Decimal]
    ) -> None:
        """Test detection of limit breaches."""
        # Crypto winter should breach 20% limit
        scenario = HYPOTHETICAL_SCENARIOS["crypto_winter"]
        result = engine.run_scenario(scenario, positions)

        assert result.breaches_limits
        assert len(result.breached_limits) > 0


# ============================================================================
# Correlation Monitor Tests
# ============================================================================


class TestCorrelationMonitor:
    """Tests for Correlation Monitor."""

    @pytest.fixture
    def monitor(self) -> CorrelationMonitor:
        """Create correlation monitor."""
        return create_correlation_monitor(window_size=30)

    @pytest.fixture
    def returns_dict(self) -> dict[str, np.ndarray]:
        """Generate correlated return data."""
        np.random.seed(42)
        n = 50

        returns_btc = np.random.normal(0, 0.03, n)
        returns_eth = returns_btc * 0.85 + np.random.normal(0, 0.01, n)
        returns_sol = returns_btc * 0.6 + np.random.normal(0, 0.02, n)

        return {
            "BTC": returns_btc,
            "ETH": returns_eth,
            "SOL": returns_sol,
        }

    def test_correlation_matrix_calculation(
        self, monitor: CorrelationMonitor, returns_dict: dict[str, np.ndarray]
    ) -> None:
        """Test correlation matrix calculation."""
        matrix = monitor.calculate_correlation_matrix(returns_dict)

        assert isinstance(matrix, CorrelationMatrix)
        assert len(matrix.symbols) == 3
        assert matrix.matrix.shape == (3, 3)

        # Diagonal should be 1
        for i in range(3):
            assert abs(matrix.matrix[i, i] - 1.0) < 0.01

        # BTC-ETH should be highly correlated
        btc_eth_corr = matrix.get_correlation("BTC", "ETH")
        assert btc_eth_corr > 0.7

    def test_update_with_returns(
        self, monitor: CorrelationMonitor, returns_dict: dict[str, np.ndarray]
    ) -> None:
        """Test updating monitor with returns."""
        matrix = monitor.update(returns_dict)

        assert matrix is not None
        assert monitor.current_matrix is not None

    def test_tail_correlation(
        self, monitor: CorrelationMonitor, returns_dict: dict[str, np.ndarray]
    ) -> None:
        """Test tail correlation calculation."""
        tail_matrix = monitor.calculate_tail_correlation(
            returns_dict, threshold_percentile=10.0
        )

        assert tail_matrix is not None
        # Tail correlations are often higher than normal correlations

    def test_concentration_metrics(self, monitor: CorrelationMonitor) -> None:
        """Test concentration metrics calculation."""
        weights = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        metrics = monitor.calculate_concentration(weights)

        assert metrics.hhi > 0
        assert metrics.effective_positions > 0
        assert metrics.max_weight == 0.5
        assert metrics.max_weight_symbol == "BTC"
        assert abs(metrics.top_3_weight - 1.0) < 0.01

    def test_concentration_detection(self, monitor: CorrelationMonitor) -> None:
        """Test concentrated portfolio detection."""
        # Concentrated portfolio
        concentrated = {"BTC": 0.8, "ETH": 0.2}
        metrics = monitor.calculate_concentration(concentrated)
        assert metrics.is_concentrated

        # Diversified portfolio
        diversified = {"BTC": 0.25, "ETH": 0.25, "SOL": 0.25, "LINK": 0.25}
        metrics = monitor.calculate_concentration(diversified)
        assert not metrics.is_concentrated

    def test_diversification_metrics(
        self, monitor: CorrelationMonitor, returns_dict: dict[str, np.ndarray]
    ) -> None:
        """Test diversification metrics calculation."""
        weights = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        metrics = monitor.calculate_diversification(returns_dict, weights)

        assert metrics.diversification_ratio >= 1.0
        assert len(metrics.marginal_risk) == 3
        assert len(metrics.risk_parity_weights) == 3

    def test_correlation_regime(
        self, monitor: CorrelationMonitor, returns_dict: dict[str, np.ndarray]
    ) -> None:
        """Test correlation regime detection."""
        monitor.update(returns_dict)

        # With highly correlated assets, should be HIGH regime
        assert monitor.regime in [CorrelationRegime.HIGH, CorrelationRegime.NORMAL]

    def test_different_methods(
        self, returns_dict: dict[str, np.ndarray]
    ) -> None:
        """Test different correlation methods."""
        for method in CorrelationMethod:
            monitor = CorrelationMonitor(method=method)
            matrix = monitor.calculate_correlation_matrix(returns_dict, method)
            assert matrix.method == method


# ============================================================================
# Margin Monitor Tests
# ============================================================================


class TestMarginMonitor:
    """Tests for Margin Monitor."""

    @pytest.fixture
    def monitor(self) -> MarginMonitor:
        """Create margin monitor."""
        return create_margin_monitor(
            account_equity=Decimal("100000"),
            max_leverage=Decimal("10"),
        )

    @pytest.fixture
    def sample_position(self) -> MarginPosition:
        """Create sample margin position."""
        return MarginPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=Decimal("1.5"),
            entry_price=Decimal("45000"),
            mark_price=Decimal("44000"),
            leverage=Decimal("10"),
            margin_mode=MarginMode.CROSS,
            initial_margin=Decimal("6750"),
            maintenance_margin=Decimal("450"),
            margin_balance=Decimal("6000"),
        )

    def test_position_creation(self, sample_position: MarginPosition) -> None:
        """Test margin position creation and calculations."""
        assert sample_position.notional_value == Decimal("66000")  # 1.5 * 44000
        assert sample_position.unrealized_pnl == Decimal("-1500")  # 1.5 * (44000 - 45000)
        assert sample_position.margin_ratio > 0
        assert sample_position.liquidation_price > 0
        assert sample_position.liquidation_price < sample_position.entry_price

    def test_update_position(
        self, monitor: MarginMonitor, sample_position: MarginPosition
    ) -> None:
        """Test updating position in monitor."""
        alerts = monitor.update_position(sample_position)

        pos = monitor.get_position("BTCUSDT")
        assert pos is not None
        assert pos.symbol == "BTCUSDT"

    def test_portfolio_margin(
        self, monitor: MarginMonitor, sample_position: MarginPosition
    ) -> None:
        """Test portfolio margin calculation."""
        monitor.update_position(sample_position)
        portfolio = monitor.get_portfolio_margin()

        assert portfolio.total_maintenance_margin > 0
        assert portfolio.margin_utilization > 0
        assert portfolio.position_count == 1

    def test_liquidation_price_long(self, monitor: MarginMonitor) -> None:
        """Test liquidation price calculation for long position."""
        liq_price = monitor.calculate_liquidation_price(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            leverage=Decimal("10"),
            side=PositionSide.LONG,
        )

        # With 10x leverage and 0.5% maintenance, liq should be ~10% below entry
        assert liq_price < Decimal("50000")
        assert liq_price > Decimal("45000")

    def test_liquidation_price_short(self, monitor: MarginMonitor) -> None:
        """Test liquidation price calculation for short position."""
        liq_price = monitor.calculate_liquidation_price(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            leverage=Decimal("10"),
            side=PositionSide.SHORT,
        )

        # For short, liq price should be above entry
        assert liq_price > Decimal("50000")

    def test_max_position_size(
        self, monitor: MarginMonitor, sample_position: MarginPosition
    ) -> None:
        """Test maximum position size calculation."""
        max_size = monitor.calculate_max_position_size(
            symbol="ETHUSDT",
            price=Decimal("3000"),
            leverage=Decimal("5"),
        )

        # With 100k equity and 5x leverage, max notional is 500k
        # At 3000 per ETH, max size should be ~166 ETH
        assert max_size > 0
        assert max_size < Decimal("200")

    def test_margin_alerts(self, monitor: MarginMonitor) -> None:
        """Test margin alert generation."""
        # Create position close to liquidation
        risky_position = MarginPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=Decimal("1"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("45500"),  # Close to liquidation
            leverage=Decimal("10"),
            margin_mode=MarginMode.ISOLATED,
            initial_margin=Decimal("5000"),
            maintenance_margin=Decimal("250"),
            margin_balance=Decimal("500"),  # Very low margin
        )

        alerts = monitor.update_position(risky_position)

        # Should generate alert due to low margin
        assert len(alerts) > 0 or risky_position.distance_to_liquidation_pct > 10

    def test_alert_levels(self) -> None:
        """Test margin alert level classification."""
        config = MarginConfig()

        # Healthy: margin_level > 1.5
        # Caution: 1.25 < margin_level < 1.5
        # Warning: 1.1 < margin_level < 1.25
        # Critical: margin_level < 1.1

        assert config.caution_threshold == 1.5
        assert config.warning_threshold == 1.25
        assert config.critical_threshold == 1.1

    def test_margin_summary(
        self, monitor: MarginMonitor, sample_position: MarginPosition
    ) -> None:
        """Test margin summary generation."""
        monitor.update_position(sample_position)
        summary = monitor.get_margin_summary()

        assert "portfolio" in summary
        assert "positions" in summary
        assert "margin_utilization" in summary["portfolio"]

    def test_mark_price_update(
        self, monitor: MarginMonitor, sample_position: MarginPosition
    ) -> None:
        """Test mark price update."""
        monitor.update_position(sample_position)
        original_pnl = sample_position.unrealized_pnl

        # Update mark price
        alerts = monitor.update_mark_price("BTCUSDT", Decimal("43000"))

        updated = monitor.get_position("BTCUSDT")
        assert updated is not None
        assert updated.mark_price == Decimal("43000")

    def test_excess_leverage_alert(self, monitor: MarginMonitor) -> None:
        """Test alert for excess leverage."""
        high_leverage_position = MarginPosition(
            symbol="SOLUSDT",
            side=PositionSide.LONG,
            size=Decimal("100"),
            entry_price=Decimal("100"),
            mark_price=Decimal("100"),
            leverage=Decimal("20"),  # Exceeds 10x limit
            margin_mode=MarginMode.ISOLATED,
            initial_margin=Decimal("500"),
            maintenance_margin=Decimal("50"),
            margin_balance=Decimal("500"),
        )

        alerts = monitor.update_position(high_leverage_position)

        # Should generate excess leverage alert
        leverage_alerts = [a for a in alerts if a.alert_type == "excess_leverage"]
        assert len(leverage_alerts) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestAdvancedRiskIntegration:
    """Integration tests combining multiple risk components."""

    def test_var_with_stress_test_correlation(self) -> None:
        """Test that VaR and stress test results are correlated."""
        np.random.seed(42)

        # Generate returns
        returns = np.random.normal(0, 0.03, 252)
        portfolio_value = Decimal("100000")

        # Calculate VaR
        var_calc = create_var_calculator()
        var_result = var_calc.calculate_historical_var(returns, portfolio_value)

        # Run stress tests
        stress_engine = create_stress_test_engine()
        positions = {"BTC": Decimal("100000")}
        stress_results = stress_engine.run_all_scenarios(positions)

        # Worst stress loss should generally exceed VaR
        worst_loss = abs(stress_results[0].total_pnl)
        assert worst_loss > var_result.var

    def test_correlation_affects_var(self) -> None:
        """Test that correlation affects portfolio VaR."""
        np.random.seed(42)
        n = 100

        # Low correlation portfolio
        uncorrelated = {
            "A": np.random.normal(0, 0.02, n),
            "B": np.random.normal(0, 0.02, n),
        }

        # High correlation portfolio
        base = np.random.normal(0, 0.02, n)
        correlated = {
            "A": base,
            "B": base * 0.95 + np.random.normal(0, 0.005, n),
        }

        values = {"A": Decimal("50000"), "B": Decimal("50000")}
        var_calc = create_var_calculator()

        var_uncorr, _ = var_calc.calculate_portfolio_var(uncorrelated, values)
        var_corr, _ = var_calc.calculate_portfolio_var(correlated, values)

        # Correlated portfolio should have higher VaR
        assert var_corr.var > var_uncorr.var

    def test_margin_and_stress_combined(self) -> None:
        """Test margin impact under stress scenarios."""
        # Create position
        position = MarginPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=Decimal("2"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            leverage=Decimal("5"),
            margin_mode=MarginMode.CROSS,
            initial_margin=Decimal("20000"),
            maintenance_margin=Decimal("1000"),
            margin_balance=Decimal("20000"),
        )

        margin_monitor = create_margin_monitor(account_equity=Decimal("100000"))
        margin_monitor.update_position(position)

        # Simulate COVID crash shock
        shock = HISTORICAL_SCENARIOS["covid_crash_2020"].shocks[0].shock_pct  # BTC shock

        new_price = position.mark_price * Decimal(str(1 + shock))
        margin_monitor.update_mark_price("BTCUSDT", new_price)

        updated_pos = margin_monitor.get_position("BTCUSDT")
        assert updated_pos is not None

        # Position should be close to or in danger of liquidation
        assert updated_pos.distance_to_liquidation_pct < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

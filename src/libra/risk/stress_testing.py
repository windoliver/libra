"""
Stress Testing Framework.

Implements comprehensive stress testing for portfolio risk assessment:
- Historical scenarios: Replays past market crises
- Hypothetical scenarios: Custom shock scenarios
- Reverse stress testing: Finds breaking scenarios
- Sensitivity analysis: Measures factor sensitivities

References:
- Basel III stress testing requirements
- Fed CCAR/DFAST frameworks
- CFTC stress testing guidelines
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Type of stress scenario."""

    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    REVERSE = "reverse"
    SENSITIVITY = "sensitivity"


class AssetClass(Enum):
    """Asset class for scenario mapping."""

    CRYPTO = "crypto"
    EQUITY = "equity"
    FX = "fx"
    COMMODITY = "commodity"
    FIXED_INCOME = "fixed_income"


@dataclass
class MarketShock:
    """A shock to a specific market factor."""

    factor: str  # e.g., "BTC", "ETH", "SPX", "interest_rate"
    shock_pct: float  # Percentage change (e.g., -0.30 for -30%)
    asset_class: AssetClass = AssetClass.CRYPTO
    correlation_impact: float = 0.0  # Impact on correlations


@dataclass
class StressScenario:
    """Definition of a stress scenario."""

    name: str
    description: str
    scenario_type: ScenarioType
    shocks: list[MarketShock]
    probability: float | None = None  # Estimated probability
    historical_date: datetime | None = None  # For historical scenarios
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionImpact:
    """Impact of stress scenario on a single position."""

    symbol: str
    current_value: Decimal
    stressed_value: Decimal
    pnl: Decimal
    pnl_pct: float
    shock_applied: float


@dataclass
class StressTestResult:
    """Result of a stress test."""

    scenario: StressScenario
    portfolio_value_before: Decimal
    portfolio_value_after: Decimal
    total_pnl: Decimal
    total_pnl_pct: float
    position_impacts: list[PositionImpact]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    breaches_limits: bool = False
    breached_limits: list[str] = field(default_factory=list)

    @property
    def is_severe(self) -> bool:
        """Check if scenario causes severe loss (>20%)."""
        return self.total_pnl_pct < -20.0


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""

    factor: str
    base_value: Decimal
    sensitivities: dict[float, Decimal]  # shock_pct -> portfolio_value
    delta: float  # First derivative (dV/dS)
    gamma: float  # Second derivative (d2V/dS2)
    elasticity: float  # (dV/V) / (dS/S)


@dataclass
class ReverseStressResult:
    """Result of reverse stress testing."""

    target_loss: Decimal
    target_loss_pct: float
    scenario_found: bool
    breaking_scenario: StressScenario | None
    iterations: int
    search_path: list[tuple[float, Decimal]]  # (shock, resulting_loss)


# ============================================================================
# Pre-defined Historical Scenarios
# ============================================================================

HISTORICAL_SCENARIOS = {
    "covid_crash_2020": StressScenario(
        name="COVID-19 Crash (March 2020)",
        description="Market crash due to COVID-19 pandemic onset",
        scenario_type=ScenarioType.HISTORICAL,
        historical_date=datetime(2020, 3, 12),
        shocks=[
            MarketShock("BTC", -0.40, AssetClass.CRYPTO),
            MarketShock("ETH", -0.45, AssetClass.CRYPTO),
            MarketShock("SOL", -0.55, AssetClass.CRYPTO),
            MarketShock("SPX", -0.12, AssetClass.EQUITY),
            MarketShock("VIX", 2.50, AssetClass.EQUITY),
        ],
        probability=0.01,
    ),
    "luna_collapse_2022": StressScenario(
        name="Luna/UST Collapse (May 2022)",
        description="Algorithmic stablecoin collapse triggering crypto contagion",
        scenario_type=ScenarioType.HISTORICAL,
        historical_date=datetime(2022, 5, 9),
        shocks=[
            MarketShock("BTC", -0.25, AssetClass.CRYPTO),
            MarketShock("ETH", -0.30, AssetClass.CRYPTO),
            MarketShock("SOL", -0.45, AssetClass.CRYPTO),
            MarketShock("LUNA", -0.999, AssetClass.CRYPTO),
            MarketShock("UST", -0.95, AssetClass.CRYPTO),
        ],
        probability=0.005,
    ),
    "ftx_collapse_2022": StressScenario(
        name="FTX Collapse (November 2022)",
        description="Exchange collapse and contagion",
        scenario_type=ScenarioType.HISTORICAL,
        historical_date=datetime(2022, 11, 8),
        shocks=[
            MarketShock("BTC", -0.22, AssetClass.CRYPTO),
            MarketShock("ETH", -0.25, AssetClass.CRYPTO),
            MarketShock("SOL", -0.60, AssetClass.CRYPTO),
            MarketShock("FTT", -0.95, AssetClass.CRYPTO),
        ],
        probability=0.005,
    ),
    "china_ban_2021": StressScenario(
        name="China Crypto Ban (September 2021)",
        description="China declares all crypto transactions illegal",
        scenario_type=ScenarioType.HISTORICAL,
        historical_date=datetime(2021, 9, 24),
        shocks=[
            MarketShock("BTC", -0.10, AssetClass.CRYPTO),
            MarketShock("ETH", -0.12, AssetClass.CRYPTO),
            MarketShock("SOL", -0.15, AssetClass.CRYPTO),
        ],
        probability=0.02,
    ),
    "black_thursday_2020": StressScenario(
        name="Black Thursday (March 2020)",
        description="Extreme crypto deleveraging event",
        scenario_type=ScenarioType.HISTORICAL,
        historical_date=datetime(2020, 3, 12),
        shocks=[
            MarketShock("BTC", -0.50, AssetClass.CRYPTO),
            MarketShock("ETH", -0.55, AssetClass.CRYPTO),
            MarketShock("SOL", -0.65, AssetClass.CRYPTO),
        ],
        probability=0.005,
    ),
    "gfc_2008": StressScenario(
        name="Global Financial Crisis (2008)",
        description="Lehman Brothers collapse and credit crisis",
        scenario_type=ScenarioType.HISTORICAL,
        historical_date=datetime(2008, 9, 15),
        shocks=[
            MarketShock("SPX", -0.20, AssetClass.EQUITY),
            MarketShock("VIX", 3.00, AssetClass.EQUITY),
            MarketShock("GOLD", 0.05, AssetClass.COMMODITY),
            MarketShock("USD", 0.10, AssetClass.FX),
        ],
        probability=0.005,
    ),
}

# Pre-defined Hypothetical Scenarios
HYPOTHETICAL_SCENARIOS = {
    "crypto_winter": StressScenario(
        name="Crypto Winter",
        description="Extended bear market with 70% drawdown",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks=[
            MarketShock("BTC", -0.70, AssetClass.CRYPTO),
            MarketShock("ETH", -0.80, AssetClass.CRYPTO),
            MarketShock("SOL", -0.90, AssetClass.CRYPTO),
        ],
        probability=0.05,
    ),
    "stablecoin_depeg": StressScenario(
        name="Major Stablecoin Depeg",
        description="USDT or USDC loses peg significantly",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks=[
            MarketShock("USDT", -0.15, AssetClass.CRYPTO),
            MarketShock("BTC", -0.25, AssetClass.CRYPTO),
            MarketShock("ETH", -0.30, AssetClass.CRYPTO),
        ],
        probability=0.02,
    ),
    "regulatory_crackdown": StressScenario(
        name="US Regulatory Crackdown",
        description="SEC classifies major cryptos as securities",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks=[
            MarketShock("BTC", -0.15, AssetClass.CRYPTO),
            MarketShock("ETH", -0.40, AssetClass.CRYPTO),
            MarketShock("SOL", -0.50, AssetClass.CRYPTO),
            MarketShock("XRP", -0.60, AssetClass.CRYPTO),
        ],
        probability=0.10,
    ),
    "exchange_hack": StressScenario(
        name="Major Exchange Hack",
        description="Large exchange loses significant funds to hack",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks=[
            MarketShock("BTC", -0.15, AssetClass.CRYPTO),
            MarketShock("ETH", -0.18, AssetClass.CRYPTO),
            MarketShock("SOL", -0.20, AssetClass.CRYPTO),
        ],
        probability=0.05,
    ),
    "flash_crash": StressScenario(
        name="Flash Crash",
        description="Sudden liquidity crisis causing rapid price drop",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks=[
            MarketShock("BTC", -0.20, AssetClass.CRYPTO),
            MarketShock("ETH", -0.25, AssetClass.CRYPTO),
            MarketShock("SOL", -0.35, AssetClass.CRYPTO),
        ],
        probability=0.10,
    ),
    "correlation_spike": StressScenario(
        name="Correlation Spike",
        description="All assets move together during stress",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks=[
            MarketShock("BTC", -0.30, AssetClass.CRYPTO, correlation_impact=0.3),
            MarketShock("ETH", -0.35, AssetClass.CRYPTO, correlation_impact=0.3),
            MarketShock("SOL", -0.40, AssetClass.CRYPTO, correlation_impact=0.3),
            MarketShock("SPX", -0.10, AssetClass.EQUITY, correlation_impact=0.2),
        ],
        probability=0.15,
    ),
}


class StressTestEngine:
    """
    Stress Testing Engine.

    Runs stress scenarios against portfolio and evaluates
    potential losses under adverse conditions.

    Example:
        engine = StressTestEngine()

        # Run historical scenario
        result = engine.run_scenario(
            scenario=HISTORICAL_SCENARIOS["ftx_collapse_2022"],
            positions={"BTC": Decimal("50000"), "SOL": Decimal("10000")},
        )

        print(f"PnL under FTX collapse: ${result.total_pnl:,.2f}")

        # Run all historical scenarios
        results = engine.run_all_historical(positions)
        for r in results:
            print(f"{r.scenario.name}: {r.total_pnl_pct:.1f}%")
    """

    def __init__(
        self,
        loss_limit_pct: float = 20.0,
        custom_scenarios: dict[str, StressScenario] | None = None,
    ) -> None:
        """
        Initialize stress test engine.

        Args:
            loss_limit_pct: Loss threshold to flag as limit breach
            custom_scenarios: Additional custom scenarios to include
        """
        self.loss_limit_pct = loss_limit_pct
        self.scenarios: dict[str, StressScenario] = {}
        self.scenarios.update(HISTORICAL_SCENARIOS)
        self.scenarios.update(HYPOTHETICAL_SCENARIOS)
        if custom_scenarios:
            self.scenarios.update(custom_scenarios)

        # Symbol to base asset mapping (for applying shocks)
        self._symbol_mapping: dict[str, str] = {}

    def set_symbol_mapping(self, mapping: dict[str, str]) -> None:
        """
        Set mapping from trading symbols to base assets.

        Args:
            mapping: Dict of symbol -> base asset (e.g., "BTCUSDT" -> "BTC")
        """
        self._symbol_mapping = mapping

    def _get_base_asset(self, symbol: str) -> str:
        """Get base asset for a symbol."""
        if symbol in self._symbol_mapping:
            return self._symbol_mapping[symbol]
        # Try common patterns
        for suffix in ["USDT", "USD", "BUSD", "USDC", "BTC", "ETH"]:
            if symbol.endswith(suffix):
                return symbol[: -len(suffix)]
        return symbol

    def run_scenario(
        self,
        scenario: StressScenario,
        positions: dict[str, Decimal],
        prices: dict[str, Decimal] | None = None,
    ) -> StressTestResult:
        """
        Run a single stress scenario.

        Args:
            scenario: Stress scenario to run
            positions: Dict of symbol -> position value
            prices: Current prices (optional, for quantity calculations)

        Returns:
            StressTestResult with PnL impact
        """
        # Build shock lookup by factor
        shock_by_factor = {shock.factor.upper(): shock for shock in scenario.shocks}

        # Calculate portfolio value before stress
        portfolio_before = sum(positions.values(), Decimal("0"))

        # Apply shocks to each position
        position_impacts: list[PositionImpact] = []
        portfolio_after = Decimal("0")

        for symbol, value in positions.items():
            base_asset = self._get_base_asset(symbol).upper()

            # Find applicable shock
            shock = shock_by_factor.get(base_asset)
            if shock:
                shock_pct = shock.shock_pct
            else:
                # No specific shock - apply default crypto shock if crypto asset
                shock_pct = 0.0
                # Check if any crypto shock should apply
                for factor, s in shock_by_factor.items():
                    if s.asset_class == AssetClass.CRYPTO and base_asset not in [
                        "USDT", "USDC", "BUSD", "DAI", "TUSD"
                    ]:
                        # Apply average crypto shock
                        crypto_shocks = [
                            sh.shock_pct for sh in scenario.shocks
                            if sh.asset_class == AssetClass.CRYPTO
                        ]
                        if crypto_shocks:
                            shock_pct = sum(crypto_shocks) / len(crypto_shocks)
                        break

            # Calculate stressed value
            stressed_value = value * Decimal(str(1 + shock_pct))
            pnl = stressed_value - value
            pnl_pct = float(pnl / value) * 100 if value != 0 else 0.0

            position_impacts.append(
                PositionImpact(
                    symbol=symbol,
                    current_value=value,
                    stressed_value=stressed_value,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    shock_applied=shock_pct,
                )
            )

            portfolio_after += stressed_value

        # Calculate total PnL
        total_pnl = portfolio_after - portfolio_before
        total_pnl_pct = (
            float(total_pnl / portfolio_before) * 100
            if portfolio_before != 0
            else 0.0
        )

        # Check limit breaches
        breaches_limits = abs(total_pnl_pct) > self.loss_limit_pct
        breached_limits = []
        if breaches_limits:
            breached_limits.append(f"Loss exceeds {self.loss_limit_pct}% limit")

        return StressTestResult(
            scenario=scenario,
            portfolio_value_before=portfolio_before,
            portfolio_value_after=portfolio_after,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            position_impacts=position_impacts,
            breaches_limits=breaches_limits,
            breached_limits=breached_limits,
        )

    def run_all_historical(
        self,
        positions: dict[str, Decimal],
    ) -> list[StressTestResult]:
        """
        Run all historical scenarios.

        Args:
            positions: Current positions

        Returns:
            List of stress test results
        """
        results = []
        for name, scenario in self.scenarios.items():
            if scenario.scenario_type == ScenarioType.HISTORICAL:
                result = self.run_scenario(scenario, positions)
                results.append(result)
        return sorted(results, key=lambda r: r.total_pnl)

    def run_all_hypothetical(
        self,
        positions: dict[str, Decimal],
    ) -> list[StressTestResult]:
        """
        Run all hypothetical scenarios.

        Args:
            positions: Current positions

        Returns:
            List of stress test results
        """
        results = []
        for name, scenario in self.scenarios.items():
            if scenario.scenario_type == ScenarioType.HYPOTHETICAL:
                result = self.run_scenario(scenario, positions)
                results.append(result)
        return sorted(results, key=lambda r: r.total_pnl)

    def run_all_scenarios(
        self,
        positions: dict[str, Decimal],
    ) -> list[StressTestResult]:
        """
        Run all available scenarios.

        Args:
            positions: Current positions

        Returns:
            List of stress test results sorted by loss
        """
        results = []
        for scenario in self.scenarios.values():
            result = self.run_scenario(scenario, positions)
            results.append(result)
        return sorted(results, key=lambda r: r.total_pnl)

    def run_custom_scenario(
        self,
        name: str,
        shocks: dict[str, float],
        positions: dict[str, Decimal],
        description: str = "Custom scenario",
    ) -> StressTestResult:
        """
        Run a custom scenario with specified shocks.

        Args:
            name: Scenario name
            shocks: Dict of factor -> shock percentage
            positions: Current positions
            description: Scenario description

        Returns:
            StressTestResult
        """
        market_shocks = [
            MarketShock(factor, shock_pct, AssetClass.CRYPTO)
            for factor, shock_pct in shocks.items()
        ]

        scenario = StressScenario(
            name=name,
            description=description,
            scenario_type=ScenarioType.HYPOTHETICAL,
            shocks=market_shocks,
        )

        return self.run_scenario(scenario, positions)

    def sensitivity_analysis(
        self,
        positions: dict[str, Decimal],
        factor: str,
        shock_range: Sequence[float] = (-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5),
    ) -> SensitivityResult:
        """
        Perform sensitivity analysis on a factor.

        Args:
            positions: Current positions
            factor: Factor to analyze (e.g., "BTC")
            shock_range: Range of shocks to apply

        Returns:
            SensitivityResult with sensitivities and greeks
        """
        base_value = sum(positions.values(), Decimal("0"))
        sensitivities: dict[float, Decimal] = {}

        for shock in shock_range:
            result = self.run_custom_scenario(
                name=f"Sensitivity {factor} {shock:+.0%}",
                shocks={factor: shock},
                positions=positions,
            )
            sensitivities[shock] = result.portfolio_value_after

        # Calculate delta (first derivative) using central difference
        shocks_arr = np.array(list(shock_range))
        values_arr = np.array([float(sensitivities[s]) for s in shock_range])

        # Find points around 0 for delta calculation
        if 0 in shock_range:
            idx_zero = list(shock_range).index(0)
            if idx_zero > 0 and idx_zero < len(shock_range) - 1:
                h = shock_range[idx_zero + 1] - shock_range[idx_zero]
                delta = (values_arr[idx_zero + 1] - values_arr[idx_zero - 1]) / (2 * h)

                # Gamma (second derivative)
                gamma = (
                    values_arr[idx_zero + 1]
                    - 2 * values_arr[idx_zero]
                    + values_arr[idx_zero - 1]
                ) / (h**2)
            else:
                delta = 0.0
                gamma = 0.0
        else:
            # Approximate using linear regression
            delta = float(np.polyfit(shocks_arr, values_arr, 1)[0])
            gamma = 0.0

        # Elasticity
        elasticity = delta * float(base_value) / float(base_value) if base_value != 0 else 0.0

        return SensitivityResult(
            factor=factor,
            base_value=base_value,
            sensitivities=sensitivities,
            delta=delta,
            gamma=gamma,
            elasticity=elasticity,
        )

    def reverse_stress_test(
        self,
        positions: dict[str, Decimal],
        target_loss_pct: float,
        factor: str = "BTC",
        max_iterations: int = 50,
        tolerance: float = 0.01,
    ) -> ReverseStressResult:
        """
        Find the shock level that produces target loss.

        Reverse stress testing finds scenarios that would cause
        a specified level of loss.

        Args:
            positions: Current positions
            target_loss_pct: Target loss percentage (e.g., 50 for 50%)
            factor: Factor to shock
            max_iterations: Maximum search iterations
            tolerance: Convergence tolerance

        Returns:
            ReverseStressResult with breaking scenario
        """
        portfolio_value = sum(positions.values(), Decimal("0"))
        target_loss = portfolio_value * Decimal(str(target_loss_pct / 100))

        # Binary search for shock level
        low_shock = -0.99  # Maximum possible loss
        high_shock = 0.0
        search_path: list[tuple[float, Decimal]] = []

        for iteration in range(max_iterations):
            mid_shock = (low_shock + high_shock) / 2

            result = self.run_custom_scenario(
                name=f"Reverse test iteration {iteration}",
                shocks={factor: mid_shock},
                positions=positions,
            )

            actual_loss = -result.total_pnl
            search_path.append((mid_shock, actual_loss))

            # Check convergence
            loss_diff = abs(float(actual_loss - target_loss) / float(target_loss))
            if loss_diff < tolerance:
                # Found breaking scenario
                breaking_scenario = StressScenario(
                    name=f"Reverse Stress: {target_loss_pct}% Loss",
                    description=f"{factor} shock of {mid_shock:.1%} causes {target_loss_pct}% loss",
                    scenario_type=ScenarioType.REVERSE,
                    shocks=[MarketShock(factor, mid_shock, AssetClass.CRYPTO)],
                )
                return ReverseStressResult(
                    target_loss=target_loss,
                    target_loss_pct=target_loss_pct,
                    scenario_found=True,
                    breaking_scenario=breaking_scenario,
                    iterations=iteration + 1,
                    search_path=search_path,
                )

            # Adjust search range
            if actual_loss < target_loss:
                high_shock = mid_shock
            else:
                low_shock = mid_shock

        # Did not converge
        return ReverseStressResult(
            target_loss=target_loss,
            target_loss_pct=target_loss_pct,
            scenario_found=False,
            breaking_scenario=None,
            iterations=max_iterations,
            search_path=search_path,
        )

    def expected_loss(
        self,
        positions: dict[str, Decimal],
        scenarios: list[StressScenario] | None = None,
    ) -> Decimal:
        """
        Calculate probability-weighted expected loss.

        Args:
            positions: Current positions
            scenarios: Scenarios to consider (defaults to all with probabilities)

        Returns:
            Expected loss across scenarios
        """
        if scenarios is None:
            scenarios = [
                s for s in self.scenarios.values() if s.probability is not None
            ]

        total_expected_loss = Decimal("0")

        for scenario in scenarios:
            if scenario.probability is None:
                continue

            result = self.run_scenario(scenario, positions)
            if result.total_pnl < 0:
                expected_loss = abs(result.total_pnl) * Decimal(str(scenario.probability))
                total_expected_loss += expected_loss

        return total_expected_loss

    def get_worst_case(
        self,
        positions: dict[str, Decimal],
    ) -> StressTestResult | None:
        """
        Find the worst-case scenario.

        Args:
            positions: Current positions

        Returns:
            StressTestResult for worst scenario, or None if no scenarios
        """
        results = self.run_all_scenarios(positions)
        return results[0] if results else None

    def add_scenario(self, name: str, scenario: StressScenario) -> None:
        """Add a custom scenario."""
        self.scenarios[name] = scenario

    def remove_scenario(self, name: str) -> None:
        """Remove a scenario."""
        self.scenarios.pop(name, None)


def create_stress_test_engine(
    loss_limit_pct: float = 20.0,
) -> StressTestEngine:
    """
    Factory function to create stress test engine.

    Args:
        loss_limit_pct: Loss limit percentage for breach detection

    Returns:
        Configured StressTestEngine
    """
    return StressTestEngine(loss_limit_pct=loss_limit_pct)

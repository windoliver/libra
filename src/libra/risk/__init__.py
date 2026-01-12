"""
LIBRA Risk: Risk management system.

This module provides comprehensive risk management for trading:

- RiskEngine: Pre-trade validation engine (all orders must pass through)
- RiskEngineConfig: Complete engine configuration
- RiskLimits: Configurable risk limits (YAML supported)
- PositionSizer: Position sizing algorithms (Fixed%, ATR, Kelly)
- CircuitBreaker: Automatic trading halt on risk events
- TokenBucketRateLimiter: Order rate limiting

Usage:
    from libra.risk import RiskEngine, RiskEngineConfig, RiskLimits

    # Configure engine
    config = RiskEngineConfig(
        limits=RiskLimits.from_yaml(Path("config/risk_limits.yaml")),
        enable_self_trade_prevention=True,
        enable_price_collar=True,
        price_collar_pct=Decimal("0.10"),  # 10%
    )

    # Create engine
    engine = RiskEngine(config=config, bus=message_bus)

    # Validate every order before execution
    result = engine.validate_order(order, current_price, instrument)
    if not result.passed:
        logger.warning("Order denied: %s", result.reason)
        return

    # Order passed all checks
    await execution_client.submit_order(order)

Architecture:
    Strategy -> RiskEngine.validate_order() -> ExecutionEngine -> Gateway

    ALL orders MUST pass through RiskEngine. This is mandatory.

Features:
    - Trading state management (ACTIVE/REDUCING/HALTED)
    - Position limit checks
    - Notional value checks
    - Order rate limiting (submit + modify)
    - Drawdown monitoring
    - Circuit breaker
    - Self-trade prevention (wash trading)
    - Price/quantity precision validation
    - Price collar (fat-finger protection)

References:
    - NautilusTrader RiskEngine pattern
    - FIA Best Practices for Automated Trading Risk Controls
"""

from libra.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from libra.risk.engine import (
    RiskCheckResult,
    RiskEngine,
    RiskEngineConfig,
    TradingState,
)
from libra.risk.limits import (
    RiskLimits,
    SymbolLimits,
)
from libra.risk.manager import (
    RiskManager,  # Backward compatibility alias
)
from libra.risk.position_sizing import (
    FixedPercentageSizer,
    FixedQuantitySizer,
    KellyCriterionSizer,
    PositionSizeResult,
    PositionSizer,
    VolatilityAdjustedSizer,
    create_sizer,
)
from libra.risk.rate_limiter import (
    MultiRateLimiter,
    TokenBucketRateLimiter,
)

# Advanced Risk Management (Issue #15)
from libra.risk.var import (
    ConfidenceLevel,
    PositionVaR,
    VaRCalculator,
    VaRConfig,
    VaRMethod,
    VaRResult,
    create_var_calculator,
)
from libra.risk.stress_testing import (
    AssetClass,
    MarketShock,
    PositionImpact,
    ReverseStressResult,
    ScenarioType,
    SensitivityResult,
    StressScenario,
    StressTestEngine,
    StressTestResult,
    create_stress_test_engine,
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS,
)
from libra.risk.correlation import (
    ConcentrationMetrics,
    CorrelationAlert,
    CorrelationMatrix,
    CorrelationMethod,
    CorrelationMonitor,
    CorrelationMonitorState,
    CorrelationRegime,
    DiversificationMetrics,
    create_correlation_monitor,
)
from libra.risk.margin import (
    MarginAlert,
    MarginAlertLevel,
    MarginConfig,
    MarginMode,
    MarginMonitor,
    MarginPosition,
    PortfolioMargin,
    PositionSide,
    create_margin_monitor,
)
from libra.risk.advanced_manager import (
    AdvancedRiskConfig,
    AdvancedRiskManager,
    AdvancedRiskReport,
    create_advanced_risk_manager,
)


__all__ = [
    # Engine (new)
    "RiskEngine",
    "RiskEngineConfig",
    "RiskCheckResult",
    "TradingState",
    # Manager (backward compatibility)
    "RiskManager",
    # Limits
    "RiskLimits",
    "SymbolLimits",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    # Rate Limiter
    "TokenBucketRateLimiter",
    "MultiRateLimiter",
    # Position Sizing
    "PositionSizer",
    "PositionSizeResult",
    "FixedPercentageSizer",
    "VolatilityAdjustedSizer",
    "KellyCriterionSizer",
    "FixedQuantitySizer",
    "create_sizer",
    # VaR Calculator (Issue #15)
    "ConfidenceLevel",
    "PositionVaR",
    "VaRCalculator",
    "VaRConfig",
    "VaRMethod",
    "VaRResult",
    "create_var_calculator",
    # Stress Testing (Issue #15)
    "AssetClass",
    "MarketShock",
    "PositionImpact",
    "ReverseStressResult",
    "ScenarioType",
    "SensitivityResult",
    "StressScenario",
    "StressTestEngine",
    "StressTestResult",
    "create_stress_test_engine",
    "HISTORICAL_SCENARIOS",
    "HYPOTHETICAL_SCENARIOS",
    # Correlation Monitor (Issue #15)
    "ConcentrationMetrics",
    "CorrelationAlert",
    "CorrelationMatrix",
    "CorrelationMethod",
    "CorrelationMonitor",
    "CorrelationMonitorState",
    "CorrelationRegime",
    "DiversificationMetrics",
    "create_correlation_monitor",
    # Margin Monitor (Issue #15)
    "MarginAlert",
    "MarginAlertLevel",
    "MarginConfig",
    "MarginMode",
    "MarginMonitor",
    "MarginPosition",
    "PortfolioMargin",
    "PositionSide",
    "create_margin_monitor",
    # Advanced Risk Manager (Issue #15)
    "AdvancedRiskConfig",
    "AdvancedRiskManager",
    "AdvancedRiskReport",
    "create_advanced_risk_manager",
]

"""
LIBRA Risk: Risk management system.

This module provides comprehensive risk management for trading:

- RiskManager: Pre-trade validation engine (all orders must pass through)
- RiskLimits: Configurable risk limits (YAML supported)
- PositionSizer: Position sizing algorithms (Fixed%, ATR, Kelly)
- CircuitBreaker: Automatic trading halt on risk events
- TokenBucketRateLimiter: Order rate limiting

Usage:
    from libra.risk import RiskManager, RiskLimits

    # Load limits from config
    limits = RiskLimits.from_yaml(Path("config/risk_limits.yaml"))

    # Create manager
    manager = RiskManager(limits=limits, bus=message_bus)

    # Validate every order before execution
    result = manager.validate_order(order, current_price)
    if not result.passed:
        logger.warning("Order rejected: %s", result.reason)
        return

    # Order passed all checks
    await gateway.submit_order(order)

Architecture:
    Strategy -> RiskManager.validate_order() -> ExecutionEngine -> Gateway

    ALL orders MUST pass through RiskManager. This is mandatory.
"""

from libra.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from libra.risk.limits import (
    RiskLimits,
    SymbolLimits,
)
from libra.risk.manager import (
    RiskCheckResult,
    RiskManager,
    TradingState,
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


__all__ = [
    # Manager
    "RiskManager",
    "RiskCheckResult",
    "TradingState",
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
]

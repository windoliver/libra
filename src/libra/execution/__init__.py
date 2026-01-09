"""
Execution Algorithm Framework.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

This module provides execution algorithms for strategic order execution,
minimizing market impact by splitting large orders into smaller child orders.

Algorithms Available:
- TWAP: Time-Weighted Average Price (equal slices over time)
- VWAP: Volume-Weighted Average Price (slices proportional to volume)
- Iceberg: Hidden orders showing only visible portion
- POV: Percentage of Volume (participation rate)

Quick Start:
    from libra.execution import create_algorithm

    # Create TWAP algorithm
    twap = create_algorithm(
        "twap",
        horizon_secs=300,    # 5 minutes
        interval_secs=30,    # 30 second intervals
    )
    twap.set_execution_client(client)
    progress = await twap.execute(order)

    # Or use direct imports
    from libra.execution import TWAPAlgorithm, TWAPConfig

    config = TWAPConfig(horizon_secs=120, interval_secs=10)
    twap = TWAPAlgorithm(config, execution_client=client)

    # Use ExecutionEngine for algorithm-based order routing
    from libra.execution import ExecutionEngine

    engine = ExecutionEngine(message_bus=bus, clock=clock, execution_client=client)
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("100"),
        exec_algorithm="twap",
        exec_algorithm_params={"horizon_secs": 300},
    )
    progress = await engine.submit_order(order)

References:
- NautilusTrader: https://nautilustrader.io/docs/nightly/concepts/execution/
- QuantConnect Lean: VolumeWeightedAveragePriceExecutionModel
"""

# Core protocol and base class
from libra.execution.algorithm import (
    AlgorithmState,
    BaseExecAlgorithm,
    ChildOrder,
    ExecAlgorithm,
    ExecutionMetrics,
    ExecutionProgress,
)

# Adaptive execution features
from libra.execution.adaptive import (
    AdaptiveConfig,
    AdaptiveMixin,
    AdaptiveState,
    MarketConditions,
    UrgencyLevel,
    create_adaptive_config,
    urgency_from_level,
)

# Execution Engine
from libra.execution.engine import (
    ActiveExecution,
    AlgorithmExecutionError,
    AlgorithmNotFoundError,
    ExecutionEngine,
    ExecutionEngineConfig,
    ExecutionEngineStats,
    OrderDeniedError,
    create_execution_engine,
)

# Iceberg algorithm
from libra.execution.iceberg import (
    IcebergAlgorithm,
    IcebergConfig,
    create_iceberg,
)

# TCA Metrics
from libra.execution.metrics import (
    AggregatedTCA,
    ExecutionTCA,
    FillRecord,
    create_tca,
)

# POV algorithm
from libra.execution.pov import (
    POVAlgorithm,
    POVConfig,
    create_pov,
)

# Registry for algorithm discovery
from libra.execution.registry import (
    AlgorithmRegistry,
    create_algorithm,
    get_algorithm_registry,
    list_algorithms,
)

# TWAP algorithm
from libra.execution.twap import (
    TWAPAlgorithm,
    TWAPConfig,
    create_twap,
)

# VWAP algorithm
from libra.execution.vwap import (
    VolumeProfile,
    VWAPAlgorithm,
    VWAPConfig,
    create_vwap,
)


__all__ = [
    # Protocol and base
    "ExecAlgorithm",
    "BaseExecAlgorithm",
    "AlgorithmState",
    "ChildOrder",
    "ExecutionProgress",
    "ExecutionMetrics",
    # Execution Engine
    "ExecutionEngine",
    "ExecutionEngineConfig",
    "ExecutionEngineStats",
    "ActiveExecution",
    "OrderDeniedError",
    "AlgorithmNotFoundError",
    "AlgorithmExecutionError",
    "create_execution_engine",
    # TWAP
    "TWAPAlgorithm",
    "TWAPConfig",
    "create_twap",
    # VWAP
    "VWAPAlgorithm",
    "VWAPConfig",
    "VolumeProfile",
    "create_vwap",
    # Iceberg
    "IcebergAlgorithm",
    "IcebergConfig",
    "create_iceberg",
    # POV
    "POVAlgorithm",
    "POVConfig",
    "create_pov",
    # Adaptive
    "AdaptiveMixin",
    "AdaptiveConfig",
    "AdaptiveState",
    "MarketConditions",
    "UrgencyLevel",
    "create_adaptive_config",
    "urgency_from_level",
    # TCA Metrics
    "ExecutionTCA",
    "AggregatedTCA",
    "FillRecord",
    "create_tca",
    # Registry
    "AlgorithmRegistry",
    "get_algorithm_registry",
    "create_algorithm",
    "list_algorithms",
]

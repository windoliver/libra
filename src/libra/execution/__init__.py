"""
Execution Algorithm Framework.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

This module provides execution algorithms for strategic order execution,
minimizing market impact by splitting large orders into smaller child orders.

Algorithms Available:
- TWAP: Time-Weighted Average Price (equal slices over time)
- VWAP: Volume-Weighted Average Price (slices proportional to volume)
- Iceberg: Hidden orders showing only visible portion

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

# Iceberg algorithm
from libra.execution.iceberg import (
    IcebergAlgorithm,
    IcebergConfig,
    create_iceberg,
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
    # Registry
    "AlgorithmRegistry",
    "get_algorithm_registry",
    "create_algorithm",
    "list_algorithms",
]

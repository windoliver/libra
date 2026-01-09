"""
Example strategies demonstrating the Strategy Protocol.

Available examples:
- SMACrossStrategy: Simple Moving Average crossover (signal generation only)
- SMACrossLiveStrategy: SMA crossover with live order execution (Actor-based)
- ScheduledRebalanceStrategy: Time-based portfolio rebalancing (Issue #24)
"""

from libra.strategies.examples.scheduled_rebalance import (
    RebalanceConfig,
    ScheduledRebalanceStrategy,
)
from libra.strategies.examples.sma_cross import SMACrossConfig, SMACrossStrategy
from libra.strategies.examples.sma_cross_live import (
    SMACrossLiveConfig,
    SMACrossLiveStrategy,
)


__all__ = [
    "RebalanceConfig",
    "SMACrossConfig",
    "SMACrossLiveConfig",
    "SMACrossLiveStrategy",
    "SMACrossStrategy",
    "ScheduledRebalanceStrategy",
]

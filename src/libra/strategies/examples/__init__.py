"""
Example strategies demonstrating the Strategy Protocol.

Available examples:
- SMACrossStrategy: Simple Moving Average crossover (signal generation only)
- SMACrossLiveStrategy: SMA crossover with live order execution (Actor-based)
"""

from libra.strategies.examples.sma_cross import SMACrossConfig, SMACrossStrategy
from libra.strategies.examples.sma_cross_live import (
    SMACrossLiveConfig,
    SMACrossLiveStrategy,
)


__all__ = [
    "SMACrossConfig",
    "SMACrossLiveConfig",
    "SMACrossLiveStrategy",
    "SMACrossStrategy",
]

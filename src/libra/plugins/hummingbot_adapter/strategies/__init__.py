"""
Market Making Strategies for Hummingbot Adapter (Issue #12).

Standalone implementations of market making algorithms:
- Avellaneda-Stoikov optimal market making
- Pure market making
- Cross-exchange market making (XEMM)
"""

from libra.plugins.hummingbot_adapter.strategies.avellaneda import AvellanedaStoikovStrategy
from libra.plugins.hummingbot_adapter.strategies.pure_mm import PureMarketMakingStrategy
from libra.plugins.hummingbot_adapter.strategies.xemm import CrossExchangeMarketMakingStrategy

__all__ = [
    "AvellanedaStoikovStrategy",
    "PureMarketMakingStrategy",
    "CrossExchangeMarketMakingStrategy",
]

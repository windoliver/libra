"""
Hummingbot Adapter Plugin for LIBRA (Issue #12).

Standalone implementation of Hummingbot-style market making strategies:
- Avellaneda-Stoikov optimal market making
- Pure market making with inventory management
- Cross-exchange market making (XEMM)

This is a lightweight implementation that doesn't require Hummingbot installation.
"""

from libra.plugins.hummingbot_adapter.adapter import HummingbotAdapter
from libra.plugins.hummingbot_adapter.config import HummingbotAdapterConfig
from libra.plugins.hummingbot_adapter.strategies.avellaneda import AvellanedaStoikovStrategy
from libra.plugins.hummingbot_adapter.strategies.pure_mm import PureMarketMakingStrategy
from libra.plugins.hummingbot_adapter.strategies.xemm import CrossExchangeMarketMakingStrategy

__all__ = [
    "HummingbotAdapter",
    "HummingbotAdapterConfig",
    "AvellanedaStoikovStrategy",
    "PureMarketMakingStrategy",
    "CrossExchangeMarketMakingStrategy",
]

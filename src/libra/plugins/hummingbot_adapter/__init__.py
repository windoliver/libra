"""
Hummingbot Adapter Plugin for LIBRA (Issue #12).

Standalone implementation of Hummingbot-style market making strategies:
- Avellaneda-Stoikov optimal market making
- Pure market making with inventory management
- Cross-exchange market making (XEMM)
- DEX gateway support (Uniswap V2/V3)
- DEX arbitrage detection and execution
- Comprehensive performance tracking

This is a lightweight implementation that doesn't require Hummingbot installation.
"""

from libra.plugins.hummingbot_adapter.adapter import HummingbotAdapter
from libra.plugins.hummingbot_adapter.config import HummingbotAdapterConfig
from libra.plugins.hummingbot_adapter.performance import (
    PerformanceStats,
    PerformanceTracker,
    Position,
    PositionSide,
    Trade,
)
from libra.plugins.hummingbot_adapter.strategies.avellaneda import AvellanedaStoikovStrategy
from libra.plugins.hummingbot_adapter.strategies.pure_mm import PureMarketMakingStrategy
from libra.plugins.hummingbot_adapter.strategies.xemm import CrossExchangeMarketMakingStrategy

# DEX imports (optional, requires web3)
try:
    from libra.plugins.hummingbot_adapter.dex import (
        DEXArbitrageStrategy,
        DEXGateway,
        DEXPool,
        DEXQuote,
        DEXSwapResult,
        Token,
        UniswapV2Gateway,
        UniswapV3Gateway,
    )

    _DEX_AVAILABLE = True
except ImportError:
    _DEX_AVAILABLE = False

__all__ = [
    # Adapter
    "HummingbotAdapter",
    "HummingbotAdapterConfig",
    # Strategies
    "AvellanedaStoikovStrategy",
    "PureMarketMakingStrategy",
    "CrossExchangeMarketMakingStrategy",
    # Performance tracking
    "PerformanceTracker",
    "PerformanceStats",
    "Position",
    "PositionSide",
    "Trade",
    # DEX (conditionally available)
    "DEXGateway",
    "DEXPool",
    "DEXQuote",
    "DEXSwapResult",
    "Token",
    "UniswapV2Gateway",
    "UniswapV3Gateway",
    "DEXArbitrageStrategy",
]

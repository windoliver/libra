"""
DEX Gateway Support for Hummingbot Adapter (Issue #12).

Provides integration with decentralized exchanges:
- Uniswap V2/V3
- SushiSwap
- PancakeSwap
- Generic AMM interface

Note: Requires web3 optional dependency.
"""

from libra.plugins.hummingbot_adapter.dex.base import (
    ChainId,
    DEXGateway,
    DEXGatewayConfig,
    DEXPool,
    DEXQuote,
    DEXSwapResult,
    DEXType,
    Token,
    USDC,
    USDT,
    WETH,
)
from libra.plugins.hummingbot_adapter.dex.uniswap import UniswapV2Gateway, UniswapV3Gateway
from libra.plugins.hummingbot_adapter.dex.arbitrage import DEXArbitrageStrategy

__all__ = [
    # Base classes
    "ChainId",
    "DEXGateway",
    "DEXGatewayConfig",
    "DEXPool",
    "DEXQuote",
    "DEXSwapResult",
    "DEXType",
    "Token",
    # Predefined tokens
    "WETH",
    "USDC",
    "USDT",
    # Gateways
    "UniswapV2Gateway",
    "UniswapV3Gateway",
    # Strategies
    "DEXArbitrageStrategy",
]

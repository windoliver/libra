"""
Prediction Market Providers.

Provider implementations for different prediction market platforms:
- Polymarket: Crypto-native, USDC-based prediction market
- Kalshi: Regulated US prediction market
- Metaculus: Reputation-based forecasting platform
- Manifold: Play-money prediction market

Each provider implements the BasePredictionProvider protocol.
"""

from libra.gateways.prediction_market.providers.base import (
    BasePredictionProvider,
    ProviderConfig,
)
from libra.gateways.prediction_market.providers.kalshi import KalshiProvider
from libra.gateways.prediction_market.providers.manifold import ManifoldProvider
from libra.gateways.prediction_market.providers.metaculus import MetaculusProvider
from libra.gateways.prediction_market.providers.polymarket import PolymarketProvider

__all__ = [
    "BasePredictionProvider",
    "ProviderConfig",
    "PolymarketProvider",
    "KalshiProvider",
    "MetaculusProvider",
    "ManifoldProvider",
]

"""
Prediction Market Gateway.

Unified interface for multiple prediction market platforms:
- Polymarket: Crypto-native prediction market on Polygon (USDC)
- Kalshi: CFTC-regulated prediction market (USD)
- Metaculus: Reputation-based forecasting platform
- Manifold Markets: Play-money prediction market (Mana)

Example:
    from libra.gateways.prediction_market import (
        PredictionMarketGateway,
        PredictionMarketCredentials,
    )

    # Initialize with credentials
    credentials = PredictionMarketCredentials(
        polymarket_api_key="your_key",
        kalshi_api_key="your_key",
    )
    gateway = PredictionMarketGateway(credentials=credentials)

    async with gateway:
        # Get markets
        markets = await gateway.get_markets(category="crypto", limit=50)

        # Get quote
        quote = await gateway.get_quote("polymarket", market_id, "yes")

        # Compare across platforms
        quotes = await gateway.get_cross_platform_quotes(market_id, "yes")

        # Trading (requires credentials)
        from libra.gateways.prediction_market.protocol import (
            PredictionOrder,
            PredictionOrderSide,
            PredictionOrderType,
        )

        order = PredictionOrder(
            market_id=market_id,
            outcome_id="yes",
            platform="polymarket",
            side=PredictionOrderSide.BUY,
            order_type=PredictionOrderType.LIMIT,
            size=Decimal("100"),
            price=Decimal("0.65"),
        )
        result = await gateway.submit_order("polymarket", order)
"""

from libra.gateways.prediction_market.gateway import (
    PredictionMarketCredentials,
    PredictionMarketGateway,
    PredictionMarketGatewayError,
    PredictionMarketNotInstalledError,
    ProviderStatus,
)
from libra.gateways.prediction_market.protocol import (
    MarketStatus,
    MarketType,
    Outcome,
    OutcomeType,
    PredictionMarket,
    PredictionMarketCapabilities,
    PredictionMarketError,
    PredictionOrder,
    PredictionOrderBook,
    PredictionOrderBookLevel,
    PredictionOrderResult,
    PredictionOrderSide,
    PredictionOrderStatus,
    PredictionOrderType,
    PredictionPosition,
    PredictionQuote,
)

__all__ = [
    # Gateway
    "PredictionMarketGateway",
    "PredictionMarketCredentials",
    "PredictionMarketGatewayError",
    "PredictionMarketNotInstalledError",
    "ProviderStatus",
    # Protocol
    "MarketStatus",
    "MarketType",
    "OutcomeType",
    "Outcome",
    "PredictionMarket",
    "PredictionMarketCapabilities",
    "PredictionQuote",
    "PredictionOrderBook",
    "PredictionOrderBookLevel",
    "PredictionOrder",
    "PredictionOrderResult",
    "PredictionOrderSide",
    "PredictionOrderType",
    "PredictionOrderStatus",
    "PredictionPosition",
    "PredictionMarketError",
]

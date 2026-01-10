"""
Prediction Market Fetchers.

TET (Transform-Extract-Transform) pipeline fetchers for prediction markets.
Following the pattern from Issue #27: Provider/Fetcher Pattern.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from libra.gateways.fetcher import GatewayFetcher
from libra.gateways.prediction_market.protocol import (
    MarketStatus,
    PredictionMarket,
    PredictionOrderBook,
    PredictionQuote,
)
from libra.gateways.prediction_market.queries import (
    MarketDetailQuery,
    PredictionMarketQuery,
    PredictionOrderBookQuery,
    PredictionQuoteQuery,
)


logger = logging.getLogger(__name__)


# Type alias for provider registry
ProviderRegistry = dict[str, Any]  # str -> BasePredictionProvider


class PredictionMarketFetcher(GatewayFetcher[PredictionMarketQuery, list[PredictionMarket]]):
    """
    Fetcher for prediction markets.

    Uses the TET pipeline to fetch markets from providers.

    Example:
        fetcher = PredictionMarketFetcher(providers)
        markets = await fetcher.fetch(
            provider="polymarket",
            category="crypto",
            status=MarketStatus.OPEN,
            limit=50,
        )
    """

    def __init__(self, providers: ProviderRegistry) -> None:
        """
        Initialize fetcher with provider registry.

        Args:
            providers: Dict mapping provider names to provider instances
        """
        self._providers = providers

    def transform_query(self, params: dict[str, Any]) -> PredictionMarketQuery:
        """Transform raw params to typed query."""
        status = params.get("status")
        if isinstance(status, str):
            status = MarketStatus(status)

        return PredictionMarketQuery(
            provider=params.get("provider"),
            category=params.get("category"),
            status=status,
            search=params.get("search"),
            limit=params.get("limit", 100),
            offset=params.get("offset", 0),
            sort_by=params.get("sort_by"),
            sort_order=params.get("sort_order", "desc"),
        )

    async def extract_data(self, query: PredictionMarketQuery, **kwargs: Any) -> list[Any]:
        """Extract markets from provider(s)."""
        all_markets: list[Any] = []

        if query.provider:
            # Single provider
            providers_to_query = [query.provider]
        else:
            # All providers
            providers_to_query = list(self._providers.keys())

        for provider_name in providers_to_query:
            provider = self._providers.get(provider_name)
            if not provider:
                logger.warning(f"Provider {provider_name} not found")
                continue

            try:
                markets = await provider.get_markets(
                    category=query.category,
                    status=query.status,
                    search=query.search,
                    limit=query.limit,
                    offset=query.offset,
                )
                all_markets.extend(markets)
            except Exception as e:
                logger.error(f"Failed to fetch from {provider_name}: {e}")

        return all_markets

    def transform_data(
        self, query: PredictionMarketQuery, raw: list[Any]
    ) -> list[PredictionMarket]:
        """Transform raw data (already PredictionMarket objects)."""
        # Data is already in correct format from providers
        markets = [m for m in raw if isinstance(m, PredictionMarket)]

        # Apply sorting if specified
        if query.sort_by:
            reverse = query.sort_order == "desc"
            if query.sort_by == "volume":
                markets.sort(key=lambda m: m.volume, reverse=reverse)
            elif query.sort_by == "liquidity":
                markets.sort(key=lambda m: m.liquidity, reverse=reverse)
            elif query.sort_by == "volume_24h":
                markets.sort(key=lambda m: m.volume_24h, reverse=reverse)

        return markets[:query.limit]


class MarketDetailFetcher(GatewayFetcher[MarketDetailQuery, PredictionMarket | None]):
    """
    Fetcher for individual market details.

    Example:
        fetcher = MarketDetailFetcher(providers)
        market = await fetcher.fetch(
            market_id="0x123abc",
            provider="polymarket",
        )
    """

    def __init__(self, providers: ProviderRegistry) -> None:
        self._providers = providers

    def transform_query(self, params: dict[str, Any]) -> MarketDetailQuery:
        """Transform raw params to typed query."""
        return MarketDetailQuery(
            market_id=params["market_id"],
            provider=params["provider"],
        )

    async def extract_data(
        self, query: MarketDetailQuery, **kwargs: Any
    ) -> PredictionMarket | None:
        """Extract market from provider."""
        provider = self._providers.get(query.provider)
        if not provider:
            raise ValueError(f"Provider {query.provider} not found")

        return await provider.get_market(query.market_id)

    def transform_data(
        self, query: MarketDetailQuery, raw: PredictionMarket | None
    ) -> PredictionMarket | None:
        """Transform data (pass through)."""
        return raw


class PredictionQuoteFetcher(GatewayFetcher[PredictionQuoteQuery, PredictionQuote | None]):
    """
    Fetcher for prediction market quotes.

    Example:
        fetcher = PredictionQuoteFetcher(providers)
        quote = await fetcher.fetch(
            market_id="0x123abc",
            outcome_id="yes",
            provider="polymarket",
        )
    """

    def __init__(self, providers: ProviderRegistry) -> None:
        self._providers = providers

    def transform_query(self, params: dict[str, Any]) -> PredictionQuoteQuery:
        """Transform raw params to typed query."""
        return PredictionQuoteQuery(
            market_id=params["market_id"],
            outcome_id=params["outcome_id"],
            provider=params["provider"],
        )

    async def extract_data(
        self, query: PredictionQuoteQuery, **kwargs: Any
    ) -> PredictionQuote | None:
        """Extract quote from provider."""
        provider = self._providers.get(query.provider)
        if not provider:
            raise ValueError(f"Provider {query.provider} not found")

        return await provider.get_quote(query.market_id, query.outcome_id)

    def transform_data(
        self, query: PredictionQuoteQuery, raw: PredictionQuote | None
    ) -> PredictionQuote | None:
        """Transform data (pass through)."""
        return raw


class PredictionOrderBookFetcher(
    GatewayFetcher[PredictionOrderBookQuery, PredictionOrderBook | None]
):
    """
    Fetcher for prediction market order books.

    Example:
        fetcher = PredictionOrderBookFetcher(providers)
        orderbook = await fetcher.fetch(
            market_id="0x123abc",
            outcome_id="yes",
            provider="polymarket",
            depth=20,
        )
    """

    def __init__(self, providers: ProviderRegistry) -> None:
        self._providers = providers

    def transform_query(self, params: dict[str, Any]) -> PredictionOrderBookQuery:
        """Transform raw params to typed query."""
        return PredictionOrderBookQuery(
            market_id=params["market_id"],
            outcome_id=params["outcome_id"],
            provider=params["provider"],
            depth=params.get("depth", 20),
        )

    async def extract_data(
        self, query: PredictionOrderBookQuery, **kwargs: Any
    ) -> PredictionOrderBook | None:
        """Extract order book from provider."""
        provider = self._providers.get(query.provider)
        if not provider:
            raise ValueError(f"Provider {query.provider} not found")

        if not provider.capabilities.supports_orderbook:
            return None

        return await provider.get_orderbook(
            query.market_id, query.outcome_id, query.depth
        )

    def transform_data(
        self, query: PredictionOrderBookQuery, raw: PredictionOrderBook | None
    ) -> PredictionOrderBook | None:
        """Transform data (pass through)."""
        return raw


class CrossPlatformQuoteFetcher(
    GatewayFetcher[PredictionQuoteQuery, dict[str, PredictionQuote | None]]
):
    """
    Fetcher for quotes across multiple platforms.

    Useful for comparing prices and finding arbitrage opportunities.

    Example:
        fetcher = CrossPlatformQuoteFetcher(providers)
        quotes = await fetcher.fetch(
            market_id="bitcoin-100k",
            outcome_id="yes",
        )
        # Returns: {"polymarket": Quote(...), "kalshi": Quote(...), ...}
    """

    def __init__(self, providers: ProviderRegistry) -> None:
        self._providers = providers

    def transform_query(self, params: dict[str, Any]) -> PredictionQuoteQuery:
        """Transform raw params to typed query."""
        return PredictionQuoteQuery(
            market_id=params["market_id"],
            outcome_id=params["outcome_id"],
            provider=params.get("provider", ""),
        )

    async def extract_data(
        self, query: PredictionQuoteQuery, **kwargs: Any
    ) -> dict[str, PredictionQuote | None]:
        """Extract quotes from all providers."""
        quotes: dict[str, PredictionQuote | None] = {}

        for provider_name, provider in self._providers.items():
            try:
                # Search for similar market
                quote = await provider.get_quote(query.market_id, query.outcome_id)
                quotes[provider_name] = quote
            except Exception as e:
                logger.debug(f"No quote from {provider_name}: {e}")
                quotes[provider_name] = None

        return quotes

    def transform_data(
        self, query: PredictionQuoteQuery, raw: dict[str, PredictionQuote | None]
    ) -> dict[str, PredictionQuote | None]:
        """Transform data (pass through)."""
        return raw

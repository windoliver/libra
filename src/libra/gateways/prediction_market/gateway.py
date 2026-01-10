"""
Prediction Market Gateway.

High-level gateway providing unified access to multiple prediction market platforms:
- Polymarket (crypto, USDC-based)
- Kalshi (regulated, USD-based)
- Metaculus (reputation-based forecasting)
- Manifold Markets (play-money)

This gateway follows the pattern established by OpenBB Gateway (Issue #28)
and uses the TET pipeline from Issue #27.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from libra.gateways.prediction_market.fetchers import (
    CrossPlatformQuoteFetcher,
    MarketDetailFetcher,
    PredictionMarketFetcher,
    PredictionOrderBookFetcher,
    PredictionQuoteFetcher,
)
from libra.gateways.prediction_market.protocol import (
    MarketStatus,
    PredictionMarket,
    PredictionMarketCapabilities,
    PredictionOrder,
    PredictionOrderBook,
    PredictionOrderResult,
    PredictionPosition,
    PredictionQuote,
    ProviderNotAvailableError,
)
from libra.gateways.prediction_market.providers.base import (
    BasePredictionProvider,
    ProviderConfig,
)
from libra.gateways.prediction_market.providers.kalshi import KalshiProvider
from libra.gateways.prediction_market.providers.manifold import ManifoldProvider
from libra.gateways.prediction_market.providers.metaculus import MetaculusProvider
from libra.gateways.prediction_market.providers.polymarket import PolymarketProvider


logger = logging.getLogger(__name__)


class PredictionMarketGatewayError(Exception):
    """Base exception for prediction market gateway errors."""


class PredictionMarketNotInstalledError(PredictionMarketGatewayError):
    """Raised when required dependencies are not installed."""

    def __init__(self, message: str | None = None) -> None:
        default_message = (
            "HTTP client not available. Install with:\n"
            "  pip install httpx"
        )
        super().__init__(message or default_message)


@dataclass
class ProviderStatus:
    """Status of a prediction market provider."""

    name: str
    available: bool
    connected: bool
    has_credentials: bool
    capabilities: PredictionMarketCapabilities | None = None
    last_error: str | None = None


@dataclass
class PredictionMarketCredentials:
    """Credentials for prediction market providers."""

    # Polymarket
    polymarket_api_key: str | None = None
    polymarket_api_secret: str | None = None
    polymarket_private_key: str | None = None  # Polygon wallet key

    # Kalshi
    kalshi_api_key: str | None = None
    kalshi_private_key: str | None = None  # RSA private key

    # Manifold
    manifold_api_key: str | None = None

    # Metaculus
    metaculus_api_token: str | None = None


@dataclass
class PredictionMarketGateway:
    """
    Prediction Market Gateway for LIBRA.

    Provides unified access to multiple prediction market platforms through
    a consistent interface.

    Example:
        gateway = PredictionMarketGateway()
        await gateway.connect()

        # Get markets across platforms
        markets = await gateway.get_markets(category="crypto", limit=50)

        # Get specific market
        market = await gateway.get_market("polymarket", market_id)

        # Get quote
        quote = await gateway.get_quote("polymarket", market_id, "yes")

        # Compare prices across platforms
        quotes = await gateway.get_cross_platform_quotes(market_id, "yes")

        # Trading (requires credentials)
        result = await gateway.submit_order("polymarket", order)

        await gateway.disconnect()
    """

    credentials: PredictionMarketCredentials = field(
        default_factory=PredictionMarketCredentials
    )
    default_provider: str = "polymarket"

    # Provider enablement
    enable_polymarket: bool = True
    enable_kalshi: bool = True
    enable_metaculus: bool = True
    enable_manifold: bool = True

    # Internal state
    _connected: bool = field(default=False, init=False)
    _providers: dict[str, BasePredictionProvider] = field(default_factory=dict, init=False)

    # Fetchers (lazy-initialized)
    _market_fetcher: PredictionMarketFetcher | None = field(default=None, init=False)
    _market_detail_fetcher: MarketDetailFetcher | None = field(default=None, init=False)
    _quote_fetcher: PredictionQuoteFetcher | None = field(default=None, init=False)
    _orderbook_fetcher: PredictionOrderBookFetcher | None = field(default=None, init=False)
    _cross_platform_fetcher: CrossPlatformQuoteFetcher | None = field(default=None, init=False)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """
        Connect to prediction market providers.

        Initializes enabled providers and establishes connections.

        Raises:
            PredictionMarketNotInstalledError: If httpx is not installed.
        """
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise PredictionMarketNotInstalledError() from e

        # Initialize providers
        if self.enable_polymarket:
            await self._init_polymarket()

        if self.enable_kalshi:
            await self._init_kalshi()

        if self.enable_metaculus:
            await self._init_metaculus()

        if self.enable_manifold:
            await self._init_manifold()

        # Initialize fetchers
        self._market_fetcher = PredictionMarketFetcher(self._providers)
        self._market_detail_fetcher = MarketDetailFetcher(self._providers)
        self._quote_fetcher = PredictionQuoteFetcher(self._providers)
        self._orderbook_fetcher = PredictionOrderBookFetcher(self._providers)
        self._cross_platform_fetcher = CrossPlatformQuoteFetcher(self._providers)

        self._connected = True
        logger.info(
            f"Prediction market gateway connected with providers: {list(self._providers.keys())}"
        )

    async def disconnect(self) -> None:
        """Disconnect from all providers."""
        for provider in self._providers.values():
            try:
                await provider.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {provider.name}: {e}")

        self._providers.clear()
        self._connected = False
        self._market_fetcher = None
        self._market_detail_fetcher = None
        self._quote_fetcher = None
        self._orderbook_fetcher = None
        self._cross_platform_fetcher = None

        logger.info("Prediction market gateway disconnected")

    async def __aenter__(self) -> PredictionMarketGateway:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if gateway is connected."""
        return self._connected

    @property
    def available_providers(self) -> list[str]:
        """List of connected provider names."""
        return list(self._providers.keys())

    def _ensure_connected(self) -> None:
        """Ensure gateway is connected."""
        if not self._connected:
            raise RuntimeError(
                "Prediction market gateway not connected. Call connect() first."
            )

    # =========================================================================
    # Market Data
    # =========================================================================

    async def get_markets(
        self,
        provider: str | None = None,
        category: str | None = None,
        status: MarketStatus | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PredictionMarket]:
        """
        Get prediction markets.

        Args:
            provider: Specific provider (None = all providers)
            category: Filter by category (crypto, politics, sports, etc.)
            status: Filter by status (open, closed, resolved)
            search: Search term
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of markets across platforms
        """
        self._ensure_connected()
        assert self._market_fetcher is not None

        return await self._market_fetcher.fetch(
            provider=provider,
            category=category,
            status=status,
            search=search,
            limit=limit,
            offset=offset,
        )

    async def get_market(
        self, provider: str, market_id: str
    ) -> PredictionMarket | None:
        """
        Get a specific market.

        Args:
            provider: Provider name
            market_id: Market identifier

        Returns:
            Market or None if not found
        """
        self._ensure_connected()
        assert self._market_detail_fetcher is not None

        return await self._market_detail_fetcher.fetch(
            market_id=market_id,
            provider=provider,
        )

    async def search_markets(
        self,
        query: str,
        providers: list[str] | None = None,
        limit: int = 50,
    ) -> list[PredictionMarket]:
        """
        Search for markets across platforms.

        Args:
            query: Search query
            providers: Specific providers to search (None = all)
            limit: Maximum results

        Returns:
            List of matching markets
        """
        self._ensure_connected()
        assert self._market_fetcher is not None

        all_markets: list[PredictionMarket] = []
        target_providers = providers or list(self._providers.keys())

        for provider_name in target_providers:
            try:
                markets = await self._market_fetcher.fetch(
                    provider=provider_name,
                    search=query,
                    limit=limit,
                )
                all_markets.extend(markets)
            except Exception as e:
                logger.warning(f"Search failed for {provider_name}: {e}")

        return all_markets[:limit]

    # =========================================================================
    # Quotes and Order Books
    # =========================================================================

    async def get_quote(
        self, provider: str, market_id: str, outcome_id: str
    ) -> PredictionQuote | None:
        """
        Get quote for a market outcome.

        Args:
            provider: Provider name
            market_id: Market identifier
            outcome_id: Outcome identifier (e.g., "yes", "no")

        Returns:
            Quote or None if not available
        """
        self._ensure_connected()
        assert self._quote_fetcher is not None

        return await self._quote_fetcher.fetch(
            market_id=market_id,
            outcome_id=outcome_id,
            provider=provider,
        )

    async def get_orderbook(
        self, provider: str, market_id: str, outcome_id: str, depth: int = 20
    ) -> PredictionOrderBook | None:
        """
        Get order book for a market outcome.

        Args:
            provider: Provider name
            market_id: Market identifier
            outcome_id: Outcome identifier
            depth: Number of levels

        Returns:
            Order book or None if not supported
        """
        self._ensure_connected()
        assert self._orderbook_fetcher is not None

        return await self._orderbook_fetcher.fetch(
            market_id=market_id,
            outcome_id=outcome_id,
            provider=provider,
            depth=depth,
        )

    async def get_cross_platform_quotes(
        self, market_id: str, outcome_id: str
    ) -> dict[str, PredictionQuote | None]:
        """
        Get quotes for a market across all platforms.

        Useful for comparing prices and finding arbitrage.

        Args:
            market_id: Market identifier (may vary by platform)
            outcome_id: Outcome identifier

        Returns:
            Dict mapping provider names to quotes
        """
        self._ensure_connected()
        assert self._cross_platform_fetcher is not None

        return await self._cross_platform_fetcher.fetch(
            market_id=market_id,
            outcome_id=outcome_id,
        )

    # =========================================================================
    # Trading
    # =========================================================================

    async def get_positions(
        self, provider: str, market_id: str | None = None
    ) -> list[PredictionPosition]:
        """
        Get user positions.

        Args:
            provider: Provider name
            market_id: Optional market filter

        Returns:
            List of positions
        """
        self._ensure_connected()
        self._ensure_provider(provider)

        return await self._providers[provider].get_positions(market_id)

    async def get_balance(self, provider: str) -> dict[str, Decimal]:
        """
        Get account balance.

        Args:
            provider: Provider name

        Returns:
            Dict mapping currency to balance
        """
        self._ensure_connected()
        self._ensure_provider(provider)

        return await self._providers[provider].get_balance()

    async def submit_order(
        self, provider: str, order: PredictionOrder
    ) -> PredictionOrderResult:
        """
        Submit an order.

        Args:
            provider: Provider name
            order: Order to submit

        Returns:
            Order result

        Raises:
            ProviderNotAvailableError: If provider doesn't support trading
        """
        self._ensure_connected()
        self._ensure_provider(provider)

        prov = self._providers[provider]
        if not prov.capabilities.supports_trading:
            raise ProviderNotAvailableError(
                f"{provider} does not support trading"
            )

        return await prov.submit_order(order)

    async def cancel_order(self, provider: str, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            provider: Provider name
            order_id: Order to cancel

        Returns:
            True if cancelled
        """
        self._ensure_connected()
        self._ensure_provider(provider)

        return await self._providers[provider].cancel_order(order_id)

    async def get_open_orders(
        self, provider: str, market_id: str | None = None
    ) -> list[PredictionOrderResult]:
        """
        Get open orders.

        Args:
            provider: Provider name
            market_id: Optional market filter

        Returns:
            List of open orders
        """
        self._ensure_connected()
        self._ensure_provider(provider)

        return await self._providers[provider].get_open_orders(market_id)

    # =========================================================================
    # Provider Status
    # =========================================================================

    def get_provider_status(self) -> list[ProviderStatus]:
        """
        Get status of all providers.

        Returns:
            List of provider statuses
        """
        statuses = []

        provider_configs = [
            ("polymarket", self.enable_polymarket, self.credentials.polymarket_api_key),
            ("kalshi", self.enable_kalshi, self.credentials.kalshi_api_key),
            ("metaculus", self.enable_metaculus, self.credentials.metaculus_api_token),
            ("manifold", self.enable_manifold, self.credentials.manifold_api_key),
        ]

        for name, enabled, creds in provider_configs:
            provider = self._providers.get(name)
            statuses.append(
                ProviderStatus(
                    name=name,
                    available=enabled,
                    connected=provider.is_connected if provider else False,
                    has_credentials=creds is not None,
                    capabilities=provider.capabilities if provider else None,
                )
            )

        return statuses

    def get_provider(self, name: str) -> BasePredictionProvider | None:
        """
        Get a specific provider instance.

        Args:
            name: Provider name

        Returns:
            Provider instance or None
        """
        return self._providers.get(name)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _ensure_provider(self, provider: str) -> None:
        """Ensure provider is available."""
        if provider not in self._providers:
            raise ProviderNotAvailableError(
                f"Provider '{provider}' not available. "
                f"Available: {list(self._providers.keys())}"
            )

    async def _init_polymarket(self) -> None:
        """Initialize Polymarket provider."""
        try:
            config = ProviderConfig(
                api_key=self.credentials.polymarket_api_key,
                api_secret=self.credentials.polymarket_api_secret,
                private_key=self.credentials.polymarket_private_key,
            )
            provider = PolymarketProvider(config)
            await provider.connect()
            self._providers["polymarket"] = provider
            logger.info("Polymarket provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Polymarket: {e}")

    async def _init_kalshi(self) -> None:
        """Initialize Kalshi provider."""
        try:
            config = ProviderConfig(
                api_key=self.credentials.kalshi_api_key,
                private_key=self.credentials.kalshi_private_key,
            )
            provider = KalshiProvider(config)
            await provider.connect()
            self._providers["kalshi"] = provider
            logger.info("Kalshi provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kalshi: {e}")

    async def _init_metaculus(self) -> None:
        """Initialize Metaculus provider."""
        try:
            config = ProviderConfig(
                api_key=self.credentials.metaculus_api_token,
            )
            provider = MetaculusProvider(config)
            await provider.connect()
            self._providers["metaculus"] = provider
            logger.info("Metaculus provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Metaculus: {e}")

    async def _init_manifold(self) -> None:
        """Initialize Manifold provider."""
        try:
            config = ProviderConfig(
                api_key=self.credentials.manifold_api_key,
            )
            provider = ManifoldProvider(config)
            await provider.connect()
            self._providers["manifold"] = provider
            logger.info("Manifold provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Manifold: {e}")

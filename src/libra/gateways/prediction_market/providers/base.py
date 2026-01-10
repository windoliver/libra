"""
Base Prediction Market Provider.

Abstract base class defining the interface for all prediction market providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from libra.gateways.prediction_market.protocol import (
    MarketStatus,
    PredictionMarket,
    PredictionMarketCapabilities,
    PredictionOrder,
    PredictionOrderBook,
    PredictionOrderResult,
    PredictionPosition,
    PredictionQuote,
)


logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a prediction market provider."""

    # API credentials
    api_key: str | None = None
    api_secret: str | None = None
    private_key: str | None = None  # For blockchain-based platforms

    # Connection settings
    base_url: str | None = None
    timeout: float = 30.0
    max_retries: int = 3

    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_buffer: float = 0.9  # Use 90% of limit

    # Additional settings
    testnet: bool = False  # Use testnet/sandbox
    proxy: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimiter:
    """Simple rate limiter for API calls."""

    max_requests: int
    window_seconds: float = 60.0
    _requests: list[float] = field(default_factory=list)

    async def acquire(self) -> None:
        """Acquire a rate limit slot (blocks if necessary)."""
        import asyncio
        import time

        now = time.time()
        # Remove old requests outside the window
        self._requests = [t for t in self._requests if now - t < self.window_seconds]

        if len(self._requests) >= self.max_requests:
            # Wait until oldest request expires
            sleep_time = self.window_seconds - (now - self._requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self._requests = self._requests[1:]

        self._requests.append(time.time())


class BasePredictionProvider(ABC):
    """
    Abstract base class for prediction market providers.

    All prediction market providers must implement this interface.

    Example:
        class MyProvider(BasePredictionProvider):
            async def connect(self) -> None:
                self._client = httpx.AsyncClient()
                self._connected = True

            async def get_markets(self, **kwargs) -> list[PredictionMarket]:
                response = await self._client.get(f"{self.base_url}/markets")
                return self._parse_markets(response.json())
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """
        Initialize provider.

        Args:
            config: Provider configuration
        """
        self._config = config or ProviderConfig()
        self._connected = False
        self._client: Any = None  # httpx.AsyncClient when connected
        self._rate_limiter: RateLimiter | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'polymarket', 'kalshi')."""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base API URL."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> PredictionMarketCapabilities:
        """Provider capabilities."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self._connected

    @property
    def config(self) -> ProviderConfig:
        """Provider configuration."""
        return self._config

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """
        Connect to the provider.

        Sets up HTTP client and verifies connectivity.
        """
        if self._connected:
            return

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for prediction market providers. "
                "Install with: pip install httpx"
            ) from e

        self._client = httpx.AsyncClient(
            timeout=self._config.timeout,
            headers=self._get_default_headers(),
        )

        self._rate_limiter = RateLimiter(
            max_requests=int(
                self._config.rate_limit_per_minute * self._config.rate_limit_buffer
            ),
            window_seconds=60.0,
        )

        # Verify connectivity
        try:
            await self._verify_connection()
            self._connected = True
            logger.info(f"{self.name} provider connected")
        except Exception as e:
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to {self.name}: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from the provider."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info(f"{self.name} provider disconnected")

    async def __aenter__(self) -> BasePredictionProvider:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # =========================================================================
    # Abstract Methods - Data
    # =========================================================================

    @abstractmethod
    async def get_markets(
        self,
        category: str | None = None,
        status: MarketStatus | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PredictionMarket]:
        """
        Get list of markets.

        Args:
            category: Filter by category
            status: Filter by status
            search: Search term
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of markets
        """
        ...

    @abstractmethod
    async def get_market(self, market_id: str) -> PredictionMarket | None:
        """
        Get a specific market.

        Args:
            market_id: Market identifier

        Returns:
            Market or None if not found
        """
        ...

    @abstractmethod
    async def get_quote(
        self, market_id: str, outcome_id: str
    ) -> PredictionQuote | None:
        """
        Get quote for a market outcome.

        Args:
            market_id: Market identifier
            outcome_id: Outcome identifier

        Returns:
            Quote or None if not available
        """
        ...

    # =========================================================================
    # Optional Methods - Override in providers that support them
    # =========================================================================

    async def get_orderbook(
        self, market_id: str, outcome_id: str, depth: int = 20
    ) -> PredictionOrderBook | None:
        """
        Get order book for a market outcome.

        Args:
            market_id: Market identifier
            outcome_id: Outcome identifier
            depth: Number of levels

        Returns:
            Order book or None if not supported
        """
        if not self.capabilities.supports_orderbook:
            return None
        raise NotImplementedError(f"{self.name} does not implement get_orderbook")

    async def get_positions(
        self, market_id: str | None = None
    ) -> list[PredictionPosition]:
        """
        Get user positions.

        Args:
            market_id: Optional filter by market

        Returns:
            List of positions
        """
        if not self.capabilities.supports_positions:
            return []
        raise NotImplementedError(f"{self.name} does not implement get_positions")

    async def get_balance(self) -> dict[str, Decimal]:
        """
        Get account balance.

        Returns:
            Dict mapping currency to balance
        """
        if not self.capabilities.supports_trading:
            return {}
        raise NotImplementedError(f"{self.name} does not implement get_balance")

    # =========================================================================
    # Trading Methods (Phase 3)
    # =========================================================================

    async def submit_order(self, order: PredictionOrder) -> PredictionOrderResult:
        """
        Submit an order.

        Args:
            order: Order to submit

        Returns:
            Order result

        Raises:
            NotImplementedError: If trading not supported
        """
        if not self.capabilities.supports_trading:
            raise NotImplementedError(f"{self.name} does not support trading")
        raise NotImplementedError(f"{self.name} does not implement submit_order")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order to cancel

        Returns:
            True if cancelled

        Raises:
            NotImplementedError: If trading not supported
        """
        if not self.capabilities.supports_trading:
            raise NotImplementedError(f"{self.name} does not support trading")
        raise NotImplementedError(f"{self.name} does not implement cancel_order")

    async def get_open_orders(
        self, market_id: str | None = None
    ) -> list[PredictionOrderResult]:
        """
        Get open orders.

        Args:
            market_id: Optional filter by market

        Returns:
            List of open orders
        """
        if not self.capabilities.supports_trading:
            return []
        raise NotImplementedError(f"{self.name} does not implement get_open_orders")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_default_headers(self) -> dict[str, str]:
        """Get default HTTP headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "LIBRA-PredictionMarket/1.0",
        }

    async def _verify_connection(self) -> None:
        """Verify connection to provider."""
        # Default implementation - fetch markets with limit 1
        await self.get_markets(limit=1)

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """
        Make an HTTP request with rate limiting.

        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            json: JSON body
            headers: Additional headers

        Returns:
            Response JSON

        Raises:
            httpx.HTTPError: On HTTP errors
        """
        if not self._client:
            raise RuntimeError("Provider not connected")

        if self._rate_limiter:
            await self._rate_limiter.acquire()

        url = f"{self.base_url}{path}"
        response = await self._client.request(
            method=method,
            url=url,
            params=params,
            json=json,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make a GET request."""
        return await self._request("GET", path, params=params, headers=headers)

    async def _post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make a POST request."""
        return await self._request("POST", path, json=json, headers=headers)

    async def _delete(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make a DELETE request."""
        return await self._request("DELETE", path, params=params, headers=headers)

    def _ensure_connected(self) -> None:
        """Ensure provider is connected."""
        if not self._connected:
            raise RuntimeError(f"{self.name} provider not connected. Call connect() first.")

"""
On-Chain Data Providers for Whale Detection.

Tier 2 implementation using blockchain data APIs (optional).

Providers:
- WhaleAlertProvider: Real-time large transaction alerts
- DuneProvider: Custom SQL queries on blockchain data

See: https://github.com/windoliver/libra/issues/38
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.signals.protocol import (
    SignalDirection,
    SignalSource,
    WhaleSignal,
    WhaleSignalType,
    WhaleThresholds,
    WhaleTransaction,
)

if TYPE_CHECKING:
    import httpx


logger = logging.getLogger(__name__)


# =============================================================================
# Base Provider
# =============================================================================


class BaseOnChainProvider(ABC):
    """
    Base class for on-chain data providers.

    All on-chain providers must implement:
    - connect(): Establish connection
    - disconnect(): Clean up resources
    - get_transactions(): Fetch whale transactions
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize provider.

        Args:
            api_key: API key for the service (if required)
        """
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._connected = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base API URL."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to provider."""
        if self._connected:
            return

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for on-chain providers. "
                "Install with: pip install httpx"
            ) from e

        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers=self._get_headers(),
        )
        self._connected = True
        logger.info(f"{self.name} provider connected")

    async def disconnect(self) -> None:
        """Disconnect from provider."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info(f"{self.name} provider disconnected")

    async def __aenter__(self) -> BaseOnChainProvider:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def _get_headers(self) -> dict[str, str]:
        """Get default HTTP headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @abstractmethod
    async def get_transactions(
        self,
        min_value_usd: Decimal = Decimal("1000000"),
        blockchain: str | None = None,
        limit: int = 100,
    ) -> list[WhaleTransaction]:
        """
        Get recent whale transactions.

        Args:
            min_value_usd: Minimum transaction value in USD
            blockchain: Filter by blockchain (None = all)
            limit: Maximum transactions to return

        Returns:
            List of whale transactions
        """
        ...

    async def get_signals(
        self,
        min_value_usd: Decimal = Decimal("1000000"),
        blockchain: str | None = None,
        limit: int = 100,
    ) -> list[WhaleSignal]:
        """
        Get whale signals from transactions.

        Args:
            min_value_usd: Minimum transaction value
            blockchain: Filter by blockchain
            limit: Maximum signals to return

        Returns:
            List of whale signals
        """
        transactions = await self.get_transactions(
            min_value_usd=min_value_usd,
            blockchain=blockchain,
            limit=limit,
        )
        return [tx.to_whale_signal() for tx in transactions]


# =============================================================================
# Whale Alert Provider
# =============================================================================


@dataclass
class WhaleAlertConfig:
    """Configuration for Whale Alert API."""

    api_key: str
    base_url: str = "https://api.whale-alert.io/v1"
    min_value_usd: Decimal = Decimal("1000000")
    blockchains: list[str] = field(default_factory=lambda: [
        "bitcoin", "ethereum", "tron", "ripple", "solana"
    ])


class WhaleAlertProvider(BaseOnChainProvider):
    """
    Whale Alert API provider.

    Tracks large cryptocurrency transactions across multiple blockchains.
    Requires API key from https://whale-alert.io/

    Features:
    - Real-time transaction alerts
    - Exchange inflow/outflow detection
    - Wallet labeling (exchange, whale, unknown)

    Pricing: $29.95/month for standard tier

    Example:
        provider = WhaleAlertProvider(api_key="your_key")
        async with provider:
            transactions = await provider.get_transactions(
                min_value_usd=Decimal("5000000"),
                blockchain="ethereum",
            )
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: WhaleAlertConfig | None = None,
    ) -> None:
        """
        Initialize Whale Alert provider.

        Args:
            api_key: API key (can also be in config)
            config: Full configuration (overrides api_key)
        """
        if config:
            self._config = config
        elif api_key:
            self._config = WhaleAlertConfig(api_key=api_key)
        else:
            raise ValueError("Either api_key or config must be provided")

        super().__init__(api_key=self._config.api_key)

    @property
    def name(self) -> str:
        return "whale_alert"

    @property
    def base_url(self) -> str:
        return self._config.base_url

    def _get_headers(self) -> dict[str, str]:
        """Get headers with API key."""
        headers = super()._get_headers()
        headers["X-WA-API-KEY"] = self._config.api_key
        return headers

    async def get_transactions(
        self,
        min_value_usd: Decimal = Decimal("1000000"),
        blockchain: str | None = None,
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[WhaleTransaction]:
        """
        Get recent whale transactions from Whale Alert API.

        Args:
            min_value_usd: Minimum transaction value in USD
            blockchain: Filter by blockchain (None = all configured)
            limit: Maximum transactions
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of whale transactions
        """
        if not self._client:
            raise RuntimeError("Provider not connected")

        # Build query params
        params: dict[str, Any] = {
            "min_value": int(min_value_usd),
            "limit": limit,
        }

        if blockchain:
            params["blockchain"] = blockchain

        if start_time:
            params["start"] = int(start_time.timestamp())

        if end_time:
            params["end"] = int(end_time.timestamp())

        # Make request
        try:
            response = await self._client.get(
                f"{self.base_url}/transactions",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Whale Alert API error: {e}")
            return []

        # Parse transactions
        transactions: list[WhaleTransaction] = []

        for tx in data.get("transactions", []):
            try:
                transactions.append(self._parse_transaction(tx))
            except Exception as e:
                logger.warning(f"Failed to parse transaction: {e}")

        return transactions

    def _parse_transaction(self, data: dict[str, Any]) -> WhaleTransaction:
        """Parse Whale Alert API transaction response."""
        return WhaleTransaction(
            tx_hash=data.get("hash", ""),
            blockchain=data.get("blockchain", "unknown"),
            timestamp=data.get("timestamp", 0),
            from_address=data.get("from", {}).get("address", ""),
            to_address=data.get("to", {}).get("address", ""),
            amount=Decimal(str(data.get("amount", 0))),
            amount_usd=Decimal(str(data.get("amount_usd", 0))),
            symbol=data.get("symbol", "").upper(),
            tx_type=data.get("transaction_type", "transfer"),
            from_owner=data.get("from", {}).get("owner", None),
            to_owner=data.get("to", {}).get("owner", None),
            from_type=data.get("from", {}).get("owner_type", None),
            to_type=data.get("to", {}).get("owner_type", None),
        )

    async def get_status(self) -> dict[str, Any]:
        """Get API status and remaining quota."""
        if not self._client:
            raise RuntimeError("Provider not connected")

        try:
            response = await self._client.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}


# =============================================================================
# Dune Analytics Provider
# =============================================================================


@dataclass
class DuneConfig:
    """Configuration for Dune Analytics API."""

    api_key: str
    base_url: str = "https://api.dune.com/api/v1"

    # Pre-configured query IDs for whale tracking
    whale_tracker_query_id: int | None = None
    exchange_flows_query_id: int | None = None


class DuneProvider(BaseOnChainProvider):
    """
    Dune Analytics API provider.

    Execute custom SQL queries on blockchain data.

    Features:
    - Custom query execution
    - Pre-built whale tracking queries
    - Historical data analysis

    Example:
        provider = DuneProvider(api_key="your_key")
        async with provider:
            # Execute custom query
            results = await provider.execute_query(query_id=12345)

            # Get whale transactions
            transactions = await provider.get_transactions()
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: DuneConfig | None = None,
    ) -> None:
        """
        Initialize Dune provider.

        Args:
            api_key: API key
            config: Full configuration
        """
        if config:
            self._config = config
        elif api_key:
            self._config = DuneConfig(api_key=api_key)
        else:
            raise ValueError("Either api_key or config must be provided")

        super().__init__(api_key=self._config.api_key)

    @property
    def name(self) -> str:
        return "dune"

    @property
    def base_url(self) -> str:
        return self._config.base_url

    def _get_headers(self) -> dict[str, str]:
        """Get headers with API key."""
        headers = super()._get_headers()
        headers["X-Dune-API-Key"] = self._config.api_key
        return headers

    async def execute_query(
        self,
        query_id: int,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a Dune query and get results.

        Args:
            query_id: Dune query ID
            params: Query parameters

        Returns:
            Query results as list of dicts
        """
        if not self._client:
            raise RuntimeError("Provider not connected")

        # Execute query
        try:
            execute_response = await self._client.post(
                f"{self.base_url}/query/{query_id}/execute",
                json={"query_parameters": params or {}},
            )
            execute_response.raise_for_status()
            execution_id = execute_response.json().get("execution_id")
        except Exception as e:
            logger.error(f"Dune execute error: {e}")
            return []

        # Poll for results
        import asyncio
        max_attempts = 30
        for _ in range(max_attempts):
            try:
                status_response = await self._client.get(
                    f"{self.base_url}/execution/{execution_id}/status"
                )
                status_response.raise_for_status()
                status = status_response.json()

                if status.get("state") == "QUERY_STATE_COMPLETED":
                    # Get results
                    results_response = await self._client.get(
                        f"{self.base_url}/execution/{execution_id}/results"
                    )
                    results_response.raise_for_status()
                    return results_response.json().get("result", {}).get("rows", [])

                elif status.get("state") in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                    logger.error(f"Dune query failed: {status}")
                    return []

            except Exception as e:
                logger.warning(f"Dune poll error: {e}")

            await asyncio.sleep(2)

        logger.error("Dune query timed out")
        return []

    async def get_transactions(
        self,
        min_value_usd: Decimal = Decimal("1000000"),
        blockchain: str | None = None,
        limit: int = 100,
    ) -> list[WhaleTransaction]:
        """
        Get whale transactions from Dune.

        Uses pre-configured whale tracker query if available.

        Args:
            min_value_usd: Minimum transaction value
            blockchain: Filter by blockchain
            limit: Maximum transactions

        Returns:
            List of whale transactions
        """
        query_id = self._config.whale_tracker_query_id
        if not query_id:
            logger.warning("No whale tracker query configured for Dune")
            return []

        params = {
            "min_value_usd": str(min_value_usd),
            "limit": limit,
        }
        if blockchain:
            params["blockchain"] = blockchain

        results = await self.execute_query(query_id, params)

        transactions: list[WhaleTransaction] = []
        for row in results:
            try:
                transactions.append(WhaleTransaction(
                    tx_hash=row.get("tx_hash", ""),
                    blockchain=row.get("blockchain", blockchain or "ethereum"),
                    timestamp=int(row.get("block_time", 0)),
                    from_address=row.get("from_address", ""),
                    to_address=row.get("to_address", ""),
                    amount=Decimal(str(row.get("amount", 0))),
                    amount_usd=Decimal(str(row.get("amount_usd", 0))),
                    symbol=row.get("symbol", "ETH"),
                    tx_type=row.get("tx_type", "transfer"),
                    from_owner=row.get("from_label"),
                    to_owner=row.get("to_label"),
                    from_type=row.get("from_type"),
                    to_type=row.get("to_type"),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse Dune row: {e}")

        return transactions

    async def get_exchange_flows(
        self,
        symbol: str = "ETH",
        hours: int = 24,
    ) -> dict[str, Decimal]:
        """
        Get exchange inflow/outflow data.

        Args:
            symbol: Token symbol
            hours: Lookback period

        Returns:
            Dict with inflow, outflow, net_flow
        """
        query_id = self._config.exchange_flows_query_id
        if not query_id:
            logger.warning("No exchange flows query configured for Dune")
            return {"inflow": Decimal("0"), "outflow": Decimal("0"), "net_flow": Decimal("0")}

        results = await self.execute_query(query_id, {
            "symbol": symbol,
            "hours": hours,
        })

        if results:
            row = results[0]
            inflow = Decimal(str(row.get("inflow", 0)))
            outflow = Decimal(str(row.get("outflow", 0)))
            return {
                "inflow": inflow,
                "outflow": outflow,
                "net_flow": inflow - outflow,
            }

        return {"inflow": Decimal("0"), "outflow": Decimal("0"), "net_flow": Decimal("0")}


# =============================================================================
# Mock Provider (for testing)
# =============================================================================


class MockOnChainProvider(BaseOnChainProvider):
    """
    Mock provider for testing and demos.

    Generates realistic-looking whale transactions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._transactions: list[WhaleTransaction] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def base_url(self) -> str:
        return "mock://localhost"

    async def connect(self) -> None:
        """Connect (no-op for mock)."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect (no-op for mock)."""
        self._connected = False

    def add_transaction(self, tx: WhaleTransaction) -> None:
        """Add a mock transaction."""
        self._transactions.append(tx)

    async def get_transactions(
        self,
        min_value_usd: Decimal = Decimal("1000000"),
        blockchain: str | None = None,
        limit: int = 100,
    ) -> list[WhaleTransaction]:
        """Get mock transactions."""
        filtered = [
            tx for tx in self._transactions
            if tx.amount_usd >= min_value_usd
            and (blockchain is None or tx.blockchain == blockchain)
        ]
        return filtered[:limit]

    @classmethod
    def with_demo_data(cls) -> MockOnChainProvider:
        """Create provider with demo transactions."""
        import time as time_module
        provider = cls()

        # Add some demo transactions
        demo_txs = [
            WhaleTransaction(
                tx_hash="0xabc123...",
                blockchain="ethereum",
                timestamp=int(time_module.time()) - 300,
                from_address="0x1234...whale",
                to_address="0x5678...binance",
                amount=Decimal("5000"),
                amount_usd=Decimal("15000000"),
                symbol="ETH",
                tx_type="transfer",
                from_owner="Unknown Whale",
                to_owner="Binance",
                from_type="whale",
                to_type="exchange",
            ),
            WhaleTransaction(
                tx_hash="0xdef456...",
                blockchain="bitcoin",
                timestamp=int(time_module.time()) - 600,
                from_address="bc1q...coinbase",
                to_address="bc1q...unknown",
                amount=Decimal("500"),
                amount_usd=Decimal("25000000"),
                symbol="BTC",
                tx_type="transfer",
                from_owner="Coinbase",
                to_owner="Unknown",
                from_type="exchange",
                to_type="unknown",
            ),
            WhaleTransaction(
                tx_hash="0xghi789...",
                blockchain="ethereum",
                timestamp=int(time_module.time()) - 120,
                from_address="0xabcd...contract",
                to_address="0x0000...burn",
                amount=Decimal("1000000"),
                amount_usd=Decimal("2000000"),
                symbol="USDC",
                tx_type="burn",
                from_owner="Circle",
                to_owner=None,
                from_type="contract",
                to_type=None,
            ),
        ]

        for tx in demo_txs:
            provider.add_transaction(tx)

        return provider


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BaseOnChainProvider",
    "WhaleAlertProvider",
    "WhaleAlertConfig",
    "DuneProvider",
    "DuneConfig",
    "MockOnChainProvider",
]

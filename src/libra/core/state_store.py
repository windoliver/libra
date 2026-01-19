"""
Redis-backed state store for crash recovery and distributed operation.

Provides:
- Connection pooling for efficient Redis access
- Order and position persistence with TTL
- Crash recovery: restore state after restart
- Distributed: share state across processes

See: https://github.com/windoliver/libra/issues/85
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec

from libra.gateways.protocol import (
    OrderResult,
    Position,
    PositionSide,
)


if TYPE_CHECKING:
    import redis.asyncio as redis


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RedisConfig:
    """Configuration for Redis state store."""

    url: str = "redis://localhost:6379"
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30

    # TTL for persisted state (seconds)
    order_ttl: int = 86400  # 24 hours
    position_ttl: int = 86400 * 7  # 7 days

    # Key prefixes
    order_prefix: str = "libra:orders"
    position_prefix: str = "libra:positions"
    metadata_prefix: str = "libra:meta"


# =============================================================================
# Encoders/Decoders
# =============================================================================

# Reuse msgspec encoders for performance (Issue #81)
_encoder = msgspec.json.Encoder()
_order_result_decoder = msgspec.json.Decoder(OrderResult)
_position_decoder = msgspec.json.Decoder(Position)


# =============================================================================
# Redis State Store
# =============================================================================


class RedisStateStore:
    """
    Redis-backed state store for order and position persistence.

    Uses connection pooling for efficient Redis access.
    Supports crash recovery by restoring open orders and positions.

    Example:
        store = RedisStateStore()
        await store.connect()

        # Save order
        await store.save_order(order_result)

        # Get open orders for crash recovery
        open_orders = await store.get_open_orders()

        # Restore after crash
        orders, positions = await store.restore_state()

        await store.close()

    Usage with context manager:
        async with RedisStateStore() as store:
            await store.save_order(order_result)
    """

    def __init__(self, config: RedisConfig | None = None) -> None:
        """
        Initialize Redis state store.

        Args:
            config: Redis configuration (uses defaults if None)
        """
        self.config = config or RedisConfig()
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Redis with connection pool."""
        if self._connected:
            return

        import redis.asyncio as redis_lib

        self._pool = redis_lib.ConnectionPool.from_url(
            self.config.url,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            retry_on_timeout=self.config.retry_on_timeout,
            health_check_interval=self.config.health_check_interval,
        )
        self._client = redis_lib.Redis(connection_pool=self._pool)

        # Test connection
        await self._client.ping()
        self._connected = True
        logger.info(
            "Connected to Redis: %s (pool_size=%d)",
            self.config.url,
            self.config.max_connections,
        )

    async def close(self) -> None:
        """Close Redis connection and pool."""
        if self._client:
            await self._client.aclose()  # type: ignore[attr-defined]
            self._client = None

        if self._pool:
            await self._pool.disconnect()
            self._pool = None

        self._connected = False
        logger.info("Redis connection closed")

    async def __aenter__(self) -> RedisStateStore:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected

    def _ensure_connected(self) -> redis.Redis:
        """Ensure connected and return client."""
        if not self._connected or self._client is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")
        return self._client

    # =========================================================================
    # Order Persistence
    # =========================================================================

    def _order_key(self, order_id: str) -> str:
        """Generate Redis key for order."""
        return f"{self.config.order_prefix}:{order_id}"

    async def save_order(self, order: OrderResult) -> None:
        """
        Save order to Redis.

        Args:
            order: Order result to persist
        """
        client = self._ensure_connected()
        key = self._order_key(order.order_id)

        # Serialize using msgspec for consistency
        data = _encoder.encode(order)

        # Store with TTL
        await client.set(key, data, ex=self.config.order_ttl)
        logger.debug("Saved order: %s (status=%s)", order.order_id, order.status.value)

    async def get_order(self, order_id: str) -> OrderResult | None:
        """
        Get order by ID.

        Args:
            order_id: Order ID to retrieve

        Returns:
            OrderResult or None if not found
        """
        client = self._ensure_connected()
        key = self._order_key(order_id)

        data = await client.get(key)
        if data is None:
            return None

        return _order_result_decoder.decode(data)

    async def delete_order(self, order_id: str) -> bool:
        """
        Delete order from Redis.

        Args:
            order_id: Order ID to delete

        Returns:
            True if deleted, False if not found
        """
        client = self._ensure_connected()
        key = self._order_key(order_id)
        result = await client.delete(key)
        return result > 0

    async def get_open_orders(self) -> list[OrderResult]:
        """
        Get all open orders (for crash recovery).

        Returns:
            List of open orders
        """
        client = self._ensure_connected()
        pattern = f"{self.config.order_prefix}:*"

        open_orders: list[OrderResult] = []
        async for key in client.scan_iter(match=pattern, count=100):
            data = await client.get(key)
            if data:
                order = _order_result_decoder.decode(data)
                if order.is_open:
                    open_orders.append(order)

        logger.info("Found %d open orders", len(open_orders))
        return open_orders

    async def get_orders_by_symbol(self, symbol: str) -> list[OrderResult]:
        """
        Get all orders for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            List of orders for the symbol
        """
        client = self._ensure_connected()
        pattern = f"{self.config.order_prefix}:*"

        orders: list[OrderResult] = []
        async for key in client.scan_iter(match=pattern, count=100):
            data = await client.get(key)
            if data:
                order = _order_result_decoder.decode(data)
                if order.symbol == symbol:
                    orders.append(order)

        return orders

    # =========================================================================
    # Position Persistence
    # =========================================================================

    def _position_key(self, symbol: str) -> str:
        """Generate Redis key for position."""
        return f"{self.config.position_prefix}:{symbol}"

    async def save_position(self, position: Position) -> None:
        """
        Save position to Redis.

        Args:
            position: Position to persist
        """
        client = self._ensure_connected()
        key = self._position_key(position.symbol)

        # Serialize using msgspec
        data = _encoder.encode(position)

        # Store with TTL
        await client.set(key, data, ex=self.config.position_ttl)
        logger.debug(
            "Saved position: %s (side=%s, amount=%s)",
            position.symbol,
            position.side.value,
            position.amount,
        )

    async def get_position(self, symbol: str) -> Position | None:
        """
        Get position by symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position or None if not found
        """
        client = self._ensure_connected()
        key = self._position_key(symbol)

        data = await client.get(key)
        if data is None:
            return None

        return _position_decoder.decode(data)

    async def delete_position(self, symbol: str) -> bool:
        """
        Delete position from Redis.

        Args:
            symbol: Trading pair symbol

        Returns:
            True if deleted, False if not found
        """
        client = self._ensure_connected()
        key = self._position_key(symbol)
        result = await client.delete(key)
        return result > 0

    async def get_all_positions(self) -> list[Position]:
        """
        Get all positions (for crash recovery).

        Returns:
            List of all positions
        """
        client = self._ensure_connected()
        pattern = f"{self.config.position_prefix}:*"

        positions: list[Position] = []
        async for key in client.scan_iter(match=pattern, count=100):
            data = await client.get(key)
            if data:
                position = _position_decoder.decode(data)
                # Only return non-flat positions
                if position.side != PositionSide.FLAT:
                    positions.append(position)

        logger.info("Found %d positions", len(positions))
        return positions

    # =========================================================================
    # Crash Recovery
    # =========================================================================

    async def restore_state(self) -> tuple[list[OrderResult], list[Position]]:
        """
        Restore full state after crash/restart.

        Returns:
            Tuple of (open_orders, positions)
        """
        logger.info("Restoring state from Redis...")

        open_orders = await self.get_open_orders()
        positions = await self.get_all_positions()

        logger.info(
            "State restored: %d open orders, %d positions",
            len(open_orders),
            len(positions),
        )

        return open_orders, positions

    async def clear_all(self) -> int:
        """
        Clear all stored state (use with caution!).

        Returns:
            Number of keys deleted
        """
        client = self._ensure_connected()
        deleted = 0

        # Delete orders
        pattern = f"{self.config.order_prefix}:*"
        async for key in client.scan_iter(match=pattern, count=100):
            await client.delete(key)
            deleted += 1

        # Delete positions
        pattern = f"{self.config.position_prefix}:*"
        async for key in client.scan_iter(match=pattern, count=100):
            await client.delete(key)
            deleted += 1

        logger.warning("Cleared %d keys from Redis", deleted)
        return deleted

    # =========================================================================
    # Metadata
    # =========================================================================

    async def save_metadata(self, key: str, value: str, ttl: int | None = None) -> None:
        """
        Save metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value
            ttl: Optional TTL in seconds
        """
        client = self._ensure_connected()
        full_key = f"{self.config.metadata_prefix}:{key}"
        await client.set(full_key, value, ex=ttl)

    async def get_metadata(self, key: str) -> str | None:
        """
        Get metadata value.

        Args:
            key: Metadata key

        Returns:
            Value or None if not found
        """
        client = self._ensure_connected()
        full_key = f"{self.config.metadata_prefix}:{key}"
        result = await client.get(full_key)
        return result.decode() if result else None

    # =========================================================================
    # Health Check
    # =========================================================================

    async def ping(self) -> bool:
        """
        Check Redis connection health.

        Returns:
            True if healthy
        """
        try:
            client = self._ensure_connected()
            await client.ping()
            return True
        except Exception:
            logger.exception("Redis ping failed")
            return False

    @property
    def stats(self) -> dict[str, int | bool]:
        """Get store statistics."""
        return {
            "connected": self._connected,
            "max_connections": self.config.max_connections,
        }

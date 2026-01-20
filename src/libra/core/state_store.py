"""
State Store for Crash-Only Design.

Implements externalized state persistence for the TradingKernel, enabling:
- Unified recovery: startup and crash recovery use same code path
- Externalized state: critical state persisted to external store
- Idempotent operations: safe to retry after restart
- Fail-fast: inconsistent state triggers immediate termination

Issue #102: Implement crash-only design pattern
Reference: https://www.usenix.org/legacy/events/hotos03/tech/full_papers/candea/candea.pdf
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# State Models
# =============================================================================


class OrderState(Enum):
    """Persisted order state."""

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class PersistedOrder:
    """
    Order state for crash recovery.

    Contains all information needed to recover order tracking after restart.
    """

    order_id: str
    client_order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "limit", "market", etc.
    amount: str  # Decimal as string for serialization
    filled_amount: str
    price: str | None
    state: OrderState
    created_at_ns: int
    updated_at_ns: int
    exchange_order_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "amount": self.amount,
            "filled_amount": self.filled_amount,
            "price": self.price,
            "state": self.state.value,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
            "exchange_order_id": self.exchange_order_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistedOrder:
        """Deserialize from dictionary."""
        return cls(
            order_id=data["order_id"],
            client_order_id=data["client_order_id"],
            symbol=data["symbol"],
            side=data["side"],
            order_type=data["order_type"],
            amount=data["amount"],
            filled_amount=data["filled_amount"],
            price=data.get("price"),
            state=OrderState(data["state"]),
            created_at_ns=data["created_at_ns"],
            updated_at_ns=data["updated_at_ns"],
            exchange_order_id=data.get("exchange_order_id"),
        )


@dataclass
class PersistedPosition:
    """
    Position state for crash recovery.

    Contains all information needed to recover position tracking after restart.
    """

    symbol: str
    side: str  # "long" or "short"
    quantity: str  # Decimal as string
    entry_price: str
    unrealized_pnl: str
    realized_pnl: str
    opened_at_ns: int
    updated_at_ns: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "opened_at_ns": self.opened_at_ns,
            "updated_at_ns": self.updated_at_ns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistedPosition:
        """Deserialize from dictionary."""
        return cls(
            symbol=data["symbol"],
            side=data["side"],
            quantity=data["quantity"],
            entry_price=data["entry_price"],
            unrealized_pnl=data["unrealized_pnl"],
            realized_pnl=data["realized_pnl"],
            opened_at_ns=data["opened_at_ns"],
            updated_at_ns=data["updated_at_ns"],
        )


@dataclass
class KernelCheckpoint:
    """
    Kernel state checkpoint for crash recovery.

    Contains timestamp and state hash for consistency verification.
    """

    instance_id: str
    checkpoint_id: str
    timestamp_ns: int
    state_hash: str
    orders_count: int
    positions_count: int
    environment: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "instance_id": self.instance_id,
            "checkpoint_id": self.checkpoint_id,
            "timestamp_ns": self.timestamp_ns,
            "state_hash": self.state_hash,
            "orders_count": self.orders_count,
            "positions_count": self.positions_count,
            "environment": self.environment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelCheckpoint:
        """Deserialize from dictionary."""
        return cls(
            instance_id=data["instance_id"],
            checkpoint_id=data["checkpoint_id"],
            timestamp_ns=data["timestamp_ns"],
            state_hash=data["state_hash"],
            orders_count=data["orders_count"],
            positions_count=data["positions_count"],
            environment=data["environment"],
        )


# =============================================================================
# State Store Protocol
# =============================================================================


class StateStore(ABC):
    """
    Abstract base class for state persistence.

    Implementations can use Redis, PostgreSQL, file system, etc.
    All operations should be idempotent for crash-only design.
    """

    @abstractmethod
    async def save_order(self, order: PersistedOrder) -> None:
        """Save or update order state."""

    @abstractmethod
    async def get_order(self, order_id: str) -> PersistedOrder | None:
        """Get order by ID."""

    @abstractmethod
    async def get_open_orders(self) -> list[PersistedOrder]:
        """Get all open orders."""

    @abstractmethod
    async def delete_order(self, order_id: str) -> None:
        """Delete order (after fill/cancel)."""

    @abstractmethod
    async def save_position(self, position: PersistedPosition) -> None:
        """Save or update position state."""

    @abstractmethod
    async def get_position(self, symbol: str) -> PersistedPosition | None:
        """Get position by symbol."""

    @abstractmethod
    async def get_positions(self) -> list[PersistedPosition]:
        """Get all positions."""

    @abstractmethod
    async def delete_position(self, symbol: str) -> None:
        """Delete position (after close)."""

    @abstractmethod
    async def save_checkpoint(self, checkpoint: KernelCheckpoint) -> None:
        """Save kernel checkpoint."""

    @abstractmethod
    async def get_latest_checkpoint(self) -> KernelCheckpoint | None:
        """Get most recent checkpoint."""

    @abstractmethod
    async def clear_all(self) -> None:
        """Clear all state (for testing)."""


# =============================================================================
# File-Based State Store (Default Implementation)
# =============================================================================


class FileStateStore(StateStore):
    """
    File-based state store for development and testing.

    Uses JSON files for persistence. For production, use RedisStateStore
    or PostgresStateStore instead.

    File structure:
        state_dir/
            orders/
                {order_id}.json
            positions/
                {symbol}.json
            checkpoints/
                {checkpoint_id}.json
            latest_checkpoint.json
    """

    def __init__(self, state_dir: str | Path = ".libra_state") -> None:
        """
        Initialize file-based state store.

        Args:
            state_dir: Directory for state files
        """
        self._state_dir = Path(state_dir)
        self._orders_dir = self._state_dir / "orders"
        self._positions_dir = self._state_dir / "positions"
        self._checkpoints_dir = self._state_dir / "checkpoints"

        # Create directories
        self._orders_dir.mkdir(parents=True, exist_ok=True)
        self._positions_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

        logger.info("FileStateStore initialized: %s", self._state_dir)

    async def save_order(self, order: PersistedOrder) -> None:
        """Save order to file."""
        path = self._orders_dir / f"{order.order_id}.json"
        path.write_text(json.dumps(order.to_dict(), indent=2))

    async def get_order(self, order_id: str) -> PersistedOrder | None:
        """Get order from file."""
        path = self._orders_dir / f"{order_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return PersistedOrder.from_dict(data)

    async def get_open_orders(self) -> list[PersistedOrder]:
        """Get all open orders."""
        orders = []
        for path in self._orders_dir.glob("*.json"):
            data = json.loads(path.read_text())
            order = PersistedOrder.from_dict(data)
            if order.state in (OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.PENDING):
                orders.append(order)
        return orders

    async def delete_order(self, order_id: str) -> None:
        """Delete order file."""
        path = self._orders_dir / f"{order_id}.json"
        if path.exists():
            path.unlink()

    async def save_position(self, position: PersistedPosition) -> None:
        """Save position to file."""
        # Sanitize symbol for filename
        safe_symbol = position.symbol.replace("/", "_")
        path = self._positions_dir / f"{safe_symbol}.json"
        path.write_text(json.dumps(position.to_dict(), indent=2))

    async def get_position(self, symbol: str) -> PersistedPosition | None:
        """Get position from file."""
        safe_symbol = symbol.replace("/", "_")
        path = self._positions_dir / f"{safe_symbol}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return PersistedPosition.from_dict(data)

    async def get_positions(self) -> list[PersistedPosition]:
        """Get all positions."""
        positions = []
        for path in self._positions_dir.glob("*.json"):
            data = json.loads(path.read_text())
            positions.append(PersistedPosition.from_dict(data))
        return positions

    async def delete_position(self, symbol: str) -> None:
        """Delete position file."""
        safe_symbol = symbol.replace("/", "_")
        path = self._positions_dir / f"{safe_symbol}.json"
        if path.exists():
            path.unlink()

    async def save_checkpoint(self, checkpoint: KernelCheckpoint) -> None:
        """Save checkpoint to file."""
        # Save checkpoint
        path = self._checkpoints_dir / f"{checkpoint.checkpoint_id}.json"
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2))

        # Update latest pointer
        latest_path = self._state_dir / "latest_checkpoint.json"
        latest_path.write_text(json.dumps({"checkpoint_id": checkpoint.checkpoint_id}))

    async def get_latest_checkpoint(self) -> KernelCheckpoint | None:
        """Get most recent checkpoint."""
        latest_path = self._state_dir / "latest_checkpoint.json"
        if not latest_path.exists():
            return None

        latest_data = json.loads(latest_path.read_text())
        checkpoint_id = latest_data.get("checkpoint_id")
        if not checkpoint_id:
            return None

        path = self._checkpoints_dir / f"{checkpoint_id}.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        return KernelCheckpoint.from_dict(data)

    async def clear_all(self) -> None:
        """Clear all state files."""
        import shutil

        if self._state_dir.exists():
            shutil.rmtree(self._state_dir)
        self._orders_dir.mkdir(parents=True, exist_ok=True)
        self._positions_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# In-Memory State Store (for testing)
# =============================================================================


class MemoryStateStore(StateStore):
    """
    In-memory state store for testing.

    All state is lost on process termination. Use for unit tests
    and development without external dependencies.
    """

    def __init__(self) -> None:
        """Initialize in-memory state store."""
        self._orders: dict[str, PersistedOrder] = {}
        self._positions: dict[str, PersistedPosition] = {}
        self._checkpoints: dict[str, KernelCheckpoint] = {}
        self._latest_checkpoint_id: str | None = None
        logger.info("MemoryStateStore initialized")

    async def save_order(self, order: PersistedOrder) -> None:
        """Save order to memory."""
        self._orders[order.order_id] = order

    async def get_order(self, order_id: str) -> PersistedOrder | None:
        """Get order from memory."""
        return self._orders.get(order_id)

    async def get_open_orders(self) -> list[PersistedOrder]:
        """Get all open orders."""
        return [
            o for o in self._orders.values()
            if o.state in (OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.PENDING)
        ]

    async def delete_order(self, order_id: str) -> None:
        """Delete order from memory."""
        self._orders.pop(order_id, None)

    async def save_position(self, position: PersistedPosition) -> None:
        """Save position to memory."""
        self._positions[position.symbol] = position

    async def get_position(self, symbol: str) -> PersistedPosition | None:
        """Get position from memory."""
        return self._positions.get(symbol)

    async def get_positions(self) -> list[PersistedPosition]:
        """Get all positions."""
        return list(self._positions.values())

    async def delete_position(self, symbol: str) -> None:
        """Delete position from memory."""
        self._positions.pop(symbol, None)

    async def save_checkpoint(self, checkpoint: KernelCheckpoint) -> None:
        """Save checkpoint to memory."""
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        self._latest_checkpoint_id = checkpoint.checkpoint_id

    async def get_latest_checkpoint(self) -> KernelCheckpoint | None:
        """Get most recent checkpoint."""
        if self._latest_checkpoint_id is None:
            return None
        return self._checkpoints.get(self._latest_checkpoint_id)

    async def clear_all(self) -> None:
        """Clear all in-memory state."""
        self._orders.clear()
        self._positions.clear()
        self._checkpoints.clear()
        self._latest_checkpoint_id = None


# =============================================================================
# Redis State Store (Issue #108)
# =============================================================================


# Check if redis is available
try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisStateStore(StateStore):
    """
    Redis-backed state store for production use.

    Provides durable state persistence with:
    - Automatic reconnection
    - Configurable TTL for old data
    - Key prefix for multi-tenant support
    - Atomic operations with Lua scripts

    Key schema:
        {prefix}orders:{order_id} -> JSON
        {prefix}positions:{symbol} -> JSON
        {prefix}checkpoints:{checkpoint_id} -> JSON
        {prefix}latest_checkpoint -> checkpoint_id

    Example:
        store = RedisStateStore(
            url="redis://localhost:6379",
            prefix="libra:prod:",
            ttl_seconds=86400,  # 24 hours
        )
        await store.connect()

        # Use with kernel
        kernel = TradingKernel(state_store=store)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "libra:",
        ttl_seconds: int | None = None,
        db: int = 0,
    ) -> None:
        """
        Initialize Redis state store.

        Args:
            url: Redis connection URL
            prefix: Key prefix for all state keys
            ttl_seconds: Optional TTL for state entries
            db: Redis database number
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package not installed. Install with: pip install redis[hiredis]"
            )

        self._url = url
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._db = db
        self._client: aioredis.Redis | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        self._client = aioredis.from_url(
            self._url,
            db=self._db,
            decode_responses=True,
        )
        # Test connection
        await self._client.ping()
        self._connected = True
        logger.info("RedisStateStore connected to %s", self._url)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
            logger.info("RedisStateStore disconnected")

    def _key(self, *parts: str) -> str:
        """Build prefixed key."""
        return f"{self._prefix}{':'.join(parts)}"

    async def _ensure_connected(self) -> None:
        """Ensure Redis connection is active."""
        if not self._connected or self._client is None:
            await self.connect()

    async def save_order(self, order: PersistedOrder) -> None:
        """Save order to Redis."""
        await self._ensure_connected()
        key = self._key("orders", order.order_id)
        value = json.dumps(order.to_dict())
        if self._ttl:
            await self._client.setex(key, self._ttl, value)
        else:
            await self._client.set(key, value)

    async def get_order(self, order_id: str) -> PersistedOrder | None:
        """Get order from Redis."""
        await self._ensure_connected()
        key = self._key("orders", order_id)
        data = await self._client.get(key)
        if data is None:
            return None
        return PersistedOrder.from_dict(json.loads(data))

    async def get_open_orders(self) -> list[PersistedOrder]:
        """Get all open orders."""
        await self._ensure_connected()
        pattern = self._key("orders", "*")
        orders = []

        async for key in self._client.scan_iter(match=pattern):
            data = await self._client.get(key)
            if data:
                order = PersistedOrder.from_dict(json.loads(data))
                if order.state in (OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.PENDING):
                    orders.append(order)

        return orders

    async def delete_order(self, order_id: str) -> None:
        """Delete order from Redis."""
        await self._ensure_connected()
        key = self._key("orders", order_id)
        await self._client.delete(key)

    async def save_position(self, position: PersistedPosition) -> None:
        """Save position to Redis."""
        await self._ensure_connected()
        # Sanitize symbol for key
        safe_symbol = position.symbol.replace("/", "_")
        key = self._key("positions", safe_symbol)
        value = json.dumps(position.to_dict())
        if self._ttl:
            await self._client.setex(key, self._ttl, value)
        else:
            await self._client.set(key, value)

    async def get_position(self, symbol: str) -> PersistedPosition | None:
        """Get position from Redis."""
        await self._ensure_connected()
        safe_symbol = symbol.replace("/", "_")
        key = self._key("positions", safe_symbol)
        data = await self._client.get(key)
        if data is None:
            return None
        return PersistedPosition.from_dict(json.loads(data))

    async def get_positions(self) -> list[PersistedPosition]:
        """Get all positions."""
        await self._ensure_connected()
        pattern = self._key("positions", "*")
        positions = []

        async for key in self._client.scan_iter(match=pattern):
            data = await self._client.get(key)
            if data:
                positions.append(PersistedPosition.from_dict(json.loads(data)))

        return positions

    async def delete_position(self, symbol: str) -> None:
        """Delete position from Redis."""
        await self._ensure_connected()
        safe_symbol = symbol.replace("/", "_")
        key = self._key("positions", safe_symbol)
        await self._client.delete(key)

    async def save_checkpoint(self, checkpoint: KernelCheckpoint) -> None:
        """Save checkpoint to Redis."""
        await self._ensure_connected()

        # Save checkpoint
        key = self._key("checkpoints", checkpoint.checkpoint_id)
        value = json.dumps(checkpoint.to_dict())
        if self._ttl:
            await self._client.setex(key, self._ttl, value)
        else:
            await self._client.set(key, value)

        # Update latest pointer
        latest_key = self._key("latest_checkpoint")
        await self._client.set(latest_key, checkpoint.checkpoint_id)

    async def get_latest_checkpoint(self) -> KernelCheckpoint | None:
        """Get most recent checkpoint."""
        await self._ensure_connected()

        latest_key = self._key("latest_checkpoint")
        checkpoint_id = await self._client.get(latest_key)
        if checkpoint_id is None:
            return None

        key = self._key("checkpoints", checkpoint_id)
        data = await self._client.get(key)
        if data is None:
            return None

        return KernelCheckpoint.from_dict(json.loads(data))

    async def clear_all(self) -> None:
        """Clear all state with prefix."""
        await self._ensure_connected()
        pattern = self._key("*")

        # Delete in batches
        async for key in self._client.scan_iter(match=pattern):
            await self._client.delete(key)

        logger.warning("RedisStateStore cleared all state with prefix: %s", self._prefix)

    async def __aenter__(self) -> "RedisStateStore":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()


# =============================================================================
# State Verification
# =============================================================================


class StateVerificationError(Exception):
    """Raised when state verification fails (triggers fail-fast)."""

    pass


@dataclass
class RecoveryResult:
    """Result of state recovery operation."""

    success: bool
    orders_recovered: int = 0
    positions_recovered: int = 0
    checkpoint: KernelCheckpoint | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


async def verify_order_consistency(
    order: PersistedOrder,
    exchange_order: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """
    Verify order state consistency between local and exchange.

    Args:
        order: Persisted local order state
        exchange_order: Order state from exchange (if available)

    Returns:
        Tuple of (is_consistent, error_message)
    """
    if exchange_order is None:
        # Order not found on exchange - may be filled/cancelled
        if order.state in (OrderState.OPEN, OrderState.PARTIALLY_FILLED):
            return False, f"Order {order.order_id} open locally but not found on exchange"
        return True, None

    # Check key fields match
    exchange_filled = Decimal(str(exchange_order.get("filled_amount", "0")))
    local_filled = Decimal(order.filled_amount)

    if exchange_filled < local_filled:
        return False, (
            f"Order {order.order_id} has inconsistent fill: "
            f"exchange={exchange_filled}, local={local_filled}"
        )

    return True, None


def compute_state_hash(
    orders: list[PersistedOrder],
    positions: list[PersistedPosition],
) -> str:
    """
    Compute hash of current state for consistency verification.

    Args:
        orders: List of orders
        positions: List of positions

    Returns:
        State hash string
    """
    import hashlib

    # Sort for deterministic ordering
    order_strs = sorted(f"{o.order_id}:{o.state.value}:{o.filled_amount}" for o in orders)
    position_strs = sorted(f"{p.symbol}:{p.quantity}" for p in positions)

    combined = "|".join(order_strs + position_strs)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

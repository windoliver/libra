"""
ExecutionClient Protocol: Order execution operations interface.

Defines the contract for all order execution client implementations.
Handles order submission, cancellation, modification, and account queries.

Design inspired by NautilusTrader's ExecutionClient architecture.
See: https://nautilustrader.io/docs/latest/concepts/adapters/

Implementations:
    - CCXTExecutionClient: 100+ exchanges via CCXT
    - PaperExecutionClient: Simulated paper trading
    - BacktestExecutionClient: Historical backtesting with fill simulation

See: https://github.com/windoliver/libra/issues/33
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from libra.gateways.protocol import (
        Balance,
        Order,
        OrderResult,
        Position,
    )


# =============================================================================
# ExecutionClient Protocol
# =============================================================================


@runtime_checkable
class ExecutionClient(Protocol):
    """
    Order execution client protocol.

    Defines the interface for submitting, cancelling, and managing orders.
    All execution operations (order management, account queries) go through here.

    This protocol uses structural subtyping - any class implementing these
    methods is considered an ExecutionClient, no inheritance required.

    Thread Safety:
        Implementations should be thread-safe. Order state should be
        protected by locks if accessed from multiple coroutines.

    Connection Management:
        - connect() must be called before any execution operations
        - disconnect() should be called for cleanup
        - Implementations should handle reconnection internally

    Order Flow:
        1. Client calls submit_order(order)
        2. ExecutionClient validates and sends to exchange
        3. Returns OrderResult with status (PENDING, OPEN, REJECTED, etc.)
        4. Client can query status via get_order() or stream via stream_order_updates()
        5. Client can cancel via cancel_order()

    Reconciliation:
        Call reconcile_orders() on startup and periodically to sync local
        state with exchange state. This catches orders filled while disconnected.

    Examples:
        async with CCXTExecutionClient("binance", api_key, secret) as client:
            # Submit order
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.1"),
                price=Decimal("50000"),
            )
            result = await client.submit_order(order)
            print(f"Order {result.order_id}: {result.status}")

            # Monitor fills
            async for update in client.stream_order_updates():
                print(f"Order {update.order_id} now {update.status}")
    """

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Client identifier (e.g., "binance-exec", "paper-exec").

        Used for logging, metrics, and routing.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """
        Check if client is connected and ready.

        Returns:
            True if connected and can execute orders, False otherwise.
        """
        ...

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to the execution venue.

        Establishes connections, authenticates with API credentials,
        and prepares for order execution.

        Raises:
            ConnectionError: If connection fails.
            AuthenticationError: If authentication fails.

        Note:
            Must be called before any execution operations.
            Implementations should handle reconnection internally.
        """
        ...

    async def disconnect(self) -> None:
        """
        Disconnect from the execution venue.

        Closes all connections and cleans up resources.
        Does NOT cancel open orders (call cancel_all_orders() first if desired).

        Note:
            Should be safe to call multiple times (idempotent).
        """
        ...

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order to the exchange.

        The order is validated locally then sent to the exchange.
        Returns immediately with initial status (usually PENDING or OPEN).

        Args:
            order: Order to submit

        Returns:
            OrderResult with order_id, status, and fill info (if any).

        Raises:
            OrderError: If order validation fails locally.
            InsufficientFundsError: If not enough balance.
            ConnectionError: If not connected.

        Note:
            - client_order_id is used for idempotent retries
            - Use stream_order_updates() to monitor fill progress
            - Market orders may fill immediately (status=FILLED)

        Example:
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("0.1"),
                client_order_id="my-order-123",
            )
            result = await client.submit_order(order)
        """
        ...

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Exchange-assigned order ID
            symbol: Trading pair (required by some exchanges)

        Returns:
            True if cancelled successfully, False if already closed.

        Raises:
            OrderNotFoundError: If order doesn't exist.
            OrderError: If cancellation fails.

        Note:
            Some exchanges require symbol for cancellation.
            Race condition: order may fill before cancel reaches exchange.
        """
        ...

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional symbol to filter. None = all symbols.

        Returns:
            Number of orders cancelled.

        Note:
            Useful for emergency shutdown or strategy restart.
            Some orders may fill before cancellation completes.
        """
        ...

    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        price: Any | None = None,  # Decimal
        amount: Any | None = None,  # Decimal
    ) -> OrderResult:
        """
        Modify an existing order.

        Not all exchanges support order modification. Those that don't
        will cancel and replace (which may lose queue position).

        Args:
            order_id: Exchange-assigned order ID
            symbol: Trading pair
            price: New price (None = keep current)
            amount: New amount (None = keep current)

        Returns:
            OrderResult with updated order state.

        Raises:
            OrderNotFoundError: If order doesn't exist.
            OrderError: If modification fails.
            NotImplementedError: If exchange doesn't support modification.

        Note:
            At least one of price or amount must be provided.
        """
        ...

    # -------------------------------------------------------------------------
    # Order Queries
    # -------------------------------------------------------------------------

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """
        Get current status of an order.

        Args:
            order_id: Exchange-assigned order ID
            symbol: Trading pair

        Returns:
            Current OrderResult with latest status and fill info.

        Raises:
            OrderNotFoundError: If order doesn't exist.
        """
        ...

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol to filter.

        Returns:
            List of open OrderResults (status OPEN or PARTIALLY_FILLED).
        """
        ...

    async def get_order_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[OrderResult]:
        """
        Get historical orders (filled, cancelled, etc.).

        Args:
            symbol: Optional symbol to filter.
            limit: Maximum number of orders to return.

        Returns:
            List of OrderResults sorted by time descending.

        Note:
            Exchange retention policies vary (some keep 30 days, some longer).
        """
        ...

    # -------------------------------------------------------------------------
    # Order Event Stream
    # -------------------------------------------------------------------------

    def stream_order_updates(self) -> AsyncIterator[OrderResult]:
        """
        Stream real-time order status updates.

        Yields OrderResult whenever an order's status changes:
        - PENDING → OPEN (acknowledged)
        - OPEN → PARTIALLY_FILLED (partial fill)
        - PARTIALLY_FILLED → FILLED (complete fill)
        - OPEN → CANCELLED (cancelled)
        - etc.

        Yields:
            OrderResult objects on each status change.

        Note:
            Subscribe to symbols via subscribe_order_updates() first
            (some implementations auto-subscribe to all orders).

        Example:
            async for update in client.stream_order_updates():
                if update.status == OrderStatus.FILLED:
                    print(f"Order {update.order_id} filled at {update.average_price}")
        """
        ...

    # -------------------------------------------------------------------------
    # Account State
    # -------------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of current positions with P&L.

        Note:
            For spot trading, positions are derived from balances.
            For derivatives, positions are tracked separately.
        """
        ...

    async def get_position(self, symbol: str) -> Position | None:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position if exists, None otherwise.
        """
        ...

    async def get_balances(self) -> dict[str, Balance]:
        """
        Get account balances for all currencies.

        Returns:
            Dict mapping currency to Balance.
            Example: {"USDT": Balance(...), "BTC": Balance(...)}
        """
        ...

    async def get_balance(self, currency: str) -> Balance | None:
        """
        Get balance for a specific currency.

        Args:
            currency: Currency code (e.g., "USDT", "BTC")

        Returns:
            Balance if exists, None otherwise.
        """
        ...

    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------

    async def reconcile_orders(self) -> int:
        """
        Reconcile local order state with exchange state.

        Fetches all open orders from exchange and updates local state.
        Publishes events for any orders that changed while disconnected.

        Returns:
            Number of orders reconciled (state changes detected).

        Note:
            Call on startup and after reconnection.
            May be called periodically (every 5-10 minutes) for safety.
        """
        ...

    async def reconcile_positions(self) -> int:
        """
        Reconcile local position state with exchange state.

        Returns:
            Number of positions reconciled.
        """
        ...


# =============================================================================
# Abstract Base Class (Optional)
# =============================================================================


class BaseExecutionClient(ABC):
    """
    Abstract base class for ExecutionClient implementations.

    Provides common functionality:
    - Connection state tracking
    - Order tracking
    - Configuration handling

    Subclasses must implement all abstract methods.
    Use this base class for code reuse, or implement ExecutionClient protocol directly.

    Examples:
        class CCXTExecutionClient(BaseExecutionClient):
            async def connect(self) -> None:
                self._exchange = ccxt.pro.binance({
                    'apiKey': self._config['api_key'],
                    'secret': self._config['secret'],
                })
                await self._exchange.load_markets()
                self._connected = True
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        """
        Initialize execution client.

        Args:
            name: Client identifier
            config: Configuration dict (API keys, etc.)
        """
        self._name = name
        self._config = config or {}
        self._connected = False
        self._open_orders: dict[str, OrderResult] = {}  # order_id -> OrderResult

    @property
    def name(self) -> str:
        """Client identifier."""
        return self._name

    @property
    def is_connected(self) -> bool:
        """Connection status."""
        return self._connected

    @property
    def open_orders(self) -> dict[str, OrderResult]:
        """Currently tracked open orders."""
        return self._open_orders.copy()

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> BaseExecutionClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Connect to execution venue."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from execution venue."""
        ...

    @abstractmethod
    async def submit_order(self, order: Order) -> OrderResult:
        """Submit order."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order."""
        ...

    @abstractmethod
    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders."""
        ...

    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        price: Any | None = None,
        amount: Any | None = None,
    ) -> OrderResult:
        """Modify order."""
        ...

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order status."""
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get open orders."""
        ...

    @abstractmethod
    async def get_order_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[OrderResult]:
        """Get order history."""
        ...

    @abstractmethod
    def stream_order_updates(self) -> AsyncIterator[OrderResult]:
        """Stream order updates."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get positions."""
        ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """Get position for symbol."""
        ...

    @abstractmethod
    async def get_balances(self) -> dict[str, Balance]:
        """Get balances."""
        ...

    @abstractmethod
    async def get_balance(self, currency: str) -> Balance | None:
        """Get balance for currency."""
        ...

    @abstractmethod
    async def reconcile_orders(self) -> int:
        """Reconcile orders."""
        ...

    @abstractmethod
    async def reconcile_positions(self) -> int:
        """Reconcile positions."""
        ...


# =============================================================================
# Exceptions
# =============================================================================


class ExecutionClientError(Exception):
    """Base exception for execution client errors."""


class OrderError(ExecutionClientError):
    """Order-related error."""


class OrderNotFoundError(OrderError):
    """Order not found."""


class InsufficientFundsError(OrderError):
    """Insufficient funds for order."""


class OrderRejectedError(OrderError):
    """Order rejected by exchange."""


class ReconciliationError(ExecutionClientError):
    """Reconciliation failed."""

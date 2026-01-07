"""
CCXTExecutionClient: Order execution via CCXT.

Supports 100+ exchanges through ccxt.pro for async order management.

Features:
- Async order submission/cancellation/modification
- WebSocket streaming for order updates
- Position and balance tracking
- Automatic reconnection with exponential backoff
- Order reconciliation on startup
- Rate limit handling

Performance optimizations:
- Uses ccxt.pro for native async support
- orjson enabled for 10x faster JSON parsing
- Connection pooling for REST calls

Usage:
    config = {
        "api_key": "...",
        "secret": "...",
        "testnet": True,
    }

    async with CCXTExecutionClient("binance", config) as client:
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
        )
        result = await client.submit_order(order)
        print(f"Order {result.order_id}: {result.status}")

        async for update in client.stream_order_updates():
            print(f"Order {update.order_id} status: {update.status}")

See: https://github.com/windoliver/libra/issues/33
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.clients.execution_client import (
    BaseExecutionClient,
    ExecutionClientError,
    InsufficientFundsError,
    OrderError,
    OrderNotFoundError,
    OrderRejectedError,
    ReconciliationError,
)
from libra.gateways.protocol import (
    Balance,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TimeInForce,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)


# =============================================================================
# CCXTExecutionClient Implementation
# =============================================================================


class CCXTExecutionClient(BaseExecutionClient):
    """
    CCXT-based execution client supporting 100+ exchanges.

    Uses ccxt.pro for async WebSocket order updates.

    Supported exchanges (partial list):
    - binance, binanceusdm (futures)
    - bybit
    - okx
    - kraken
    - coinbase
    - kucoin
    - gate
    - huobi
    - ... and 100+ more

    Configuration:
        config = {
            "api_key": "your-api-key",
            "secret": "your-secret",
            "password": "passphrase",  # Optional, for some exchanges
            "testnet": True,  # Use testnet/sandbox
            "options": {
                "defaultType": "future",  # spot, future, swap
            },
        }

    Example:
        client = CCXTExecutionClient("binance", config)
        await client.connect()

        # Submit order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.001"),
        )
        result = await client.submit_order(order)

        # Monitor order updates
        async for update in client.stream_order_updates():
            if update.status == OrderStatus.FILLED:
                print(f"Order {update.order_id} filled!")

        await client.disconnect()
    """

    # Reconnection settings
    RECONNECT_DELAY_MIN = 1.0
    RECONNECT_DELAY_MAX = 60.0
    RECONNECT_BACKOFF_MULTIPLIER = 2.0

    def __init__(
        self,
        exchange_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize CCXT execution client.

        Args:
            exchange_id: CCXT exchange ID (e.g., "binance", "bybit")
            config: Configuration with API credentials and options
        """
        super().__init__(name=f"{exchange_id}-exec", config=config)
        self._exchange_id = exchange_id
        self._exchange: Any = None  # ccxt.pro exchange instance
        self._order_update_queue: asyncio.Queue[OrderResult] = asyncio.Queue(
            maxsize=10000
        )
        self._stream_tasks: list[asyncio.Task[Any]] = []
        self._stop_streaming = asyncio.Event()
        self._position_cache: dict[str, Position] = {}
        self._balance_cache: dict[str, Balance] = {}
        self._last_reconcile_time: float = 0.0

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to the exchange.

        Initializes CCXT exchange, loads markets, and authenticates.

        Raises:
            ConnectionError: If connection fails
            ExecutionClientError: If authentication fails
        """
        if self._connected:
            logger.warning(f"{self.name}: Already connected")
            return

        try:
            # Import ccxt.pro (async version)
            import ccxt.pro as ccxtpro

            # Get exchange class
            exchange_class = getattr(ccxtpro, self._exchange_id, None)
            if exchange_class is None:
                raise ConnectionError(f"Unknown exchange: {self._exchange_id}")

            # Build exchange options
            options: dict[str, Any] = {
                "enableRateLimit": True,
            }

            # Enable orjson for faster JSON parsing
            try:
                import orjson  # noqa: F401

                options["enableOrjson"] = True
                logger.debug(f"{self.name}: orjson enabled for faster parsing")
            except ImportError:
                logger.debug(f"{self.name}: orjson not available")

            # Configure testnet if requested
            if self._config.get("testnet", False):
                options["sandbox"] = True

            # Add user-provided options
            if "options" in self._config:
                options.update(self._config["options"])

            # Create exchange instance
            exchange_config: dict[str, Any] = {
                "apiKey": self._config.get("api_key"),
                "secret": self._config.get("secret"),
                "password": self._config.get("password"),
                "options": options,
            }

            self._exchange = exchange_class(exchange_config)

            # Load markets
            logger.info(f"{self.name}: Loading markets...")
            await self._exchange.load_markets()
            logger.info(f"{self.name}: Loaded {len(self._exchange.markets)} markets")

            # Verify authentication (required for execution)
            if not self._config.get("api_key"):
                raise ExecutionClientError(
                    "API credentials required for execution client"
                )

            try:
                await self._exchange.fetch_balance()
                logger.info(f"{self.name}: Authentication successful")
            except Exception as e:
                raise ExecutionClientError(f"Authentication failed: {e}") from e

            self._connected = True
            self._stop_streaming.clear()

            # Start order update streaming
            task = asyncio.create_task(
                self._stream_my_trades(),
                name="order_stream",
            )
            self._stream_tasks.append(task)

            # Initial reconciliation
            await self.reconcile_orders()
            await self.reconcile_positions()

            logger.info(f"{self.name}: Connected successfully")

        except ExecutionClientError:
            raise
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.name}: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from the exchange.

        Stops all streaming tasks and closes connections.
        Does NOT cancel open orders.
        """
        if not self._connected:
            return

        logger.info(f"{self.name}: Disconnecting...")

        # Signal streams to stop
        self._stop_streaming.set()

        # Cancel streaming tasks
        for task in self._stream_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._stream_tasks.clear()

        # Close exchange connection
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.warning(f"{self.name}: Error closing connection: {e}")

        self._connected = False
        logger.info(f"{self.name}: Disconnected")

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order to the exchange.

        Args:
            order: Order to submit

        Returns:
            OrderResult with status and fill info

        Raises:
            OrderError: If order is rejected
            InsufficientFundsError: If not enough balance
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            # Build CCXT order params
            params: dict[str, Any] = {}

            if order.reduce_only:
                params["reduceOnly"] = True

            if order.post_only:
                params["postOnly"] = True

            if order.client_order_id:
                params["clientOrderId"] = order.client_order_id

            if order.time_in_force != TimeInForce.GTC:
                params["timeInForce"] = order.time_in_force.value

            # Submit order via CCXT
            if order.order_type == OrderType.MARKET:
                result = await self._exchange.create_order(
                    symbol=order.symbol,
                    type="market",
                    side=order.side.value,
                    amount=float(order.amount),
                    params=params,
                )
            elif order.order_type == OrderType.LIMIT:
                if order.price is None:
                    raise OrderError("Limit order requires price")
                result = await self._exchange.create_order(
                    symbol=order.symbol,
                    type="limit",
                    side=order.side.value,
                    amount=float(order.amount),
                    price=float(order.price),
                    params=params,
                )
            elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
                if order.stop_price is None:
                    raise OrderError("Stop order requires stop_price")
                params["stopPrice"] = float(order.stop_price)

                if order.order_type == OrderType.STOP_LIMIT:
                    if order.price is None:
                        raise OrderError("Stop-limit order requires price")
                    result = await self._exchange.create_order(
                        symbol=order.symbol,
                        type="stopLimit",
                        side=order.side.value,
                        amount=float(order.amount),
                        price=float(order.price),
                        params=params,
                    )
                else:
                    result = await self._exchange.create_order(
                        symbol=order.symbol,
                        type="stopMarket",
                        side=order.side.value,
                        amount=float(order.amount),
                        params=params,
                    )
            else:
                raise OrderError(f"Unsupported order type: {order.order_type}")

            order_result = self._convert_order_result(result)

            # Track open orders
            if order_result.status in (
                OrderStatus.OPEN,
                OrderStatus.PENDING,
                OrderStatus.PARTIALLY_FILLED,
            ):
                self._open_orders[order_result.order_id] = order_result

            logger.info(
                f"{self.name}: Order submitted - {order_result.order_id} "
                f"{order_result.status}"
            )
            return order_result

        except Exception as e:
            error_str = str(e).lower()
            if "insufficient" in error_str or "balance" in error_str:
                raise InsufficientFundsError(str(e)) from e
            if "rejected" in error_str:
                raise OrderRejectedError(str(e)) from e
            if "rate" in error_str and "limit" in error_str:
                raise OrderError(f"Rate limit exceeded: {e}") from e
            raise OrderError(str(e)) from e

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            True if cancelled, False if already closed
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            await self._exchange.cancel_order(order_id, symbol)

            # Remove from tracked orders
            self._open_orders.pop(order_id, None)

            logger.info(f"{self.name}: Order {order_id} cancelled")
            return True

        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "not exist" in error_str:
                raise OrderNotFoundError(f"Order {order_id} not found") from e
            if "already" in error_str and (
                "filled" in error_str or "cancelled" in error_str
            ):
                self._open_orders.pop(order_id, None)
                return False
            raise OrderError(str(e)) from e

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional symbol to filter

        Returns:
            Number of orders cancelled
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            if symbol:
                result = await self._exchange.cancel_all_orders(symbol)
            else:
                result = await self._exchange.cancel_all_orders()

            # Clear tracked orders
            if symbol:
                self._open_orders = {
                    oid: o
                    for oid, o in self._open_orders.items()
                    if o.symbol != symbol
                }
            else:
                self._open_orders.clear()

            count = len(result) if isinstance(result, list) else 0
            logger.info(f"{self.name}: Cancelled {count} orders")
            return count

        except Exception as e:
            raise OrderError(f"Failed to cancel orders: {e}") from e

    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        price: Any | None = None,
        amount: Any | None = None,
    ) -> OrderResult:
        """
        Modify an existing order.

        Some exchanges support native modification; others cancel and replace.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair
            price: New price (None = keep current)
            amount: New amount (None = keep current)

        Returns:
            OrderResult with updated order state
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        if price is None and amount is None:
            raise OrderError("Must provide price or amount to modify")

        try:
            # Try native edit if exchange supports it
            if hasattr(self._exchange, "edit_order"):
                params = {}
                edit_price = float(price) if price is not None else None
                edit_amount = float(amount) if amount is not None else None

                result = await self._exchange.edit_order(
                    order_id,
                    symbol,
                    type=None,  # Keep same type
                    side=None,  # Keep same side
                    amount=edit_amount,
                    price=edit_price,
                    params=params,
                )
                order_result = self._convert_order_result(result)
                self._open_orders[order_result.order_id] = order_result
                return order_result

            # Fallback: cancel and replace
            # First get current order details
            current = await self.get_order(order_id, symbol)

            # Cancel the order
            await self.cancel_order(order_id, symbol)

            # Create new order with modifications
            new_order = Order(
                symbol=symbol,
                side=current.side,
                order_type=current.order_type,
                amount=Decimal(str(amount)) if amount else current.amount,
                price=Decimal(str(price)) if price else current.price,
            )

            return await self.submit_order(new_order)

        except OrderNotFoundError:
            raise
        except Exception as e:
            raise OrderError(f"Failed to modify order: {e}") from e

    # -------------------------------------------------------------------------
    # Order Queries
    # -------------------------------------------------------------------------

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """
        Get current status of an order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            Current OrderResult
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            result = await self._exchange.fetch_order(order_id, symbol)
            return self._convert_order_result(result)

        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "not exist" in error_str:
                raise OrderNotFoundError(f"Order {order_id} not found") from e
            raise ExecutionClientError(str(e)) from e

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol to filter

        Returns:
            List of open OrderResults
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            if symbol:
                orders = await self._exchange.fetch_open_orders(symbol)
            else:
                orders = await self._exchange.fetch_open_orders()

            return [self._convert_order_result(o) for o in orders]

        except Exception as e:
            raise ExecutionClientError(f"Failed to fetch open orders: {e}") from e

    async def get_order_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[OrderResult]:
        """
        Get historical orders.

        Args:
            symbol: Optional symbol to filter
            limit: Maximum number of orders to return

        Returns:
            List of OrderResults sorted by time descending
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            if symbol:
                orders = await self._exchange.fetch_closed_orders(symbol, limit=limit)
            else:
                orders = await self._exchange.fetch_closed_orders(limit=limit)

            return [self._convert_order_result(o) for o in orders]

        except Exception as e:
            raise ExecutionClientError(f"Failed to fetch order history: {e}") from e

    # -------------------------------------------------------------------------
    # Order Update Stream
    # -------------------------------------------------------------------------

    async def _stream_my_trades(self) -> None:
        """
        Internal order update streaming loop.

        Watches for order status changes and queues updates.
        """
        logger.info(f"{self.name}: Starting order update stream")
        reconnect_delay = self.RECONNECT_DELAY_MIN

        while not self._stop_streaming.is_set():
            try:
                # Use watch_my_trades for real-time order updates
                if hasattr(self._exchange, "watch_my_trades"):
                    trades = await self._exchange.watch_my_trades()
                    for trade in trades:
                        order_result = self._trade_to_order_result(trade)
                        await self._queue_order_update(order_result)
                        reconnect_delay = self.RECONNECT_DELAY_MIN  # Reset on success
                elif hasattr(self._exchange, "watch_orders"):
                    # Fallback to watch_orders if available
                    orders = await self._exchange.watch_orders()
                    for order in orders:
                        order_result = self._convert_order_result(order)
                        await self._queue_order_update(order_result)
                        reconnect_delay = self.RECONNECT_DELAY_MIN
                else:
                    # Polling fallback
                    await asyncio.sleep(1.0)
                    await self._poll_order_updates()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self.name}: Order stream error: {e}")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * self.RECONNECT_BACKOFF_MULTIPLIER,
                    self.RECONNECT_DELAY_MAX,
                )

        logger.info(f"{self.name}: Order update stream stopped")

    async def _poll_order_updates(self) -> None:
        """Poll open orders for status changes (fallback for exchanges without WS)."""
        try:
            current_orders = await self.get_open_orders()
            current_ids = {o.order_id for o in current_orders}

            # Check for closed orders
            for order_id, tracked in list(self._open_orders.items()):
                if order_id not in current_ids:
                    # Order was closed - fetch final status
                    try:
                        final = await self.get_order(order_id, tracked.symbol)
                        await self._queue_order_update(final)
                        del self._open_orders[order_id]
                    except OrderNotFoundError:
                        del self._open_orders[order_id]

            # Update tracked orders
            for order in current_orders:
                old = self._open_orders.get(order.order_id)
                if old is None or old.status != order.status:
                    await self._queue_order_update(order)
                self._open_orders[order.order_id] = order

        except Exception as e:
            logger.warning(f"{self.name}: Poll error: {e}")

    async def _queue_order_update(self, order_result: OrderResult) -> None:
        """Queue an order update, handling overflow."""
        try:
            self._order_update_queue.put_nowait(order_result)
        except asyncio.QueueFull:
            # Drop oldest if queue full
            try:
                self._order_update_queue.get_nowait()
                self._order_update_queue.put_nowait(order_result)
            except asyncio.QueueEmpty:
                pass

    async def stream_order_updates(self) -> AsyncIterator[OrderResult]:
        """
        Stream real-time order status updates.

        Yields OrderResult whenever an order's status changes.

        Yields:
            OrderResult objects on each status change
        """
        while not self._stop_streaming.is_set():
            try:
                update = await asyncio.wait_for(
                    self._order_update_queue.get(),
                    timeout=1.0,
                )
                yield update
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _trade_to_order_result(self, trade: dict[str, Any]) -> OrderResult:
        """Convert a trade update to OrderResult."""
        # Trades represent fills - reconstruct order state
        return OrderResult(
            order_id=str(trade.get("order", trade.get("id", ""))),
            symbol=trade.get("symbol", ""),
            status=OrderStatus.PARTIALLY_FILLED,  # Trades indicate fills
            side=OrderSide(trade.get("side", "buy")),
            order_type=OrderType.LIMIT,  # Assume limit
            amount=Decimal(str(trade.get("amount", 0) or 0)),
            filled_amount=Decimal(str(trade.get("amount", 0) or 0)),
            remaining_amount=Decimal("0"),
            average_price=Decimal(str(trade.get("price", 0) or 0)),
            fee=Decimal(str(trade.get("fee", {}).get("cost", 0) or 0)),
            fee_currency=trade.get("fee", {}).get("currency", ""),
            timestamp_ns=int((trade.get("timestamp", 0) or time.time() * 1000) * 1_000_000),
            created_ns=int((trade.get("timestamp", 0) or time.time() * 1000) * 1_000_000),
        )

    # -------------------------------------------------------------------------
    # Account State
    # -------------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of current positions with P&L
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            positions = await self._exchange.fetch_positions()
            result = [
                self._convert_position(p)
                for p in positions
                if float(p.get("contracts", 0) or p.get("position", 0) or 0) != 0
            ]

            # Update cache
            self._position_cache = {p.symbol: p for p in result}

            return result

        except Exception as e:
            # Spot exchanges don't support positions
            if "not supported" in str(e).lower():
                return []
            raise ExecutionClientError(f"Failed to fetch positions: {e}") from e

    async def get_position(self, symbol: str) -> Position | None:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position if exists, None otherwise
        """
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def get_balances(self) -> dict[str, Balance]:
        """
        Get account balances for all currencies.

        Returns:
            Dict mapping currency to Balance
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            data = await self._exchange.fetch_balance()
            balances: dict[str, Balance] = {}

            for currency, info in data.items():
                if isinstance(info, dict) and "total" in info:
                    total = Decimal(str(info.get("total", 0) or 0))
                    if total > 0:
                        balances[currency] = Balance(
                            currency=currency,
                            total=total,
                            available=Decimal(str(info.get("free", 0) or 0)),
                            locked=Decimal(str(info.get("used", 0) or 0)),
                        )

            # Update cache
            self._balance_cache = balances

            return balances

        except Exception as e:
            raise ExecutionClientError(f"Failed to fetch balances: {e}") from e

    async def get_balance(self, currency: str) -> Balance | None:
        """
        Get balance for a specific currency.

        Args:
            currency: Currency code (e.g., "USDT")

        Returns:
            Balance if exists, None otherwise
        """
        balances = await self.get_balances()
        return balances.get(currency)

    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------

    async def reconcile_orders(self) -> int:
        """
        Reconcile local order state with exchange state.

        Fetches all open orders from exchange and updates local tracking.
        Detects orders that changed while disconnected.

        Returns:
            Number of orders reconciled (state changes detected)
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            exchange_orders = await self.get_open_orders()
            exchange_ids = {o.order_id for o in exchange_orders}
            changes = 0

            # Detect orders that closed while we were away
            for order_id, tracked in list(self._open_orders.items()):
                if order_id not in exchange_ids:
                    # Order is no longer open - fetch final status
                    try:
                        final = await self.get_order(order_id, tracked.symbol)
                        if final.status != tracked.status:
                            await self._queue_order_update(final)
                            changes += 1
                    except OrderNotFoundError:
                        pass
                    del self._open_orders[order_id]

            # Update/add current open orders
            for order in exchange_orders:
                old = self._open_orders.get(order.order_id)
                if old is None or old.status != order.status:
                    changes += 1
                self._open_orders[order.order_id] = order

            self._last_reconcile_time = time.time()
            logger.info(f"{self.name}: Reconciled orders - {changes} changes detected")
            return changes

        except Exception as e:
            raise ReconciliationError(f"Order reconciliation failed: {e}") from e

    async def reconcile_positions(self) -> int:
        """
        Reconcile local position state with exchange state.

        Returns:
            Number of positions reconciled
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        try:
            positions = await self.get_positions()
            old_cache = self._position_cache.copy()
            changes = 0

            # Compare with cached positions
            for pos in positions:
                old_pos = old_cache.get(pos.symbol)
                if old_pos is None or old_pos.amount != pos.amount:
                    changes += 1

            # Check for closed positions
            for symbol in old_cache:
                if symbol not in {p.symbol for p in positions}:
                    changes += 1

            logger.info(
                f"{self.name}: Reconciled positions - {changes} changes detected"
            )
            return changes

        except Exception as e:
            raise ReconciliationError(f"Position reconciliation failed: {e}") from e

    # -------------------------------------------------------------------------
    # Conversion Helpers
    # -------------------------------------------------------------------------

    def _convert_order_result(self, data: dict[str, Any]) -> OrderResult:
        """Convert CCXT order to OrderResult."""
        # Map CCXT status to OrderStatus
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        status = status_map.get(data.get("status", ""), OrderStatus.PENDING)

        # Check for partial fill
        filled = Decimal(str(data.get("filled", 0) or 0))
        amount = Decimal(str(data.get("amount", 0) or 0))
        if status == OrderStatus.OPEN and filled > 0:
            status = OrderStatus.PARTIALLY_FILLED

        # Extract fee info
        fee_data = data.get("fee") or {}
        fee = Decimal(str(fee_data.get("cost", 0) or 0))
        fee_currency = fee_data.get("currency", "")

        # Map order type
        type_str = data.get("type", "market")
        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
            "stopLimit": OrderType.STOP_LIMIT,
            "stopMarket": OrderType.STOP,
        }
        order_type = type_map.get(type_str, OrderType.MARKET)

        return OrderResult(
            order_id=str(data.get("id", "")),
            symbol=data.get("symbol", ""),
            status=status,
            side=OrderSide(data.get("side", "buy")),
            order_type=order_type,
            amount=amount,
            filled_amount=filled,
            remaining_amount=Decimal(str(data.get("remaining", 0) or 0)),
            average_price=(
                Decimal(str(data.get("average", 0) or 0))
                if data.get("average")
                else None
            ),
            fee=fee,
            fee_currency=fee_currency,
            timestamp_ns=int(
                (data.get("timestamp", 0) or time.time() * 1000) * 1_000_000
            ),
            created_ns=int(
                (data.get("timestamp", 0) or time.time() * 1000) * 1_000_000
            ),
            client_order_id=data.get("clientOrderId"),
            price=(
                Decimal(str(data.get("price", 0) or 0)) if data.get("price") else None
            ),
            stop_price=(
                Decimal(str(data.get("stopPrice", 0) or 0))
                if data.get("stopPrice")
                else None
            ),
        )

    def _convert_position(self, data: dict[str, Any]) -> Position:
        """Convert CCXT position to Position."""
        side_str = data.get("side", "long")
        if side_str == "long":
            side = PositionSide.LONG
        elif side_str == "short":
            side = PositionSide.SHORT
        else:
            side = PositionSide.FLAT

        # Handle different position amount fields
        amount = abs(
            float(
                data.get("contracts", 0)
                or data.get("position", 0)
                or data.get("positionAmt", 0)
                or 0
            )
        )

        return Position(
            symbol=data.get("symbol", ""),
            side=side,
            amount=Decimal(str(amount)),
            entry_price=Decimal(str(data.get("entryPrice", 0) or 0)),
            current_price=Decimal(
                str(data.get("markPrice", 0) or data.get("entryPrice", 0) or 0)
            ),
            unrealized_pnl=Decimal(str(data.get("unrealizedPnl", 0) or 0)),
            realized_pnl=Decimal(str(data.get("realizedPnl", 0) or 0)),
            leverage=int(data.get("leverage", 1) or 1),
            liquidation_price=(
                Decimal(str(data.get("liquidationPrice", 0)))
                if data.get("liquidationPrice")
                else None
            ),
            margin=(
                Decimal(str(data.get("initialMargin", 0) or 0))
                if data.get("initialMargin")
                else None
            ),
            margin_type=data.get("marginType"),
            timestamp_ns=int(
                (data.get("timestamp", 0) or time.time() * 1000) * 1_000_000
            ),
        )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_market_info(self, symbol: str) -> dict[str, Any] | None:
        """
        Get market information for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Market info dict or None if not found
        """
        if self._exchange and symbol in self._exchange.markets:
            return self._exchange.markets[symbol]
        return None

    @property
    def exchange_id(self) -> str:
        """Get CCXT exchange ID."""
        return self._exchange_id

    @property
    def exchange(self) -> Any:
        """Get underlying CCXT exchange instance (for advanced usage)."""
        return self._exchange

    @property
    def cached_positions(self) -> dict[str, Position]:
        """Get cached positions (may be stale)."""
        return self._position_cache.copy()

    @property
    def cached_balances(self) -> dict[str, Balance]:
        """Get cached balances (may be stale)."""
        return self._balance_cache.copy()

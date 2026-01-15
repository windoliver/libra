"""
Alpaca Gateway Implementation.

Provides execution and market data via Alpaca's API.
Supports stocks, ETFs, and options (single-leg and multi-leg).

Issue #61: Alpaca Gateway - Stock & Options Execution
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.core.sessions import MarketSessionManager
from libra.gateways.alpaca.config import AlpacaConfig
from libra.gateways.alpaca.symbols import (
    is_option_symbol,
    normalize_symbol,
)
from libra.gateways.protocol import (
    Balance,
    BaseGateway,
    GatewayCapabilities,
    GatewayError,
    MarketClosedError,
    Order,
    OrderBook,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    RateLimitError,
    Tick,
    TimeInForce,
)


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# Alpaca gateway capabilities
ALPACA_CAPABILITIES = GatewayCapabilities(
    # Order types
    market_orders=True,
    limit_orders=True,
    stop_orders=True,
    stop_limit_orders=True,
    trailing_stop_orders=True,
    bracket_orders=True,
    oco_orders=True,
    oto_orders=True,
    # Time-in-force
    tif_gtc=True,
    tif_ioc=True,
    tif_fok=True,
    tif_day=True,
    # Market data
    streaming_ticks=True,
    streaming_orderbook=False,  # Not available via WebSocket
    streaming_trades=True,
    historical_bars=True,
    historical_trades=True,
    # Trading features
    margin_trading=True,
    futures_trading=False,
    options_trading=True,
    short_selling=True,
    reduce_only=False,
    post_only=False,
    # Position management
    position_tracking=True,
    hedge_mode=False,
    # Rate limits
    max_orders_per_second=10,
    max_requests_per_minute=200,
)


# Order type mappings
LIBRA_TO_ALPACA_ORDER_TYPE = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
    OrderType.STOP: "stop",
    OrderType.STOP_LIMIT: "stop_limit",
}

ALPACA_TO_LIBRA_ORDER_TYPE = {v: k for k, v in LIBRA_TO_ALPACA_ORDER_TYPE.items()}

# Order side mappings
LIBRA_TO_ALPACA_SIDE = {
    OrderSide.BUY: "buy",
    OrderSide.SELL: "sell",
}

ALPACA_TO_LIBRA_SIDE = {v: k for k, v in LIBRA_TO_ALPACA_SIDE.items()}

# Time-in-force mappings
LIBRA_TO_ALPACA_TIF = {
    TimeInForce.DAY: "day",
    TimeInForce.GTC: "gtc",
    TimeInForce.IOC: "ioc",
    TimeInForce.FOK: "fok",
}

ALPACA_TO_LIBRA_TIF = {v: k for k, v in LIBRA_TO_ALPACA_TIF.items()}

# Order status mappings
ALPACA_TO_LIBRA_STATUS = {
    "new": OrderStatus.OPEN,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.EXPIRED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.OPEN,
    "pending_replace": OrderStatus.OPEN,
    "pending_new": OrderStatus.PENDING,
    "accepted": OrderStatus.OPEN,
    "accepted_for_bidding": OrderStatus.OPEN,
    "stopped": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.OPEN,
    "calculated": OrderStatus.OPEN,
}


class AlpacaGateway(BaseGateway):
    """
    Alpaca gateway for US stocks and options.

    Features:
    - Commission-free stock/ETF trading
    - Options trading (Level 1-3)
    - Paper trading mode
    - Real-time WebSocket streaming
    - Bracket orders (entry + take-profit + stop-loss)

    Example:
        config = AlpacaConfig.from_env(paper=True)
        gateway = AlpacaGateway(config)

        async with gateway:
            # Place market order
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("10"),
            )
            result = await gateway.submit_order(order)
            print(f"Order {result.order_id}: {result.status}")

            # Get positions
            positions = await gateway.get_positions()
            for pos in positions:
                print(f"{pos.symbol}: {pos.unrealized_pnl}")
    """

    def __init__(self, config: AlpacaConfig) -> None:
        """
        Initialize Alpaca gateway.

        Args:
            config: Alpaca gateway configuration
        """
        # Create session manager for market hours validation (Issue #62)
        session_manager = MarketSessionManager()
        super().__init__(name="alpaca", config=config.__dict__, session_manager=session_manager)
        self._config = config
        self._trading_client: Any = None
        self._data_client: Any = None
        self._option_data_client: Any = None
        self._data_stream: Any = None
        self._trading_stream: Any = None
        self._tick_queue: asyncio.Queue[Tick] = asyncio.Queue()
        self._stream_task: asyncio.Task | None = None
        self._last_request_time: float = 0.0
        self._request_count: int = 0

    @property
    def capabilities(self) -> GatewayCapabilities:
        """Get gateway capabilities."""
        return ALPACA_CAPABILITIES

    async def connect(self) -> None:
        """
        Connect to Alpaca API.

        Initializes trading and data clients.
        Verifies account status and trading permissions.
        """
        if self._connected:
            return

        try:
            # Import Alpaca SDK
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.live import StockDataStream
            from alpaca.trading.stream import TradingStream
        except ImportError as e:
            raise ImportError(
                "alpaca-py is not installed. Install with: pip install alpaca-py"
            ) from e

        logger.info(
            "Connecting to Alpaca (%s mode, %s feed)",
            "paper" if self._config.paper else "live",
            self._config.data_feed,
        )

        # Initialize trading client
        self._trading_client = TradingClient(
            api_key=self._config.credentials.api_key,
            secret_key=self._config.credentials.secret_key,
            paper=self._config.paper,
        )

        # Verify account
        try:
            account = self._trading_client.get_account()
            if account.trading_blocked:
                raise GatewayError("Account trading is blocked")
            if account.account_blocked:
                raise GatewayError("Account is blocked")

            logger.info(
                "Alpaca account connected: %s (Equity: $%s, Buying Power: $%s)",
                account.account_number,
                account.equity,
                account.buying_power,
            )
        except Exception as e:
            raise GatewayError(f"Failed to verify Alpaca account: {e}") from e

        # Initialize data client
        self._data_client = StockHistoricalDataClient(
            api_key=self._config.credentials.api_key,
            secret_key=self._config.credentials.secret_key,
        )

        # Initialize streaming clients
        self._data_stream = StockDataStream(
            api_key=self._config.credentials.api_key,
            secret_key=self._config.credentials.secret_key,
            feed=self._config.data_feed,
        )

        self._trading_stream = TradingStream(
            api_key=self._config.credentials.api_key,
            secret_key=self._config.credentials.secret_key,
            paper=self._config.paper,
        )

        self._connected = True
        logger.info("Alpaca gateway connected")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        if not self._connected:
            return

        logger.info("Disconnecting from Alpaca")

        # Stop streaming
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._data_stream:
            try:
                await self._data_stream.stop_ws()
            except Exception:
                pass

        if self._trading_stream:
            try:
                await self._trading_stream.stop_ws()
            except Exception:
                pass

        self._trading_client = None
        self._data_client = None
        self._data_stream = None
        self._trading_stream = None
        self._connected = False

        logger.info("Alpaca gateway disconnected")

    async def _rate_limit(self) -> None:
        """Apply rate limiting to avoid 429 errors."""
        now = time.time()
        elapsed = now - self._last_request_time

        # Reset counter every minute
        if elapsed >= 60:
            self._request_count = 0
            self._last_request_time = now

        # Check if at limit
        if self._request_count >= self._config.rate_limit_per_minute:
            wait_time = 60 - elapsed
            if wait_time > 0:
                logger.warning("Rate limit reached, waiting %.1f seconds", wait_time)
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_request_time = time.time()

        self._request_count += 1

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time market data."""
        if not self._connected:
            raise GatewayError("Gateway not connected")

        normalized = [normalize_symbol(s) for s in symbols]
        self._subscribed_symbols.update(normalized)

        # Set up quote handler
        async def on_quote(quote: Any) -> None:
            tick = Tick(
                symbol=quote.symbol,
                bid=Decimal(str(quote.bid_price)),
                ask=Decimal(str(quote.ask_price)),
                last=Decimal(str(quote.bid_price)),  # Use bid as proxy
                timestamp_ns=int(quote.timestamp.timestamp() * 1_000_000_000),
                bid_size=Decimal(str(quote.bid_size)),
                ask_size=Decimal(str(quote.ask_size)),
            )
            await self._tick_queue.put(tick)

        # Subscribe
        self._data_stream.subscribe_quotes(on_quote, *normalized)

        # Start streaming if not already running
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._run_stream())

        logger.info("Subscribed to quotes: %s", normalized)

    async def _run_stream(self) -> None:
        """Run the WebSocket stream."""
        try:
            await self._data_stream._run_forever()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Stream error: %s", e)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from market data."""
        normalized = [normalize_symbol(s) for s in symbols]
        self._subscribed_symbols.difference_update(normalized)

        if self._data_stream:
            self._data_stream.unsubscribe_quotes(*normalized)

        logger.info("Unsubscribed from quotes: %s", normalized)

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """Stream real-time tick data."""
        while self._connected:
            try:
                tick = await asyncio.wait_for(self._tick_queue.get(), timeout=1.0)
                yield tick
            except asyncio.TimeoutError:
                continue

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """
        Get order book snapshot.

        Note: Alpaca doesn't provide full order book via API.
        Returns best bid/ask only.
        """
        await self._rate_limit()
        ticker = await self.get_ticker(symbol)
        return OrderBook(
            symbol=symbol,
            bids=[(ticker.bid, ticker.bid_size or Decimal("0"))],
            asks=[(ticker.ask, ticker.ask_size or Decimal("0"))],
            timestamp_ns=ticker.timestamp_ns,
        )

    async def get_ticker(self, symbol: str) -> Tick:
        """Get current quote for a symbol."""
        await self._rate_limit()

        if not self._data_client:
            raise GatewayError("Gateway not connected")

        from alpaca.data.requests import StockLatestQuoteRequest

        symbol = normalize_symbol(symbol)
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)

        try:
            quotes = self._data_client.get_stock_latest_quote(request)
            quote = quotes[symbol]

            return Tick(
                symbol=symbol,
                bid=Decimal(str(quote.bid_price)),
                ask=Decimal(str(quote.ask_price)),
                last=Decimal(str((quote.bid_price + quote.ask_price) / 2)),
                timestamp_ns=int(quote.timestamp.timestamp() * 1_000_000_000),
                bid_size=Decimal(str(quote.bid_size)),
                ask_size=Decimal(str(quote.ask_size)),
            )
        except Exception as e:
            raise GatewayError(f"Failed to get ticker for {symbol}: {e}") from e

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order to Alpaca.

        Supports stocks and options (single-leg).
        For multi-leg options, use submit_multileg_order().

        Validates market session before submission (Issue #62):
        - Regular hours: All order types allowed
        - Extended hours: Only allowed if order.extended_hours=True
        - Market closed: Raises MarketClosedError
        """
        # Validate market session (Issue #62)
        valid, reason = self.validate_session(order)
        if not valid:
            raise MarketClosedError(reason or "Market is closed")

        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        # Determine if this is an option order
        if is_option_symbol(order.symbol):
            return await self._submit_option_order(order)
        return await self._submit_stock_order(order)

    async def _submit_stock_order(self, order: Order) -> OrderResult:
        """Submit a stock order."""
        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
            StopLimitOrderRequest,
        )
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF

        symbol = normalize_symbol(order.symbol)
        side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
        tif = AlpacaTIF(LIBRA_TO_ALPACA_TIF.get(order.time_in_force, "day"))

        try:
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=float(order.amount),
                    side=side,
                    time_in_force=tif,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.LIMIT:
                if order.price is None:
                    raise GatewayError("Limit order requires price")
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=float(order.amount),
                    side=side,
                    time_in_force=tif,
                    limit_price=float(order.price),
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.STOP:
                if order.stop_price is None:
                    raise GatewayError("Stop order requires stop_price")
                request = StopOrderRequest(
                    symbol=symbol,
                    qty=float(order.amount),
                    side=side,
                    time_in_force=tif,
                    stop_price=float(order.stop_price),
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.STOP_LIMIT:
                if order.price is None or order.stop_price is None:
                    raise GatewayError("Stop-limit order requires price and stop_price")
                request = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=float(order.amount),
                    side=side,
                    time_in_force=tif,
                    limit_price=float(order.price),
                    stop_price=float(order.stop_price),
                    client_order_id=order.client_order_id,
                )
            else:
                raise GatewayError(f"Unsupported order type: {order.order_type}")

            result = self._trading_client.submit_order(request)
            return self._convert_order_result(result)

        except Exception as e:
            if "rate limit" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded: {e}") from e
            raise GatewayError(f"Failed to submit order: {e}") from e

    async def _submit_option_order(self, order: Order) -> OrderResult:
        """Submit a single-leg option order."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF

        # Options must use limit orders
        if order.order_type != OrderType.LIMIT:
            raise GatewayError("Options orders must be limit orders")
        if order.price is None:
            raise GatewayError("Options order requires price")

        side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL

        try:
            request = LimitOrderRequest(
                symbol=order.symbol,  # OCC format
                qty=float(order.amount),
                side=side,
                time_in_force=AlpacaTIF.DAY,  # Options only support day orders
                limit_price=float(order.price),
                client_order_id=order.client_order_id,
            )

            result = self._trading_client.submit_order(request)
            return self._convert_order_result(result)

        except Exception as e:
            raise GatewayError(f"Failed to submit option order: {e}") from e

    async def submit_multileg_order(
        self,
        legs: list[dict[str, Any]],
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        extended_hours: bool = False,
    ) -> OrderResult:
        """
        Submit a multi-leg options order.

        Args:
            legs: List of leg definitions, each with:
                - symbol: OCC option symbol
                - side: "buy" or "sell"
                - qty: Quantity (integer)
            order_type: MARKET or LIMIT
            limit_price: Net debit/credit for limit orders
            time_in_force: Time-in-force (DAY only for options)
            extended_hours: Allow extended hours trading (Issue #62)

        Returns:
            OrderResult for the multi-leg order

        Example:
            # Bull call spread
            result = await gateway.submit_multileg_order(
                legs=[
                    {"symbol": "AAPL250117C00150000", "side": "buy", "qty": 1},
                    {"symbol": "AAPL250117C00160000", "side": "sell", "qty": 1},
                ],
                order_type=OrderType.LIMIT,
                limit_price=Decimal("2.50"),
            )
        """
        # Validate market session (Issue #62)
        # Create a dummy order for validation
        dummy_order = Order(
            symbol=legs[0]["symbol"] if legs else "",
            side=OrderSide.BUY,
            order_type=order_type,
            amount=Decimal("1"),
            extended_hours=extended_hours,
        )
        valid, reason = self.validate_session(dummy_order)
        if not valid:
            raise MarketClosedError(reason or "Market is closed")

        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        if self._config.options_level < 3:
            raise GatewayError("Multi-leg orders require options level 3")

        try:
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide
            from alpaca.trading.enums import OrderClass, OrderType as AlpacaOrderType

            # Build leg requests
            leg_requests = []
            for leg in legs:
                from alpaca.trading.requests import OptionLegRequest

                side = (
                    AlpacaOrderSide.BUY
                    if leg["side"].lower() == "buy"
                    else AlpacaOrderSide.SELL
                )
                leg_requests.append(
                    OptionLegRequest(
                        symbol=leg["symbol"],
                        side=side,
                        qty=int(leg["qty"]),
                    )
                )

            # Build order kwargs
            order_kwargs: dict[str, Any] = {
                "legs": leg_requests,
                "order_class": OrderClass.MLEG,
                "time_in_force": "day",
            }

            if order_type == OrderType.LIMIT:
                order_kwargs["type"] = AlpacaOrderType.LIMIT
                if limit_price is None:
                    raise GatewayError("Limit price required for limit orders")
                order_kwargs["limit_price"] = float(limit_price)
            else:
                order_kwargs["type"] = AlpacaOrderType.MARKET

            result = self._trading_client.submit_order(**order_kwargs)
            return self._convert_order_result(result)

        except Exception as e:
            raise GatewayError(f"Failed to submit multi-leg order: {e}") from e

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info("Cancelled order: %s", order_id)
            return True
        except Exception as e:
            if "not found" in str(e).lower():
                return False
            raise GatewayError(f"Failed to cancel order {order_id}: {e}") from e

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all open orders."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            responses = self._trading_client.cancel_orders()
            count = len(responses) if responses else 0
            logger.info("Cancelled %d orders", count)
            return count
        except Exception as e:
            raise GatewayError(f"Failed to cancel orders: {e}") from e

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order status."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            order = self._trading_client.get_order_by_id(order_id)
            return self._convert_order_result(order)
        except Exception as e:
            raise GatewayError(f"Failed to get order {order_id}: {e}") from e

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get all open orders."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._trading_client.get_orders(request)

            results = [self._convert_order_result(o) for o in orders]

            if symbol:
                symbol = normalize_symbol(symbol)
                results = [r for r in results if r.symbol == symbol]

            return results
        except Exception as e:
            raise GatewayError(f"Failed to get open orders: {e}") from e

    async def get_order_history(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list[OrderResult]:
        """Get order history."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=limit or 100,
            )
            orders = self._trading_client.get_orders(request)

            results = [self._convert_order_result(o) for o in orders]

            if symbol:
                symbol = normalize_symbol(symbol)
                results = [r for r in results if r.symbol == symbol]

            return results
        except Exception as e:
            raise GatewayError(f"Failed to get order history: {e}") from e

    async def get_trades(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list:
        """Get trade/fill history."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=limit or 100,
            )
            orders = self._trading_client.get_orders(request)

            # Filter to only filled orders
            filled = [o for o in orders if o.status == "filled"]

            if symbol:
                symbol = normalize_symbol(symbol)
                filled = [o for o in filled if o.symbol == symbol]

            return filled
        except Exception as e:
            raise GatewayError(f"Failed to get trades: {e}") from e

    def _convert_order_result(self, alpaca_order: Any) -> OrderResult:
        """Convert Alpaca order to LIBRA OrderResult."""
        status = ALPACA_TO_LIBRA_STATUS.get(alpaca_order.status, OrderStatus.PENDING)
        side = ALPACA_TO_LIBRA_SIDE.get(alpaca_order.side, OrderSide.BUY)
        order_type = ALPACA_TO_LIBRA_ORDER_TYPE.get(
            alpaca_order.type, OrderType.MARKET
        )

        filled_qty = Decimal(str(alpaca_order.filled_qty or 0))
        qty = Decimal(str(alpaca_order.qty or 0))

        return OrderResult(
            order_id=str(alpaca_order.id),
            symbol=alpaca_order.symbol,
            status=status,
            side=side,
            order_type=order_type,
            amount=qty,
            filled_amount=filled_qty,
            remaining_amount=qty - filled_qty,
            average_price=(
                Decimal(str(alpaca_order.filled_avg_price))
                if alpaca_order.filled_avg_price
                else None
            ),
            fee=Decimal("0"),  # Alpaca is commission-free
            fee_currency="USD",
            timestamp_ns=int(alpaca_order.updated_at.timestamp() * 1_000_000_000),
            created_ns=(
                int(alpaca_order.created_at.timestamp() * 1_000_000_000)
                if alpaca_order.created_at
                else None
            ),
            client_order_id=alpaca_order.client_order_id,
            price=(
                Decimal(str(alpaca_order.limit_price))
                if alpaca_order.limit_price
                else None
            ),
            stop_price=(
                Decimal(str(alpaca_order.stop_price))
                if alpaca_order.stop_price
                else None
            ),
        )

    # -------------------------------------------------------------------------
    # Positions & Account
    # -------------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            positions = self._trading_client.get_all_positions()
            return [self._convert_position(p) for p in positions]
        except Exception as e:
            raise GatewayError(f"Failed to get positions: {e}") from e

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            symbol = normalize_symbol(symbol)
            position = self._trading_client.get_open_position(symbol)
            return self._convert_position(position)
        except Exception as e:
            if "not found" in str(e).lower():
                return None
            raise GatewayError(f"Failed to get position for {symbol}: {e}") from e

    def _convert_position(self, alpaca_pos: Any) -> Position:
        """Convert Alpaca position to LIBRA Position."""
        qty = Decimal(str(alpaca_pos.qty))
        side = PositionSide.LONG if qty > 0 else PositionSide.SHORT

        return Position(
            symbol=alpaca_pos.symbol,
            side=side,
            amount=abs(qty),
            entry_price=Decimal(str(alpaca_pos.avg_entry_price)),
            current_price=Decimal(str(alpaca_pos.current_price)),
            unrealized_pnl=Decimal(str(alpaca_pos.unrealized_pl)),
            realized_pnl=Decimal("0"),  # Not available per-position
            leverage=1,
            timestamp_ns=time.time_ns(),
        )

    async def get_balances(self) -> dict[str, Balance]:
        """Get account balances."""
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            account = self._trading_client.get_account()

            # Alpaca uses USD as base
            return {
                "USD": Balance(
                    currency="USD",
                    total=Decimal(str(account.equity)),
                    available=Decimal(str(account.buying_power)),
                    locked=Decimal(str(account.equity))
                    - Decimal(str(account.buying_power)),
                ),
            }
        except Exception as e:
            raise GatewayError(f"Failed to get balances: {e}") from e

    async def get_balance(self, currency: str) -> Balance | None:
        """Get balance for a specific currency."""
        balances = await self.get_balances()
        return balances.get(currency.upper())

    async def get_account_info(self) -> dict[str, Any]:
        """
        Get detailed account information.

        Returns dict with:
        - account_number: Account ID
        - equity: Total equity
        - cash: Cash balance
        - buying_power: Available buying power
        - daytrading_buying_power: Day trading buying power (if PDT)
        - portfolio_value: Portfolio value
        - status: Account status
        """
        await self._rate_limit()

        if not self._trading_client:
            raise GatewayError("Gateway not connected")

        try:
            account = self._trading_client.get_account()
            return {
                "account_number": account.account_number,
                "equity": str(account.equity),
                "cash": str(account.cash),
                "buying_power": str(account.buying_power),
                "daytrading_buying_power": str(account.daytrading_buying_power),
                "portfolio_value": str(account.portfolio_value),
                "status": account.status,
                "trading_blocked": account.trading_blocked,
                "pattern_day_trader": account.pattern_day_trader,
            }
        except Exception as e:
            raise GatewayError(f"Failed to get account info: {e}") from e

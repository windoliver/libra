"""
Interactive Brokers Gateway Implementation.

Provides institutional-grade access to stocks, options, and futures
via TWS API using ib_async library.

Issue #64: Interactive Brokers Gateway - Full Options Lifecycle
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.gateways.protocol import (
    Balance,
    BaseGateway,
    GatewayCapabilities,
    GatewayError,
    Order,
    OrderBook,
    OrderResult,
    OrderStatus,
    Position,
    Tick,
)

from libra.gateways.ibkr.config import IBKRConfig
from libra.gateways.ibkr.contracts import build_option, build_stock, parse_symbol
from libra.gateways.ibkr.converters import (
    ib_account_value_to_balance,
    ib_position_to_position,
    ticker_to_greeks,
    ticker_to_tick,
    trade_to_order_result,
)
from libra.gateways.ibkr.orders import (
    apply_extended_hours,
    apply_time_in_force,
    build_ib_order,
    map_ib_status,
)


if TYPE_CHECKING:
    from libra.core.options import Greeks, OptionContract
    from libra.core.sessions import MarketSessionManager


logger = logging.getLogger(__name__)


# =============================================================================
# Gateway Capabilities
# =============================================================================

IBKR_CAPABILITIES = GatewayCapabilities(
    # Order types - IB supports everything
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
    tif_gtd=True,
    tif_day=True,
    # Market data
    streaming_ticks=True,
    streaming_orderbook=True,
    streaming_trades=True,
    historical_bars=True,
    historical_trades=True,
    # Trading features
    margin_trading=True,
    futures_trading=True,
    options_trading=True,
    short_selling=True,
    # Position management
    position_tracking=True,
    hedge_mode=False,
    # Rate limits - IB is generous
    max_orders_per_second=50,
    max_requests_per_minute=600,
)


# =============================================================================
# Exceptions
# =============================================================================


class IBKRNotInstalledError(GatewayError):
    """ib_async library is not installed."""


class IBKRConnectionError(GatewayError):
    """Failed to connect to TWS/IB Gateway."""


class IBKRNotConnectedError(GatewayError):
    """Gateway is not connected."""


# =============================================================================
# Gateway Implementation
# =============================================================================


class IBKRGateway(BaseGateway):
    """
    Interactive Brokers gateway for stocks and options.

    Provides access to global markets with full options lifecycle support:
    - Stock and options trading
    - Real-time Greeks via market data
    - Option chains without throttling
    - Exercise/assignment handling
    - Multi-leg combo orders

    Requires TWS or IB Gateway running locally.

    Example:
        config = IBKRConfig(port=7497)  # Paper trading
        async with IBKRGateway("ibkr", config) as gw:
            # Get positions
            positions = await gw.get_positions()

            # Place stock order
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("100"),
            )
            result = await gw.submit_order(order)

            # Get option Greeks
            from libra.core.options import OptionContract, OptionType
            contract = OptionContract(...)
            greeks = await gw.get_greeks(contract)
    """

    def __init__(
        self,
        name: str = "ibkr",
        config: IBKRConfig | None = None,
        session_manager: MarketSessionManager | None = None,
    ) -> None:
        """
        Initialize IBKR gateway.

        Args:
            name: Gateway identifier
            config: IBKR configuration (uses defaults if None)
            session_manager: Market session manager for trading hours
        """
        super().__init__(name, config=None, session_manager=session_manager)
        self._config = config or IBKRConfig()
        self._ib: Any | None = None
        self._tick_queue: asyncio.Queue[Tick] = asyncio.Queue()
        self._subscriptions: dict[str, Any] = {}  # symbol -> IB contract
        self._reconnect_task: asyncio.Task | None = None

    @property
    def capabilities(self) -> GatewayCapabilities:
        """Gateway capabilities."""
        return IBKR_CAPABILITIES

    @property
    def config(self) -> IBKRConfig:
        """Gateway configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to TWS or IB Gateway.

        Raises:
            IBKRNotInstalledError: If ib_async is not installed
            IBKRConnectionError: If connection fails
        """
        if self._connected:
            logger.warning("Already connected to IBKR")
            return

        try:
            from ib_async import IB
        except ImportError as e:
            raise IBKRNotInstalledError(
                "ib_async is not installed. Install with: pip install ib_async"
            ) from e

        self._ib = IB()

        # Register event handlers
        self._ib.errorEvent += self._on_error
        self._ib.disconnectedEvent += self._on_disconnect

        try:
            logger.info(
                f"Connecting to IBKR at {self._config.host}:{self._config.port} "
                f"(clientId={self._config.client_id})"
            )
            await self._ib.connectAsync(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
                readonly=self._config.readonly,
                timeout=self._config.timeout,
            )
            self._connected = True
            logger.info(f"Connected to IBKR ({'paper' if self._config.is_paper else 'live'})")

            # Request account updates if account specified
            if self._config.account:
                self._ib.reqAccountUpdates(True, self._config.account)

        except Exception as e:
            self._ib = None
            raise IBKRConnectionError(
                f"Failed to connect to IBKR at {self._config.host}:{self._config.port}. "
                f"Ensure TWS/IB Gateway is running with API enabled. Error: {e}"
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from TWS/IB Gateway."""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._connected = False
                self._ib = None
                self._subscriptions.clear()
                logger.info("Disconnected from IBKR")

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Any) -> None:
        """Handle IB error events."""
        # Some error codes are just warnings
        warning_codes = {2104, 2106, 2158}  # Market data farm connections
        if errorCode in warning_codes:
            logger.debug(f"IBKR info [{errorCode}]: {errorString}")
        else:
            logger.error(f"IBKR error [{errorCode}] reqId={reqId}: {errorString}")

    def _on_disconnect(self) -> None:
        """Handle disconnect event."""
        logger.warning("Disconnected from IBKR")
        self._connected = False

        if self._config.auto_reconnect:
            logger.info("Scheduling reconnection...")
            self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        delay = self._config.reconnect_delay
        for attempt in range(self._config.max_reconnect_attempts):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{self._config.max_reconnect_attempts}")
                await self.connect()
                logger.info("Reconnected successfully")
                return
            except Exception as e:
                logger.warning(f"Reconnection failed: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)  # Exponential backoff, max 60s

        logger.error("Max reconnection attempts reached")

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected or self._ib is None:
            raise IBKRNotConnectedError("Not connected to IBKR")

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data for symbols."""
        self._ensure_connected()

        for symbol in symbols:
            if symbol in self._subscriptions:
                continue

            asset_class, underlying = parse_symbol(symbol)

            if asset_class == "stock":
                contract = build_stock(underlying)
            else:
                # For options, we need full OCC symbol parsing
                # For now, treat as stock
                contract = build_stock(underlying)

            await self._ib.qualifyContractsAsync(contract)
            ticker = self._ib.reqMktData(contract, "", False, False)

            # Set up callback
            ticker.updateEvent += lambda t, sym=symbol: self._on_tick_update(t, sym)

            self._subscriptions[symbol] = contract
            self._subscribed_symbols.add(symbol)
            logger.debug(f"Subscribed to {symbol}")

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from market data."""
        self._ensure_connected()

        for symbol in symbols:
            if symbol not in self._subscriptions:
                continue

            contract = self._subscriptions.pop(symbol)
            self._ib.cancelMktData(contract)
            self._subscribed_symbols.discard(symbol)
            logger.debug(f"Unsubscribed from {symbol}")

    def _on_tick_update(self, ticker: Any, symbol: str) -> None:
        """Handle tick update from IB."""
        try:
            tick = ticker_to_tick(ticker, symbol)
            self._tick_queue.put_nowait(tick)
        except Exception as e:
            logger.warning(f"Error processing tick for {symbol}: {e}")

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """Stream tick data."""
        while self._connected:
            try:
                tick = await asyncio.wait_for(self._tick_queue.get(), timeout=1.0)
                yield tick
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    async def get_ticker(self, symbol: str) -> Tick:
        """Get current ticker for symbol."""
        self._ensure_connected()

        asset_class, underlying = parse_symbol(symbol)
        contract = build_stock(underlying)

        await self._ib.qualifyContractsAsync(contract)
        ticker = self._ib.reqMktData(contract, "", True, False)  # snapshot=True

        # Wait for data
        await self._ib.sleep(2)
        self._ib.cancelMktData(contract)

        return ticker_to_tick(ticker, symbol)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book for symbol."""
        self._ensure_connected()

        asset_class, underlying = parse_symbol(symbol)
        contract = build_stock(underlying)

        await self._ib.qualifyContractsAsync(contract)
        orderbook = await self._ib.reqMktDepthAsync(contract, numRows=depth)

        # Convert to libra format
        bids = [(Decimal(str(row.price)), Decimal(str(row.size))) for row in orderbook if row.side == 1]
        asks = [(Decimal(str(row.price)), Decimal(str(row.size))) for row in orderbook if row.side == 0]

        return OrderBook(
            symbol=symbol,
            bids=sorted(bids, reverse=True),
            asks=sorted(asks),
            timestamp_ns=time.time_ns(),
        )

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """Submit order to IBKR."""
        self._ensure_connected()

        # Validate session
        is_valid, error = self.validate_session(order)
        if not is_valid:
            return OrderResult(
                order_id="",
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                side=order.side,
                order_type=order.order_type,
                amount=order.amount,
                filled_amount=Decimal("0"),
                remaining_amount=order.amount,
                average_price=None,
                fee=Decimal("0"),
                fee_currency="USD",
                timestamp_ns=time.time_ns(),
            )

        # Build contract
        asset_class, underlying = parse_symbol(order.symbol)
        if asset_class == "stock":
            contract = build_stock(underlying)
        else:
            contract = build_stock(underlying)  # TODO: Option support

        await self._ib.qualifyContractsAsync(contract)

        # Build order
        ib_order = build_ib_order(order)
        apply_time_in_force(ib_order, order.time_in_force)
        apply_extended_hours(ib_order, order.extended_hours)

        # Submit
        trade = self._ib.placeOrder(contract, ib_order)

        # Wait for acknowledgment
        await self._ib.sleep(0.5)

        return trade_to_order_result(trade)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        self._ensure_connected()

        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                await self._ib.sleep(0.5)
                return True

        return False

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all open orders."""
        self._ensure_connected()

        cancelled = 0
        for trade in self._ib.openTrades():
            if symbol is None or trade.contract.symbol == symbol:
                self._ib.cancelOrder(trade.order)
                cancelled += 1

        if cancelled:
            await self._ib.sleep(0.5)

        return cancelled

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order status."""
        self._ensure_connected()

        # Check open trades
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                return trade_to_order_result(trade)

        # Check completed trades
        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                return trade_to_order_result(trade)

        raise ValueError(f"Order {order_id} not found")

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get all open orders."""
        self._ensure_connected()

        results = []
        for trade in self._ib.openTrades():
            if symbol is None or trade.contract.symbol == symbol:
                results.append(trade_to_order_result(trade))

        return results

    async def get_order_history(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list[OrderResult]:
        """Get order history."""
        self._ensure_connected()

        results = []
        for trade in self._ib.trades():
            if symbol is None or trade.contract.symbol == symbol:
                results.append(trade_to_order_result(trade))

        if limit:
            results = results[:limit]

        return results

    async def get_trades(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list:
        """Get trade/fill history."""
        self._ensure_connected()

        fills = []
        for trade in self._ib.trades():
            if symbol is None or trade.contract.symbol == symbol:
                for fill in trade.fills:
                    fills.append({
                        "order_id": str(trade.order.orderId),
                        "symbol": trade.contract.symbol,
                        "side": trade.order.action,
                        "quantity": fill.execution.shares,
                        "price": fill.execution.price,
                        "commission": fill.commissionReport.commission if fill.commissionReport else 0,
                        "timestamp": fill.execution.time,
                    })

        if limit:
            fills = fills[:limit]

        return fills

    # -------------------------------------------------------------------------
    # Positions and Balances
    # -------------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Get all positions."""
        self._ensure_connected()

        positions = self._ib.positions()
        return [ib_position_to_position(p) for p in positions]

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for symbol."""
        self._ensure_connected()

        for pos in self._ib.positions():
            if pos.contract.symbol == symbol or pos.contract.localSymbol == symbol:
                return ib_position_to_position(pos)

        return None

    async def get_balances(self) -> dict[str, Balance]:
        """Get account balances."""
        self._ensure_connected()

        account_values = self._ib.accountValues()
        currencies = {av.currency for av in account_values if av.currency}

        balances = {}
        for currency in currencies:
            if currency in ("BASE", ""):  # Skip non-currency entries
                continue
            balances[currency] = ib_account_value_to_balance(account_values, currency)

        return balances

    async def get_balance(self, currency: str) -> Balance | None:
        """Get balance for specific currency."""
        balances = await self.get_balances()
        return balances.get(currency)

    # -------------------------------------------------------------------------
    # Options-Specific Methods
    # -------------------------------------------------------------------------

    async def get_greeks(self, contract: OptionContract) -> Greeks:
        """
        Get real-time Greeks for an option.

        Args:
            contract: Core OptionContract

        Returns:
            Greeks object with delta, gamma, theta, vega, rho, iv

        Raises:
            ValueError: If Greeks are not available
        """
        self._ensure_connected()

        ib_option = build_option(contract)
        await self._ib.qualifyContractsAsync(ib_option)

        # Request market data with Greeks (tick type 106)
        ticker = self._ib.reqMktData(ib_option, genericTickList="106", snapshot=False)

        # Wait for data
        await self._ib.sleep(2)

        try:
            return ticker_to_greeks(ticker)
        finally:
            self._ib.cancelMktData(ib_option)

    async def get_option_chain(self, underlying: str) -> dict:
        """
        Get option chain for underlying.

        Args:
            underlying: Underlying ticker (e.g., "AAPL")

        Returns:
            Dict with expirations and strikes
        """
        self._ensure_connected()

        stock = build_stock(underlying)
        await self._ib.qualifyContractsAsync(stock)

        chains = await self._ib.reqSecDefOptParamsAsync(
            underlyingSymbol=stock.symbol,
            futFopExchange="",
            underlyingSecType=stock.secType,
            underlyingConId=stock.conId,
        )

        from libra.gateways.ibkr.converters import option_chain_params_to_dict

        return option_chain_params_to_dict(chains)

    async def exercise_option(
        self,
        contract: OptionContract,
        quantity: int,
        override: bool = False,
    ) -> None:
        """
        Exercise an option position.

        Args:
            contract: Option contract to exercise
            quantity: Number of contracts to exercise
            override: Override system's price check

        Note:
            This is an irreversible action. Use with caution.
        """
        self._ensure_connected()

        ib_option = build_option(contract)
        await self._ib.qualifyContractsAsync(ib_option)

        self._ib.exerciseOptions(
            contract=ib_option,
            exerciseAction=1,  # 1=exercise
            exerciseQuantity=quantity,
            account=self._config.account or "",
            override=1 if override else 0,
        )

        logger.info(f"Exercised {quantity} contracts of {contract.symbol}")

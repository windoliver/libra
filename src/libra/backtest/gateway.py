"""
BacktestGateway: Gateway adapter for backtest mode.

Wraps BacktestExecutionClient to provide the Gateway protocol interface,
enabling unified strategy code to run in both backtest and live environments.

This is the key component for Issue #37 - Event-Driven Backtest Engine
with Unified Strategy Code.

Usage:
    # Create backtest gateway
    exec_client = BacktestExecutionClient(clock, initial_balance)
    gateway = BacktestGateway(exec_client)

    # Use with unified strategy (same as live)
    strategy = MyStrategy(gateway)
    await strategy.on_bar(bar)  # Strategy calls gateway.submit_order() internally
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.gateways.protocol import (
    Balance,
    BaseGateway,
    Order,
    OrderBook,
    OrderResult,
    Position,
    Tick,
)


if TYPE_CHECKING:
    from libra.clients.backtest_execution_client import BacktestExecutionClient
    from libra.core.message_bus import MessageBus


logger = logging.getLogger(__name__)


class BacktestGateway(BaseGateway):
    """
    Gateway adapter for backtesting.

    Wraps BacktestExecutionClient to provide the standard Gateway interface.
    This allows the same strategy code to work in backtest and live modes.

    Key features:
    - Implements full Gateway protocol
    - Delegates order execution to BacktestExecutionClient
    - Supports position and balance queries
    - Provides tick streaming (from bar data during backtest)

    Example:
        # In BacktestEngine
        exec_client = BacktestExecutionClient(clock, {"USDT": Decimal("100000")})
        gateway = BacktestGateway(exec_client)

        # Strategy uses gateway (same as live)
        class MyStrategy(BaseStrategy):
            async def on_bar(self, bar: Bar) -> None:
                if should_buy:
                    await self.buy_market(bar.symbol, amount)
    """

    def __init__(
        self,
        execution_client: BacktestExecutionClient,
        bus: MessageBus | None = None,
    ) -> None:
        """
        Initialize backtest gateway.

        Args:
            execution_client: BacktestExecutionClient for order simulation
            bus: Optional message bus for event publishing
        """
        super().__init__(name="backtest")
        self._exec_client = execution_client
        self._bus = bus
        self._current_bar: Any = None  # Current bar for tick simulation

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def execution_client(self) -> BacktestExecutionClient:
        """Access underlying execution client."""
        return self._exec_client

    @property
    def is_connected(self) -> bool:
        """Check if gateway is connected."""
        return self._exec_client._connected

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Connect the gateway (connects execution client)."""
        await self._exec_client.connect()
        self._connected = True
        logger.info("BacktestGateway connected")

    async def disconnect(self) -> None:
        """Disconnect the gateway."""
        await self._exec_client.disconnect()
        self._connected = False
        logger.info("BacktestGateway disconnected")

    # =========================================================================
    # Market Data
    # =========================================================================

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols (no-op in backtest, data is pre-loaded)."""
        self._subscribed_symbols.update(symbols)
        logger.debug("Backtest subscribed to: %s", symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        self._subscribed_symbols -= set(symbols)

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """
        Stream ticks (not used in backtest - bars are replayed instead).

        In backtest mode, the engine replays bars directly.
        This is provided for protocol compliance.
        """
        # In backtest, we don't stream ticks - bars are replayed by engine
        # Yield nothing to satisfy the async iterator protocol
        return
        yield  # type: ignore[misc]  # Make this a generator

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """
        Get order book (simulated from current bar).

        In backtest, we simulate a simple order book from the current price.
        """
        # Get latest tick/price
        tick = self._exec_client._latest_ticks.get(symbol)
        if tick is None:
            # Return empty order book
            return OrderBook(
                symbol=symbol,
                bids=[],
                asks=[],
                timestamp_ns=0,
            )

        price = tick.last
        # Simulate simple order book around current price
        spread = price * Decimal("0.0001")  # 1 bps spread

        # OrderBook uses tuples: (price, size)
        bids: list[tuple[Decimal, Decimal]] = [
            (price - spread * (i + 1), Decimal("10"))
            for i in range(min(depth, 5))
        ]
        asks: list[tuple[Decimal, Decimal]] = [
            (price + spread * (i + 1), Decimal("10"))
            for i in range(min(depth, 5))
        ]

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp_ns=tick.timestamp_ns,
        )

    async def get_ticker(self, symbol: str) -> Tick:
        """Get current ticker from execution client's latest ticks."""
        tick = self._exec_client._latest_ticks.get(symbol)
        if tick is None:
            # Return a zero tick if no data yet
            return Tick(
                symbol=symbol,
                bid=Decimal("0"),
                ask=Decimal("0"),
                last=Decimal("0"),
                timestamp_ns=0,
            )
        return tick

    def set_current_bar(self, bar: Any) -> None:
        """
        Set current bar (called by engine during simulation).

        This updates the latest tick for the symbol.
        """
        self._current_bar = bar

    # =========================================================================
    # Trading
    # =========================================================================

    async def submit_order(self, order: Order) -> OrderResult:
        """Submit order through execution client."""
        return await self._exec_client.submit_order(order)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order through execution client."""
        return await self._exec_client.cancel_order(order_id, symbol)

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders through execution client."""
        return await self._exec_client.cancel_all_orders(symbol)

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order status from execution client."""
        return await self._exec_client.get_order(order_id, symbol)

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get open orders from execution client."""
        return await self._exec_client.get_open_orders(symbol)

    # =========================================================================
    # Account
    # =========================================================================

    async def get_positions(self) -> list[Position]:
        """Get all positions from execution client."""
        return list(self._exec_client._positions.values())

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for symbol from execution client."""
        return self._exec_client._positions.get(symbol)

    async def get_balances(self) -> dict[str, Balance]:
        """Get all balances from execution client."""
        return dict(self._exec_client._balances)

    async def get_balance(self, currency: str) -> Balance | None:
        """Get balance for currency from execution client."""
        return self._exec_client._balances.get(currency)

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> BacktestGateway:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        _exc_type: Any,
        _exc_val: Any,
        _exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def __repr__(self) -> str:
        """String representation."""
        return f"BacktestGateway(connected={self.is_connected})"

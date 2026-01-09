"""
Demo Execution Client for TUI.

Bridges DemoTrader with ExecutionClient protocol for Issue #36.

This adapter allows the ExecutionEngine and execution algorithms
to work seamlessly with the demo trading simulation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.gateways.protocol import (
    Balance,
    Order,
    OrderResult,
    OrderStatus,
    Position,
    PositionSide,
)
from libra.tui.demo_trader import DemoTrader


# =============================================================================
# Demo Execution Client
# =============================================================================


class DemoExecutionClient:
    """
    ExecutionClient implementation wrapping DemoTrader.

    Provides ExecutionClient protocol compliance for the demo
    trading engine, enabling ExecutionEngine and algorithms
    to execute orders through the simulation.

    Usage:
        demo_trader = DemoTrader()
        client = DemoExecutionClient(demo_trader)

        # Use with ExecutionEngine
        engine = create_execution_engine(
            execution_client=client,
            enable_risk_checks=False,
        )

        # Submit algo order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            exec_algorithm="twap",
            exec_algorithm_params={"horizon_secs": 60},
        )
        result = await engine.submit_order(order)
    """

    def __init__(
        self,
        demo_trader: DemoTrader,
        name: str = "demo-exec",
    ) -> None:
        """
        Initialize the demo execution client.

        Args:
            demo_trader: The DemoTrader instance to wrap
            name: Client identifier
        """
        self._demo_trader = demo_trader
        self._name = name
        self._connected = False
        self._open_orders: dict[str, OrderResult] = {}
        self._order_history: list[OrderResult] = []
        self._order_update_queue: asyncio.Queue[OrderResult] = asyncio.Queue()

    @property
    def name(self) -> str:
        """Client identifier."""
        return self._name

    @property
    def is_connected(self) -> bool:
        """Connection status."""
        return self._connected

    @property
    def demo_trader(self) -> DemoTrader:
        """Get the underlying DemoTrader instance."""
        return self._demo_trader

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Connect to the demo trading simulation."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the demo trading simulation."""
        self._connected = False

    async def __aenter__(self) -> DemoExecutionClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # =========================================================================
    # Order Management
    # =========================================================================

    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order to the demo trading simulation.

        Args:
            order: Order to submit

        Returns:
            OrderResult with execution details
        """
        if not self._connected:
            raise RuntimeError("Client not connected")

        # Generate order ID if not present
        order_id = order.client_order_id or f"demo-{uuid.uuid4().hex[:8]}"
        timestamp_ns = time.time_ns()

        # Map side
        side_str = order.side.value if hasattr(order.side, "value") else str(order.side)

        # Execute through DemoTrader
        success, message = self._demo_trader.execute_order(
            symbol=order.symbol,
            side=side_str,
            quantity=order.amount,
            price=order.price,
        )

        # Get fill price
        fill_price = order.price or self._demo_trader.get_price(order.symbol)

        # Create result
        if success:
            result = OrderResult(
                order_id=order_id,
                symbol=order.symbol,
                status=OrderStatus.FILLED,
                side=order.side,
                order_type=order.order_type,
                amount=order.amount,
                filled_amount=order.amount,
                remaining_amount=Decimal("0"),
                average_price=fill_price,
                fee=fill_price * order.amount * Decimal("0.001"),  # 0.1% fee
                fee_currency="USDT",
                timestamp_ns=timestamp_ns,
                client_order_id=order_id,
            )
        else:
            result = OrderResult(
                order_id=order_id,
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                side=order.side,
                order_type=order.order_type,
                amount=order.amount,
                filled_amount=Decimal("0"),
                remaining_amount=order.amount,
                average_price=None,
                fee=Decimal("0"),
                fee_currency="USDT",
                timestamp_ns=timestamp_ns,
                client_order_id=order_id,
            )

        # Track order
        self._order_history.append(result)

        # Queue update for streaming
        await self._order_update_queue.put(result)

        return result

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order (demo orders fill instantly, so nothing to cancel)."""
        return False

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders (demo orders fill instantly)."""
        return 0

    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        price: Any | None = None,
        amount: Any | None = None,
    ) -> OrderResult:
        """Modify an order (not supported in demo mode)."""
        raise NotImplementedError("Order modification not supported in demo mode")

    # =========================================================================
    # Order Queries
    # =========================================================================

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order by ID."""
        for order in self._order_history:
            if order.order_id == order_id:
                return order
        raise ValueError(f"Order {order_id} not found")

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get open orders (demo orders fill instantly, so always empty)."""
        return []

    async def get_order_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[OrderResult]:
        """Get order history."""
        history = self._order_history
        if symbol:
            history = [o for o in history if o.symbol == symbol]
        return history[-limit:]

    # =========================================================================
    # Order Event Stream
    # =========================================================================

    async def stream_order_updates(self) -> AsyncIterator[OrderResult]:
        """Stream order updates."""
        while True:
            try:
                result = await asyncio.wait_for(
                    self._order_update_queue.get(),
                    timeout=1.0,
                )
                yield result
            except asyncio.TimeoutError:
                continue

    # =========================================================================
    # Account State
    # =========================================================================

    def _convert_side(self, side_str: str) -> PositionSide:
        """Convert DemoTrader side string to PositionSide enum."""
        side_map = {
            "LONG": PositionSide.LONG,
            "SHORT": PositionSide.SHORT,
            "FLAT": PositionSide.FLAT,
        }
        return side_map.get(side_str.upper(), PositionSide.FLAT)

    async def get_positions(self) -> list[Position]:
        """Get all positions from DemoTrader."""
        positions = []
        for symbol, pos in self._demo_trader.positions.items():
            if pos.side != "FLAT" and pos.quantity > 0:
                positions.append(
                    Position(
                        symbol=symbol,
                        side=self._convert_side(pos.side),
                        amount=pos.quantity,
                        entry_price=pos.entry_price,
                        current_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl=Decimal("0"),
                    )
                )
        return positions

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for symbol."""
        pos = self._demo_trader.positions.get(symbol)
        if not pos or pos.side == "FLAT":
            return None
        return Position(
            symbol=symbol,
            side=self._convert_side(pos.side),
            amount=pos.quantity,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl,
            realized_pnl=Decimal("0"),
        )

    async def get_balances(self) -> dict[str, Balance]:
        """Get account balances."""
        return {
            "USDT": Balance(
                currency="USDT",
                total=self._demo_trader.balance,
                available=self._demo_trader.balance,
                locked=Decimal("0"),
            ),
        }

    async def get_balance(self, currency: str) -> Balance | None:
        """Get balance for currency."""
        if currency == "USDT":
            return Balance(
                currency="USDT",
                total=self._demo_trader.balance,
                available=self._demo_trader.balance,
                locked=Decimal("0"),
            )
        return None

    # =========================================================================
    # Reconciliation
    # =========================================================================

    async def reconcile_orders(self) -> int:
        """Reconcile orders (demo has instant fills, always reconciled)."""
        return 0

    async def reconcile_positions(self) -> int:
        """Reconcile positions (demo tracks positions internally)."""
        return 0

"""
Cache: Shared state cache for instruments, orders, positions, and market data.

Provides centralized access to:
- Order state (open, filled, canceled)
- Position state
- Market data (quotes, bars)
- Instrument definitions

Thread-safe with asyncio locks for concurrent access.

Design references:
- NautilusTrader cache pattern
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.gateways.protocol import (
    Balance,
    Order,
    OrderResult,
    OrderStatus,
    Position,
    Tick,
)
from libra.strategies.protocol import Bar


if TYPE_CHECKING:
    pass


class Cache:
    """
    Shared state cache for the trading system.

    Stores:
    - Orders: Active and historical orders
    - Positions: Current positions per symbol
    - Quotes: Latest tick data per symbol
    - Bars: Recent bar data per symbol/timeframe
    - Balances: Account balances per currency

    Thread-safe for concurrent access.

    Example:
        cache = Cache()

        # Store and retrieve orders
        cache.add_order(order)
        order = cache.order("client_order_123")

        # Store and retrieve positions
        cache.update_position(position)
        pos = cache.position("BTC/USDT")

        # Store and retrieve quotes
        cache.update_quote(tick)
        quote = cache.quote("BTC/USDT")
    """

    def __init__(self, max_bars: int = 1000, max_orders: int = 10000) -> None:
        """
        Initialize cache.

        Args:
            max_bars: Maximum bars to retain per symbol/timeframe
            max_orders: Maximum orders to retain (FIFO eviction)
        """
        self._max_bars = max_bars
        self._max_orders = max_orders

        # Orders: client_order_id -> OrderResult
        self._orders: dict[str, OrderResult] = {}
        self._order_ids: deque[str] = deque()  # O(1) FIFO eviction (Issue #67)

        # Positions: symbol -> Position
        self._positions: dict[str, Position] = {}

        # Quotes: symbol -> Tick
        self._quotes: dict[str, Tick] = {}

        # Bars: (symbol, timeframe) -> list[Bar]
        self._bars: dict[tuple[str, str], list[Bar]] = defaultdict(list)

        # Balances: currency -> Balance
        self._balances: dict[str, Balance] = {}

        # Locks for thread safety
        self._order_lock = asyncio.Lock()
        self._position_lock = asyncio.Lock()
        self._quote_lock = asyncio.Lock()
        self._bar_lock = asyncio.Lock()
        self._balance_lock = asyncio.Lock()

    # =========================================================================
    # Order Methods
    # =========================================================================

    async def add_order(self, result: OrderResult) -> None:
        """
        Add or update an order.

        Args:
            result: Order result to cache
        """
        async with self._order_lock:
            order_id = result.client_order_id or result.order_id

            # Add to order list for eviction tracking
            if order_id not in self._orders:
                self._order_ids.append(order_id)

                # Evict old orders if over limit - O(1) with deque.popleft()
                while len(self._order_ids) > self._max_orders:
                    old_id = self._order_ids.popleft()
                    self._orders.pop(old_id, None)

            self._orders[order_id] = result

    def order(self, client_order_id: str) -> OrderResult | None:
        """
        Get order by client order ID.

        Args:
            client_order_id: Client-assigned order ID

        Returns:
            OrderResult if found, None otherwise
        """
        return self._orders.get(client_order_id)

    def order_by_exchange_id(self, order_id: str) -> OrderResult | None:
        """
        Get order by exchange order ID.

        Args:
            order_id: Exchange-assigned order ID

        Returns:
            OrderResult if found, None otherwise
        """
        for result in self._orders.values():
            if result.order_id == order_id:
                return result
        return None

    def orders(
        self,
        symbol: str | None = None,
        status: OrderStatus | None = None,
    ) -> list[OrderResult]:
        """
        Get orders with optional filtering.

        Args:
            symbol: Filter by symbol
            status: Filter by status

        Returns:
            List of matching orders
        """
        results = list(self._orders.values())

        if symbol:
            results = [r for r in results if r.symbol == symbol]

        if status:
            results = [r for r in results if r.status == status]

        return results

    def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        return [
            r
            for r in self._orders.values()
            if r.is_open and (symbol is None or r.symbol == symbol)
        ]

    # =========================================================================
    # Position Methods
    # =========================================================================

    async def update_position(self, position: Position) -> None:
        """
        Update position for a symbol.

        Args:
            position: Position to cache
        """
        async with self._position_lock:
            if position.amount == Decimal("0"):
                self._positions.pop(position.symbol, None)
            else:
                self._positions[position.symbol] = position

    def position(self, symbol: str) -> Position | None:
        """
        Get position for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position if exists, None otherwise
        """
        return self._positions.get(symbol)

    def positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of positions
        """
        return list(self._positions.values())

    def has_position(self, symbol: str) -> bool:
        """
        Check if position exists for symbol.

        Args:
            symbol: Trading pair

        Returns:
            True if position exists with non-zero amount
        """
        pos = self._positions.get(symbol)
        return pos is not None and pos.amount > Decimal("0")

    # =========================================================================
    # Quote Methods
    # =========================================================================

    async def update_quote(self, tick: Tick) -> None:
        """
        Update quote (tick) for a symbol.

        Args:
            tick: Tick data to cache
        """
        async with self._quote_lock:
            self._quotes[tick.symbol] = tick

    def quote(self, symbol: str) -> Tick | None:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Tick if available, None otherwise
        """
        return self._quotes.get(symbol)

    def quotes(self) -> dict[str, Tick]:
        """
        Get all quotes.

        Returns:
            Dict mapping symbol to Tick
        """
        return dict(self._quotes)

    def price(self, symbol: str) -> Decimal | None:
        """
        Get last price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Last price if available, None otherwise
        """
        tick = self._quotes.get(symbol)
        return tick.last if tick else None

    def mid_price(self, symbol: str) -> Decimal | None:
        """
        Get mid price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Mid price if available, None otherwise
        """
        tick = self._quotes.get(symbol)
        return tick.mid if tick else None

    # =========================================================================
    # Bar Methods
    # =========================================================================

    async def add_bar(self, bar: Bar) -> None:
        """
        Add a bar to the cache.

        Args:
            bar: Bar data to cache
        """
        async with self._bar_lock:
            key = (bar.symbol, bar.timeframe)
            bars = self._bars[key]
            bars.append(bar)

            # Trim to max size
            if len(bars) > self._max_bars:
                self._bars[key] = bars[-self._max_bars :]

    def bar(self, symbol: str, timeframe: str) -> Bar | None:
        """
        Get latest bar for a symbol/timeframe.

        Args:
            symbol: Trading pair
            timeframe: Bar timeframe (e.g., "1h")

        Returns:
            Latest bar if available, None otherwise
        """
        bars = self._bars.get((symbol, timeframe))
        return bars[-1] if bars else None

    def bars(
        self,
        symbol: str,
        timeframe: str,
        count: int | None = None,
    ) -> list[Bar]:
        """
        Get bars for a symbol/timeframe.

        Args:
            symbol: Trading pair
            timeframe: Bar timeframe
            count: Number of bars (default: all)

        Returns:
            List of bars (oldest first)
        """
        all_bars = self._bars.get((symbol, timeframe), [])
        if count:
            return all_bars[-count:]
        return list(all_bars)

    # =========================================================================
    # Balance Methods
    # =========================================================================

    async def update_balance(self, balance: Balance) -> None:
        """
        Update balance for a currency.

        Args:
            balance: Balance to cache
        """
        async with self._balance_lock:
            self._balances[balance.currency] = balance

    async def update_balances(self, balances: dict[str, Balance]) -> None:
        """
        Update multiple balances.

        Args:
            balances: Dict of currency -> Balance
        """
        async with self._balance_lock:
            self._balances.update(balances)

    def balance(self, currency: str) -> Balance | None:
        """
        Get balance for a currency.

        Args:
            currency: Currency code (e.g., "USDT")

        Returns:
            Balance if available, None otherwise
        """
        return self._balances.get(currency)

    def balances(self) -> dict[str, Balance]:
        """
        Get all balances.

        Returns:
            Dict mapping currency to Balance
        """
        return dict(self._balances)

    def available_balance(self, currency: str) -> Decimal:
        """
        Get available balance for a currency.

        Args:
            currency: Currency code

        Returns:
            Available balance or 0
        """
        balance = self._balances.get(currency)
        return balance.available if balance else Decimal("0")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def clear(self) -> None:
        """Clear all cached data."""
        async with self._order_lock:
            self._orders.clear()
            self._order_ids.clear()

        async with self._position_lock:
            self._positions.clear()

        async with self._quote_lock:
            self._quotes.clear()

        async with self._bar_lock:
            self._bars.clear()

        async with self._balance_lock:
            self._balances.clear()

    def stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with counts of cached items
        """
        return {
            "orders": len(self._orders),
            "positions": len(self._positions),
            "quotes": len(self._quotes),
            "bar_series": len(self._bars),
            "balances": len(self._balances),
        }

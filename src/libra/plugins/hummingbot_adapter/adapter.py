"""
Hummingbot Adapter Plugin for LIBRA (Issue #12).

Main adapter class that implements the StrategyPlugin interface
and provides market making capabilities.

Supported strategies:
- Avellaneda-Stoikov optimal market making
- Pure market making
- Cross-exchange market making (XEMM)

Features:
- DEX gateway support (Uniswap V2/V3)
- Comprehensive performance tracking
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import polars as pl

from libra.plugins.base import PluginMetadata, StrategyPlugin
from libra.plugins.hummingbot_adapter.config import (
    HummingbotAdapterConfig,
    StrategyType,
)
from libra.plugins.hummingbot_adapter.performance import (
    PerformanceStats,
    PerformanceTracker,
)
from libra.plugins.hummingbot_adapter.strategies.avellaneda import (
    AvellanedaStoikovStrategy,
    Quote,
)
from libra.plugins.hummingbot_adapter.strategies.pure_mm import (
    PureMarketMakingStrategy,
    TwoSidedOrder,
)
from libra.plugins.hummingbot_adapter.strategies.xemm import (
    CrossExchangeMarketMakingStrategy,
    CrossExchangeQuote,
    XEMMOrder,
)
from libra.strategies.protocol import Signal, SignalType


class HummingbotAdapter(StrategyPlugin):
    """
    Hummingbot-style market making adapter for LIBRA.

    This adapter provides standalone implementations of market making
    strategies inspired by Hummingbot, without requiring Hummingbot installation.

    Usage:
        adapter = HummingbotAdapter()
        await adapter.initialize({
            "strategy_type": "avellaneda_stoikov",
            "symbol": "BTC/USDT",
            "order_amount": "0.01",
            "min_spread": "0.001",
            "max_spread": "0.05",
            "avellaneda": {
                "risk_aversion": 0.5,
                "order_book_depth": 1.5,
            },
        })

        signal = await adapter.on_data(ohlcv_dataframe)
    """

    VERSION = "0.1.0"

    def __init__(self, enable_performance_tracking: bool = True) -> None:
        """
        Initialize the adapter.

        Args:
            enable_performance_tracking: Whether to enable P&L and trade tracking
        """
        self._config: HummingbotAdapterConfig | None = None
        self._initialized = False
        self._symbols: list[str] = []

        # Strategy instances
        self._avellaneda: AvellanedaStoikovStrategy | None = None
        self._pure_mm: PureMarketMakingStrategy | None = None
        self._xemm: CrossExchangeMarketMakingStrategy | None = None

        # State tracking
        self._last_quote: Quote | TwoSidedOrder | None = None
        self._last_signal_time_ns: int = 0
        self._pending_orders: list[dict[str, Any]] = []

        # Performance tracking
        self._performance_tracking_enabled = enable_performance_tracking
        self._performance_tracker: PerformanceTracker | None = None

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata.create(
            name="hummingbot-adapter",
            version=cls.VERSION,
            description="Market making strategies (Avellaneda-Stoikov, Pure MM, XEMM)",
            author="LIBRA Team",
            requires=[],  # No external dependencies
        )

    @property
    def name(self) -> str:
        """Strategy name."""
        if self._config:
            return f"hummingbot_{self._config.strategy_type.value}"
        return "hummingbot_adapter"

    @property
    def symbols(self) -> list[str]:
        """Trading symbols."""
        return self._symbols

    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the adapter with configuration.

        Args:
            config: Configuration dictionary
        """
        # Extract initial_capital before parsing config (not part of strategy config)
        initial_capital = Decimal(str(config.pop("initial_capital", 100000)))

        # Parse configuration
        self._config = HummingbotAdapterConfig.from_dict(config)
        self._config.validate()

        self._symbols = [self._config.symbol]

        # Initialize appropriate strategy
        if self._config.strategy_type == StrategyType.AVELLANEDA_STOIKOV:
            self._avellaneda = AvellanedaStoikovStrategy(
                config=self._config.avellaneda,
                inventory_config=self._config.inventory,
                min_spread=self._config.min_spread,
                max_spread=self._config.max_spread,
            )

        elif self._config.strategy_type == StrategyType.PURE_MARKET_MAKING:
            self._pure_mm = PureMarketMakingStrategy(
                inventory_config=self._config.inventory,
                base_spread=self._config.min_spread,
                order_levels=self._config.order_levels,
                level_spread=self._config.level_spread,
                min_spread=self._config.min_spread,
                max_spread=self._config.max_spread,
            )

        elif self._config.strategy_type == StrategyType.CROSS_EXCHANGE_MM:
            self._xemm = CrossExchangeMarketMakingStrategy(
                config=self._config.xemm,
                min_spread=self._config.min_spread,
                max_spread=self._config.max_spread,
            )

        # Initialize performance tracker
        if self._performance_tracking_enabled:
            self._performance_tracker = PerformanceTracker(
                initial_capital=initial_capital,
            )

        self._initialized = True

    async def on_data(self, data: pl.DataFrame) -> Signal | None:
        """
        Process market data and generate trading signal.

        For market making, this generates quotes (bid/ask) that should
        be converted to orders by the execution layer.

        Args:
            data: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            Signal with quote information in metadata, or None
        """
        if not self._initialized or self._config is None:
            return None

        # Get latest data point
        if data.is_empty():
            return None

        latest = data.tail(1)
        close_price = Decimal(str(latest["close"][0]))
        timestamp_ns = int(time.time_ns())

        # Check order refresh time
        refresh_time_ns = int(self._config.order_refresh_time * 1_000_000_000)
        if timestamp_ns - self._last_signal_time_ns < refresh_time_ns:
            return None

        signal: Signal | None = None

        if self._config.strategy_type == StrategyType.AVELLANEDA_STOIKOV:
            signal = await self._process_avellaneda(close_price, timestamp_ns)

        elif self._config.strategy_type == StrategyType.PURE_MARKET_MAKING:
            signal = await self._process_pure_mm(close_price, timestamp_ns)

        elif self._config.strategy_type == StrategyType.CROSS_EXCHANGE_MM:
            # XEMM requires quotes from both exchanges
            # For now, return None and handle via separate method
            signal = None

        if signal:
            self._last_signal_time_ns = timestamp_ns

        return signal

    async def _process_avellaneda(
        self,
        mid_price: Decimal,
        timestamp_ns: int,
    ) -> Signal | None:
        """Process Avellaneda-Stoikov strategy."""
        if not self._avellaneda or not self._config:
            return None

        # Generate quote
        quote = self._avellaneda.generate_quote(
            mid_price=mid_price,
            order_size=self._config.order_amount,
            timestamp_ns=timestamp_ns,
        )

        if quote is None:
            return None

        self._last_quote = quote

        # Create signal with quote information
        return Signal(
            signal_type=SignalType.HOLD,  # Market making is continuous
            symbol=self._config.symbol,
            timestamp_ns=timestamp_ns,
            strength=1.0,
            price=mid_price,
            metadata={
                "strategy": "avellaneda_stoikov",
                "action": "quote",
                "bid_price": str(quote.bid_price),
                "ask_price": str(quote.ask_price),
                "bid_size": str(quote.bid_size),
                "ask_size": str(quote.ask_size),
                "spread": str(quote.spread),
                "spread_pct": quote.spread_pct,
            },
        )

    async def _process_pure_mm(
        self,
        mid_price: Decimal,
        timestamp_ns: int,
    ) -> Signal | None:
        """Process Pure Market Making strategy."""
        if not self._pure_mm or not self._config:
            return None

        # Check if we need to refresh
        if not self._pure_mm.should_refresh_orders(
            self._last_quote if isinstance(self._last_quote, TwoSidedOrder) else None,
            mid_price,
        ):
            return None

        # Generate orders
        orders = self._pure_mm.generate_orders(
            mid_price=mid_price,
            order_size=self._config.order_amount,
            timestamp_ns=timestamp_ns,
        )

        self._last_quote = orders

        # Convert to signal
        bids = [{"price": str(b.price), "size": str(b.size), "level": b.level} for b in orders.bids]
        asks = [{"price": str(a.price), "size": str(a.size), "level": a.level} for a in orders.asks]

        return Signal(
            signal_type=SignalType.HOLD,
            symbol=self._config.symbol,
            timestamp_ns=timestamp_ns,
            strength=1.0,
            price=mid_price,
            metadata={
                "strategy": "pure_market_making",
                "action": "quote",
                "bids": bids,
                "asks": asks,
                "mid_price": str(mid_price),
                "spread": str(orders.spread) if orders.spread else None,
            },
        )

    async def process_xemm_quotes(
        self,
        maker_bid: Decimal,
        maker_ask: Decimal,
        taker_bid: Decimal,
        taker_ask: Decimal,
        timestamp_ns: int,
    ) -> list[XEMMOrder]:
        """
        Process XEMM with quotes from both exchanges.

        This method should be called with quotes from both maker and taker exchanges.

        Args:
            maker_bid: Bid price on maker exchange
            maker_ask: Ask price on maker exchange
            taker_bid: Bid price on taker exchange
            taker_ask: Ask price on taker exchange
            timestamp_ns: Current timestamp

        Returns:
            List of orders to place on maker exchange
        """
        if not self._xemm or not self._config:
            return []

        quotes = CrossExchangeQuote(
            maker_bid=maker_bid,
            maker_ask=maker_ask,
            taker_bid=taker_bid,
            taker_ask=taker_ask,
            timestamp_ns=timestamp_ns,
        )

        return self._xemm.generate_maker_orders(quotes, self._config.order_amount)

    def set_inventory(
        self,
        base_balance: Decimal,
        quote_balance: Decimal,
        mid_price: Decimal,
    ) -> None:
        """
        Update inventory for the active strategy.

        Args:
            base_balance: Amount of base currency
            quote_balance: Amount of quote currency
            mid_price: Current mid price
        """
        if self._avellaneda:
            self._avellaneda.set_inventory(base_balance, quote_balance, mid_price)
        if self._pure_mm:
            self._pure_mm.set_inventory(base_balance, quote_balance, mid_price)

    async def on_fill(self, order_result: dict[str, Any]) -> None:
        """
        Handle order fill notification.

        For XEMM, this triggers hedge orders on the taker exchange.

        Args:
            order_result: Order fill information
        """
        if not self._xemm or not self._config:
            return

        if self._config.strategy_type != StrategyType.CROSS_EXCHANGE_MM:
            return

        # Extract fill details
        side = order_result.get("side", "")
        size = Decimal(str(order_result.get("filled_amount", 0)))
        price = Decimal(str(order_result.get("average_price", 0)))

        if size <= 0:
            return

        # Generate hedge instruction
        hedge = self._xemm.on_maker_fill(side, size, price)

        if hedge:
            # Store hedge instruction for execution
            self._pending_orders.append({
                "type": "hedge",
                "exchange": self._config.xemm.taker_exchange,
                "side": hedge.side.value,
                "size": str(hedge.size),
                "max_price": str(hedge.max_price) if hedge.max_price else None,
                "min_price": str(hedge.min_price) if hedge.min_price else None,
                "urgency": hedge.urgency,
            })

    def get_pending_orders(self) -> list[dict[str, Any]]:
        """Get and clear pending orders (e.g., hedge orders for XEMM)."""
        orders = self._pending_orders.copy()
        self._pending_orders.clear()
        return orders

    def get_statistics(self) -> dict[str, Any]:
        """Get strategy statistics."""
        stats: dict[str, Any] = {
            "strategy_type": self._config.strategy_type.value if self._config else None,
            "symbol": self._config.symbol if self._config else None,
            "initialized": self._initialized,
        }

        if self._xemm:
            stats.update({
                "unhedged_position": str(self._xemm.unhedged_position),
                "total_profit": str(self._xemm.total_profit),
                "trade_count": self._xemm.trade_count,
            })

        return stats

    def get_performance_stats(self) -> PerformanceStats | None:
        """
        Get comprehensive performance statistics.

        Returns:
            PerformanceStats with P&L, trade stats, and risk metrics,
            or None if performance tracking is disabled.
        """
        if self._performance_tracker is None:
            return None
        return self._performance_tracker.get_stats()

    def get_performance_summary(self) -> dict[str, Any] | None:
        """
        Get performance summary as dictionary.

        Returns:
            Dictionary with equity, positions, P&L, and key statistics,
            or None if performance tracking is disabled.
        """
        if self._performance_tracker is None:
            return None
        return self._performance_tracker.to_dict()

    def update_price(self, symbol: str, price: Decimal) -> None:
        """
        Update market price for performance tracking.

        Args:
            symbol: Trading symbol
            price: Current market price
        """
        if self._performance_tracker:
            self._performance_tracker.update_price(symbol, price)

    async def shutdown(self) -> None:
        """Shutdown the adapter and clean up resources."""
        # Reset all strategies
        if self._avellaneda:
            self._avellaneda.reset()
        if self._pure_mm:
            self._pure_mm.reset()
        if self._xemm:
            self._xemm.reset()

        # Reset performance tracker
        if self._performance_tracker:
            self._performance_tracker.reset()

        self._last_quote = None
        self._pending_orders.clear()
        self._initialized = False

"""
UnifiedSMACross: Example strategy demonstrating unified backtest/live code.

This strategy uses the same code for both backtesting and live trading,
implementing Issue #37 - Event-Driven Backtest Engine with Unified Strategy Code.

Key features:
- Uses BaseStrategy from strategies.strategy (not strategies.base)
- Async on_bar() method for both backtest and live
- Direct order execution via gateway
- Receives order filled events

Usage:
    # In backtest (gateway is injected by BacktestEngine)
    strategy = UnifiedSMACross(None, config)  # Gateway set by engine
    engine = BacktestEngine()
    engine.add_strategy(strategy)
    result = await engine.run()

    # In live trading
    async with CCXTGateway("binance", api_config) as gateway:
        strategy = UnifiedSMACross(gateway, config)
        kernel = TradingKernel(config)
        kernel.add_strategy(strategy)
        await kernel.start_async()
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.strategies.strategy import (
    BaseStrategy,
    OrderFilledEvent,
    PositionOpenedEvent,
    PositionClosedEvent,
)


if TYPE_CHECKING:
    from libra.gateways.protocol import Gateway
    from libra.strategies.protocol import Bar


logger = logging.getLogger(__name__)


@dataclass
class UnifiedSMACrossConfig:
    """Configuration for UnifiedSMACross strategy."""

    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    fast_period: int = 10
    slow_period: int = 30
    position_size_pct: float = 0.95  # Use 95% of available capital


class UnifiedSMACross(BaseStrategy):
    """
    SMA Crossover strategy with unified backtest/live code.

    This strategy demonstrates the key principle of Issue #37:
    **Same code runs in both backtest and live trading.**

    Trading Logic:
    - Calculate fast and slow simple moving averages
    - BUY when fast SMA crosses above slow SMA (bullish)
    - SELL when fast SMA crosses below slow SMA (bearish)

    Example:
        # Works in both backtest and live!
        config = UnifiedSMACrossConfig(
            symbol="BTC/USDT",
            fast_period=10,
            slow_period=30,
        )
        strategy = UnifiedSMACross(gateway, config)
    """

    def __init__(
        self,
        gateway: Gateway | None,
        config: UnifiedSMACrossConfig | None = None,
    ) -> None:
        """
        Initialize strategy.

        Args:
            gateway: Gateway for order execution (None in backtest, set by engine)
            config: Strategy configuration
        """
        # Gateway can be None - BacktestEngine will inject it
        super().__init__(gateway)  # type: ignore[arg-type]

        self._config = config or UnifiedSMACrossConfig()
        self._closes: deque[Decimal] = deque(maxlen=self._config.slow_period)
        self._has_position = False

    @property
    def name(self) -> str:
        """Strategy name."""
        return f"unified_sma_{self._config.fast_period}_{self._config.slow_period}"

    @property
    def timeframe(self) -> str:
        """Strategy timeframe."""
        return self._config.timeframe

    async def on_start(self) -> None:
        """Called when strategy starts."""
        await super().on_start()
        self._closes.clear()
        self._has_position = False
        self.log.info(
            "UnifiedSMACross started: symbol=%s fast=%d slow=%d",
            self._config.symbol,
            self._config.fast_period,
            self._config.slow_period,
        )

    async def on_bar(self, bar: Bar) -> None:
        """
        Process bar data - IDENTICAL in backtest and live.

        This is the key method that runs the same in both environments.
        """
        # Store close price
        self._closes.append(bar.close)

        # Need enough data for slow SMA
        if len(self._closes) < self._config.slow_period:
            return

        # Calculate SMAs
        closes_list = list(self._closes)
        fast_sma = sum(closes_list[-self._config.fast_period:]) / self._config.fast_period
        slow_sma = sum(closes_list) / self._config.slow_period

        # Trading logic
        if fast_sma > slow_sma and not self._has_position:
            # Bullish crossover - open long
            await self._open_long(bar)

        elif fast_sma < slow_sma and self._has_position:
            # Bearish crossover - close long
            await self._close_long(bar)

    async def _open_long(self, bar: Bar) -> None:
        """Open a long position."""
        # Get available balance
        balance = await self._gateway.get_balance(
            bar.symbol.split("/")[1] if "/" in bar.symbol else "USDT"
        )
        if balance is None or balance.available <= 0:
            return

        # Calculate position size
        order_value = balance.available * Decimal(str(self._config.position_size_pct))
        amount = order_value / bar.close

        # Execute market buy
        self.log.info("Opening long: %s @ %s", amount, bar.close)
        await self.buy_market(bar.symbol, amount)
        self._has_position = True

    async def _close_long(self, bar: Bar) -> None:
        """Close long position."""
        await self.close_position(bar.symbol)
        self._has_position = False
        self.log.info("Closed long position @ %s", bar.close)

    async def on_order_filled(self, event: OrderFilledEvent) -> None:
        """
        Handle order fill - IDENTICAL in backtest and live.

        This demonstrates that the strategy receives order events
        in both environments.
        """
        self.log.info(
            "Order filled: %s @ %s (amount=%s)",
            event.symbol,
            event.fill_price,
            event.fill_amount,
        )

    async def on_position_opened(self, event: PositionOpenedEvent) -> None:
        """Handle position opened event."""
        self.log.info(
            "Position opened: %s %s @ %s",
            event.position.side.value,
            event.position.amount,
            event.position.entry_price,
        )

    async def on_position_closed(self, event: PositionClosedEvent) -> None:
        """Handle position closed event."""
        self.log.info(
            "Position closed: %s PnL=%s",
            event.symbol,
            event.realized_pnl,
        )

    async def on_stop(self) -> None:
        """Called when strategy stops."""
        # Close any open positions
        if self._has_position:
            self.log.info("Stopping - closing open position")
            await self.close_all_positions()
        await super().on_stop()

    async def on_reset(self) -> None:
        """Reset strategy state."""
        await super().on_reset()
        self._closes.clear()
        self._has_position = False

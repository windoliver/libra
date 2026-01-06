"""
SMA Crossover Live Strategy: Actor-based implementation with order execution.

A trend-following strategy using the new Actor/Strategy pattern that:
- Follows full lifecycle (INITIALIZED → RUNNING → STOPPED → DISPOSED)
- Executes orders through Gateway
- Handles order and position events
- Supports live trading, paper trading, and backtesting

Demonstrates:
- BaseStrategy usage (extends BaseActor)
- Order execution with event handling
- Position management
- State persistence
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.core.events import EventType
from libra.strategies.protocol import Bar, SignalType, StrategyConfig
from libra.strategies.strategy import (
    BaseStrategy,
    OrderFilledEvent,
    PositionClosedEvent,
    PositionOpenedEvent,
)


if TYPE_CHECKING:
    from libra.gateways.protocol import Gateway


@dataclass
class SMACrossLiveConfig(StrategyConfig):
    """
    Configuration for SMA Crossover Live Strategy.

    Attributes:
        symbol: Trading pair (inherited)
        timeframe: Bar timeframe (inherited)
        fast_period: Fast SMA period (default 10)
        slow_period: Slow SMA period (default 20)
        order_size: Order size in base currency (e.g., 0.1 BTC)
        use_market_orders: Use market orders (True) or limit orders (False)
    """

    fast_period: int = 10
    slow_period: int = 20
    order_size: Decimal = Decimal("0.1")
    use_market_orders: bool = True


class SMACrossLiveStrategy(BaseStrategy):
    """
    SMA Crossover Strategy with live order execution.

    Uses the Actor/Strategy pattern with full lifecycle support:
    - Subscribes to BAR events on start
    - Generates signals on MA crossover
    - Executes orders through Gateway
    - Tracks positions and P&L

    Example:
        # Create gateway and strategy
        gateway = PaperGateway(config)
        config = SMACrossLiveConfig(
            symbol="BTC/USDT",
            fast_period=10,
            slow_period=20,
            order_size=Decimal("0.1"),
        )
        strategy = SMACrossLiveStrategy(gateway, config)

        # Initialize with message bus
        await strategy.initialize(message_bus)

        # Run strategy
        async with strategy:
            # Strategy is now running and processing events
            await asyncio.sleep(3600)  # Run for 1 hour
    """

    def __init__(self, gateway: Gateway, config: SMACrossLiveConfig) -> None:
        """
        Initialize SMA Crossover Live Strategy.

        Args:
            gateway: Gateway for order execution
            config: Strategy configuration
        """
        super().__init__(gateway)
        self._strategy_config = config

        # Price buffers for MA calculation
        self._closes: deque[Decimal] = deque(maxlen=config.slow_period)

        # State tracking
        self._prev_fast_ma: Decimal | None = None
        self._prev_slow_ma: Decimal | None = None

        # P&L tracking
        self._total_pnl = Decimal("0")
        self._trades_won = 0
        self._trades_lost = 0

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "sma_cross_live"

    @property
    def symbol(self) -> str:
        """Trading symbol."""
        return self._strategy_config.symbol

    @property
    def fast_period(self) -> int:
        """Fast SMA period."""
        return self._strategy_config.fast_period

    @property
    def slow_period(self) -> int:
        """Slow SMA period."""
        return self._strategy_config.slow_period

    @property
    def order_size(self) -> Decimal:
        """Order size."""
        return self._strategy_config.order_size

    @property
    def total_pnl(self) -> Decimal:
        """Total realized P&L."""
        return self._total_pnl

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        total = self._trades_won + self._trades_lost
        if total == 0:
            return 0.0
        return (self._trades_won / total) * 100

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    async def on_start(self) -> None:
        """Initialize strategy and subscribe to events."""
        await super().on_start()

        # Clear state
        self._closes.clear()
        self._prev_fast_ma = None
        self._prev_slow_ma = None

        # Subscribe to bar events
        await self.subscribe(EventType.BAR)

        self.log.info(
            "Started %s: symbol=%s, fast=%d, slow=%d, size=%s",
            self.name,
            self.symbol,
            self.fast_period,
            self.slow_period,
            self.order_size,
        )

    async def on_stop(self) -> None:
        """Close positions and cleanup."""
        # Close any open position before stopping
        if not self.is_flat(self.symbol):
            self.log.info("Closing position on stop")
            await self.close_position(self.symbol)

        await super().on_stop()

        self.log.info(
            "Stopped %s: pnl=%s, trades=%d, win_rate=%.1f%%",
            self.name,
            self.total_pnl,
            self._trades_won + self._trades_lost,
            self.win_rate,
        )

    async def on_reset(self) -> None:
        """Reset strategy state between runs."""
        await super().on_reset()
        self._closes.clear()
        self._prev_fast_ma = None
        self._prev_slow_ma = None
        self._total_pnl = Decimal("0")
        self._trades_won = 0
        self._trades_lost = 0

    # =========================================================================
    # Bar Processing
    # =========================================================================

    async def on_bar(self, bar: Bar) -> None:
        """
        Process bar and execute trades on MA crossover.

        Args:
            bar: OHLCV bar data
        """
        # Only process bars for our symbol
        if bar.symbol != self.symbol:
            return

        # Add close price to buffer
        self._closes.append(bar.close)

        # Need enough data for slow MA
        if len(self._closes) < self.slow_period:
            return

        # Calculate MAs
        fast_ma = self._calculate_sma(self.fast_period)
        slow_ma = self._calculate_sma(self.slow_period)

        # Check for crossover (need previous values)
        if self._prev_fast_ma is not None and self._prev_slow_ma is not None:
            await self._check_crossover(bar, fast_ma, slow_ma)

        # Update previous values
        self._prev_fast_ma = fast_ma
        self._prev_slow_ma = slow_ma

    async def _check_crossover(
        self,
        bar: Bar,
        fast_ma: Decimal,
        slow_ma: Decimal,
    ) -> None:
        """Check for MA crossover and execute trades."""
        # Golden Cross: fast crosses above slow
        prev_below = self._prev_fast_ma <= self._prev_slow_ma
        curr_above = fast_ma > slow_ma

        if prev_below and curr_above and self.is_flat(self.symbol):
            self.log.info(
                "Golden Cross detected: fast_ma=%s > slow_ma=%s",
                fast_ma,
                slow_ma,
            )

            # Create signal
            signal = self.create_signal(
                SignalType.LONG,
                self.symbol,
                price=bar.close,
                metadata={
                    "reason": "golden_cross",
                    "fast_ma": str(fast_ma),
                    "slow_ma": str(slow_ma),
                },
            )

            # Execute buy order
            if self._strategy_config.use_market_orders:
                await self.buy_market(self.symbol, self.order_size)
            else:
                await self.buy_limit(self.symbol, self.order_size, bar.close)

        # Death Cross: fast crosses below slow
        prev_above = self._prev_fast_ma >= self._prev_slow_ma
        curr_below = fast_ma < slow_ma

        if prev_above and curr_below and self.is_long(self.symbol):
            self.log.info(
                "Death Cross detected: fast_ma=%s < slow_ma=%s",
                fast_ma,
                slow_ma,
            )

            # Create signal
            signal = self.create_signal(
                SignalType.CLOSE_LONG,
                self.symbol,
                price=bar.close,
                metadata={
                    "reason": "death_cross",
                    "fast_ma": str(fast_ma),
                    "slow_ma": str(slow_ma),
                },
            )

            # Close position
            await self.close_position(self.symbol)

    def _calculate_sma(self, period: int) -> Decimal:
        """Calculate SMA for given period."""
        values = list(self._closes)[-period:]
        return sum(values) / Decimal(period)

    # =========================================================================
    # Order/Position Event Handlers
    # =========================================================================

    async def on_order_filled(self, event: OrderFilledEvent) -> None:
        """Handle order fill event."""
        self.log.info(
            "Order filled: %s @ %s (amount=%s)",
            event.order_id,
            event.fill_price,
            event.fill_amount,
        )

    async def on_position_opened(self, event: PositionOpenedEvent) -> None:
        """Handle new position event."""
        self.log.info(
            "Position opened: %s %s @ %s",
            event.position.side.value,
            event.position.amount,
            event.position.entry_price,
        )

    async def on_position_closed(self, event: PositionClosedEvent) -> None:
        """Handle position closed event."""
        pnl = event.realized_pnl
        self._total_pnl += pnl

        if pnl > 0:
            self._trades_won += 1
        else:
            self._trades_lost += 1

        self.log.info(
            "Position closed: pnl=%s (total=%s, win_rate=%.1f%%)",
            pnl,
            self._total_pnl,
            self.win_rate,
        )

    # =========================================================================
    # State Persistence
    # =========================================================================

    def on_save(self) -> dict[str, bytes]:
        """Save strategy state."""
        state = {
            "closes": [str(c) for c in self._closes],
            "prev_fast_ma": str(self._prev_fast_ma) if self._prev_fast_ma else None,
            "prev_slow_ma": str(self._prev_slow_ma) if self._prev_slow_ma else None,
            "total_pnl": str(self._total_pnl),
            "trades_won": self._trades_won,
            "trades_lost": self._trades_lost,
            "signal_count": self.signal_count,
            "order_count": self.order_count,
        }
        return {"state": json.dumps(state).encode()}

    def on_load(self, state: dict[str, bytes]) -> None:
        """Load strategy state."""
        if "state" not in state:
            return

        data = json.loads(state["state"].decode())
        self._closes = deque(
            [Decimal(c) for c in data.get("closes", [])],
            maxlen=self.slow_period,
        )
        prev_fast = data.get("prev_fast_ma")
        self._prev_fast_ma = Decimal(prev_fast) if prev_fast else None
        prev_slow = data.get("prev_slow_ma")
        self._prev_slow_ma = Decimal(prev_slow) if prev_slow else None
        self._total_pnl = Decimal(data.get("total_pnl", "0"))
        self._trades_won = data.get("trades_won", 0)
        self._trades_lost = data.get("trades_lost", 0)

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_current_mas(self) -> tuple[Decimal | None, Decimal | None]:
        """
        Get current MA values for debugging/display.

        Returns:
            Tuple of (fast_ma, slow_ma), None if not enough data
        """
        if len(self._closes) < self.slow_period:
            return None, None

        fast_ma = self._calculate_sma(self.fast_period)
        slow_ma = self._calculate_sma(self.slow_period)
        return fast_ma, slow_ma

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "state": self.state.name,
            "signal_count": self.signal_count,
            "order_count": self.order_count,
            "total_pnl": str(self._total_pnl),
            "trades_won": self._trades_won,
            "trades_lost": self._trades_lost,
            "win_rate": f"{self.win_rate:.1f}%",
            "fast_ma": str(self._prev_fast_ma) if self._prev_fast_ma else None,
            "slow_ma": str(self._prev_slow_ma) if self._prev_slow_ma else None,
        }

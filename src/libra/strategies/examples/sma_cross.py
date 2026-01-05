"""
SMA Crossover Strategy: Reference implementation.

A classic trend-following strategy that generates:
- LONG signal when fast MA crosses above slow MA (golden cross)
- CLOSE_LONG signal when fast MA crosses below slow MA (death cross)

This serves as a reference implementation demonstrating:
- Strategy Protocol compliance
- BaseStrategy usage
- Proper lifecycle management
- Signal generation
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from libra.strategies.base import BaseStrategy
from libra.strategies.protocol import (
    Bar,
    Signal,
    StrategyConfig,
)


@dataclass
class SMACrossConfig(StrategyConfig):
    """
    Configuration for SMA Crossover Strategy.

    Attributes:
        symbol: Trading pair (inherited)
        timeframe: Bar timeframe (inherited)
        fast_period: Fast SMA period (default 10)
        slow_period: Slow SMA period (default 20)
        position_size: Position size as decimal (default 1.0 = 100%)

    Examples:
        config = SMACrossConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            fast_period=10,
            slow_period=20,
        )
    """

    fast_period: int = 10
    slow_period: int = 20
    position_size: Decimal = Decimal("1.0")


class SMACrossStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Generates trading signals based on SMA crossovers:
    - Golden Cross (fast > slow): LONG signal
    - Death Cross (fast < slow): CLOSE_LONG signal

    This is a long-only strategy suitable for trending markets.

    Example:
        config = SMACrossConfig(
            symbol="BTC/USDT",
            fast_period=10,
            slow_period=20,
        )
        strategy = SMACrossStrategy(config)
        strategy.on_start()

        for bar in bars:
            signal = strategy.on_bar(bar)
            if signal:
                print(f"Signal: {signal.signal_type}")
    """

    def __init__(self, config: SMACrossConfig) -> None:
        """
        Initialize SMA Crossover Strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        self._config: SMACrossConfig = config

        # Price buffers for MA calculation
        self._closes: deque[Decimal] = deque(maxlen=config.slow_period)

        # State tracking
        self._prev_fast_ma: Decimal | None = None
        self._prev_slow_ma: Decimal | None = None
        self._in_position = False

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "sma_cross"

    @property
    def fast_period(self) -> int:
        """Fast SMA period."""
        return self._config.fast_period

    @property
    def slow_period(self) -> int:
        """Slow SMA period."""
        return self._config.slow_period

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """Initialize strategy state."""
        super().on_start()
        self._closes.clear()
        self._prev_fast_ma = None
        self._prev_slow_ma = None
        self._in_position = False

    def on_stop(self) -> None:
        """Cleanup strategy state."""
        super().on_stop()

    def on_reset(self) -> None:
        """Reset strategy between backtest runs."""
        super().on_reset()
        self._closes.clear()
        self._prev_fast_ma = None
        self._prev_slow_ma = None
        self._in_position = False

    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> Signal | None:
        """
        Process bar and generate signal on MA crossover.

        Args:
            bar: OHLCV bar data

        Returns:
            LONG on golden cross, CLOSE_LONG on death cross, None otherwise
        """
        # Add close price to buffer
        self._closes.append(bar.close)

        # Need enough data for slow MA
        if len(self._closes) < self.slow_period:
            return None

        # Calculate MAs
        fast_ma = self.sma(self._closes, self.fast_period)
        slow_ma = self.sma(self._closes, self.slow_period)

        signal: Signal | None = None

        # Check for crossover (need previous values)
        if self._prev_fast_ma is not None and self._prev_slow_ma is not None:
            # Golden Cross: fast crosses above slow
            prev_below = self._prev_fast_ma <= self._prev_slow_ma
            curr_above = fast_ma > slow_ma

            if prev_below and curr_above and not self._in_position:
                signal = self._long(
                    strength=float(self._config.position_size),
                    price=bar.close,
                    metadata={
                        "reason": "golden_cross",
                        "fast_ma": str(fast_ma),
                        "slow_ma": str(slow_ma),
                    },
                )
                self._in_position = True

            # Death Cross: fast crosses below slow
            prev_above = self._prev_fast_ma >= self._prev_slow_ma
            curr_below = fast_ma < slow_ma

            if prev_above and curr_below and self._in_position:
                signal = self._close_long(
                    price=bar.close,
                    metadata={
                        "reason": "death_cross",
                        "fast_ma": str(fast_ma),
                        "slow_ma": str(slow_ma),
                    },
                )
                self._in_position = False

        # Update previous values
        self._prev_fast_ma = fast_ma
        self._prev_slow_ma = slow_ma

        return signal

    # -------------------------------------------------------------------------
    # State Persistence
    # -------------------------------------------------------------------------

    def on_save(self) -> dict[str, bytes]:
        """Save strategy state."""
        state = {
            "closes": [str(c) for c in self._closes],
            "prev_fast_ma": str(self._prev_fast_ma) if self._prev_fast_ma else None,
            "prev_slow_ma": str(self._prev_slow_ma) if self._prev_slow_ma else None,
            "in_position": self._in_position,
            "bar_count": self._bar_count,
            "signal_count": self._signal_count,
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
        self._in_position = data.get("in_position", False)
        self._bar_count = data.get("bar_count", 0)
        self._signal_count = data.get("signal_count", 0)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_current_mas(self) -> tuple[Decimal | None, Decimal | None]:
        """
        Get current MA values for debugging/display.

        Returns:
            Tuple of (fast_ma, slow_ma), None if not enough data
        """
        if len(self._closes) < self.slow_period:
            return None, None

        fast_ma = self.sma(self._closes, self.fast_period)
        slow_ma = self.sma(self._closes, self.slow_period)
        return fast_ma, slow_ma

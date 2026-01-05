"""
Base Strategy: Abstract base class for strategy implementations.

Provides:
- Default implementations for optional methods
- Common utility methods
- Indicator helpers
- State management

Subclass this for concrete strategy implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque  # noqa: TC003 - used at runtime in type annotations
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.strategies.protocol import (
    Bar,
    Signal,
    SignalType,
    StrategyConfig,
)


if TYPE_CHECKING:
    from libra.gateways.protocol import Tick


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Provides common functionality:
    - Configuration management
    - State tracking
    - Indicator utilities (SMA, EMA, etc.)
    - Default implementations for optional methods

    Subclasses must implement:
    - name property
    - on_bar() method

    Example:
        class MyStrategy(BaseStrategy):
            def __init__(self, config: StrategyConfig):
                super().__init__(config)
                self._sma_buffer: deque[Decimal] = deque(maxlen=20)

            @property
            def name(self) -> str:
                return "my_strategy"

            def on_bar(self, bar: Bar) -> Signal | None:
                self._sma_buffer.append(bar.close)
                if len(self._sma_buffer) < 20:
                    return None

                sma = sum(self._sma_buffer) / len(self._sma_buffer)
                if bar.close > sma:
                    return Signal.create(SignalType.LONG, bar.symbol)
                return None
    """

    def __init__(self, config: StrategyConfig) -> None:
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy configuration
        """
        self._config = config
        self._is_running = False
        self._bar_count = 0
        self._signal_count = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (must be implemented by subclass)."""
        ...

    @property
    def config(self) -> StrategyConfig:
        """Strategy configuration."""
        return self._config

    @property
    def symbol(self) -> str:
        """Trading symbol from config."""
        return self._config.symbol

    @property
    def timeframe(self) -> str:
        """Timeframe from config."""
        return self._config.timeframe

    @property
    def is_running(self) -> bool:
        """Check if strategy is currently running."""
        return self._is_running

    @property
    def bar_count(self) -> int:
        """Number of bars processed."""
        return self._bar_count

    @property
    def signal_count(self) -> int:
        """Number of signals generated."""
        return self._signal_count

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """
        Called when strategy starts.

        Override to add custom initialization.
        Always call super().on_start() first.
        """
        self._is_running = True
        self._bar_count = 0
        self._signal_count = 0

    def on_stop(self) -> None:
        """
        Called when strategy stops.

        Override to add custom cleanup.
        Always call super().on_stop() first.
        """
        self._is_running = False

    def on_reset(self) -> None:
        """
        Reset strategy state between backtest runs.

        Override to reset custom state.
        Always call super().on_reset() first.
        """
        self._bar_count = 0
        self._signal_count = 0

    # -------------------------------------------------------------------------
    # Data Handlers (must be implemented)
    # -------------------------------------------------------------------------

    @abstractmethod
    def on_bar(self, bar: Bar) -> Signal | None:
        """
        Process OHLCV bar data (must be implemented by subclass).

        Args:
            bar: OHLCV bar data

        Returns:
            Signal if strategy wants to trade, None otherwise.
        """
        ...

    def on_tick(self, tick: Tick) -> Signal | None:  # noqa: ARG002
        """
        Process tick/quote data (optional override).

        Default implementation returns None (no signal).

        Args:
            tick: Real-time tick data

        Returns:
            Signal if strategy wants to trade, None otherwise.
        """
        return None

    # -------------------------------------------------------------------------
    # State Persistence (optional override)
    # -------------------------------------------------------------------------

    def on_save(self) -> dict[str, bytes]:
        """
        Save strategy state (optional override).

        Default implementation returns empty dict.
        """
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:  # noqa: B027
        """
        Load strategy state (optional override).

        Default implementation does nothing.
        """

    # -------------------------------------------------------------------------
    # Signal Helpers
    # -------------------------------------------------------------------------

    def _create_signal(
        self,
        signal_type: SignalType,
        strength: float = 1.0,
        price: Decimal | None = None,
        metadata: dict | None = None,
    ) -> Signal:
        """
        Create a signal with automatic symbol and tracking.

        Args:
            signal_type: Type of signal
            strength: Signal confidence (0.0 to 1.0)
            price: Reference price
            metadata: Additional data

        Returns:
            New Signal instance
        """
        self._signal_count += 1
        return Signal.create(
            signal_type=signal_type,
            symbol=self.symbol,
            strength=strength,
            price=price,
            metadata=metadata,
        )

    def _long(
        self,
        strength: float = 1.0,
        price: Decimal | None = None,
        metadata: dict | None = None,
    ) -> Signal:
        """Create a LONG signal."""
        return self._create_signal(SignalType.LONG, strength, price, metadata)

    def _short(
        self,
        strength: float = 1.0,
        price: Decimal | None = None,
        metadata: dict | None = None,
    ) -> Signal:
        """Create a SHORT signal."""
        return self._create_signal(SignalType.SHORT, strength, price, metadata)

    def _close_long(
        self,
        strength: float = 1.0,
        price: Decimal | None = None,
        metadata: dict | None = None,
    ) -> Signal:
        """Create a CLOSE_LONG signal."""
        return self._create_signal(SignalType.CLOSE_LONG, strength, price, metadata)

    def _close_short(
        self,
        strength: float = 1.0,
        price: Decimal | None = None,
        metadata: dict | None = None,
    ) -> Signal:
        """Create a CLOSE_SHORT signal."""
        return self._create_signal(SignalType.CLOSE_SHORT, strength, price, metadata)

    # -------------------------------------------------------------------------
    # Indicator Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def sma(values: list[Decimal] | deque[Decimal], period: int | None = None) -> Decimal:
        """
        Calculate Simple Moving Average.

        Args:
            values: List of values (most recent last)
            period: Number of periods (default: use all values)

        Returns:
            SMA value

        Raises:
            ValueError: If not enough values
        """
        if period is None:
            period = len(values)
        if len(values) < period:
            raise ValueError(f"Need at least {period} values, got {len(values)}")

        # Use last `period` values
        recent = list(values)[-period:]
        return sum(recent) / Decimal(period)

    @staticmethod
    def ema(
        values: list[Decimal] | deque[Decimal],
        period: int,
        smoothing: float = 2.0,
    ) -> Decimal:
        """
        Calculate Exponential Moving Average.

        Args:
            values: List of values (most recent last)
            period: EMA period
            smoothing: Smoothing factor (default 2.0)

        Returns:
            EMA value

        Raises:
            ValueError: If not enough values
        """
        if len(values) < period:
            raise ValueError(f"Need at least {period} values, got {len(values)}")

        multiplier = Decimal(str(smoothing / (period + 1)))

        # Start with SMA of first `period` values
        values_list = list(values)
        ema = sum(values_list[:period]) / Decimal(period)

        # Calculate EMA for remaining values
        for value in values_list[period:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))

        return ema

    @staticmethod
    def rsi(
        values: list[Decimal] | deque[Decimal],
        period: int = 14,
    ) -> Decimal:
        """
        Calculate Relative Strength Index.

        Args:
            values: List of closing prices (most recent last)
            period: RSI period (default 14)

        Returns:
            RSI value (0-100)

        Raises:
            ValueError: If not enough values
        """
        if len(values) < period + 1:
            raise ValueError(f"Need at least {period + 1} values, got {len(values)}")

        values_list = list(values)[-(period + 1) :]

        gains: list[Decimal] = []
        losses: list[Decimal] = []

        for i in range(1, len(values_list)):
            change = values_list[i] - values_list[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        avg_gain = sum(gains) / Decimal(period)
        avg_loss = sum(losses) / Decimal(period)

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        return Decimal("100") - (Decimal("100") / (1 + rs))

    @staticmethod
    def stddev(values: list[Decimal] | deque[Decimal], period: int | None = None) -> Decimal:
        """
        Calculate Standard Deviation.

        Args:
            values: List of values
            period: Number of periods (default: use all values)

        Returns:
            Standard deviation

        Raises:
            ValueError: If not enough values
        """
        if period is None:
            period = len(values)
        if len(values) < period:
            raise ValueError(f"Need at least {period} values, got {len(values)}")

        recent = list(values)[-period:]
        mean = sum(recent) / Decimal(period)
        variance = sum((x - mean) ** 2 for x in recent) / Decimal(period)
        return variance.sqrt()

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _process_bar(self, bar: Bar) -> Signal | None:
        """
        Internal bar processing with tracking.

        Called by the engine, delegates to on_bar().
        """
        self._bar_count += 1
        return self.on_bar(bar)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name!r}, symbol={self.symbol!r})"

"""
LIBRA Strategies: Unified strategy interface for backtest and live trading.

This module provides:
- Strategy Protocol: Standard interface for all strategies
- Signal types: LONG, SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
- BacktestResult: Comprehensive performance metrics
- BaseStrategy: Abstract base class with common utilities
- Example strategies: Reference implementations

Same interface works for both backtest and live trading (event-driven architecture).

Examples:
    # Using the SMA crossover example strategy
    from libra.strategies import SMACrossStrategy, SMACrossConfig

    config = SMACrossConfig(
        symbol="BTC/USDT",
        fast_period=10,
        slow_period=20,
    )
    strategy = SMACrossStrategy(config)
    strategy.on_start()

    # Process bars
    for bar in bars:
        signal = strategy.on_bar(bar)
        if signal:
            print(f"{signal.signal_type}: {signal.symbol}")

    # Creating custom strategies
    from libra.strategies import BaseStrategy, Bar, Signal, SignalType

    class MyStrategy(BaseStrategy):
        @property
        def name(self) -> str:
            return "my_strategy"

        def on_bar(self, bar: Bar) -> Signal | None:
            # Your strategy logic here
            if some_condition:
                return self._long(price=bar.close)
            return None
"""

# Protocol and types
# Base class
from libra.strategies.base import BaseStrategy

# Example strategies
from libra.strategies.examples import SMACrossConfig, SMACrossStrategy
from libra.strategies.protocol import (
    BacktestResult,
    Bar,
    Signal,
    SignalType,
    Strategy,
    StrategyConfig,
    decode_bar,
    decode_signal,
    encode_bar,
    encode_signal,
)


__all__ = [
    "BacktestResult",
    "Bar",
    "BaseStrategy",
    "SMACrossConfig",
    "SMACrossStrategy",
    "Signal",
    "SignalType",
    "Strategy",
    "StrategyConfig",
    "decode_bar",
    "decode_signal",
    "encode_bar",
    "encode_signal",
]

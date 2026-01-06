"""
LIBRA Strategies: Unified strategy interface for backtest and live trading.

This module provides:
- Actor Protocol: Base component with lifecycle and event handling
- Strategy Protocol: Trading strategy with order management
- Signal types: LONG, SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
- BacktestResult: Comprehensive performance metrics
- BaseActor: Abstract base class for actors
- BaseStrategy: Abstract base class for strategies with order routing
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

    # Creating custom strategies with lifecycle
    from libra.strategies import BaseStrategy, Bar, Signal, SignalType

    class MyStrategy(BaseStrategy):
        @property
        def name(self) -> str:
            return "my_strategy"

        async def on_bar(self, bar: Bar) -> None:
            if some_condition:
                await self.buy_market(bar.symbol, Decimal("0.1"))

        async def on_order_filled(self, event: OrderFilledEvent) -> None:
            self.log.info(f"Filled: {event.fill_amount}")
"""

# Actor module - lifecycle and event handling
from libra.strategies.actor import (
    Actor,
    BaseActor,
    ComponentState,
    InvalidStateTransition,
)

# Legacy base class (for backwards compatibility)
from libra.strategies.base import BaseStrategy as LegacyBaseStrategy

# Example strategies
from libra.strategies.examples import SMACrossConfig, SMACrossStrategy
from libra.strategies.protocol import (
    BacktestResult,
    Bar,
    Signal,
    SignalType,
    Strategy as LegacyStrategy,
    StrategyConfig,
    decode_bar,
    decode_signal,
    encode_bar,
    encode_signal,
)

# New Strategy module with order management
from libra.strategies.strategy import (
    BaseStrategy,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderSubmittedEvent,
    PositionChangedEvent,
    PositionClosedEvent,
    PositionOpenedEvent,
    Strategy,
)


__all__ = [
    # Actor
    "Actor",
    "BaseActor",
    "ComponentState",
    "InvalidStateTransition",
    # Strategy
    "BacktestResult",
    "Bar",
    "BaseStrategy",
    "LegacyBaseStrategy",
    "LegacyStrategy",
    "OrderAcceptedEvent",
    "OrderCanceledEvent",
    "OrderFilledEvent",
    "OrderRejectedEvent",
    "OrderSubmittedEvent",
    "PositionChangedEvent",
    "PositionClosedEvent",
    "PositionOpenedEvent",
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

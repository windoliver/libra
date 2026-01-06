# Strategy Development Guide

This guide covers how to develop trading strategies using Libra's Actor/Strategy framework.

## Overview

Libra follows the NautilusTrader-inspired Actor/Strategy pattern:

- **Actor**: Base component with lifecycle hooks and event handling
- **Strategy**: Extends Actor with order management and position tracking
- **MessageBus**: Central event routing system

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  MessageBus │────▶│   Strategy   │────▶│   Gateway   │
│  (Events)   │◀────│   (Actor)    │◀────│  (Orders)   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │
       │                   ▼
       │            ┌──────────────┐
       └───────────▶│   Events     │
                    │ TICK, BAR,   │
                    │ ORDER, etc.  │
                    └──────────────┘
```

## Component States

Strategies follow a state machine lifecycle:

```
PRE_INITIALIZED ─▶ READY ─▶ STARTING ─▶ RUNNING
                     │                      │
                     │                      ▼
                     │               DEGRADED ──────┐
                     │                      │       │
                     ▼                      ▼       │
                 DISPOSING          STOPPING ◀─────┘
                     │                      │
                     ▼                      ▼
                 DISPOSED            STOPPED
                                        │
                                        ▼
                                    FAULTED
```

Valid transitions:
- `PRE_INITIALIZED` → `READY` (after initialize)
- `READY` → `STARTING` → `RUNNING` (after start)
- `RUNNING` → `DEGRADED` (after degrade)
- `RUNNING/DEGRADED` → `STOPPING` → `STOPPED` (after stop)
- `STOPPED` → `RUNNING` (after resume)
- `STOPPED` → `DISPOSED` (after dispose)
- Any state → `FAULTED` (on unrecoverable error)

## Quick Start

### 1. Create a Strategy

```python
from decimal import Decimal
from libra.strategies.strategy import BaseStrategy
from libra.strategies.protocol import Bar, SignalType, StrategyConfig
from libra.core.events import EventType
from dataclasses import dataclass

@dataclass
class MyStrategyConfig(StrategyConfig):
    """Strategy configuration."""
    fast_period: int = 10
    slow_period: int = 20
    order_size: Decimal = Decimal("0.1")

class MyStrategy(BaseStrategy):
    """My custom trading strategy."""

    def __init__(self, gateway, config: MyStrategyConfig):
        super().__init__(gateway)
        self.config = config
        self._prices = []

    @property
    def name(self) -> str:
        return "my_strategy"

    async def on_start(self) -> None:
        """Called when strategy starts."""
        await super().on_start()
        # Subscribe to bar events
        await self.subscribe(EventType.BAR)
        self.log.info("Strategy started for %s", self.config.symbol)

    async def on_stop(self) -> None:
        """Called when strategy stops."""
        # Close any open positions
        if not self.is_flat(self.config.symbol):
            await self.close_position(self.config.symbol)
        await super().on_stop()

    async def on_bar(self, bar: Bar) -> None:
        """Process incoming bar data."""
        if bar.symbol != self.config.symbol:
            return

        self._prices.append(bar.close)

        # Implement your signal logic here
        if self._should_buy():
            await self.buy_market(
                self.config.symbol,
                self.config.order_size
            )
        elif self._should_sell():
            await self.close_position(self.config.symbol)

    def _should_buy(self) -> bool:
        """Your buy signal logic."""
        # Example: Buy when we have enough data
        return len(self._prices) >= self.config.slow_period

    def _should_sell(self) -> bool:
        """Your sell signal logic."""
        return False
```

### 2. Run the Strategy

```python
import asyncio
from libra.core.message_bus import MessageBus
from libra.gateways.paper_gateway import PaperGateway

async def main():
    # Create gateway
    gateway_config = {
        "initial_balance": {"USDT": Decimal("10000")},
        "slippage_model": "fixed",
        "slippage_bps": 5,
    }
    gateway = PaperGateway(gateway_config)

    # Create strategy
    config = MyStrategyConfig(
        symbol="BTC/USDT",
        fast_period=10,
        slow_period=20,
    )
    strategy = MyStrategy(gateway, config)

    # Create message bus
    bus = MessageBus()

    # Initialize and run
    await gateway.connect()
    await strategy.initialize(bus)

    async with bus:
        async with strategy:
            # Strategy is now running
            # Publish bars to message_bus or let gateway stream them
            await asyncio.sleep(3600)  # Run for 1 hour

    await gateway.disconnect()

asyncio.run(main())
```

## Lifecycle Hooks

Override these methods to customize strategy behavior:

### `on_start()`
Called when the strategy starts. Subscribe to events here.

```python
async def on_start(self) -> None:
    await super().on_start()
    await self.subscribe(EventType.BAR)
    await self.subscribe(EventType.TICK)
```

### `on_stop()`
Called when the strategy stops. Clean up resources here.

```python
async def on_stop(self) -> None:
    # Close positions gracefully
    if not self.is_flat(self.symbol):
        await self.close_position(self.symbol)
    await super().on_stop()
```

### `on_resume()`
Called when a stopped strategy resumes.

```python
async def on_resume(self) -> None:
    await super().on_resume()
    self.log.info("Strategy resumed")
```

### `on_reset()`
Called to reset strategy state between runs.

```python
async def on_reset(self) -> None:
    await super().on_reset()
    self._prices.clear()
    self._signals.clear()
```

### `on_degrade(reason)`
Called when the strategy enters degraded mode.

```python
async def on_degrade(self, reason: str) -> None:
    self.log.warning("Degraded: %s", reason)
    # Stop opening new positions, only close existing
```

### `on_fault(error)`
Called when an unrecoverable error occurs.

```python
async def on_fault(self, error: Exception) -> None:
    self.log.error("Faulted: %s", error)
    # Emergency cleanup
```

## Event Handling

### Bar Events

```python
async def on_bar(self, bar: Bar) -> None:
    """
    Process OHLCV bar data.

    Bar fields:
        - symbol: str
        - timestamp_ns: int
        - open: Decimal
        - high: Decimal
        - low: Decimal
        - close: Decimal
        - volume: Decimal
        - timeframe: str
    """
    self.log.info(
        "Bar: %s close=%s volume=%s",
        bar.symbol, bar.close, bar.volume
    )
```

### Tick Events

```python
async def on_tick(self, tick: Tick) -> None:
    """
    Process tick data.

    Tick fields:
        - symbol: str
        - bid: Decimal
        - ask: Decimal
        - last: Decimal
        - timestamp_ns: int
    """
    spread = tick.ask - tick.bid
    self.log.debug("Tick: %s spread=%s", tick.symbol, spread)
```

### Order Events

```python
async def on_order_filled(self, event: OrderFilledEvent) -> None:
    """Called when an order is filled."""
    self.log.info(
        "Order filled: %s @ %s",
        event.fill_amount, event.fill_price
    )

async def on_order_rejected(self, event: OrderRejectedEvent) -> None:
    """Called when an order is rejected."""
    self.log.error("Order rejected: %s", event.reason)
```

### Position Events

```python
async def on_position_opened(self, event: PositionOpenedEvent) -> None:
    """Called when a new position is opened."""
    pos = event.position
    self.log.info(
        "Position opened: %s %s @ %s",
        pos.side, pos.amount, pos.entry_price
    )

async def on_position_closed(self, event: PositionClosedEvent) -> None:
    """Called when a position is closed."""
    self.log.info("P&L: %s", event.realized_pnl)
```

## Order Management

### Market Orders

```python
# Buy at market price
result = await self.buy_market("BTC/USDT", Decimal("0.1"))

# Sell at market price
result = await self.sell_market("BTC/USDT", Decimal("0.1"))
```

### Limit Orders

```python
# Buy limit order
result = await self.buy_limit(
    "BTC/USDT",
    amount=Decimal("0.1"),
    price=Decimal("49000")
)

# Sell limit order
result = await self.sell_limit(
    "BTC/USDT",
    amount=Decimal("0.1"),
    price=Decimal("51000")
)
```

### Position Management

```python
# Check position status
if self.is_flat("BTC/USDT"):
    # No position
    pass
elif self.is_long("BTC/USDT"):
    # Long position
    pass
elif self.is_short("BTC/USDT"):
    # Short position
    pass

# Close position
await self.close_position("BTC/USDT")

# Cancel all orders
await self.cancel_all_orders("BTC/USDT")
```

## Signal Generation

```python
from libra.strategies.protocol import SignalType

# Create a signal
signal = self.create_signal(
    SignalType.LONG,
    symbol="BTC/USDT",
    price=Decimal("50000"),
    metadata={
        "reason": "golden_cross",
        "fast_ma": "50100",
        "slow_ma": "49900",
    }
)

# Signal types
SignalType.LONG        # Open long position
SignalType.SHORT       # Open short position
SignalType.CLOSE_LONG  # Close long position
SignalType.CLOSE_SHORT # Close short position
SignalType.HOLD        # No action
```

## State Persistence

Save and restore strategy state for crash recovery:

```python
def on_save(self) -> dict[str, bytes]:
    """Save strategy state."""
    import json
    state = {
        "prices": [str(p) for p in self._prices],
        "signal_count": self.signal_count,
    }
    return {"state": json.dumps(state).encode()}

def on_load(self, state: dict[str, bytes]) -> None:
    """Load strategy state."""
    import json
    if "state" in state:
        data = json.loads(state["state"].decode())
        self._prices = [Decimal(p) for p in data.get("prices", [])]
```

## Example: SMA Crossover Strategy

See `src/libra/strategies/examples/sma_cross_live.py` for a complete example:

```python
from libra.strategies.examples.sma_cross_live import (
    SMACrossLiveConfig,
    SMACrossLiveStrategy,
)

config = SMACrossLiveConfig(
    symbol="BTC/USDT",
    timeframe="1h",
    fast_period=10,
    slow_period=20,
    order_size=Decimal("0.1"),
    use_market_orders=True,
)
strategy = SMACrossLiveStrategy(gateway, config)
```

## Testing Strategies

### Unit Tests

```python
import pytest
from decimal import Decimal
from libra.core.message_bus import MessageBus
from libra.gateways.paper_gateway import PaperGateway

@pytest.fixture
def gateway():
    return PaperGateway({
        "initial_balance": {"USDT": Decimal("10000")}
    })

@pytest.fixture
def bus():
    return MessageBus()

@pytest.mark.asyncio
async def test_strategy_lifecycle(gateway, bus):
    strategy = MyStrategy(gateway, MyStrategyConfig())
    await gateway.connect()
    await strategy.initialize(bus)

    async with bus:
        await strategy.start()
        assert strategy.is_running

        await strategy.stop()
        assert strategy.is_stopped
```

### Integration Tests

See `tests/integration/test_actor_messagebus.py` for examples.

### E2E Tests

See `tests/e2e/test_trading_flow.py` for full trading flow tests.

## Best Practices

1. **Always call `super()`** in lifecycle hooks
2. **Subscribe to events in `on_start()`**, not in `__init__`
3. **Close positions in `on_stop()`** for graceful shutdown
4. **Use `is_flat()` checks** before opening positions
5. **Log important events** for debugging
6. **Implement `on_save()`/`on_load()`** for crash recovery
7. **Handle order rejections** gracefully
8. **Use paper trading** for testing before live trading

## API Reference

### BaseStrategy Properties

| Property | Type | Description |
|----------|------|-------------|
| `state` | `ComponentState` | Current lifecycle state |
| `is_running` | `bool` | True if RUNNING |
| `is_stopped` | `bool` | True if STOPPED |
| `is_degraded` | `bool` | True if DEGRADED |
| `signal_count` | `int` | Number of signals generated |
| `order_count` | `int` | Number of orders submitted |

### BaseStrategy Methods

| Method | Description |
|--------|-------------|
| `initialize(bus)` | Initialize with MessageBus |
| `start()` | Start the strategy |
| `stop()` | Stop the strategy |
| `resume()` | Resume from stopped |
| `reset()` | Reset strategy state |
| `degrade(reason)` | Enter degraded mode |
| `dispose()` | Dispose resources |
| `subscribe(event_type)` | Subscribe to events |
| `buy_market(symbol, amount)` | Submit market buy |
| `sell_market(symbol, amount)` | Submit market sell |
| `buy_limit(symbol, amount, price)` | Submit limit buy |
| `sell_limit(symbol, amount, price)` | Submit limit sell |
| `close_position(symbol)` | Close position |
| `is_flat(symbol)` | Check if no position |
| `is_long(symbol)` | Check if long |
| `is_short(symbol)` | Check if short |
| `get_position(symbol)` | Get position details |
| `create_signal(...)` | Create a trading signal |

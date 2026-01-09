# Extensible Protocol Design (Issue #24)

This guide covers the extensible protocol patterns in LIBRA for building flexible strategies and gateways.

## Overview

LIBRA's extensible protocol design provides:

- **Strategy Modes**: EVENT_DRIVEN, SCHEDULED, VECTORIZED, HYBRID
- **Gateway Capabilities**: Feature detection inspired by CCXT's `exchange.has` pattern
- **Order Extensions**: OCO, OTO, trailing stops, contingency orders
- **Immutable Extension Methods**: `order.with_*()` pattern for frozen structs

## Strategy Types

### StrategyType Enum

```python
from libra.strategies.protocol import StrategyType

class StrategyType(str, Enum):
    EVENT_DRIVEN = "event_driven"  # React to on_bar, on_tick
    SCHEDULED = "scheduled"        # Time-based execution
    VECTORIZED = "vectorized"      # Batch processing (backtest)
    HYBRID = "hybrid"              # Combination
```

### Event-Driven Strategies

The default mode. Strategies react to market events:

```python
from libra.strategies.base import BaseStrategy
from libra.strategies.protocol import Bar, Signal, StrategyType

class MyEventStrategy(BaseStrategy):
    strategy_type = StrategyType.EVENT_DRIVEN

    def on_bar(self, bar: Bar) -> Signal | None:
        """Called on every new bar."""
        if bar.close > self.sma(self._closes, 20):
            return self._long()
        return None

    def on_tick(self, tick: Tick) -> Signal | None:
        """Called on every tick (optional)."""
        pass
```

### Scheduled Strategies

Execute on time-based schedules (like Zipline's `schedule_function`):

```python
from libra.strategies.protocol import (
    DateRule,
    ScheduledTask,
    StrategyType,
    TimeRule,
)

class RebalanceStrategy(BaseStrategy):
    strategy_type = StrategyType.SCHEDULED

    def __init__(self, config):
        super().__init__(config)
        self._scheduled_tasks = [
            ScheduledTask(
                func_name="rebalance",
                date_rule=DateRule.WEEK_START,  # Every Monday
                time_rule=TimeRule.MARKET_OPEN,
                offset_minutes=30,  # 30 min after open
            ),
        ]

    def rebalance(self) -> list[Signal]:
        """Called on schedule."""
        return [self._long(symbol) for symbol in self.underweight_symbols]
```

### DateRule Options

| Rule | Description |
|------|-------------|
| `EVERY_DAY` | Run every trading day |
| `WEEK_START` | Run on Monday |
| `WEEK_END` | Run on Friday |
| `MONTH_START` | Run on first trading day of month |
| `MONTH_END` | Run on last trading day of month |

### TimeRule Options

| Rule | Description |
|------|-------------|
| `MARKET_OPEN` | At market open |
| `MARKET_CLOSE` | At market close |
| `EVERY_MINUTE` | Every minute |
| `EVERY_HOUR` | Every hour |

## Gateway Capabilities

Inspired by CCXT's `exchange.has` dict for feature detection.

### GatewayCapabilities

```python
from libra.gateways import GatewayCapabilities

@dataclass(frozen=True)
class GatewayCapabilities:
    # Order Types
    market_orders: bool = True
    limit_orders: bool = True
    stop_orders: bool = False
    stop_limit_orders: bool = False
    trailing_stop_orders: bool = False
    bracket_orders: bool = False
    oco_orders: bool = False
    oto_orders: bool = False

    # Time-in-Force
    gtc: bool = True
    ioc: bool = False
    fok: bool = False
    gtd: bool = False

    # Market Data
    real_time_data: bool = True
    historical_data: bool = True
    orderbook_data: bool = True

    # Trading Features
    margin_trading: bool = False
    futures_trading: bool = False
    options_trading: bool = False

    # Rate Limits
    rate_limit_orders_per_second: int | None = None
    rate_limit_requests_per_minute: int | None = None
```

### Usage

```python
from libra.gateways import CCXTGateway, PaperGateway

# Check capabilities before using features
gateway = CCXTGateway("binance", config)
await gateway.connect()

if gateway.capabilities.trailing_stop_orders:
    order = Order(...).with_trailing_stop(
        offset=Decimal("100"),
        offset_type="absolute"
    )
    await gateway.submit_order(order)
else:
    # Fallback to regular stop order
    pass

# Check supported order types
print(gateway.capabilities.supported_order_types)
# frozenset({OrderType.MARKET, OrderType.LIMIT, ...})

# Check specific order type
if gateway.capabilities.supports_order_type(OrderType.STOP_LIMIT):
    # Use stop-limit orders
    pass
```

### Default Capabilities

```python
# Paper gateway - full simulation support
PAPER_GATEWAY_CAPABILITIES = GatewayCapabilities(
    market_orders=True,
    limit_orders=True,
    stop_orders=True,
    trailing_stop_orders=True,
    oco_orders=True,
    margin_trading=True,
)

# CCXT default - common exchange features
CCXT_DEFAULT_CAPABILITIES = GatewayCapabilities(
    market_orders=True,
    limit_orders=True,
    stop_orders=True,
    stop_limit_orders=True,
    gtc=True,
    ioc=True,
    fok=True,
    real_time_data=True,
    historical_data=True,
)
```

## Order Extensions

### Contingency Orders

Link orders together for OCO (One-Cancels-Other) and OTO (One-Triggers-Other):

```python
from libra.gateways import Order, OrderSide, OrderType, ContingencyType

# OCO Order - Cancel other if one fills
take_profit = Order(
    symbol="BTC/USDT",
    side=OrderSide.SELL,
    order_type=OrderType.LIMIT,
    amount=Decimal("1.0"),
    price=Decimal("55000"),
)

stop_loss = Order(
    symbol="BTC/USDT",
    side=OrderSide.SELL,
    order_type=OrderType.STOP,
    amount=Decimal("1.0"),
    stop_price=Decimal("45000"),
)

# Link as OCO
oco_take_profit = take_profit.with_contingency(
    contingency_type=ContingencyType.OCO,
    linked_order_ids=["stop_loss_order_id"],
)

# OTO Order - Trigger other when this fills
entry = Order(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    amount=Decimal("1.0"),
    price=Decimal("50000"),
)

oto_entry = entry.with_contingency(
    contingency_type=ContingencyType.OTO,
    linked_order_ids=["take_profit_id", "stop_loss_id"],
)
```

### ContingencyType Options

| Type | Description |
|------|-------------|
| `NONE` | No contingency (default) |
| `OCO` | One-Cancels-Other |
| `OTO` | One-Triggers-Other |
| `OUO` | One-Updates-Other |

### Trailing Stops

```python
from libra.gateways import Order, TriggerType

# Absolute trailing stop ($100 from peak)
order = Order(
    symbol="BTC/USDT",
    side=OrderSide.SELL,
    order_type=OrderType.STOP,
    amount=Decimal("1.0"),
).with_trailing_stop(
    offset=Decimal("100"),
    offset_type="absolute",
    trigger=TriggerType.LAST_PRICE,
)

# Percentage trailing stop (2% from peak)
order = Order(
    symbol="BTC/USDT",
    side=OrderSide.SELL,
    order_type=OrderType.STOP,
    amount=Decimal("1.0"),
).with_trailing_stop(
    offset=Decimal("0.02"),
    offset_type="percentage",
    trigger=TriggerType.MID_POINT,
)
```

### TriggerType Options

| Type | Description |
|------|-------------|
| `LAST_PRICE` | Last traded price |
| `BID_ASK` | Bid or ask price |
| `MID_POINT` | Mid-point of bid/ask |
| `MARK_PRICE` | Mark price (futures) |

### Checking Order Properties

```python
# Check if order is contingent
if order.is_contingent:
    print(f"Contingency: {order.contingency_type}")
    print(f"Linked orders: {order.linked_order_ids}")

# Check if order is trailing
if order.is_trailing_stop:
    print(f"Trailing offset: {order.trailing_offset}")
    print(f"Offset type: {order.trailing_offset_type}")
    print(f"Trigger: {order.trigger_type}")
```

## Extension Methods (Immutable Pattern)

The `Order` class is immutable (frozen msgspec.Struct). Use `with_*()` methods to create modified copies:

```python
# Base order
order = Order(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    amount=Decimal("1.0"),
)

# Add client order ID
order = order.with_client_id("my-order-001")

# Add timestamp
order = order.with_timestamp()

# Add contingency
order = order.with_contingency(ContingencyType.OCO, ["other-id"])

# Add trailing stop
order = order.with_trailing_stop(Decimal("100"), "absolute")

# Chain methods
order = (
    Order(symbol="BTC/USDT", side=OrderSide.BUY, ...)
    .with_client_id("my-order")
    .with_timestamp()
    .with_contingency(ContingencyType.OTO, ["sl-id", "tp-id"])
)
```

## Example: Complete Trading Flow

```python
from libra.gateways import (
    CCXTGateway,
    Order,
    OrderSide,
    OrderType,
    ContingencyType,
)

async def trade_with_risk_management():
    gateway = CCXTGateway("binance", config)
    await gateway.connect()

    # Check capabilities
    caps = gateway.capabilities

    # Entry order
    entry = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("0.1"),
        price=Decimal("50000"),
    ).with_client_id("entry-001")

    entry_result = await gateway.submit_order(entry)

    if entry_result.is_filled:
        # Set up exit orders
        if caps.oco_orders:
            # Use native OCO
            take_profit = Order(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.1"),
                price=Decimal("55000"),
            )

            stop_loss = Order(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.STOP,
                amount=Decimal("0.1"),
                stop_price=Decimal("48000"),
            )

            # Link as OCO
            tp = take_profit.with_contingency(
                ContingencyType.OCO,
                linked_order_ids=[stop_loss.client_order_id],
            )
            sl = stop_loss.with_contingency(
                ContingencyType.OCO,
                linked_order_ids=[take_profit.client_order_id],
            )

            await gateway.submit_order(tp)
            await gateway.submit_order(sl)
        else:
            # Manual OCO management
            pass
```

## Best Practices

### 1. Check Capabilities Before Using Features

```python
if gateway.capabilities.trailing_stop_orders:
    order = order.with_trailing_stop(...)
else:
    # Implement manual trailing stop logic
    pass
```

### 2. Use Strategy Type Declarations

```python
class MyStrategy(BaseStrategy):
    strategy_type = StrategyType.SCHEDULED  # Explicit type
```

### 3. Register Scheduled Tasks in __init__

```python
def __init__(self, config):
    super().__init__(config)
    self._scheduled_tasks = [
        ScheduledTask(func_name="daily_check", ...),
        ScheduledTask(func_name="weekly_rebalance", ...),
    ]
```

### 4. Use Immutable Order Extensions

```python
# Good - creates new order
order = base_order.with_trailing_stop(...)

# Order is immutable - original unchanged
assert base_order.trailing_offset is None
```

## See Also

- [Strategy Development Guide](strategy_development.md)
- [Plugin Development Guide](plugin_development.md)
- [NautilusTrader Contingency Orders](https://nautilustrader.io)
- [Zipline schedule_function](https://www.zipline.io/appendix.html#zipline.api.schedule_function)

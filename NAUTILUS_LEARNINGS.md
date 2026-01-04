# Nautilus Trader Learnings & Integration Opportunities for LIBRA

> Deep research analysis comparing [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) (17K+ stars) with LIBRA to identify patterns, learnings, and architectural improvements.

## Executive Summary

NautilusTrader is a mature, production-grade algorithmic trading platform with a Rust core and Python bindings. It shares LIBRA's event-driven architecture philosophy but is significantly more mature. Key differentiator: **identical code runs in backtest and live trading**.

| Aspect | NautilusTrader | LIBRA |
|--------|----------------|-------|
| **Maturity** | 6+ years, 17K stars, production use | Phase 1 complete |
| **Core Language** | Rust + Cython + Python | Python (Rust planned) |
| **Performance** | Nanosecond precision, Rust hot paths | ~2.5M events/sec Python |
| **Message Bus** | Topic-based pub/sub + request/response | Priority-based queues |
| **Strategy Model** | Actor-based with lifecycle hooks | Not yet implemented |
| **Backtest** | Full event-driven engine | Planned Phase 2 |
| **Live Trading** | 16+ venue integrations | CCXT + Paper (partial) |

---

## Key Learnings from NautilusTrader

### 1. Actor Model for Strategies (HIGH VALUE)

NautilusTrader uses an **Actor → Strategy inheritance model**:

```
Actor (base)
  ├── receives data, handles events, manages state
  ├── on_start(), on_stop(), on_reset(), on_dispose()
  └── access to: Cache, Portfolio, Clock, MessageBus, Logger

Strategy (extends Actor)
  ├── adds order management capabilities
  ├── on_bar(), on_quote_tick(), on_trade_tick()
  ├── on_order_filled(), on_position_opened()
  └── submit_order(), cancel_order(), close_position()
```

**Recommendation for LIBRA**: Adopt this Actor/Strategy model.

```python
# Proposed: src/libra/strategies/actor.py
class Actor(Protocol):
    """Base actor with event handling and state access."""

    # Lifecycle
    async def on_start(self) -> None: ...
    async def on_stop(self) -> None: ...
    async def on_reset(self) -> None: ...

    # Event handlers
    async def on_event(self, event: Event) -> None: ...
    async def on_bar(self, bar: BarEvent) -> None: ...
    async def on_tick(self, tick: TickEvent) -> None: ...

    # State access
    @property
    def cache(self) -> Cache: ...
    @property
    def portfolio(self) -> Portfolio: ...
    @property
    def clock(self) -> Clock: ...
    @property
    def bus(self) -> MessageBus: ...

# Proposed: src/libra/strategies/strategy.py
class Strategy(Actor):
    """Trading strategy with order management."""

    # Order management
    async def submit_order(self, order: Order) -> OrderResult: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def close_position(self, symbol: str) -> None: ...

    # Position events
    async def on_order_filled(self, event: OrderFilledEvent) -> None: ...
    async def on_position_opened(self, event: PositionOpenedEvent) -> None: ...
```

### 2. Unified Backtest/Live Architecture (HIGH VALUE)

NautilusTrader's core principle: **same strategy code runs in backtest and live**.

```
┌─────────────────────────────────────────────────────────────┐
│                      Strategy Code                          │
│              (identical in all environments)                │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Backtest │        │  Sandbox │        │   Live   │
    │  Engine  │        │  Engine  │        │  Engine  │
    └──────────┘        └──────────┘        └──────────┘
```

**Implementation pattern:**
- Abstract `DataClient` and `ExecutionClient` interfaces
- Backtest engine provides simulated implementations
- Live engine provides venue-specific implementations
- Strategy only interacts with abstract interfaces

**Recommendation for LIBRA**: Design gateway protocol to support this.

```python
# Gateway protocol already supports this pattern!
# Key: ensure Paper gateway behavior matches live gateways exactly
```

### 3. DataClient / ExecutionClient Separation (HIGH VALUE)

NautilusTrader cleanly separates:

| Component | Responsibility |
|-----------|----------------|
| **DataClient** | Market data subscriptions, historical requests, normalization |
| **ExecutionClient** | Order submission, fills, account updates, reconciliation |

**Recommendation for LIBRA**: Split current Gateway protocol.

```python
# Proposed split
class DataClient(Protocol):
    """Market data operations."""
    async def subscribe_ticks(self, symbol: str) -> None: ...
    async def subscribe_bars(self, symbol: str, interval: str) -> None: ...
    async def request_historical(self, symbol: str, start: datetime, end: datetime) -> list[Bar]: ...
    async def get_orderbook(self, symbol: str) -> OrderBook: ...

class ExecutionClient(Protocol):
    """Order execution operations."""
    async def submit_order(self, order: Order) -> OrderResult: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_positions(self) -> list[Position]: ...
    async def get_balances(self) -> list[Balance]: ...
```

### 4. NautilusKernel: Central Orchestrator (MEDIUM VALUE)

NautilusTrader has a `NautilusKernel` that manages:
- Component initialization and lifecycle
- Message bus setup
- Cache initialization
- Engine coordination

**Recommendation for LIBRA**: Create a `TradingKernel`.

```python
# Proposed: src/libra/core/kernel.py
@dataclass
class TradingKernel:
    """Central orchestrator for all trading components."""

    bus: MessageBus
    cache: Cache
    data_engine: DataEngine
    execution_engine: ExecutionEngine
    risk_engine: RiskEngine

    async def start(self) -> None:
        """Initialize and start all components in correct order."""
        await self.bus.start()
        await self.cache.initialize()
        await self.data_engine.start()
        await self.execution_engine.start()
        await self.risk_engine.start()

    async def stop(self) -> None:
        """Graceful shutdown in reverse order."""
        await self.risk_engine.stop()
        await self.execution_engine.stop()
        await self.data_engine.stop()
        await self.bus.stop()
```

### 5. Execution Algorithm Framework (MEDIUM VALUE)

NautilusTrader supports pluggable execution algorithms:
- TWAP (Time-Weighted Average Price)
- Custom algorithms via `ExecAlgorithm` base class
- Spawns child orders from parent orders

```python
# NautilusTrader pattern
class TWAPExecAlgorithm(ExecAlgorithm):
    def on_start(self):
        self.set_timer("twap_slice", interval=self.config.slice_interval)

    def on_timer(self, event):
        self.spawn_market(quantity=self.slice_qty)
```

**Recommendation for LIBRA**: Add execution algorithm support.

```python
# Proposed: src/libra/execution/algorithms.py
class ExecAlgorithm(Protocol):
    """Base for execution algorithms."""

    async def execute(self, order: Order) -> list[Order]:
        """Split parent order into child orders."""
        ...

class TWAPAlgorithm(ExecAlgorithm):
    """Time-weighted average price execution."""

    def __init__(self, slices: int, interval_seconds: float):
        self.slices = slices
        self.interval = interval_seconds

    async def execute(self, order: Order) -> list[Order]:
        slice_qty = order.quantity / self.slices
        # Return list of timed child orders
        ...
```

### 6. Topic-Based Message Bus (MEDIUM VALUE)

NautilusTrader uses **topic-based pub/sub** vs LIBRA's **priority queues**:

| LIBRA (Current) | NautilusTrader |
|-----------------|----------------|
| Priority levels (RISK, ORDERS, SIGNALS, MARKET_DATA) | Named topics (strings) |
| O(1) enqueue/dequeue | Topic-based routing |
| Implicit ordering by priority | Explicit subscription |

**Assessment**: Both approaches are valid. LIBRA's priority queue is better for latency-sensitive trading. Keep current approach but add topic support for custom messaging.

```python
# Proposed enhancement: Add topic support alongside priority queues
class MessageBus:
    # Existing priority-based dispatch
    async def publish(self, event: Event) -> bool: ...

    # New: topic-based messaging for custom use
    async def publish_topic(self, topic: str, message: Any) -> None: ...
    def subscribe_topic(self, topic: str, handler: Callable) -> None: ...
```

### 7. Comprehensive Lifecycle Hooks (HIGH VALUE)

NautilusTrader actors have rich lifecycle:

| Hook | Purpose |
|------|---------|
| `on_start()` | Initialize, subscribe to data |
| `on_stop()` | Cancel orders, cleanup |
| `on_resume()` | Resume from stopped state |
| `on_reset()` | Clear state between backtests |
| `on_degrade()` | Handle partial functionality |
| `on_fault()` | Handle critical errors |
| `on_dispose()` | Final cleanup |

**Recommendation for LIBRA**: Implement full lifecycle.

```python
class ComponentLifecycle(Protocol):
    """Standard lifecycle for all components."""

    async def on_start(self) -> None: ...
    async def on_stop(self) -> None: ...
    async def on_resume(self) -> None: ...
    async def on_reset(self) -> None: ...
    async def on_degrade(self, reason: str) -> None: ...
    async def on_fault(self, error: Exception) -> None: ...
    async def on_dispose(self) -> None: ...
```

### 8. Risk Engine Pre-Trade Checks (HIGH VALUE)

NautilusTrader routes ALL orders through RiskEngine:

```
Strategy → OrderEmulator → ExecAlgorithm → RiskEngine → ExecutionEngine → Venue
                                              ↓
                                    OrderDenied (if rejected)
```

**Risk checks include:**
- Position limits
- Order rate throttling
- Notional value limits
- Self-trade prevention

**Recommendation for LIBRA**: Implement Risk Engine as mandatory order pipeline step.

```python
# Proposed: src/libra/risk/engine.py
class RiskEngine:
    """Pre-trade risk validation."""

    def __init__(self, config: RiskConfig):
        self.max_position_size: dict[str, Decimal] = config.max_positions
        self.max_order_rate: int = config.max_orders_per_second
        self.max_notional: Decimal = config.max_notional_per_order

    async def validate(self, order: Order) -> RiskResult:
        """Validate order against risk limits."""
        checks = [
            self._check_position_limit(order),
            self._check_order_rate(order),
            self._check_notional(order),
            self._check_self_trade(order),
        ]
        for check in checks:
            if not check.passed:
                return RiskResult(approved=False, reason=check.reason)
        return RiskResult(approved=True)
```

### 9. Data Wrangler Pattern (MEDIUM VALUE)

NautilusTrader separates data loading from transformation:

```
Raw Data (CSV/JSON/DBN)
         ↓
    DataLoader → DataFrame (standardized schema)
         ↓
    DataWrangler → Nautilus Objects (QuoteTick, Bar, etc.)
```

**Recommendation for LIBRA**: Adopt for backtest data ingestion.

```python
# Proposed: src/libra/data/wrangler.py
class BarDataWrangler:
    """Transform DataFrame to Bar events."""

    def __init__(self, instrument: Instrument):
        self.instrument = instrument

    def process(self, df: pl.DataFrame) -> list[BarEvent]:
        """Convert DataFrame rows to BarEvent objects."""
        return [
            BarEvent(
                symbol=self.instrument.symbol,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                timestamp_ns=row["timestamp"],
            )
            for row in df.iter_rows(named=True)
        ]
```

### 10. Precision Modes (LOW VALUE)

NautilusTrader supports dual precision:
- **High-precision**: 128-bit integers (16 decimal places)
- **Standard**: 64-bit integers (9 decimal places)

**Assessment**: Useful for crypto with many decimal places. Consider for Phase 2.

---

## Architecture Comparison

### Event Flow

```
NautilusTrader:
┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│  Venue   │───▶│ DataClient  │───▶│  Cache   │───▶│ Strategy │
│ Adapter  │    │             │    │          │    │ (Actor)  │
└──────────┘    └─────────────┘    └──────────┘    └──────────┘
                                                         │
                                                         ▼
┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│  Venue   │◀───│  Execution  │◀───│   Risk   │◀───│  Order   │
│          │    │   Client    │    │  Engine  │    │          │
└──────────┘    └─────────────┘    └──────────┘    └──────────┘

LIBRA (Current):
┌──────────┐    ┌─────────────┐    ┌──────────┐
│ Gateway  │───▶│  Message    │───▶│ Handler  │
│          │    │    Bus      │    │          │
└──────────┘    │ (Priority)  │    └──────────┘
                └─────────────┘
```

### Recommended LIBRA Evolution

```
LIBRA (Proposed):
┌──────────────────────────────────────────────────────────────────┐
│                        TradingKernel                             │
│  ┌────────────┐  ┌───────────┐  ┌────────────┐  ┌─────────────┐ │
│  │ DataEngine │  │   Cache   │  │ ExecEngine │  │ RiskEngine  │ │
│  └────────────┘  └───────────┘  └────────────┘  └─────────────┘ │
│         │              │               │               │         │
│         └──────────────┴───────────────┴───────────────┘         │
│                              │                                    │
│                    ┌─────────────────┐                           │
│                    │   MessageBus    │                           │
│                    │ (Priority + Topics)                         │
│                    └─────────────────┘                           │
│                              │                                    │
│              ┌───────────────┼───────────────┐                   │
│              ▼               ▼               ▼                   │
│        ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│        │ Strategy │    │ Strategy │    │  Actor   │             │
│        │    1     │    │    2     │    │ (custom) │             │
│        └──────────┘    └──────────┘    └──────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

---

## What We Can Reuse / Learn

### Direct Adoption (High Priority)

| Pattern | Benefit | Effort |
|---------|---------|--------|
| Actor/Strategy model | Clean separation, testable | Medium |
| Lifecycle hooks | Proper state management | Low |
| DataClient/ExecutionClient split | Clear responsibilities | Medium |
| Risk engine pipeline | Safe order flow | Medium |

### Learn From (Medium Priority)

| Pattern | Benefit | Effort |
|---------|---------|--------|
| TradingKernel orchestrator | Clean initialization | Medium |
| Execution algorithms (TWAP) | Better fills | Medium |
| Data wrangler pattern | Clean backtest data | Low |
| Topic-based messaging | Custom pub/sub | Low |

### Consider Later (Low Priority)

| Pattern | Benefit | Effort |
|---------|---------|--------|
| Precision modes | Crypto support | Low |
| Order book L3 support | Market making | High |
| Redis persistence | Fault tolerance | Medium |

---

## Recommended GitHub Issues

### High Priority

1. **[Feature] Actor/Strategy Base Classes with Lifecycle Hooks**
   - Implement Actor base class with full lifecycle
   - Extend to Strategy with order management
   - Enable consistent component behavior

2. **[Feature] Split Gateway into DataClient/ExecutionClient**
   - Separate data and execution concerns
   - Enable independent scaling
   - Support unified backtest/live architecture

3. **[Feature] Risk Engine with Pre-Trade Validation**
   - Mandatory order validation pipeline
   - Position limits, rate limiting, notional checks
   - OrderDenied events for rejected orders

4. **[Feature] TradingKernel Orchestrator**
   - Central component initialization
   - Proper startup/shutdown ordering
   - Unified configuration

### Medium Priority

5. **[Feature] Execution Algorithm Framework**
   - TWAP, VWAP base implementations
   - Parent/child order spawning
   - Algorithm selection per order

6. **[Feature] Data Wrangler for Backtest Ingestion**
   - CSV/Parquet to event conversion
   - Standardized DataFrame schemas
   - Efficient batch processing

7. **[Enhancement] Add Topic-Based Messaging to MessageBus**
   - Custom pub/sub alongside priority queues
   - String-based topic routing
   - Flexible inter-component messaging

---

## Implementation Roadmap

### Phase 2 Enhancement (from Nautilus learnings)

1. **Refactor Gateway** → Split into DataClient + ExecutionClient
2. **Add Actor/Strategy** → Base classes with lifecycle hooks
3. **Add TradingKernel** → Central orchestrator
4. **Add Risk Engine** → Pre-trade validation pipeline
5. **Backtest Engine** → Event-driven with identical strategy code

### Phase 3 Enhancement

1. **Execution Algorithms** → TWAP, VWAP, Iceberg
2. **Data Wranglers** → Clean backtest data ingestion
3. **Topic Messaging** → Custom pub/sub

---

## Summary

| Learning | Priority | Complexity | Impact |
|----------|----------|------------|--------|
| Actor/Strategy model | HIGH | Medium | Foundation for all strategies |
| DataClient/ExecutionClient split | HIGH | Medium | Clean architecture |
| Risk Engine pipeline | HIGH | Medium | Safe trading |
| TradingKernel orchestrator | HIGH | Low | Clean initialization |
| Lifecycle hooks | HIGH | Low | Proper state management |
| Execution algorithms | MEDIUM | Medium | Better fills |
| Data wrangler | MEDIUM | Low | Clean backtest data |
| Topic messaging | LOW | Low | Flexibility |

NautilusTrader validates LIBRA's core architecture choices (event-driven, priority-based) while providing mature patterns for the strategy layer, risk management, and backtest engine. The Actor/Strategy model and DataClient/ExecutionClient separation are the highest-value adoptions.

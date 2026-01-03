# First Issue: Phase 1 Foundation

## Overview

Build the core infrastructure for LIBRA (Aquarius) trading platform using a **hybrid Rust/Python architecture**.

**Duration**: 4 weeks
**Priority**: P0 (Critical Path)
**Approach**: Python MVP first, then Rust hot paths

---

## Phase 1A: Python MVP (Weeks 1-2)

### Goal
Get a working end-to-end system in pure Python to validate architecture.

---

### Task 1.1: Project Setup

**Files to create:**
```
libra/
├── pyproject.toml
├── src/libra/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── events.py
│   │   └── message_bus.py
│   ├── gateways/
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   ├── ccxt_gateway.py
│   │   └── paper_gateway.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   └── examples/
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── limits.py
│   │   ├── manager.py
│   │   └── position_sizing.py
│   └── tui/
│       ├── __init__.py
│       ├── app.py
│       └── widgets/
└── tests/
```

**pyproject.toml:**
```toml
[project]
name = "libra"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    "pydantic>=2.0",
    "aiohttp>=3.9",
    "ccxt>=4.0",
    "pandas>=2.0",
    "numpy>=1.26",
    "textual>=0.50",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-benchmark>=4.0",
    "ruff>=0.1",
    "mypy>=1.8",
]
```

**Acceptance Criteria:**
- [ ] `uv sync` installs all dependencies
- [ ] `pytest` runs (even with no tests)
- [ ] `python -c "import libra"` works

**Effort:** 2 hours

---

### Task 1.2: Event System

**File:** `src/libra/core/events.py`

**Best Practices Applied:**
- Frozen dataclass for immutability (thread-safe)
- Slots for memory efficiency (<256 bytes per event target)
- W3C Trace Context compatible trace_id/span_id for distributed tracing
- Priority field for LMAX Disruptor-style routing

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import Any
from uuid import uuid4

class EventType(Enum):
    # Market Data
    TICK = auto()
    BAR = auto()
    ORDER_BOOK = auto()

    # Orders
    ORDER_NEW = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()

    # Positions
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()

    # Risk (highest priority)
    RISK_LIMIT_BREACH = auto()
    DRAWDOWN_WARNING = auto()
    CIRCUIT_BREAKER = auto()

    # Signals
    SIGNAL = auto()

    # System
    GATEWAY_CONNECTED = auto()
    GATEWAY_DISCONNECTED = auto()

class Priority(IntEnum):
    """Priority levels (lower = higher priority, processed first)."""
    RISK = 0
    ORDERS = 1
    SIGNALS = 2
    MARKET_DATA = 3

def _generate_trace_id() -> str:
    """Generate W3C Trace Context compatible trace ID (32 hex chars)."""
    return format(uuid4().int >> 64, '032x')

def _generate_span_id() -> str:
    """Generate W3C Trace Context compatible span ID (16 hex chars)."""
    return format(uuid4().int >> 96, '016x')

@dataclass(frozen=True, slots=True)
class Event:
    """
    Immutable event for message bus.

    Memory target: <256 bytes per event
    - type: 8 bytes (enum reference)
    - timestamp: 8 bytes
    - source: ~50 bytes typical
    - payload: varies (dict reference)
    - trace_id: 32 bytes
    - span_id: 16 bytes
    - priority: 4 bytes
    """
    event_type: EventType
    timestamp: datetime
    source: str
    payload: dict[str, Any]
    trace_id: str = field(default_factory=_generate_trace_id)
    span_id: str = field(default_factory=_generate_span_id)
    priority: Priority = Priority.MARKET_DATA

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source: str,
        payload: dict[str, Any],
        parent_trace_id: str | None = None,
    ) -> "Event":
        """
        Factory method with automatic priority assignment.

        Args:
            event_type: The type of event
            source: Component that created the event
            payload: Event-specific data
            parent_trace_id: For correlation with parent events
        """
        # Auto-assign priority based on event type
        priority_map = {
            EventType.RISK_LIMIT_BREACH: Priority.RISK,
            EventType.DRAWDOWN_WARNING: Priority.RISK,
            EventType.CIRCUIT_BREAKER: Priority.RISK,
            EventType.ORDER_NEW: Priority.ORDERS,
            EventType.ORDER_FILLED: Priority.ORDERS,
            EventType.ORDER_CANCELLED: Priority.ORDERS,
            EventType.ORDER_REJECTED: Priority.ORDERS,
            EventType.SIGNAL: Priority.SIGNALS,
        }
        priority = priority_map.get(event_type, Priority.MARKET_DATA)

        return cls(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            source=source,
            payload=payload,
            trace_id=parent_trace_id or _generate_trace_id(),
            span_id=_generate_span_id(),
            priority=priority,
        )

    def child_event(
        self,
        event_type: EventType,
        source: str,
        payload: dict[str, Any],
    ) -> "Event":
        """Create a child event with same trace_id for correlation."""
        return Event.create(
            event_type=event_type,
            source=source,
            payload=payload,
            parent_trace_id=self.trace_id,
        )
```

**Acceptance Criteria:**
- [ ] All event types defined (Market Data, Orders, Positions, Risk, Signals, System)
- [ ] Events are immutable (frozen dataclass)
- [ ] Events use slots for memory efficiency
- [ ] W3C Trace Context compatible trace_id/span_id
- [ ] Priority auto-assigned based on event type
- [ ] Child event creation preserves trace_id
- [ ] Unit tests pass with 100% coverage

**Effort:** 3 hours (increased for tracing fields)

---

### Task 1.3: Message Bus (Python MVP)

**File:** `src/libra/core/message_bus.py`

**Best Practices Applied (from LMAX Disruptor & NautilusTrader research):**
- Priority-based event routing (Risk > Orders > Signals > Market Data)
- Correlation IDs for distributed tracing (W3C Trace Context)
- Graceful shutdown with queue draining
- Handler error isolation

```python
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any
from enum import IntEnum
from uuid import uuid4
import asyncio
import logging

from .events import Event, EventType

logger = logging.getLogger(__name__)

class Priority(IntEnum):
    """Event priority levels (lower = higher priority)."""
    RISK = 0        # Circuit breakers, limit breaches
    ORDERS = 1      # Order lifecycle events
    SIGNALS = 2     # Trading signals
    MARKET_DATA = 3  # Ticks, bars, order book

# Map event types to priorities
EVENT_PRIORITIES = {
    EventType.RISK_LIMIT_BREACH: Priority.RISK,
    EventType.DRAWDOWN_WARNING: Priority.RISK,
    EventType.CIRCUIT_BREAKER: Priority.RISK,
    EventType.ORDER_NEW: Priority.ORDERS,
    EventType.ORDER_FILLED: Priority.ORDERS,
    EventType.ORDER_CANCELLED: Priority.ORDERS,
    EventType.ORDER_REJECTED: Priority.ORDERS,
    # Default: MARKET_DATA
}

@dataclass(order=True)
class PrioritizedEvent:
    """Wrapper for priority queue ordering."""
    priority: int
    sequence: int  # For FIFO within same priority
    event: Event = field(compare=False)

Handler = Callable[[Event], Coroutine[Any, Any, None]]

class MessageBus:
    """
    High-performance async message bus with priority routing.

    Inspired by:
    - LMAX Disruptor: Priority processing, mechanical sympathy
    - NautilusTrader: Pub/sub pattern, loose coupling

    Phase 1A targets:
    - Publish latency: <10μs
    - Dispatch latency: <100μs
    - Throughput: 100K-500K events/sec
    """

    def __init__(self, max_handlers_per_event: int = 100):
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._queue: asyncio.PriorityQueue[PrioritizedEvent] = asyncio.PriorityQueue()
        self._running = False
        self._accepting_new = True
        self._sequence = 0
        self._max_handlers = max_handlers_per_event
        self._running_handlers: set[asyncio.Task] = set()

    def subscribe(
        self,
        event_type: EventType,
        handler: Handler,
        filter_fn: Callable[[Event], bool] | None = None
    ) -> str:
        """Subscribe to an event type with optional filtering."""
        if len(self._handlers[event_type]) >= self._max_handlers:
            raise RuntimeError(f"Max handlers ({self._max_handlers}) reached for {event_type}")

        # Wrap handler with filter if provided
        if filter_fn:
            async def filtered_handler(event: Event):
                if filter_fn(event):
                    await handler(event)
            self._handlers[event_type].append(filtered_handler)
        else:
            self._handlers[event_type].append(handler)

        subscription_id = str(uuid4())
        return subscription_id

    def unsubscribe(self, event_type: EventType, handler: Handler) -> bool:
        """Unsubscribe from an event type. Returns True if found."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            return True
        return False

    async def publish(self, event: Event) -> None:
        """Publish an event with automatic priority assignment."""
        if not self._accepting_new:
            logger.warning(f"Rejecting event during shutdown: {event.type}")
            return

        priority = EVENT_PRIORITIES.get(event.type, Priority.MARKET_DATA)
        self._sequence += 1

        prioritized = PrioritizedEvent(
            priority=priority,
            sequence=self._sequence,
            event=event
        )
        await self._queue.put(prioritized)

    async def start(self) -> None:
        """Start the message bus dispatch loop."""
        self._running = True
        self._accepting_new = True
        logger.info("MessageBus started")

        while self._running:
            try:
                prioritized = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1
                )
                await self._dispatch(prioritized.event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("MessageBus received cancellation")
                break

    async def stop(self, drain_timeout: float = 5.0) -> int:
        """
        Graceful shutdown with queue draining.

        Args:
            drain_timeout: Max seconds to wait for queue drain

        Returns:
            Number of events dropped (if any)
        """
        self._accepting_new = False
        self._running = False
        dropped = 0

        # Drain remaining events with timeout
        try:
            async with asyncio.timeout(drain_timeout):
                while not self._queue.empty():
                    prioritized = await self._queue.get()
                    await self._dispatch(prioritized.event)
        except asyncio.TimeoutError:
            dropped = self._queue.qsize()
            logger.warning(f"Shutdown timeout, {dropped} events dropped")

        # Cancel any still-running handlers
        for task in list(self._running_handlers):
            if not task.done():
                task.cancel()

        if self._running_handlers:
            await asyncio.gather(
                *self._running_handlers,
                return_exceptions=True
            )

        logger.info(f"MessageBus stopped, dropped {dropped} events")
        return dropped

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all registered handlers."""
        handlers = self._handlers.get(event.type, [])
        if not handlers:
            return

        # Create tasks for parallel execution
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(
                self._safe_handle(handler, event)
            )
            self._running_handlers.add(task)
            task.add_done_callback(self._running_handlers.discard)
            tasks.append(task)

        # Wait for all handlers (with error isolation)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, handler: Handler, event: Event) -> None:
        """Execute handler with error isolation."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                f"Handler error for {event.type}: {e}",
                exc_info=True,
                extra={
                    "trace_id": getattr(event, 'trace_id', None),
                    "event_type": event.type.name,
                }
            )

    @property
    def queue_size(self) -> int:
        """Current number of pending events."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Whether the bus is actively processing."""
        return self._running
```

**Acceptance Criteria:**
- [ ] Events dispatched in priority order (Risk first)
- [ ] FIFO ordering within same priority level
- [ ] Handler errors don't crash bus (isolated via try/except)
- [ ] Graceful shutdown drains queue with timeout
- [ ] Correlation IDs propagated for tracing
- [ ] Benchmark: measure baseline latency (<100μs dispatch target)

**Effort:** 8 hours (increased for priority + tracing features)

---

### Task 1.4: Gateway Protocol

**File:** `src/libra/gateways/protocol.py`

```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import AsyncIterator, Protocol, runtime_checkable

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    id: str | None
    symbol: str
    side: OrderSide
    type: OrderType
    amount: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None

@dataclass
class Position:
    symbol: str
    side: OrderSide
    amount: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal

@dataclass
class Tick:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    timestamp: datetime

@runtime_checkable
class Gateway(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def is_connected(self) -> bool: ...

    async def connect(self, config: dict) -> None: ...
    async def disconnect(self) -> None: ...
    async def subscribe(self, symbols: list[str]) -> None: ...
    def stream_ticks(self) -> AsyncIterator[Tick]: ...
    async def order(self, order: Order) -> dict: ...
    async def cancel(self, order_id: str) -> bool: ...
    async def positions(self) -> list[Position]: ...
    async def balance(self) -> dict[str, Decimal]: ...
```

**Acceptance Criteria:**
- [ ] Protocol is runtime_checkable
- [ ] All types are well-defined
- [ ] Async throughout

**Effort:** 3 hours

---

### Task 1.5: CCXT Gateway

**File:** `src/libra/gateways/ccxt_gateway.py`

Implement the Gateway protocol using CCXT for Binance testnet.

**Acceptance Criteria:**
- [ ] Connects to Binance testnet
- [ ] Streams real-time ticks via WebSocket
- [ ] Submits market and limit orders
- [ ] Handles disconnections gracefully
- [ ] Integration test passes

**Effort:** 12 hours

---

### Task 1.6: Paper Trading Gateway

**File:** `src/libra/gateways/paper_gateway.py`

Simulated gateway for testing without real exchange.

**Acceptance Criteria:**
- [ ] Simulates order fills (market = instant, limit = price match)
- [ ] Tracks positions accurately
- [ ] Calculates P&L correctly
- [ ] Configurable slippage

**Effort:** 6 hours

---

### Task 1.7: Risk Manager

**Files:** `src/libra/risk/`

**Acceptance Criteria:**
- [ ] Pre-trade risk checks work
- [ ] Position size limits enforced
- [ ] Daily loss limits tracked
- [ ] Circuit breaker triggers correctly
- [ ] Rate limiting works

**Effort:** 10 hours

---

### Task 1.8: Strategy Protocol

**File:** `src/libra/strategies/protocol.py`

**Acceptance Criteria:**
- [ ] Signal types defined (LONG, SHORT, CLOSE_*, HOLD)
- [ ] Strategy Protocol defined
- [ ] Example SMA crossover strategy works
- [ ] Same code works for backtest and live

**Effort:** 4 hours

---

### Task 1.9: TUI Shell

**Files:** `src/libra/tui/`

Basic Textual application shell.

**Acceptance Criteria:**
- [ ] App launches without errors
- [ ] Shows connection status
- [ ] Displays account balance
- [ ] Shows log messages
- [ ] Keyboard navigation works

**Effort:** 8 hours

---

## Phase 1A Milestone Checklist

```
[ ] M1A.1: Message bus working (events flow)
[ ] M1A.2: CCXT gateway connects to Binance testnet
[ ] M1A.3: Paper trading works end-to-end
[ ] M1A.4: Risk checks prevent bad orders
[ ] M1A.5: TUI launches and shows data
```

---

## Phase 1B: Rust Hot Paths (Weeks 3-4)

### Goal
Migrate performance-critical components to Rust for <100ms signal-to-order latency.

---

### Task 1.10: Rust Crate Setup

**Files to create:**
```
libra-core/
├── Cargo.toml
├── pyproject.toml      # maturin config
├── src/
│   ├── lib.rs
│   ├── events.rs
│   ├── message_bus.rs
│   ├── risk.rs
│   ├── indicators.rs
│   └── ws_parser.rs
└── python/
    └── libra_core/
        ├── __init__.py
        └── __init__.pyi   # Type stubs
```

**Cargo.toml:**
```toml
[package]
name = "libra-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "libra_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
tokio = { version = "1", features = ["full"] }
crossbeam-channel = "0.5"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
simd-json = "0.14"
rust_decimal = "1"
```

**Acceptance Criteria:**
- [ ] `maturin develop` builds successfully
- [ ] `import libra_core` works in Python
- [ ] Basic smoke test passes

**Effort:** 6 hours

---

### Task 1.11: Rust Message Bus

**File:** `libra-core/src/message_bus.rs`

**Best Practices Applied (from LMAX Disruptor research):**
- Lock-free ring buffer with cache-line padding
- Separate ring buffers per priority level (no heap for priority queue)
- SPSC queues for each priority → single dispatcher thread
- PyO3 integration with `allow_threads` for GIL release

**Recommended Crates:**
- `ringbuf` (9M+ downloads) - Lock-free SPSC ring buffer
- `crossbeam-channel` - Battle-tested MPSC for multi-producer scenarios
- `crossfire` (v2.1, 2025) - Async-capable lockless channels

```rust
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache-line padded sequence counter (prevents false sharing)
/// See: https://trishagee.com/2011/07/22/dissecting_the_disruptor_why_its_so_fast_part_two__magic_cache_line_padding/
#[repr(C)]
pub struct PaddedSequence {
    _pad_left: [u64; 7],    // 56 bytes padding
    pub value: AtomicU64,    // 8 bytes (the critical value)
    _pad_right: [u64; 7],   // 56 bytes padding
}
// Total: 120 bytes, ensuring `value` is never on a shared cache line

impl PaddedSequence {
    pub fn new(initial: u64) -> Self {
        Self {
            _pad_left: [0; 7],
            value: AtomicU64::new(initial),
            _pad_right: [0; 7],
        }
    }

    #[inline]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    #[inline]
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Release);
    }
}

/// Priority levels matching Python implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Priority {
    Risk = 0,
    Orders = 1,
    Signals = 2,
    MarketData = 3,
}

/// Multi-priority message bus using separate ring buffers per priority
/// This avoids heap allocation overhead of a priority queue
pub struct MessageBus {
    // One ring buffer per priority level
    risk_queue: ringbuf::HeapRb<Event>,
    orders_queue: ringbuf::HeapRb<Event>,
    signals_queue: ringbuf::HeapRb<Event>,
    market_data_queue: ringbuf::HeapRb<Event>,

    // Sequence counters for each queue
    risk_sequence: PaddedSequence,
    orders_sequence: PaddedSequence,
    signals_sequence: PaddedSequence,
    market_data_sequence: PaddedSequence,

    running: AtomicBool,
}

impl MessageBus {
    /// Process events in priority order: Risk → Orders → Signals → Market Data
    pub fn dispatch_next(&mut self) -> Option<Event> {
        // Check queues in priority order
        if let Some(event) = self.risk_queue.pop() {
            return Some(event);
        }
        if let Some(event) = self.orders_queue.pop() {
            return Some(event);
        }
        if let Some(event) = self.signals_queue.pop() {
            return Some(event);
        }
        self.market_data_queue.pop()
    }
}
```

**PyO3 Integration Pattern:**
```rust
use pyo3::prelude::*;

#[pyclass]
pub struct RustMessageBus {
    inner: MessageBus,
}

#[pymethods]
impl RustMessageBus {
    #[new]
    fn new(buffer_size: usize) -> Self {
        Self {
            inner: MessageBus::new(buffer_size),
        }
    }

    /// Publish with GIL release for performance
    fn publish(&mut self, py: Python<'_>, event_bytes: &[u8]) -> PyResult<()> {
        // Release GIL while doing Rust work
        py.allow_threads(|| {
            let event: Event = deserialize(event_bytes)?;
            self.inner.publish(event);
            Ok(())
        })
    }
}
```

**Acceptance Criteria:**
- [ ] <1ms dispatch latency (P99) - validated via benchmarks
- [ ] 10x+ faster than Python implementation
- [ ] Cache-line padded sequence counters (no false sharing)
- [ ] Separate ring buffers per priority (no heap allocation in hot path)
- [ ] Drop-in replacement API via PyO3
- [ ] `allow_threads` used for GIL release
- [ ] No GC pauses (Rust has no GC)

**Effort:** 20 hours (increased for cache-line padding + PyO3 integration)

---

### Task 1.12: Rust Risk Engine

**File:** `libra-core/src/risk.rs`

Pre-trade risk checks with zero allocations.

**Acceptance Criteria:**
- [ ] <100μs per risk check
- [ ] Same logic as Python (validated by tests)
- [ ] Zero heap allocations in hot path

**Effort:** 12 hours

---

### Task 1.13: Rust Technical Indicators

**File:** `libra-core/src/indicators.rs`

SIMD-optimized technical indicators.

**Acceptance Criteria:**
- [ ] EMA, SMA, RSI, MACD, Bollinger Bands
- [ ] 50x+ faster than pandas-ta
- [ ] Identical numerical results
- [ ] Works with NumPy arrays directly

**Effort:** 12 hours

---

### Task 1.14: Rust WebSocket Parser

**File:** `libra-core/src/ws_parser.rs`

SIMD-accelerated JSON parsing for market data.

**Acceptance Criteria:**
- [ ] <10μs per message parse
- [ ] Supports Binance/Bybit formats
- [ ] Handles malformed data gracefully

**Effort:** 8 hours

---

## Phase 1B Milestone Checklist

```
[ ] M1B.1: Rust crate builds and imports
[ ] M1B.2: Message bus <1ms dispatch
[ ] M1B.3: Risk checks <100μs
[ ] M1B.4: Indicators 50x faster than pandas-ta
[ ] M1B.5: WS parsing <10μs
```

---

## Performance Validation Gate

Before Phase 2, all metrics must pass:

| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Message dispatch | <1ms P99 | ___ | [ ] |
| Risk check | <100μs | ___ | [ ] |
| Indicator calc | 50x vs pandas | ___ | [ ] |
| WS message parse | <10μs | ___ | [ ] |
| Signal-to-order | <100ms | ___ | [ ] |

---

## Definition of Done

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Benchmarks documented
- [ ] Performance gates met
- [ ] Code reviewed
- [ ] Documentation updated

---

## Dependencies & Blockers

### External Dependencies
- Binance testnet API access
- CCXT library compatibility

### Potential Blockers
- PyO3/maturin build issues on CI
- CCXT WebSocket stability
- Rust async/Python async interop

---

## Notes

### Why Python First?
1. Validate architecture before optimizing
2. Faster iteration on interfaces
3. Profile to find real bottlenecks
4. Reduce risk of wrong abstractions

### Key Rust Crates
- `pyo3` - Python bindings
- `tokio` - Async runtime
- `crossbeam` - Lock-free data structures
- `simd-json` - Fast JSON parsing
- `rust_decimal` - Financial precision

### Reference Implementation
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) - Same Rust+Python pattern

---

## Research References & Best Practices

### LMAX Disruptor Architecture
- [Martin Fowler: The LMAX Architecture](https://martinfowler.com/articles/lmax.html) - Core design principles
- [LMAX Disruptor Official Docs](https://lmax-exchange.github.io/disruptor/) - Implementation details
- [Trisha Gee: Cache Line Padding](https://trishagee.com/2011/07/22/dissecting_the_disruptor_why_its_so_fast_part_two__magic_cache_line_padding/) - Why padding prevents false sharing
- [Baeldung: LMAX Concurrency](https://www.baeldung.com/lmax-disruptor-concurrency) - Java implementation guide

### Rust Crates for High-Performance Queues
- [ringbuf](https://crates.io/crates/ringbuf) - Lock-free SPSC ring buffer (9M+ downloads)
- [crossbeam-channel](https://crates.io/crates/crossbeam-channel) - Battle-tested MPSC/MPMC
- [crossfire](https://crates.io/crates/crossfire) - Async lockless channels (v2.1, Sept 2025)
- [jonhoo/bus](https://github.com/jonhoo/bus) - Efficient lock-free broadcast channel
- [disruptor-rs](https://github.com/khaledyassin/disruptor-rs) - Rust port of LMAX Disruptor

### NautilusTrader Reference
- [NautilusTrader MessageBus](https://nautilustrader.io/docs/nightly/concepts/message_bus/) - Production architecture
- [NautilusTrader Architecture](https://nautilustrader.io/docs/latest/concepts/architecture/) - Hybrid Rust/Python design
- [nautilus-infrastructure crate](https://crates.io/crates/nautilus-infrastructure) - Redis + PyO3 integration

### PyO3 Best Practices
- [PyO3 Performance Guide](https://pyo3.rs/main/performance) - GIL release, vectorcall
- [PyO3 Async Runtimes](https://github.com/PyO3/pyo3-async-runtimes) - Tokio/asyncio interop
- [PyO3 Thread Communication](https://github.com/PyO3/pyo3/discussions/3438) - MPSC patterns

### Distributed Tracing
- [OpenTelemetry Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/) - W3C Trace Context
- [Honeycomb: Tracing with Message Bus](https://www.honeycomb.io/blog/understanding-distributed-tracing-message-bus) - Span links
- [OpenTelemetry Traces](https://opentelemetry.io/docs/concepts/signals/traces/) - Trace/span IDs

### Python Async Best Practices
- [Roguelynn: Asyncio Graceful Shutdowns](https://roguelynn.com/words/asyncio-graceful-shutdowns/) - Signal handling
- [QuantStart: Event-Driven Backtesting](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/) - Priority queue patterns
- [AsyncAlgoTrading/aat](https://github.com/AsyncAlgoTrading/aat) - Production async trading

### Performance Benchmarks & Targets

| Component | Python MVP Target | Rust Target | Industry Benchmark |
|-----------|-------------------|-------------|-------------------|
| Publish latency | <10μs | <100ns | LMAX: <1μs |
| Dispatch latency | <100μs | <1μs | NautilusTrader: <1ms |
| Throughput | 100K-500K/sec | >1M/sec | LMAX: 6M/sec |
| Memory/event | <256 bytes | <128 bytes | Disruptor: varies |

### Key Learnings

1. **Cache Line Padding is Critical**: Without padding, false sharing can kill performance by 10-100x
2. **Separate Queues per Priority**: Faster than heap-based priority queue in hot path
3. **Python GIL is the Bottleneck**: Only achievable via Rust core for <1μs latency
4. **Use SPSC When Possible**: Much faster than MPSC/MPMC
5. **Batch Operations**: `push_slice`/`pop_slice` reduce cache synchronization overhead
6. **Graceful Shutdown**: Don't use `asyncio.run()` - need custom shutdown handler

---

*Created: January 2026*
*Status: Ready for Implementation*
*Research Updated: January 2026*

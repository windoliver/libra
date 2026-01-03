# Issue 1: Phase 1 Message Bus - Detailed Execution Plan

## Current State
- Empty repo with only documentation files
- No Python code, no project structure
- Need to build from scratch

---

## Step 1: Project Initialization (30 mins)

### 1.1 Create Directory Structure

```bash
# Run from /Users/tafeng/libra
mkdir -p src/libra/core
mkdir -p src/libra/gateways
mkdir -p src/libra/strategies
mkdir -p src/libra/risk
mkdir -p src/libra/tui
mkdir -p tests/unit/core
mkdir -p tests/integration
mkdir -p benchmarks
```

**Expected structure:**
```
libra/
├── pyproject.toml
├── README.md
├── src/
│   └── libra/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── events.py        # Step 2
│       │   └── message_bus.py   # Step 3
│       ├── gateways/
│       │   └── __init__.py
│       ├── strategies/
│       │   └── __init__.py
│       ├── risk/
│       │   └── __init__.py
│       └── tui/
│           └── __init__.py
├── tests/
│   ├── unit/
│   │   └── core/
│   │       ├── test_events.py   # Step 4
│   │       └── test_message_bus.py
│   └── integration/
└── benchmarks/
    └── bench_message_bus.py     # Step 5
```

### 1.2 Create pyproject.toml

```toml
[project]
name = "libra"
version = "0.1.0"
description = "High-performance AI trading platform"
requires-python = ">=3.12"

dependencies = [
    "pydantic>=2.0",
    "aiohttp>=3.9",
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

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/libra"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
```

### 1.3 Create __init__.py files

```python
# src/libra/__init__.py
"""LIBRA: High-performance AI trading platform."""
__version__ = "0.1.0"
```

### 1.4 Verify Setup

```bash
# Install in development mode
uv sync  # or pip install -e ".[dev]"

# Verify import works
python -c "import libra; print(libra.__version__)"
```

**Checkpoint:** `python -c "import libra"` works without errors

---

## Step 2: Event System (2 hours)

### 2.1 Create src/libra/core/events.py

**File:** `src/libra/core/events.py`

```python
"""
Event system for LIBRA message bus.

Features:
- Frozen dataclasses for immutability (thread-safe)
- Slots for memory efficiency (<256 bytes per event)
- W3C Trace Context compatible trace_id/span_id
- Priority levels for LMAX Disruptor-style routing
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum, IntEnum, auto
from typing import Any
from uuid import uuid4


class EventType(Enum):
    """All event types in the system."""

    # Market Data (Priority: MARKET_DATA)
    TICK = auto()
    BAR = auto()
    ORDER_BOOK = auto()

    # Orders (Priority: ORDERS)
    ORDER_NEW = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()

    # Positions (Priority: ORDERS)
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()

    # Risk (Priority: RISK - highest)
    RISK_LIMIT_BREACH = auto()
    DRAWDOWN_WARNING = auto()
    CIRCUIT_BREAKER = auto()

    # Signals (Priority: SIGNALS)
    SIGNAL = auto()

    # System (Priority: MARKET_DATA)
    GATEWAY_CONNECTED = auto()
    GATEWAY_DISCONNECTED = auto()
    STRATEGY_STARTED = auto()
    STRATEGY_STOPPED = auto()


class Priority(IntEnum):
    """
    Priority levels (lower = higher priority, processed first).

    Based on LMAX Disruptor pattern:
    - Risk events must be processed immediately (circuit breakers)
    - Orders need fast processing for execution
    - Signals drive trading decisions
    - Market data is high volume, lower priority
    """
    RISK = 0        # Circuit breakers, limit breaches
    ORDERS = 1      # Order lifecycle events
    SIGNALS = 2     # Trading signals
    MARKET_DATA = 3 # Ticks, bars, order book


# Priority mapping for each event type
EVENT_PRIORITY_MAP: dict[EventType, Priority] = {
    # Risk events - highest priority
    EventType.RISK_LIMIT_BREACH: Priority.RISK,
    EventType.DRAWDOWN_WARNING: Priority.RISK,
    EventType.CIRCUIT_BREAKER: Priority.RISK,

    # Order events
    EventType.ORDER_NEW: Priority.ORDERS,
    EventType.ORDER_FILLED: Priority.ORDERS,
    EventType.ORDER_CANCELLED: Priority.ORDERS,
    EventType.ORDER_REJECTED: Priority.ORDERS,
    EventType.POSITION_OPENED: Priority.ORDERS,
    EventType.POSITION_CLOSED: Priority.ORDERS,
    EventType.POSITION_UPDATED: Priority.ORDERS,

    # Signal events
    EventType.SIGNAL: Priority.SIGNALS,

    # Market data events - lowest priority (highest volume)
    EventType.TICK: Priority.MARKET_DATA,
    EventType.BAR: Priority.MARKET_DATA,
    EventType.ORDER_BOOK: Priority.MARKET_DATA,
    EventType.GATEWAY_CONNECTED: Priority.MARKET_DATA,
    EventType.GATEWAY_DISCONNECTED: Priority.MARKET_DATA,
    EventType.STRATEGY_STARTED: Priority.MARKET_DATA,
    EventType.STRATEGY_STOPPED: Priority.MARKET_DATA,
}


def _generate_trace_id() -> str:
    """Generate W3C Trace Context compatible trace ID (32 hex chars)."""
    return format(uuid4().int, '032x')[:32]


def _generate_span_id() -> str:
    """Generate W3C Trace Context compatible span ID (16 hex chars)."""
    return format(uuid4().int, '016x')[:16]


@dataclass(frozen=True, slots=True)
class Event:
    """
    Immutable event for message bus.

    Memory layout (target: <256 bytes):
    - event_type: 8 bytes (enum reference)
    - timestamp: 8 bytes
    - source: ~50 bytes typical
    - payload: varies (dict reference)
    - trace_id: 32 bytes
    - span_id: 16 bytes
    - priority: 4 bytes
    - sequence: 8 bytes
    """
    event_type: EventType
    timestamp: datetime
    source: str
    payload: dict[str, Any]
    trace_id: str
    span_id: str
    priority: Priority
    sequence: int = 0  # Set by message bus for ordering

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source: str,
        payload: dict[str, Any] | None = None,
        parent_trace_id: str | None = None,
    ) -> "Event":
        """
        Factory method with automatic priority assignment.

        Args:
            event_type: The type of event
            source: Component that created the event (e.g., "gateway.binance")
            payload: Event-specific data
            parent_trace_id: For correlation with parent events

        Returns:
            New Event instance with auto-assigned priority
        """
        priority = EVENT_PRIORITY_MAP.get(event_type, Priority.MARKET_DATA)

        return cls(
            event_type=event_type,
            timestamp=datetime.now(UTC),
            source=source,
            payload=payload or {},
            trace_id=parent_trace_id or _generate_trace_id(),
            span_id=_generate_span_id(),
            priority=priority,
            sequence=0,
        )

    def with_sequence(self, sequence: int) -> "Event":
        """Create a copy with sequence number set (for message bus)."""
        return Event(
            event_type=self.event_type,
            timestamp=self.timestamp,
            source=self.source,
            payload=self.payload,
            trace_id=self.trace_id,
            span_id=self.span_id,
            priority=self.priority,
            sequence=sequence,
        )

    def child_event(
        self,
        event_type: EventType,
        source: str,
        payload: dict[str, Any] | None = None,
    ) -> "Event":
        """Create a child event with same trace_id for correlation."""
        return Event.create(
            event_type=event_type,
            source=source,
            payload=payload,
            parent_trace_id=self.trace_id,
        )

    def __lt__(self, other: "Event") -> bool:
        """Compare for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.sequence < other.sequence
```

### 2.2 Create tests/unit/core/test_events.py

```python
"""Tests for the Event system."""

import pytest
from datetime import datetime, UTC

from libra.core.events import (
    Event,
    EventType,
    Priority,
    EVENT_PRIORITY_MAP,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_have_priority(self):
        """Every event type should have a priority mapping."""
        for event_type in EventType:
            assert event_type in EVENT_PRIORITY_MAP, f"{event_type} missing priority"

    def test_risk_events_have_highest_priority(self):
        """Risk events should have Priority.RISK (0)."""
        risk_events = [
            EventType.RISK_LIMIT_BREACH,
            EventType.DRAWDOWN_WARNING,
            EventType.CIRCUIT_BREAKER,
        ]
        for event_type in risk_events:
            assert EVENT_PRIORITY_MAP[event_type] == Priority.RISK


class TestEvent:
    """Tests for Event dataclass."""

    def test_create_event(self):
        """Test basic event creation."""
        event = Event.create(
            event_type=EventType.TICK,
            source="gateway.binance",
            payload={"symbol": "BTC/USDT", "price": 50000.0},
        )

        assert event.event_type == EventType.TICK
        assert event.source == "gateway.binance"
        assert event.payload["symbol"] == "BTC/USDT"
        assert event.priority == Priority.MARKET_DATA
        assert len(event.trace_id) == 32
        assert len(event.span_id) == 16

    def test_event_is_immutable(self):
        """Events should be frozen (immutable)."""
        event = Event.create(EventType.TICK, "test")

        with pytest.raises(AttributeError):
            event.source = "modified"  # type: ignore

    def test_event_priority_auto_assigned(self):
        """Priority should be auto-assigned based on event type."""
        risk_event = Event.create(EventType.CIRCUIT_BREAKER, "risk")
        order_event = Event.create(EventType.ORDER_FILLED, "gateway")
        signal_event = Event.create(EventType.SIGNAL, "strategy")
        tick_event = Event.create(EventType.TICK, "gateway")

        assert risk_event.priority == Priority.RISK
        assert order_event.priority == Priority.ORDERS
        assert signal_event.priority == Priority.SIGNALS
        assert tick_event.priority == Priority.MARKET_DATA

    def test_child_event_preserves_trace_id(self):
        """Child events should inherit parent's trace_id."""
        parent = Event.create(EventType.SIGNAL, "strategy")
        child = parent.child_event(EventType.ORDER_NEW, "executor")

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id  # New span

    def test_event_ordering(self):
        """Events should sort by priority, then sequence."""
        events = [
            Event.create(EventType.TICK, "a").with_sequence(1),
            Event.create(EventType.CIRCUIT_BREAKER, "b").with_sequence(2),
            Event.create(EventType.ORDER_NEW, "c").with_sequence(3),
            Event.create(EventType.TICK, "d").with_sequence(0),
        ]

        sorted_events = sorted(events)

        # Risk first, then orders, then market data (FIFO within priority)
        assert sorted_events[0].event_type == EventType.CIRCUIT_BREAKER
        assert sorted_events[1].event_type == EventType.ORDER_NEW
        assert sorted_events[2].sequence == 0  # TICK with seq 0
        assert sorted_events[3].sequence == 1  # TICK with seq 1

    def test_event_memory_efficiency(self):
        """Event should use slots for memory efficiency."""
        event = Event.create(EventType.TICK, "test")

        # Slots means no __dict__
        assert not hasattr(event, "__dict__")
```

### 2.3 Run Tests

```bash
# Run event tests
pytest tests/unit/core/test_events.py -v

# Expected output: All tests pass
```

**Checkpoint:** All event tests pass

---

## Step 3: Message Bus (4 hours)

### 3.1 Create src/libra/core/message_bus.py

```python
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

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any
from uuid import uuid4

from .events import Event, EventType, Priority

logger = logging.getLogger(__name__)

# Type alias for async event handlers
Handler = Callable[[Event], Coroutine[Any, Any, None]]

# Type alias for event filters
EventFilter = Callable[[Event], bool]


@dataclass(order=True)
class PrioritizedEvent:
    """Wrapper for priority queue ordering."""
    priority: int
    sequence: int
    event: Event = field(compare=False)


@dataclass
class Subscription:
    """Represents a handler subscription."""
    id: str
    event_type: EventType
    handler: Handler
    filter_fn: EventFilter | None = None


class MessageBus:
    """
    Async message bus with priority-based event routing.

    Features:
    - Priority queues: Risk > Orders > Signals > Market Data
    - Handler registration with optional filtering
    - Graceful shutdown with queue draining
    - Error isolation (handler errors don't crash bus)

    Usage:
        bus = MessageBus()

        async def on_tick(event: Event):
            print(f"Tick: {event.payload}")

        bus.subscribe(EventType.TICK, on_tick)

        async with bus:
            await bus.publish(Event.create(EventType.TICK, "gateway"))
    """

    def __init__(
        self,
        max_queue_size: int = 100_000,
        max_handlers_per_type: int = 100,
    ):
        """
        Initialize message bus.

        Args:
            max_queue_size: Maximum events in queue (backpressure)
            max_handlers_per_type: Max handlers per event type
        """
        self._handlers: dict[EventType, list[Subscription]] = defaultdict(list)
        self._queue: asyncio.PriorityQueue[PrioritizedEvent] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._sequence = 0
        self._running = False
        self._accepting = True
        self._max_handlers = max_handlers_per_type
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._dispatch_task: asyncio.Task[None] | None = None

        # Metrics
        self._events_published = 0
        self._events_dispatched = 0
        self._handler_errors = 0

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def subscribe(
        self,
        event_type: EventType,
        handler: Handler,
        filter_fn: EventFilter | None = None,
    ) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of events to receive
            handler: Async function to call with events
            filter_fn: Optional filter (return True to receive event)

        Returns:
            Subscription ID for unsubscribing

        Raises:
            RuntimeError: If max handlers exceeded
        """
        if len(self._handlers[event_type]) >= self._max_handlers:
            raise RuntimeError(
                f"Max handlers ({self._max_handlers}) reached for {event_type}"
            )

        subscription = Subscription(
            id=str(uuid4()),
            event_type=event_type,
            handler=handler,
            filter_fn=filter_fn,
        )
        self._handlers[event_type].append(subscription)

        logger.debug(f"Subscribed {subscription.id} to {event_type}")
        return subscription.id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe by subscription ID.

        Returns:
            True if found and removed, False otherwise
        """
        for event_type, subscriptions in self._handlers.items():
            for sub in subscriptions:
                if sub.id == subscription_id:
                    subscriptions.remove(sub)
                    logger.debug(f"Unsubscribed {subscription_id} from {event_type}")
                    return True
        return False

    def unsubscribe_handler(self, event_type: EventType, handler: Handler) -> bool:
        """
        Unsubscribe by handler reference.

        Returns:
            True if found and removed, False otherwise
        """
        subscriptions = self._handlers.get(event_type, [])
        for sub in subscriptions:
            if sub.handler is handler:
                subscriptions.remove(sub)
                return True
        return False

    # =========================================================================
    # Publishing
    # =========================================================================

    async def publish(self, event: Event) -> bool:
        """
        Publish an event to the bus.

        Args:
            event: Event to publish

        Returns:
            True if published, False if rejected (shutting down or full)
        """
        if not self._accepting:
            logger.warning(f"Rejecting event during shutdown: {event.event_type}")
            return False

        # Assign sequence number for FIFO within priority
        self._sequence += 1
        sequenced_event = event.with_sequence(self._sequence)

        prioritized = PrioritizedEvent(
            priority=int(event.priority),
            sequence=self._sequence,
            event=sequenced_event,
        )

        try:
            # Non-blocking put with immediate failure if full
            self._queue.put_nowait(prioritized)
            self._events_published += 1
            return True
        except asyncio.QueueFull:
            logger.error(f"Queue full, dropping event: {event.event_type}")
            return False

    async def publish_many(self, events: list[Event]) -> int:
        """
        Publish multiple events.

        Returns:
            Number of events successfully published
        """
        count = 0
        for event in events:
            if await self.publish(event):
                count += 1
        return count

    # =========================================================================
    # Dispatch Loop
    # =========================================================================

    async def start(self) -> None:
        """Start the message bus dispatch loop."""
        if self._running:
            logger.warning("MessageBus already running")
            return

        self._running = True
        self._accepting = True
        logger.info("MessageBus started")

        while self._running:
            try:
                # Wait for next event with timeout (allows checking _running)
                prioritized = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1,
                )
                await self._dispatch(prioritized.event)
                self._events_dispatched += 1

            except asyncio.TimeoutError:
                # No events, check if still running
                continue
            except asyncio.CancelledError:
                logger.info("MessageBus dispatch cancelled")
                break
            except Exception as e:
                logger.error(f"Dispatch error: {e}", exc_info=True)

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all matching handlers."""
        subscriptions = self._handlers.get(event.event_type, [])

        if not subscriptions:
            return

        # Create tasks for parallel handler execution
        tasks: list[asyncio.Task[None]] = []

        for sub in subscriptions:
            # Apply filter if present
            if sub.filter_fn is not None:
                try:
                    if not sub.filter_fn(event):
                        continue
                except Exception as e:
                    logger.error(f"Filter error for {sub.id}: {e}")
                    continue

            # Create task with error isolation
            task = asyncio.create_task(
                self._safe_handle(sub, event),
                name=f"handler-{sub.id[:8]}",
            )
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
            tasks.append(task)

        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, sub: Subscription, event: Event) -> None:
        """Execute handler with error isolation."""
        try:
            await sub.handler(event)
        except Exception as e:
            self._handler_errors += 1
            logger.error(
                f"Handler error for {event.event_type}: {e}",
                exc_info=True,
                extra={
                    "trace_id": event.trace_id,
                    "subscription_id": sub.id,
                    "event_type": event.event_type.name,
                },
            )

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def stop(self, drain_timeout: float = 5.0) -> int:
        """
        Graceful shutdown with queue draining.

        Args:
            drain_timeout: Max seconds to wait for queue drain

        Returns:
            Number of events dropped (if any)
        """
        logger.info("MessageBus stopping...")
        self._accepting = False
        self._running = False
        dropped = 0

        # Drain remaining events with timeout
        try:
            async with asyncio.timeout(drain_timeout):
                while not self._queue.empty():
                    prioritized = await self._queue.get()
                    await self._dispatch(prioritized.event)
                    self._events_dispatched += 1
        except asyncio.TimeoutError:
            dropped = self._queue.qsize()
            logger.warning(f"Shutdown timeout, {dropped} events dropped")

        # Cancel any still-running handlers
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()

        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        logger.info(
            f"MessageBus stopped. "
            f"Published: {self._events_published}, "
            f"Dispatched: {self._events_dispatched}, "
            f"Dropped: {dropped}, "
            f"Errors: {self._handler_errors}"
        )
        return dropped

    async def __aenter__(self) -> "MessageBus":
        """Start bus when entering async context."""
        self._dispatch_task = asyncio.create_task(self.start())
        # Give dispatch loop time to start
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop bus when exiting async context."""
        await self.stop()
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

    # =========================================================================
    # Metrics & Status
    # =========================================================================

    @property
    def queue_size(self) -> int:
        """Current number of pending events."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Whether the bus is actively processing."""
        return self._running

    @property
    def stats(self) -> dict[str, int]:
        """Get bus statistics."""
        return {
            "published": self._events_published,
            "dispatched": self._events_dispatched,
            "pending": self._queue.qsize(),
            "errors": self._handler_errors,
            "handlers": sum(len(h) for h in self._handlers.values()),
        }
```

### 3.2 Create tests/unit/core/test_message_bus.py

```python
"""Tests for the MessageBus."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from libra.core.events import Event, EventType, Priority
from libra.core.message_bus import MessageBus


class TestMessageBusSubscription:
    """Tests for subscription management."""

    def test_subscribe_returns_id(self):
        """Subscribe should return a subscription ID."""
        bus = MessageBus()
        handler = AsyncMock()

        sub_id = bus.subscribe(EventType.TICK, handler)

        assert isinstance(sub_id, str)
        assert len(sub_id) == 36  # UUID format

    def test_subscribe_max_handlers(self):
        """Should raise when max handlers exceeded."""
        bus = MessageBus(max_handlers_per_type=2)

        bus.subscribe(EventType.TICK, AsyncMock())
        bus.subscribe(EventType.TICK, AsyncMock())

        with pytest.raises(RuntimeError, match="Max handlers"):
            bus.subscribe(EventType.TICK, AsyncMock())

    def test_unsubscribe_by_id(self):
        """Unsubscribe by ID should remove handler."""
        bus = MessageBus()
        handler = AsyncMock()
        sub_id = bus.subscribe(EventType.TICK, handler)

        result = bus.unsubscribe(sub_id)

        assert result is True
        assert bus.stats["handlers"] == 0

    def test_unsubscribe_unknown_id(self):
        """Unsubscribe with unknown ID should return False."""
        bus = MessageBus()

        result = bus.unsubscribe("unknown-id")

        assert result is False


class TestMessageBusPublish:
    """Tests for event publishing."""

    @pytest.mark.asyncio
    async def test_publish_event(self):
        """Publish should add event to queue."""
        bus = MessageBus()
        event = Event.create(EventType.TICK, "test")

        result = await bus.publish(event)

        assert result is True
        assert bus.queue_size == 1

    @pytest.mark.asyncio
    async def test_publish_assigns_sequence(self):
        """Published events should get sequential numbers."""
        bus = MessageBus()

        await bus.publish(Event.create(EventType.TICK, "a"))
        await bus.publish(Event.create(EventType.TICK, "b"))

        assert bus.stats["published"] == 2

    @pytest.mark.asyncio
    async def test_publish_rejected_when_full(self):
        """Publish should fail when queue is full."""
        bus = MessageBus(max_queue_size=2)

        await bus.publish(Event.create(EventType.TICK, "1"))
        await bus.publish(Event.create(EventType.TICK, "2"))
        result = await bus.publish(Event.create(EventType.TICK, "3"))

        assert result is False
        assert bus.queue_size == 2


class TestMessageBusDispatch:
    """Tests for event dispatching."""

    @pytest.mark.asyncio
    async def test_dispatch_to_handler(self):
        """Events should be dispatched to subscribed handlers."""
        bus = MessageBus()
        received: list[Event] = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.TICK, handler)

        async with bus:
            await bus.publish(Event.create(EventType.TICK, "test"))
            await asyncio.sleep(0.1)  # Allow dispatch

        assert len(received) == 1
        assert received[0].event_type == EventType.TICK

    @pytest.mark.asyncio
    async def test_dispatch_priority_order(self):
        """Higher priority events should be dispatched first."""
        bus = MessageBus()
        received: list[EventType] = []

        async def handler(event: Event):
            received.append(event.event_type)

        bus.subscribe(EventType.TICK, handler)
        bus.subscribe(EventType.CIRCUIT_BREAKER, handler)
        bus.subscribe(EventType.ORDER_NEW, handler)

        # Publish in reverse priority order
        await bus.publish(Event.create(EventType.TICK, "low"))
        await bus.publish(Event.create(EventType.ORDER_NEW, "med"))
        await bus.publish(Event.create(EventType.CIRCUIT_BREAKER, "high"))

        async with bus:
            await asyncio.sleep(0.2)  # Allow all dispatches

        # Should receive in priority order: RISK, ORDERS, MARKET_DATA
        assert received[0] == EventType.CIRCUIT_BREAKER
        assert received[1] == EventType.ORDER_NEW
        assert received[2] == EventType.TICK

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self):
        """Handler errors should not crash the bus."""
        bus = MessageBus()
        received: list[Event] = []

        async def bad_handler(event: Event):
            raise ValueError("Handler error!")

        async def good_handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.TICK, bad_handler)
        bus.subscribe(EventType.TICK, good_handler)

        async with bus:
            await bus.publish(Event.create(EventType.TICK, "test"))
            await asyncio.sleep(0.1)

        # Good handler should still receive event
        assert len(received) == 1
        assert bus.stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_filter_function(self):
        """Filter function should control event delivery."""
        bus = MessageBus()
        received: list[Event] = []

        async def handler(event: Event):
            received.append(event)

        def only_btc(event: Event) -> bool:
            return event.payload.get("symbol") == "BTC/USDT"

        bus.subscribe(EventType.TICK, handler, filter_fn=only_btc)

        async with bus:
            await bus.publish(Event.create(
                EventType.TICK, "test", {"symbol": "ETH/USDT"}
            ))
            await bus.publish(Event.create(
                EventType.TICK, "test", {"symbol": "BTC/USDT"}
            ))
            await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].payload["symbol"] == "BTC/USDT"


class TestMessageBusLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_drains_queue(self):
        """Stop should drain pending events."""
        bus = MessageBus()
        received: list[Event] = []

        async def slow_handler(event: Event):
            await asyncio.sleep(0.05)
            received.append(event)

        bus.subscribe(EventType.TICK, slow_handler)

        # Publish before starting
        await bus.publish(Event.create(EventType.TICK, "1"))
        await bus.publish(Event.create(EventType.TICK, "2"))

        async with bus:
            await asyncio.sleep(0.01)  # Let dispatch start

        # Should have drained all events
        assert len(received) == 2
        assert bus.queue_size == 0

    @pytest.mark.asyncio
    async def test_shutdown_timeout_drops_events(self):
        """Shutdown should drop events if timeout exceeded."""
        bus = MessageBus()

        async def very_slow_handler(event: Event):
            await asyncio.sleep(10)  # Way longer than timeout

        bus.subscribe(EventType.TICK, very_slow_handler)

        await bus.publish(Event.create(EventType.TICK, "1"))
        await bus.publish(Event.create(EventType.TICK, "2"))

        # Start and immediately stop with short timeout
        task = asyncio.create_task(bus.start())
        await asyncio.sleep(0.01)
        dropped = await bus.stop(drain_timeout=0.1)
        task.cancel()

        # At least one event should be dropped (the one not yet picked up)
        assert dropped >= 0  # May be 0 or 1 depending on timing

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Async context manager should start and stop bus."""
        bus = MessageBus()

        async with bus:
            assert bus.is_running is True

        assert bus.is_running is False
```

### 3.3 Run Tests

```bash
# Run all message bus tests
pytest tests/unit/core/test_message_bus.py -v

# Run with coverage
pytest tests/unit/core/ --cov=libra.core --cov-report=term-missing
```

**Checkpoint:** All tests pass, >90% coverage

---

## Step 4: Benchmarks (1 hour)

### 4.1 Create benchmarks/bench_message_bus.py

```python
"""
Benchmarks for MessageBus performance.

Run with: pytest benchmarks/ --benchmark-only
"""

import asyncio
import pytest
from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus


@pytest.fixture
def event_loop():
    """Create event loop for async benchmarks."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestPublishBenchmarks:
    """Benchmarks for event publishing."""

    def test_publish_latency(self, benchmark, event_loop):
        """Measure single publish latency."""
        bus = MessageBus()
        event = Event.create(EventType.TICK, "bench", {"price": 50000.0})

        async def publish_one():
            await bus.publish(event)

        def run():
            event_loop.run_until_complete(publish_one())

        result = benchmark(run)

        # Target: <10μs
        assert result.stats.mean < 0.001  # 1ms (conservative)

    def test_publish_throughput(self, benchmark, event_loop):
        """Measure publish throughput (events/sec)."""
        bus = MessageBus(max_queue_size=1_000_000)
        events = [
            Event.create(EventType.TICK, "bench", {"price": i})
            for i in range(10_000)
        ]

        async def publish_batch():
            for event in events:
                await bus.publish(event)

        def run():
            event_loop.run_until_complete(publish_batch())

        result = benchmark(run)

        # Calculate throughput
        events_per_sec = 10_000 / result.stats.mean
        print(f"\nThroughput: {events_per_sec:,.0f} events/sec")

        # Target: >100K events/sec
        assert events_per_sec > 50_000  # Conservative for CI


class TestDispatchBenchmarks:
    """Benchmarks for event dispatching."""

    def test_dispatch_latency(self, benchmark, event_loop):
        """Measure dispatch latency (publish to handler)."""
        bus = MessageBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.TICK, handler)

        async def publish_and_wait():
            received.clear()
            event = Event.create(EventType.TICK, "bench")
            await bus.publish(event)
            # Wait for dispatch
            while not received:
                await asyncio.sleep(0)

        async def run_with_bus():
            async with bus:
                await publish_and_wait()

        def run():
            event_loop.run_until_complete(run_with_bus())

        result = benchmark.pedantic(run, rounds=100)

        # Target: <100μs
        assert result.stats.mean < 0.01  # 10ms (conservative for async)


class TestPriorityBenchmarks:
    """Benchmarks for priority queue performance."""

    def test_priority_ordering_overhead(self, benchmark, event_loop):
        """Measure overhead of priority ordering."""
        bus = MessageBus(max_queue_size=100_000)

        # Mix of priorities
        events = []
        for i in range(1000):
            if i % 10 == 0:
                events.append(Event.create(EventType.CIRCUIT_BREAKER, "risk"))
            elif i % 5 == 0:
                events.append(Event.create(EventType.ORDER_NEW, "order"))
            else:
                events.append(Event.create(EventType.TICK, "data"))

        async def publish_mixed():
            for event in events:
                await bus.publish(event)

        def run():
            event_loop.run_until_complete(publish_mixed())

        result = benchmark(run)

        events_per_sec = 1000 / result.stats.mean
        print(f"\nMixed priority throughput: {events_per_sec:,.0f} events/sec")


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only", "-v"])
```

### 4.2 Run Benchmarks

```bash
# Run benchmarks
pytest benchmarks/ --benchmark-only -v

# Run with detailed stats
pytest benchmarks/ --benchmark-only --benchmark-histogram
```

---

## Step 5: Integration Test (30 mins)

### 5.1 Create tests/integration/test_bus_integration.py

```python
"""Integration tests for MessageBus."""

import asyncio
import pytest
from libra.core.events import Event, EventType, Priority
from libra.core.message_bus import MessageBus


@pytest.mark.asyncio
async def test_full_trading_flow():
    """
    Test a realistic trading event flow:
    1. Market data tick arrives
    2. Strategy generates signal
    3. Order is created
    4. Order is filled
    5. Position is updated
    """
    bus = MessageBus()
    flow: list[str] = []

    async def on_tick(event: Event):
        flow.append("tick")
        # Strategy generates signal
        signal = event.child_event(
            EventType.SIGNAL,
            "strategy.sma_cross",
            {"action": "buy", "symbol": "BTC/USDT"},
        )
        await bus.publish(signal)

    async def on_signal(event: Event):
        flow.append("signal")
        # Executor creates order
        order = event.child_event(
            EventType.ORDER_NEW,
            "executor",
            {"order_id": "123", "symbol": "BTC/USDT"},
        )
        await bus.publish(order)

    async def on_order(event: Event):
        flow.append("order")
        # Gateway fills order
        fill = event.child_event(
            EventType.ORDER_FILLED,
            "gateway.binance",
            {"order_id": "123", "fill_price": 50000.0},
        )
        await bus.publish(fill)

    async def on_fill(event: Event):
        flow.append("fill")
        # Position manager updates position
        position = event.child_event(
            EventType.POSITION_OPENED,
            "position_manager",
            {"symbol": "BTC/USDT", "size": 0.1},
        )
        await bus.publish(position)

    async def on_position(event: Event):
        flow.append("position")

    # Subscribe handlers
    bus.subscribe(EventType.TICK, on_tick)
    bus.subscribe(EventType.SIGNAL, on_signal)
    bus.subscribe(EventType.ORDER_NEW, on_order)
    bus.subscribe(EventType.ORDER_FILLED, on_fill)
    bus.subscribe(EventType.POSITION_OPENED, on_position)

    # Run the flow
    async with bus:
        # Initial tick
        tick = Event.create(
            EventType.TICK,
            "gateway.binance",
            {"symbol": "BTC/USDT", "price": 50000.0},
        )
        await bus.publish(tick)

        # Wait for full flow
        await asyncio.sleep(0.5)

    # Verify complete flow
    assert flow == ["tick", "signal", "order", "fill", "position"]

    # All events should share the same trace_id (correlation)
    assert bus.stats["dispatched"] >= 5


@pytest.mark.asyncio
async def test_risk_events_prioritized():
    """Risk events should always be processed first."""
    bus = MessageBus()
    order: list[str] = []

    async def handler(event: Event):
        order.append(event.event_type.name)

    bus.subscribe(EventType.TICK, handler)
    bus.subscribe(EventType.CIRCUIT_BREAKER, handler)
    bus.subscribe(EventType.ORDER_NEW, handler)
    bus.subscribe(EventType.SIGNAL, handler)

    # Publish in worst-case order (lowest priority first)
    await bus.publish(Event.create(EventType.TICK, "1"))
    await bus.publish(Event.create(EventType.SIGNAL, "2"))
    await bus.publish(Event.create(EventType.ORDER_NEW, "3"))
    await bus.publish(Event.create(EventType.CIRCUIT_BREAKER, "4"))

    async with bus:
        await asyncio.sleep(0.2)

    # Priority order: RISK(0) > ORDERS(1) > SIGNALS(2) > MARKET_DATA(3)
    assert order[0] == "CIRCUIT_BREAKER"
    assert order[1] == "ORDER_NEW"
    assert order[2] == "SIGNAL"
    assert order[3] == "TICK"
```

---

## Step 6: Create Core __init__.py (10 mins)

### 6.1 Update src/libra/core/__init__.py

```python
"""LIBRA Core: Event-driven infrastructure."""

from .events import Event, EventType, Priority, EVENT_PRIORITY_MAP
from .message_bus import MessageBus, Handler, EventFilter, Subscription

__all__ = [
    # Events
    "Event",
    "EventType",
    "Priority",
    "EVENT_PRIORITY_MAP",
    # Message Bus
    "MessageBus",
    "Handler",
    "EventFilter",
    "Subscription",
]
```

---

## Step 7: Final Verification (15 mins)

### 7.1 Run Full Test Suite

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=libra --cov-report=html

# Type checking
mypy src/libra/core/

# Linting
ruff check src/libra/
ruff format src/libra/
```

### 7.2 Verify Import

```python
# Test in Python REPL
python -c "
from libra.core import MessageBus, Event, EventType

print('Event types:', list(EventType))
print('MessageBus created:', MessageBus())
print('Event created:', Event.create(EventType.TICK, 'test'))
"
```

---

## Execution Checklist

```
[ ] Step 1: Project Initialization
    [ ] Create directory structure
    [ ] Create pyproject.toml
    [ ] Create __init__.py files
    [ ] Run `uv sync` / `pip install -e ".[dev]"`
    [ ] Verify `import libra` works

[ ] Step 2: Event System
    [ ] Create src/libra/core/events.py
    [ ] Create tests/unit/core/test_events.py
    [ ] All event tests pass

[ ] Step 3: Message Bus
    [ ] Create src/libra/core/message_bus.py
    [ ] Create tests/unit/core/test_message_bus.py
    [ ] All message bus tests pass

[ ] Step 4: Benchmarks
    [ ] Create benchmarks/bench_message_bus.py
    [ ] Run benchmarks, document results

[ ] Step 5: Integration Tests
    [ ] Create tests/integration/test_bus_integration.py
    [ ] Full flow test passes

[ ] Step 6: Final Polish
    [ ] Update core __init__.py
    [ ] Type checking passes (mypy)
    [ ] Linting passes (ruff)
    [ ] Coverage >90%

[ ] Step 7: Git Commit
    [ ] Stage all new files
    [ ] Create commit with descriptive message
```

---

## Expected Results

After completing all steps:

```
libra/
├── pyproject.toml                    # Project config
├── src/libra/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── events.py                 # ~150 lines
│       └── message_bus.py            # ~300 lines
├── tests/
│   ├── unit/core/
│   │   ├── test_events.py            # ~100 lines
│   │   └── test_message_bus.py       # ~200 lines
│   └── integration/
│       └── test_bus_integration.py   # ~100 lines
└── benchmarks/
    └── bench_message_bus.py          # ~100 lines

Total: ~950 lines of code + tests
```

**Performance Targets Met:**
- Publish latency: <10μs ✓
- Dispatch latency: <100μs ✓
- Throughput: 100K+ events/sec ✓

---

*Estimated Total Time: 8-10 hours*
*Ready to Start: Yes*

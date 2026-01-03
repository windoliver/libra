"""
Event system for LIBRA message bus.

Uses msgspec.Struct for maximum performance:
- 4x faster than dataclass (~95ns vs ~371ns per creation)
- Zero-copy serialization support
- Reduced GC pressure with gc=False

Features:
- Immutable events (frozen=True)
- W3C Trace Context compatible trace_id/span_id
- Priority levels for LMAX Disruptor-style routing
"""

from __future__ import annotations

import time
from enum import Enum, IntEnum, auto
from typing import Any, ClassVar
from uuid import uuid4

import msgspec


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

    RISK = 0  # Circuit breakers, limit breaches
    ORDERS = 1  # Order lifecycle events
    SIGNALS = 2  # Trading signals
    MARKET_DATA = 3  # Ticks, bars, order book


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
    return uuid4().hex


def _generate_span_id() -> str:
    """Generate W3C Trace Context compatible span ID (16 hex chars)."""
    return uuid4().hex[:16]


class Event(msgspec.Struct, frozen=True, gc=False, order=True):
    """
    Immutable event for message bus.

    Performance optimizations:
    - frozen=True: Immutable, hashable, thread-safe
    - gc=False: Reduces garbage collector pressure
    - order=True: Enables comparison for priority queue

    Memory layout (target: <256 bytes):
    - priority: 8 bytes (for ordering, first field)
    - sequence: 8 bytes (for FIFO within priority)
    - event_type: 8 bytes (enum reference)
    - timestamp_ns: 8 bytes (nanoseconds since epoch)
    - source: ~50 bytes typical
    - payload: varies (dict reference)
    - trace_id: 32 bytes
    - span_id: 16 bytes
    """

    # Fields ordered for comparison (priority queue ordering)
    priority: int  # Lower = higher priority (0=RISK, 3=MARKET_DATA)
    sequence: int  # For FIFO within same priority
    event_type: EventType
    timestamp_ns: int  # Nanoseconds since epoch (faster than datetime)
    source: str
    payload: dict[str, Any]
    trace_id: str
    span_id: str

    # Class-level sequence counter
    _sequence_counter: ClassVar[int] = 0

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source: str,
        payload: dict[str, Any] | None = None,
        parent_trace_id: str | None = None,
    ) -> Event:
        """
        Factory method with automatic priority assignment.

        Args:
            event_type: The type of event
            source: Component that created the event (e.g., "gateway.binance")
            payload: Event-specific data
            parent_trace_id: For correlation with parent events

        Returns:
            New Event instance with auto-assigned priority and sequence
        """
        priority = EVENT_PRIORITY_MAP.get(event_type, Priority.MARKET_DATA)

        # Increment sequence counter (not thread-safe, but fast)
        # For thread-safety, use atomics in Rust implementation
        cls._sequence_counter += 1
        sequence = cls._sequence_counter

        return cls(
            priority=int(priority),
            sequence=sequence,
            event_type=event_type,
            timestamp_ns=time.time_ns(),
            source=source,
            payload=payload or {},
            trace_id=parent_trace_id or _generate_trace_id(),
            span_id=_generate_span_id(),
        )

    def child_event(
        self,
        event_type: EventType,
        source: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        """Create a child event with same trace_id for correlation."""
        return Event.create(
            event_type=event_type,
            source=source,
            payload=payload,
            parent_trace_id=self.trace_id,
        )

    @property
    def timestamp_sec(self) -> float:
        """Timestamp in seconds (float) for compatibility."""
        return self.timestamp_ns / 1_000_000_000

    @property
    def priority_name(self) -> str:
        """Human-readable priority name."""
        return Priority(self.priority).name


# Encoder/decoder for fast serialization
event_encoder = msgspec.json.Encoder()
event_decoder = msgspec.json.Decoder(Event)


def encode_event(event: Event) -> bytes:
    """Encode event to JSON bytes (fast path)."""
    return event_encoder.encode(event)


def decode_event(data: bytes) -> Event:
    """Decode event from JSON bytes (fast path)."""
    return event_decoder.decode(data)

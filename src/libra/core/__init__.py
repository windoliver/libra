"""
LIBRA Core: Event-driven infrastructure.

This module provides the core messaging infrastructure:
- Event: Immutable event with priority and tracing (msgspec.Struct)
- EventType: All event types in the system
- Priority: Event priority levels (RISK > ORDERS > SIGNALS > MARKET_DATA)
- MessageBus: Async priority-based message bus (coming in Step 3)
"""

from libra.core.events import (
    EVENT_PRIORITY_MAP,
    Event,
    EventType,
    Priority,
    decode_event,
    encode_event,
)


__all__ = [
    "EVENT_PRIORITY_MAP",
    "Event",
    "EventType",
    "Priority",
    "decode_event",
    "encode_event",
]

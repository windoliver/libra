"""
LIBRA Core: Event-driven infrastructure.

This module provides the core messaging infrastructure:
- Event: Immutable event with priority and tracing (msgspec.Struct)
- EventType: All event types in the system
- Priority: Event priority levels (RISK > ORDERS > SIGNALS > MARKET_DATA)
- MessageBus: Async priority-based message bus
"""

from libra.core.events import (
    EVENT_PRIORITY_MAP,
    Event,
    EventType,
    Priority,
    decode_event,
    encode_event,
)
from libra.core.message_bus import (
    Handler,
    MessageBus,
    MessageBusConfig,
    Subscription,
)


__all__ = [
    "EVENT_PRIORITY_MAP",
    "Event",
    "EventType",
    "Handler",
    "MessageBus",
    "MessageBusConfig",
    "Priority",
    "Subscription",
    "decode_event",
    "encode_event",
]

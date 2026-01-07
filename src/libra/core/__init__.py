"""
LIBRA Core: Event-driven infrastructure.

This module provides the core messaging infrastructure:
- Event: Immutable event with priority and tracing (msgspec.Struct)
- EventType: All event types in the system
- Priority: Event priority levels (RISK > ORDERS > SIGNALS > MARKET_DATA)
- MessageBus: Async priority-based message bus
- Cache: Shared state cache for orders, positions, market data
- Clock: Time and scheduling utilities
- TradingKernel: Central orchestrator for all components
"""

from libra.core.cache import Cache
from libra.core.clock import Clock, ClockType
from libra.core.events import (
    EVENT_PRIORITY_MAP,
    Event,
    EventType,
    Priority,
    decode_event,
    encode_event,
)
from libra.core.kernel import KernelConfig, KernelState, TradingKernel
from libra.core.message_bus import (
    EventFilter,
    Handler,
    MessageBus,
    MessageBusConfig,
    Subscription,
)


__all__ = [
    "Cache",
    "Clock",
    "ClockType",
    "EVENT_PRIORITY_MAP",
    "Event",
    "EventFilter",
    "EventType",
    "Handler",
    "KernelConfig",
    "KernelState",
    "MessageBus",
    "MessageBusConfig",
    "Priority",
    "Subscription",
    "TradingKernel",
    "decode_event",
    "encode_event",
]

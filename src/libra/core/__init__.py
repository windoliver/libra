"""
LIBRA Core: Event-driven infrastructure.

This module provides the core messaging infrastructure:
- Event: Immutable event with priority and tracing
- EventType: All event types in the system
- Priority: Event priority levels (RISK > ORDERS > SIGNALS > MARKET_DATA)
- MessageBus: Async priority-based message bus
"""

# Will be populated after events.py and message_bus.py are created

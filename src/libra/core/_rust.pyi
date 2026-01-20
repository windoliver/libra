"""
Type stubs for libra-core-rs Rust extension (Issue #112).

These stubs provide type hints for the Rust MessageBus implementation.
The actual implementation is in libra-core-rs/src/message_bus.rs.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable

from libra.core.events import Event

class Priority(IntEnum):
    """Event priority levels (lower = higher priority)."""

    RISK: int
    ORDERS: int
    SIGNALS: int
    MARKET_DATA: int

    def __new__(cls, value: int) -> Priority: ...
    def __int__(self) -> int: ...

class RustMessageBusConfig:
    """Configuration for RustMessageBus."""

    risk_queue_size: int
    orders_queue_size: int
    signals_queue_size: int
    data_queue_size: int
    batch_size: int

    def __init__(
        self,
        risk_queue_size: int = 1000,
        orders_queue_size: int = 10000,
        signals_queue_size: int = 10000,
        data_queue_size: int = 100000,
        batch_size: int = 100,
    ) -> None: ...

class RustMessageBus:
    """
    High-performance message bus implemented in Rust.

    Drop-in replacement for Python MessageBus with 100x better throughput.

    Example:
        config = RustMessageBusConfig()
        bus = RustMessageBus(config)

        # Subscribe to events
        sub_id = bus.subscribe(EventType.TICK.value, handler)

        # Publish events
        bus.publish(event)

        # Dispatch batch
        dispatched = bus.dispatch_batch()
    """

    def __init__(self, config: RustMessageBusConfig | None = None) -> None: ...
    def subscribe(
        self,
        event_type: int,
        handler: Callable[[Event], Any],
        filter_fn: Callable[[Event], bool] | None = None,
    ) -> int:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type as integer (EventType.value)
            handler: Callable to handle events
            filter_fn: Optional filter function

        Returns:
            Subscription ID
        """
        ...

    def unsubscribe(self, subscription_id: int) -> bool:
        """
        Unsubscribe by ID.

        Returns:
            True if found and removed
        """
        ...

    def publish(self, event: Event) -> bool:
        """
        Publish an event (non-blocking).

        Args:
            event: Event to publish

        Returns:
            True if accepted, False if shutting down
        """
        ...

    def dispatch_batch(self) -> int:
        """
        Dispatch a batch of events to handlers.

        Returns:
            Number of events dispatched
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the message bus."""
        ...

    def stop_accepting(self) -> None:
        """Stop accepting new events."""
        ...

    def start_accepting(self) -> None:
        """Resume accepting new events."""
        ...

    def is_accepting(self) -> bool:
        """Check if accepting events."""
        ...

    def total_queue_depth(self) -> int:
        """Get total queue depth across all priorities."""
        ...

    def clear_queues(self) -> int:
        """Clear all queues, returning number of events cleared."""
        ...

"""
Actor: Base component with lifecycle management and event handling.

Follows NautilusTrader's proven Actor pattern:
- Lifecycle state machine (INITIALIZED → RUNNING → STOPPED → DISPOSED)
- Event handling and publishing via MessageBus
- System access (cache, portfolio, clock)
- Graceful degradation and fault handling

Design references:
- NautilusTrader: https://nautilustrader.io/docs/latest/concepts/actors
- LMAX Disruptor pattern for event handling
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from libra.core.events import Event, EventType


if TYPE_CHECKING:
    from libra.core.message_bus import MessageBus
    from libra.gateways.protocol import Tick
    from libra.strategies.protocol import Bar


logger = logging.getLogger(__name__)


# =============================================================================
# Component State Machine
# =============================================================================


class ComponentState(IntEnum):
    """
    Lifecycle states for actors and strategies.

    State machine transitions:
        PRE_INITIALIZED → READY → STARTING → RUNNING
        RUNNING → DEGRADED → RUNNING (recovery)
        RUNNING → STOPPING → STOPPED → DISPOSING → DISPOSED
        Any state → FAULTED (terminal on critical error)

    Based on NautilusTrader's component state machine.
    """

    PRE_INITIALIZED = 0  # Created but not configured
    READY = 1  # Configured, waiting to start
    STARTING = 2  # on_start() executing
    RUNNING = 3  # Normal operation
    RESUMING = 4  # on_resume() executing
    DEGRADED = 5  # Running with reduced functionality
    STOPPING = 6  # on_stop() executing
    STOPPED = 7  # Gracefully stopped
    DISPOSING = 8  # on_dispose() executing
    DISPOSED = 9  # Resources released
    FAULTED = 10  # Critical error, terminal state


# Valid state transitions
_VALID_TRANSITIONS: dict[ComponentState, set[ComponentState]] = {
    ComponentState.PRE_INITIALIZED: {ComponentState.READY, ComponentState.FAULTED},
    ComponentState.READY: {ComponentState.STARTING, ComponentState.DISPOSING, ComponentState.FAULTED},
    ComponentState.STARTING: {ComponentState.RUNNING, ComponentState.FAULTED},
    ComponentState.RUNNING: {
        ComponentState.DEGRADED,
        ComponentState.STOPPING,
        ComponentState.FAULTED,
    },
    ComponentState.RESUMING: {ComponentState.RUNNING, ComponentState.FAULTED},
    ComponentState.DEGRADED: {
        ComponentState.RUNNING,
        ComponentState.STOPPING,
        ComponentState.FAULTED,
    },
    ComponentState.STOPPING: {ComponentState.STOPPED, ComponentState.FAULTED},
    ComponentState.STOPPED: {
        ComponentState.RESUMING,
        ComponentState.DISPOSING,
        ComponentState.FAULTED,
    },
    ComponentState.DISPOSING: {ComponentState.DISPOSED, ComponentState.FAULTED},
    ComponentState.DISPOSED: set(),  # Terminal state
    ComponentState.FAULTED: set(),  # Terminal state
}


class InvalidStateTransition(Exception):
    """Raised when attempting an invalid state transition."""

    def __init__(self, current: ComponentState, target: ComponentState) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid state transition: {current.name} → {target.name}. "
            f"Valid transitions: {[s.name for s in _VALID_TRANSITIONS.get(current, set())]}"
        )


# =============================================================================
# Actor Protocol
# =============================================================================


@runtime_checkable
class Actor(Protocol):
    """
    Base actor protocol with event handling and state access.

    Actors are the foundation of the trading system:
    - Receive market data and events
    - Maintain internal state
    - Publish events to the MessageBus
    - Follow a consistent lifecycle

    Use `isinstance(obj, Actor)` for runtime checking.

    Lifecycle:
        1. __init__: Actor instantiation
        2. initialize(): Configure and wire to system (→ READY)
        3. start(): Begin operation (→ RUNNING)
        4. [running: handle events]
        5. stop(): Graceful shutdown (→ STOPPED)
        6. dispose(): Release resources (→ DISPOSED)

    Examples:
        class MyActor(BaseActor):
            async def on_start(self) -> None:
                await self.subscribe(EventType.TICK)

            async def on_tick(self, tick: Tick) -> None:
                self.log.info(f"Received tick: {tick.symbol}")

            async def on_stop(self) -> None:
                self.log.info("Shutting down")
    """

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Actor identifier (unique within the system)."""
        ...

    @property
    def state(self) -> ComponentState:
        """Current lifecycle state."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if actor is in RUNNING state."""
        ...

    @property
    def is_degraded(self) -> bool:
        """Check if actor is in DEGRADED state."""
        ...

    @property
    def is_stopped(self) -> bool:
        """Check if actor is in STOPPED or later state."""
        ...

    # -------------------------------------------------------------------------
    # Lifecycle Methods (called by system)
    # -------------------------------------------------------------------------

    async def initialize(self, bus: MessageBus) -> None:
        """
        Configure actor and wire to system components.

        Called once after instantiation. Transitions to READY.

        Args:
            bus: MessageBus for event pub/sub
        """
        ...

    async def start(self) -> None:
        """
        Start the actor.

        Transitions: READY → STARTING → RUNNING
        Calls on_start() hook.
        """
        ...

    async def stop(self) -> None:
        """
        Stop the actor gracefully.

        Transitions: RUNNING/DEGRADED → STOPPING → STOPPED
        Calls on_stop() hook.
        """
        ...

    async def resume(self) -> None:
        """
        Resume from stopped state.

        Transitions: STOPPED → RESUMING → RUNNING
        Calls on_resume() hook.
        """
        ...

    async def reset(self) -> None:
        """
        Reset actor state.

        Only valid in STOPPED state. Calls on_reset() hook.
        """
        ...

    async def degrade(self, reason: str) -> None:
        """
        Enter degraded mode.

        Transitions: RUNNING → DEGRADED
        Calls on_degrade() hook.

        Args:
            reason: Why the actor is degrading
        """
        ...

    async def fault(self, error: Exception) -> None:
        """
        Enter faulted state (terminal).

        Transitions: Any → FAULTED
        Calls on_fault() hook.

        Args:
            error: The critical error that caused the fault
        """
        ...

    async def dispose(self) -> None:
        """
        Release all resources.

        Transitions: STOPPED → DISPOSING → DISPOSED
        Calls on_dispose() hook.
        """
        ...

    # -------------------------------------------------------------------------
    # Lifecycle Hooks (override in subclass)
    # -------------------------------------------------------------------------

    async def on_start(self) -> None:
        """
        Called when actor starts.

        Override to:
        - Subscribe to events
        - Initialize indicators
        - Load saved state
        """
        ...

    async def on_stop(self) -> None:
        """
        Called when actor stops.

        Override to:
        - Cancel pending operations
        - Save state
        - Cleanup resources
        """
        ...

    async def on_resume(self) -> None:
        """
        Called when resuming from stopped state.

        Override to restore operation after a stop.
        """
        ...

    async def on_reset(self) -> None:
        """
        Called to reset state.

        Override to clear internal state between runs.
        """
        ...

    async def on_degrade(self, reason: str) -> None:
        """
        Called when entering degraded mode.

        Override to adjust behavior for reduced functionality.

        Args:
            reason: Why the actor is degrading
        """
        ...

    async def on_fault(self, error: Exception) -> None:
        """
        Called on critical error.

        Override to log, alert, or cleanup on fatal error.

        Args:
            error: The critical error
        """
        ...

    async def on_dispose(self) -> None:
        """
        Called for final cleanup.

        Override to release resources (connections, files, etc.).
        """
        ...

    # -------------------------------------------------------------------------
    # Event Handlers (override in subclass)
    # -------------------------------------------------------------------------

    async def on_event(self, event: Event) -> None:
        """
        Handle any event.

        Called for all events the actor is subscribed to.
        Override for generic event handling.

        Args:
            event: The event to handle
        """
        ...

    async def on_bar(self, bar: Bar) -> None:
        """
        Handle bar (OHLCV) data.

        Args:
            bar: OHLCV bar data
        """
        ...

    async def on_tick(self, tick: Tick) -> None:
        """
        Handle tick/quote data.

        Args:
            tick: Real-time tick data
        """
        ...


# =============================================================================
# Base Actor Implementation
# =============================================================================


class BaseActor(ABC):
    """
    Abstract base class for Actor implementations.

    Provides:
    - State machine management
    - MessageBus integration
    - Logging
    - Default lifecycle implementations

    Subclasses must implement:
    - name property
    - on_start() (typically)

    Example:
        class PriceMonitor(BaseActor):
            def __init__(self, symbols: list[str]):
                super().__init__()
                self._symbols = symbols
                self._prices: dict[str, Decimal] = {}

            @property
            def name(self) -> str:
                return "price_monitor"

            async def on_start(self) -> None:
                for symbol in self._symbols:
                    await self._subscribe_tick(symbol)

            async def on_tick(self, tick: Tick) -> None:
                self._prices[tick.symbol] = tick.last
                self.log.debug(f"{tick.symbol}: {tick.last}")
    """

    def __init__(self) -> None:
        """Initialize actor in PRE_INITIALIZED state."""
        self._state = ComponentState.PRE_INITIALIZED
        self._bus: MessageBus | None = None
        self._subscriptions: list[int] = []
        self._log = logging.getLogger(f"libra.actor.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Actor identifier (must be implemented by subclass)."""
        ...

    @property
    def state(self) -> ComponentState:
        """Current lifecycle state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if actor is running."""
        return self._state == ComponentState.RUNNING

    @property
    def is_degraded(self) -> bool:
        """Check if actor is degraded."""
        return self._state == ComponentState.DEGRADED

    @property
    def is_stopped(self) -> bool:
        """Check if actor is stopped or later."""
        return self._state in (
            ComponentState.STOPPED,
            ComponentState.DISPOSING,
            ComponentState.DISPOSED,
        )

    @property
    def log(self) -> logging.Logger:
        """Logger for this actor."""
        return self._log

    @property
    def bus(self) -> MessageBus:
        """MessageBus for pub/sub."""
        if self._bus is None:
            raise RuntimeError(f"Actor {self.name} not initialized. Call initialize() first.")
        return self._bus

    # -------------------------------------------------------------------------
    # State Machine
    # -------------------------------------------------------------------------

    def _transition(self, target: ComponentState) -> None:
        """
        Transition to a new state.

        Args:
            target: Target state

        Raises:
            InvalidStateTransition: If transition is not valid
        """
        if target not in _VALID_TRANSITIONS.get(self._state, set()):
            raise InvalidStateTransition(self._state, target)

        old_state = self._state
        self._state = target
        self._log.debug("State transition: %s → %s", old_state.name, target.name)

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def initialize(self, bus: MessageBus) -> None:
        """Configure actor and wire to MessageBus."""
        if self._state != ComponentState.PRE_INITIALIZED:
            raise RuntimeError(
                f"Cannot initialize actor in state {self._state.name}. "
                "Must be PRE_INITIALIZED."
            )

        self._bus = bus
        self._transition(ComponentState.READY)
        self._log.info("Actor %s initialized", self.name)

    async def start(self) -> None:
        """Start the actor."""
        if self._state != ComponentState.READY:
            raise RuntimeError(
                f"Cannot start actor in state {self._state.name}. Must be READY."
            )

        self._transition(ComponentState.STARTING)
        self._log.info("Actor %s starting...", self.name)

        try:
            await self.on_start()
            self._transition(ComponentState.RUNNING)
            self._log.info("Actor %s running", self.name)
        except Exception as e:
            self._log.exception("Error in on_start(), faulting")
            await self.fault(e)
            raise

    async def stop(self) -> None:
        """Stop the actor gracefully."""
        if self._state not in (ComponentState.RUNNING, ComponentState.DEGRADED):
            self._log.warning(
                "Cannot stop actor in state %s. Must be RUNNING or DEGRADED.",
                self._state.name,
            )
            return

        self._transition(ComponentState.STOPPING)
        self._log.info("Actor %s stopping...", self.name)

        try:
            # Unsubscribe from all events
            for sub_id in self._subscriptions:
                self.bus.unsubscribe(sub_id)
            self._subscriptions.clear()

            await self.on_stop()
            self._transition(ComponentState.STOPPED)
            self._log.info("Actor %s stopped", self.name)
        except Exception as e:
            self._log.exception("Error in on_stop(), faulting")
            await self.fault(e)
            raise

    async def resume(self) -> None:
        """Resume from stopped state."""
        if self._state != ComponentState.STOPPED:
            raise RuntimeError(
                f"Cannot resume actor in state {self._state.name}. Must be STOPPED."
            )

        self._transition(ComponentState.RESUMING)
        self._log.info("Actor %s resuming...", self.name)

        try:
            await self.on_resume()
            self._transition(ComponentState.RUNNING)
            self._log.info("Actor %s running", self.name)
        except Exception as e:
            self._log.exception("Error in on_resume(), faulting")
            await self.fault(e)
            raise

    async def reset(self) -> None:
        """Reset actor state."""
        if self._state != ComponentState.STOPPED:
            raise RuntimeError(
                f"Cannot reset actor in state {self._state.name}. Must be STOPPED."
            )

        self._log.info("Actor %s resetting...", self.name)
        await self.on_reset()
        self._log.info("Actor %s reset complete", self.name)

    async def degrade(self, reason: str) -> None:
        """Enter degraded mode."""
        if self._state != ComponentState.RUNNING:
            self._log.warning(
                "Cannot degrade actor in state %s. Must be RUNNING.",
                self._state.name,
            )
            return

        self._transition(ComponentState.DEGRADED)
        self._log.warning("Actor %s degraded: %s", self.name, reason)

        try:
            await self.on_degrade(reason)
        except Exception as e:
            self._log.exception("Error in on_degrade(), faulting")
            await self.fault(e)
            raise

    async def fault(self, error: Exception) -> None:
        """Enter faulted state (terminal)."""
        if self._state == ComponentState.FAULTED:
            return  # Already faulted, idempotent

        old_state = self._state
        self._state = ComponentState.FAULTED  # Direct assignment, always valid
        self._log.critical(
            "Actor %s faulted from %s: %s",
            self.name,
            old_state.name,
            error,
        )

        try:
            await self.on_fault(error)
        except Exception:
            self._log.exception("Error in on_fault() handler")

    async def dispose(self) -> None:
        """Release all resources."""
        if self._state not in (ComponentState.STOPPED, ComponentState.FAULTED):
            raise RuntimeError(
                f"Cannot dispose actor in state {self._state.name}. "
                "Must be STOPPED or FAULTED."
            )

        self._transition(ComponentState.DISPOSING)
        self._log.info("Actor %s disposing...", self.name)

        try:
            await self.on_dispose()
            self._transition(ComponentState.DISPOSED)
            self._log.info("Actor %s disposed", self.name)
        except Exception as e:
            self._log.exception("Error in on_dispose(), faulting")
            await self.fault(e)
            raise

    # -------------------------------------------------------------------------
    # Event Subscription Helpers
    # -------------------------------------------------------------------------

    async def subscribe(self, event_type: EventType) -> int:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of events to receive

        Returns:
            Subscription ID
        """
        sub_id = self.bus.subscribe(event_type, self._handle_event)
        self._subscriptions.append(sub_id)
        self._log.debug("Subscribed to %s (id=%d)", event_type.name, sub_id)
        return sub_id

    async def _handle_event(self, event: Event) -> None:
        """Internal event router."""
        if self._state not in (ComponentState.RUNNING, ComponentState.DEGRADED):
            return  # Ignore events when not running

        try:
            # Route to specific handlers based on event type
            if event.event_type == EventType.TICK:
                tick = event.payload.get("tick")
                if tick:
                    await self.on_tick(tick)
            elif event.event_type == EventType.BAR:
                bar = event.payload.get("bar")
                if bar:
                    await self.on_bar(bar)

            # Always call generic handler
            await self.on_event(event)

        except Exception:
            self._log.exception(
                "Error handling event %s (trace_id=%s)",
                event.event_type.name,
                event.trace_id,
            )

    # -------------------------------------------------------------------------
    # Event Publishing
    # -------------------------------------------------------------------------

    def publish(self, event: Event) -> bool:
        """
        Publish an event to the MessageBus.

        Args:
            event: Event to publish

        Returns:
            True if accepted, False if bus is shutting down
        """
        return self.bus.publish(event)

    def publish_event(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """
        Create and publish an event.

        Args:
            event_type: Type of event
            payload: Event payload

        Returns:
            True if accepted
        """
        event = Event.create(
            event_type=event_type,
            source=f"actor.{self.name}",
            payload=payload,
        )
        return self.publish(event)

    # -------------------------------------------------------------------------
    # Lifecycle Hooks (override in subclass)
    # -------------------------------------------------------------------------

    async def on_start(self) -> None:
        """Called when actor starts. Override to add initialization."""
        pass

    async def on_stop(self) -> None:
        """Called when actor stops. Override to add cleanup."""
        pass

    async def on_resume(self) -> None:
        """Called when resuming. Override to restore operation."""
        pass

    async def on_reset(self) -> None:
        """Called to reset state. Override to clear internal state."""
        pass

    async def on_degrade(self, reason: str) -> None:  # noqa: ARG002
        """Called when degrading. Override to adjust behavior."""
        pass

    async def on_fault(self, error: Exception) -> None:  # noqa: ARG002
        """Called on fault. Override for error handling."""
        pass

    async def on_dispose(self) -> None:
        """Called for final cleanup. Override to release resources."""
        pass

    # -------------------------------------------------------------------------
    # Event Handlers (override in subclass)
    # -------------------------------------------------------------------------

    async def on_event(self, event: Event) -> None:  # noqa: ARG002
        """Handle any event. Override for generic handling."""
        pass

    async def on_bar(self, bar: Bar) -> None:  # noqa: ARG002
        """Handle bar data. Override in subclass."""
        pass

    async def on_tick(self, tick: Tick) -> None:  # noqa: ARG002
        """Handle tick data. Override in subclass."""
        pass

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> BaseActor:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._state in (ComponentState.RUNNING, ComponentState.DEGRADED):
            await self.stop()
        if self._state == ComponentState.STOPPED:
            await self.dispose()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name!r}, state={self._state.name})"

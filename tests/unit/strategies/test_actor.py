"""Unit tests for Actor lifecycle and state machine."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus
from libra.strategies.actor import (
    BaseActor,
    ComponentState,
    InvalidStateTransition,
    _VALID_TRANSITIONS,
)


# =============================================================================
# Test Actor Implementation
# =============================================================================


class SampleActor(BaseActor):
    """Concrete Actor implementation for testing."""

    def __init__(self, name: str = "test_actor") -> None:
        super().__init__()
        self._name = name
        self.on_start_called = False
        self.on_stop_called = False
        self.on_resume_called = False
        self.on_reset_called = False
        self.on_degrade_called = False
        self.on_degrade_reason: str | None = None
        self.on_fault_called = False
        self.on_fault_error: Exception | None = None
        self.on_dispose_called = False
        self.events_received: list[Event] = []

    @property
    def name(self) -> str:
        return self._name

    async def on_start(self) -> None:
        self.on_start_called = True

    async def on_stop(self) -> None:
        self.on_stop_called = True

    async def on_resume(self) -> None:
        self.on_resume_called = True

    async def on_reset(self) -> None:
        self.on_reset_called = True

    async def on_degrade(self, reason: str) -> None:
        self.on_degrade_called = True
        self.on_degrade_reason = reason

    async def on_fault(self, error: Exception) -> None:
        self.on_fault_called = True
        self.on_fault_error = error

    async def on_dispose(self) -> None:
        self.on_dispose_called = True

    async def on_event(self, event: Event) -> None:
        self.events_received.append(event)


class FailingActor(BaseActor):
    """Actor that fails in lifecycle hooks for testing."""

    def __init__(self, fail_on: str = "start") -> None:
        super().__init__()
        self.fail_on = fail_on

    @property
    def name(self) -> str:
        return "failing_actor"

    async def on_start(self) -> None:
        if self.fail_on == "start":
            raise RuntimeError("Intentional start failure")

    async def on_stop(self) -> None:
        if self.fail_on == "stop":
            raise RuntimeError("Intentional stop failure")

    async def on_resume(self) -> None:
        if self.fail_on == "resume":
            raise RuntimeError("Intentional resume failure")

    async def on_degrade(self, reason: str) -> None:
        if self.fail_on == "degrade":
            raise RuntimeError("Intentional degrade failure")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def message_bus() -> MessageBus:
    """Create a MessageBus instance."""
    return MessageBus()


@pytest.fixture
def actor() -> SampleActor:
    """Create a test actor."""
    return SampleActor()


@pytest.fixture
async def initialized_actor(actor: SampleActor, message_bus: MessageBus) -> SampleActor:
    """Create an initialized actor."""
    await actor.initialize(message_bus)
    return actor


@pytest.fixture
async def running_actor(initialized_actor: SampleActor) -> SampleActor:
    """Create a running actor."""
    await initialized_actor.start()
    return initialized_actor


# =============================================================================
# Component State Tests
# =============================================================================


class TestComponentState:
    """Tests for ComponentState enum."""

    def test_state_values(self) -> None:
        """Test that states have expected integer values."""
        assert ComponentState.PRE_INITIALIZED == 0
        assert ComponentState.READY == 1
        assert ComponentState.STARTING == 2
        assert ComponentState.RUNNING == 3
        assert ComponentState.RESUMING == 4
        assert ComponentState.DEGRADED == 5
        assert ComponentState.STOPPING == 6
        assert ComponentState.STOPPED == 7
        assert ComponentState.DISPOSING == 8
        assert ComponentState.DISPOSED == 9
        assert ComponentState.FAULTED == 10

    def test_valid_transitions_from_pre_initialized(self) -> None:
        """Test valid transitions from PRE_INITIALIZED."""
        valid = _VALID_TRANSITIONS[ComponentState.PRE_INITIALIZED]
        assert ComponentState.READY in valid
        assert ComponentState.FAULTED in valid
        assert len(valid) == 2

    def test_valid_transitions_from_ready(self) -> None:
        """Test valid transitions from READY."""
        valid = _VALID_TRANSITIONS[ComponentState.READY]
        assert ComponentState.STARTING in valid
        assert ComponentState.DISPOSING in valid
        assert ComponentState.FAULTED in valid

    def test_valid_transitions_from_running(self) -> None:
        """Test valid transitions from RUNNING."""
        valid = _VALID_TRANSITIONS[ComponentState.RUNNING]
        assert ComponentState.DEGRADED in valid
        assert ComponentState.STOPPING in valid
        assert ComponentState.FAULTED in valid

    def test_terminal_states_have_no_transitions(self) -> None:
        """Test that terminal states have no valid transitions."""
        assert len(_VALID_TRANSITIONS[ComponentState.DISPOSED]) == 0
        assert len(_VALID_TRANSITIONS[ComponentState.FAULTED]) == 0


# =============================================================================
# Actor Initialization Tests
# =============================================================================


class TestActorInitialization:
    """Tests for Actor initialization."""

    def test_initial_state(self, actor: SampleActor) -> None:
        """Test actor starts in PRE_INITIALIZED state."""
        assert actor.state == ComponentState.PRE_INITIALIZED
        assert not actor.is_running
        assert not actor.is_degraded
        assert not actor.is_stopped

    @pytest.mark.asyncio
    async def test_initialize(self, actor: SampleActor, message_bus: MessageBus) -> None:
        """Test actor initialization."""
        await actor.initialize(message_bus)
        assert actor.state == ComponentState.READY
        assert actor.bus == message_bus

    @pytest.mark.asyncio
    async def test_initialize_twice_fails(
        self, initialized_actor: SampleActor, message_bus: MessageBus
    ) -> None:
        """Test that initializing twice raises error."""
        with pytest.raises(RuntimeError, match="Cannot initialize"):
            await initialized_actor.initialize(message_bus)

    @pytest.mark.asyncio
    async def test_bus_access_before_init_fails(self, actor: SampleActor) -> None:
        """Test that accessing bus before init raises error."""
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = actor.bus


# =============================================================================
# Actor Lifecycle Tests
# =============================================================================


class TestActorLifecycle:
    """Tests for Actor lifecycle transitions."""

    @pytest.mark.asyncio
    async def test_start(self, initialized_actor: SampleActor) -> None:
        """Test actor start."""
        await initialized_actor.start()
        assert initialized_actor.state == ComponentState.RUNNING
        assert initialized_actor.is_running
        assert initialized_actor.on_start_called

    @pytest.mark.asyncio
    async def test_start_not_ready_fails(self, actor: SampleActor) -> None:
        """Test that starting from non-READY state fails."""
        with pytest.raises(RuntimeError, match="Must be READY"):
            await actor.start()

    @pytest.mark.asyncio
    async def test_stop(self, running_actor: SampleActor) -> None:
        """Test actor stop."""
        await running_actor.stop()
        assert running_actor.state == ComponentState.STOPPED
        assert running_actor.is_stopped
        assert running_actor.on_stop_called

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_ignored(
        self, initialized_actor: SampleActor
    ) -> None:
        """Test that stop when not running is a no-op."""
        await initialized_actor.stop()
        assert initialized_actor.state == ComponentState.READY  # Unchanged
        assert not initialized_actor.on_stop_called

    @pytest.mark.asyncio
    async def test_resume(self, running_actor: SampleActor) -> None:
        """Test actor resume."""
        await running_actor.stop()
        assert running_actor.state == ComponentState.STOPPED

        await running_actor.resume()
        assert running_actor.state == ComponentState.RUNNING
        assert running_actor.on_resume_called

    @pytest.mark.asyncio
    async def test_resume_not_stopped_fails(self, running_actor: SampleActor) -> None:
        """Test that resume from non-STOPPED state fails."""
        with pytest.raises(RuntimeError, match="Must be STOPPED"):
            await running_actor.resume()

    @pytest.mark.asyncio
    async def test_reset(self, running_actor: SampleActor) -> None:
        """Test actor reset."""
        await running_actor.stop()
        await running_actor.reset()
        assert running_actor.on_reset_called

    @pytest.mark.asyncio
    async def test_reset_not_stopped_fails(self, running_actor: SampleActor) -> None:
        """Test that reset from non-STOPPED state fails."""
        with pytest.raises(RuntimeError, match="Must be STOPPED"):
            await running_actor.reset()

    @pytest.mark.asyncio
    async def test_degrade(self, running_actor: SampleActor) -> None:
        """Test actor degradation."""
        await running_actor.degrade("test reason")
        assert running_actor.state == ComponentState.DEGRADED
        assert running_actor.is_degraded
        assert running_actor.on_degrade_called
        assert running_actor.on_degrade_reason == "test reason"

    @pytest.mark.asyncio
    async def test_degrade_not_running_is_ignored(
        self, initialized_actor: SampleActor
    ) -> None:
        """Test that degrade when not running is a no-op."""
        await initialized_actor.degrade("test")
        assert initialized_actor.state == ComponentState.READY
        assert not initialized_actor.on_degrade_called

    @pytest.mark.asyncio
    async def test_stop_from_degraded(self, running_actor: SampleActor) -> None:
        """Test stopping from degraded state."""
        await running_actor.degrade("test")
        await running_actor.stop()
        assert running_actor.state == ComponentState.STOPPED

    @pytest.mark.asyncio
    async def test_dispose(self, running_actor: SampleActor) -> None:
        """Test actor disposal."""
        await running_actor.stop()
        await running_actor.dispose()
        assert running_actor.state == ComponentState.DISPOSED
        assert running_actor.on_dispose_called

    @pytest.mark.asyncio
    async def test_dispose_not_stopped_fails(self, running_actor: SampleActor) -> None:
        """Test that dispose from non-STOPPED state fails."""
        with pytest.raises(RuntimeError, match="Must be STOPPED"):
            await running_actor.dispose()


# =============================================================================
# Fault Handling Tests
# =============================================================================


class TestActorFaultHandling:
    """Tests for Actor fault handling."""

    @pytest.mark.asyncio
    async def test_fault(self, running_actor: SampleActor) -> None:
        """Test actor fault."""
        error = RuntimeError("Test error")
        await running_actor.fault(error)
        assert running_actor.state == ComponentState.FAULTED
        assert running_actor.on_fault_called
        assert running_actor.on_fault_error == error

    @pytest.mark.asyncio
    async def test_fault_is_idempotent(self, running_actor: SampleActor) -> None:
        """Test that fault is idempotent."""
        error1 = RuntimeError("First error")
        error2 = RuntimeError("Second error")

        await running_actor.fault(error1)
        await running_actor.fault(error2)

        assert running_actor.state == ComponentState.FAULTED
        # Only first error should be recorded
        assert running_actor.on_fault_error == error1

    @pytest.mark.asyncio
    async def test_fault_from_any_state(self, actor: SampleActor, message_bus: MessageBus) -> None:
        """Test that fault can happen from any state."""
        # From PRE_INITIALIZED
        await actor.fault(RuntimeError("test"))
        assert actor.state == ComponentState.FAULTED

    @pytest.mark.asyncio
    async def test_start_failure_causes_fault(self, message_bus: MessageBus) -> None:
        """Test that failure in on_start causes fault."""
        actor = FailingActor(fail_on="start")
        await actor.initialize(message_bus)

        with pytest.raises(RuntimeError, match="Intentional start failure"):
            await actor.start()

        assert actor.state == ComponentState.FAULTED

    @pytest.mark.asyncio
    async def test_stop_failure_causes_fault(self, message_bus: MessageBus) -> None:
        """Test that failure in on_stop causes fault."""
        actor = FailingActor(fail_on="stop")
        await actor.initialize(message_bus)
        await actor.start()

        with pytest.raises(RuntimeError, match="Intentional stop failure"):
            await actor.stop()

        assert actor.state == ComponentState.FAULTED


# =============================================================================
# Invalid Transition Tests
# =============================================================================


class TestInvalidStateTransition:
    """Tests for InvalidStateTransition exception."""

    def test_exception_message(self) -> None:
        """Test exception message format."""
        exc = InvalidStateTransition(ComponentState.RUNNING, ComponentState.READY)
        assert "RUNNING" in str(exc)
        assert "READY" in str(exc)
        assert exc.current == ComponentState.RUNNING
        assert exc.target == ComponentState.READY


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestActorContextManager:
    """Tests for Actor context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager(self, message_bus: MessageBus) -> None:
        """Test actor as context manager."""
        actor = SampleActor()
        await actor.initialize(message_bus)

        async with actor:
            assert actor.state == ComponentState.RUNNING
            assert actor.on_start_called

        assert actor.state == ComponentState.DISPOSED
        assert actor.on_stop_called
        assert actor.on_dispose_called


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestActorEventSubscription:
    """Tests for Actor event subscription."""

    @pytest.mark.asyncio
    async def test_subscribe(self, running_actor: SampleActor) -> None:
        """Test subscribing to events."""
        sub_id = await running_actor.subscribe(EventType.TICK)
        assert sub_id > 0
        assert len(running_actor._subscriptions) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_on_stop(
        self, running_actor: SampleActor, message_bus: MessageBus
    ) -> None:
        """Test that subscriptions are removed on stop."""
        await running_actor.subscribe(EventType.TICK)
        assert len(running_actor._subscriptions) == 1

        await running_actor.stop()
        assert len(running_actor._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_publish_event(self, running_actor: SampleActor) -> None:
        """Test publishing events."""
        result = running_actor.publish_event(EventType.SIGNAL, {"test": "data"})
        assert result is True


# =============================================================================
# String Representation Tests
# =============================================================================


class TestActorRepr:
    """Tests for Actor string representation."""

    def test_repr(self, actor: SampleActor) -> None:
        """Test __repr__ format."""
        repr_str = repr(actor)
        assert "SampleActor" in repr_str
        assert "test_actor" in repr_str
        assert "PRE_INITIALIZED" in repr_str

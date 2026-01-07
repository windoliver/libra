"""Tests for the TradingKernel."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.core.kernel import KernelConfig, KernelState, TradingKernel
from libra.core.message_bus import MessageBus
from libra.strategies.actor import BaseActor, ComponentState


# =============================================================================
# Test Fixtures
# =============================================================================


class MockActor(BaseActor):
    """Mock actor for testing."""

    def __init__(self, name: str = "mock_actor") -> None:
        super().__init__()
        self._name = name
        self.start_called = False
        self.stop_called = False

    @property
    def name(self) -> str:
        return self._name

    async def on_start(self) -> None:
        self.start_called = True

    async def on_stop(self) -> None:
        self.stop_called = True


class MockStrategy(BaseActor):
    """Mock strategy for testing (simplified, extends BaseActor)."""

    def __init__(self, name: str = "mock_strategy") -> None:
        super().__init__()
        self._name = name
        self.start_called = False
        self.stop_called = False

    @property
    def name(self) -> str:
        return self._name

    async def on_start(self) -> None:
        self.start_called = True

    async def on_stop(self) -> None:
        self.stop_called = True


class MockGateway:
    """Mock gateway for testing."""

    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True


# =============================================================================
# KernelConfig Tests
# =============================================================================


class TestKernelConfig:
    """Tests for KernelConfig."""

    def test_default_config(self) -> None:
        """Default config should have reasonable values."""
        config = KernelConfig()

        assert config.environment == "sandbox"
        assert config.load_state is False
        assert config.save_state is True
        assert config.startup_timeout == 30.0
        assert config.shutdown_timeout == 10.0
        assert len(config.instance_id) == 12

    def test_custom_config(self) -> None:
        """Custom config values should be set correctly."""
        config = KernelConfig(
            instance_id="test123",
            environment="live",
            load_state=True,
            save_state=True,
            startup_timeout=60.0,
            shutdown_timeout=20.0,
        )

        assert config.instance_id == "test123"
        assert config.environment == "live"
        assert config.load_state is True
        assert config.startup_timeout == 60.0
        assert config.shutdown_timeout == 20.0

    def test_backtest_environment(self) -> None:
        """Backtest environment should be valid."""
        config = KernelConfig(environment="backtest")
        assert config.environment == "backtest"

    def test_invalid_startup_timeout(self) -> None:
        """Invalid startup timeout should raise ValueError."""
        with pytest.raises(ValueError, match="startup_timeout must be positive"):
            KernelConfig(startup_timeout=0)

        with pytest.raises(ValueError, match="startup_timeout must be positive"):
            KernelConfig(startup_timeout=-1)

    def test_invalid_shutdown_timeout(self) -> None:
        """Invalid shutdown timeout should raise ValueError."""
        with pytest.raises(ValueError, match="shutdown_timeout must be positive"):
            KernelConfig(shutdown_timeout=0)


# =============================================================================
# KernelState Tests
# =============================================================================


class TestKernelState:
    """Tests for KernelState enum."""

    def test_state_values(self) -> None:
        """States should have correct integer values."""
        assert KernelState.INITIALIZED == 0
        assert KernelState.STARTING == 1
        assert KernelState.RUNNING == 2
        assert KernelState.STOPPING == 3
        assert KernelState.STOPPED == 4
        assert KernelState.DISPOSING == 5
        assert KernelState.DISPOSED == 6

    def test_state_ordering(self) -> None:
        """States should be ordered correctly."""
        assert KernelState.INITIALIZED < KernelState.RUNNING
        assert KernelState.RUNNING < KernelState.STOPPED


# =============================================================================
# TradingKernel Creation Tests
# =============================================================================


class TestKernelCreation:
    """Tests for TradingKernel creation."""

    def test_create_with_defaults(self) -> None:
        """Kernel should create with default config."""
        kernel = TradingKernel()

        assert kernel.state == KernelState.INITIALIZED
        assert kernel.environment == "sandbox"
        assert kernel.is_running is False
        assert kernel.is_stopped is False

    def test_create_with_config(self) -> None:
        """Kernel should use provided config."""
        config = KernelConfig(
            instance_id="test_kernel",
            environment="live",
        )
        kernel = TradingKernel(config)

        assert kernel.instance_id == "test_kernel"
        assert kernel.environment == "live"

    def test_infrastructure_created(self) -> None:
        """Core infrastructure should be created on init."""
        kernel = TradingKernel()

        assert kernel.bus is not None
        assert isinstance(kernel.bus, MessageBus)
        assert kernel.cache is not None
        assert kernel.clock is not None

    def test_backtest_clock_type(self) -> None:
        """Backtest environment should use backtest clock."""
        config = KernelConfig(environment="backtest")
        kernel = TradingKernel(config)

        assert kernel.clock.is_backtest

    def test_live_clock_type(self) -> None:
        """Live environment should use live clock."""
        config = KernelConfig(environment="live")
        kernel = TradingKernel(config)

        assert kernel.clock.is_live

    def test_timestamps_set(self) -> None:
        """Created timestamp should be set."""
        kernel = TradingKernel()

        assert kernel.ts_created > 0
        assert kernel.ts_started is None
        assert kernel.ts_stopped is None


# =============================================================================
# Component Registration Tests
# =============================================================================


class TestComponentRegistration:
    """Tests for component registration."""

    def test_add_actor(self) -> None:
        """Actor should be registered."""
        kernel = TradingKernel()
        actor = MockActor("test_actor")

        kernel.add_actor(actor)

        assert len(kernel.actors) == 1
        assert kernel.get_actor("test_actor") is actor

    def test_add_multiple_actors(self) -> None:
        """Multiple actors should be registered."""
        kernel = TradingKernel()
        actor1 = MockActor("actor1")
        actor2 = MockActor("actor2")

        kernel.add_actor(actor1)
        kernel.add_actor(actor2)

        assert len(kernel.actors) == 2
        assert kernel.get_actor("actor1") is actor1
        assert kernel.get_actor("actor2") is actor2

    def test_add_duplicate_actor_name(self) -> None:
        """Duplicate actor name should raise ValueError."""
        kernel = TradingKernel()
        actor1 = MockActor("same_name")
        actor2 = MockActor("same_name")

        kernel.add_actor(actor1)

        with pytest.raises(ValueError, match="already registered"):
            kernel.add_actor(actor2)

    def test_add_strategy(self) -> None:
        """Strategy should be registered."""
        kernel = TradingKernel()
        strategy = MockStrategy("test_strategy")

        kernel.add_strategy(strategy)

        assert len(kernel.strategies) == 1
        assert kernel.get_strategy("test_strategy") is strategy

    def test_add_duplicate_strategy_name(self) -> None:
        """Duplicate strategy name should raise ValueError."""
        kernel = TradingKernel()
        strategy1 = MockStrategy("same_name")
        strategy2 = MockStrategy("same_name")

        kernel.add_strategy(strategy1)

        with pytest.raises(ValueError, match="already registered"):
            kernel.add_strategy(strategy2)

    def test_set_gateway(self) -> None:
        """Gateway should be set."""
        kernel = TradingKernel()
        gateway = MockGateway()

        kernel.set_gateway(gateway)

        assert kernel.gateway is gateway

    def test_get_nonexistent_actor(self) -> None:
        """Getting nonexistent actor should return None."""
        kernel = TradingKernel()
        assert kernel.get_actor("nonexistent") is None

    def test_get_nonexistent_strategy(self) -> None:
        """Getting nonexistent strategy should return None."""
        kernel = TradingKernel()
        assert kernel.get_strategy("nonexistent") is None


# =============================================================================
# Startup Tests
# =============================================================================


class TestKernelStartup:
    """Tests for kernel startup."""

    @pytest.mark.asyncio
    async def test_start_sets_running_state(self) -> None:
        """Start should transition to RUNNING state."""
        kernel = TradingKernel()

        await kernel.start_async()

        assert kernel.state == KernelState.RUNNING
        assert kernel.is_running is True
        assert kernel.ts_started is not None

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_start_initializes_actors(self) -> None:
        """Start should initialize and start all actors."""
        kernel = TradingKernel()
        actor = MockActor("test")
        kernel.add_actor(actor)

        await kernel.start_async()

        assert actor.state == ComponentState.RUNNING
        assert actor.start_called is True

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_start_initializes_strategies(self) -> None:
        """Start should initialize and start all strategies."""
        kernel = TradingKernel()
        strategy = MockStrategy("test")
        kernel.add_strategy(strategy)

        await kernel.start_async()

        assert strategy.state == ComponentState.RUNNING
        assert strategy.start_called is True

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_start_connects_gateway(self) -> None:
        """Start should connect gateway if set."""
        kernel = TradingKernel()
        gateway = MockGateway()
        kernel.set_gateway(gateway)

        await kernel.start_async()

        assert gateway.connected is True

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_start_starts_bus(self) -> None:
        """Start should start the message bus."""
        kernel = TradingKernel()

        await kernel.start_async()

        assert kernel.bus.is_running is True

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_cannot_start_twice(self) -> None:
        """Starting an already running kernel should fail."""
        kernel = TradingKernel()
        await kernel.start_async()

        with pytest.raises(RuntimeError, match="Cannot start kernel"):
            await kernel.start_async()

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_cannot_add_actor_after_start(self) -> None:
        """Cannot add actor after kernel is started."""
        kernel = TradingKernel()
        await kernel.start_async()

        with pytest.raises(RuntimeError, match="Cannot add actor"):
            kernel.add_actor(MockActor())

        await kernel.stop_async()


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestKernelShutdown:
    """Tests for kernel shutdown."""

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_state(self) -> None:
        """Stop should transition to STOPPED state."""
        kernel = TradingKernel()
        await kernel.start_async()

        await kernel.stop_async()

        assert kernel.state == KernelState.STOPPED
        assert kernel.is_stopped is True
        assert kernel.ts_stopped is not None

    @pytest.mark.asyncio
    async def test_stop_stops_strategies_first(self) -> None:
        """Strategies should be stopped before actors."""
        kernel = TradingKernel()
        actor = MockActor("actor")
        strategy = MockStrategy("strategy")

        kernel.add_actor(actor)
        kernel.add_strategy(strategy)

        await kernel.start_async()
        await kernel.stop_async()

        assert strategy.stop_called is True
        assert actor.stop_called is True

    @pytest.mark.asyncio
    async def test_stop_disconnects_gateway(self) -> None:
        """Stop should disconnect gateway if set."""
        kernel = TradingKernel()
        gateway = MockGateway()
        kernel.set_gateway(gateway)

        await kernel.start_async()
        await kernel.stop_async()

        assert gateway.disconnected is True

    @pytest.mark.asyncio
    async def test_stop_stops_bus(self) -> None:
        """Stop should stop the message bus."""
        kernel = TradingKernel()

        await kernel.start_async()
        await kernel.stop_async()

        assert kernel.bus.is_running is False

    @pytest.mark.asyncio
    async def test_stop_idempotent(self) -> None:
        """Stopping multiple times should be safe."""
        kernel = TradingKernel()
        await kernel.start_async()

        await kernel.stop_async()
        await kernel.stop_async()  # Should not raise

        assert kernel.state == KernelState.STOPPED


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_starts_and_stops(self) -> None:
        """Context manager should start and stop kernel."""
        config = KernelConfig()

        async with TradingKernel(config) as kernel:
            assert kernel.is_running is True

        assert kernel.state == KernelState.DISPOSED

    @pytest.mark.asyncio
    async def test_context_manager_with_actors(self) -> None:
        """Context manager should manage actor lifecycle."""
        actor = MockActor("test")

        kernel = TradingKernel()
        kernel.add_actor(actor)

        async with kernel:
            assert actor.state == ComponentState.RUNNING

        assert actor.state == ComponentState.DISPOSED

    @pytest.mark.asyncio
    async def test_context_manager_handles_exception(self) -> None:
        """Context manager should stop on exception."""
        kernel = TradingKernel()

        with pytest.raises(ValueError):
            async with kernel:
                assert kernel.is_running is True
                raise ValueError("Test error")

        assert kernel.state == KernelState.DISPOSED


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_running(self) -> None:
        """Health check should return status when running."""
        kernel = TradingKernel()
        await kernel.start_async()

        health = kernel.health_check()

        assert health["kernel"]["state"] == "RUNNING"
        assert health["bus"]["running"] is True
        assert "cache" in health
        assert "clock" in health

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_health_check_includes_actors(self) -> None:
        """Health check should include actor status."""
        kernel = TradingKernel()
        actor = MockActor("test_actor")
        kernel.add_actor(actor)

        await kernel.start_async()

        health = kernel.health_check()

        assert "test_actor" in health["actors"]
        assert health["actors"]["test_actor"]["state"] == "RUNNING"

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_is_healthy_when_running(self) -> None:
        """is_healthy should return True when all components running."""
        kernel = TradingKernel()
        actor = MockActor("test")
        kernel.add_actor(actor)

        await kernel.start_async()

        assert kernel.is_healthy() is True

        await kernel.stop_async()

    def test_is_healthy_when_not_running(self) -> None:
        """is_healthy should return False when not running."""
        kernel = TradingKernel()
        assert kernel.is_healthy() is False


# =============================================================================
# Dispose Tests
# =============================================================================


class TestDispose:
    """Tests for kernel disposal."""

    @pytest.mark.asyncio
    async def test_dispose_releases_resources(self) -> None:
        """Dispose should release all resources."""
        kernel = TradingKernel()
        await kernel.start_async()
        await kernel.stop_async()

        await kernel.dispose()

        assert kernel.state == KernelState.DISPOSED

    @pytest.mark.asyncio
    async def test_dispose_disposes_actors(self) -> None:
        """Dispose should dispose all actors."""
        kernel = TradingKernel()
        actor = MockActor("test")
        kernel.add_actor(actor)

        await kernel.start_async()
        await kernel.stop_async()
        await kernel.dispose()

        assert actor.state == ComponentState.DISPOSED

    @pytest.mark.asyncio
    async def test_dispose_idempotent(self) -> None:
        """Disposing multiple times should be safe."""
        kernel = TradingKernel()
        await kernel.start_async()
        await kernel.stop_async()

        await kernel.dispose()
        await kernel.dispose()  # Should not raise

        assert kernel.state == KernelState.DISPOSED

    @pytest.mark.asyncio
    async def test_dispose_stops_if_running(self) -> None:
        """Dispose should stop kernel if still running."""
        kernel = TradingKernel()
        await kernel.start_async()

        await kernel.dispose()

        assert kernel.state == KernelState.DISPOSED


# =============================================================================
# Representation Tests
# =============================================================================


class TestRepresentation:
    """Tests for string representation."""

    def test_repr(self) -> None:
        """Repr should include key information."""
        config = KernelConfig(instance_id="test123", environment="live")
        kernel = TradingKernel(config)

        repr_str = repr(kernel)

        assert "test123" in repr_str
        assert "live" in repr_str
        assert "INITIALIZED" in repr_str

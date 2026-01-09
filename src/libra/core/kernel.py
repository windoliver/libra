"""
TradingKernel: Central orchestrator for all trading components.

Following NautilusTrader's NautilusKernel pattern, this provides:
- Single entry point for system startup/shutdown
- Correct component initialization ordering
- Graceful shutdown with resource cleanup
- Unified configuration
- Environment-aware operation (backtest/sandbox/live)

Design references:
- NautilusTrader: https://nautilustrader.io/docs/latest/concepts/architecture
- Issue: https://github.com/windoliver/libra/issues/35
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from libra.core.cache import Cache
from libra.core.clock import Clock, ClockType
from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus, MessageBusConfig
from libra.plugins.loader import (
    discover_gateways,
    discover_strategies,
    list_strategy_plugins,
)
from libra.strategies.actor import BaseActor, ComponentState


if TYPE_CHECKING:
    from libra.clients.data_client import DataClient, Instrument
    from libra.clients.execution_client import ExecutionClient
    from libra.gateways.protocol import Gateway, Order, OrderResult
    from libra.risk.engine import RiskEngine
    from libra.risk.manager import RiskManager
    from libra.strategies.strategy import BaseStrategy


logger = logging.getLogger(__name__)


# =============================================================================
# Kernel State
# =============================================================================


class KernelState(IntEnum):
    """
    TradingKernel lifecycle states.

    State machine transitions:
        INITIALIZED → STARTING → RUNNING → STOPPING → STOPPED → DISPOSING → DISPOSED

    Unlike components, kernel cannot be DEGRADED or FAULTED - if kernel fails,
    the system should terminate (crash-only design for data integrity).
    """

    INITIALIZED = 0  # Created but not started
    STARTING = 1  # start() in progress
    RUNNING = 2  # Normal operation
    STOPPING = 3  # stop() in progress
    STOPPED = 4  # Gracefully stopped
    DISPOSING = 5  # dispose() in progress
    DISPOSED = 6  # Resources released


# =============================================================================
# Kernel Configuration
# =============================================================================


@dataclass
class KernelConfig:
    """
    Configuration for TradingKernel.

    Attributes:
        instance_id: Unique identifier for this kernel instance
        environment: Operating environment (backtest/sandbox/live)
        load_state: Load actor/strategy state on startup
        save_state: Save actor/strategy state on shutdown
        bus_config: MessageBus configuration
        startup_timeout: Max time to wait for component startup (seconds)
        shutdown_timeout: Max time to wait for graceful shutdown (seconds)
    """

    # Identity
    instance_id: str = field(default_factory=lambda: uuid4().hex[:12])

    # Environment
    environment: Literal["backtest", "sandbox", "live"] = "sandbox"

    # State persistence
    load_state: bool = False
    save_state: bool = True

    # Component configs
    bus_config: MessageBusConfig = field(default_factory=MessageBusConfig)

    # Timeouts
    startup_timeout: float = 30.0
    shutdown_timeout: float = 10.0

    # Cache settings
    cache_max_bars: int = 1000
    cache_max_orders: int = 10000

    # Plugin discovery (Issue #29)
    discover_plugins: bool = True  # Discover registered plugins at startup
    log_plugins: bool = True  # Log discovered plugins

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.environment == "live" and not self.save_state:
            logger.warning(
                "save_state=False in LIVE environment - state will not persist on shutdown"
            )

        if self.startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")

        if self.shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be positive")


# =============================================================================
# TradingKernel
# =============================================================================


class TradingKernel:
    """
    Central orchestrator for all trading components.

    The TradingKernel is the single entry point for:
    - System initialization and startup
    - Component lifecycle management
    - Graceful shutdown and cleanup
    - State persistence

    Components are started in dependency order:
    1. MessageBus (communication backbone)
    2. Cache (shared state)
    3. Clock (time services)
    4. RiskManager (safety checks)
    5. Gateway (exchange connection)
    6. Actors (non-trading components)
    7. Strategies (trading logic)

    Shutdown occurs in reverse order for clean resource release.

    Example:
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        # Register components
        kernel.add_strategy(MyStrategy(gateway))

        # Run
        async with kernel:
            await asyncio.sleep(3600)  # Run for 1 hour

        # Or manual control
        await kernel.start_async()
        try:
            await asyncio.sleep(3600)
        finally:
            await kernel.stop_async()
    """

    def __init__(self, config: KernelConfig | None = None) -> None:
        """
        Initialize TradingKernel.

        Args:
            config: Kernel configuration (uses defaults if not provided)
        """
        self._config = config or KernelConfig()
        self._state = KernelState.INITIALIZED

        # Timestamps
        self._ts_created = time.time_ns()
        self._ts_started: int | None = None
        self._ts_stopped: int | None = None

        # Core infrastructure (created immediately)
        self._bus = MessageBus(self._config.bus_config)
        self._cache = Cache(
            max_bars=self._config.cache_max_bars,
            max_orders=self._config.cache_max_orders,
        )
        self._clock = Clock(
            clock_type=ClockType.BACKTEST
            if self._config.environment == "backtest"
            else ClockType.LIVE
        )

        # Optional components (set by user)
        self._gateway: Gateway | None = None
        self._risk_manager: RiskManager | None = None  # Legacy
        self._risk_engine: RiskEngine | None = None  # New (Issue #34)

        # New client architecture (Issue #33)
        self._data_client: DataClient | None = None
        self._execution_client: ExecutionClient | None = None

        # Actors and strategies (registered by user)
        self._actors: list[BaseActor] = []
        self._strategies: list[BaseStrategy] = []

        # Internal tasks
        self._bus_task: asyncio.Task[None] | None = None

        logger.info(
            "TradingKernel created: instance_id=%s environment=%s",
            self._config.instance_id,
            self._config.environment,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> KernelConfig:
        """Kernel configuration."""
        return self._config

    @property
    def state(self) -> KernelState:
        """Current kernel state."""
        return self._state

    @property
    def instance_id(self) -> str:
        """Unique instance identifier."""
        return self._config.instance_id

    @property
    def environment(self) -> str:
        """Operating environment (backtest/sandbox/live)."""
        return self._config.environment

    @property
    def is_running(self) -> bool:
        """Check if kernel is running."""
        return self._state == KernelState.RUNNING

    @property
    def is_stopped(self) -> bool:
        """Check if kernel is stopped."""
        return self._state in (KernelState.STOPPED, KernelState.DISPOSED)

    # =========================================================================
    # Core Infrastructure Access
    # =========================================================================

    @property
    def bus(self) -> MessageBus:
        """MessageBus for event pub/sub."""
        return self._bus

    @property
    def cache(self) -> Cache:
        """Shared state cache."""
        return self._cache

    @property
    def clock(self) -> Clock:
        """Clock for time services."""
        return self._clock

    @property
    def gateway(self) -> Gateway | None:
        """Exchange gateway (if set) - legacy, prefer data_client/execution_client."""
        return self._gateway

    @property
    def risk_manager(self) -> RiskManager | None:
        """Risk manager (if set) - legacy, prefer risk_engine."""
        return self._risk_manager

    @property
    def risk_engine(self) -> RiskEngine | None:
        """Risk engine for pre-trade validation (if set)."""
        return self._risk_engine

    @property
    def data_client(self) -> DataClient | None:
        """Market data client (if set)."""
        return self._data_client

    @property
    def execution_client(self) -> ExecutionClient | None:
        """Order execution client (if set)."""
        return self._execution_client

    # =========================================================================
    # Timestamps
    # =========================================================================

    @property
    def ts_created(self) -> int:
        """Timestamp (ns) when kernel was created."""
        return self._ts_created

    @property
    def ts_started(self) -> int | None:
        """Timestamp (ns) when kernel was last started."""
        return self._ts_started

    @property
    def ts_stopped(self) -> int | None:
        """Timestamp (ns) when kernel was last stopped."""
        return self._ts_stopped

    # =========================================================================
    # Component Registration
    # =========================================================================

    def set_gateway(self, gateway: Gateway) -> None:
        """
        Set the exchange gateway.

        Args:
            gateway: Gateway for order execution

        Raises:
            RuntimeError: If kernel is already running
        """
        if self._state != KernelState.INITIALIZED:
            raise RuntimeError(
                f"Cannot set gateway in state {self._state.name}. "
                "Must be INITIALIZED."
            )
        self._gateway = gateway
        logger.info("Gateway set: %s", type(gateway).__name__)

    def set_risk_manager(self, risk_manager: RiskManager) -> None:
        """
        Set the risk manager (legacy).

        Prefer set_risk_engine() for new code.

        Args:
            risk_manager: Risk manager for pre-trade validation

        Raises:
            RuntimeError: If kernel is already running
        """
        if self._state != KernelState.INITIALIZED:
            raise RuntimeError(
                f"Cannot set risk manager in state {self._state.name}. "
                "Must be INITIALIZED."
            )
        self._risk_manager = risk_manager
        logger.info("RiskManager set (legacy)")

    def set_risk_engine(self, risk_engine: RiskEngine) -> None:
        """
        Set the risk engine for pre-trade validation.

        All orders submitted through submit_order() will be validated
        by the risk engine before being sent to the execution client.

        Args:
            risk_engine: RiskEngine for pre-trade validation

        Raises:
            RuntimeError: If kernel is already running
        """
        if self._state != KernelState.INITIALIZED:
            raise RuntimeError(
                f"Cannot set risk engine in state {self._state.name}. "
                "Must be INITIALIZED."
            )
        self._risk_engine = risk_engine
        # Wire up message bus for event publishing
        if risk_engine.bus is None:
            risk_engine.bus = self._bus
        logger.info("RiskEngine set")

    def set_data_client(self, client: DataClient) -> None:
        """
        Set the market data client.

        Args:
            client: DataClient for market data subscriptions and requests

        Raises:
            RuntimeError: If kernel is already running
        """
        if self._state != KernelState.INITIALIZED:
            raise RuntimeError(
                f"Cannot set data client in state {self._state.name}. "
                "Must be INITIALIZED."
            )
        self._data_client = client
        logger.info("DataClient set: %s", client.name)

    def set_execution_client(self, client: ExecutionClient) -> None:
        """
        Set the order execution client.

        Args:
            client: ExecutionClient for order submission and management

        Raises:
            RuntimeError: If kernel is already running
        """
        if self._state != KernelState.INITIALIZED:
            raise RuntimeError(
                f"Cannot set execution client in state {self._state.name}. "
                "Must be INITIALIZED."
            )
        self._execution_client = client
        logger.info("ExecutionClient set: %s", client.name)

    def set_clients(
        self,
        data_client: DataClient,
        execution_client: ExecutionClient,
    ) -> None:
        """
        Set both data and execution clients at once.

        Convenience method for setting both clients together.

        Args:
            data_client: DataClient for market data
            execution_client: ExecutionClient for order execution

        Raises:
            RuntimeError: If kernel is already running
        """
        self.set_data_client(data_client)
        self.set_execution_client(execution_client)

    def add_actor(self, actor: BaseActor) -> None:
        """
        Register an actor with the kernel.

        The actor will be:
        - Initialized with access to bus, cache, clock
        - Started when kernel starts
        - Stopped when kernel stops

        Args:
            actor: Actor to register

        Raises:
            RuntimeError: If kernel is already running
            ValueError: If actor with same name already registered
        """
        if self._state != KernelState.INITIALIZED:
            raise RuntimeError(
                f"Cannot add actor in state {self._state.name}. "
                "Must be INITIALIZED."
            )

        # Check for duplicate names
        for existing in self._actors:
            if existing.name == actor.name:
                raise ValueError(f"Actor with name '{actor.name}' already registered")

        self._actors.append(actor)
        logger.info("Actor registered: %s", actor.name)

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Register a strategy with the kernel.

        The strategy will be:
        - Initialized with access to bus, cache, clock, gateway
        - Started when kernel starts (after actors)
        - Stopped when kernel stops (before actors)

        Args:
            strategy: Strategy to register

        Raises:
            RuntimeError: If kernel is already running
            ValueError: If strategy with same name already registered
        """
        if self._state != KernelState.INITIALIZED:
            raise RuntimeError(
                f"Cannot add strategy in state {self._state.name}. "
                "Must be INITIALIZED."
            )

        # Check for duplicate names
        for existing in self._strategies:
            if existing.name == strategy.name:
                raise ValueError(
                    f"Strategy with name '{strategy.name}' already registered"
                )

        self._strategies.append(strategy)
        logger.info("Strategy registered: %s", strategy.name)

    # =========================================================================
    # Component Access
    # =========================================================================

    @property
    def actors(self) -> list[BaseActor]:
        """List of registered actors."""
        return list(self._actors)

    @property
    def strategies(self) -> list[BaseStrategy]:
        """List of registered strategies."""
        return list(self._strategies)

    def get_actor(self, name: str) -> BaseActor | None:
        """Get actor by name."""
        for actor in self._actors:
            if actor.name == name:
                return actor
        return None

    def get_strategy(self, name: str) -> BaseStrategy | None:
        """Get strategy by name."""
        for strategy in self._strategies:
            if strategy.name == name:
                return strategy
        return None

    # =========================================================================
    # Plugin Discovery (Issue #29)
    # =========================================================================

    def discover_plugins(self) -> dict[str, dict[str, type]]:
        """
        Discover all registered plugins via entry_points.

        Returns:
            Dictionary with 'strategies' and 'gateways' keys,
            each mapping plugin names to their classes.

        Example:
            plugins = kernel.discover_plugins()
            print(plugins['strategies'])  # {'freqtrade': FreqtradeAdapter}
            print(plugins['gateways'])    # {'paper': PaperGateway, 'ccxt': CCXTGateway}
        """
        strategies = discover_strategies()
        gateways = discover_gateways()

        if self._config.log_plugins:
            logger.info(
                "Discovered %d strategy plugin(s): %s",
                len(strategies),
                list(strategies.keys()) or "(none)",
            )
            logger.info(
                "Discovered %d gateway plugin(s): %s",
                len(gateways),
                list(gateways.keys()) or "(none)",
            )

            # Log detailed plugin metadata
            for meta in list_strategy_plugins():
                logger.debug(
                    "  Strategy plugin: %s v%s - %s",
                    meta.name,
                    meta.version,
                    meta.description,
                )

        return {
            "strategies": strategies,
            "gateways": gateways,
        }

    def get_available_gateways(self) -> dict[str, type]:
        """
        Get all available gateway plugins.

        Returns:
            Dictionary mapping gateway names to their classes.
        """
        return discover_gateways()

    def get_available_strategies(self) -> dict[str, type]:
        """
        Get all available strategy plugins.

        Returns:
            Dictionary mapping strategy plugin names to their classes.
        """
        return discover_strategies()

    # =========================================================================
    # Startup
    # =========================================================================

    async def start_async(self) -> None:
        """
        Start the kernel and all components.

        Initialization order:
        1. MessageBus
        2. Cache
        3. Clock
        4. RiskManager (if set)
        5. Gateway (if set)
        6. Actors
        7. Strategies

        Raises:
            RuntimeError: If kernel is not in INITIALIZED state
            asyncio.TimeoutError: If startup exceeds timeout
        """
        if self._state != KernelState.INITIALIZED:
            raise RuntimeError(
                f"Cannot start kernel in state {self._state.name}. "
                "Must be INITIALIZED."
            )

        self._state = KernelState.STARTING
        logger.info(
            "TradingKernel starting: instance_id=%s components=%d",
            self._config.instance_id,
            len(self._actors) + len(self._strategies),
        )

        try:
            async with asyncio.timeout(self._config.startup_timeout):
                await self._startup_sequence()
        except asyncio.TimeoutError:
            logger.error(
                "Startup timeout after %.1fs", self._config.startup_timeout
            )
            # Attempt cleanup
            await self._emergency_shutdown()
            raise

        self._state = KernelState.RUNNING
        self._ts_started = time.time_ns()

        # Publish kernel started event
        self._bus.publish(
            Event.create(
                event_type=EventType.ACTOR_STARTED,
                source="kernel",
                payload={
                    "instance_id": self._config.instance_id,
                    "environment": self._config.environment,
                    "actors": len(self._actors),
                    "strategies": len(self._strategies),
                },
            )
        )

        logger.info(
            "TradingKernel running: actors=%d strategies=%d",
            len(self._actors),
            len(self._strategies),
        )

    async def _startup_sequence(self) -> None:
        """Execute the startup sequence in correct order."""
        # 0. Discover registered plugins (Issue #29)
        if self._config.discover_plugins:
            self.discover_plugins()

        # 1. Start MessageBus
        logger.debug("Starting MessageBus...")
        self._bus_task = asyncio.create_task(self._bus.start())
        await asyncio.sleep(0)  # Let bus start

        # 2. Cache is ready (no async init needed)
        logger.debug("Cache ready")

        # 3. Start Clock
        logger.debug("Starting Clock...")
        self._clock.start()

        # 4. RiskManager (if set)
        if self._risk_manager is not None:
            logger.debug("Initializing RiskManager...")
            # RiskManager doesn't have async start, just ensure it's ready

        # 5. Gateway connection (if set) - legacy
        if self._gateway is not None:
            logger.debug("Connecting Gateway...")
            if hasattr(self._gateway, "connect"):
                await self._gateway.connect()

        # 5b. DataClient connection (if set)
        if self._data_client is not None:
            logger.debug("Connecting DataClient: %s", self._data_client.name)
            await self._data_client.connect()

        # 5c. ExecutionClient connection (if set)
        if self._execution_client is not None:
            logger.debug("Connecting ExecutionClient: %s", self._execution_client.name)
            await self._execution_client.connect()

        # 6. Initialize and start Actors
        for actor in self._actors:
            logger.debug("Initializing actor: %s", actor.name)
            await actor.initialize(self._bus)

            if self._config.load_state:
                await self._load_actor_state(actor)

            logger.debug("Starting actor: %s", actor.name)
            await actor.start()

        # 7. Initialize and start Strategies
        for strategy in self._strategies:
            logger.debug("Initializing strategy: %s", strategy.name)
            await strategy.initialize(self._bus)

            if self._config.load_state:
                await self._load_actor_state(strategy)

            logger.debug("Starting strategy: %s", strategy.name)
            await strategy.start()

    async def _load_actor_state(self, actor: BaseActor) -> None:
        """Load saved state for an actor."""
        # TODO: Implement state persistence
        # This would load from Redis/PostgreSQL/file
        pass

    # =========================================================================
    # Shutdown
    # =========================================================================

    async def stop_async(self) -> None:
        """
        Stop the kernel and all components gracefully.

        Shutdown order (reverse of startup):
        1. Strategies (stop trading first)
        2. Actors
        3. Gateway
        4. RiskManager
        5. Clock
        6. MessageBus (drain events)
        7. Save state (if configured)

        Raises:
            RuntimeError: If kernel is not running
        """
        if self._state not in (KernelState.RUNNING, KernelState.STARTING):
            logger.warning(
                "Cannot stop kernel in state %s", self._state.name
            )
            return

        self._state = KernelState.STOPPING
        logger.info("TradingKernel stopping...")

        try:
            async with asyncio.timeout(self._config.shutdown_timeout):
                await self._shutdown_sequence()
        except asyncio.TimeoutError:
            logger.error(
                "Shutdown timeout after %.1fs, forcing...",
                self._config.shutdown_timeout,
            )
            await self._emergency_shutdown()

        self._state = KernelState.STOPPED
        self._ts_stopped = time.time_ns()

        # Publish kernel stopped event (bus may be stopped)
        logger.info(
            "TradingKernel stopped: instance_id=%s uptime_sec=%.1f",
            self._config.instance_id,
            (self._ts_stopped - (self._ts_started or self._ts_created)) / 1e9,
        )

    async def _shutdown_sequence(self) -> None:
        """Execute the shutdown sequence in correct order."""
        # 1. Stop Strategies (stop trading first)
        for strategy in reversed(self._strategies):
            logger.debug("Stopping strategy: %s", strategy.name)
            try:
                if strategy.state in (ComponentState.RUNNING, ComponentState.DEGRADED):
                    await strategy.stop()

                if self._config.save_state:
                    await self._save_actor_state(strategy)
            except Exception:
                logger.exception("Error stopping strategy: %s", strategy.name)

        # 2. Stop Actors
        for actor in reversed(self._actors):
            logger.debug("Stopping actor: %s", actor.name)
            try:
                if actor.state in (ComponentState.RUNNING, ComponentState.DEGRADED):
                    await actor.stop()

                if self._config.save_state:
                    await self._save_actor_state(actor)
            except Exception:
                logger.exception("Error stopping actor: %s", actor.name)

        # 3. Disconnect Gateway (legacy)
        if self._gateway is not None:
            logger.debug("Disconnecting Gateway...")
            try:
                if hasattr(self._gateway, "disconnect"):
                    await self._gateway.disconnect()
            except Exception:
                logger.exception("Error disconnecting gateway")

        # 3b. Disconnect ExecutionClient
        if self._execution_client is not None:
            logger.debug("Disconnecting ExecutionClient: %s", self._execution_client.name)
            try:
                await self._execution_client.disconnect()
            except Exception:
                logger.exception("Error disconnecting execution client")

        # 3c. Disconnect DataClient
        if self._data_client is not None:
            logger.debug("Disconnecting DataClient: %s", self._data_client.name)
            try:
                await self._data_client.disconnect()
            except Exception:
                logger.exception("Error disconnecting data client")

        # 4. Stop Clock
        logger.debug("Stopping Clock...")
        self._clock.stop()

        # 5. Stop MessageBus (drain remaining events)
        logger.debug("Stopping MessageBus...")
        await self._bus.stop(drain=True)

        if self._bus_task:
            self._bus_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._bus_task

    async def _save_actor_state(self, actor: BaseActor) -> None:
        """Save actor state for recovery."""
        # TODO: Implement state persistence
        # This would save to Redis/PostgreSQL/file
        pass

    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown - cancel everything immediately."""
        logger.warning("Emergency shutdown initiated")

        # Force stop all strategies
        for strategy in self._strategies:
            try:
                if hasattr(strategy, "fault"):
                    await strategy.fault(RuntimeError("Emergency shutdown"))
            except Exception:
                pass

        # Force stop all actors
        for actor in self._actors:
            try:
                if hasattr(actor, "fault"):
                    await actor.fault(RuntimeError("Emergency shutdown"))
            except Exception:
                pass

        # Force stop bus
        try:
            await self._bus.stop(drain=False)
        except Exception:
            pass

        if self._bus_task:
            self._bus_task.cancel()

        self._clock.stop()

    # =========================================================================
    # Dispose
    # =========================================================================

    async def dispose(self) -> None:
        """
        Dispose of kernel and release all resources.

        This is idempotent - calling multiple times has same effect.
        After disposal, the kernel cannot be used again.
        """
        if self._state == KernelState.DISPOSED:
            return

        if self._state == KernelState.RUNNING:
            await self.stop_async()

        self._state = KernelState.DISPOSING
        logger.debug("TradingKernel disposing...")

        # Dispose strategies
        for strategy in self._strategies:
            try:
                if strategy.state == ComponentState.STOPPED:
                    await strategy.dispose()
            except Exception:
                logger.exception("Error disposing strategy: %s", strategy.name)

        # Dispose actors
        for actor in self._actors:
            try:
                if actor.state == ComponentState.STOPPED:
                    await actor.dispose()
            except Exception:
                logger.exception("Error disposing actor: %s", actor.name)

        # Clear cache
        await self._cache.clear()

        self._state = KernelState.DISPOSED
        logger.info("TradingKernel disposed")

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """
        Check health of kernel and all components.

        Returns:
            Dictionary with health status of each component
        """
        health: dict[str, Any] = {
            "kernel": {
                "state": self._state.name,
                "instance_id": self._config.instance_id,
                "environment": self._config.environment,
                "uptime_sec": self._uptime_seconds(),
            },
            "bus": {
                "running": self._bus.is_running,
                "stats": self._bus.stats,
            },
            "cache": self._cache.stats(),
            "clock": self._clock.stats(),
            "actors": {},
            "strategies": {},
        }

        # Actor health
        for actor in self._actors:
            health["actors"][actor.name] = {
                "state": actor.state.name,
                "is_running": actor.is_running,
                "is_degraded": actor.is_degraded,
            }

        # Strategy health
        for strategy in self._strategies:
            health["strategies"][strategy.name] = {
                "state": strategy.state.name,
                "is_running": strategy.is_running,
                "is_degraded": strategy.is_degraded,
            }

        return health

    def _uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        if self._ts_started is None:
            return 0.0

        end_time = self._ts_stopped or time.time_ns()
        return (end_time - self._ts_started) / 1e9

    def is_healthy(self) -> bool:
        """
        Quick health check.

        Returns:
            True if kernel and all components are healthy
        """
        if self._state != KernelState.RUNNING:
            return False

        if not self._bus.is_running:
            return False

        # Check all actors are running or degraded (not stopped/faulted)
        for actor in self._actors:
            if actor.state not in (ComponentState.RUNNING, ComponentState.DEGRADED):
                return False

        # Check all strategies are running or degraded
        for strategy in self._strategies:
            if strategy.state not in (ComponentState.RUNNING, ComponentState.DEGRADED):
                return False

        return True

    # =========================================================================
    # Order Submission (with Risk Validation)
    # =========================================================================

    async def submit_order(
        self,
        order: Order,
        current_price: Decimal,
        instrument: Instrument | None = None,
    ) -> OrderResult:
        """
        Submit an order with mandatory risk validation.

        This is the RECOMMENDED way to submit orders. It ensures:
        1. Risk engine validation (if configured)
        2. Order submission to execution client
        3. Open order tracking for self-trade prevention

        Args:
            order: Order to submit
            current_price: Current market price (for risk checks)
            instrument: Optional instrument (for precision validation)

        Returns:
            OrderResult from execution client

        Raises:
            RuntimeError: If kernel is not running or execution client not set
            OrderDeniedException: If order fails risk validation
        """
        from decimal import Decimal as Dec

        if self._state != KernelState.RUNNING:
            raise RuntimeError(
                f"Cannot submit orders in state {self._state.name}. "
                "Kernel must be RUNNING."
            )

        if self._execution_client is None:
            raise RuntimeError("ExecutionClient not set. Cannot submit orders.")

        # Risk validation (mandatory if engine is set)
        if self._risk_engine is not None:
            result = self._risk_engine.validate_order(
                order, current_price, instrument
            )
            if not result.passed:
                # Create a denied order result
                from libra.gateways.protocol import OrderResult, OrderStatus

                logger.warning(
                    "Order DENIED by RiskEngine: %s - %s",
                    result.check_name,
                    result.reason,
                )

                return OrderResult(
                    order_id="",
                    symbol=order.symbol,
                    status=OrderStatus.REJECTED,
                    side=order.side,
                    order_type=order.order_type,
                    amount=order.amount,
                    filled_amount=Dec("0"),
                    remaining_amount=order.amount,
                    average_price=None,
                    fee=Dec("0"),
                    fee_currency="",
                    timestamp_ns=time.time_ns(),
                    client_order_id=order.client_order_id,
                    price=order.price,
                    stop_price=order.stop_price,
                )

        # Submit to execution client
        order_result = await self._execution_client.submit_order(order)

        # Track open order for self-trade prevention
        if self._risk_engine is not None and order_result.is_open:
            self._risk_engine.add_open_order(order)

        return order_result

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
    ) -> bool:
        """
        Cancel an order and update risk engine tracking.

        Args:
            order_id: Order ID to cancel
            symbol: Order symbol

        Returns:
            True if cancellation was successful

        Raises:
            RuntimeError: If kernel is not running or execution client not set
        """
        if self._state != KernelState.RUNNING:
            raise RuntimeError(
                f"Cannot cancel orders in state {self._state.name}. "
                "Kernel must be RUNNING."
            )

        if self._execution_client is None:
            raise RuntimeError("ExecutionClient not set. Cannot cancel orders.")

        success = await self._execution_client.cancel_order(order_id, symbol)

        # Remove from open order tracking
        if success and self._risk_engine is not None:
            self._risk_engine.remove_open_order(symbol, order_id=order_id)

        return success

    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        price: Decimal | None = None,
        amount: Decimal | None = None,
        current_price: Decimal | None = None,
        instrument: Instrument | None = None,
    ) -> OrderResult:
        """
        Modify an order with risk validation.

        Args:
            order_id: Order ID to modify
            symbol: Order symbol
            price: New price (if changing)
            amount: New amount (if changing)
            current_price: Current market price (for collar check)
            instrument: Instrument (for precision validation)

        Returns:
            OrderResult with updated order state

        Raises:
            RuntimeError: If kernel is not running or execution client not set
        """
        from decimal import Decimal as Dec

        if self._state != KernelState.RUNNING:
            raise RuntimeError(
                f"Cannot modify orders in state {self._state.name}. "
                "Kernel must be RUNNING."
            )

        if self._execution_client is None:
            raise RuntimeError("ExecutionClient not set. Cannot modify orders.")

        # Risk validation for modifications
        if self._risk_engine is not None:
            result = self._risk_engine.validate_modify(
                order_id,
                new_price=price,
                new_amount=amount,
                current_price=current_price,
                instrument=instrument,
            )
            if not result.passed:
                from libra.gateways.protocol import OrderResult, OrderStatus, OrderSide, OrderType

                logger.warning(
                    "Order modification DENIED by RiskEngine: %s - %s",
                    result.check_name,
                    result.reason,
                )

                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    status=OrderStatus.REJECTED,
                    side=OrderSide.BUY,  # Unknown, placeholder
                    order_type=OrderType.LIMIT,  # Unknown, placeholder
                    amount=amount or Dec("0"),
                    filled_amount=Dec("0"),
                    remaining_amount=amount or Dec("0"),
                    average_price=None,
                    fee=Dec("0"),
                    fee_currency="",
                    timestamp_ns=time.time_ns(),
                    price=price,
                )

        return await self._execution_client.modify_order(
            order_id, symbol, price, amount
        )

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> TradingKernel:
        """Async context manager entry - start the kernel."""
        await self.start_async()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - stop and dispose."""
        if exc_type is not None:
            logger.error(
                "Kernel context exiting with exception: %s: %s",
                exc_type.__name__,
                exc_val,
            )

        await self.stop_async()
        await self.dispose()

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TradingKernel("
            f"instance_id={self._config.instance_id!r}, "
            f"environment={self._config.environment!r}, "
            f"state={self._state.name}, "
            f"actors={len(self._actors)}, "
            f"strategies={len(self._strategies)})"
        )

"""Integration tests for TradingKernel with full component stack."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from libra.core.events import Event, EventType
from libra.core.kernel import KernelConfig, KernelState, TradingKernel
from libra.gateways.protocol import Tick
from libra.strategies.actor import BaseActor, ComponentState
from libra.strategies.protocol import Bar


# =============================================================================
# Test Components
# =============================================================================


class DataFeedActor(BaseActor):
    """Simulates a data feed that publishes market data."""

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        super().__init__()
        self._symbol = symbol
        self.ticks_published = 0

    @property
    def name(self) -> str:
        return f"data_feed_{self._symbol.replace('/', '_')}"

    async def publish_tick(self, price: Decimal) -> None:
        """Publish a tick event."""
        tick = Tick(
            symbol=self._symbol,
            bid=price - Decimal("0.5"),
            ask=price + Decimal("0.5"),
            last=price,
            timestamp_ns=self.bus._events_published,
        )
        self.publish_event(EventType.TICK, {"tick": tick})
        self.ticks_published += 1


class PriceMonitorActor(BaseActor):
    """Actor that monitors prices and tracks statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.prices: dict[str, list[Decimal]] = {}
        self.events_received = 0

    @property
    def name(self) -> str:
        return "price_monitor"

    async def on_start(self) -> None:
        await self.subscribe(EventType.TICK)

    async def on_event(self, event: Event) -> None:
        self.events_received += 1
        tick = event.payload.get("tick")
        if tick:
            symbol = tick.symbol
            if symbol not in self.prices:
                self.prices[symbol] = []
            self.prices[symbol].append(tick.last)


class SignalGeneratorActor(BaseActor):
    """Actor that generates trading signals based on price movements."""

    def __init__(self, threshold: Decimal = Decimal("10")) -> None:
        super().__init__()
        self._threshold = threshold
        self._last_prices: dict[str, Decimal] = {}
        self.signals_generated = 0

    @property
    def name(self) -> str:
        return "signal_generator"

    async def on_start(self) -> None:
        await self.subscribe(EventType.TICK)

    async def on_event(self, event: Event) -> None:
        tick = event.payload.get("tick")
        if not tick:
            return

        symbol = tick.symbol
        current_price = tick.last

        if symbol in self._last_prices:
            price_change = current_price - self._last_prices[symbol]
            if abs(price_change) >= self._threshold:
                action = "BUY" if price_change > 0 else "SELL"
                self.publish_event(
                    EventType.SIGNAL,
                    {
                        "symbol": symbol,
                        "action": action,
                        "price": str(current_price),
                        "change": str(price_change),
                    },
                )
                self.signals_generated += 1

        self._last_prices[symbol] = current_price


class MockStrategy(BaseActor):
    """Mock strategy that reacts to signals."""

    def __init__(self) -> None:
        super().__init__()
        self.signals_received: list[dict[str, Any]] = []
        self.orders_placed = 0

    @property
    def name(self) -> str:
        return "mock_strategy"

    async def on_start(self) -> None:
        await self.subscribe(EventType.SIGNAL)

    async def on_event(self, event: Event) -> None:
        if event.event_type == EventType.SIGNAL:
            self.signals_received.append(event.payload)
            # Simulate order placement
            self.orders_placed += 1
            self.publish_event(
                EventType.ORDER_NEW,
                {
                    "symbol": event.payload.get("symbol"),
                    "side": event.payload.get("action"),
                    "amount": "0.1",
                },
            )


class MockGateway:
    """Mock exchange gateway."""

    def __init__(self) -> None:
        self.connected = False
        self.orders_received: list[dict[str, Any]] = []

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False


# =============================================================================
# Integration Tests
# =============================================================================


class TestKernelIntegration:
    """Integration tests for TradingKernel with multiple components."""

    @pytest.mark.asyncio
    async def test_full_event_pipeline(self) -> None:
        """Test complete event flow through kernel-managed components."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        # Create components
        data_feed = DataFeedActor("BTC/USDT")
        price_monitor = PriceMonitorActor()
        signal_gen = SignalGeneratorActor(threshold=Decimal("5"))
        strategy = MockStrategy()

        # Register with kernel
        kernel.add_actor(data_feed)
        kernel.add_actor(price_monitor)
        kernel.add_actor(signal_gen)
        kernel.add_strategy(strategy)

        async with kernel:
            # Verify all components are running
            assert kernel.is_running
            assert data_feed.is_running
            assert price_monitor.is_running
            assert signal_gen.is_running
            assert strategy.is_running

            # Publish price updates
            await data_feed.publish_tick(Decimal("50000"))
            await asyncio.sleep(0.05)

            await data_feed.publish_tick(Decimal("50010"))  # +10, should trigger
            await asyncio.sleep(0.1)

            await data_feed.publish_tick(Decimal("50005"))  # -5, should trigger
            await asyncio.sleep(0.1)

        # Verify event processing
        assert data_feed.ticks_published == 3
        assert price_monitor.events_received == 3
        assert len(price_monitor.prices["BTC/USDT"]) == 3

        # Signal should have been generated and received
        assert signal_gen.signals_generated == 2  # Two threshold crossings
        assert len(strategy.signals_received) == 2
        assert strategy.orders_placed == 2

    @pytest.mark.asyncio
    async def test_kernel_manages_component_lifecycle(self) -> None:
        """Test that kernel properly manages component lifecycle."""
        kernel = TradingKernel()

        actor1 = PriceMonitorActor()
        actor2 = SignalGeneratorActor()

        kernel.add_actor(actor1)
        kernel.add_actor(actor2)

        # Before start
        assert actor1.state == ComponentState.PRE_INITIALIZED
        assert actor2.state == ComponentState.PRE_INITIALIZED

        await kernel.start_async()

        # After start
        assert actor1.state == ComponentState.RUNNING
        assert actor2.state == ComponentState.RUNNING

        await kernel.stop_async()

        # After stop
        assert actor1.state == ComponentState.STOPPED
        assert actor2.state == ComponentState.STOPPED

        await kernel.dispose()

        # After dispose
        assert actor1.state == ComponentState.DISPOSED
        assert actor2.state == ComponentState.DISPOSED

    @pytest.mark.asyncio
    async def test_kernel_with_gateway(self) -> None:
        """Test kernel with gateway integration."""
        kernel = TradingKernel()
        gateway = MockGateway()

        kernel.set_gateway(gateway)

        await kernel.start_async()
        assert gateway.connected is True

        await kernel.stop_async()
        assert gateway.connected is False

    @pytest.mark.asyncio
    async def test_kernel_health_check_with_components(self) -> None:
        """Test health check with multiple components."""
        kernel = TradingKernel()

        kernel.add_actor(PriceMonitorActor())
        kernel.add_actor(SignalGeneratorActor())
        kernel.add_strategy(MockStrategy())

        await kernel.start_async()

        health = kernel.health_check()

        assert health["kernel"]["state"] == "RUNNING"
        assert "price_monitor" in health["actors"]
        assert "signal_generator" in health["actors"]
        assert "mock_strategy" in health["strategies"]

        assert kernel.is_healthy()

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_backtest_environment(self) -> None:
        """Test kernel in backtest environment uses backtest clock."""
        config = KernelConfig(environment="backtest")
        kernel = TradingKernel(config)

        assert kernel.clock.is_backtest

        actor = PriceMonitorActor()
        kernel.add_actor(actor)

        await kernel.start_async()
        assert kernel.is_running

        await kernel.stop_async()

    @pytest.mark.asyncio
    async def test_multiple_data_feeds(self) -> None:
        """Test kernel handling multiple data feeds."""
        kernel = TradingKernel()

        btc_feed = DataFeedActor("BTC/USDT")
        eth_feed = DataFeedActor("ETH/USDT")
        monitor = PriceMonitorActor()

        kernel.add_actor(btc_feed)
        kernel.add_actor(eth_feed)
        kernel.add_actor(monitor)

        async with kernel:
            # Publish from both feeds
            await btc_feed.publish_tick(Decimal("50000"))
            await eth_feed.publish_tick(Decimal("3000"))
            await asyncio.sleep(0.1)

            await btc_feed.publish_tick(Decimal("50100"))
            await eth_feed.publish_tick(Decimal("3010"))
            await asyncio.sleep(0.1)

        # Monitor should have received all ticks
        assert monitor.events_received == 4
        assert "BTC/USDT" in monitor.prices
        assert "ETH/USDT" in monitor.prices
        assert len(monitor.prices["BTC/USDT"]) == 2
        assert len(monitor.prices["ETH/USDT"]) == 2

    @pytest.mark.asyncio
    async def test_actor_failure_isolation(self) -> None:
        """Test that one actor's failure doesn't crash others."""

        class FailingActor(BaseActor):
            """Actor that fails on event processing."""

            def __init__(self) -> None:
                super().__init__()
                self.fail_count = 0

            @property
            def name(self) -> str:
                return "failing_actor"

            async def on_start(self) -> None:
                await self.subscribe(EventType.TICK)

            async def on_event(self, event: Event) -> None:
                self.fail_count += 1
                raise ValueError("Simulated failure")

        kernel = TradingKernel()

        failing = FailingActor()
        monitor = PriceMonitorActor()

        kernel.add_actor(failing)
        kernel.add_actor(monitor)

        async with kernel:
            # Publish tick - failing actor will error, but monitor should work
            tick = Tick(
                symbol="BTC/USDT",
                bid=Decimal("50000"),
                ask=Decimal("50001"),
                last=Decimal("50000"),
                timestamp_ns=1000000000,
            )
            kernel.bus.publish(Event.create(EventType.TICK, "test", {"tick": tick}))
            await asyncio.sleep(0.1)

        # Failing actor attempted to process
        assert failing.fail_count >= 1

        # Monitor still processed successfully
        assert monitor.events_received >= 1

    @pytest.mark.asyncio
    async def test_kernel_uptime_tracking(self) -> None:
        """Test that kernel tracks uptime correctly."""
        kernel = TradingKernel()

        assert kernel.ts_started is None
        assert kernel.ts_stopped is None

        await kernel.start_async()
        start_time = kernel.ts_started
        assert start_time is not None

        await asyncio.sleep(0.1)

        await kernel.stop_async()
        stop_time = kernel.ts_stopped
        assert stop_time is not None
        assert stop_time > start_time

        health = kernel.health_check()
        assert health["kernel"]["uptime_sec"] > 0

    @pytest.mark.asyncio
    async def test_kernel_repr(self) -> None:
        """Test kernel string representation."""
        config = KernelConfig(instance_id="test_123", environment="live")
        kernel = TradingKernel(config)

        kernel.add_actor(PriceMonitorActor())
        kernel.add_strategy(MockStrategy())

        repr_str = repr(kernel)

        assert "test_123" in repr_str
        assert "live" in repr_str
        assert "actors=1" in repr_str
        assert "strategies=1" in repr_str


class TestKernelEventCorrelation:
    """Test event correlation through the kernel."""

    @pytest.mark.asyncio
    async def test_trace_id_propagation(self) -> None:
        """Test that trace IDs propagate through event chain."""

        class TraceCollectorActor(BaseActor):
            def __init__(self) -> None:
                super().__init__()
                self.trace_ids: list[str] = []

            @property
            def name(self) -> str:
                return "trace_collector"

            async def on_start(self) -> None:
                await self.subscribe(EventType.TICK)
                await self.subscribe(EventType.SIGNAL)

            async def on_event(self, event: Event) -> None:
                self.trace_ids.append(event.trace_id)
                if event.event_type == EventType.TICK:
                    # Create child event with same trace_id
                    child = event.child_event(
                        EventType.SIGNAL,
                        source=f"actor.{self.name}",
                        payload={"derived_from": "tick"},
                    )
                    self.bus.publish(child)

        kernel = TradingKernel()
        collector = TraceCollectorActor()
        kernel.add_actor(collector)

        async with kernel:
            # Publish initial tick
            tick = Tick(
                symbol="BTC/USDT",
                bid=Decimal("50000"),
                ask=Decimal("50001"),
                last=Decimal("50000"),
                timestamp_ns=1000000000,
            )
            initial_event = Event.create(EventType.TICK, "test", {"tick": tick})
            kernel.bus.publish(initial_event)

            await asyncio.sleep(0.15)

        # Should have received both events
        assert len(collector.trace_ids) == 2

        # Both should have the same trace_id
        assert collector.trace_ids[0] == collector.trace_ids[1]

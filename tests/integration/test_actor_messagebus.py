"""Integration tests for Actor with MessageBus event flow."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus
from libra.gateways.protocol import Tick
from libra.strategies.actor import BaseActor, ComponentState
from libra.strategies.protocol import Bar


# =============================================================================
# Test Actors
# =============================================================================


class TickCollectorActor(BaseActor):
    """Actor that collects tick events."""

    def __init__(self) -> None:
        super().__init__()
        self.ticks_received: list[Tick] = []
        self.events_received: list[Event] = []

    @property
    def name(self) -> str:
        return "tick_collector"

    async def on_start(self) -> None:
        await self.subscribe(EventType.TICK)

    async def on_tick(self, tick: Tick) -> None:
        self.ticks_received.append(tick)

    async def on_event(self, event: Event) -> None:
        self.events_received.append(event)


class BarProcessorActor(BaseActor):
    """Actor that processes bar events and publishes signals."""

    def __init__(self) -> None:
        super().__init__()
        self.bars_received: list[Bar] = []
        self.signals_published = 0

    @property
    def name(self) -> str:
        return "bar_processor"

    async def on_start(self) -> None:
        await self.subscribe(EventType.BAR)

    async def on_bar(self, bar: Bar) -> None:
        self.bars_received.append(bar)
        # Publish a signal for each bar
        self.publish_event(
            EventType.SIGNAL,
            {"symbol": bar.symbol, "action": "BUY" if bar.close > bar.open else "SELL"},
        )
        self.signals_published += 1


class SignalReceiverActor(BaseActor):
    """Actor that receives signal events."""

    def __init__(self) -> None:
        super().__init__()
        self.signals_received: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "signal_receiver"

    async def on_start(self) -> None:
        await self.subscribe(EventType.SIGNAL)

    async def on_event(self, event: Event) -> None:
        if event.event_type == EventType.SIGNAL:
            self.signals_received.append(event.payload)


class MultiEventActor(BaseActor):
    """Actor that subscribes to multiple event types."""

    def __init__(self) -> None:
        super().__init__()
        self.tick_count = 0
        self.bar_count = 0
        self.signal_count = 0

    @property
    def name(self) -> str:
        return "multi_event"

    async def on_start(self) -> None:
        await self.subscribe(EventType.TICK)
        await self.subscribe(EventType.BAR)
        await self.subscribe(EventType.SIGNAL)

    async def on_event(self, event: Event) -> None:
        if event.event_type == EventType.TICK:
            self.tick_count += 1
        elif event.event_type == EventType.BAR:
            self.bar_count += 1
        elif event.event_type == EventType.SIGNAL:
            self.signal_count += 1


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def message_bus() -> MessageBus:
    """Create a MessageBus instance."""
    return MessageBus()


# =============================================================================
# Integration Tests
# =============================================================================


class TestActorMessageBusIntegration:
    """Integration tests for Actor with MessageBus."""

    @pytest.mark.asyncio
    async def test_actor_receives_events(self, message_bus: MessageBus) -> None:
        """Test that actor receives events published to MessageBus."""
        actor = TickCollectorActor()
        await actor.initialize(message_bus)

        async with message_bus:
            await actor.start()

            # Publish tick events
            tick = Tick(
                symbol="BTC/USDT",
                bid=Decimal("50000"),
                ask=Decimal("50001"),
                last=Decimal("50000.50"),
                timestamp_ns=1000000000,
            )
            message_bus.publish(
                Event.create(EventType.TICK, "test", {"tick": tick})
            )

            # Wait for event processing
            await asyncio.sleep(0.05)

            await actor.stop()

        assert len(actor.events_received) == 1
        assert actor.events_received[0].event_type == EventType.TICK

    @pytest.mark.asyncio
    async def test_actor_publishes_events(self, message_bus: MessageBus) -> None:
        """Test that actor can publish events to MessageBus."""
        processor = BarProcessorActor()
        receiver = SignalReceiverActor()

        await processor.initialize(message_bus)
        await receiver.initialize(message_bus)

        async with message_bus:
            await processor.start()
            await receiver.start()

            # Publish a bar event
            bar = Bar(
                symbol="BTC/USDT",
                timestamp_ns=1000000000,
                open=Decimal("49000"),
                high=Decimal("51000"),
                low=Decimal("48500"),
                close=Decimal("50500"),  # Close > Open = bullish
                volume=Decimal("1000"),
                timeframe="1h",
            )
            message_bus.publish(
                Event.create(EventType.BAR, "test", {"bar": bar})
            )

            # Wait for event processing chain
            await asyncio.sleep(0.1)

            await processor.stop()
            await receiver.stop()

        # Processor should have received the bar
        assert len(processor.bars_received) == 1
        assert processor.signals_published == 1

        # Receiver should have received the signal
        assert len(receiver.signals_received) == 1
        assert receiver.signals_received[0]["symbol"] == "BTC/USDT"
        assert receiver.signals_received[0]["action"] == "BUY"

    @pytest.mark.asyncio
    async def test_multiple_actors_same_event_type(self, message_bus: MessageBus) -> None:
        """Test that multiple actors can subscribe to the same event type."""
        actor1 = TickCollectorActor()
        actor2 = TickCollectorActor()

        await actor1.initialize(message_bus)
        await actor2.initialize(message_bus)

        async with message_bus:
            await actor1.start()
            await actor2.start()

            # Publish tick
            tick = Tick(
                symbol="ETH/USDT",
                bid=Decimal("2000"),
                ask=Decimal("2001"),
                last=Decimal("2000.50"),
                timestamp_ns=1000000000,
            )
            message_bus.publish(
                Event.create(EventType.TICK, "test", {"tick": tick})
            )

            await asyncio.sleep(0.05)

            await actor1.stop()
            await actor2.stop()

        # Both actors should receive the event
        assert len(actor1.events_received) == 1
        assert len(actor2.events_received) == 1

    @pytest.mark.asyncio
    async def test_actor_multiple_subscriptions(self, message_bus: MessageBus) -> None:
        """Test actor with multiple event subscriptions."""
        actor = MultiEventActor()
        await actor.initialize(message_bus)

        async with message_bus:
            await actor.start()

            # Publish different event types
            tick = Tick(
                symbol="BTC/USDT",
                bid=Decimal("50000"),
                ask=Decimal("50001"),
                last=Decimal("50000"),
                timestamp_ns=1000000000,
            )
            bar = Bar(
                symbol="BTC/USDT",
                timestamp_ns=1000000000,
                open=Decimal("49000"),
                high=Decimal("51000"),
                low=Decimal("48500"),
                close=Decimal("50500"),
                volume=Decimal("1000"),
                timeframe="1h",
            )

            message_bus.publish(Event.create(EventType.TICK, "test", {"tick": tick}))
            message_bus.publish(Event.create(EventType.BAR, "test", {"bar": bar}))
            message_bus.publish(Event.create(EventType.SIGNAL, "test", {"action": "BUY"}))

            await asyncio.sleep(0.1)

            await actor.stop()

        assert actor.tick_count == 1
        assert actor.bar_count == 1
        assert actor.signal_count == 1

    @pytest.mark.asyncio
    async def test_actor_unsubscribes_on_stop(self, message_bus: MessageBus) -> None:
        """Test that actor unsubscribes from events when stopped."""
        actor = TickCollectorActor()
        await actor.initialize(message_bus)

        async with message_bus:
            await actor.start()

            # Publish first tick
            tick1 = Tick(
                symbol="BTC/USDT",
                bid=Decimal("50000"),
                ask=Decimal("50001"),
                last=Decimal("50000"),
                timestamp_ns=1000000000,
            )
            message_bus.publish(Event.create(EventType.TICK, "test", {"tick": tick1}))
            await asyncio.sleep(0.05)

            # Stop actor
            await actor.stop()

            # Publish second tick (actor should not receive)
            tick2 = Tick(
                symbol="BTC/USDT",
                bid=Decimal("51000"),
                ask=Decimal("51001"),
                last=Decimal("51000"),
                timestamp_ns=2000000000,
            )
            message_bus.publish(Event.create(EventType.TICK, "test", {"tick": tick2}))
            await asyncio.sleep(0.05)

        # Actor should only have received the first tick
        assert len(actor.events_received) == 1

    @pytest.mark.asyncio
    async def test_actor_ignores_events_when_not_running(
        self, message_bus: MessageBus
    ) -> None:
        """Test that actor ignores events when not in RUNNING state."""
        actor = TickCollectorActor()
        await actor.initialize(message_bus)

        # Manually subscribe without starting
        actor.bus.subscribe(EventType.TICK, actor._handle_event)

        async with message_bus:
            # Actor is in READY state, not RUNNING
            assert actor.state == ComponentState.READY

            tick = Tick(
                symbol="BTC/USDT",
                bid=Decimal("50000"),
                ask=Decimal("50001"),
                last=Decimal("50000"),
                timestamp_ns=1000000000,
            )
            message_bus.publish(Event.create(EventType.TICK, "test", {"tick": tick}))
            await asyncio.sleep(0.05)

        # Actor should not have processed the event
        assert len(actor.events_received) == 0

    @pytest.mark.asyncio
    async def test_event_chain_processing(self, message_bus: MessageBus) -> None:
        """Test a chain of actors processing events."""
        # Actor 1: Receives ticks, publishes bars
        class TickToBarActor(BaseActor):
            def __init__(self) -> None:
                super().__init__()
                self.tick_count = 0

            @property
            def name(self) -> str:
                return "tick_to_bar"

            async def on_start(self) -> None:
                await self.subscribe(EventType.TICK)

            async def on_event(self, event: Event) -> None:
                if event.event_type == EventType.TICK:
                    self.tick_count += 1
                    # Convert tick to bar (simplified)
                    bar = Bar(
                        symbol="BTC/USDT",
                        timestamp_ns=event.timestamp_ns,
                        open=Decimal("50000"),
                        high=Decimal("50100"),
                        low=Decimal("49900"),
                        close=Decimal("50050"),
                        volume=Decimal("100"),
                        timeframe="1m",
                    )
                    self.publish_event(EventType.BAR, {"bar": bar})

        # Actor 2: Receives bars, publishes signals
        class BarToSignalActor(BaseActor):
            def __init__(self) -> None:
                super().__init__()
                self.bar_count = 0

            @property
            def name(self) -> str:
                return "bar_to_signal"

            async def on_start(self) -> None:
                await self.subscribe(EventType.BAR)

            async def on_event(self, event: Event) -> None:
                if event.event_type == EventType.BAR:
                    self.bar_count += 1
                    self.publish_event(EventType.SIGNAL, {"action": "BUY"})

        # Actor 3: Receives signals
        class SignalLoggerActor(BaseActor):
            def __init__(self) -> None:
                super().__init__()
                self.signal_count = 0

            @property
            def name(self) -> str:
                return "signal_logger"

            async def on_start(self) -> None:
                await self.subscribe(EventType.SIGNAL)

            async def on_event(self, event: Event) -> None:
                if event.event_type == EventType.SIGNAL:
                    self.signal_count += 1

        tick_to_bar = TickToBarActor()
        bar_to_signal = BarToSignalActor()
        signal_logger = SignalLoggerActor()

        await tick_to_bar.initialize(message_bus)
        await bar_to_signal.initialize(message_bus)
        await signal_logger.initialize(message_bus)

        async with message_bus:
            await tick_to_bar.start()
            await bar_to_signal.start()
            await signal_logger.start()

            # Publish a tick - should cascade through all actors
            tick = Tick(
                symbol="BTC/USDT",
                bid=Decimal("50000"),
                ask=Decimal("50001"),
                last=Decimal("50000"),
                timestamp_ns=1000000000,
            )
            message_bus.publish(Event.create(EventType.TICK, "test", {"tick": tick}))

            # Wait for cascade
            await asyncio.sleep(0.15)

            await tick_to_bar.stop()
            await bar_to_signal.stop()
            await signal_logger.stop()

        # Verify the chain processed correctly
        assert tick_to_bar.tick_count == 1
        assert bar_to_signal.bar_count == 1
        assert signal_logger.signal_count == 1

    @pytest.mark.asyncio
    async def test_high_volume_events(self, message_bus: MessageBus) -> None:
        """Test actor handling high volume of events."""
        actor = TickCollectorActor()
        await actor.initialize(message_bus)

        num_events = 100

        async with message_bus:
            await actor.start()

            # Publish many events rapidly
            for i in range(num_events):
                tick = Tick(
                    symbol="BTC/USDT",
                    bid=Decimal(str(50000 + i)),
                    ask=Decimal(str(50001 + i)),
                    last=Decimal(str(50000.5 + i)),
                    timestamp_ns=1000000000 + i,
                )
                message_bus.publish(Event.create(EventType.TICK, "test", {"tick": tick}))

            # Wait for all events to be processed
            await asyncio.sleep(0.5)

            await actor.stop()

        # All events should be received
        assert len(actor.events_received) == num_events

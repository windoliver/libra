"""
Integration tests for MessageBus.

Tests realistic trading event flows and priority handling.
"""

from __future__ import annotations

import asyncio

import pytest

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus


class TestTradingFlow:
    """Integration tests for complete trading event flows."""

    @pytest.mark.asyncio
    async def test_full_trading_flow(self) -> None:
        """
        Test a realistic trading event flow:
        1. Market data tick arrives
        2. Strategy generates signal
        3. Order is created
        4. Order is filled
        5. Position is updated

        All events share the same trace_id for correlation.
        """
        bus = MessageBus()
        flow: list[str] = []
        trace_ids: list[str] = []

        async def on_tick(event: Event) -> None:
            flow.append("tick")
            trace_ids.append(event.trace_id)
            # Strategy generates signal
            signal = event.child_event(
                EventType.SIGNAL,
                "strategy.sma_cross",
                {"action": "buy", "symbol": "BTC/USDT"},
            )
            bus.publish(signal)

        async def on_signal(event: Event) -> None:
            flow.append("signal")
            trace_ids.append(event.trace_id)
            # Executor creates order
            order = event.child_event(
                EventType.ORDER_NEW,
                "executor",
                {"order_id": "123", "symbol": "BTC/USDT"},
            )
            bus.publish(order)

        async def on_order(event: Event) -> None:
            flow.append("order")
            trace_ids.append(event.trace_id)
            # Gateway fills order
            fill = event.child_event(
                EventType.ORDER_FILLED,
                "gateway.binance",
                {"order_id": "123", "fill_price": 50000.0},
            )
            bus.publish(fill)

        async def on_fill(event: Event) -> None:
            flow.append("fill")
            trace_ids.append(event.trace_id)
            # Position manager updates position
            position = event.child_event(
                EventType.POSITION_OPENED,
                "position_manager",
                {"symbol": "BTC/USDT", "size": 0.1},
            )
            bus.publish(position)

        async def on_position(event: Event) -> None:
            flow.append("position")
            trace_ids.append(event.trace_id)

        # Subscribe handlers
        bus.subscribe(EventType.TICK, on_tick)
        bus.subscribe(EventType.SIGNAL, on_signal)
        bus.subscribe(EventType.ORDER_NEW, on_order)
        bus.subscribe(EventType.ORDER_FILLED, on_fill)
        bus.subscribe(EventType.POSITION_OPENED, on_position)

        # Run the flow
        async with bus:
            # Initial tick
            tick = Event.create(
                EventType.TICK,
                "gateway.binance",
                {"symbol": "BTC/USDT", "price": 50000.0},
            )
            bus.publish(tick)

            # Wait for full flow to complete
            await asyncio.sleep(0.5)

        # Verify complete flow executed in order
        assert flow == ["tick", "signal", "order", "fill", "position"]

        # All events should share the same trace_id (correlation)
        assert len(set(trace_ids)) == 1, "All events should have same trace_id"

        # Verify all events were dispatched
        assert bus.stats["dispatched"] >= 5

    @pytest.mark.asyncio
    async def test_multiple_concurrent_flows(self) -> None:
        """Test multiple trading flows happening concurrently."""
        bus = MessageBus()
        completed_flows: list[str] = []

        async def on_tick(event: Event) -> None:
            symbol = event.payload["symbol"]
            signal = event.child_event(
                EventType.SIGNAL,
                "strategy",
                {"action": "buy", "symbol": symbol},
            )
            bus.publish(signal)

        async def on_signal(event: Event) -> None:
            symbol = event.payload["symbol"]
            completed_flows.append(symbol)

        bus.subscribe(EventType.TICK, on_tick)
        bus.subscribe(EventType.SIGNAL, on_signal)

        async with bus:
            # Publish ticks for multiple symbols
            for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
                tick = Event.create(
                    EventType.TICK,
                    "gateway",
                    {"symbol": symbol, "price": 100.0},
                )
                bus.publish(tick)

            await asyncio.sleep(0.3)

        # All symbols should complete their flow
        assert len(completed_flows) == 3
        assert set(completed_flows) == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}


class TestPriorityHandling:
    """Integration tests for priority-based event handling."""

    @pytest.mark.asyncio
    async def test_risk_events_prioritized(self) -> None:
        """Risk events should always be processed first."""
        bus = MessageBus()
        order: list[str] = []

        async def handler(event: Event) -> None:
            order.append(event.event_type.name)

        bus.subscribe(EventType.TICK, handler)
        bus.subscribe(EventType.CIRCUIT_BREAKER, handler)
        bus.subscribe(EventType.ORDER_NEW, handler)
        bus.subscribe(EventType.SIGNAL, handler)

        # Publish in worst-case order (lowest priority first)
        bus.publish(Event.create(EventType.TICK, "1"))
        bus.publish(Event.create(EventType.SIGNAL, "2"))
        bus.publish(Event.create(EventType.ORDER_NEW, "3"))
        bus.publish(Event.create(EventType.CIRCUIT_BREAKER, "4"))

        async with bus:
            await asyncio.sleep(0.2)

        # Priority order: RISK(0) > ORDERS(1) > SIGNALS(2) > MARKET_DATA(3)
        assert order[0] == "CIRCUIT_BREAKER"
        assert order[1] == "ORDER_NEW"
        assert order[2] == "SIGNAL"
        assert order[3] == "TICK"

    @pytest.mark.asyncio
    async def test_circuit_breaker_halts_trading(self) -> None:
        """Circuit breaker should be processed before pending orders."""
        bus = MessageBus()
        events_processed: list[tuple[str, int]] = []

        async def handler(event: Event) -> None:
            events_processed.append((event.event_type.name, event.payload.get("seq", 0)))

        bus.subscribe(EventType.ORDER_NEW, handler)
        bus.subscribe(EventType.CIRCUIT_BREAKER, handler)

        # Simulate: many orders in queue, then circuit breaker triggers
        for i in range(10):
            bus.publish(Event.create(EventType.ORDER_NEW, "executor", {"seq": i}))

        # Circuit breaker triggered!
        bus.publish(Event.create(EventType.CIRCUIT_BREAKER, "risk", {"reason": "drawdown"}))

        async with bus:
            await asyncio.sleep(0.2)

        # Circuit breaker should be first even though it was published last
        assert events_processed[0][0] == "CIRCUIT_BREAKER"


class TestErrorIsolation:
    """Integration tests for error handling and isolation."""

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_break_flow(self) -> None:
        """A failing handler shouldn't stop other handlers."""
        bus = MessageBus()
        successful_handlers: list[str] = []

        async def failing_handler(event: Event) -> None:
            raise ValueError("Simulated failure")

        async def success_handler_1(event: Event) -> None:
            successful_handlers.append("handler1")

        async def success_handler_2(event: Event) -> None:
            successful_handlers.append("handler2")

        bus.subscribe(EventType.TICK, failing_handler)
        bus.subscribe(EventType.TICK, success_handler_1)
        bus.subscribe(EventType.TICK, success_handler_2)

        async with bus:
            bus.publish(Event.create(EventType.TICK, "test"))
            await asyncio.sleep(0.1)

        # Both successful handlers should have run
        assert "handler1" in successful_handlers
        assert "handler2" in successful_handlers
        assert bus.stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_flow_continues_after_handler_error(self) -> None:
        """Event flow should continue even if one handler fails."""
        bus = MessageBus()
        flow: list[str] = []

        async def on_tick(event: Event) -> None:
            flow.append("tick")
            raise ValueError("Tick handler error")

        async def on_tick_backup(event: Event) -> None:
            flow.append("tick_backup")
            # Still generate signal
            signal = event.child_event(EventType.SIGNAL, "backup", {"action": "buy"})
            bus.publish(signal)

        async def on_signal(event: Event) -> None:
            flow.append("signal")

        bus.subscribe(EventType.TICK, on_tick)
        bus.subscribe(EventType.TICK, on_tick_backup)
        bus.subscribe(EventType.SIGNAL, on_signal)

        async with bus:
            bus.publish(Event.create(EventType.TICK, "test"))
            await asyncio.sleep(0.2)

        # Flow should continue despite error
        assert "tick" in flow
        assert "tick_backup" in flow
        assert "signal" in flow


class TestGracefulShutdown:
    """Integration tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_pending_events_drained_on_shutdown(self) -> None:
        """All pending events should be processed before shutdown."""
        bus = MessageBus()
        processed: list[int] = []

        async def handler(event: Event) -> None:
            processed.append(event.payload["seq"])

        bus.subscribe(EventType.TICK, handler)

        # Publish events before starting bus
        for i in range(100):
            bus.publish(Event.create(EventType.TICK, "test", {"seq": i}))

        async with bus:
            await asyncio.sleep(0.3)

        # All events should be processed
        assert len(processed) == 100
        assert bus.total_pending == 0

    @pytest.mark.asyncio
    async def test_no_new_events_accepted_during_shutdown(self) -> None:
        """Events published during shutdown should be rejected."""
        bus = MessageBus()
        processed: list[int] = []

        async def handler(event: Event) -> None:
            processed.append(event.payload["seq"])

        bus.subscribe(EventType.TICK, handler)

        async with bus:
            bus.publish(Event.create(EventType.TICK, "test", {"seq": 1}))
            await asyncio.sleep(0.1)

        # Bus is now stopped
        result = bus.publish(Event.create(EventType.TICK, "test", {"seq": 2}))
        assert result is False  # Should be rejected


class TestEventCorrelation:
    """Integration tests for event correlation via trace_id."""

    @pytest.mark.asyncio
    async def test_trace_id_preserved_across_chain(self) -> None:
        """Child events should inherit parent's trace_id."""
        bus = MessageBus()
        collected_trace_ids: dict[str, str] = {}

        async def on_tick(event: Event) -> None:
            collected_trace_ids["tick"] = event.trace_id
            signal = event.child_event(EventType.SIGNAL, "strategy", {})
            bus.publish(signal)

        async def on_signal(event: Event) -> None:
            collected_trace_ids["signal"] = event.trace_id
            order = event.child_event(EventType.ORDER_NEW, "executor", {})
            bus.publish(order)

        async def on_order(event: Event) -> None:
            collected_trace_ids["order"] = event.trace_id

        bus.subscribe(EventType.TICK, on_tick)
        bus.subscribe(EventType.SIGNAL, on_signal)
        bus.subscribe(EventType.ORDER_NEW, on_order)

        async with bus:
            tick = Event.create(EventType.TICK, "gateway", {})
            bus.publish(tick)
            await asyncio.sleep(0.2)

        # All events in the chain should share the same trace_id
        assert len(collected_trace_ids) == 3
        trace_ids = set(collected_trace_ids.values())
        assert len(trace_ids) == 1, f"Expected 1 unique trace_id, got {trace_ids}"

    @pytest.mark.asyncio
    async def test_independent_flows_have_different_trace_ids(self) -> None:
        """Separate event flows should have different trace_ids."""
        bus = MessageBus()
        trace_ids: list[str] = []

        async def handler(event: Event) -> None:
            trace_ids.append(event.trace_id)

        bus.subscribe(EventType.TICK, handler)

        async with bus:
            # Two independent ticks
            bus.publish(Event.create(EventType.TICK, "gateway1", {}))
            bus.publish(Event.create(EventType.TICK, "gateway2", {}))
            await asyncio.sleep(0.1)

        # Should have 2 different trace_ids
        assert len(trace_ids) == 2
        assert trace_ids[0] != trace_ids[1]

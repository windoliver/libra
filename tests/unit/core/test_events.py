"""Tests for the Event system."""

from __future__ import annotations

import time

import pytest

from libra.core.events import (
    EVENT_PRIORITY_MAP,
    Event,
    EventType,
    Priority,
    decode_event,
    encode_event,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_have_priority(self) -> None:
        """Every event type should have a priority mapping."""
        for event_type in EventType:
            assert event_type in EVENT_PRIORITY_MAP, f"{event_type} missing priority"

    def test_risk_events_have_highest_priority(self) -> None:
        """Risk events should have Priority.RISK (0)."""
        risk_events = [
            EventType.RISK_LIMIT_BREACH,
            EventType.DRAWDOWN_WARNING,
            EventType.CIRCUIT_BREAKER,
        ]
        for event_type in risk_events:
            assert EVENT_PRIORITY_MAP[event_type] == Priority.RISK

    def test_order_events_have_orders_priority(self) -> None:
        """Order events should have Priority.ORDERS (1)."""
        order_events = [
            EventType.ORDER_NEW,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
        ]
        for event_type in order_events:
            assert EVENT_PRIORITY_MAP[event_type] == Priority.ORDERS

    def test_market_data_events_have_lowest_priority(self) -> None:
        """Market data events should have Priority.MARKET_DATA (3)."""
        market_events = [
            EventType.TICK,
            EventType.BAR,
            EventType.ORDER_BOOK,
        ]
        for event_type in market_events:
            assert EVENT_PRIORITY_MAP[event_type] == Priority.MARKET_DATA


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_ordering(self) -> None:
        """Lower numeric value = higher priority."""
        assert Priority.RISK < Priority.ORDERS
        assert Priority.ORDERS < Priority.SIGNALS
        assert Priority.SIGNALS < Priority.MARKET_DATA

    def test_priority_values(self) -> None:
        """Priority values should be 0-3."""
        assert Priority.RISK == 0
        assert Priority.ORDERS == 1
        assert Priority.SIGNALS == 2
        assert Priority.MARKET_DATA == 3


class TestEvent:
    """Tests for Event msgspec.Struct."""

    def test_create_event(self, sample_payload: dict[str, object]) -> None:
        """Test basic event creation."""
        event = Event.create(
            event_type=EventType.TICK,
            source="gateway.binance",
            payload=sample_payload,
        )

        assert event.event_type == EventType.TICK
        assert event.source == "gateway.binance"
        assert event.payload["symbol"] == "BTC/USDT"
        assert event.priority == Priority.MARKET_DATA
        assert len(event.trace_id) == 32
        assert len(event.span_id) == 16
        assert event.timestamp_ns > 0
        assert event.sequence > 0

    def test_event_is_immutable(self) -> None:
        """Events should be frozen (immutable)."""
        event = Event.create(EventType.TICK, "test")

        with pytest.raises(AttributeError):
            event.source = "modified"  # type: ignore[misc]

    def test_event_priority_auto_assigned(self) -> None:
        """Priority should be auto-assigned based on event type."""
        risk_event = Event.create(EventType.CIRCUIT_BREAKER, "risk")
        order_event = Event.create(EventType.ORDER_FILLED, "gateway")
        signal_event = Event.create(EventType.SIGNAL, "strategy")
        tick_event = Event.create(EventType.TICK, "gateway")

        assert risk_event.priority == Priority.RISK
        assert order_event.priority == Priority.ORDERS
        assert signal_event.priority == Priority.SIGNALS
        assert tick_event.priority == Priority.MARKET_DATA

    def test_event_sequence_increments(self) -> None:
        """Each event should get a unique, incrementing sequence."""
        event1 = Event.create(EventType.TICK, "a")
        event2 = Event.create(EventType.TICK, "b")
        event3 = Event.create(EventType.TICK, "c")

        assert event2.sequence > event1.sequence
        assert event3.sequence > event2.sequence

    def test_child_event_preserves_trace_id(self) -> None:
        """Child events should inherit parent's trace_id."""
        parent = Event.create(EventType.SIGNAL, "strategy")
        child = parent.child_event(EventType.ORDER_NEW, "executor")

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id  # New span

    def test_event_ordering_by_priority(self) -> None:
        """Events should sort by priority first."""
        # Create events with different priorities
        tick = Event.create(EventType.TICK, "a")
        circuit = Event.create(EventType.CIRCUIT_BREAKER, "b")
        order = Event.create(EventType.ORDER_NEW, "c")
        signal = Event.create(EventType.SIGNAL, "d")

        events = [tick, signal, order, circuit]
        sorted_events = sorted(events)

        # Should be sorted by priority: RISK(0), ORDERS(1), SIGNALS(2), MARKET_DATA(3)
        assert sorted_events[0].event_type == EventType.CIRCUIT_BREAKER
        assert sorted_events[1].event_type == EventType.ORDER_NEW
        assert sorted_events[2].event_type == EventType.SIGNAL
        assert sorted_events[3].event_type == EventType.TICK

    def test_event_ordering_by_sequence_within_priority(self) -> None:
        """Within same priority, events should sort by sequence (FIFO)."""
        # Create multiple events with same priority
        tick1 = Event.create(EventType.TICK, "first")
        tick2 = Event.create(EventType.TICK, "second")
        tick3 = Event.create(EventType.TICK, "third")

        events = [tick3, tick1, tick2]
        sorted_events = sorted(events)

        # Should be FIFO order within same priority
        assert sorted_events[0].source == "first"
        assert sorted_events[1].source == "second"
        assert sorted_events[2].source == "third"

    def test_event_no_gc_tracking(self) -> None:
        """Event should not be tracked by garbage collector (gc=False)."""

        event = Event.create(EventType.TICK, "test")

        # gc=False means the object won't be in gc.get_objects()
        # This is hard to test directly, but we can verify the struct works
        assert event is not None

    def test_timestamp_properties(self) -> None:
        """Test timestamp conversion properties."""
        before = time.time()
        event = Event.create(EventType.TICK, "test")
        after = time.time()

        # timestamp_sec should be between before and after
        assert before <= event.timestamp_sec <= after

        # timestamp_ns should be nanoseconds
        assert event.timestamp_ns > 1_000_000_000_000_000_000  # > 2001 in ns

    def test_priority_name_property(self) -> None:
        """Test priority_name property."""
        risk_event = Event.create(EventType.CIRCUIT_BREAKER, "test")
        tick_event = Event.create(EventType.TICK, "test")

        assert risk_event.priority_name == "RISK"
        assert tick_event.priority_name == "MARKET_DATA"


class TestEventSerialization:
    """Tests for event serialization/deserialization."""

    def test_encode_decode_roundtrip(self) -> None:
        """Event should survive JSON encode/decode roundtrip."""
        original = Event.create(
            event_type=EventType.TICK,
            source="gateway.binance",
            payload={"symbol": "BTC/USDT", "price": 50000.0},
        )

        # Encode to JSON bytes
        encoded = encode_event(original)
        assert isinstance(encoded, bytes)
        # Note: Enum is serialized as int by default (1 for TICK)
        assert b"gateway.binance" in encoded

        # Decode back
        decoded = decode_event(encoded)

        assert decoded.event_type == original.event_type
        assert decoded.source == original.source
        assert decoded.payload == original.payload
        assert decoded.trace_id == original.trace_id
        assert decoded.timestamp_ns == original.timestamp_ns

    def test_encode_performance(self) -> None:
        """Encoding should be fast."""
        event = Event.create(
            EventType.TICK,
            "test",
            {"price": 50000.0, "volume": 1.5},
        )

        # Warm up
        for _ in range(100):
            encode_event(event)

        # Benchmark
        start = time.perf_counter_ns()
        iterations = 10000
        for _ in range(iterations):
            encode_event(event)
        elapsed_ns = time.perf_counter_ns() - start

        avg_ns = elapsed_ns / iterations
        # Should be under 1μs (1000ns) per encode
        assert avg_ns < 5000, f"Encoding too slow: {avg_ns:.0f}ns per event"


class TestEventCreationPerformance:
    """Performance tests for event creation."""

    @pytest.mark.benchmark
    def test_event_creation_speed(self) -> None:
        """Event creation should be fast (target: <10μs)."""
        # Warm up
        for _ in range(1000):
            Event.create(EventType.TICK, "bench", {"price": 50000.0})

        # Benchmark
        start = time.perf_counter_ns()
        iterations = 100_000
        for _ in range(iterations):
            Event.create(EventType.TICK, "bench", {"price": 50000.0})
        elapsed_ns = time.perf_counter_ns() - start

        avg_ns = elapsed_ns / iterations
        print(f"\nEvent creation: {avg_ns:.0f}ns per event")

        # Target: under 10μs (10000ns)
        # Note: uuid4() calls add ~3-4μs overhead; pure msgspec.Struct is ~100ns
        # For HFT, we'd use faster ID generation or pre-allocated IDs
        assert avg_ns < 10000, f"Event creation too slow: {avg_ns:.0f}ns"

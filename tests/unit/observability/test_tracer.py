"""
Tests for distributed tracing (Issue #25).
"""

import pytest
import time
import uuid

from libra.observability.tracer import (
    Span,
    SpanStatus,
    Trace,
    TraceRegistry,
    get_registry,
    set_registry,
)


class TestSpan:
    """Tests for Span."""

    def test_span_init(self):
        """Span initializes correctly."""
        span = Span(
            span_id="span123",
            trace_id="trace456",
            parent_span_id=None,
            name="test_operation",
            start_time=time.time(),
        )

        assert span.span_id == "span123"
        assert span.trace_id == "trace456"
        assert span.parent_span_id is None
        assert span.name == "test_operation"
        assert span.status == SpanStatus.UNSET
        assert span.is_active

    def test_span_finish(self):
        """Span finishes correctly."""
        span = Span(
            span_id="span123",
            trace_id="trace456",
            parent_span_id=None,
            name="test",
            start_time=time.time(),
        )

        span.finish(SpanStatus.OK)

        assert not span.is_active
        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    def test_span_duration(self):
        """Span calculates duration."""
        start = time.time()
        span = Span(
            span_id="span123",
            trace_id="trace456",
            parent_span_id=None,
            name="test",
            start_time=start,
        )

        # Active span has no duration
        assert span.duration_ms is None

        # Finish span
        time.sleep(0.01)  # 10ms
        span.finish()

        assert span.duration_ms is not None
        assert span.duration_ms >= 10  # At least 10ms

    def test_span_attributes(self):
        """Span stores attributes."""
        span = Span(
            span_id="span123",
            trace_id="trace456",
            parent_span_id=None,
            name="test",
            start_time=time.time(),
        )

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 42

    def test_span_events(self):
        """Span records events."""
        span = Span(
            span_id="span123",
            trace_id="trace456",
            parent_span_id=None,
            name="test",
            start_time=time.time(),
        )

        span.add_event("event1", {"detail": "info"})
        span.add_event("event2")

        assert len(span.events) == 2
        assert span.events[0]["name"] == "event1"
        assert span.events[0]["attributes"]["detail"] == "info"

    def test_span_to_dict(self):
        """Span exports to dictionary."""
        span = Span(
            span_id="span123",
            trace_id="trace456",
            parent_span_id="parent789",
            name="test",
            start_time=time.time(),
        )
        span.finish(SpanStatus.OK)

        data = span.to_dict()

        assert data["span_id"] == "span123"
        assert data["trace_id"] == "trace456"
        assert data["parent_span_id"] == "parent789"
        assert data["name"] == "test"
        assert data["status"] == "ok"
        assert data["duration_ms"] is not None


class TestTrace:
    """Tests for Trace."""

    def test_trace_init(self):
        """Trace initializes correctly."""
        trace = Trace(trace_id="trace123")

        assert trace.trace_id == "trace123"
        assert trace.root_span_id is None
        assert trace.span_count == 0

    def test_trace_add_span(self):
        """Trace adds spans."""
        trace = Trace(trace_id="trace123")

        span1 = Span(
            span_id="span1",
            trace_id="trace123",
            parent_span_id=None,
            name="root",
            start_time=time.time(),
        )
        span2 = Span(
            span_id="span2",
            trace_id="trace123",
            parent_span_id="span1",
            name="child",
            start_time=time.time(),
        )

        trace.add_span(span1)
        trace.add_span(span2)

        assert trace.span_count == 2
        assert trace.root_span_id == "span1"  # First span becomes root

    def test_trace_get_span(self):
        """Trace retrieves spans."""
        trace = Trace(trace_id="trace123")
        span = Span(
            span_id="span1",
            trace_id="trace123",
            parent_span_id=None,
            name="test",
            start_time=time.time(),
        )
        trace.add_span(span)

        assert trace.get_span("span1") is span
        assert trace.get_span("nonexistent") is None

    def test_trace_is_complete(self):
        """Trace checks completion status."""
        trace = Trace(trace_id="trace123")
        span = Span(
            span_id="span1",
            trace_id="trace123",
            parent_span_id=None,
            name="test",
            start_time=time.time(),
        )
        trace.add_span(span)

        assert not trace.is_complete

        span.finish()
        assert trace.is_complete

    def test_trace_duration(self):
        """Trace calculates duration."""
        trace = Trace(trace_id="trace123")

        # Unfinished trace has no duration
        assert trace.duration_ms is None

        time.sleep(0.01)
        trace.finish()

        assert trace.duration_ms is not None
        assert trace.duration_ms >= 10

    def test_trace_to_dict(self):
        """Trace exports to dictionary."""
        trace = Trace(trace_id="trace123", source="gateway")
        span = Span(
            span_id="span1",
            trace_id="trace123",
            parent_span_id=None,
            name="test",
            start_time=time.time(),
        )
        trace.add_span(span)
        span.finish()
        trace.finish()

        data = trace.to_dict()

        assert data["trace_id"] == "trace123"
        assert data["source"] == "gateway"
        assert data["span_count"] == 1
        assert "spans" in data


class TestTraceRegistry:
    """Tests for TraceRegistry."""

    def test_registry_init(self):
        """Registry initializes correctly."""
        registry = TraceRegistry(max_traces=100)
        stats = registry.get_stats()

        assert stats["max_traces"] == 100
        assert stats["current_traces"] == 0
        assert stats["total_recorded"] == 0

    def test_registry_start_span(self):
        """Registry creates spans."""
        registry = TraceRegistry()
        trace_id = uuid.uuid4().hex[:32]

        span = registry.start_span(trace_id, "test_operation")

        assert span.trace_id == trace_id
        assert span.name == "test_operation"
        assert span.is_active

        # Trace should be created
        trace = registry.get_trace(trace_id)
        assert trace is not None
        assert trace.span_count == 1

    def test_registry_finish_span(self):
        """Registry finishes spans."""
        registry = TraceRegistry()
        trace_id = uuid.uuid4().hex[:32]

        span = registry.start_span(trace_id, "test")
        registry.finish_span(span.span_id, SpanStatus.OK)

        # Verify span is finished
        updated_span = registry.get_span(span.span_id)
        assert not updated_span.is_active
        assert updated_span.status == SpanStatus.OK

    def test_registry_child_spans(self):
        """Registry creates child spans."""
        registry = TraceRegistry()
        trace_id = uuid.uuid4().hex[:32]

        parent = registry.start_span(trace_id, "parent")
        child = registry.start_span(trace_id, "child", parent_span_id=parent.span_id)

        assert child.parent_span_id == parent.span_id

        trace = registry.get_trace(trace_id)
        assert trace.span_count == 2

    def test_registry_get_trace_events(self):
        """Registry retrieves trace events in order."""
        registry = TraceRegistry()
        trace_id = uuid.uuid4().hex[:32]

        # Create spans with slight delays
        registry.start_span(trace_id, "first")
        time.sleep(0.001)
        registry.start_span(trace_id, "second")
        time.sleep(0.001)
        registry.start_span(trace_id, "third")

        events = registry.get_trace_events(trace_id)

        assert len(events) == 3
        assert events[0].name == "first"
        assert events[1].name == "second"
        assert events[2].name == "third"

    def test_registry_search(self):
        """Registry searches traces."""
        registry = TraceRegistry()

        # Create traces with different sources
        for source in ["gateway", "strategy", "gateway"]:
            trace_id = uuid.uuid4().hex[:32]
            registry.start_span(trace_id, "test", attributes={"source": source})
            # Manually set source on trace
            trace = registry.get_trace(trace_id)
            trace.source = source

        # Search by source
        gateway_traces = registry.search(source="gateway")
        assert len(gateway_traces) == 2

        strategy_traces = registry.search(source="strategy")
        assert len(strategy_traces) == 1

    def test_registry_search_time_range(self):
        """Registry searches by time range."""
        registry = TraceRegistry()

        # Create older trace
        trace_id1 = uuid.uuid4().hex[:32]
        registry.start_span(trace_id1, "old")

        time.sleep(0.02)
        cutoff = time.time()
        time.sleep(0.02)

        # Create newer trace
        trace_id2 = uuid.uuid4().hex[:32]
        registry.start_span(trace_id2, "new")

        # Search for traces after cutoff
        recent = registry.search(start_time=cutoff)
        assert len(recent) == 1

    def test_registry_eviction(self):
        """Registry evicts old traces when full."""
        registry = TraceRegistry(max_traces=5)

        # Create more traces than max
        trace_ids = []
        for i in range(10):
            trace_id = uuid.uuid4().hex[:32]
            trace_ids.append(trace_id)
            registry.start_span(trace_id, f"trace_{i}")

        stats = registry.get_stats()
        assert stats["current_traces"] == 5
        assert stats["total_evicted"] == 5

        # Old traces should be gone
        assert registry.get_trace(trace_ids[0]) is None
        # New traces should exist
        assert registry.get_trace(trace_ids[-1]) is not None

    def test_registry_clear(self):
        """Registry clears all traces."""
        registry = TraceRegistry()

        for i in range(5):
            registry.start_span(uuid.uuid4().hex[:32], f"trace_{i}")

        registry.clear()

        stats = registry.get_stats()
        assert stats["current_traces"] == 0


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry(self):
        """get_registry returns default instance."""
        registry = get_registry()
        assert registry is not None
        assert isinstance(registry, TraceRegistry)

    def test_set_registry(self):
        """set_registry changes default instance."""
        original = get_registry()
        new_registry = TraceRegistry(max_traces=50)

        set_registry(new_registry)
        assert get_registry() is new_registry

        # Restore original
        set_registry(original)

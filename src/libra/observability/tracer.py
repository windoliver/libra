"""
Distributed Tracing: Trace Registry and Span Management.

Provides:
- Trace registry for event correlation
- Span hierarchy tracking
- Query operations for debugging
- W3C Trace Context compatibility

See: https://github.com/windoliver/libra/issues/25
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from libra.core.events import Event

logger = logging.getLogger(__name__)


class SpanStatus(str, Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """
    A single span in a trace.

    Represents one operation (e.g., handler execution).

    Attributes:
        span_id: Unique span identifier
        trace_id: Parent trace identifier
        parent_span_id: Parent span (if nested)
        name: Operation name
        start_time: Start timestamp (seconds)
        end_time: End timestamp (seconds) or None if active
        status: Span status
        attributes: Additional metadata
        events: Events within this span
    """

    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def is_active(self) -> bool:
        """Check if span is still active."""
        return self.end_time is None

    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        """Mark span as finished."""
        self.end_time = time.time()
        self.status = status

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to this span."""
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class Trace:
    """
    A complete trace containing multiple spans.

    Represents a single request/event flow through the system.

    Attributes:
        trace_id: Unique trace identifier
        root_span_id: ID of the root span
        spans: All spans in this trace
        start_time: When trace started
        end_time: When trace ended
        source: Origin of the trace
        metadata: Additional trace metadata
    """

    trace_id: str
    root_span_id: str | None = None
    spans: dict[str, Span] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_span(self, span: Span) -> None:
        """Add a span to this trace."""
        self.spans[span.span_id] = span
        if self.root_span_id is None:
            self.root_span_id = span.span_id

    def get_span(self, span_id: str) -> Span | None:
        """Get a span by ID."""
        return self.spans.get(span_id)

    def finish(self) -> None:
        """Mark trace as complete."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float | None:
        """Total trace duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def span_count(self) -> int:
        """Number of spans in trace."""
        return len(self.spans)

    @property
    def is_complete(self) -> bool:
        """Check if all spans are finished."""
        return all(not span.is_active for span in self.spans.values())

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "trace_id": self.trace_id,
            "root_span_id": self.root_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "source": self.source,
            "span_count": self.span_count,
            "spans": {sid: s.to_dict() for sid, s in self.spans.items()},
            "metadata": self.metadata,
        }


class TraceRegistry:
    """
    Central registry for traces.

    Features:
    - Store traces in circular buffer (memory bounded)
    - Query traces by ID, time range, source
    - Span hierarchy tracking
    - Event correlation

    Example:
        registry = TraceRegistry(max_traces=10000)

        # Record event
        registry.record_event(event)

        # Query trace
        trace = registry.get_trace(trace_id)
        events = registry.get_trace_events(trace_id)

        # Search traces
        recent = registry.search(start_time=time.time() - 3600)
    """

    def __init__(self, max_traces: int = 10000) -> None:
        self._max_traces = max_traces
        self._traces: OrderedDict[str, Trace] = OrderedDict()
        self._span_to_trace: dict[str, str] = {}  # span_id -> trace_id
        self._lock = threading.Lock()

        # Statistics
        self._total_recorded = 0
        self._total_evicted = 0

    def record_event(self, event: Event) -> Span:
        """
        Record an event in the registry.

        Creates or updates trace and span.

        Args:
            event: Event to record

        Returns:
            Created span for this event
        """
        with self._lock:
            trace_id = event.trace_id
            span_id = event.span_id

            # Get or create trace
            if trace_id not in self._traces:
                self._traces[trace_id] = Trace(
                    trace_id=trace_id,
                    source=event.source,
                )
                # Evict old traces if needed
                while len(self._traces) > self._max_traces:
                    self._traces.popitem(last=False)
                    self._total_evicted += 1

            trace = self._traces[trace_id]

            # Create span for this event
            span = Span(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=None,  # Could track parent from event
                name=f"{event.event_type.name}",
                start_time=event.timestamp_sec,
                attributes={
                    "source": event.source,
                    "priority": event.priority_name,
                    "event_type": event.event_type.name,
                },
            )

            trace.add_span(span)
            self._span_to_trace[span_id] = trace_id
            self._total_recorded += 1

            return span

    def start_span(
        self,
        trace_id: str,
        name: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Start a new span in an existing trace.

        Args:
            trace_id: Trace to add span to
            name: Span name
            parent_span_id: Parent span ID
            attributes: Span attributes

        Returns:
            New span
        """
        import uuid

        span_id = uuid.uuid4().hex[:16]

        with self._lock:
            if trace_id not in self._traces:
                self._traces[trace_id] = Trace(trace_id=trace_id)

            span = Span(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                name=name,
                start_time=time.time(),
                attributes=attributes or {},
            )

            self._traces[trace_id].add_span(span)
            self._span_to_trace[span_id] = trace_id

            return span

    def finish_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.OK,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Finish a span.

        Args:
            span_id: Span to finish
            status: Final status
            attributes: Additional attributes to add
        """
        with self._lock:
            trace_id = self._span_to_trace.get(span_id)
            if trace_id and trace_id in self._traces:
                span = self._traces[trace_id].get_span(span_id)
                if span:
                    span.finish(status)
                    if attributes:
                        span.attributes.update(attributes)

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get trace by ID."""
        with self._lock:
            return self._traces.get(trace_id)

    def get_span(self, span_id: str) -> Span | None:
        """Get span by ID."""
        with self._lock:
            trace_id = self._span_to_trace.get(span_id)
            if trace_id and trace_id in self._traces:
                return self._traces[trace_id].get_span(span_id)
            return None

    def get_trace_events(self, trace_id: str) -> list[Span]:
        """
        Get all spans (events) in a trace.

        Returns list sorted by start time.
        """
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return []
            return sorted(trace.spans.values(), key=lambda s: s.start_time)

    def search(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[Trace]:
        """
        Search traces by criteria.

        Args:
            start_time: Filter by start time (>= this)
            end_time: Filter by end time (<= this)
            source: Filter by source
            limit: Max results

        Returns:
            List of matching traces
        """
        with self._lock:
            results = []
            for trace in reversed(self._traces.values()):
                if start_time and trace.start_time < start_time:
                    continue
                if end_time and trace.start_time > end_time:
                    continue
                if source and trace.source != source:
                    continue
                results.append(trace)
                if len(results) >= limit:
                    break
            return results

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            return {
                "total_recorded": self._total_recorded,
                "total_evicted": self._total_evicted,
                "current_traces": len(self._traces),
                "max_traces": self._max_traces,
                "total_spans": len(self._span_to_trace),
            }

    def clear(self) -> None:
        """Clear all traces (for testing)."""
        with self._lock:
            self._traces.clear()
            self._span_to_trace.clear()


# =============================================================================
# Global Registry Instance
# =============================================================================

_default_registry: TraceRegistry | None = None


def get_registry() -> TraceRegistry:
    """Get the default trace registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = TraceRegistry()
    return _default_registry


def set_registry(registry: TraceRegistry) -> None:
    """Set the default trace registry."""
    global _default_registry
    _default_registry = registry

"""
Trace Viewer Widget for Observability (Issue #25).

Displays distributed traces with span hierarchies:
- Trace list with filtering
- Span timeline visualization
- Span details with attributes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import DataTable, Rule, Static, Tree
from textual.widgets.tree import TreeNode


@dataclass
class SpanData:
    """Data for a single span."""

    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: str = "ok"  # ok, error, unset
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TraceData:
    """Data for a complete trace."""

    trace_id: str
    root_span_id: str | None = None
    spans: dict[str, SpanData] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float | None = None
    duration_ms: float | None = None
    source: str = ""
    span_count: int = 0


class TraceListPanel(Vertical):
    """Panel showing list of recent traces."""

    DEFAULT_CSS = """
    TraceListPanel {
        width: 1fr;
        min-width: 40;
        height: 100%;
        border: round $primary-darken-1;
    }

    TraceListPanel > Static.panel-title {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }

    TraceListPanel > DataTable {
        height: 1fr;
    }
    """

    class TraceSelected(Message):
        """Emitted when a trace is selected."""

        def __init__(self, trace_id: str) -> None:
            super().__init__()
            self.trace_id = trace_id

    def compose(self) -> ComposeResult:
        yield Static("RECENT TRACES", classes="panel-title")
        table = DataTable(id="trace-list")
        table.cursor_type = "row"
        table.add_columns("Time", "Source", "Spans", "Duration")
        yield table

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle trace selection."""
        table = self.query_one("#trace-list", DataTable)
        row_key = event.row_key
        if row_key:
            # row_key.value is the trace_id we stored
            trace_id = str(row_key.value)
            self.post_message(self.TraceSelected(trace_id))

    def update_traces(self, traces: list[TraceData]) -> None:
        """Update the trace list."""
        try:
            table = self.query_one("#trace-list", DataTable)
            table.clear()

            for trace in traces[:50]:  # Limit to 50 traces
                # Format time
                time_str = datetime.fromtimestamp(
                    trace.start_time, tz=timezone.utc
                ).strftime("%H:%M:%S")

                # Format duration
                if trace.duration_ms is not None:
                    if trace.duration_ms < 1:
                        dur_str = f"{trace.duration_ms * 1000:.0f}μs"
                    elif trace.duration_ms < 1000:
                        dur_str = f"{trace.duration_ms:.1f}ms"
                    else:
                        dur_str = f"{trace.duration_ms / 1000:.2f}s"
                else:
                    dur_str = "..."

                # Source - truncate if too long
                source = trace.source[:15] + "..." if len(trace.source) > 18 else trace.source

                table.add_row(
                    time_str,
                    source,
                    str(trace.span_count),
                    dur_str,
                    key=trace.trace_id,
                )
        except Exception:
            pass


class SpanTreePanel(Vertical):
    """Panel showing span hierarchy as a tree."""

    DEFAULT_CSS = """
    SpanTreePanel {
        width: 2fr;
        min-width: 50;
        height: 100%;
        border: round $primary-darken-1;
    }

    SpanTreePanel > Static.panel-title {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }

    SpanTreePanel > VerticalScroll {
        height: 1fr;
    }

    SpanTreePanel Tree {
        height: auto;
    }
    """

    class SpanSelected(Message):
        """Emitted when a span is selected."""

        def __init__(self, span_id: str, trace_id: str) -> None:
            super().__init__()
            self.span_id = span_id
            self.trace_id = trace_id

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._current_trace: TraceData | None = None

    def compose(self) -> ComposeResult:
        yield Static("SPAN HIERARCHY", classes="panel-title")
        with VerticalScroll():
            yield Tree("No trace selected", id="span-tree")

    def show_trace(self, trace: TraceData) -> None:
        """Display spans for a trace."""
        self._current_trace = trace
        try:
            tree = self.query_one("#span-tree", Tree)
            tree.clear()
            tree.root.set_label(f"Trace: {trace.trace_id[:8]}...")
            tree.root.expand()

            # Build span hierarchy
            self._build_span_tree(tree.root, trace)
        except Exception:
            pass

    def _build_span_tree(self, parent: TreeNode, trace: TraceData) -> None:
        """Build span tree recursively."""
        # Find root spans (no parent)
        root_spans = [
            span for span in trace.spans.values()
            if span.parent_span_id is None
        ]

        # Sort by start time
        root_spans.sort(key=lambda s: s.start_time)

        for span in root_spans:
            self._add_span_node(parent, span, trace)

    def _add_span_node(
        self, parent: TreeNode, span: SpanData, trace: TraceData
    ) -> None:
        """Add a span and its children to the tree."""
        # Format span label
        duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "..."

        # Status icon
        if span.status == "error":
            status_icon = "[red]✗[/red]"
        elif span.status == "ok":
            status_icon = "[green]✓[/green]"
        else:
            status_icon = "[dim]○[/dim]"

        label = f"{status_icon} {span.name} ({duration})"
        node = parent.add(label, data={"span_id": span.span_id, "trace_id": trace.trace_id})

        # Add child spans
        children = [
            s for s in trace.spans.values()
            if s.parent_span_id == span.span_id
        ]
        children.sort(key=lambda s: s.start_time)

        for child in children:
            self._add_span_node(node, child, trace)

        # Expand by default
        node.expand()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle span selection."""
        node = event.node
        if node.data and "span_id" in node.data:
            self.post_message(self.SpanSelected(
                node.data["span_id"],
                node.data["trace_id"],
            ))


class SpanDetailPanel(Vertical):
    """Panel showing span details."""

    DEFAULT_CSS = """
    SpanDetailPanel {
        width: 1fr;
        min-width: 40;
        height: 100%;
        border: round $primary-darken-1;
    }

    SpanDetailPanel > Static.panel-title {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }

    SpanDetailPanel > VerticalScroll {
        height: 1fr;
        padding: 1;
    }

    SpanDetailPanel .detail-label {
        color: $text-muted;
        text-style: bold;
    }

    SpanDetailPanel .detail-value {
        margin-left: 2;
        margin-bottom: 1;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._span: SpanData | None = None

    def compose(self) -> ComposeResult:
        yield Static("SPAN DETAILS", classes="panel-title")
        with VerticalScroll():
            yield Static("Select a span to view details", id="span-details")

    def show_span(self, span: SpanData) -> None:
        """Display span details."""
        self._span = span
        try:
            details = self.query_one("#span-details", Static)

            # Build details text
            lines = [
                f"[bold]Span ID:[/bold] {span.span_id}",
                f"[bold]Name:[/bold] {span.name}",
                f"[bold]Status:[/bold] {self._format_status(span.status)}",
                "",
                f"[bold]Duration:[/bold] {span.duration_ms:.2f}ms" if span.duration_ms else "[bold]Duration:[/bold] ...",
                f"[bold]Start:[/bold] {self._format_time(span.start_time)}",
            ]

            if span.end_time:
                lines.append(f"[bold]End:[/bold] {self._format_time(span.end_time)}")

            if span.parent_span_id:
                lines.append(f"[bold]Parent:[/bold] {span.parent_span_id[:8]}...")

            # Attributes
            if span.attributes:
                lines.append("")
                lines.append("[bold]Attributes:[/bold]")
                for key, value in span.attributes.items():
                    lines.append(f"  {key}: {value}")

            # Events
            if span.events:
                lines.append("")
                lines.append(f"[bold]Events ({len(span.events)}):[/bold]")
                for event in span.events[:5]:  # Limit to 5
                    lines.append(f"  • {event.get('name', 'unknown')}")

            details.update("\n".join(lines))
        except Exception:
            pass

    def _format_status(self, status: str) -> str:
        """Format status with color."""
        if status == "error":
            return "[red]ERROR[/red]"
        elif status == "ok":
            return "[green]OK[/green]"
        return "[dim]UNSET[/dim]"

    def _format_time(self, timestamp: float) -> str:
        """Format timestamp."""
        return datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).strftime("%H:%M:%S.%f")[:-3]


class TraceViewer(Horizontal):
    """
    Complete trace viewer for distributed tracing.

    Shows:
    - List of recent traces
    - Span hierarchy tree
    - Span details panel

    Example:
        viewer = TraceViewer(id="trace-viewer")

        # Update with traces from TraceRegistry
        traces = registry.search(limit=50)
        viewer.update_traces(traces)
    """

    DEFAULT_CSS = """
    TraceViewer {
        height: 100%;
        min-height: 20;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._traces: dict[str, TraceData] = {}

    def compose(self) -> ComposeResult:
        yield TraceListPanel(id="trace-list-panel")
        yield SpanTreePanel(id="span-tree-panel")
        yield SpanDetailPanel(id="span-detail-panel")

    def on_trace_list_panel_trace_selected(
        self, event: TraceListPanel.TraceSelected
    ) -> None:
        """Handle trace selection."""
        trace = self._traces.get(event.trace_id)
        if trace:
            try:
                self.query_one("#span-tree-panel", SpanTreePanel).show_trace(trace)
            except Exception:
                pass

    def on_span_tree_panel_span_selected(
        self, event: SpanTreePanel.SpanSelected
    ) -> None:
        """Handle span selection."""
        trace = self._traces.get(event.trace_id)
        if trace and event.span_id in trace.spans:
            span = trace.spans[event.span_id]
            try:
                self.query_one("#span-detail-panel", SpanDetailPanel).show_span(span)
            except Exception:
                pass

    def update_traces(self, traces: list[TraceData]) -> None:
        """Update with new trace data."""
        self._traces = {t.trace_id: t for t in traces}
        try:
            self.query_one("#trace-list-panel", TraceListPanel).update_traces(traces)
        except Exception:
            pass

    def update_from_registry(self, registry: Any) -> None:
        """
        Update from TraceRegistry.

        Args:
            registry: TraceRegistry instance
        """
        # Get recent traces
        raw_traces = registry.search(limit=50)

        # Convert to TraceData
        traces = []
        for raw in raw_traces:
            spans = {}
            for span_id, raw_span in raw.spans.items():
                spans[span_id] = SpanData(
                    span_id=raw_span.span_id,
                    trace_id=raw_span.trace_id,
                    parent_span_id=raw_span.parent_span_id,
                    name=raw_span.name,
                    start_time=raw_span.start_time,
                    end_time=raw_span.end_time,
                    duration_ms=raw_span.duration_ms,
                    status=raw_span.status.value if hasattr(raw_span.status, 'value') else raw_span.status,
                    attributes=raw_span.attributes,
                    events=raw_span.events,
                )

            traces.append(TraceData(
                trace_id=raw.trace_id,
                root_span_id=raw.root_span_id,
                spans=spans,
                start_time=raw.start_time,
                end_time=raw.end_time,
                duration_ms=raw.duration_ms,
                source=raw.source,
                span_count=raw.span_count,
            ))

        self.update_traces(traces)


def create_demo_trace_data() -> list[TraceData]:
    """Create demo trace data for testing."""
    import random
    import time
    import uuid

    traces = []
    sources = ["gateway.binance", "strategy.sma_cross", "risk.validator", "execution.twap"]

    for i in range(10):
        trace_id = uuid.uuid4().hex[:32]
        start = time.time() - random.uniform(0, 3600)

        # Create spans
        spans = {}
        root_id = uuid.uuid4().hex[:16]

        # Root span
        root_duration = random.uniform(1, 100)
        spans[root_id] = SpanData(
            span_id=root_id,
            trace_id=trace_id,
            parent_span_id=None,
            name=random.choice(["TICK", "ORDER_FILLED", "SIGNAL", "RISK_CHECK"]),
            start_time=start,
            end_time=start + root_duration / 1000,
            duration_ms=root_duration,
            status=random.choice(["ok", "ok", "ok", "error"]),
            attributes={
                "source": random.choice(sources),
                "priority": random.choice(["RISK", "ORDERS", "SIGNALS", "MARKET_DATA"]),
            },
        )

        # Child spans
        num_children = random.randint(0, 3)
        child_start = start + root_duration / 4000
        for j in range(num_children):
            child_id = uuid.uuid4().hex[:16]
            child_duration = random.uniform(0.1, root_duration / 2)
            spans[child_id] = SpanData(
                span_id=child_id,
                trace_id=trace_id,
                parent_span_id=root_id,
                name=random.choice(["validate", "persist", "notify", "calculate"]),
                start_time=child_start,
                end_time=child_start + child_duration / 1000,
                duration_ms=child_duration,
                status="ok",
            )
            child_start += child_duration / 1000

        traces.append(TraceData(
            trace_id=trace_id,
            root_span_id=root_id,
            spans=spans,
            start_time=start,
            end_time=start + root_duration / 1000,
            duration_ms=root_duration,
            source=random.choice(sources),
            span_count=len(spans),
        ))

    # Sort by start time (newest first)
    traces.sort(key=lambda t: t.start_time, reverse=True)
    return traces

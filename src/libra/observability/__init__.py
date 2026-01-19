"""
Observability Layer for LIBRA Trading Platform.

Provides:
- Distributed tracing with correlation IDs
- Metrics collection (counters, gauges, histograms)
- Event recording and replay
- Health monitoring
- Memory profiling with tracemalloc
- Prometheus/OpenTelemetry export

See: https://github.com/windoliver/libra/issues/25
"""

from libra.observability.collector import (
    Counter,
    Gauge,
    Histogram,
    HistogramBuckets,
    MetricsCollector,
    MetricType,
    get_collector,
    set_collector,
)
from libra.observability.health import (
    ComponentHealth,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    check_cpu,
    check_disk,
    check_memory,
    get_monitor,
    set_monitor,
)
from libra.observability.memory import (
    MemoryDiff,
    MemoryMonitor,
    MemorySnapshot,
    get_memory_monitor,
    set_memory_monitor,
)
from libra.observability.recorder import (
    EventRecorder,
    Recording,
    RecordingMetadata,
)
from libra.observability.replayer import (
    EventReplayer,
    ReplayConfig,
    ReplayMode,
    ReplayStats,
    quick_replay,
)
from libra.observability.tracer import (
    Span,
    SpanStatus,
    Trace,
    TraceRegistry,
    get_registry,
    set_registry,
)


__all__ = [
    # Metrics
    "Counter",
    "Gauge",
    "Histogram",
    "HistogramBuckets",
    "MetricsCollector",
    "MetricType",
    "get_collector",
    "set_collector",
    # Tracing
    "Span",
    "SpanStatus",
    "Trace",
    "TraceRegistry",
    "get_registry",
    "set_registry",
    # Recording
    "EventRecorder",
    "Recording",
    "RecordingMetadata",
    # Replay
    "EventReplayer",
    "ReplayConfig",
    "ReplayMode",
    "ReplayStats",
    "quick_replay",
    # Health
    "ComponentHealth",
    "HealthCheck",
    "HealthMonitor",
    "HealthStatus",
    "check_cpu",
    "check_disk",
    "check_memory",
    "get_monitor",
    "set_monitor",
    # Memory (Issue #92)
    "MemoryDiff",
    "MemoryMonitor",
    "MemorySnapshot",
    "get_memory_monitor",
    "set_memory_monitor",
]

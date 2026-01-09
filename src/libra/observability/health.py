"""
Health Monitoring: Component health checks and system monitoring.

Provides:
- Health check protocols
- Component health aggregation
- System resource monitoring
- Health status dashboard data

See: https://github.com/windoliver/libra/issues/25
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """
    Health status of a single component.

    Attributes:
        name: Component name
        status: Current health status
        message: Status message
        last_check: When last checked
        latency_ms: Health check latency
        details: Additional details
    """

    name: str
    status: HealthStatus
    message: str = ""
    last_check: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def age_seconds(self) -> float:
        """Time since last check."""
        return time.time() - self.last_check

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check,
            "latency_ms": self.latency_ms,
            "age_seconds": self.age_seconds,
            "details": self.details,
        }


@runtime_checkable
class HealthCheck(Protocol):
    """Protocol for health checkable components."""

    @property
    def name(self) -> str:
        """Component name."""
        ...

    async def health_check(self) -> ComponentHealth:
        """Perform health check."""
        ...


class HealthMonitor:
    """
    Central health monitoring for all components.

    Features:
    - Register health checkable components
    - Periodic health checks
    - Aggregated health status
    - Alerting hooks

    Example:
        monitor = HealthMonitor()

        # Register components
        monitor.register(gateway)
        monitor.register(message_bus)
        monitor.register(risk_engine)

        # Run periodic checks
        await monitor.start_monitoring(interval=30)

        # Get overall health
        health = await monitor.check_all()
        print(health["status"])  # "healthy" or "degraded" or "unhealthy"

        # Get specific component
        gateway_health = await monitor.check("gateway.binance")
    """

    def __init__(self) -> None:
        self._components: dict[str, HealthCheck] = {}
        self._custom_checks: dict[str, Callable[[], ComponentHealth]] = {}
        self._last_results: dict[str, ComponentHealth] = {}
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None

    def register(self, component: HealthCheck) -> None:
        """
        Register a health checkable component.

        Args:
            component: Component implementing HealthCheck protocol
        """
        with self._lock:
            self._components[component.name] = component
            logger.debug(f"Registered health check: {component.name}")

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register a custom health check function.

        Args:
            name: Check name
            check_fn: Function returning ComponentHealth
        """
        with self._lock:
            self._custom_checks[name] = check_fn
            logger.debug(f"Registered custom health check: {name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister a component.

        Args:
            name: Component name

        Returns:
            True if removed
        """
        with self._lock:
            if name in self._components:
                del self._components[name]
                return True
            if name in self._custom_checks:
                del self._custom_checks[name]
                return True
            return False

    async def check(self, name: str) -> ComponentHealth:
        """
        Check health of specific component.

        Args:
            name: Component name

        Returns:
            Component health
        """
        start_time = time.time()

        # Try registered component
        if name in self._components:
            try:
                health = await self._components[name].health_check()
                health.latency_ms = (time.time() - start_time) * 1000
                return health
            except Exception as e:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Try custom check
        if name in self._custom_checks:
            try:
                health = self._custom_checks[name]()
                health.latency_ms = (time.time() - start_time) * 1000
                return health
            except Exception as e:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

        return ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            message="Component not found",
        )

    async def check_all(self) -> dict[str, Any]:
        """
        Check health of all registered components.

        Returns:
            Aggregated health status with all component statuses
        """
        results: dict[str, ComponentHealth] = {}
        statuses: list[HealthStatus] = []

        # Check all components
        all_names = list(self._components.keys()) + list(self._custom_checks.keys())

        for name in all_names:
            health = await self.check(name)
            results[name] = health
            statuses.append(health.status)

        # Store results
        with self._lock:
            self._last_results = results

        # Determine overall status
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        return {
            "status": overall.value,
            "timestamp": time.time(),
            "components": {name: h.to_dict() for name, h in results.items()},
            "summary": {
                "total": len(results),
                "healthy": sum(1 for s in statuses if s == HealthStatus.HEALTHY),
                "degraded": sum(1 for s in statuses if s == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for s in statuses if s == HealthStatus.UNHEALTHY),
                "unknown": sum(1 for s in statuses if s == HealthStatus.UNKNOWN),
            },
        }

    async def start_monitoring(self, interval: float = 30.0) -> None:
        """
        Start periodic health monitoring.

        Args:
            interval: Check interval in seconds
        """
        if self._monitoring:
            return

        self._monitoring = True

        async def monitor_loop() -> None:
            while self._monitoring:
                try:
                    await self.check_all()
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")

                await asyncio.sleep(interval)

        self._monitor_task = asyncio.create_task(monitor_loop())
        logger.info(f"Started health monitoring (interval={interval}s)")

    async def stop_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Stopped health monitoring")

    def get_last_results(self) -> dict[str, ComponentHealth]:
        """Get results from last health check."""
        with self._lock:
            return dict(self._last_results)

    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._monitoring

    @property
    def component_count(self) -> int:
        """Number of registered components."""
        with self._lock:
            return len(self._components) + len(self._custom_checks)


# =============================================================================
# System Health Checks
# =============================================================================


def check_memory() -> ComponentHealth:
    """Check system memory usage."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        percent = memory.percent

        if percent < 80:
            status = HealthStatus.HEALTHY
            message = f"Memory usage: {percent:.1f}%"
        elif percent < 90:
            status = HealthStatus.DEGRADED
            message = f"High memory usage: {percent:.1f}%"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Critical memory usage: {percent:.1f}%"

        return ComponentHealth(
            name="system.memory",
            status=status,
            message=message,
            details={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent": percent,
            },
        )
    except ImportError:
        return ComponentHealth(
            name="system.memory",
            status=HealthStatus.UNKNOWN,
            message="psutil not installed",
        )
    except Exception as e:
        return ComponentHealth(
            name="system.memory",
            status=HealthStatus.UNKNOWN,
            message=str(e),
        )


def check_cpu() -> ComponentHealth:
    """Check CPU usage."""
    try:
        import psutil

        percent = psutil.cpu_percent(interval=0.1)

        if percent < 70:
            status = HealthStatus.HEALTHY
            message = f"CPU usage: {percent:.1f}%"
        elif percent < 90:
            status = HealthStatus.DEGRADED
            message = f"High CPU usage: {percent:.1f}%"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Critical CPU usage: {percent:.1f}%"

        return ComponentHealth(
            name="system.cpu",
            status=status,
            message=message,
            details={
                "percent": percent,
                "count": psutil.cpu_count(),
            },
        )
    except ImportError:
        return ComponentHealth(
            name="system.cpu",
            status=HealthStatus.UNKNOWN,
            message="psutil not installed",
        )
    except Exception as e:
        return ComponentHealth(
            name="system.cpu",
            status=HealthStatus.UNKNOWN,
            message=str(e),
        )


def check_disk(path: str = "/") -> ComponentHealth:
    """Check disk usage."""
    try:
        import psutil

        disk = psutil.disk_usage(path)
        percent = disk.percent

        if percent < 80:
            status = HealthStatus.HEALTHY
            message = f"Disk usage: {percent:.1f}%"
        elif percent < 90:
            status = HealthStatus.DEGRADED
            message = f"High disk usage: {percent:.1f}%"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Critical disk usage: {percent:.1f}%"

        return ComponentHealth(
            name="system.disk",
            status=status,
            message=message,
            details={
                "path": path,
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent": percent,
            },
        )
    except ImportError:
        return ComponentHealth(
            name="system.disk",
            status=HealthStatus.UNKNOWN,
            message="psutil not installed",
        )
    except Exception as e:
        return ComponentHealth(
            name="system.disk",
            status=HealthStatus.UNKNOWN,
            message=str(e),
        )


# =============================================================================
# Global Monitor Instance
# =============================================================================

_default_monitor: HealthMonitor | None = None


def get_monitor() -> HealthMonitor:
    """Get the default health monitor."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = HealthMonitor()
        # Register system checks
        _default_monitor.register_check("system.memory", check_memory)
        _default_monitor.register_check("system.cpu", check_cpu)
        _default_monitor.register_check("system.disk", check_disk)
    return _default_monitor


def set_monitor(monitor: HealthMonitor) -> None:
    """Set the default health monitor."""
    global _default_monitor
    _default_monitor = monitor

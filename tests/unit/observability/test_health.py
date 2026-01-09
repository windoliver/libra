"""
Tests for health monitoring (Issue #25).
"""

import pytest
import asyncio
import time

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


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Health status has expected values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Tests for ComponentHealth."""

    def test_health_init(self):
        """ComponentHealth initializes correctly."""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All good",
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All good"
        assert health.is_healthy

    def test_health_is_healthy(self):
        """is_healthy property works."""
        healthy = ComponentHealth(name="test", status=HealthStatus.HEALTHY)
        degraded = ComponentHealth(name="test", status=HealthStatus.DEGRADED)
        unhealthy = ComponentHealth(name="test", status=HealthStatus.UNHEALTHY)

        assert healthy.is_healthy
        assert not degraded.is_healthy
        assert not unhealthy.is_healthy

    def test_health_age(self):
        """age_seconds property works."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            last_check=time.time() - 10,
        )

        assert health.age_seconds >= 10

    def test_health_to_dict(self):
        """ComponentHealth exports to dictionary."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            latency_ms=5.5,
            details={"extra": "info"},
        )

        data = health.to_dict()

        assert data["name"] == "test"
        assert data["status"] == "healthy"
        assert data["message"] == "OK"
        assert data["latency_ms"] == 5.5
        assert data["details"]["extra"] == "info"


class MockHealthComponent:
    """Mock component implementing HealthCheck protocol."""

    def __init__(self, name: str, status: HealthStatus = HealthStatus.HEALTHY):
        self._name = name
        self._status = status

    @property
    def name(self) -> str:
        return self._name

    async def health_check(self) -> ComponentHealth:
        return ComponentHealth(
            name=self._name,
            status=self._status,
            message=f"{self._name} is {self._status.value}",
        )


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    def test_monitor_init(self):
        """HealthMonitor initializes correctly."""
        monitor = HealthMonitor()

        assert monitor.component_count == 0
        assert not monitor.is_monitoring

    def test_monitor_register_component(self):
        """Monitor registers components."""
        monitor = HealthMonitor()
        component = MockHealthComponent("test")

        monitor.register(component)

        assert monitor.component_count == 1

    def test_monitor_register_check(self):
        """Monitor registers custom checks."""
        monitor = HealthMonitor()

        def custom_check():
            return ComponentHealth(
                name="custom",
                status=HealthStatus.HEALTHY,
            )

        monitor.register_check("custom_check", custom_check)

        assert monitor.component_count == 1

    def test_monitor_unregister(self):
        """Monitor unregisters components."""
        monitor = HealthMonitor()
        component = MockHealthComponent("test")
        monitor.register(component)

        assert monitor.unregister("test")
        assert monitor.component_count == 0

        # Unregistering nonexistent returns False
        assert not monitor.unregister("nonexistent")

    @pytest.mark.asyncio
    async def test_monitor_check_single(self):
        """Monitor checks single component."""
        monitor = HealthMonitor()
        component = MockHealthComponent("test", HealthStatus.HEALTHY)
        monitor.register(component)

        health = await monitor.check("test")

        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_monitor_check_not_found(self):
        """Monitor returns unknown for missing component."""
        monitor = HealthMonitor()

        health = await monitor.check("nonexistent")

        assert health.status == HealthStatus.UNKNOWN
        assert "not found" in health.message.lower()

    @pytest.mark.asyncio
    async def test_monitor_check_error(self):
        """Monitor handles check errors."""
        monitor = HealthMonitor()

        def failing_check():
            raise RuntimeError("Check failed!")

        monitor.register_check("failing", failing_check)

        health = await monitor.check("failing")

        assert health.status == HealthStatus.UNHEALTHY
        assert "failed" in health.message.lower()

    @pytest.mark.asyncio
    async def test_monitor_check_all(self):
        """Monitor checks all components."""
        monitor = HealthMonitor()
        monitor.register(MockHealthComponent("comp1", HealthStatus.HEALTHY))
        monitor.register(MockHealthComponent("comp2", HealthStatus.HEALTHY))
        monitor.register(MockHealthComponent("comp3", HealthStatus.DEGRADED))

        result = await monitor.check_all()

        assert result["status"] == "degraded"  # Overall degraded due to comp3
        assert result["summary"]["total"] == 3
        assert result["summary"]["healthy"] == 2
        assert result["summary"]["degraded"] == 1
        assert "comp1" in result["components"]
        assert "comp2" in result["components"]
        assert "comp3" in result["components"]

    @pytest.mark.asyncio
    async def test_monitor_overall_status_healthy(self):
        """Monitor reports healthy when all healthy."""
        monitor = HealthMonitor()
        monitor.register(MockHealthComponent("comp1", HealthStatus.HEALTHY))
        monitor.register(MockHealthComponent("comp2", HealthStatus.HEALTHY))

        result = await monitor.check_all()

        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_monitor_overall_status_unhealthy(self):
        """Monitor reports unhealthy when any unhealthy."""
        monitor = HealthMonitor()
        monitor.register(MockHealthComponent("comp1", HealthStatus.HEALTHY))
        monitor.register(MockHealthComponent("comp2", HealthStatus.UNHEALTHY))

        result = await monitor.check_all()

        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_monitor_last_results(self):
        """Monitor stores last results."""
        monitor = HealthMonitor()
        monitor.register(MockHealthComponent("test", HealthStatus.HEALTHY))

        await monitor.check_all()

        results = monitor.get_last_results()
        assert "test" in results
        assert results["test"].status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_monitor_start_stop_monitoring(self):
        """Monitor starts and stops periodic monitoring."""
        monitor = HealthMonitor()
        monitor.register(MockHealthComponent("test"))

        # Start monitoring
        await monitor.start_monitoring(interval=0.1)
        assert monitor.is_monitoring

        # Wait for at least one check
        await asyncio.sleep(0.15)

        # Stop monitoring
        await monitor.stop_monitoring()
        assert not monitor.is_monitoring

        # Results should be populated
        results = monitor.get_last_results()
        assert "test" in results


class TestSystemHealthChecks:
    """Tests for system health check functions."""

    def test_check_memory(self):
        """Memory check returns valid result."""
        health = check_memory()

        assert health.name == "system.memory"
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
        if health.status != HealthStatus.UNKNOWN:
            assert "percent" in health.details

    def test_check_cpu(self):
        """CPU check returns valid result."""
        health = check_cpu()

        assert health.name == "system.cpu"
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
        if health.status != HealthStatus.UNKNOWN:
            assert "percent" in health.details

    def test_check_disk(self):
        """Disk check returns valid result."""
        health = check_disk()

        assert health.name == "system.disk"
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
        if health.status != HealthStatus.UNKNOWN:
            assert "percent" in health.details


class TestGlobalMonitor:
    """Tests for global monitor functions."""

    def test_get_monitor(self):
        """get_monitor returns default instance."""
        monitor = get_monitor()
        assert monitor is not None
        assert isinstance(monitor, HealthMonitor)

        # Should have system checks registered
        assert monitor.component_count >= 3  # memory, cpu, disk

    def test_set_monitor(self):
        """set_monitor changes default instance."""
        original = get_monitor()
        new_monitor = HealthMonitor()

        set_monitor(new_monitor)
        assert get_monitor() is new_monitor

        # Restore original
        set_monitor(original)

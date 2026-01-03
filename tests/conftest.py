"""
Pytest configuration and shared fixtures for LIBRA tests.
"""

import pytest


# =============================================================================
# Async fixtures
# =============================================================================


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio as the async backend."""
    return "asyncio"


# =============================================================================
# Event fixtures (will be populated after events.py is created)
# =============================================================================


@pytest.fixture
def sample_payload() -> dict[str, object]:
    """Sample event payload for testing."""
    return {
        "symbol": "BTC/USDT",
        "price": 50000.0,
        "quantity": 0.1,
    }


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")

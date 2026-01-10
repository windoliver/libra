"""Tests for OpenBB Gateway."""

from __future__ import annotations

from datetime import date

import pytest

from libra.gateways.openbb.gateway import (
    OpenBBCredentials,
    OpenBBGateway,
    OpenBBNotInstalledError,
    ProviderStatus,
)


class TestOpenBBCredentials:
    """Tests for OpenBBCredentials."""

    def test_default_credentials(self) -> None:
        """Test default credentials (all None)."""
        creds = OpenBBCredentials()

        assert creds.fmp_api_key is None
        assert creds.polygon_api_key is None
        assert creds.fred_api_key is None

    def test_with_api_keys(self) -> None:
        """Test credentials with API keys."""
        creds = OpenBBCredentials(
            fmp_api_key="test_fmp_key",
            polygon_api_key="test_polygon_key",
            fred_api_key="test_fred_key",
        )

        assert creds.fmp_api_key == "test_fmp_key"
        assert creds.polygon_api_key == "test_polygon_key"
        assert creds.fred_api_key == "test_fred_key"


class TestProviderStatus:
    """Tests for ProviderStatus."""

    def test_create_status(self) -> None:
        """Test creating provider status."""
        status = ProviderStatus(
            name="yfinance",
            available=True,
            has_credentials=False,
            rate_limit="2000/hour",
        )

        assert status.name == "yfinance"
        assert status.available is True
        assert status.has_credentials is False
        assert status.rate_limit == "2000/hour"


class TestOpenBBNotInstalledError:
    """Tests for OpenBBNotInstalledError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = OpenBBNotInstalledError()

        assert "pip install openbb" in str(error)
        assert "openbb-yfinance" in str(error)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = OpenBBNotInstalledError("Custom message")

        assert str(error) == "Custom message"


class TestOpenBBGateway:
    """Tests for OpenBBGateway."""

    def test_init_defaults(self) -> None:
        """Test gateway initialization with defaults."""
        gateway = OpenBBGateway()

        assert gateway.default_equity_provider == "yfinance"
        assert gateway.default_crypto_provider == "yfinance"
        assert gateway.default_fundamentals_provider == "fmp"
        assert gateway.default_options_provider == "cboe"
        assert gateway.default_economic_provider == "fred"
        assert gateway.is_connected is False

    def test_init_with_credentials(self) -> None:
        """Test gateway with custom credentials."""
        creds = OpenBBCredentials(fmp_api_key="test_key")
        gateway = OpenBBGateway(credentials=creds)

        assert gateway.credentials.fmp_api_key == "test_key"

    def test_init_with_custom_providers(self) -> None:
        """Test gateway with custom default providers."""
        gateway = OpenBBGateway(
            default_equity_provider="polygon",
            default_fundamentals_provider="intrinio",
        )

        assert gateway.default_equity_provider == "polygon"
        assert gateway.default_fundamentals_provider == "intrinio"

    def test_not_connected_raises(self) -> None:
        """Test methods raise when not connected."""
        gateway = OpenBBGateway()

        assert gateway.is_connected is False

        with pytest.raises(RuntimeError, match="not connected"):
            gateway._ensure_connected()

    def test_get_provider_status(self) -> None:
        """Test getting provider status."""
        creds = OpenBBCredentials(fmp_api_key="test_key")
        gateway = OpenBBGateway(credentials=creds)

        statuses = gateway.get_provider_status()

        assert len(statuses) > 0
        assert all(isinstance(s, ProviderStatus) for s in statuses)

        # Check FMP has credentials
        fmp_status = next(s for s in statuses if s.name == "fmp")
        assert fmp_status.has_credentials is True

        # Check yfinance (no API key required)
        yf_status = next(s for s in statuses if s.name == "yfinance")
        assert yf_status.available is True

    def test_get_available_providers(self) -> None:
        """Test getting available providers by data type."""
        gateway = OpenBBGateway()

        equity_providers = gateway.get_available_providers("equity")
        assert "yfinance" in equity_providers
        assert "polygon" in equity_providers

        crypto_providers = gateway.get_available_providers("crypto")
        assert "yfinance" in crypto_providers
        assert "fmp" in crypto_providers

        options_providers = gateway.get_available_providers("options")
        assert "cboe" in options_providers

        economic_providers = gateway.get_available_providers("economic")
        assert "fred" in economic_providers

    def test_unknown_data_type_providers(self) -> None:
        """Test getting providers for unknown data type."""
        gateway = OpenBBGateway()

        providers = gateway.get_available_providers("unknown")
        assert providers == []


class TestOpenBBGatewayIntegration:
    """Integration tests requiring OpenBB (skipped if not installed)."""

    @pytest.fixture
    def gateway(self) -> OpenBBGateway:
        """Create a gateway instance."""
        return OpenBBGateway()

    @pytest.mark.asyncio
    async def test_connect_without_openbb(self, gateway: OpenBBGateway) -> None:
        """Test connect raises when OpenBB not installed."""
        # This test will pass if OpenBB is not installed
        # and skip if it is installed
        try:
            import openbb
            pytest.skip("OpenBB is installed, skipping not-installed test")
        except ImportError:
            with pytest.raises(OpenBBNotInstalledError):
                await gateway.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, gateway: OpenBBGateway) -> None:
        """Test disconnect clears state."""
        # Manually set connected state
        gateway._connected = True
        gateway._equity_fetcher = "mock"  # type: ignore

        await gateway.disconnect()

        assert gateway.is_connected is False
        assert gateway._equity_fetcher is None

"""Tests for Prediction Market Gateway."""

from __future__ import annotations

import pytest

from libra.gateways.prediction_market.gateway import (
    PredictionMarketCredentials,
    PredictionMarketGateway,
    PredictionMarketGatewayError,
    PredictionMarketNotInstalledError,
    ProviderStatus,
)
from libra.gateways.prediction_market.protocol import (
    PredictionMarketCapabilities,
)


class TestPredictionMarketCredentials:
    """Tests for PredictionMarketCredentials."""

    def test_default_credentials(self) -> None:
        """Test default credentials (all None)."""
        creds = PredictionMarketCredentials()

        assert creds.polymarket_api_key is None
        assert creds.polymarket_api_secret is None
        assert creds.polymarket_private_key is None
        assert creds.kalshi_api_key is None
        assert creds.kalshi_private_key is None
        assert creds.manifold_api_key is None
        assert creds.metaculus_api_token is None

    def test_with_polymarket_credentials(self) -> None:
        """Test credentials with Polymarket keys."""
        creds = PredictionMarketCredentials(
            polymarket_api_key="test_key",
            polymarket_api_secret="test_secret",
            polymarket_private_key="0xprivatekey",
        )

        assert creds.polymarket_api_key == "test_key"
        assert creds.polymarket_api_secret == "test_secret"
        assert creds.polymarket_private_key == "0xprivatekey"

    def test_with_kalshi_credentials(self) -> None:
        """Test credentials with Kalshi keys."""
        creds = PredictionMarketCredentials(
            kalshi_api_key="kalshi_key",
            kalshi_private_key="-----BEGIN RSA PRIVATE KEY-----...",
        )

        assert creds.kalshi_api_key == "kalshi_key"
        assert creds.kalshi_private_key is not None

    def test_with_all_credentials(self) -> None:
        """Test credentials with all platforms."""
        creds = PredictionMarketCredentials(
            polymarket_api_key="poly_key",
            kalshi_api_key="kalshi_key",
            manifold_api_key="manifold_key",
            metaculus_api_token="metaculus_token",
        )

        assert creds.polymarket_api_key == "poly_key"
        assert creds.kalshi_api_key == "kalshi_key"
        assert creds.manifold_api_key == "manifold_key"
        assert creds.metaculus_api_token == "metaculus_token"


class TestProviderStatus:
    """Tests for ProviderStatus."""

    def test_create_status(self) -> None:
        """Test creating provider status."""
        status = ProviderStatus(
            name="polymarket",
            available=True,
            connected=True,
            has_credentials=True,
        )

        assert status.name == "polymarket"
        assert status.available is True
        assert status.connected is True
        assert status.has_credentials is True

    def test_status_with_capabilities(self) -> None:
        """Test status with capabilities."""
        caps = PredictionMarketCapabilities(
            supports_trading=True,
            supports_limit_orders=True,
        )
        status = ProviderStatus(
            name="kalshi",
            available=True,
            connected=False,
            has_credentials=False,
            capabilities=caps,
        )

        assert status.capabilities is not None
        assert status.capabilities.supports_trading is True

    def test_status_with_error(self) -> None:
        """Test status with error message."""
        status = ProviderStatus(
            name="polymarket",
            available=True,
            connected=False,
            has_credentials=True,
            last_error="Connection timeout",
        )

        assert status.last_error == "Connection timeout"


class TestPredictionMarketNotInstalledError:
    """Tests for PredictionMarketNotInstalledError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = PredictionMarketNotInstalledError()

        assert "pip install httpx" in str(error)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = PredictionMarketNotInstalledError("Custom message")

        assert str(error) == "Custom message"


class TestPredictionMarketGatewayError:
    """Tests for PredictionMarketGatewayError."""

    def test_error_message(self) -> None:
        """Test error with message."""
        error = PredictionMarketGatewayError("Something went wrong")

        assert str(error) == "Something went wrong"


class TestPredictionMarketGateway:
    """Tests for PredictionMarketGateway."""

    def test_init_defaults(self) -> None:
        """Test gateway initialization with defaults."""
        gateway = PredictionMarketGateway()

        assert gateway.default_provider == "polymarket"
        assert gateway.enable_polymarket is True
        assert gateway.enable_kalshi is True
        assert gateway.enable_metaculus is True
        assert gateway.enable_manifold is True
        assert gateway.is_connected is False

    def test_init_with_credentials(self) -> None:
        """Test gateway with custom credentials."""
        creds = PredictionMarketCredentials(
            polymarket_api_key="test_key"
        )
        gateway = PredictionMarketGateway(credentials=creds)

        assert gateway.credentials.polymarket_api_key == "test_key"

    def test_init_disable_providers(self) -> None:
        """Test gateway with disabled providers."""
        gateway = PredictionMarketGateway(
            enable_polymarket=False,
            enable_kalshi=True,
            enable_metaculus=False,
            enable_manifold=True,
        )

        assert gateway.enable_polymarket is False
        assert gateway.enable_kalshi is True
        assert gateway.enable_metaculus is False
        assert gateway.enable_manifold is True

    def test_available_providers_before_connect(self) -> None:
        """Test available_providers before connect."""
        gateway = PredictionMarketGateway()

        # Should be empty before connect
        assert gateway.available_providers == []

    def test_not_connected_error(self) -> None:
        """Test error when calling methods without connecting."""
        gateway = PredictionMarketGateway()

        with pytest.raises(RuntimeError, match="not connected"):
            gateway._ensure_connected()

    def test_get_provider_status(self) -> None:
        """Test get_provider_status method."""
        creds = PredictionMarketCredentials(
            polymarket_api_key="poly_key",
            kalshi_api_key="kalshi_key",
        )
        gateway = PredictionMarketGateway(credentials=creds)

        statuses = gateway.get_provider_status()

        assert len(statuses) == 4

        # Check polymarket status
        poly_status = next(s for s in statuses if s.name == "polymarket")
        assert poly_status.available is True
        assert poly_status.has_credentials is True

        # Check kalshi status
        kalshi_status = next(s for s in statuses if s.name == "kalshi")
        assert kalshi_status.available is True
        assert kalshi_status.has_credentials is True

        # Check metaculus (no credentials)
        meta_status = next(s for s in statuses if s.name == "metaculus")
        assert meta_status.has_credentials is False


class TestPredictionMarketGatewayAsync:
    """Async tests for PredictionMarketGateway."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        """Test connect and disconnect lifecycle."""
        pytest.importorskip("httpx")

        gateway = PredictionMarketGateway()

        # Connect
        await gateway.connect()
        assert gateway.is_connected is True
        assert len(gateway.available_providers) > 0

        # Disconnect
        await gateway.disconnect()
        assert gateway.is_connected is False
        assert len(gateway.available_providers) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        pytest.importorskip("httpx")

        async with PredictionMarketGateway() as gateway:
            assert gateway.is_connected is True
            assert len(gateway.available_providers) > 0

        # After context exit
        assert gateway.is_connected is False

    @pytest.mark.asyncio
    async def test_get_markets_polymarket(self) -> None:
        """Test getting markets from Polymarket."""
        pytest.importorskip("httpx")

        gateway = PredictionMarketGateway(
            enable_kalshi=False,
            enable_metaculus=False,
            enable_manifold=False,
        )

        async with gateway:
            markets = await gateway.get_markets(
                provider="polymarket",
                limit=5,
            )

            # Should return some markets (may be empty if API issues)
            assert isinstance(markets, list)
            if markets:
                market = markets[0]
                assert market.platform == "polymarket"
                assert market.market_id is not None
                assert market.title is not None

    @pytest.mark.asyncio
    async def test_get_markets_all_providers(self) -> None:
        """Test getting markets from all providers."""
        pytest.importorskip("httpx")

        gateway = PredictionMarketGateway()

        async with gateway:
            markets = await gateway.get_markets(limit=10)

            # Should aggregate from multiple providers
            assert isinstance(markets, list)

    @pytest.mark.asyncio
    async def test_search_markets(self) -> None:
        """Test searching markets."""
        pytest.importorskip("httpx")

        gateway = PredictionMarketGateway(
            enable_kalshi=False,
            enable_metaculus=False,
            enable_manifold=False,
        )

        async with gateway:
            markets = await gateway.search_markets(
                query="bitcoin",
                limit=5,
            )

            assert isinstance(markets, list)

    @pytest.mark.asyncio
    async def test_get_market_detail(self) -> None:
        """Test getting a specific market."""
        pytest.importorskip("httpx")

        gateway = PredictionMarketGateway(
            enable_kalshi=False,
            enable_metaculus=False,
            enable_manifold=False,
        )

        async with gateway:
            # First get a market to get its ID
            markets = await gateway.get_markets(
                provider="polymarket",
                limit=1,
            )

            if markets:
                market_id = markets[0].market_id
                market = await gateway.get_market("polymarket", market_id)

                assert market is not None
                assert market.market_id == market_id


class TestPredictionMarketGatewayIntegration:
    """Integration tests requiring network access."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_quote(self) -> None:
        """Test getting a quote (requires network)."""
        pytest.importorskip("httpx")

        gateway = PredictionMarketGateway(
            enable_kalshi=False,
            enable_metaculus=False,
            enable_manifold=False,
        )

        async with gateway:
            # Get a market first
            markets = await gateway.get_markets(
                provider="polymarket",
                limit=1,
            )

            if markets and markets[0].outcomes:
                market = markets[0]
                outcome = market.outcomes[0]

                quote = await gateway.get_quote(
                    "polymarket",
                    market.market_id,
                    outcome.outcome_id,
                )

                # Quote may or may not be available
                if quote:
                    assert quote.market_id == market.market_id
                    assert quote.bid >= 0
                    assert quote.ask >= 0

"""Tests for Polymarket Provider."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.gateways.prediction_market.protocol import (
    PredictionOrder,
    PredictionOrderSide,
    PredictionOrderType,
)
from libra.gateways.prediction_market.providers.base import ProviderConfig
from libra.gateways.prediction_market.providers.polymarket import (
    POLYMARKET_CONTRACTS,
    POLYGON_CHAIN_ID,
    PolymarketAuthError,
    PolymarketProvider,
)


class TestPolymarketProviderBasic:
    """Basic tests for PolymarketProvider."""

    def test_provider_name(self) -> None:
        """Test provider name."""
        provider = PolymarketProvider()
        assert provider.name == "polymarket"

    def test_provider_base_url(self) -> None:
        """Test provider base URL."""
        provider = PolymarketProvider()
        assert "gamma-api.polymarket.com" in provider.base_url

    def test_capabilities(self) -> None:
        """Test provider capabilities."""
        provider = PolymarketProvider()
        caps = provider.capabilities

        assert caps.supports_trading is True
        assert caps.supports_limit_orders is True
        assert caps.supports_orderbook is True

    def test_not_connected_initially(self) -> None:
        """Test provider is not connected initially."""
        provider = PolymarketProvider()

        assert provider.is_connected is False
        assert provider.is_trading_ready is False
        assert provider.wallet_address is None

    def test_config_without_private_key(self) -> None:
        """Test provider config without private key."""
        config = ProviderConfig()
        provider = PolymarketProvider(config)

        assert provider._config.private_key is None

    def test_config_with_private_key(self) -> None:
        """Test provider config with private key."""
        config = ProviderConfig(
            private_key="0x" + "a" * 64,  # Mock private key
        )
        provider = PolymarketProvider(config)

        assert provider._config.private_key is not None


class TestPolymarketContractAddresses:
    """Test contract addresses are correctly defined."""

    def test_usdc_address(self) -> None:
        """Test USDC.e address on Polygon."""
        assert POLYMARKET_CONTRACTS["usdc"] == "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

    def test_ctf_address(self) -> None:
        """Test Conditional Token Framework address."""
        assert POLYMARKET_CONTRACTS["ctf"] == "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

    def test_exchange_address(self) -> None:
        """Test Exchange contract address."""
        assert POLYMARKET_CONTRACTS["exchange"] == "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

    def test_chain_id(self) -> None:
        """Test Polygon chain ID."""
        assert POLYGON_CHAIN_ID == 137


class TestPolymarketOrderStatusParsing:
    """Tests for order status parsing."""

    def test_parse_pending_status(self) -> None:
        """Test parsing pending status."""
        provider = PolymarketProvider()
        from libra.gateways.prediction_market.protocol import PredictionOrderStatus

        assert provider._parse_order_status("pending") == PredictionOrderStatus.PENDING

    def test_parse_open_status(self) -> None:
        """Test parsing open status."""
        provider = PolymarketProvider()
        from libra.gateways.prediction_market.protocol import PredictionOrderStatus

        assert provider._parse_order_status("open") == PredictionOrderStatus.OPEN
        assert provider._parse_order_status("live") == PredictionOrderStatus.OPEN

    def test_parse_filled_status(self) -> None:
        """Test parsing filled status."""
        provider = PolymarketProvider()
        from libra.gateways.prediction_market.protocol import PredictionOrderStatus

        assert provider._parse_order_status("filled") == PredictionOrderStatus.FILLED
        assert provider._parse_order_status("matched") == PredictionOrderStatus.FILLED

    def test_parse_cancelled_status(self) -> None:
        """Test parsing cancelled status."""
        provider = PolymarketProvider()
        from libra.gateways.prediction_market.protocol import PredictionOrderStatus

        assert provider._parse_order_status("cancelled") == PredictionOrderStatus.CANCELLED
        assert provider._parse_order_status("canceled") == PredictionOrderStatus.CANCELLED

    def test_parse_unknown_status(self) -> None:
        """Test parsing unknown status defaults to pending."""
        provider = PolymarketProvider()
        from libra.gateways.prediction_market.protocol import PredictionOrderStatus

        assert provider._parse_order_status("unknown") == PredictionOrderStatus.PENDING


class TestPolymarketAllowances:
    """Tests for allowance checking."""

    def test_check_allowances_without_wallet_sync(self) -> None:
        """Test checking allowances without wallet returns False."""
        import asyncio
        provider = PolymarketProvider()

        result = asyncio.get_event_loop().run_until_complete(provider.check_allowances())
        assert result == {"usdc": False, "ctf": False}

    def test_set_allowances_without_private_key_sync(self) -> None:
        """Test setting allowances without private key raises error."""
        import asyncio
        provider = PolymarketProvider()

        with pytest.raises(PolymarketAuthError):
            asyncio.get_event_loop().run_until_complete(provider.set_allowances())


class TestPolymarketOrderValidation:
    """Tests for order validation."""

    def test_create_order_object(self) -> None:
        """Test creating a prediction order."""
        order = PredictionOrder(
            market_id="0x123",
            outcome_id="yes_token",
            platform="polymarket",
            side=PredictionOrderSide.BUY,
            order_type=PredictionOrderType.LIMIT,
            size=Decimal("10"),
            price=Decimal("0.65"),
        )

        assert order.market_id == "0x123"
        assert order.outcome_id == "yes_token"
        assert order.side == PredictionOrderSide.BUY
        assert order.order_type == PredictionOrderType.LIMIT
        assert order.size == Decimal("10")
        assert order.price == Decimal("0.65")

    def test_order_types_available(self) -> None:
        """Test all order types are available."""
        assert hasattr(PredictionOrderType, "MARKET")
        assert hasattr(PredictionOrderType, "LIMIT")
        assert hasattr(PredictionOrderType, "FOK")
        assert hasattr(PredictionOrderType, "IOC")
        assert hasattr(PredictionOrderType, "GTC")


class TestPolymarketProviderProperties:
    """Tests for provider properties."""

    def test_clob_client_none_initially(self) -> None:
        """Test CLOB client is None initially."""
        provider = PolymarketProvider()
        assert provider.clob_client is None

    def test_is_trading_ready_false_initially(self) -> None:
        """Test trading is not ready initially."""
        provider = PolymarketProvider()
        assert provider.is_trading_ready is False

    def test_wallet_address_none_initially(self) -> None:
        """Test wallet address is None initially."""
        provider = PolymarketProvider()
        assert provider.wallet_address is None

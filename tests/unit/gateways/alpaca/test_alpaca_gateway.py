"""
Unit tests for Alpaca Gateway.

Tests use mocked Alpaca SDK responses to avoid API calls.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from libra.gateways.alpaca.config import AlpacaConfig, AlpacaCredentials
from libra.gateways.alpaca.symbols import (
    from_occ_symbol,
    to_occ_symbol,
    is_option_symbol,
    normalize_symbol,
    format_option_display,
)
from libra.gateways.protocol import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)


# =============================================================================
# Symbol Tests
# =============================================================================


class TestOCCSymbol:
    """Tests for OCC symbol conversion."""

    def test_to_occ_symbol_call(self) -> None:
        """Test converting call option to OCC format."""
        result = to_occ_symbol(
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            option_type="call",
            strike=150.00,
        )
        assert result == "AAPL250117C00150000"

    def test_to_occ_symbol_put(self) -> None:
        """Test converting put option to OCC format."""
        result = to_occ_symbol(
            underlying="TSLA",
            expiration=date(2025, 3, 21),
            option_type="put",
            strike=200.50,
        )
        assert result == "TSLA250321P00200500"

    def test_to_occ_symbol_short_type(self) -> None:
        """Test using C/P shorthand for option type."""
        result = to_occ_symbol(
            underlying="SPY",
            expiration=date(2025, 6, 20),
            option_type="C",
            strike=450.00,
        )
        assert result == "SPY250620C00450000"

    def test_to_occ_symbol_decimal_strike(self) -> None:
        """Test strike with decimal places."""
        result = to_occ_symbol(
            underlying="QQQ",
            expiration=date(2025, 2, 14),
            option_type="call",
            strike=375.50,
        )
        assert result == "QQQ250214C00375500"

    def test_to_occ_symbol_invalid_underlying(self) -> None:
        """Test invalid underlying raises error."""
        with pytest.raises(ValueError, match="Invalid underlying"):
            to_occ_symbol(
                underlying="",
                expiration=date(2025, 1, 17),
                option_type="call",
                strike=150.00,
            )

    def test_to_occ_symbol_long_underlying(self) -> None:
        """Test underlying > 6 chars raises error."""
        with pytest.raises(ValueError, match="Invalid underlying"):
            to_occ_symbol(
                underlying="TOOLONG",
                expiration=date(2025, 1, 17),
                option_type="call",
                strike=150.00,
            )

    def test_to_occ_symbol_invalid_type(self) -> None:
        """Test invalid option type raises error."""
        with pytest.raises(ValueError, match="Invalid option type"):
            to_occ_symbol(
                underlying="AAPL",
                expiration=date(2025, 1, 17),
                option_type="invalid",
                strike=150.00,
            )

    def test_to_occ_symbol_negative_strike(self) -> None:
        """Test negative strike raises error."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            to_occ_symbol(
                underlying="AAPL",
                expiration=date(2025, 1, 17),
                option_type="call",
                strike=-150.00,
            )

    def test_from_occ_symbol_call(self) -> None:
        """Test parsing call option OCC symbol."""
        result = from_occ_symbol("AAPL250117C00150000")
        assert result.underlying == "AAPL"
        assert result.expiration == date(2025, 1, 17)
        assert result.option_type == "call"
        assert result.strike == 150.0

    def test_from_occ_symbol_put(self) -> None:
        """Test parsing put option OCC symbol."""
        result = from_occ_symbol("TSLA250321P00200500")
        assert result.underlying == "TSLA"
        assert result.expiration == date(2025, 3, 21)
        assert result.option_type == "put"
        assert result.strike == 200.5

    def test_from_occ_symbol_roundtrip(self) -> None:
        """Test roundtrip conversion."""
        original = "SPY250620C00450000"
        components = from_occ_symbol(original)
        result = components.to_occ()
        assert result == original

    def test_from_occ_symbol_invalid(self) -> None:
        """Test invalid OCC symbol raises error."""
        with pytest.raises(ValueError, match="Invalid OCC symbol"):
            from_occ_symbol("INVALID")

    def test_is_option_symbol_true(self) -> None:
        """Test is_option_symbol returns True for OCC symbols."""
        assert is_option_symbol("AAPL250117C00150000") is True
        assert is_option_symbol("TSLA250321P00200500") is True

    def test_is_option_symbol_false(self) -> None:
        """Test is_option_symbol returns False for stock symbols."""
        assert is_option_symbol("AAPL") is False
        assert is_option_symbol("BTC/USD") is False
        assert is_option_symbol("") is False

    def test_normalize_symbol(self) -> None:
        """Test symbol normalization."""
        assert normalize_symbol("aapl") == "AAPL"
        assert normalize_symbol("  MSFT  ") == "MSFT"
        assert normalize_symbol("GOOG.US") == "GOOG"
        assert normalize_symbol("BRK.B") == "BRK.B"

    def test_format_option_display(self) -> None:
        """Test human-readable option display."""
        result = format_option_display("AAPL250117C00150000")
        assert "AAPL" in result
        assert "150.00" in result
        assert "Call" in result


# =============================================================================
# Config Tests
# =============================================================================


class TestAlpacaConfig:
    """Tests for Alpaca configuration."""

    def test_credentials_creation(self) -> None:
        """Test creating credentials directly."""
        creds = AlpacaCredentials(
            api_key="PKTEST123",
            secret_key="secretkey",
        )
        assert creds.api_key == "PKTEST123"
        assert creds.is_paper_key() is True

    def test_credentials_live_key(self) -> None:
        """Test detecting live API key."""
        creds = AlpacaCredentials(
            api_key="AKTEST123",  # Live keys start with AK
            secret_key="secretkey",
        )
        assert creds.is_paper_key() is False

    def test_config_defaults(self) -> None:
        """Test config default values."""
        creds = AlpacaCredentials(api_key="PKTEST", secret_key="secret")
        config = AlpacaConfig(credentials=creds)

        assert config.paper is True
        assert config.data_feed == "iex"
        assert config.max_retries == 5
        assert config.rate_limit_per_minute == 200
        assert config.options_level == 2

    def test_config_paper_url(self) -> None:
        """Test paper trading URL."""
        creds = AlpacaCredentials(api_key="PKTEST", secret_key="secret")
        config = AlpacaConfig(credentials=creds, paper=True)
        assert config.base_url == "https://paper-api.alpaca.markets"

    def test_config_live_url(self) -> None:
        """Test live trading URL."""
        creds = AlpacaCredentials(api_key="AKTEST", secret_key="secret")
        config = AlpacaConfig(credentials=creds, paper=False)
        assert config.base_url == "https://api.alpaca.markets"

    def test_config_invalid_data_feed(self) -> None:
        """Test invalid data feed raises error."""
        creds = AlpacaCredentials(api_key="PKTEST", secret_key="secret")
        with pytest.raises(ValueError, match="Invalid data_feed"):
            AlpacaConfig(credentials=creds, data_feed="invalid")

    def test_config_invalid_options_level(self) -> None:
        """Test invalid options level raises error."""
        creds = AlpacaCredentials(api_key="PKTEST", secret_key="secret")
        with pytest.raises(ValueError, match="Invalid options_level"):
            AlpacaConfig(credentials=creds, options_level=5)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "PKENV123", "ALPACA_SECRET_KEY": "envsecret"})
    def test_credentials_from_env(self) -> None:
        """Test loading credentials from environment."""
        creds = AlpacaCredentials.from_env()
        assert creds.api_key == "PKENV123"
        assert creds.secret_key == "envsecret"

    @patch.dict("os.environ", {}, clear=True)
    def test_credentials_from_env_missing(self) -> None:
        """Test missing env vars raises error."""
        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            AlpacaCredentials.from_env()


# =============================================================================
# Gateway Tests (Mocked)
# =============================================================================


class TestAlpacaGateway:
    """Tests for AlpacaGateway with mocked SDK."""

    @pytest.fixture
    def mock_config(self) -> AlpacaConfig:
        """Create test config."""
        return AlpacaConfig(
            credentials=AlpacaCredentials(
                api_key="PKTEST123",
                secret_key="testsecret",
            ),
            paper=True,
        )

    @pytest.fixture
    def mock_account(self) -> MagicMock:
        """Create mock Alpaca account."""
        account = MagicMock()
        account.account_number = "TEST123"
        account.equity = "100000.00"
        account.buying_power = "50000.00"
        account.cash = "50000.00"
        account.trading_blocked = False
        account.account_blocked = False
        account.status = "ACTIVE"
        account.pattern_day_trader = False
        account.portfolio_value = "100000.00"
        account.daytrading_buying_power = "200000.00"
        return account

    @pytest.fixture
    def mock_order(self) -> MagicMock:
        """Create mock Alpaca order."""
        order = MagicMock()
        order.id = "order-123"
        order.symbol = "AAPL"
        order.status = "filled"
        order.side = "buy"
        order.type = "market"
        order.qty = "10"
        order.filled_qty = "10"
        order.filled_avg_price = "150.00"
        order.limit_price = None
        order.stop_price = None
        order.client_order_id = "client-123"
        order.created_at = datetime(2025, 1, 15, 10, 0, 0)
        order.updated_at = datetime(2025, 1, 15, 10, 0, 1)
        return order

    @pytest.fixture
    def mock_position(self) -> MagicMock:
        """Create mock Alpaca position."""
        position = MagicMock()
        position.symbol = "AAPL"
        position.qty = "100"
        position.avg_entry_price = "145.00"
        position.current_price = "150.00"
        position.unrealized_pl = "500.00"
        return position

    @pytest.mark.asyncio
    async def test_connect(self, mock_config: AlpacaConfig, mock_account: MagicMock) -> None:
        """Test gateway connection."""
        with patch("libra.gateways.alpaca.gateway.TradingClient") as MockTradingClient, \
             patch("libra.gateways.alpaca.gateway.StockHistoricalDataClient"), \
             patch("libra.gateways.alpaca.gateway.StockDataStream"), \
             patch("libra.gateways.alpaca.gateway.TradingStream"):

            # Mock the imports
            import sys
            mock_alpaca = MagicMock()
            sys.modules["alpaca"] = mock_alpaca
            sys.modules["alpaca.trading"] = MagicMock()
            sys.modules["alpaca.trading.client"] = MagicMock()
            sys.modules["alpaca.data"] = MagicMock()
            sys.modules["alpaca.data.historical"] = MagicMock()
            sys.modules["alpaca.data.live"] = MagicMock()
            sys.modules["alpaca.trading.stream"] = MagicMock()

            MockTradingClient.return_value.get_account.return_value = mock_account

            from libra.gateways.alpaca.gateway import AlpacaGateway

            gateway = AlpacaGateway(mock_config)
            await gateway.connect()

            assert gateway.is_connected is True
            MockTradingClient.return_value.get_account.assert_called_once()

            # Cleanup
            await gateway.disconnect()

    @pytest.mark.asyncio
    async def test_submit_market_order(
        self, mock_config: AlpacaConfig, mock_account: MagicMock, mock_order: MagicMock
    ) -> None:
        """Test submitting a market order."""
        with patch("libra.gateways.alpaca.gateway.TradingClient") as MockTradingClient, \
             patch("libra.gateways.alpaca.gateway.StockHistoricalDataClient"), \
             patch("libra.gateways.alpaca.gateway.StockDataStream"), \
             patch("libra.gateways.alpaca.gateway.TradingStream"):

            MockTradingClient.return_value.get_account.return_value = mock_account
            MockTradingClient.return_value.submit_order.return_value = mock_order

            from libra.gateways.alpaca.gateway import AlpacaGateway

            gateway = AlpacaGateway(mock_config)
            await gateway.connect()

            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("10"),
            )

            result = await gateway.submit_order(order)

            assert result.order_id == "order-123"
            assert result.status == OrderStatus.FILLED
            assert result.filled_amount == Decimal("10")

            await gateway.disconnect()

    @pytest.mark.asyncio
    async def test_get_positions(
        self, mock_config: AlpacaConfig, mock_account: MagicMock, mock_position: MagicMock
    ) -> None:
        """Test getting positions."""
        with patch("libra.gateways.alpaca.gateway.TradingClient") as MockTradingClient, \
             patch("libra.gateways.alpaca.gateway.StockHistoricalDataClient"), \
             patch("libra.gateways.alpaca.gateway.StockDataStream"), \
             patch("libra.gateways.alpaca.gateway.TradingStream"):

            MockTradingClient.return_value.get_account.return_value = mock_account
            MockTradingClient.return_value.get_all_positions.return_value = [mock_position]

            from libra.gateways.alpaca.gateway import AlpacaGateway

            gateway = AlpacaGateway(mock_config)
            await gateway.connect()

            positions = await gateway.get_positions()

            assert len(positions) == 1
            assert positions[0].symbol == "AAPL"
            assert positions[0].amount == Decimal("100")
            assert positions[0].unrealized_pnl == Decimal("500.00")

            await gateway.disconnect()

    @pytest.mark.asyncio
    async def test_get_balances(
        self, mock_config: AlpacaConfig, mock_account: MagicMock
    ) -> None:
        """Test getting account balances."""
        with patch("libra.gateways.alpaca.gateway.TradingClient") as MockTradingClient, \
             patch("libra.gateways.alpaca.gateway.StockHistoricalDataClient"), \
             patch("libra.gateways.alpaca.gateway.StockDataStream"), \
             patch("libra.gateways.alpaca.gateway.TradingStream"):

            MockTradingClient.return_value.get_account.return_value = mock_account

            from libra.gateways.alpaca.gateway import AlpacaGateway

            gateway = AlpacaGateway(mock_config)
            await gateway.connect()

            balances = await gateway.get_balances()

            assert "USD" in balances
            assert balances["USD"].total == Decimal("100000.00")
            assert balances["USD"].available == Decimal("50000.00")

            await gateway.disconnect()

    @pytest.mark.asyncio
    async def test_cancel_order(
        self, mock_config: AlpacaConfig, mock_account: MagicMock
    ) -> None:
        """Test cancelling an order."""
        with patch("libra.gateways.alpaca.gateway.TradingClient") as MockTradingClient, \
             patch("libra.gateways.alpaca.gateway.StockHistoricalDataClient"), \
             patch("libra.gateways.alpaca.gateway.StockDataStream"), \
             patch("libra.gateways.alpaca.gateway.TradingStream"):

            MockTradingClient.return_value.get_account.return_value = mock_account
            MockTradingClient.return_value.cancel_order_by_id.return_value = None

            from libra.gateways.alpaca.gateway import AlpacaGateway

            gateway = AlpacaGateway(mock_config)
            await gateway.connect()

            result = await gateway.cancel_order("order-123", "AAPL")

            assert result is True
            MockTradingClient.return_value.cancel_order_by_id.assert_called_once_with("order-123")

            await gateway.disconnect()


# =============================================================================
# Integration Tests (require real API - skipped by default)
# =============================================================================


@pytest.mark.skip(reason="Requires real Alpaca API credentials")
class TestAlpacaGatewayIntegration:
    """Integration tests with real Alpaca paper trading API."""

    @pytest.fixture
    async def gateway(self):
        """Create connected gateway."""
        from libra.gateways.alpaca import AlpacaGateway, AlpacaConfig

        config = AlpacaConfig.from_env(paper=True)
        gateway = AlpacaGateway(config)
        await gateway.connect()
        yield gateway
        await gateway.disconnect()

    @pytest.mark.asyncio
    async def test_real_get_account(self, gateway) -> None:
        """Test getting real account info."""
        info = await gateway.get_account_info()
        assert "account_number" in info
        assert "equity" in info

    @pytest.mark.asyncio
    async def test_real_get_ticker(self, gateway) -> None:
        """Test getting real ticker data."""
        ticker = await gateway.get_ticker("AAPL")
        assert ticker.symbol == "AAPL"
        assert ticker.bid > 0
        assert ticker.ask > 0

"""
End-to-end tests for OpenBB Data Gateway with real data.

These tests use real API calls to verify the OpenBB integration.
They require network access and may be slow.

Skip if:
- OpenBB is not installed
- Network is unavailable
- Rate limits are exceeded
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest


# Check if OpenBB is available
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False

requires_openbb = pytest.mark.skipif(
    not OPENBB_AVAILABLE,
    reason="OpenBB not installed (pip install openbb openbb-yfinance)"
)


@requires_openbb
class TestOpenBBEquityData:
    """E2E tests for equity data fetching."""

    @pytest.fixture
    async def gateway(self):
        """Create and connect gateway."""
        from libra.gateways.openbb import OpenBBGateway

        gw = OpenBBGateway()
        await gw.connect()
        yield gw
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_fetch_aapl_historical(self, gateway) -> None:
        """Test fetching AAPL historical data."""
        bars = await gateway.get_equity_historical(
            symbol="AAPL",
            interval="1d",
            provider="yfinance",
        )

        assert len(bars) > 0
        assert all(bar.symbol == "AAPL" for bar in bars)
        assert all(bar.open > 0 for bar in bars)
        assert all(bar.high >= bar.low for bar in bars)
        assert all(bar.volume >= 0 for bar in bars)

        print(f"\n=== AAPL Historical Data ===")
        print(f"Fetched {len(bars)} bars")
        if bars:
            latest = bars[-1]
            print(f"Latest: {latest.datetime} - O:{latest.open} H:{latest.high} L:{latest.low} C:{latest.close}")

    @pytest.mark.asyncio
    async def test_fetch_with_date_range(self, gateway) -> None:
        """Test fetching with specific date range."""
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        bars = await gateway.get_equity_historical(
            symbol="MSFT",
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            provider="yfinance",
        )

        assert len(bars) > 0
        # Should have roughly 20-22 trading days
        assert len(bars) >= 15
        assert len(bars) <= 25

    @pytest.mark.asyncio
    async def test_fetch_quote(self, gateway) -> None:
        """Test fetching current quote."""
        quote = await gateway.get_quote(symbol="GOOGL", provider="yfinance")

        if quote:  # May be None outside market hours
            assert quote.symbol == "GOOGL"
            print(f"\n=== GOOGL Quote ===")
            print(f"Last: {quote.last}, Bid: {quote.bid}, Ask: {quote.ask}")


@requires_openbb
class TestOpenBBCryptoData:
    """E2E tests for crypto data fetching."""

    @pytest.fixture
    async def gateway(self):
        """Create and connect gateway."""
        from libra.gateways.openbb import OpenBBGateway

        gw = OpenBBGateway()
        await gw.connect()
        yield gw
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_fetch_btc_historical(self, gateway) -> None:
        """Test fetching BTC historical data."""
        bars = await gateway.get_crypto_historical(
            symbol="BTC-USD",
            interval="1d",
            provider="yfinance",
        )

        assert len(bars) > 0
        print(f"\n=== BTC-USD Historical Data ===")
        print(f"Fetched {len(bars)} bars")
        if bars:
            latest = bars[-1]
            print(f"Latest: {latest.datetime} - Close: ${float(latest.close):,.2f}")


@requires_openbb
class TestOpenBBFundamentals:
    """E2E tests for fundamental data fetching."""

    @pytest.fixture
    async def gateway(self):
        """Create and connect gateway."""
        from libra.gateways.openbb import OpenBBGateway

        gw = OpenBBGateway()
        await gw.connect()
        yield gw
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_fetch_company_profile(self, gateway) -> None:
        """Test fetching company profile."""
        profile = await gateway.get_company_profile(symbol="AAPL", provider="yfinance")

        if profile:
            assert profile.symbol == "AAPL"
            assert profile.name is not None
            print(f"\n=== AAPL Company Profile ===")
            print(f"Name: {profile.name}")
            print(f"Sector: {profile.sector}")
            print(f"Market Cap: ${float(profile.market_cap or 0):,.0f}")
            print(f"P/E: {profile.pe_ratio}")

    @pytest.mark.asyncio
    async def test_fetch_income_statement(self, gateway) -> None:
        """Test fetching income statement."""
        try:
            records = await gateway.get_income_statement(
                symbol="AAPL",
                period="annual",
                limit=4,
                provider="yfinance",
            )

            print(f"\n=== AAPL Income Statement ===")
            print(f"Fetched {len(records)} periods")
            for record in records[:2]:
                print(f"{record.period}: Revenue={record.revenue}, Net Income={record.net_income}")
        except Exception as e:
            pytest.skip(f"Income statement fetch failed: {e}")

    @pytest.mark.asyncio
    async def test_fetch_balance_sheet(self, gateway) -> None:
        """Test fetching balance sheet."""
        try:
            records = await gateway.get_balance_sheet(
                symbol="MSFT",
                period="annual",
                limit=2,
                provider="yfinance",
            )

            print(f"\n=== MSFT Balance Sheet ===")
            print(f"Fetched {len(records)} periods")
            for record in records[:1]:
                print(f"{record.period}: Assets={record.total_assets}, Equity={record.total_equity}")
        except Exception as e:
            pytest.skip(f"Balance sheet fetch failed: {e}")


@requires_openbb
class TestOpenBBOptions:
    """E2E tests for options data fetching."""

    @pytest.fixture
    async def gateway(self):
        """Create and connect gateway."""
        from libra.gateways.openbb import OpenBBGateway

        gw = OpenBBGateway()
        await gw.connect()
        yield gw
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_fetch_options_chain(self, gateway) -> None:
        """Test fetching options chain."""
        try:
            contracts = await gateway.get_options_chain(
                symbol="AAPL",
                provider="cboe",
            )

            if contracts:
                print(f"\n=== AAPL Options Chain ===")
                print(f"Fetched {len(contracts)} contracts")

                calls = [c for c in contracts if c.option_type == "call"]
                puts = [c for c in contracts if c.option_type == "put"]
                print(f"Calls: {len(calls)}, Puts: {len(puts)}")

                if calls:
                    sample = calls[0]
                    print(f"Sample Call: Strike={sample.strike}, Bid={sample.bid}, Ask={sample.ask}")
                    print(f"Greeks: Delta={sample.delta}, Gamma={sample.gamma}, Theta={sample.theta}")
        except Exception as e:
            pytest.skip(f"Options chain fetch failed: {e}")

    @pytest.mark.asyncio
    async def test_fetch_options_expirations(self, gateway) -> None:
        """Test fetching option expirations."""
        try:
            expirations = await gateway.get_options_expirations(symbol="SPY")

            if expirations:
                print(f"\n=== SPY Option Expirations ===")
                print(f"Found {len(expirations)} expirations")
                print(f"Next 5: {expirations[:5]}")
        except Exception as e:
            pytest.skip(f"Options expirations fetch failed: {e}")


@requires_openbb
class TestOpenBBEconomicData:
    """E2E tests for economic data fetching."""

    @pytest.fixture
    async def gateway(self):
        """Create and connect gateway."""
        from libra.gateways.openbb import OpenBBGateway

        gw = OpenBBGateway()
        await gw.connect()
        yield gw
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_fetch_gdp(self, gateway) -> None:
        """Test fetching GDP data."""
        try:
            points = await gateway.get_economic_series(
                series_id="GDP",
                provider="fred",
            )

            if points:
                print(f"\n=== GDP Data ===")
                print(f"Fetched {len(points)} data points")
                latest = points[-1]
                print(f"Latest: {latest.date} = {float(latest.value):,.1f}")
        except Exception as e:
            pytest.skip(f"GDP fetch failed (may need FRED API key): {e}")

    @pytest.mark.asyncio
    async def test_fetch_unemployment(self, gateway) -> None:
        """Test fetching unemployment rate."""
        try:
            points = await gateway.get_economic_series(
                series_id="UNRATE",
                provider="fred",
            )

            if points:
                print(f"\n=== Unemployment Rate ===")
                print(f"Fetched {len(points)} data points")
                latest = points[-1]
                print(f"Latest: {latest.date} = {float(latest.value):.1f}%")
        except Exception as e:
            pytest.skip(f"Unemployment fetch failed: {e}")

    @pytest.mark.asyncio
    async def test_fetch_cpi_with_transform(self, gateway) -> None:
        """Test fetching CPI with percent change transformation."""
        try:
            points = await gateway.get_economic_series(
                series_id="CPIAUCSL",
                transform="pc1",  # Year-over-year percent change
                provider="fred",
            )

            if points:
                print(f"\n=== CPI YoY Change ===")
                print(f"Fetched {len(points)} data points")
                latest = points[-1]
                print(f"Latest: {latest.date} = {float(latest.value):.2f}% YoY")
        except Exception as e:
            pytest.skip(f"CPI fetch failed: {e}")


@requires_openbb
class TestOpenBBGatewayLifecycle:
    """E2E tests for gateway lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        """Test gateway connect/disconnect cycle."""
        from libra.gateways.openbb import OpenBBGateway

        gateway = OpenBBGateway()

        assert gateway.is_connected is False

        await gateway.connect()
        assert gateway.is_connected is True
        assert gateway._equity_fetcher is not None

        await gateway.disconnect()
        assert gateway.is_connected is False
        assert gateway._equity_fetcher is None

    @pytest.mark.asyncio
    async def test_multiple_connect_disconnect(self) -> None:
        """Test multiple connect/disconnect cycles."""
        from libra.gateways.openbb import OpenBBGateway

        gateway = OpenBBGateway()

        for i in range(3):
            await gateway.connect()
            assert gateway.is_connected is True

            await gateway.disconnect()
            assert gateway.is_connected is False


@requires_openbb
class TestOpenBBDataIntegrity:
    """E2E tests for data integrity."""

    @pytest.fixture
    async def gateway(self):
        """Create and connect gateway."""
        from libra.gateways.openbb import OpenBBGateway

        gw = OpenBBGateway()
        await gw.connect()
        yield gw
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_bar_data_types(self, gateway) -> None:
        """Test that bar data has correct types."""
        bars = await gateway.get_equity_historical(
            symbol="AAPL",
            interval="1d",
            provider="yfinance",
        )

        assert len(bars) > 0
        bar = bars[0]

        # Check types
        assert isinstance(bar.symbol, str)
        assert isinstance(bar.timestamp_ns, int)
        assert isinstance(bar.open, Decimal)
        assert isinstance(bar.high, Decimal)
        assert isinstance(bar.low, Decimal)
        assert isinstance(bar.close, Decimal)
        assert isinstance(bar.volume, Decimal)
        assert isinstance(bar.interval, str)

        # Check values are sensible
        assert bar.timestamp_ns > 0
        assert bar.open > 0
        assert bar.high >= bar.open
        assert bar.high >= bar.close
        assert bar.low <= bar.open
        assert bar.low <= bar.close
        assert bar.volume >= 0

    @pytest.mark.asyncio
    async def test_bar_datetime_property(self, gateway) -> None:
        """Test bar datetime property."""
        from datetime import datetime

        bars = await gateway.get_equity_historical(
            symbol="MSFT",
            interval="1d",
            provider="yfinance",
        )

        assert len(bars) > 0
        bar = bars[0]

        dt = bar.datetime
        assert isinstance(dt, datetime)
        assert dt.year >= 2020  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

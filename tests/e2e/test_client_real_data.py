"""
End-to-End tests using REAL market data from Binance.

Tests the client architecture with actual historical price data
for multiple cryptocurrencies (BTC, ETH, SOL, etc.).

This provides more realistic testing than synthetic data.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from libra.clients import (
    BacktestDataClient,
    BacktestExecutionClient,
    InMemoryDataSource,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    Tick,
)
from libra.clients.backtest_execution_client import SlippageModel
from libra.core.clock import Clock, ClockType
from libra.strategies.protocol import Bar


# =============================================================================
# Real Data Fetching
# =============================================================================


async def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch real OHLCV data from Binance public API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Timeframe (e.g., "1m", "5m", "1h", "1d")
        limit: Number of candles to fetch (max 1000)

    Returns:
        List of OHLCV dicts
    """
    import urllib.request
    import json

    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        # Convert to dict format
        klines = []
        for k in data:
            klines.append({
                "timestamp_ms": k[0],
                "open": Decimal(k[1]),
                "high": Decimal(k[2]),
                "low": Decimal(k[3]),
                "close": Decimal(k[4]),
                "volume": Decimal(k[5]),
                "close_time_ms": k[6],
                "quote_volume": Decimal(k[7]),
                "trades": k[8],
            })
        return klines
    except Exception as e:
        pytest.skip(f"Could not fetch Binance data: {e}")
        return []


def klines_to_bars(klines: list[dict[str, Any]], symbol: str, timeframe: str) -> list[Bar]:
    """Convert Binance klines to Bar objects."""
    bars = []
    for k in klines:
        bar = Bar(
            symbol=symbol,
            timestamp_ns=k["timestamp_ms"] * 1_000_000,  # ms to ns
            open=k["open"],
            high=k["high"],
            low=k["low"],
            close=k["close"],
            volume=k["volume"],
            timeframe=timeframe,
        )
        bars.append(bar)
    return bars


def bars_to_ticks(bars: list[Bar]) -> list[Tick]:
    """Generate ticks from bars (using close price with spread)."""
    ticks = []
    for bar in bars:
        spread = bar.close * Decimal("0.0001")  # 1 bps spread
        tick = Tick(
            symbol=bar.symbol,
            bid=bar.close - spread,
            ask=bar.close + spread,
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
            volume_24h=bar.volume * Decimal("24"),
        )
        ticks.append(tick)
    return ticks


# =============================================================================
# Fixtures with Real Data
# =============================================================================


@pytest.fixture
def clock() -> Clock:
    """Create a backtest clock."""
    return Clock(ClockType.BACKTEST)


@pytest.fixture
async def btc_bars() -> list[Bar]:
    """Fetch real BTC/USDT 1h bars from Binance."""
    klines = await fetch_binance_klines("BTCUSDT", "1h", 100)
    return klines_to_bars(klines, "BTC/USDT", "1h")


@pytest.fixture
async def eth_bars() -> list[Bar]:
    """Fetch real ETH/USDT 1h bars from Binance."""
    klines = await fetch_binance_klines("ETHUSDT", "1h", 100)
    return klines_to_bars(klines, "ETH/USDT", "1h")


@pytest.fixture
async def sol_bars() -> list[Bar]:
    """Fetch real SOL/USDT 1h bars from Binance."""
    klines = await fetch_binance_klines("SOLUSDT", "1h", 100)
    return klines_to_bars(klines, "SOL/USDT", "1h")


@pytest.fixture
async def multi_crypto_data() -> dict[str, list[Bar]]:
    """Fetch data for multiple cryptocurrencies."""
    symbols = [
        ("BTCUSDT", "BTC/USDT"),
        ("ETHUSDT", "ETH/USDT"),
        ("SOLUSDT", "SOL/USDT"),
        ("BNBUSDT", "BNB/USDT"),
        ("XRPUSDT", "XRP/USDT"),
    ]

    data = {}
    for binance_symbol, libra_symbol in symbols:
        try:
            klines = await fetch_binance_klines(binance_symbol, "1h", 50)
            if klines:
                data[libra_symbol] = klines_to_bars(klines, libra_symbol, "1h")
        except Exception:
            continue

    if not data:
        pytest.skip("Could not fetch any crypto data from Binance")

    return data


# =============================================================================
# Real Data Tests
# =============================================================================


class TestRealBTCData:
    """Test with real BTC/USDT data."""

    @pytest.mark.asyncio
    async def test_btc_data_fetch(self, btc_bars: list[Bar]) -> None:
        """Verify we can fetch real BTC data."""
        assert len(btc_bars) > 0, "Should fetch BTC bars"
        assert btc_bars[0].symbol == "BTC/USDT"

        # Check price is reasonable (BTC should be > $10k)
        latest_price = btc_bars[-1].close
        assert latest_price > Decimal("10000"), f"BTC price {latest_price} seems too low"
        assert latest_price < Decimal("500000"), f"BTC price {latest_price} seems too high"

        print(f"\nBTC/USDT Latest: ${btc_bars[-1].close:,.2f}")
        print(f"High: ${max(b.high for b in btc_bars):,.2f}")
        print(f"Low: ${min(b.low for b in btc_bars):,.2f}")

    @pytest.mark.asyncio
    async def test_btc_backtest_buy_and_hold(
        self, btc_bars: list[Bar], clock: Clock
    ) -> None:
        """Test buy-and-hold strategy with real BTC data."""
        ticks = bars_to_ticks(btc_bars)

        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", btc_bars)
        data_source.add_ticks("BTC/USDT", ticks)

        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("5"),
        )

        await data_client.connect()
        await exec_client.connect()

        # Configure range
        start_time = datetime.fromtimestamp(btc_bars[0].timestamp_ns / 1_000_000_000)
        end_time = datetime.fromtimestamp(btc_bars[-1].timestamp_ns / 1_000_000_000)
        data_client.configure_range(start_time, end_time)
        await data_client.subscribe_bars("BTC/USDT", "1h")

        # Get initial balance
        initial_balance = (await exec_client.get_balance("USDT")).total

        # Buy on first bar
        first_bar = btc_bars[0]
        first_tick = Tick(
            symbol="BTC/USDT",
            bid=first_bar.close - Decimal("10"),
            ask=first_bar.close + Decimal("10"),
            last=first_bar.close,
            timestamp_ns=first_bar.timestamp_ns,
        )
        await exec_client.process_tick(first_tick)

        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),  # Buy 1 BTC
        )
        buy_result = await exec_client.submit_order(buy_order)
        assert buy_result.status == OrderStatus.FILLED
        buy_price = buy_result.average_price

        # Sell on last bar
        last_bar = btc_bars[-1]
        last_tick = Tick(
            symbol="BTC/USDT",
            bid=last_bar.close - Decimal("10"),
            ask=last_bar.close + Decimal("10"),
            last=last_bar.close,
            timestamp_ns=last_bar.timestamp_ns,
        )
        await exec_client.process_tick(last_tick)

        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        sell_result = await exec_client.submit_order(sell_order)
        assert sell_result.status == OrderStatus.FILLED
        sell_price = sell_result.average_price

        # Calculate P&L
        final_balance = (await exec_client.get_balance("USDT")).total
        pnl = final_balance - initial_balance
        pnl_percent = (pnl / initial_balance) * 100

        print(f"\n=== BTC Buy & Hold Results ===")
        print(f"Buy price:  ${buy_price:,.2f}")
        print(f"Sell price: ${sell_price:,.2f}")
        print(f"P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)")
        print(f"Period: {len(btc_bars)} hours")

        await data_client.disconnect()
        await exec_client.disconnect()


class TestRealETHData:
    """Test with real ETH/USDT data."""

    @pytest.mark.asyncio
    async def test_eth_data_fetch(self, eth_bars: list[Bar]) -> None:
        """Verify we can fetch real ETH data."""
        assert len(eth_bars) > 0, "Should fetch ETH bars"
        assert eth_bars[0].symbol == "ETH/USDT"

        latest_price = eth_bars[-1].close
        assert latest_price > Decimal("500"), f"ETH price {latest_price} seems too low"
        assert latest_price < Decimal("50000"), f"ETH price {latest_price} seems too high"

        print(f"\nETH/USDT Latest: ${eth_bars[-1].close:,.2f}")

    @pytest.mark.asyncio
    async def test_eth_momentum_strategy(
        self, eth_bars: list[Bar], clock: Clock
    ) -> None:
        """Test simple momentum strategy with real ETH data."""
        ticks = bars_to_ticks(eth_bars)

        data_source = InMemoryDataSource()
        data_source.add_bars("ETH/USDT", "1h", eth_bars)
        data_source.add_ticks("ETH/USDT", ticks)

        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("50000")},
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("10"),
        )

        await data_client.connect()
        await exec_client.connect()

        start_time = datetime.fromtimestamp(eth_bars[0].timestamp_ns / 1_000_000_000)
        end_time = datetime.fromtimestamp(eth_bars[-1].timestamp_ns / 1_000_000_000)
        data_client.configure_range(start_time, end_time)
        await data_client.subscribe_bars("ETH/USDT", "1h")

        # Simple momentum: buy when price > 5-bar SMA, sell when price < 5-bar SMA
        initial_balance = (await exec_client.get_balance("USDT")).total
        position_size = Decimal("0")
        trades = 0

        prices: list[Decimal] = []
        async for bar in data_client.stream_bars():
            prices.append(bar.close)

            # Update tick for execution
            tick = Tick(
                symbol=bar.symbol,
                bid=bar.close - Decimal("1"),
                ask=bar.close + Decimal("1"),
                last=bar.close,
                timestamp_ns=bar.timestamp_ns,
            )
            await exec_client.process_tick(tick)

            if len(prices) < 5:
                continue

            # Calculate 5-bar SMA
            sma = sum(prices[-5:]) / 5

            # Trading logic
            if bar.close > sma and position_size == 0:
                # Buy signal
                order = Order(
                    symbol="ETH/USDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    amount=Decimal("5.0"),  # Buy 5 ETH
                )
                result = await exec_client.submit_order(order)
                if result.status == OrderStatus.FILLED:
                    position_size = Decimal("5.0")
                    trades += 1

            elif bar.close < sma and position_size > 0:
                # Sell signal
                order = Order(
                    symbol="ETH/USDT",
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=position_size,
                )
                result = await exec_client.submit_order(order)
                if result.status == OrderStatus.FILLED:
                    position_size = Decimal("0")
                    trades += 1

        # Close any remaining position
        if position_size > 0:
            last_tick = Tick(
                symbol="ETH/USDT",
                bid=eth_bars[-1].close - Decimal("1"),
                ask=eth_bars[-1].close + Decimal("1"),
                last=eth_bars[-1].close,
                timestamp_ns=eth_bars[-1].timestamp_ns,
            )
            await exec_client.process_tick(last_tick)
            order = Order(
                symbol="ETH/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=position_size,
            )
            await exec_client.submit_order(order)
            trades += 1

        final_balance = (await exec_client.get_balance("USDT")).total
        pnl = final_balance - initial_balance
        pnl_percent = (pnl / initial_balance) * 100

        print(f"\n=== ETH Momentum Strategy Results ===")
        print(f"Initial: ${initial_balance:,.2f}")
        print(f"Final:   ${final_balance:,.2f}")
        print(f"P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)")
        print(f"Trades: {trades}")

        await data_client.disconnect()
        await exec_client.disconnect()


class TestMultiCryptoPortfolio:
    """Test with multiple cryptocurrencies."""

    @pytest.mark.asyncio
    async def test_multi_crypto_fetch(self, multi_crypto_data: dict[str, list[Bar]]) -> None:
        """Verify we can fetch data for multiple cryptos."""
        print("\n=== Multi-Crypto Latest Prices ===")
        for symbol, bars in multi_crypto_data.items():
            if bars:
                print(f"{symbol}: ${bars[-1].close:,.2f}")

        assert len(multi_crypto_data) >= 2, "Should have at least 2 crypto pairs"

    @pytest.mark.asyncio
    async def test_multi_crypto_equal_weight_portfolio(
        self, multi_crypto_data: dict[str, list[Bar]], clock: Clock
    ) -> None:
        """Test equal-weight portfolio across multiple cryptos."""
        data_source = InMemoryDataSource()
        for symbol, bars in multi_crypto_data.items():
            data_source.add_bars(symbol, "1h", bars)
            ticks = bars_to_ticks(bars)
            data_source.add_ticks(symbol, ticks)

        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},  # $100k
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("10"),
        )

        await data_client.connect()
        await exec_client.connect()

        # Allocate equal weight to each crypto
        initial_balance = (await exec_client.get_balance("USDT")).total
        allocation_per_crypto = initial_balance / len(multi_crypto_data)

        print(f"\n=== Equal Weight Portfolio ===")
        print(f"Starting capital: ${initial_balance:,.2f}")
        print(f"Allocation per crypto: ${allocation_per_crypto:,.2f}")

        # Buy each crypto
        entry_prices = {}
        for symbol, bars in multi_crypto_data.items():
            first_bar = bars[0]
            tick = Tick(
                symbol=symbol,
                bid=first_bar.close - first_bar.close * Decimal("0.001"),
                ask=first_bar.close + first_bar.close * Decimal("0.001"),
                last=first_bar.close,
                timestamp_ns=first_bar.timestamp_ns,
            )
            await exec_client.process_tick(tick)

            # Calculate amount to buy
            amount = allocation_per_crypto / first_bar.close

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=amount.quantize(Decimal("0.0001")),
            )
            result = await exec_client.submit_order(order)
            if result.status == OrderStatus.FILLED:
                entry_prices[symbol] = result.average_price
                print(f"  Bought {symbol} at ${result.average_price:,.2f}")

        # Sell all at end
        exit_prices = {}
        for symbol, bars in multi_crypto_data.items():
            last_bar = bars[-1]
            tick = Tick(
                symbol=symbol,
                bid=last_bar.close - last_bar.close * Decimal("0.001"),
                ask=last_bar.close + last_bar.close * Decimal("0.001"),
                last=last_bar.close,
                timestamp_ns=last_bar.timestamp_ns,
            )
            await exec_client.process_tick(tick)

            position = await exec_client.get_position(symbol)
            if position and position.amount > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=position.amount,
                )
                result = await exec_client.submit_order(order)
                if result.status == OrderStatus.FILLED:
                    exit_prices[symbol] = result.average_price

        final_balance = (await exec_client.get_balance("USDT")).total
        pnl = final_balance - initial_balance
        pnl_percent = (pnl / initial_balance) * 100

        print(f"\n=== Portfolio Performance ===")
        for symbol in multi_crypto_data:
            if symbol in entry_prices and symbol in exit_prices:
                crypto_return = (exit_prices[symbol] - entry_prices[symbol]) / entry_prices[symbol] * 100
                print(f"  {symbol}: {crypto_return:+.2f}%")

        print(f"\nPortfolio P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)")

        await data_client.disconnect()
        await exec_client.disconnect()


class TestRealSOLData:
    """Test with real SOL/USDT data."""

    @pytest.mark.asyncio
    async def test_sol_data_fetch(self, sol_bars: list[Bar]) -> None:
        """Verify we can fetch real SOL data."""
        assert len(sol_bars) > 0, "Should fetch SOL bars"
        assert sol_bars[0].symbol == "SOL/USDT"

        latest_price = sol_bars[-1].close
        print(f"\nSOL/USDT Latest: ${latest_price:,.2f}")

    @pytest.mark.asyncio
    async def test_sol_mean_reversion_strategy(
        self, sol_bars: list[Bar], clock: Clock
    ) -> None:
        """Test mean reversion strategy with real SOL data."""
        ticks = bars_to_ticks(sol_bars)

        data_source = InMemoryDataSource()
        data_source.add_bars("SOL/USDT", "1h", sol_bars)
        data_source.add_ticks("SOL/USDT", ticks)

        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("20000")},
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("15"),  # Higher slippage for SOL
        )

        await data_client.connect()
        await exec_client.connect()

        start_time = datetime.fromtimestamp(sol_bars[0].timestamp_ns / 1_000_000_000)
        end_time = datetime.fromtimestamp(sol_bars[-1].timestamp_ns / 1_000_000_000)
        data_client.configure_range(start_time, end_time)
        await data_client.subscribe_bars("SOL/USDT", "1h")

        # Mean reversion: buy when price drops 2% below 20-bar SMA, sell when it reverts
        initial_balance = (await exec_client.get_balance("USDT")).total
        position_size = Decimal("0")
        trades = 0

        prices: list[Decimal] = []
        async for bar in data_client.stream_bars():
            prices.append(bar.close)

            tick = Tick(
                symbol=bar.symbol,
                bid=bar.close - Decimal("0.1"),
                ask=bar.close + Decimal("0.1"),
                last=bar.close,
                timestamp_ns=bar.timestamp_ns,
            )
            await exec_client.process_tick(tick)

            if len(prices) < 20:
                continue

            sma = sum(prices[-20:]) / 20
            deviation = (bar.close - sma) / sma

            # Buy when price is 2% below SMA
            if deviation < Decimal("-0.02") and position_size == 0:
                order = Order(
                    symbol="SOL/USDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    amount=Decimal("50"),  # Buy 50 SOL
                )
                result = await exec_client.submit_order(order)
                if result.status == OrderStatus.FILLED:
                    position_size = Decimal("50")
                    trades += 1

            # Sell when price reverts to SMA or goes 1% above
            elif deviation > Decimal("0.01") and position_size > 0:
                order = Order(
                    symbol="SOL/USDT",
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=position_size,
                )
                result = await exec_client.submit_order(order)
                if result.status == OrderStatus.FILLED:
                    position_size = Decimal("0")
                    trades += 1

        # Close position
        if position_size > 0:
            tick = Tick(
                symbol="SOL/USDT",
                bid=sol_bars[-1].close - Decimal("0.1"),
                ask=sol_bars[-1].close + Decimal("0.1"),
                last=sol_bars[-1].close,
                timestamp_ns=sol_bars[-1].timestamp_ns,
            )
            await exec_client.process_tick(tick)
            order = Order(
                symbol="SOL/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=position_size,
            )
            await exec_client.submit_order(order)
            trades += 1

        final_balance = (await exec_client.get_balance("USDT")).total
        pnl = final_balance - initial_balance
        pnl_percent = (pnl / initial_balance) * 100

        print(f"\n=== SOL Mean Reversion Results ===")
        print(f"P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)")
        print(f"Trades: {trades}")

        await data_client.disconnect()
        await exec_client.disconnect()

"""
End-to-End tests for RiskEngine with REAL market data from Binance.

Tests the RiskEngine pre-trade validation with actual cryptocurrency prices
to ensure realistic risk management behavior.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from datetime import datetime
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
    Tick,
)
from libra.clients.backtest_execution_client import SlippageModel
from libra.clients.data_client import Instrument
from libra.core.clock import Clock, ClockType
from libra.core.events import EventType
from libra.core.kernel import KernelConfig, TradingKernel
from libra.risk import (
    RiskEngine,
    RiskEngineConfig,
    RiskLimits,
    SymbolLimits,
    TradingState,
)
from libra.strategies.protocol import Bar


# =============================================================================
# Real Data Fetching
# =============================================================================


async def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch real OHLCV data from Binance public API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        klines = []
        for k in data:
            klines.append({
                "timestamp_ms": k[0],
                "open": Decimal(k[1]),
                "high": Decimal(k[2]),
                "low": Decimal(k[3]),
                "close": Decimal(k[4]),
                "volume": Decimal(k[5]),
            })
        return klines
    except Exception as e:
        pytest.skip(f"Could not fetch Binance data: {e}")
        return []


async def fetch_binance_ticker(symbol: str = "BTCUSDT") -> dict[str, Decimal]:
    """Fetch current ticker from Binance."""
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        return {
            "last": Decimal(data["lastPrice"]),
            "bid": Decimal(data["bidPrice"]),
            "ask": Decimal(data["askPrice"]),
            "high": Decimal(data["highPrice"]),
            "low": Decimal(data["lowPrice"]),
            "volume": Decimal(data["volume"]),
        }
    except Exception as e:
        pytest.skip(f"Could not fetch Binance ticker: {e}")
        return {}


def klines_to_bars(klines: list[dict[str, Any]], symbol: str, timeframe: str) -> list[Bar]:
    """Convert Binance klines to Bar objects."""
    return [
        Bar(
            symbol=symbol,
            timestamp_ns=k["timestamp_ms"] * 1_000_000,
            open=k["open"],
            high=k["high"],
            low=k["low"],
            close=k["close"],
            volume=k["volume"],
            timeframe=timeframe,
        )
        for k in klines
    ]


def bars_to_ticks(bars: list[Bar]) -> list[Tick]:
    """Generate ticks from bars."""
    return [
        Tick(
            symbol=bar.symbol,
            bid=bar.close - bar.close * Decimal("0.0001"),
            ask=bar.close + bar.close * Decimal("0.0001"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
            volume_24h=bar.volume * Decimal("24"),
        )
        for bar in bars
    ]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clock() -> Clock:
    """Create a backtest clock."""
    return Clock(ClockType.BACKTEST)


@pytest.fixture
async def btc_data() -> tuple[list[Bar], Decimal]:
    """Fetch real BTC data and current price."""
    klines = await fetch_binance_klines("BTCUSDT", "1h", 100)
    bars = klines_to_bars(klines, "BTC/USDT", "1h")
    current_price = bars[-1].close if bars else Decimal("50000")
    return bars, current_price


@pytest.fixture
async def eth_data() -> tuple[list[Bar], Decimal]:
    """Fetch real ETH data and current price."""
    klines = await fetch_binance_klines("ETHUSDT", "1h", 100)
    bars = klines_to_bars(klines, "ETH/USDT", "1h")
    current_price = bars[-1].close if bars else Decimal("3000")
    return bars, current_price


@pytest.fixture
def btc_instrument() -> Instrument:
    """BTC instrument with real Binance specs."""
    return Instrument(
        symbol="BTC/USDT",
        base="BTC",
        quote="USDT",
        exchange="binance",
        lot_size=Decimal("0.00001"),  # 5 decimals
        tick_size=Decimal("0.01"),  # 2 decimals for price
        min_notional=Decimal("10"),
    )


@pytest.fixture
def eth_instrument() -> Instrument:
    """ETH instrument with real Binance specs."""
    return Instrument(
        symbol="ETH/USDT",
        base="ETH",
        quote="USDT",
        exchange="binance",
        lot_size=Decimal("0.0001"),  # 4 decimals
        tick_size=Decimal("0.01"),
        min_notional=Decimal("10"),
    )


@pytest.fixture
def risk_engine_config() -> RiskEngineConfig:
    """Realistic risk engine configuration."""
    return RiskEngineConfig(
        limits=RiskLimits(
            max_total_exposure=Decimal("100000"),  # $100k max
            max_single_position_pct=Decimal("0.25"),  # 25% max per position
            max_daily_loss_pct=Decimal("-0.05"),  # -5% daily loss limit
            max_weekly_loss_pct=Decimal("-0.10"),  # -10% weekly
            max_total_drawdown_pct=Decimal("-0.20"),  # -20% max drawdown
            max_orders_per_second=5,
            max_orders_per_minute=100,
            symbol_limits={
                "BTC/USDT": SymbolLimits(
                    max_position_size=Decimal("2.0"),  # Max 2 BTC
                    max_notional_per_order=Decimal("50000"),  # $50k max per order
                    max_order_rate=3,
                ),
                "ETH/USDT": SymbolLimits(
                    max_position_size=Decimal("20.0"),  # Max 20 ETH
                    max_notional_per_order=Decimal("30000"),  # $30k max per order
                    max_order_rate=3,
                ),
            },
        ),
        enable_self_trade_prevention=True,
        enable_price_collar=True,
        price_collar_pct=Decimal("0.05"),  # 5% price collar
        enable_precision_validation=True,
        max_modify_rate=10,
    )


# =============================================================================
# E2E Tests with Real Data
# =============================================================================


class TestRiskEngineWithRealBTCData:
    """Test RiskEngine with real BTC market data."""

    @pytest.mark.asyncio
    async def test_position_limit_with_real_btc_price(
        self, btc_data: tuple[list[Bar], Decimal], risk_engine_config: RiskEngineConfig
    ) -> None:
        """Test position limit enforcement with real BTC prices."""
        bars, current_price = btc_data
        engine = RiskEngine(config=risk_engine_config)

        print(f"\n=== BTC Position Limit Test ===")
        print(f"Current BTC price: ${current_price:,.2f}")
        print(f"Max position: 2.0 BTC")
        print(f"Max notional: $50,000")

        # Calculate max amount that fits within notional limit
        max_notional = Decimal("50000")
        max_amount_by_notional = (max_notional / current_price).quantize(Decimal("0.00001"))

        # Order within both position limit and notional limit should pass
        valid_amount = min(Decimal("0.5"), max_amount_by_notional)
        valid_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=valid_amount,
        )
        result = engine.validate_order(valid_order, current_price)
        assert result.passed is True, f"Valid order should pass: {result.reason}"
        print(f"Order for {valid_amount} BTC (${float(valid_amount * current_price):,.2f}): PASSED")

        # Order exceeding position limit should fail
        invalid_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("2.5"),  # 2.5 BTC > 2.0 limit
        )
        result = engine.validate_order(invalid_order, current_price)
        assert result.passed is False
        # Could fail on either position_limit or notional_limit depending on price
        assert result.check_name in ("position_limit", "notional_limit")
        print(f"Order for 2.5 BTC (${Decimal('2.5') * current_price:,.2f}): DENIED - {result.check_name}")

    @pytest.mark.asyncio
    async def test_notional_limit_with_real_btc_price(
        self, btc_data: tuple[list[Bar], Decimal], risk_engine_config: RiskEngineConfig
    ) -> None:
        """Test notional limit enforcement with real BTC prices."""
        bars, current_price = btc_data
        engine = RiskEngine(config=risk_engine_config)

        print(f"\n=== BTC Notional Limit Test ===")
        print(f"Current BTC price: ${current_price:,.2f}")
        print(f"Max notional per order: $50,000")

        # Calculate amount that would exceed $50k notional
        max_amount = Decimal("50000") / current_price
        exceeding_amount = max_amount * Decimal("1.2")  # 20% over

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=exceeding_amount.quantize(Decimal("0.00001")),
        )
        notional = exceeding_amount * current_price

        result = engine.validate_order(order, current_price)
        if notional > Decimal("50000"):
            assert result.passed is False
            assert result.check_name == "notional_limit"
            print(f"Order notional ${notional:,.2f}: DENIED (exceeds $50k)")
        else:
            print(f"Order notional ${notional:,.2f}: PASSED")

    @pytest.mark.asyncio
    async def test_price_collar_with_real_btc_price(
        self,
        btc_data: tuple[list[Bar], Decimal],
        btc_instrument: Instrument,
        risk_engine_config: RiskEngineConfig,
    ) -> None:
        """Test price collar (fat-finger protection) with real prices."""
        bars, current_price = btc_data
        engine = RiskEngine(config=risk_engine_config)

        print(f"\n=== BTC Price Collar Test (5% collar) ===")
        print(f"Current BTC price: ${current_price:,.2f}")
        print(f"Acceptable range: ${current_price * Decimal('0.95'):,.2f} - ${current_price * Decimal('1.05'):,.2f}")

        # Order within collar should pass
        valid_price = (current_price * Decimal("1.03")).quantize(Decimal("0.01"))  # 3% above
        valid_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=valid_price,
        )
        result = engine.validate_order(valid_order, current_price, btc_instrument)
        assert result.passed is True
        print(f"Limit order at ${valid_price:,.2f} (+3%): PASSED")

        # Order outside collar should fail
        invalid_price = (current_price * Decimal("1.10")).quantize(Decimal("0.01"))  # 10% above
        invalid_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=invalid_price,
        )
        result = engine.validate_order(invalid_order, current_price, btc_instrument)
        assert result.passed is False
        assert result.check_name == "price_collar"
        print(f"Limit order at ${invalid_price:,.2f} (+10%): DENIED - price collar")

    @pytest.mark.asyncio
    async def test_precision_validation_with_real_btc_specs(
        self,
        btc_data: tuple[list[Bar], Decimal],
        btc_instrument: Instrument,
        risk_engine_config: RiskEngineConfig,
    ) -> None:
        """Test price/quantity precision validation with real specs."""
        bars, current_price = btc_data
        engine = RiskEngine(config=risk_engine_config)

        print(f"\n=== BTC Precision Validation Test ===")
        print(f"Tick size: {btc_instrument.tick_size}")
        print(f"Lot size: {btc_instrument.lot_size}")

        # Valid precision
        valid_price = (current_price).quantize(Decimal("0.01"))  # 2 decimals
        valid_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.10000"),  # Valid: 5 decimals
            price=valid_price,
        )
        result = engine.validate_order(valid_order, current_price, btc_instrument)
        assert result.passed is True
        print(f"Order with price {valid_price} and amount 0.10000: PASSED")

        # Invalid price precision (3 decimals when tick_size is 0.01)
        invalid_price = current_price + Decimal("0.001")  # 3 decimals
        invalid_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=invalid_price,
        )
        result = engine.validate_order(invalid_order, current_price, btc_instrument)
        assert result.passed is False
        assert result.check_name == "price_precision"
        print(f"Order with invalid price {invalid_price}: DENIED - precision")


class TestRiskEngineWithRealETHData:
    """Test RiskEngine with real ETH market data."""

    @pytest.mark.asyncio
    async def test_eth_trading_scenario(
        self,
        eth_data: tuple[list[Bar], Decimal],
        eth_instrument: Instrument,
        risk_engine_config: RiskEngineConfig,
    ) -> None:
        """Test realistic ETH trading scenario with risk checks."""
        bars, current_price = eth_data
        engine = RiskEngine(config=risk_engine_config)

        print(f"\n=== ETH Trading Scenario ===")
        print(f"Current ETH price: ${current_price:,.2f}")
        print(f"Max position: 20 ETH")
        print(f"Max notional: $30,000")

        # Simulate a series of trades
        trades_attempted = 0
        trades_passed = 0
        trades_denied = 0

        # Try to build a position incrementally
        for i in range(5):
            amount = Decimal("5.0")  # 5 ETH per order
            order = Order(
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=amount,
            )

            result = engine.validate_order(order, current_price, eth_instrument)
            trades_attempted += 1

            if result.passed:
                trades_passed += 1
                # Simulate position update
                from libra.gateways.protocol import Position, PositionSide
                engine.update_position(Position(
                    symbol="ETH/USDT",
                    side=PositionSide.LONG,
                    amount=amount * (i + 1),
                    entry_price=current_price,
                    current_price=current_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"),
                ))
                print(f"Trade {i+1}: BUY 5 ETH - PASSED (position now: {5*(i+1)} ETH)")
            else:
                trades_denied += 1
                print(f"Trade {i+1}: BUY 5 ETH - DENIED: {result.reason}")

        print(f"\nSummary: {trades_passed} passed, {trades_denied} denied out of {trades_attempted}")

        # Should hit position limit after 4 trades (20 ETH)
        assert trades_passed == 4
        assert trades_denied == 1


class TestRiskEngineSelfTradeWithRealData:
    """Test self-trade prevention with realistic order book scenarios."""

    @pytest.mark.asyncio
    async def test_self_trade_prevention_realistic_scenario(
        self,
        btc_data: tuple[list[Bar], Decimal],
        btc_instrument: Instrument,
        risk_engine_config: RiskEngineConfig,
    ) -> None:
        """Test self-trade prevention with real market prices."""
        bars, current_price = btc_data
        engine = RiskEngine(config=risk_engine_config)

        print(f"\n=== Self-Trade Prevention Test ===")
        print(f"Current BTC price: ${current_price:,.2f}")

        # Place a sell limit order above current price
        sell_price = (current_price * Decimal("1.02")).quantize(Decimal("0.01"))
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=sell_price,
            client_order_id="sell-001",
        )

        # First sell order should pass
        result = engine.validate_order(sell_order, current_price, btc_instrument)
        assert result.passed is True
        engine.add_open_order(sell_order)
        print(f"SELL order at ${sell_price:,.2f}: PASSED (now tracked)")

        # Try to place buy order at or above sell price - should be blocked
        buy_price = sell_price  # Same price = would self-trade
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.3"),
            price=buy_price,
            client_order_id="buy-001",
        )

        result = engine.validate_order(buy_order, current_price, btc_instrument)
        assert result.passed is False
        assert result.check_name == "self_trade"
        print(f"BUY order at ${buy_price:,.2f}: DENIED - would self-trade with open SELL")

        # Buy order at lower price should pass
        safe_buy_price = (sell_price * Decimal("0.98")).quantize(Decimal("0.01"))
        safe_buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.3"),
            price=safe_buy_price,
        )

        result = engine.validate_order(safe_buy_order, current_price, btc_instrument)
        assert result.passed is True
        print(f"BUY order at ${safe_buy_price:,.2f}: PASSED (below open SELL)")


class TestRiskEngineFullBacktest:
    """Full backtest with RiskEngine integrated."""

    @pytest.mark.asyncio
    async def test_btc_momentum_strategy_with_risk_engine(
        self,
        btc_data: tuple[list[Bar], Decimal],
        btc_instrument: Instrument,
        risk_engine_config: RiskEngineConfig,
        clock: Clock,
    ) -> None:
        """Run momentum strategy with full risk engine integration."""
        bars, current_price = btc_data
        if len(bars) < 20:
            pytest.skip("Not enough data for test")

        ticks = bars_to_ticks(bars)
        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", bars)
        data_source.add_ticks("BTC/USDT", ticks)

        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("5"),
        )

        engine = RiskEngine(config=risk_engine_config)

        await data_client.connect()
        await exec_client.connect()

        start_time = datetime.fromtimestamp(bars[0].timestamp_ns / 1_000_000_000)
        end_time = datetime.fromtimestamp(bars[-1].timestamp_ns / 1_000_000_000)
        data_client.configure_range(start_time, end_time)
        await data_client.subscribe_bars("BTC/USDT", "1h")

        print(f"\n=== BTC Momentum Strategy with Risk Engine ===")
        print(f"Period: {start_time} to {end_time}")
        print(f"Bars: {len(bars)}")

        initial_balance = (await exec_client.get_balance("USDT")).total
        position_size = Decimal("0")
        prices: list[Decimal] = []

        orders_attempted = 0
        orders_passed = 0
        orders_denied = 0

        async for bar in data_client.stream_bars():
            prices.append(bar.close)

            tick = Tick(
                symbol=bar.symbol,
                bid=bar.close - Decimal("10"),
                ask=bar.close + Decimal("10"),
                last=bar.close,
                timestamp_ns=bar.timestamp_ns,
            )
            await exec_client.process_tick(tick)

            if len(prices) < 10:
                continue

            sma = sum(prices[-10:]) / 10

            # Buy signal: price > SMA and no position
            if bar.close > sma and position_size == 0:
                order = Order(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    amount=Decimal("0.5"),
                )

                orders_attempted += 1
                risk_result = engine.validate_order(order, bar.close, btc_instrument)

                if risk_result.passed:
                    orders_passed += 1
                    result = await exec_client.submit_order(order)
                    if result.status == OrderStatus.FILLED:
                        position_size = Decimal("0.5")
                        engine.add_open_order(order)
                else:
                    orders_denied += 1

            # Sell signal: price < SMA and has position
            elif bar.close < sma and position_size > 0:
                order = Order(
                    symbol="BTC/USDT",
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=position_size,
                )

                orders_attempted += 1
                risk_result = engine.validate_order(order, bar.close, btc_instrument)

                if risk_result.passed:
                    orders_passed += 1
                    result = await exec_client.submit_order(order)
                    if result.status == OrderStatus.FILLED:
                        position_size = Decimal("0")
                        engine.clear_open_orders("BTC/USDT")
                else:
                    orders_denied += 1

        # Close any remaining position
        if position_size > 0:
            last_tick = Tick(
                symbol="BTC/USDT",
                bid=bars[-1].close - Decimal("10"),
                ask=bars[-1].close + Decimal("10"),
                last=bars[-1].close,
                timestamp_ns=bars[-1].timestamp_ns,
            )
            await exec_client.process_tick(last_tick)
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=position_size,
            )
            await exec_client.submit_order(order)

        final_balance = (await exec_client.get_balance("USDT")).total
        pnl = final_balance - initial_balance
        pnl_percent = (pnl / initial_balance) * 100

        print(f"\n=== Results ===")
        print(f"Initial balance: ${initial_balance:,.2f}")
        print(f"Final balance: ${final_balance:,.2f}")
        print(f"P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)")
        print(f"\nRisk Engine Stats:")
        print(f"  Orders attempted: {orders_attempted}")
        print(f"  Orders passed: {orders_passed}")
        print(f"  Orders denied: {orders_denied}")

        stats = engine.get_stats()
        print(f"  Avg check latency: {stats['avg_check_latency_us']:.1f}μs")

        await data_client.disconnect()
        await exec_client.disconnect()


class TestRiskEngineTradingKernelE2E:
    """Test RiskEngine integrated with TradingKernel."""

    @pytest.mark.asyncio
    async def test_kernel_order_flow_with_real_data(
        self,
        btc_data: tuple[list[Bar], Decimal],
        btc_instrument: Instrument,
        risk_engine_config: RiskEngineConfig,
        clock: Clock,
    ) -> None:
        """Test complete order flow through TradingKernel with RiskEngine."""
        bars, current_price = btc_data

        ticks = bars_to_ticks(bars)
        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", bars)
        data_source.add_ticks("BTC/USDT", ticks)

        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
        )

        config = KernelConfig(environment="backtest")
        kernel = TradingKernel(config)

        engine = RiskEngine(config=risk_engine_config)
        kernel.set_risk_engine(engine)
        kernel.set_data_client(data_client)
        kernel.set_execution_client(exec_client)

        # Track ORDER_DENIED events
        denied_events = []

        def capture_denied(event):
            denied_events.append(event)

        print(f"\n=== TradingKernel + RiskEngine E2E Test ===")
        print(f"BTC price: ${current_price:,.2f}")

        async with kernel:
            # Subscribe to denied events
            kernel.bus.subscribe(EventType.ORDER_DENIED, capture_denied)

            # Process initial tick
            first_tick = ticks[0]
            await exec_client.process_tick(first_tick)

            # Test 1: Valid order through kernel.submit_order()
            valid_order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("0.5"),
            )
            result = await kernel.submit_order(valid_order, current_price, btc_instrument)
            assert result.status == OrderStatus.FILLED
            print(f"Valid order (0.5 BTC): {result.status.value}")

            # Test 2: Order exceeding position limit
            large_order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("5.0"),  # Would exceed 2.0 limit
            )
            result = await kernel.submit_order(large_order, current_price, btc_instrument)
            assert result.status == OrderStatus.REJECTED
            print(f"Large order (5.0 BTC): {result.status.value} - position limit")

            # Test 3: Order with bad price (outside collar)
            bad_price_order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.1"),
                price=current_price * Decimal("1.20"),  # 20% above market
            )
            result = await kernel.submit_order(bad_price_order, current_price, btc_instrument)
            assert result.status == OrderStatus.REJECTED
            print(f"Bad price order: {result.status.value} - price collar")

            # Give time for events to process
            await asyncio.sleep(0.1)

            print(f"\nORDER_DENIED events captured: {len(denied_events)}")
            for event in denied_events:
                print(f"  - {event.payload.get('check')}: {event.payload.get('reason', '')[:50]}")

        print(f"\nKernel health check: {kernel.is_healthy()}")
        print(f"Risk engine stats: {engine.get_stats()['orders_checked']} orders checked")

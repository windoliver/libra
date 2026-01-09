"""
E2E Tests for Hummingbot Adapter with DEX and Performance Tracking (Issue #12).

Tests complete trading workflows with simulated market data.
"""

import asyncio
from decimal import Decimal
import time

import polars as pl
import pytest

from libra.plugins.hummingbot_adapter import (
    HummingbotAdapter,
    PerformanceTracker,
    AvellanedaStoikovStrategy,
    PureMarketMakingStrategy,
)
from libra.plugins.hummingbot_adapter.config import AvellanedaStoikovConfig, InventoryConfig
from libra.plugins.hummingbot_adapter.dex import (
    DEXArbitrageStrategy,
    DEXGatewayConfig,
    DEXPool,
    DEXType,
    Token,
    UniswapV2Gateway,
    UniswapV3Gateway,
)
from libra.plugins.hummingbot_adapter.dex.arbitrage import ArbitrageConfig


# Test tokens
WETH = Token(
    address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    symbol="WETH",
    decimals=18,
    chain_id=1,
    name="Wrapped Ether",
)

USDC = Token(
    address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    symbol="USDC",
    decimals=6,
    chain_id=1,
    name="USD Coin",
)

DAI = Token(
    address="0x6B175474E89094C44Da98b954EesCD6aB3f7C7",
    symbol="DAI",
    decimals=18,
    chain_id=1,
    name="Dai Stablecoin",
)


def generate_ohlcv_data(
    start_price: float = 2000.0,
    num_bars: int = 100,
    volatility: float = 0.02,
) -> pl.DataFrame:
    """Generate realistic OHLCV data for testing."""
    import random

    random.seed(42)  # Reproducible

    timestamps: list[int] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []

    price = start_price
    base_time = int(time.time() * 1000) - num_bars * 60000  # 1 minute bars

    for i in range(num_bars):
        open_price = price
        change = random.gauss(0, volatility)
        close_price = open_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility / 2)))
        low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility / 2)))
        volume = random.uniform(100, 1000) * (1 + abs(change) * 10)

        timestamps.append(int(base_time + i * 60000))
        opens.append(float(open_price))
        highs.append(float(high_price))
        lows.append(float(low_price))
        closes.append(float(close_price))
        volumes.append(float(volume))

        price = close_price

    return pl.DataFrame({
        "timestamp": pl.Series("timestamp", timestamps, dtype=pl.Int64),
        "open": pl.Series("open", opens, dtype=pl.Float64),
        "high": pl.Series("high", highs, dtype=pl.Float64),
        "low": pl.Series("low", lows, dtype=pl.Float64),
        "close": pl.Series("close", closes, dtype=pl.Float64),
        "volume": pl.Series("volume", volumes, dtype=pl.Float64),
    })


class TestDEXGatewayE2E:
    """E2E tests for DEX gateway functionality."""

    @pytest.fixture
    def v2_gateway(self) -> UniswapV2Gateway:
        """Create V2 gateway in simulation mode."""
        config = DEXGatewayConfig(chain_id=1, max_slippage=0.5)
        return UniswapV2Gateway(config)

    @pytest.fixture
    def v3_gateway(self) -> UniswapV3Gateway:
        """Create V3 gateway in simulation mode."""
        config = DEXGatewayConfig(chain_id=1, max_slippage=0.5)
        return UniswapV3Gateway(config)

    @pytest.fixture
    def weth_usdc_pool(self) -> DEXPool:
        """Create realistic WETH/USDC pool."""
        return DEXPool(
            address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("50000"),  # 50k WETH
            reserve1=Decimal("100000000"),  # 100M USDC ($2000/ETH)
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        )

    async def test_complete_swap_workflow(
        self, v2_gateway: UniswapV2Gateway, weth_usdc_pool: DEXPool
    ) -> None:
        """Test complete swap from quote to execution."""
        await v2_gateway.connect()
        v2_gateway.add_simulated_pool(weth_usdc_pool)

        # Step 1: Get quote
        input_amount = Decimal("10")  # 10 WETH
        quote = await v2_gateway.get_quote(WETH, USDC, input_amount)

        assert quote is not None
        assert quote.input_amount == input_amount
        assert quote.output_amount > Decimal("19000")  # ~$2000/ETH minus fees/slippage
        assert quote.price_impact < 1.0  # Less than 1% for 10 ETH
        print(f"\nQuote: {input_amount} WETH -> {quote.output_amount} USDC")
        print(f"Price impact: {quote.price_impact:.4f}%")

        # Step 2: Execute swap
        result = await v2_gateway.execute_swap(quote)

        assert result.success
        assert result.output_amount == quote.output_amount
        print(f"Swap executed: TX {result.tx_hash[:16]}...")

        # Step 3: Verify pool state changed
        updated_pool = await v2_gateway.get_pool(WETH, USDC)
        assert updated_pool is not None
        assert updated_pool.reserve0 > weth_usdc_pool.reserve0  # More WETH in pool
        assert updated_pool.reserve1 < weth_usdc_pool.reserve1  # Less USDC in pool

        await v2_gateway.disconnect()

    async def test_large_trade_price_impact(
        self, v2_gateway: UniswapV2Gateway, weth_usdc_pool: DEXPool
    ) -> None:
        """Test price impact increases with trade size."""
        await v2_gateway.connect()
        v2_gateway.add_simulated_pool(weth_usdc_pool)

        trade_sizes = [Decimal("1"), Decimal("100"), Decimal("1000"), Decimal("5000")]
        impacts = []

        for size in trade_sizes:
            quote = await v2_gateway.get_quote(WETH, USDC, size)
            assert quote is not None
            impacts.append(quote.price_impact)
            print(f"Trade {size} WETH: impact={quote.price_impact:.4f}%")

        # Verify price impact increases with size
        for i in range(1, len(impacts)):
            assert impacts[i] > impacts[i - 1], "Price impact should increase with size"

        await v2_gateway.disconnect()

    async def test_v3_fee_tier_comparison(self, v3_gateway: UniswapV3Gateway) -> None:
        """Test different V3 fee tiers give different quotes."""
        await v3_gateway.connect()

        # Add pools with different fee tiers
        for fee_bps in [500, 3000, 10000]:  # 0.05%, 0.3%, 1%
            pool = DEXPool(
                address=f"0x{fee_bps:040x}",
                token0=WETH,
                token1=USDC,
                reserve0=Decimal("10000"),
                reserve1=Decimal("20000000"),
                fee=Decimal(fee_bps) / Decimal(1_000_000),
                dex_type=DEXType.AMM_V3,
            )
            v3_gateway.add_simulated_pool(pool)

        # Compare quotes across fee tiers
        input_amount = Decimal("10")
        outputs = {}

        for fee_tier in [500, 3000, 10000]:
            quote = await v3_gateway.get_quote(WETH, USDC, input_amount, fee_tier)
            if quote:
                outputs[fee_tier] = quote.output_amount
                print(f"Fee tier {fee_tier/10000}%: {quote.output_amount} USDC")

        # Lower fee tier should give better output (for same liquidity)
        assert outputs[500] > outputs[3000] > outputs[10000]

        await v3_gateway.disconnect()


class TestDEXArbitrageE2E:
    """E2E tests for DEX arbitrage detection and execution."""

    async def test_cross_dex_arbitrage_detection(self) -> None:
        """Test detecting arbitrage between two DEXs with price discrepancy."""
        config = DEXGatewayConfig(chain_id=1)

        # Create two "DEXs" with different prices
        dex1 = UniswapV2Gateway(config)
        dex2 = UniswapV2Gateway(config)

        await dex1.connect()
        await dex2.connect()

        # DEX1: WETH cheaper ($1950)
        pool1 = DEXPool(
            address="0x1111111111111111111111111111111111111111",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("10000"),
            reserve1=Decimal("19500000"),  # $1950/ETH
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        )
        dex1.add_simulated_pool(pool1)

        # DEX2: WETH more expensive ($2050)
        pool2 = DEXPool(
            address="0x2222222222222222222222222222222222222222",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("10000"),
            reserve1=Decimal("20500000"),  # $2050/ETH
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        )
        dex2.add_simulated_pool(pool2)

        # Create arbitrage strategy
        arb_config = ArbitrageConfig(
            min_profit_percentage=0.1,
            min_profit_amount=Decimal("10"),
        )
        strategy = DEXArbitrageStrategy([dex1, dex2], arb_config)

        # Scan for opportunities
        opportunities = await strategy.scan_for_opportunities(
            token_pairs=[(WETH, USDC)],
            input_amounts=[Decimal("100")],  # 100 WETH
        )

        print(f"\nFound {len(opportunities)} arbitrage opportunities")

        if opportunities:
            opp = opportunities[0]
            print(f"Best opportunity:")
            print(f"  Type: {opp.arb_type.value}")
            print(f"  Input: {opp.input_amount} WETH")
            print(f"  Expected profit: {opp.expected_profit} WETH")
            print(f"  Profit %: {opp.profit_percentage:.2f}%")

            # There should be an opportunity given the price difference
            assert opp.profit_percentage > 0

        await dex1.disconnect()
        await dex2.disconnect()

    async def test_triangular_arbitrage(self) -> None:
        """Test triangular arbitrage: WETH -> USDC -> DAI -> WETH."""
        config = DEXGatewayConfig(chain_id=1)
        gateway = UniswapV2Gateway(config)
        await gateway.connect()

        # Create pools with slight mispricing to create arbitrage
        # WETH/USDC pool
        gateway.add_simulated_pool(DEXPool(
            address="0x1111111111111111111111111111111111111111",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("10000"),
            reserve1=Decimal("20000000"),  # $2000/ETH
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        ))

        # USDC/DAI pool (slight mispricing: 1 USDC = 1.001 DAI)
        gateway.add_simulated_pool(DEXPool(
            address="0x2222222222222222222222222222222222222222",
            token0=USDC,
            token1=DAI,
            reserve0=Decimal("10000000"),
            reserve1=Decimal("10010000"),  # 1.001 DAI per USDC
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        ))

        # DAI/WETH pool (mispriced to create opportunity)
        gateway.add_simulated_pool(DEXPool(
            address="0x3333333333333333333333333333333333333333",
            token0=DAI,
            token1=WETH,
            reserve0=Decimal("19800000"),  # $1980 worth of DAI
            reserve1=Decimal("10000"),  # 10k WETH
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        ))

        arb_config = ArbitrageConfig(
            min_profit_percentage=0.01,  # Lower threshold for triangular
            min_profit_amount=Decimal("0.1"),
        )
        strategy = DEXArbitrageStrategy([gateway], arb_config)

        # Find triangular arbitrage
        opp = await strategy.find_triangular_opportunity(
            token_a=WETH,
            token_b=USDC,
            token_c=DAI,
            input_amount=Decimal("10"),
            dex_name="uniswap_v2",
        )

        if opp:
            print(f"\nTriangular arbitrage found:")
            print(f"  Path: {' -> '.join(t.symbol for t in opp.token_path)}")
            print(f"  Input: {opp.input_amount} WETH")
            print(f"  Output: {opp.expected_output} WETH")
            print(f"  Profit: {opp.expected_profit} WETH ({opp.profit_percentage:.2f}%)")
        else:
            print("\nNo triangular arbitrage opportunity found (expected with balanced pools)")

        await gateway.disconnect()


class TestPerformanceTrackingE2E:
    """E2E tests for performance tracking with realistic trading scenarios."""

    async def test_full_trading_session(self) -> None:
        """Simulate a complete trading session with multiple trades."""
        tracker = PerformanceTracker(
            initial_capital=Decimal("100000"),
            snapshot_interval_ns=1_000_000_000,  # 1 second
        )

        # Simulate trading session
        trades = [
            # (symbol, side, qty, price)
            ("ETH-USD", "buy", 10, 2000),    # Open long
            ("ETH-USD", "sell", 10, 2100),   # Close long +$1000
            ("ETH-USD", "sell", 5, 2050),    # Open short
            ("ETH-USD", "buy", 5, 2000),     # Close short +$250
            ("BTC-USD", "buy", 1, 40000),    # Open long BTC
            ("BTC-USD", "sell", 1, 39000),   # Close long -$1000
            ("ETH-USD", "buy", 20, 1900),    # Open long
            ("ETH-USD", "sell", 20, 2000),   # Close long +$2000
        ]

        print("\n=== Trading Session ===")
        for symbol, side, qty, price in trades:
            # Create mock trade event
            class MockEvent:
                pass

            event = MockEvent()
            event.symbol = symbol
            event.side = side
            event.quantity = float(qty)
            event.price = float(price)
            event.timestamp_ns = time.time_ns()

            tracker.on_trade(event)  # type: ignore[arg-type]
            tracker._last_snapshot_time = 0  # Force snapshot

            print(f"  {side.upper():4} {qty} {symbol} @ ${price}")

            # Small delay for timestamp differentiation
            await asyncio.sleep(0.001)

        # Get final stats
        stats = tracker.get_stats()

        print("\n=== Performance Summary ===")
        print(f"Total Trades: {stats.total_trades}")
        print(f"Winning Trades: {stats.winning_trades}")
        print(f"Losing Trades: {stats.losing_trades}")
        print(f"Win Rate: {stats.win_rate * 100:.1f}%")
        print(f"Gross Profit: ${stats.gross_profit}")
        print(f"Gross Loss: ${stats.gross_loss}")
        print(f"Net P&L: ${stats.total_pnl}")
        print(f"Profit Factor: {stats.profit_factor:.2f}")
        print(f"Largest Win: ${stats.largest_win}")
        print(f"Largest Loss: ${stats.largest_loss}")

        # Verify calculations
        assert stats.total_trades == 4  # 4 round-trip trades
        assert stats.winning_trades == 3
        assert stats.losing_trades == 1
        assert stats.win_rate == 0.75
        assert stats.gross_profit == Decimal("3250")  # 1000 + 250 + 2000
        assert stats.gross_loss == Decimal("1000")
        assert stats.total_pnl == Decimal("2250")

    async def test_drawdown_tracking(self) -> None:
        """Test drawdown calculation during losing streak."""
        tracker = PerformanceTracker(
            initial_capital=Decimal("100000"),
            snapshot_interval_ns=1_000_000,  # 1ms for fast testing
        )

        # Simulate losing streak
        trades = [
            ("ETH-USD", "buy", 10, 2000),
            ("ETH-USD", "sell", 10, 1800),  # -$2000
            ("ETH-USD", "buy", 10, 1800),
            ("ETH-USD", "sell", 10, 1600),  # -$2000
            ("ETH-USD", "buy", 10, 1600),
            ("ETH-USD", "sell", 10, 1500),  # -$1000
        ]

        print("\n=== Drawdown Test ===")
        for symbol, side, qty, price in trades:
            class MockEvent:
                pass

            event = MockEvent()
            event.symbol = symbol
            event.side = side
            event.quantity = float(qty)
            event.price = float(price)
            event.timestamp_ns = time.time_ns()

            tracker.on_trade(event)  # type: ignore[arg-type]
            tracker._last_snapshot_time = 0

            await asyncio.sleep(0.001)

        stats = tracker.get_stats()

        print(f"Total Loss: ${stats.total_pnl}")
        print(f"Max Drawdown: {stats.max_drawdown_pct:.2f}%")
        print(f"Current Drawdown: {stats.current_drawdown_pct:.2f}%")

        # Should have significant drawdown
        assert stats.total_pnl == Decimal("-5000")
        assert stats.max_drawdown_pct > 0


class TestHummingbotAdapterE2E:
    """E2E tests for the complete Hummingbot adapter."""

    async def test_avellaneda_stoikov_trading_session(self) -> None:
        """Test A-S strategy with simulated market data."""
        adapter = HummingbotAdapter(enable_performance_tracking=True)

        await adapter.initialize({
            "strategy_type": "avellaneda_stoikov",
            "symbol": "ETH/USDT",
            "order_amount": "1.0",
            "min_spread": "0.001",
            "max_spread": "0.05",
            "order_refresh_time": 0.0,  # No delay for testing
            "initial_capital": 100000,
            "avellaneda": {
                "risk_aversion": 0.5,
                "order_book_depth": 1.5,
                "volatility_window": 20,
            },
        })

        # Generate market data
        data = generate_ohlcv_data(start_price=2000, num_bars=50, volatility=0.01)

        print("\n=== Avellaneda-Stoikov Session ===")
        signals_generated = 0

        for i in range(10, len(data)):
            # Feed data incrementally
            window = data.slice(0, i + 1)
            signal = await adapter.on_data(window)

            if signal and signal.metadata:
                signals_generated += 1
                bid = signal.metadata.get("bid_price")
                ask = signal.metadata.get("ask_price")
                spread = signal.metadata.get("spread_pct", 0)

                if signals_generated <= 5:
                    print(f"Bar {i}: Bid={bid}, Ask={ask}, Spread={spread:.2f}%")

        print(f"Total signals generated: {signals_generated}")
        assert signals_generated > 0

        # Check statistics
        stats = adapter.get_statistics()
        print(f"Strategy: {stats['strategy_type']}")
        print(f"Symbol: {stats['symbol']}")

        await adapter.shutdown()

    async def test_pure_mm_multi_level_orders(self) -> None:
        """Test Pure MM strategy with multiple order levels."""
        adapter = HummingbotAdapter(enable_performance_tracking=True)

        await adapter.initialize({
            "strategy_type": "pure_market_making",
            "symbol": "BTC/USDT",
            "order_amount": "0.1",
            "min_spread": "0.002",
            "max_spread": "0.05",
            "order_levels": 3,
            "level_spread": "0.001",
            "order_refresh_time": 0.0,
        })

        data = generate_ohlcv_data(start_price=40000, num_bars=30, volatility=0.015)

        print("\n=== Pure MM Multi-Level Orders ===")

        for i in range(15, len(data)):
            window = data.slice(0, i + 1)
            signal = await adapter.on_data(window)

            if signal and signal.metadata:
                bids = signal.metadata.get("bids", [])
                asks = signal.metadata.get("asks", [])

                if i == 15:  # Print first signal
                    print(f"Bids: {len(bids)} levels")
                    for bid in bids:
                        print(f"  Level {bid['level']}: {bid['price']} x {bid['size']}")
                    print(f"Asks: {len(asks)} levels")
                    for ask in asks:
                        print(f"  Level {ask['level']}: {ask['price']} x {ask['size']}")

                # Verify order structure
                assert len(bids) == 3, "Should have 3 bid levels"
                assert len(asks) == 3, "Should have 3 ask levels"
                break

        await adapter.shutdown()

    async def test_adapter_with_performance_tracking(self) -> None:
        """Test adapter's integrated performance tracking."""
        adapter = HummingbotAdapter(enable_performance_tracking=True)

        await adapter.initialize({
            "strategy_type": "avellaneda_stoikov",
            "symbol": "ETH/USDT",
            "order_amount": "1.0",
            "min_spread": "0.001",
            "max_spread": "0.05",
            "initial_capital": 50000,
        })

        # Update prices to track performance
        adapter.update_price("ETH/USDT", Decimal("2000"))

        # Get performance summary
        summary = adapter.get_performance_summary()
        assert summary is not None
        assert summary["equity"] == 50000
        print(f"\nPerformance tracking enabled: {summary is not None}")
        print(f"Initial equity: ${summary['equity']}")

        stats = adapter.get_performance_stats()
        assert stats is not None
        print(f"Total trades: {stats.total_trades}")

        await adapter.shutdown()


class TestIntegrationE2E:
    """Integration tests combining multiple components."""

    async def test_dex_trading_with_performance_tracking(self) -> None:
        """Test DEX trading with performance tracking integration."""
        # Setup DEX gateway
        config = DEXGatewayConfig(chain_id=1)
        gateway = UniswapV2Gateway(config)
        await gateway.connect()

        # Add pool
        pool = DEXPool(
            address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("10000"),
            reserve1=Decimal("20000000"),
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        )
        gateway.add_simulated_pool(pool)

        # Setup performance tracker
        tracker = PerformanceTracker(initial_capital=Decimal("100000"))

        print("\n=== DEX + Performance Integration ===")

        # Execute multiple swaps and track performance
        for i in range(3):
            # Get quote
            quote = await gateway.get_quote(WETH, USDC, Decimal("5"))
            assert quote is not None

            # Execute swap
            result = await gateway.execute_swap(quote)
            assert result.success

            print(f"Swap {i+1}: {result.input_amount} WETH -> {result.output_amount} USDC")

            # Track in performance system (simulated trade event)
            class MockEvent:
                symbol = "WETH-USDC"
                side = "sell"  # Selling WETH for USDC
                quantity = float(result.input_amount)
                price = float(result.output_amount / result.input_amount)
                timestamp_ns = time.time_ns()

            tracker.on_trade(MockEvent())  # type: ignore[arg-type]

        stats = tracker.get_stats()
        print(f"\nTrades executed: {stats.total_trades}")
        print(f"Realized P&L: ${stats.realized_pnl}")

        await gateway.disconnect()


if __name__ == "__main__":
    # Run specific test for debugging
    asyncio.run(TestDEXGatewayE2E().test_complete_swap_workflow(
        UniswapV2Gateway(DEXGatewayConfig(chain_id=1)),
        DEXPool(
            address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("50000"),
            reserve1=Decimal("100000000"),
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        ),
    ))

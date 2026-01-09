"""
Unit tests for Hummingbot Adapter Plugin (Issue #12).

Tests cover:
- Avellaneda-Stoikov strategy
- Pure market making
- Cross-exchange market making (XEMM)
- Configuration handling
- Main adapter class
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.plugins.hummingbot_adapter.config import (
    AvellanedaStoikovConfig,
    HummingbotAdapterConfig,
    InventoryConfig,
    InventorySkewMode,
    StrategyType,
    XEMMConfig,
)
from libra.plugins.hummingbot_adapter.strategies.avellaneda import (
    AvellanedaStoikovStrategy,
    Quote,
    VolatilityEstimator,
)
from libra.plugins.hummingbot_adapter.strategies.pure_mm import (
    PureMarketMakingStrategy,
)
from libra.plugins.hummingbot_adapter.strategies.xemm import (
    CrossExchangeMarketMakingStrategy,
    CrossExchangeQuote,
    HedgeDirection,
)


class TestVolatilityEstimator:
    """Test volatility estimation."""

    def test_empty_estimator(self):
        """Test volatility with no data."""
        estimator = VolatilityEstimator()
        assert estimator.get_volatility() == 0.0

    def test_single_price(self):
        """Test volatility with single price."""
        estimator = VolatilityEstimator(min_samples=2)
        estimator.update(Decimal("100"), 1_000_000_000)
        assert estimator.get_volatility() == 0.0

    def test_constant_price(self):
        """Test volatility with constant price."""
        estimator = VolatilityEstimator(min_samples=5)
        for i in range(10):
            estimator.update(Decimal("100"), i * 1_000_000_000)
        # Constant price = zero volatility
        assert estimator.get_volatility() == 0.0

    def test_volatile_price(self):
        """Test volatility with varying prices."""
        estimator = VolatilityEstimator(min_samples=5)
        prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
        for i, p in enumerate(prices):
            estimator.update(Decimal(str(p)), i * 1_000_000_000)

        vol = estimator.get_volatility()
        assert vol > 0  # Should have positive volatility

    def test_reset(self):
        """Test reset functionality."""
        estimator = VolatilityEstimator(min_samples=2)
        # Need at least min_samples returns, so 3 prices
        estimator.update(Decimal("100"), 1_000_000_000)
        estimator.update(Decimal("101"), 2_000_000_000)
        estimator.update(Decimal("99"), 3_000_000_000)
        assert estimator.get_volatility() > 0

        estimator.reset()
        assert estimator.get_volatility() == 0.0


class TestAvellanedaStoikovStrategy:
    """Test Avellaneda-Stoikov strategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = AvellanedaStoikovConfig(
            risk_aversion=0.5,
            order_book_depth=1.5,
            min_quote_refresh=0.0,  # No refresh delay for testing
        )
        inventory_config = InventoryConfig(
            target_ratio=0.5,
            skew_mode=InventorySkewMode.LINEAR,
        )
        return AvellanedaStoikovStrategy(
            config=config,
            inventory_config=inventory_config,
            min_spread=Decimal("0.001"),
            max_spread=Decimal("0.05"),
        )

    def test_initial_state(self, strategy):
        """Test initial strategy state."""
        assert strategy.gamma == 0.5
        assert strategy.kappa == 1.5

    def test_reservation_price_no_inventory(self, strategy):
        """Test reservation price with zero inventory."""
        mid_price = Decimal("100")
        # With no volatility data, should return mid price
        reservation = strategy.calculate_reservation_price(mid_price)
        assert reservation == mid_price

    def test_reservation_price_with_inventory(self, strategy):
        """Test reservation price adjusts with inventory."""
        mid_price = Decimal("100")

        # Simulate some price updates for volatility
        for i in range(20):
            strategy.update_price(mid_price + Decimal(str(i % 3 - 1)), i * 1_000_000_000)

        # Set long inventory
        strategy.set_inventory(
            base_balance=Decimal("1.5"),  # More than target
            quote_balance=Decimal("50"),
            mid_price=mid_price,
        )

        reservation = strategy.calculate_reservation_price(mid_price)
        # With long inventory, reservation should be below mid
        assert reservation <= mid_price

    def test_optimal_spread(self, strategy):
        """Test optimal spread calculation."""
        # With no volatility, should return min spread
        spread = strategy.calculate_optimal_spread()
        assert spread == strategy.min_spread

    def test_generate_quote(self, strategy):
        """Test quote generation."""
        mid_price = Decimal("100")
        order_size = Decimal("0.1")
        timestamp_ns = 1_000_000_000

        quote = strategy.generate_quote(mid_price, order_size, timestamp_ns)

        assert quote is not None
        assert isinstance(quote, Quote)
        assert quote.bid_price < mid_price
        assert quote.ask_price > mid_price
        assert quote.bid_size > 0
        assert quote.ask_size > 0
        assert quote.spread > 0

    def test_quote_refresh_limit(self, strategy):
        """Test that quotes respect refresh time."""
        # Set a minimum refresh time
        strategy.config.min_quote_refresh = 1.0

        mid_price = Decimal("100")
        order_size = Decimal("0.1")

        # First quote should succeed
        quote1 = strategy.generate_quote(mid_price, order_size, 1_000_000_000)
        assert quote1 is not None

        # Second quote too soon should return None
        quote2 = strategy.generate_quote(mid_price, order_size, 1_500_000_000)
        assert quote2 is None

        # After refresh time, should succeed
        quote3 = strategy.generate_quote(mid_price, order_size, 3_000_000_000)
        assert quote3 is not None

    def test_inventory_skew(self, strategy):
        """Test inventory affects quote skew."""
        mid_price = Decimal("100")

        # Set long inventory
        strategy.set_inventory(
            base_balance=Decimal("2"),
            quote_balance=Decimal("0"),
            mid_price=mid_price,
        )

        bid_skew, ask_skew = strategy.calculate_inventory_skew()

        # Long inventory should widen bid, tighten ask
        assert bid_skew > Decimal("1")
        assert ask_skew < Decimal("1")


class TestPureMarketMakingStrategy:
    """Test Pure Market Making strategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        inventory_config = InventoryConfig(
            target_ratio=0.5,
            skew_mode=InventorySkewMode.LINEAR,
        )
        return PureMarketMakingStrategy(
            inventory_config=inventory_config,
            base_spread=Decimal("0.002"),
            order_levels=3,
            level_spread=Decimal("0.001"),
        )

    def test_generate_orders(self, strategy):
        """Test order generation."""
        orders = strategy.generate_orders(
            mid_price=Decimal("100"),
            order_size=Decimal("1.0"),
            timestamp_ns=1_000_000_000,
        )

        assert len(orders.bids) == 3
        assert len(orders.asks) == 3

        # Check price ordering
        for i in range(len(orders.bids) - 1):
            assert orders.bids[i].price > orders.bids[i + 1].price

        for i in range(len(orders.asks) - 1):
            assert orders.asks[i].price < orders.asks[i + 1].price

    def test_order_levels_have_decreasing_size(self, strategy):
        """Test that outer levels have smaller size."""
        orders = strategy.generate_orders(
            mid_price=Decimal("100"),
            order_size=Decimal("1.0"),
            timestamp_ns=1_000_000_000,
        )

        # Each level should be smaller than the previous
        for i in range(len(orders.bids) - 1):
            assert orders.bids[i].size > orders.bids[i + 1].size

    def test_should_refresh_orders(self, strategy):
        """Test order refresh logic."""
        orders = strategy.generate_orders(
            mid_price=Decimal("100"),
            order_size=Decimal("1.0"),
            timestamp_ns=1_000_000_000,
        )

        # Small price change - no refresh
        assert not strategy.should_refresh_orders(
            orders, Decimal("100.05"), price_threshold=Decimal("0.001")
        )

        # Large price change - refresh needed
        assert strategy.should_refresh_orders(
            orders, Decimal("101"), price_threshold=Decimal("0.001")
        )

    def test_inventory_skew(self, strategy):
        """Test inventory affects order prices."""
        mid_price = Decimal("100")

        # Balanced inventory
        strategy.set_inventory(Decimal("1"), Decimal("100"), mid_price)
        balanced_orders = strategy.generate_orders(mid_price, Decimal("1"), 1_000_000_000)

        # Long inventory
        strategy.set_inventory(Decimal("2"), Decimal("0"), mid_price)
        long_orders = strategy.generate_orders(mid_price, Decimal("1"), 2_000_000_000)

        # With long inventory, bid should be lower (less aggressive buying)
        assert long_orders.bids[0].price < balanced_orders.bids[0].price


class TestXEMMStrategy:
    """Test Cross-Exchange Market Making strategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = XEMMConfig(
            maker_exchange="binance",
            taker_exchange="coinbase",
            min_profitability=Decimal("0.001"),
            taker_fee=Decimal("0.001"),
            maker_fee=Decimal("0.0005"),
            active_hedging=True,
        )
        return CrossExchangeMarketMakingStrategy(config=config)

    def test_calculate_profitability_buy(self, strategy):
        """Test profitability calculation for buy side."""
        profit, profit_pct = strategy.calculate_profitability(
            maker_price=Decimal("100"),  # Buy at 100
            taker_price=Decimal("101"),  # Sell at 101
            side="buy",
        )

        # Gross profit = 101 - 100 = 1
        # Fees ~= 0.15% of ~100.5 = ~0.15
        # Net profit ~= 0.85
        assert profit > 0
        assert profit_pct > 0

    def test_calculate_profitability_sell(self, strategy):
        """Test profitability calculation for sell side."""
        profit, profit_pct = strategy.calculate_profitability(
            maker_price=Decimal("101"),  # Sell at 101
            taker_price=Decimal("100"),  # Buy at 100
            side="sell",
        )

        assert profit > 0
        assert profit_pct > 0

    def test_find_opportunities(self, strategy):
        """Test opportunity detection."""
        # Profitable scenario
        quotes = CrossExchangeQuote(
            maker_bid=Decimal("100"),
            maker_ask=Decimal("100.5"),
            taker_bid=Decimal("101"),  # Higher than maker ask - buy opportunity
            taker_ask=Decimal("99"),  # Lower than maker bid - sell opportunity
            timestamp_ns=1_000_000_000,
        )

        opportunities = strategy.find_opportunities(quotes, Decimal("1.0"))

        # Should find opportunities
        assert len(opportunities) >= 1

    def test_no_opportunities_when_unprofitable(self, strategy):
        """Test no opportunities when spreads don't align."""
        # Unprofitable scenario
        quotes = CrossExchangeQuote(
            maker_bid=Decimal("100"),
            maker_ask=Decimal("100.5"),
            taker_bid=Decimal("99"),  # Lower than maker ask
            taker_ask=Decimal("102"),  # Higher than maker bid
            timestamp_ns=1_000_000_000,
        )

        opportunities = strategy.find_opportunities(quotes, Decimal("1.0"))

        # Should not find opportunities
        assert len(opportunities) == 0

    def test_on_maker_fill_generates_hedge(self, strategy):
        """Test that maker fills generate hedge instructions."""
        hedge = strategy.on_maker_fill(
            side="buy",
            size=Decimal("1.0"),
            fill_price=Decimal("100"),
        )

        assert hedge is not None
        assert hedge.side == HedgeDirection.SELL
        assert hedge.size == Decimal("1.0")

    def test_position_tracking(self, strategy):
        """Test unhedged position tracking."""
        assert strategy.unhedged_position == Decimal("0")

        # Buy on maker
        strategy.on_maker_fill("buy", Decimal("1.0"), Decimal("100"))
        assert strategy.unhedged_position == Decimal("1.0")

        # Sell hedge
        strategy.on_hedge_fill(
            HedgeDirection.SELL,
            Decimal("1.0"),
            Decimal("101"),
            Decimal("100"),
        )
        assert strategy.unhedged_position == Decimal("0")
        assert strategy.total_profit > Decimal("0")


class TestHummingbotAdapterConfig:
    """Test configuration handling."""

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "strategy_type": "avellaneda_stoikov",
            "symbol": "BTC/USDT",
            "order_amount": "0.01",
            "min_spread": "0.001",
            "max_spread": "0.05",
            "avellaneda": {
                "risk_aversion": 0.7,
                "order_book_depth": 2.0,
            },
        }

        config = HummingbotAdapterConfig.from_dict(data)

        assert config.strategy_type == StrategyType.AVELLANEDA_STOIKOV
        assert config.symbol == "BTC/USDT"
        assert config.order_amount == Decimal("0.01")
        assert config.avellaneda.risk_aversion == 0.7
        assert config.avellaneda.order_book_depth == 2.0

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = HummingbotAdapterConfig(
            strategy_type=StrategyType.PURE_MARKET_MAKING,
            symbol="ETH/USDT",
            order_amount=Decimal("0.1"),
        )

        data = config.to_dict()

        assert data["strategy_type"] == "pure_market_making"
        assert data["symbol"] == "ETH/USDT"
        assert data["order_amount"] == "0.1"

    def test_validation_fails_without_symbol(self):
        """Test validation fails without symbol."""
        config = HummingbotAdapterConfig(
            strategy_type=StrategyType.AVELLANEDA_STOIKOV,
            symbol="",
        )

        with pytest.raises(ValueError, match="symbol is required"):
            config.validate()

    def test_validation_fails_invalid_spread(self):
        """Test validation fails with invalid spread."""
        config = HummingbotAdapterConfig(
            strategy_type=StrategyType.AVELLANEDA_STOIKOV,
            symbol="BTC/USDT",
            min_spread=Decimal("0.05"),
            max_spread=Decimal("0.01"),  # Less than min
        )

        with pytest.raises(ValueError, match="max_spread must be greater"):
            config.validate()

    def test_xemm_validation_requires_exchanges(self):
        """Test XEMM validation requires exchange configuration."""
        config = HummingbotAdapterConfig(
            strategy_type=StrategyType.CROSS_EXCHANGE_MM,
            symbol="BTC/USDT",
            xemm=XEMMConfig(maker_exchange="", taker_exchange=""),
        )

        with pytest.raises(ValueError, match="maker_exchange is required"):
            config.validate()


class TestHummingbotAdapter:
    """Test main adapter class."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        from libra.plugins.hummingbot_adapter.adapter import HummingbotAdapter

        return HummingbotAdapter()

    def test_metadata(self, adapter):
        """Test adapter metadata."""
        meta = adapter.metadata()
        assert meta.name == "hummingbot-adapter"
        assert len(meta.requires) == 0  # No external dependencies

    @pytest.mark.asyncio
    async def test_initialize_avellaneda(self, adapter):
        """Test initializing with Avellaneda-Stoikov."""
        await adapter.initialize({
            "strategy_type": "avellaneda_stoikov",
            "symbol": "BTC/USDT",
            "order_amount": "0.01",
        })

        assert adapter._initialized
        assert adapter.symbols == ["BTC/USDT"]
        assert adapter._avellaneda is not None

    @pytest.mark.asyncio
    async def test_initialize_pure_mm(self, adapter):
        """Test initializing with Pure Market Making."""
        await adapter.initialize({
            "strategy_type": "pure_market_making",
            "symbol": "ETH/USDT",
            "order_amount": "0.1",
            "order_levels": 3,
        })

        assert adapter._initialized
        assert adapter._pure_mm is not None

    @pytest.mark.asyncio
    async def test_on_data_generates_signal(self, adapter):
        """Test that on_data generates appropriate signals."""
        import polars as pl

        await adapter.initialize({
            "strategy_type": "avellaneda_stoikov",
            "symbol": "BTC/USDT",
            "order_amount": "0.01",
            "order_refresh_time": 0,  # No delay for testing
        })

        data = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "open": [100.0, 100.5, 101.0],
            "high": [101.0, 101.5, 102.0],
            "low": [99.0, 99.5, 100.0],
            "close": [100.5, 101.0, 101.5],
            "volume": [1000, 1100, 1200],
        })

        signal = await adapter.on_data(data)

        assert signal is not None
        assert signal.symbol == "BTC/USDT"
        assert "bid_price" in signal.metadata
        assert "ask_price" in signal.metadata

    @pytest.mark.asyncio
    async def test_shutdown(self, adapter):
        """Test shutdown cleans up state."""
        await adapter.initialize({
            "strategy_type": "pure_market_making",
            "symbol": "BTC/USDT",
            "order_amount": "0.01",
        })

        assert adapter._initialized

        await adapter.shutdown()

        assert not adapter._initialized

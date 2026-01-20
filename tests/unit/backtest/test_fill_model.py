"""
Tests for FillModel (Issue #107).

Tests realistic order fill simulation with slippage, partial fills,
and queue position modeling.
"""

from decimal import Decimal

import pytest

from libra.backtest.fill_model import (
    FillModel,
    FillModelType,
    FillResult,
    SlippageType,
    create_conservative_model,
    create_immediate_model,
    create_order_book_model,
    create_realistic_model,
)
from libra.gateways.protocol import OrderBook, OrderSide


class TestFillResult:
    """Tests for FillResult dataclass."""

    def test_no_fill_factory(self) -> None:
        """Test no-fill factory method."""
        result = FillResult.no_fill("price_not_reached")

        assert result.filled is False
        assert result.fill_price is None
        assert result.fill_quantity == Decimal("0")
        assert result.reason == "price_not_reached"

    def test_full_fill_factory(self) -> None:
        """Test full-fill factory method."""
        result = FillResult.full_fill(
            price=Decimal("50000"),
            quantity=Decimal("1.5"),
            slippage_ticks=2,
        )

        assert result.filled is True
        assert result.fill_price == Decimal("50000")
        assert result.fill_quantity == Decimal("1.5")
        assert result.slippage_ticks == 2
        assert result.is_partial is False


class TestFillModelValidation:
    """Tests for FillModel validation."""

    def test_valid_default_config(self) -> None:
        """Test default configuration is valid."""
        model = FillModel()
        assert model.model_type == FillModelType.PROBABILISTIC

    def test_invalid_prob_slippage(self) -> None:
        """Test invalid slippage probability raises error."""
        with pytest.raises(ValueError, match="prob_slippage must be between"):
            FillModel(prob_slippage=1.5)

    def test_invalid_prob_fill_on_limit(self) -> None:
        """Test invalid fill probability raises error."""
        with pytest.raises(ValueError, match="prob_fill_on_limit must be between"):
            FillModel(prob_fill_on_limit=-0.1)

    def test_invalid_queue_position(self) -> None:
        """Test invalid queue position raises error."""
        with pytest.raises(ValueError, match="queue_position_pct must be between"):
            FillModel(queue_position_pct=1.5)

    def test_invalid_min_fill_pct(self) -> None:
        """Test invalid min fill percentage raises error."""
        with pytest.raises(ValueError, match="min_fill_pct must be between"):
            FillModel(min_fill_pct=0)

    def test_invalid_slippage_ticks(self) -> None:
        """Test invalid slippage ticks raises error."""
        with pytest.raises(ValueError, match="slippage_ticks must be non-negative"):
            FillModel(slippage_ticks=-1)

    def test_invalid_max_slippage(self) -> None:
        """Test max slippage less than slippage raises error."""
        with pytest.raises(ValueError, match="max_slippage_ticks must be >= slippage_ticks"):
            FillModel(slippage_ticks=5, max_slippage_ticks=2)


class TestImmediateFillModel:
    """Tests for immediate fill model."""

    @pytest.fixture
    def model(self) -> FillModel:
        """Create immediate fill model."""
        return create_immediate_model()

    def test_market_order_no_slippage(self, model: FillModel) -> None:
        """Test market order fills at exact price."""
        result = model.simulate_market_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            current_price=Decimal("50000"),
        )

        assert result.filled is True
        assert result.fill_price == Decimal("50000")
        assert result.fill_quantity == Decimal("1.0")
        assert result.slippage_ticks == 0

    def test_limit_order_immediate_fill(self, model: FillModel) -> None:
        """Test limit order fills immediately at favorable price."""
        result = model.simulate_limit_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            limit_price=Decimal("50000"),
            current_price=Decimal("49900"),
        )

        assert result.filled is True
        assert result.fill_price == Decimal("50000")
        assert result.fill_quantity == Decimal("1.0")

    def test_limit_order_no_fill_unfavorable(self, model: FillModel) -> None:
        """Test limit order doesn't fill at unfavorable price."""
        result = model.simulate_limit_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            limit_price=Decimal("50000"),
            current_price=Decimal("50100"),
        )

        assert result.filled is False
        assert result.reason == "price_not_reached"


class TestProbabilisticFillModel:
    """Tests for probabilistic fill model."""

    @pytest.fixture
    def model(self) -> FillModel:
        """Create probabilistic model with fixed seed."""
        return create_realistic_model().with_seed(42)

    def test_market_order_with_slippage(self, model: FillModel) -> None:
        """Test market order experiences slippage."""
        results = []
        for seed in range(100):
            m = model.with_seed(seed)
            result = m.simulate_market_fill(
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                current_price=Decimal("50000"),
            )
            results.append(result)

        # All should fill
        assert all(r.filled for r in results)

        # Some should have slippage (probability is 0.3)
        with_slippage = sum(1 for r in results if r.slippage_ticks > 0)
        assert with_slippage > 10  # At least some slippage
        assert with_slippage < 50  # But not all

    def test_limit_order_probabilistic_fill(self) -> None:
        """Test limit order fill is probabilistic."""
        fills = 0
        no_fills = 0

        for seed in range(100):
            model = FillModel(
                model_type=FillModelType.PROBABILISTIC,
                prob_fill_on_limit=0.5,
                queue_position_pct=0.3,
                random_seed=seed,
            )

            result = model.simulate_limit_fill(
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                limit_price=Decimal("50000"),
                current_price=Decimal("49900"),
                time_at_price_pct=0.5,
            )

            if result.filled:
                fills += 1
            else:
                no_fills += 1

        # Should have mix of fills and no-fills
        assert fills > 20
        assert no_fills > 20

    def test_queue_position_affects_fill(self) -> None:
        """Test queue position affects fill probability."""
        front_fills = 0
        back_fills = 0

        for seed in range(100):
            # Front of queue
            front_model = FillModel(
                model_type=FillModelType.PROBABILISTIC,
                prob_fill_on_limit=0.7,
                queue_position_pct=0.1,  # Front
                queue_position_std=0.05,
                random_seed=seed,
            )
            result = front_model.simulate_limit_fill(
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                limit_price=Decimal("50000"),
                current_price=Decimal("49900"),
            )
            if result.filled:
                front_fills += 1

            # Back of queue
            back_model = FillModel(
                model_type=FillModelType.PROBABILISTIC,
                prob_fill_on_limit=0.7,
                queue_position_pct=0.9,  # Back
                queue_position_std=0.05,
                random_seed=seed,
            )
            result = back_model.simulate_limit_fill(
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                limit_price=Decimal("50000"),
                current_price=Decimal("49900"),
            )
            if result.filled:
                back_fills += 1

        # Front of queue should fill more often
        assert front_fills > back_fills


class TestStopOrderFills:
    """Tests for stop order simulation."""

    @pytest.fixture
    def model(self) -> FillModel:
        """Create model with fixed seed."""
        return create_realistic_model().with_seed(42)

    def test_sell_stop_triggers_on_price_drop(self, model: FillModel) -> None:
        """Test sell stop triggers when price falls."""
        result = model.simulate_stop_fill(
            side=OrderSide.SELL,
            quantity=Decimal("1.0"),
            stop_price=Decimal("48000"),
            current_price=Decimal("47500"),
        )

        assert result.filled is True
        assert result.fill_price is not None

    def test_sell_stop_no_trigger_above_price(self, model: FillModel) -> None:
        """Test sell stop doesn't trigger when price is above."""
        result = model.simulate_stop_fill(
            side=OrderSide.SELL,
            quantity=Decimal("1.0"),
            stop_price=Decimal("48000"),
            current_price=Decimal("49000"),
        )

        assert result.filled is False
        assert result.reason == "stop_not_triggered"

    def test_buy_stop_triggers_on_price_rise(self, model: FillModel) -> None:
        """Test buy stop triggers when price rises."""
        result = model.simulate_stop_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            stop_price=Decimal("52000"),
            current_price=Decimal("52500"),
        )

        assert result.filled is True

    def test_stop_with_bar_high_low(self, model: FillModel) -> None:
        """Test stop uses bar high/low for trigger check."""
        # Price didn't touch stop but bar high did
        result = model.simulate_stop_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            stop_price=Decimal("52000"),
            current_price=Decimal("51000"),
            bar_high=Decimal("52500"),
            bar_low=Decimal("50500"),
        )

        assert result.filled is True


class TestOrderBookFills:
    """Tests for order book-based fills."""

    @pytest.fixture
    def model(self) -> FillModel:
        """Create order book fill model."""
        return create_order_book_model(tick_size=Decimal("0.01"))

    @pytest.fixture
    def sample_book(self) -> OrderBook:
        """Create sample order book."""
        return OrderBook(
            symbol="BTC/USDT",
            bids=[
                (Decimal("49990"), Decimal("1.0")),
                (Decimal("49980"), Decimal("2.0")),
                (Decimal("49970"), Decimal("3.0")),
            ],
            asks=[
                (Decimal("50010"), Decimal("1.0")),
                (Decimal("50020"), Decimal("2.0")),
                (Decimal("50030"), Decimal("3.0")),
            ],
            timestamp_ns=0,
        )

    def test_small_order_fills_at_best(self, model: FillModel, sample_book: OrderBook) -> None:
        """Test small order fills at best price."""
        result = model.simulate_order_book_fill(
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_book=sample_book,
        )

        assert result.filled is True
        assert result.fill_price == Decimal("50010")  # Best ask
        assert result.fill_quantity == Decimal("0.5")
        assert result.slippage_ticks == 0

    def test_large_order_walks_book(self, model: FillModel, sample_book: OrderBook) -> None:
        """Test large order walks through price levels."""
        result = model.simulate_order_book_fill(
            side=OrderSide.BUY,
            quantity=Decimal("2.5"),
            order_book=sample_book,
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("2.5")
        # VWAP should be between first and second level
        assert result.fill_price > Decimal("50010")
        assert result.fill_price < Decimal("50020")
        assert result.slippage_ticks > 0  # Has slippage from book walk

    def test_order_exceeds_liquidity(self, model: FillModel, sample_book: OrderBook) -> None:
        """Test order larger than available liquidity."""
        result = model.simulate_order_book_fill(
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),  # Only 6 available
            order_book=sample_book,
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("6.0")  # All available
        assert result.is_partial is True

    def test_empty_book_no_fill(self, model: FillModel) -> None:
        """Test empty order book returns no fill."""
        empty_book = OrderBook(symbol="BTC/USDT", bids=[], asks=[], timestamp_ns=0)

        result = model.simulate_order_book_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_book=empty_book,
        )

        assert result.filled is False
        assert result.reason == "empty_order_book"


class TestPartialFills:
    """Tests for partial fill behavior."""

    def test_partial_fill_with_volume_limit(self) -> None:
        """Test partial fill when volume exceeds limit."""
        model = FillModel(
            model_type=FillModelType.PROBABILISTIC,
            enable_partial_fills=True,
            max_volume_pct=0.1,  # 10% of volume
            min_fill_pct=0.1,
            random_seed=42,
        )

        result = model.simulate_market_fill(
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # Large order
            current_price=Decimal("50000"),
            bar_volume=Decimal("500"),  # 10% = 50
        )

        assert result.filled is True
        assert result.fill_quantity < Decimal("100")
        assert result.is_partial is True

    def test_partial_fills_disabled(self) -> None:
        """Test all-or-nothing when partial fills disabled."""
        model = FillModel(
            model_type=FillModelType.PROBABILISTIC,
            enable_partial_fills=False,
            random_seed=42,
        )

        result = model.simulate_market_fill(
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            current_price=Decimal("50000"),
            bar_volume=Decimal("500"),
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("100")
        assert result.is_partial is False


class TestSlippageTypes:
    """Tests for different slippage types."""

    def test_no_slippage(self) -> None:
        """Test no slippage model."""
        model = FillModel(
            model_type=FillModelType.PROBABILISTIC,
            slippage_type=SlippageType.NONE,
        )

        result = model.simulate_market_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            current_price=Decimal("50000"),
        )

        assert result.slippage_ticks == 0
        assert result.fill_price == Decimal("50000")

    def test_fixed_slippage(self) -> None:
        """Test fixed slippage model."""
        model = FillModel(
            model_type=FillModelType.PROBABILISTIC,
            slippage_type=SlippageType.FIXED,
            prob_slippage=1.0,  # Always slippage
            slippage_ticks=2,
            tick_size=Decimal("0.01"),
        )

        result = model.simulate_market_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            current_price=Decimal("50000"),
        )

        assert result.slippage_ticks == 2
        assert result.fill_price == Decimal("50000.02")

    def test_volume_slippage(self) -> None:
        """Test volume-based slippage increases with size."""
        model = FillModel(
            model_type=FillModelType.PROBABILISTIC,
            slippage_type=SlippageType.VOLUME,
            prob_slippage=1.0,
            volume_impact=Decimal("0.1"),
            tick_size=Decimal("0.01"),
        )

        # Small order
        small_result = model.simulate_market_fill(
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            current_price=Decimal("50000"),
            bar_volume=Decimal("1000"),
        )

        # Large order
        large_result = model.simulate_market_fill(
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            current_price=Decimal("50000"),
            bar_volume=Decimal("1000"),
        )

        # Large order should have more slippage
        assert large_result.slippage_ticks >= small_result.slippage_ticks


class TestPresetModels:
    """Tests for preset model factories."""

    def test_immediate_model(self) -> None:
        """Test immediate model has no slippage or probabilities."""
        model = create_immediate_model()

        assert model.model_type == FillModelType.IMMEDIATE
        assert model.slippage_type == SlippageType.NONE
        assert model.enable_partial_fills is False

    def test_realistic_model(self) -> None:
        """Test realistic model has balanced settings."""
        model = create_realistic_model(
            slippage_bps=Decimal("10"),
            fill_probability=0.8,
            queue_position=0.3,
        )

        assert model.model_type == FillModelType.PROBABILISTIC
        assert model.slippage_bps == Decimal("10")
        assert model.prob_fill_on_limit == 0.8
        assert model.queue_position_pct == 0.3

    def test_conservative_model(self) -> None:
        """Test conservative model is pessimistic."""
        model = create_conservative_model()

        assert model.model_type == FillModelType.PROBABILISTIC
        assert model.prob_fill_on_limit < 0.5  # Low fill probability
        assert model.queue_position_pct > 0.7  # Back of queue
        assert model.max_volume_pct < 0.05  # Small participation

    def test_order_book_model(self) -> None:
        """Test order book model."""
        model = create_order_book_model(tick_size=Decimal("0.001"))

        assert model.model_type == FillModelType.ORDER_BOOK
        assert model.tick_size == Decimal("0.001")


class TestReproducibility:
    """Tests for random seed reproducibility."""

    def test_same_seed_same_results(self) -> None:
        """Test same seed produces identical results."""
        model1 = create_realistic_model().with_seed(12345)
        model2 = create_realistic_model().with_seed(12345)

        results1 = []
        results2 = []

        for _ in range(10):
            r1 = model1.simulate_market_fill(
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                current_price=Decimal("50000"),
            )
            r2 = model2.simulate_market_fill(
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                current_price=Decimal("50000"),
            )
            results1.append(r1.fill_price)
            results2.append(r2.fill_price)

        assert results1 == results2

    def test_different_seeds_different_results(self) -> None:
        """Test different seeds produce different results."""
        results = set()

        for seed in range(10):
            model = create_realistic_model().with_seed(seed)
            # Run multiple to get some slippage
            for _ in range(5):
                result = model.simulate_market_fill(
                    side=OrderSide.BUY,
                    quantity=Decimal("1.0"),
                    current_price=Decimal("50000"),
                )
                results.add(result.fill_price)

        # Should have some variation
        assert len(results) > 1

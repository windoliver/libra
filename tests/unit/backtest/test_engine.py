"""
Unit tests for BacktestEngine.

Tests:
- Engine initialization
- Configuration validation
- Basic backtest execution
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.backtest.engine import BacktestConfig, BacktestEngine
from libra.clients.backtest_execution_client import SlippageModel
from libra.strategies.base import BaseStrategy
from libra.strategies.protocol import Bar, Signal, SignalType, StrategyConfig


class SimpleMovingAverageStrategy(BaseStrategy):
    """Simple moving average crossover strategy for testing."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        if config is None:
            config = StrategyConfig(symbol="BTC/USDT", timeframe="1h")
        super().__init__(config)
        self._closes: list[Decimal] = []
        self._fast_period = 5
        self._slow_period = 10

    @property
    def name(self) -> str:
        return "simple_sma"

    def on_bar(self, bar: Bar) -> Signal | None:
        """Process bar and generate signal based on SMA crossover."""
        self._closes.append(bar.close)

        if len(self._closes) < self._slow_period:
            return None

        # Calculate SMAs
        fast_sma = sum(self._closes[-self._fast_period:]) / self._fast_period
        slow_sma = sum(self._closes[-self._slow_period:]) / self._slow_period

        # Generate signal on crossover
        if fast_sma > slow_sma:
            return self._long(price=bar.close)

        return None

    def on_reset(self) -> None:
        """Reset strategy state."""
        super().on_reset()
        self._closes.clear()


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = BacktestConfig()

        assert config.initial_capital == Decimal("100000")
        assert config.slippage_model == SlippageModel.FIXED
        assert config.slippage_bps == Decimal("5")
        assert config.maker_fee_rate == Decimal("0.001")
        assert config.taker_fee_rate == Decimal("0.001")

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=Decimal("50000"),
            slippage_model=SlippageModel.NONE,
            slippage_bps=Decimal("0"),
            maker_fee_rate=Decimal("0.0005"),
            taker_fee_rate=Decimal("0.001"),
        )

        assert config.initial_capital == Decimal("50000")
        assert config.slippage_model == SlippageModel.NONE
        assert config.maker_fee_rate == Decimal("0.0005")

    def test_invalid_capital(self) -> None:
        """Test validation of negative capital."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            BacktestConfig(initial_capital=Decimal("-100"))

    def test_invalid_slippage(self) -> None:
        """Test validation of negative slippage."""
        with pytest.raises(ValueError, match="slippage_bps cannot be negative"):
            BacktestConfig(slippage_bps=Decimal("-5"))

    def test_to_dict(self) -> None:
        """Test configuration serialization."""
        config = BacktestConfig(initial_capital=Decimal("100000"))
        d = config.to_dict()

        assert d["initial_capital"] == "100000"
        assert "instance_id" in d


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_initialization(self) -> None:
        """Test engine initialization."""
        config = BacktestConfig(initial_capital=Decimal("100000"))
        engine = BacktestEngine(config)

        assert engine.config.initial_capital == Decimal("100000")
        assert not engine.is_running
        assert engine.strategy is None
        assert engine.symbols == []

    def test_add_strategy(self) -> None:
        """Test adding a strategy."""
        engine = BacktestEngine()
        strategy = SimpleMovingAverageStrategy()

        engine.add_strategy(strategy)

        assert engine.strategy is not None
        assert engine.strategy.name == "simple_sma"

    def test_add_strategy_twice_fails(self) -> None:
        """Test that adding two strategies fails."""
        engine = BacktestEngine()
        strategy1 = SimpleMovingAverageStrategy()
        strategy2 = SimpleMovingAverageStrategy()

        engine.add_strategy(strategy1)

        with pytest.raises(ValueError, match="Strategy already set"):
            engine.add_strategy(strategy2)

    def test_add_bars(self) -> None:
        """Test adding bar data."""
        engine = BacktestEngine()

        # Create sample bars
        bars = [
            Bar(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ns=1704067200_000_000_000 + i * 3600_000_000_000,
                open=Decimal("42000") + Decimal(str(i * 100)),
                high=Decimal("42500") + Decimal(str(i * 100)),
                low=Decimal("41500") + Decimal(str(i * 100)),
                close=Decimal("42200") + Decimal(str(i * 100)),
                volume=Decimal("1000"),
            )
            for i in range(100)
        ]

        engine.add_bars("BTC/USDT", bars)

        assert "BTC/USDT" in engine.symbols

    def test_add_bars_empty_fails(self) -> None:
        """Test that adding empty bars fails."""
        engine = BacktestEngine()

        with pytest.raises(ValueError, match="bars list cannot be empty"):
            engine.add_bars("BTC/USDT", [])

    def test_add_bars_unsorted_fails(self) -> None:
        """Test that adding unsorted bars fails."""
        engine = BacktestEngine()

        # Create bars with wrong order
        bars = [
            Bar(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ns=1704067200_000_000_000 + i * 3600_000_000_000,
                open=Decimal("42000"),
                high=Decimal("42500"),
                low=Decimal("41500"),
                close=Decimal("42200"),
                volume=Decimal("1000"),
            )
            for i in [0, 2, 1]  # Wrong order
        ]

        with pytest.raises(ValueError, match="Bars must be sorted"):
            engine.add_bars("BTC/USDT", bars)

    def test_run_no_strategy_fails(self) -> None:
        """Test that running without strategy fails."""
        engine = BacktestEngine()

        with pytest.raises(ValueError, match="No strategy configured"):
            import asyncio
            asyncio.run(engine.run())

    def test_run_no_data_fails(self) -> None:
        """Test that running without data fails."""
        engine = BacktestEngine()
        engine.add_strategy(SimpleMovingAverageStrategy())

        with pytest.raises(ValueError, match="No bar data configured"):
            import asyncio
            asyncio.run(engine.run())

    def test_repr(self) -> None:
        """Test string representation."""
        engine = BacktestEngine()
        engine.add_strategy(SimpleMovingAverageStrategy())

        repr_str = repr(engine)

        assert "BacktestEngine" in repr_str
        assert "simple_sma" in repr_str


class TestBacktestEngineRun:
    """Integration tests for running backtests."""

    @pytest.fixture
    def sample_bars(self) -> list[Bar]:
        """Create sample bar data."""
        import random
        random.seed(42)  # Reproducible

        bars = []
        price = Decimal("42000")

        for i in range(100):
            # Random walk
            change = Decimal(str(random.uniform(-0.02, 0.02)))
            price *= 1 + change

            bars.append(
                Bar(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    timestamp_ns=1704067200_000_000_000 + i * 3600_000_000_000,
                    open=price,
                    high=price * Decimal("1.01"),
                    low=price * Decimal("0.99"),
                    close=price,
                    volume=Decimal("1000"),
                )
            )

        return bars

    @pytest.mark.asyncio
    async def test_basic_backtest(self, sample_bars: list[Bar]) -> None:
        """Test running a basic backtest."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.NONE,
            verbose=False,
        )
        engine = BacktestEngine(config)

        strategy = SimpleMovingAverageStrategy(
            StrategyConfig(symbol="BTC/USDT", timeframe="1h")
        )
        engine.add_strategy(strategy)
        engine.add_bars("BTC/USDT", sample_bars)

        result = await engine.run()

        # Basic validation
        assert result is not None
        assert result.summary.strategy_name == "simple_sma"
        assert result.summary.symbol == "BTC/USDT"
        assert result.summary.bars_processed == 100
        assert len(result.equity_curve) > 0

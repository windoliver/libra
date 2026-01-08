"""Unit tests for FreqtradeAdapter."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from libra.plugins.base import PluginMetadata
from libra.plugins.freqtrade_adapter.adapter import FreqtradeAdapter
from libra.strategies.protocol import SignalType


class MockStrategy:
    """Mock Freqtrade strategy for testing."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize mock strategy."""
        self.config = config or {}
        self.timeframe = "1h"
        self.stoploss = -0.10
        self.minimal_roi = {"0": 0.1}
        self.startup_candle_count = 30
        self.can_short = True
        self.use_exit_signal = True

    def populate_indicators(
        self,
        dataframe: Any,  # pandas DataFrame from Freqtrade
        metadata: dict[str, Any],
    ) -> Any:
        """Add indicators to dataframe."""
        dataframe["sma_20"] = dataframe["close"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(
        self,
        dataframe: Any,  # pandas DataFrame
        metadata: dict[str, Any],
    ) -> Any:
        """Add entry signals."""
        dataframe.loc[
            dataframe["close"] > dataframe["sma_20"],
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(
        self,
        dataframe: Any,  # pandas DataFrame
        metadata: dict[str, Any],
    ) -> Any:
        """Add exit signals."""
        dataframe.loc[
            dataframe["close"] < dataframe["sma_20"],
            "exit_long",
        ] = 1
        return dataframe


class TestFreqtradeAdapter:
    """Tests for FreqtradeAdapter."""

    @pytest.fixture
    def adapter(self) -> FreqtradeAdapter:
        """Create an adapter instance."""
        return FreqtradeAdapter()

    @pytest.fixture
    def sample_config(self) -> dict[str, Any]:
        """Create sample configuration."""
        return {
            "strategy_name": "TestStrategy",
            "timeframe": "1h",
            "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
            "stake_currency": "USDT",
        }

    def test_metadata(self) -> None:
        """Test plugin metadata."""
        metadata = FreqtradeAdapter.metadata()

        assert isinstance(metadata, PluginMetadata)
        assert metadata.name == "freqtrade-adapter"
        assert "0.1.0" in metadata.version
        assert "Freqtrade" in metadata.description
        assert len(metadata.requires) > 0
        assert any("freqtrade" in req for req in metadata.requires)

    def test_initial_state(self, adapter: FreqtradeAdapter) -> None:
        """Test adapter initial state."""
        assert adapter.is_initialized is False
        assert adapter.strategy is None
        assert adapter.config is None
        assert adapter.symbols == []
        assert "uninitialized" in adapter.name

    @pytest.mark.asyncio
    async def test_initialize_requires_strategy_name(
        self,
        adapter: FreqtradeAdapter,
    ) -> None:
        """Test that initialize requires strategy_name."""
        with pytest.raises(ValueError, match="strategy_name is required"):
            await adapter.initialize({})

    @pytest.mark.asyncio
    async def test_initialize_with_mock_strategy(
        self,
        adapter: FreqtradeAdapter,
        sample_config: dict[str, Any],
    ) -> None:
        """Test initialization with mocked strategy loading."""
        adapter._loader._freqtrade_available = False
        adapter._loader.load_strategy = MagicMock(return_value=MockStrategy)

        await adapter.initialize(sample_config)

        assert adapter.is_initialized is True
        assert adapter.symbols == ["BTC/USDT", "ETH/USDT"]
        assert adapter.config is not None
        assert adapter.config.strategy_name == "TestStrategy"

    @pytest.mark.asyncio
    async def test_on_data_not_initialized(
        self,
        adapter: FreqtradeAdapter,
    ) -> None:
        """Test on_data raises when not initialized."""
        df = pl.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.on_data(df)

    @pytest.mark.asyncio
    async def test_on_data_empty_dataframe(
        self,
        adapter: FreqtradeAdapter,
        sample_config: dict[str, Any],
    ) -> None:
        """Test on_data with empty DataFrame."""
        adapter._loader._freqtrade_available = False
        adapter._loader.load_strategy = MagicMock(return_value=MockStrategy)
        await adapter.initialize(sample_config)

        result = await adapter.on_data(pl.DataFrame())
        assert result is None

    @pytest.mark.asyncio
    async def test_on_data_generates_signal(
        self,
        adapter: FreqtradeAdapter,
        sample_config: dict[str, Any],
    ) -> None:
        """Test on_data generates signal from strategy."""
        # Create test data where last close > sma_20
        data = pl.DataFrame({
            "open": [100.0] * 30,
            "high": [105.0] * 30,
            "low": [95.0] * 30,
            "close": [100.0] * 29 + [110.0],  # Last close is high
            "volume": [1000.0] * 30,
        })

        adapter._loader._freqtrade_available = False
        adapter._loader.load_strategy = MagicMock(return_value=MockStrategy)
        await adapter.initialize(sample_config)

        result = await adapter.on_data(data)

        assert result is not None
        assert result.signal_type == SignalType.LONG
        assert result.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_on_tick_returns_none(
        self,
        adapter: FreqtradeAdapter,
    ) -> None:
        """Test on_tick returns None (Freqtrade is bar-based)."""
        tick = {"price": 50000.0, "volume": 1.5}
        result = await adapter.on_tick(tick)
        assert result is None

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        adapter: FreqtradeAdapter,
        sample_config: dict[str, Any],
    ) -> None:
        """Test shutdown cleans up resources."""
        adapter._loader._freqtrade_available = False
        adapter._loader.load_strategy = MagicMock(return_value=MockStrategy)
        await adapter.initialize(sample_config)

        assert adapter.is_initialized is True

        await adapter.shutdown()

        assert adapter.is_initialized is False
        assert adapter.strategy is None

    @pytest.mark.asyncio
    async def test_backtest_not_initialized(
        self,
        adapter: FreqtradeAdapter,
    ) -> None:
        """Test backtest raises when not initialized."""
        df = pl.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.backtest(df, Decimal("10000"))

    def test_list_strategies(self, adapter: FreqtradeAdapter) -> None:
        """Test list_strategies delegates to loader."""
        with patch.object(
            adapter._loader,
            "list_strategies",
            return_value=["Strategy1", "Strategy2"],
        ) as mock:
            result = adapter.list_strategies(Path("/test"))

        mock.assert_called_once_with(Path("/test"))
        assert result == ["Strategy1", "Strategy2"]

    def test_get_strategy_info(self, adapter: FreqtradeAdapter) -> None:
        """Test get_strategy_info delegates to loader."""
        expected_info = {
            "name": "TestStrategy",
            "timeframe": "1h",
        }

        with patch.object(
            adapter._loader,
            "get_strategy_info",
            return_value=expected_info,
        ) as mock:
            result = adapter.get_strategy_info("TestStrategy", Path("/test"))

        mock.assert_called_once_with("TestStrategy", Path("/test"))
        assert result == expected_info

    @pytest.mark.asyncio
    async def test_get_strategy_parameters(
        self,
        adapter: FreqtradeAdapter,
        sample_config: dict[str, Any],
    ) -> None:
        """Test get_strategy_parameters returns strategy attributes."""
        adapter._loader._freqtrade_available = False
        adapter._loader.load_strategy = MagicMock(return_value=MockStrategy)
        await adapter.initialize(sample_config)

        params = adapter.get_strategy_parameters()

        assert params["timeframe"] == "1h"
        assert params["stoploss"] == -0.10
        assert params["can_short"] is True
        assert params["use_exit_signal"] is True
        assert params["startup_candle_count"] == 30

    def test_get_strategy_parameters_not_initialized(
        self,
        adapter: FreqtradeAdapter,
    ) -> None:
        """Test get_strategy_parameters returns empty dict when not initialized."""
        params = adapter.get_strategy_parameters()
        assert params == {}

    @pytest.mark.asyncio
    async def test_optimize_not_implemented(
        self,
        adapter: FreqtradeAdapter,
        sample_config: dict[str, Any],
    ) -> None:
        """Test optimize raises NotImplementedError."""
        adapter._loader._freqtrade_available = False
        adapter._loader.load_strategy = MagicMock(return_value=MockStrategy)
        await adapter.initialize(sample_config)

        with pytest.raises(NotImplementedError, match="Hyperopt"):
            await adapter.optimize(
                pl.DataFrame(),
                {"param1": [1, 2, 3]},
            )

    @pytest.mark.asyncio
    async def test_on_fill_calls_callback(
        self,
        adapter: FreqtradeAdapter,
        sample_config: dict[str, Any],
    ) -> None:
        """Test on_fill calls strategy callback if available."""
        # Create a mock strategy with order_filled method
        mock_strategy = MagicMock()
        mock_strategy.timeframe = "1h"
        mock_strategy.stoploss = -0.10
        mock_strategy.order_filled = MagicMock()

        adapter._loader._freqtrade_available = False
        adapter._loader.load_strategy = MagicMock(return_value=lambda c: mock_strategy)
        await adapter.initialize(sample_config)
        adapter._strategy = mock_strategy

        order_result = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.1,
            "price": 50000,
        }

        await adapter.on_fill(order_result)

        # Should have attempted to call order_filled
        mock_strategy.order_filled.assert_called_once()

    def test_freqtrade_available_property(self, adapter: FreqtradeAdapter) -> None:
        """Test freqtrade_available property."""
        # Just verify it returns a boolean
        result = adapter.freqtrade_available
        assert isinstance(result, bool)


class TestFreqtradeAdapterIntegration:
    """Integration tests for FreqtradeAdapter."""

    @pytest.fixture
    def adapter(self) -> FreqtradeAdapter:
        """Create an adapter instance."""
        return FreqtradeAdapter()

    @pytest.mark.asyncio
    async def test_full_workflow(self, adapter: FreqtradeAdapter) -> None:
        """Test full workflow: initialize -> on_data -> backtest -> shutdown."""
        config = {
            "strategy_name": "TestStrategy",
            "pair_whitelist": ["BTC/USDT"],
        }

        # Initialize
        adapter._loader._freqtrade_available = False
        adapter._loader.load_strategy = MagicMock(return_value=MockStrategy)
        await adapter.initialize(config)

        assert adapter.is_initialized

        # Create sample data using Polars
        data = pl.DataFrame({
            "open": [100.0] * 50,
            "high": [105.0] * 50,
            "low": [95.0] * 50,
            "close": list(range(100, 150)),
            "volume": [1000.0] * 50,
        })

        # Process data
        _ = await adapter.on_data(data)
        # Signal generation depends on strategy logic

        # Run backtest
        result = await adapter.backtest(data, Decimal("10000"))
        assert result is not None

        # Shutdown
        await adapter.shutdown()
        assert not adapter.is_initialized

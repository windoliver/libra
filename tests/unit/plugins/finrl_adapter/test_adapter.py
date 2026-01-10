"""Tests for FinRL adapter."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.plugins.finrl_adapter.adapter import FinRLAdapter, FinRLNotInstalledError
from libra.plugins.finrl_adapter.config import RLAlgorithm

# Check if numpy is available
try:
    import numpy
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

requires_numpy = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")


class TestFinRLAdapter:
    """Tests for FinRLAdapter class."""

    def test_metadata(self) -> None:
        """Test plugin metadata."""
        metadata = FinRLAdapter.metadata()

        assert metadata.name == "finrl-adapter"
        assert metadata.version == "0.1.0"
        assert "RL" in metadata.description or "FinRL" in metadata.description
        assert len(metadata.requires) > 0

    def test_init(self) -> None:
        """Test adapter initialization."""
        adapter = FinRLAdapter()

        assert adapter.is_initialized is False
        assert adapter.model_loaded is False
        assert adapter.name == "finrl:uninitialized"
        assert adapter.symbols == []

    @requires_numpy
    @pytest.mark.asyncio
    async def test_initialize_basic(self) -> None:
        """Test basic initialization."""
        adapter = FinRLAdapter()

        await adapter.initialize({
            "algorithm": "ppo",
            "pair_whitelist": ["AAPL", "GOOGL"],
            "stock_dim": 2,
        })

        assert adapter.is_initialized is True
        assert adapter.name == "finrl:ppo"
        assert adapter.symbols == ["AAPL", "GOOGL"]
        assert adapter.config is not None
        assert adapter.config.algorithm == RLAlgorithm.PPO

    @requires_numpy
    @pytest.mark.asyncio
    async def test_initialize_with_indicators(self) -> None:
        """Test initialization with custom indicators."""
        adapter = FinRLAdapter()

        await adapter.initialize({
            "algorithm": "sac",
            "tech_indicators": ["macd", "rsi", "boll"],
            "use_turbulence": True,
        })

        assert adapter.config is not None
        assert adapter.config.algorithm == RLAlgorithm.SAC
        assert "macd" in adapter.config.tech_indicators

    @pytest.mark.asyncio
    async def test_on_data_without_init(self) -> None:
        """Test on_data raises error without initialization."""
        adapter = FinRLAdapter()

        import polars as pl
        df = pl.DataFrame({"close": [100.0, 101.0]})

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.on_data(df)

    @requires_numpy
    @pytest.mark.asyncio
    async def test_on_data_without_model(self) -> None:
        """Test on_data returns None without model."""
        adapter = FinRLAdapter()
        await adapter.initialize({"algorithm": "ppo"})

        import polars as pl
        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
        })

        signal = await adapter.on_data(df)
        assert signal is None

    @requires_numpy
    @pytest.mark.asyncio
    async def test_on_tick(self) -> None:
        """Test on_tick returns None (RL is bar-based)."""
        adapter = FinRLAdapter()
        await adapter.initialize({"algorithm": "ppo"})

        signal = await adapter.on_tick({"price": 100.0, "volume": 10})
        assert signal is None

    @requires_numpy
    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test adapter shutdown."""
        adapter = FinRLAdapter()
        await adapter.initialize({"algorithm": "ppo"})

        assert adapter.is_initialized is True

        await adapter.shutdown()

        assert adapter.is_initialized is False
        assert adapter.model_loaded is False

    def test_get_supported_algorithms(self) -> None:
        """Test getting supported algorithms."""
        adapter = FinRLAdapter()
        algos = adapter.get_supported_algorithms()

        assert "ppo" in algos
        assert "sac" in algos
        assert "a2c" in algos
        assert "td3" in algos
        assert "ddpg" in algos

    def test_get_model_info_no_model(self) -> None:
        """Test model info when no model loaded."""
        adapter = FinRLAdapter()
        info = adapter.get_model_info()

        assert info["loaded"] is False

    @pytest.mark.asyncio
    async def test_train_without_init(self) -> None:
        """Test train raises error without initialization."""
        adapter = FinRLAdapter()

        import polars as pl
        df = pl.DataFrame({"close": [100.0]})

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.train(df)

    @requires_numpy
    @pytest.mark.asyncio
    async def test_backtest_without_model(self) -> None:
        """Test backtest raises error without model."""
        adapter = FinRLAdapter()
        await adapter.initialize({"algorithm": "ppo"})

        import polars as pl
        df = pl.DataFrame({"close": [100.0]})

        with pytest.raises(RuntimeError, match="No model loaded"):
            await adapter.backtest(df, Decimal("10000"))


class TestFinRLNotInstalledError:
    """Tests for FinRLNotInstalledError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = FinRLNotInstalledError()
        assert "pip install" in str(error)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = FinRLNotInstalledError("Custom message")
        assert str(error) == "Custom message"

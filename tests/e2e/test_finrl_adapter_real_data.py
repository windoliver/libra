"""
E2E tests for FinRL Adapter using REAL market data from Binance.

Demonstrates the FinRL Adapter Plugin (Phase 4) working with:
- Real OHLCV data from Binance public API
- Feature engineering with real price data
- RL environment with real market conditions
- Reward function calculations with real returns

Uses direct Binance API calls (no CCXT dependency) for maximum portability.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import polars as pl
import pytest

# Check for numpy availability
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

requires_numpy = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")

from libra.plugins.finrl_adapter.config import (
    FinRLAdapterConfig,
    RewardType,
    RLAlgorithm,
)
from libra.plugins.finrl_adapter.rewards.base import BasicReward
from libra.plugins.finrl_adapter.rewards.sharpe import (
    CompositeReward,
    DifferentialSharpeReward,
    SharpeReward,
)


# =============================================================================
# Real Data Fetchers (Direct Binance API)
# =============================================================================


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    limit: int = 100,
) -> pl.DataFrame:
    """
    Fetch real OHLCV data from Binance public API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT").
        interval: Candle interval (e.g., "1h", "4h", "1d").
        limit: Number of candles to fetch.

    Returns:
        Polars DataFrame with OHLCV data.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw_data = json.loads(response.read().decode())
    except Exception as e:
        pytest.skip(f"Could not fetch Binance data: {e}")

    # Convert to Polars DataFrame with date column
    import datetime

    return pl.DataFrame({
        "date": [datetime.datetime.fromtimestamp(k[0] / 1000).strftime("%Y-%m-%d") for k in raw_data],
        "timestamp": [int(k[0]) for k in raw_data],
        "open": [float(k[1]) for k in raw_data],
        "high": [float(k[2]) for k in raw_data],
        "low": [float(k[3]) for k in raw_data],
        "close": [float(k[4]) for k in raw_data],
        "volume": [float(k[5]) for k in raw_data],
    })


def fetch_multi_symbol_data(
    symbols: list[str],
    interval: str = "1d",
    limit: int = 100,
) -> pl.DataFrame:
    """
    Fetch data for multiple symbols and combine.

    Args:
        symbols: List of trading pairs.
        interval: Candle interval.
        limit: Number of candles per symbol.

    Returns:
        Combined DataFrame with 'tic' column for symbol.
    """
    dfs = []
    for symbol in symbols:
        try:
            df = fetch_binance_klines(symbol, interval, limit)
            df = df.with_columns(pl.lit(symbol).alias("tic"))
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        pytest.skip("Could not fetch any symbol data")

    return pl.concat(dfs)


# =============================================================================
# Test: Data Integrity
# =============================================================================


class TestDataIntegrity:
    """Verify real data has expected structure."""

    @pytest.fixture
    def real_data(self) -> pl.DataFrame:
        """Fetch real market data."""
        return fetch_binance_klines(symbol="BTCUSDT", interval="1d", limit=100)

    def test_data_structure(self, real_data: pl.DataFrame) -> None:
        """Verify real data has expected columns."""
        assert real_data.height >= 50
        assert "date" in real_data.columns
        assert "open" in real_data.columns
        assert "high" in real_data.columns
        assert "low" in real_data.columns
        assert "close" in real_data.columns
        assert "volume" in real_data.columns

    def test_data_validity(self, real_data: pl.DataFrame) -> None:
        """Verify data values are valid."""
        close_min = real_data["close"].min()
        volume_min = real_data["volume"].min()

        assert close_min is not None and float(close_min) > 0
        assert volume_min is not None and float(volume_min) >= 0
        assert (real_data["high"] >= real_data["low"]).all()
        assert (real_data["high"] >= real_data["close"]).all()
        assert (real_data["high"] >= real_data["open"]).all()


# =============================================================================
# Test: Feature Engineering with Real Data
# =============================================================================


@requires_numpy
class TestFeatureEngineeringRealData:
    """Test feature engineering with real market data."""

    @pytest.fixture
    def real_data(self) -> pl.DataFrame:
        """Fetch real market data."""
        return fetch_binance_klines(symbol="BTCUSDT", interval="1d", limit=200)

    @pytest.fixture
    def multi_symbol_data(self) -> pl.DataFrame:
        """Fetch multi-symbol data."""
        return fetch_multi_symbol_data(
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            interval="1d",
            limit=100,
        )

    def test_technical_indicators_real_data(self, real_data: pl.DataFrame) -> None:
        """Test technical indicator calculation with real data."""
        from libra.plugins.finrl_adapter.features.technical import TechnicalIndicators

        calc = TechnicalIndicators()
        pandas_df = real_data.to_pandas()

        # Add indicators
        result = calc.add_all_indicators(pandas_df)

        # Verify indicators were added
        assert "macd" in result.columns
        assert "rsi_14" in result.columns or any("rsi" in c for c in result.columns)
        assert "boll_ub" in result.columns
        assert "boll_lb" in result.columns

        # Verify indicator values are reasonable
        assert not result["macd"].isna().all()

        print("\n=== Technical Indicators on Real BTC Data ===")
        print(f"Data points: {len(result)}")
        print(f"MACD range: [{result['macd'].min():.2f}, {result['macd'].max():.2f}]")
        if "rsi_14" in result.columns:
            print(f"RSI range: [{result['rsi_14'].min():.2f}, {result['rsi_14'].max():.2f}]")

    def test_feature_engineering_pipeline(self, real_data: pl.DataFrame) -> None:
        """Test full feature engineering pipeline with real data."""
        from libra.plugins.finrl_adapter.features.engineering import FeatureEngineer

        engineer = FeatureEngineer(
            tech_indicators=["macd", "rsi", "boll", "sma"],
            normalize=True,
            add_turbulence=False,  # Skip turbulence for single stock
        )

        pandas_df = real_data.to_pandas()
        result = engineer.process(pandas_df, fit=True)

        # Verify features were added
        feature_cols = engineer.get_feature_columns(result)
        assert len(feature_cols) > 6  # More than just OHLCV

        # Verify normalization
        stats = engineer.get_feature_stats()
        assert len(stats) > 0

        print("\n=== Feature Engineering Results ===")
        print(f"Original columns: {len(real_data.columns)}")
        print(f"Final columns: {len(result.columns)}")
        print(f"Feature columns: {len(feature_cols)}")
        print(f"Normalized features: {len(stats)}")

    def test_turbulence_calculation(self, multi_symbol_data: pl.DataFrame) -> None:
        """Test turbulence calculation with multi-symbol data."""
        from libra.plugins.finrl_adapter.features.technical import TechnicalIndicators

        pandas_df = multi_symbol_data.to_pandas()
        result = TechnicalIndicators.calculate_turbulence(pandas_df, lookback=20)

        assert "turbulence" in result.columns
        assert not result["turbulence"].isna().all()

        print("\n=== Turbulence Index ===")
        print(f"Min: {result['turbulence'].min():.2f}")
        print(f"Max: {result['turbulence'].max():.2f}")
        print(f"Mean: {result['turbulence'].mean():.2f}")


# =============================================================================
# Test: Reward Functions with Real Returns
# =============================================================================


@requires_numpy
class TestRewardFunctionsRealData:
    """Test reward functions with real market returns."""

    @pytest.fixture
    def real_data(self) -> pl.DataFrame:
        """Fetch real market data."""
        return fetch_binance_klines(symbol="BTCUSDT", interval="1d", limit=100)

    @pytest.fixture
    def real_returns(self, real_data: pl.DataFrame) -> list[tuple[float, float]]:
        """Calculate real portfolio values from market data."""
        closes = real_data["close"].to_list()
        # Simulate portfolio values (starting with 10000 invested in BTC)
        initial = 10000.0
        values = [(initial / closes[0]) * p for p in closes]
        return [(values[i], values[i + 1]) for i in range(len(values) - 1)]

    def test_basic_reward_real_returns(
        self,
        real_returns: list[tuple[float, float]],
    ) -> None:
        """Test basic reward with real returns."""
        reward_fn = BasicReward(scaling=1e-4)

        rewards = [reward_fn(prev, new, None) for prev, new in real_returns]

        print("\n=== Basic Reward on Real BTC Returns ===")
        print(f"Total rewards: {sum(rewards):.6f}")
        print(f"Positive rewards: {sum(1 for r in rewards if r > 0)}")
        print(f"Negative rewards: {sum(1 for r in rewards if r < 0)}")

    def test_sharpe_reward_real_returns(
        self,
        real_returns: list[tuple[float, float]],
    ) -> None:
        """Test Sharpe reward with real returns."""
        reward_fn = SharpeReward(window_size=20)

        rewards = []
        for prev, new in real_returns:
            r = reward_fn(prev, new, None)
            rewards.append(r)

        print("\n=== Sharpe Reward on Real BTC Returns ===")
        print(f"Final Sharpe-based reward: {rewards[-1]:.4f}")
        print(f"Mean reward: {np.mean(rewards):.4f}")

    def test_differential_sharpe_real_returns(
        self,
        real_returns: list[tuple[float, float]],
    ) -> None:
        """Test differential Sharpe with real returns."""
        reward_fn = DifferentialSharpeReward(eta=0.01)

        rewards = []
        for prev, new in real_returns:
            r = reward_fn(prev, new, None)
            rewards.append(r)

        print("\n=== Differential Sharpe on Real BTC Returns ===")
        print(f"Mean reward: {np.mean(rewards):.6f}")
        print(f"Std reward: {np.std(rewards):.6f}")

    def test_composite_reward_real_returns(
        self,
        real_returns: list[tuple[float, float]],
    ) -> None:
        """Test composite reward with real returns."""
        reward_fn = CompositeReward()

        rewards = []
        for prev, new in real_returns:
            r = reward_fn(prev, new, None)
            rewards.append(r)

        print("\n=== Composite Reward on Real BTC Returns ===")
        print(f"Mean reward: {np.mean(rewards):.4f}")
        print(f"Total reward: {sum(rewards):.4f}")


# =============================================================================
# Test: Trading Environment with Real Data
# =============================================================================


@requires_numpy
class TestTradingEnvironmentRealData:
    """Test trading environment with real market data."""

    @pytest.fixture
    def real_data_with_features(self) -> Any:
        """Fetch and process real data."""
        from libra.plugins.finrl_adapter.features.engineering import FeatureEngineer

        data = fetch_binance_klines(symbol="BTCUSDT", interval="1d", limit=200)
        pandas_df = data.to_pandas()

        engineer = FeatureEngineer(
            tech_indicators=["macd", "rsi", "boll"],
            add_turbulence=False,
        )
        return engineer.process(pandas_df, fit=True)

    def test_environment_creation(self, real_data_with_features: Any) -> None:
        """Test creating environment with real data."""
        from libra.plugins.finrl_adapter.environment import TradingEnvironment

        config = FinRLAdapterConfig(
            stock_dim=1,
            hmax=100,
            initial_amount=10000.0,
            tech_indicators=("macd",),
            use_turbulence=False,
        )

        env = TradingEnvironment(
            df=real_data_with_features,
            config=config,
            mode="test",
        )

        assert env.action_space is not None
        assert env.observation_space is not None

        print("\n=== Environment Created ===")
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space: {env.action_space.shape}")

    def test_environment_reset(self, real_data_with_features: Any) -> None:
        """Test environment reset with real data."""
        from libra.plugins.finrl_adapter.environment import TradingEnvironment

        config = FinRLAdapterConfig(
            stock_dim=1,
            initial_amount=10000.0,
            use_turbulence=False,
        )

        env = TradingEnvironment(df=real_data_with_features, config=config)
        obs, info = env.reset()

        assert obs is not None
        assert len(obs) > 0
        assert info["balance"] == 10000.0

        print("\n=== Environment Reset ===")
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial balance: ${info['balance']:.2f}")

    def test_environment_step(self, real_data_with_features: Any) -> None:
        """Test environment step with real data."""
        from libra.plugins.finrl_adapter.environment import TradingEnvironment

        config = FinRLAdapterConfig(
            stock_dim=1,
            initial_amount=10000.0,
            use_turbulence=False,
        )

        env = TradingEnvironment(df=real_data_with_features, config=config)
        obs, _ = env.reset()

        # Take some actions
        total_reward = 0.0
        for i in range(10):
            # Random action
            action = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print("\n=== Environment Steps ===")
        print(f"Steps taken: {i + 1}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Final balance: ${info['balance']:.2f}")
        print(f"Portfolio value: ${info['portfolio_value']:.2f}")

    def test_environment_full_episode(self, real_data_with_features: Any) -> None:
        """Test running a full episode with real data."""
        from libra.plugins.finrl_adapter.environment import TradingEnvironment

        config = FinRLAdapterConfig(
            stock_dim=1,
            initial_amount=10000.0,
            reward_type=RewardType.SHARPE,
            use_turbulence=False,
        )

        env = TradingEnvironment(df=real_data_with_features, config=config)
        obs, _ = env.reset()

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Simple momentum strategy: buy if recent return positive
            if steps > 0 and len(env.returns_memory) > 0:
                action = np.array([1.0 if env.returns_memory[-1] > 0 else -1.0], dtype=np.float32)
            else:
                action = np.array([0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print("\n=== Full Episode Results (Real BTC Data) ===")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Total return: {info.get('total_return', 0) * 100:.2f}%")
        print(f"Sharpe ratio: {info.get('sharpe_ratio', 0):.2f}")
        print(f"Max drawdown: {info.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Total trades: {info.get('total_trades', 0)}")


# =============================================================================
# Test: Model Registry
# =============================================================================


class TestModelRegistryRealScenario:
    """Test model registry with realistic scenarios."""

    def test_register_and_compare_models(self, tmp_path: Any) -> None:
        """Test registering and comparing models."""
        from libra.plugins.finrl_adapter.models.registry import ModelRegistry

        registry = ModelRegistry(tmp_path)

        # Create dummy model files
        model1_path = tmp_path / "model_v1.zip"
        model1_path.write_text("dummy model 1")

        model2_path = tmp_path / "model_v2.zip"
        model2_path.write_text("dummy model 2")

        # Register models with realistic metrics
        m1 = registry.register_model(
            model_path=model1_path,
            name="btc_trader",
            version="1.0.0",
            algorithm="ppo",
            performance_metrics={
                "sharpe_ratio": 1.2,
                "total_return": 0.15,
                "max_drawdown": 0.08,
            },
            tags=["crypto", "btc"],
        )

        m2 = registry.register_model(
            model_path=model2_path,
            name="btc_trader",
            version="2.0.0",
            algorithm="sac",
            performance_metrics={
                "sharpe_ratio": 1.8,
                "total_return": 0.22,
                "max_drawdown": 0.05,
            },
            tags=["crypto", "btc", "improved"],
        )

        # Compare models
        comparison = registry.compare_models([m1.model_id, m2.model_id])

        assert len(comparison["models"]) == 2
        assert "sharpe_ratio" in comparison["metrics"]

        # v2 should be better
        sharpe_values = comparison["metrics"]["sharpe_ratio"]["values"]
        assert sharpe_values[1] > sharpe_values[0]

        print("\n=== Model Comparison ===")
        print(f"Model v1 Sharpe: {sharpe_values[0]:.2f}")
        print(f"Model v2 Sharpe: {sharpe_values[1]:.2f}")
        print(f"Best model index: {comparison['metrics']['sharpe_ratio']['best_idx']}")

    def test_model_promotion_workflow(self, tmp_path: Any) -> None:
        """Test model promotion workflow."""
        from libra.plugins.finrl_adapter.models.registry import ModelRegistry

        registry = ModelRegistry(tmp_path)

        # Create and register model
        model_path = tmp_path / "model.zip"
        model_path.write_text("dummy model")

        metadata = registry.register_model(
            model_path=model_path,
            name="eth_trader",
            version="1.0.0",
            algorithm="ppo",
        )

        assert metadata.status == "ready"

        # Promote to staging
        staging = registry.update_model(metadata.model_id, status="staging")
        assert staging is not None
        assert staging.status == "staging"

        # Promote to production
        production = registry.promote_model(metadata.model_id, stage="production")
        assert production is not None
        assert production.status == "production"

        # Verify we can get production model
        prod_model = registry.get_production_model("eth_trader")
        assert prod_model is not None
        assert prod_model.model_id == metadata.model_id

        print("\n=== Model Promotion ===")
        print(f"Model: {prod_model.name} v{prod_model.version}")
        print(f"Status: {prod_model.status}")


# =============================================================================
# Test: Configuration with Real Scenarios
# =============================================================================


class TestConfigurationRealScenarios:
    """Test configuration for real trading scenarios."""

    def test_crypto_trading_config(self) -> None:
        """Test configuration for crypto trading."""
        config = FinRLAdapterConfig(
            stock_dim=5,  # 5 crypto assets
            hmax=10,  # Max 10 units per trade
            initial_amount=10000.0,
            transaction_cost_pct=0.001,  # 0.1% typical for crypto
            algorithm=RLAlgorithm.PPO,
            total_timesteps=100000,
            reward_type=RewardType.SHARPE,
            tech_indicators=(
                "macd",
                "rsi_30",
                "boll_ub",
                "boll_lb",
                "close_30_sma",
            ),
            use_turbulence=True,
            turbulence_threshold=100.0,
        )

        assert config.get_state_space_dim() > 10
        assert config.get_action_space_dim() == 5

        print("\n=== Crypto Trading Config ===")
        print(f"State space dim: {config.get_state_space_dim()}")
        print(f"Action space dim: {config.get_action_space_dim()}")
        print(f"Algorithm: {config.algorithm.value}")
        print(f"Reward type: {config.reward_type.value}")

    def test_stock_trading_config(self) -> None:
        """Test configuration for stock trading."""
        config = FinRLAdapterConfig(
            stock_dim=30,  # DOW 30
            hmax=100,
            initial_amount=1000000.0,
            transaction_cost_pct=0.0001,  # Lower for stocks
            algorithm=RLAlgorithm.SAC,
            total_timesteps=500000,
            reward_type=RewardType.COMPOSITE,
            gamma=0.99,
        )

        kwargs = config.get_algorithm_kwargs()
        assert "buffer_size" in kwargs  # SAC uses replay buffer
        assert kwargs["gamma"] == 0.99

        print("\n=== Stock Trading Config ===")
        print(f"Stock dim: {config.stock_dim}")
        print(f"Initial capital: ${config.initial_amount:,.0f}")
        print(f"Algorithm kwargs: {list(kwargs.keys())}")


# =============================================================================
# Test: Full Adapter Workflow (without training)
# =============================================================================


@requires_numpy
class TestFinRLAdapterRealData:
    """Test FinRL adapter with real data (without actual training)."""

    @pytest.fixture
    def real_data(self) -> pl.DataFrame:
        """Fetch real market data."""
        return fetch_binance_klines(symbol="BTCUSDT", interval="1d", limit=200)

    def test_adapter_metadata(self) -> None:
        """Test adapter metadata."""
        from libra.plugins.finrl_adapter.adapter import FinRLAdapter

        metadata = FinRLAdapter.metadata()

        assert metadata.name == "finrl-adapter"
        assert "0.1.0" in metadata.version
        assert len(metadata.requires) > 0

        print("\n=== FinRL Adapter Metadata ===")
        print(f"Name: {metadata.name}")
        print(f"Version: {metadata.version}")
        print(f"Requires: {metadata.requires}")

    @pytest.mark.asyncio
    async def test_adapter_initialization(self) -> None:
        """Test adapter initialization."""
        from libra.plugins.finrl_adapter.adapter import FinRLAdapter

        adapter = FinRLAdapter()

        await adapter.initialize({
            "algorithm": "ppo",
            "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
            "stock_dim": 2,
            "tech_indicators": ["macd", "rsi"],
        })

        assert adapter.is_initialized
        assert adapter.name == "finrl:ppo"
        assert len(adapter.symbols) == 2

        print("\n=== Adapter Initialized ===")
        print(f"Name: {adapter.name}")
        print(f"Symbols: {adapter.symbols}")
        print(f"Model loaded: {adapter.model_loaded}")

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_on_data_no_model(self, real_data: pl.DataFrame) -> None:
        """Test on_data returns None without model."""
        from libra.plugins.finrl_adapter.adapter import FinRLAdapter

        adapter = FinRLAdapter()
        await adapter.initialize({
            "algorithm": "ppo",
            "stock_dim": 1,
        })

        signal = await adapter.on_data(real_data)

        # Without a trained model, should return None
        assert signal is None

        await adapter.shutdown()

    def test_supported_algorithms(self) -> None:
        """Test getting supported algorithms."""
        from libra.plugins.finrl_adapter.adapter import FinRLAdapter

        adapter = FinRLAdapter()
        algos = adapter.get_supported_algorithms()

        assert "ppo" in algos
        assert "sac" in algos
        assert "a2c" in algos
        assert "td3" in algos
        assert "ddpg" in algos

        print("\n=== Supported Algorithms ===")
        for algo in algos:
            print(f"  - {algo}")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

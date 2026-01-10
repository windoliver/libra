"""
Main FinRL Adapter for LIBRA.

Implements the StrategyPlugin interface to enable training and deploying
FinRL reinforcement learning strategies within the LIBRA trading platform.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from libra.plugins.base import PluginMetadata, StrategyPlugin
from libra.plugins.finrl_adapter.config import (
    FinRLAdapterConfig,
    InferenceConfig,
    RLAlgorithm,
    TrainingConfig,
)
from libra.plugins.finrl_adapter.features.engineering import FeatureEngineer
from libra.strategies.protocol import BacktestResult, Signal, SignalType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

try:
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class FinRLNotInstalledError(ImportError):
    """Raised when FinRL dependencies are not available."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "FinRL dependencies not installed. Install with: pip install libra[finrl]"
        )


class FinRLAdapter(StrategyPlugin):
    """
    Adapter for training and deploying FinRL RL strategies in LIBRA.

    This adapter enables integration of reinforcement learning strategies
    with LIBRA's trading infrastructure. It provides:

    - RL environment creation for trading
    - Training with Stable-Baselines3 algorithms
    - Hyperparameter tuning with Optuna
    - Model versioning and registry
    - Live inference with risk management
    - Signal conversion to LIBRA format

    Usage:
        # Create adapter
        adapter = FinRLAdapter()

        # Initialize with configuration
        await adapter.initialize({
            "algorithm": "ppo",
            "model_path": "/path/to/model.zip",
            "pair_whitelist": ["AAPL", "GOOGL", "MSFT"],
        })

        # Process market data
        signal = await adapter.on_data(ohlcv_dataframe)
        if signal:
            print(f"Signal: {signal.signal_type}")

        # Train a new model
        result = await adapter.train(historical_data)
        print(f"Training complete: {result}")

    Lifecycle:
        1. Create adapter instance
        2. Call initialize() with configuration
        3. Either:
           a. Call on_data() for live inference
           b. Call train() for model training
        4. Call shutdown() when done

    Thread Safety:
        The adapter is NOT thread-safe. Use separate instances for concurrent
        processing or implement external synchronization.
    """

    # Plugin version
    VERSION = "0.1.0"

    # Minimum SB3 version
    MIN_SB3_VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the FinRL adapter."""
        self._config: FinRLAdapterConfig | None = None
        self._inference_config: InferenceConfig | None = None
        self._training_config: TrainingConfig | None = None
        self._model: Any = None
        self._feature_engineer: FeatureEngineer | None = None
        self._initialized = False
        self._symbols: list[str] = []
        self._last_observation: Any = None

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """
        Return plugin metadata.

        Returns:
            PluginMetadata describing this adapter.
        """
        return PluginMetadata.create(
            name="finrl-adapter",
            version=cls.VERSION,
            description="Train and deploy RL trading strategies with FinRL",
            author="LIBRA Team",
            requires=[
                f"stable-baselines3>={cls.MIN_SB3_VERSION}",
                "gymnasium>=0.28.1",
            ],
        )

    @property
    def name(self) -> str:
        """
        Strategy name.

        Returns:
            Combined name of adapter and algorithm.
        """
        if self._config:
            return f"finrl:{self._config.algorithm.value}"
        return "finrl:uninitialized"

    @property
    def symbols(self) -> list[str]:
        """
        Symbols this strategy trades.

        Returns:
            List of trading symbols.
        """
        return self._symbols

    @property
    def is_initialized(self) -> bool:
        """Whether the adapter has been initialized."""
        return self._initialized

    @property
    def model_loaded(self) -> bool:
        """Whether a model is loaded for inference."""
        return self._model is not None

    @property
    def config(self) -> FinRLAdapterConfig | None:
        """Access the adapter configuration."""
        return self._config

    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the adapter with configuration.

        Args:
            config: Configuration dictionary with keys:
                - algorithm: RL algorithm (ppo, sac, a2c, td3, ddpg)
                - model_path: Path to pre-trained model (optional)
                - pair_whitelist: List of trading symbols
                - tech_indicators: Technical indicators to use
                - Plus any FinRLAdapterConfig fields

        Raises:
            FinRLNotInstalledError: If required dependencies not installed.
            ValueError: If configuration is invalid.
        """
        if not NUMPY_AVAILABLE:
            raise FinRLNotInstalledError("NumPy is required")

        # Parse configuration
        self._config = FinRLAdapterConfig.from_dict(config)

        # Parse inference config if present
        if "inference" in config:
            self._inference_config = InferenceConfig(**config["inference"])
        else:
            self._inference_config = InferenceConfig()

        # Parse training config if present
        if "training" in config:
            self._training_config = TrainingConfig(**config["training"])
        else:
            self._training_config = TrainingConfig()

        # Set symbols
        self._symbols = list(self._config.pair_whitelist)

        # Initialize feature engineer
        self._feature_engineer = FeatureEngineer(
            tech_indicators=list(self._config.tech_indicators),
            add_turbulence=self._config.use_turbulence,
        )

        # Load model if path provided
        if self._config.model_path and self._config.model_path.exists():
            await self._load_model(self._config.model_path)

        self._initialized = True
        logger.info(
            "Initialized FinRLAdapter: algorithm=%s, symbols=%s, model_loaded=%s",
            self._config.algorithm.value,
            self._symbols,
            self.model_loaded,
        )

    async def _load_model(self, model_path: Path) -> None:
        """Load a pre-trained model."""
        if not SB3_AVAILABLE:
            raise FinRLNotInstalledError("stable-baselines3 is required")

        algorithm_map = {
            RLAlgorithm.PPO: PPO,
            RLAlgorithm.A2C: A2C,
            RLAlgorithm.SAC: SAC,
            RLAlgorithm.TD3: TD3,
            RLAlgorithm.DDPG: DDPG,
        }

        AlgoClass = algorithm_map.get(self._config.algorithm)  # type: ignore[union-attr]
        if AlgoClass is None:
            raise ValueError(f"Unsupported algorithm: {self._config.algorithm}")  # type: ignore[union-attr]

        self._model = AlgoClass.load(str(model_path))
        logger.info("Loaded model from %s", model_path)

    async def on_data(self, data: pl.DataFrame) -> Signal | None:
        """
        Process market data and generate trading signal.

        Runs the RL model on the provided data and converts the
        action to a LIBRA Signal.

        Args:
            data: Polars OHLCV DataFrame.

        Returns:
            Signal if the model generates a trade signal, None otherwise.

        Raises:
            RuntimeError: If adapter is not initialized or no model loaded.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        if self._model is None:
            logger.warning("No model loaded, cannot generate signals")
            return None

        if data.is_empty():
            return None

        # Convert to pandas and add features
        pandas_df = data.to_pandas()
        if self._feature_engineer:
            pandas_df = self._feature_engineer.process(pandas_df, fit=False)

        # Build observation
        observation = self._build_observation(pandas_df)
        if observation is None:
            return None

        self._last_observation = observation

        # Get action from model
        action, _ = self._model.predict(observation, deterministic=True)
        action = action.flatten()

        # Convert action to signal
        return self._action_to_signal(action, pandas_df)

    def _build_observation(self, df: Any) -> Any:
        """Build observation vector from DataFrame."""
        if not NUMPY_AVAILABLE:
            return None

        # Get the most recent row
        if len(df) == 0:
            return None

        latest = df.iloc[-1]

        # Build state vector similar to environment
        state_parts = []

        # Normalized balance (placeholder)
        state_parts.append(1.0)

        # Stock prices (normalized)
        if "close" in df.columns:
            close_prices = df["close"].values[-self._config.stock_dim :]  # type: ignore[union-attr]
            if len(close_prices) < self._config.stock_dim:  # type: ignore[union-attr]
                close_prices = np.pad(
                    close_prices, (0, self._config.stock_dim - len(close_prices))  # type: ignore[union-attr]
                )
            state_parts.extend(close_prices / close_prices[0] if close_prices[0] != 0 else close_prices)

        # Holdings (placeholder zeros)
        state_parts.extend([0.0] * self._config.stock_dim)  # type: ignore[union-attr]

        # Technical indicators
        for indicator in self._config.tech_indicators:  # type: ignore[union-attr]
            if indicator in df.columns:
                vals = df[indicator].values[-self._config.stock_dim :]  # type: ignore[union-attr]
                if len(vals) < self._config.stock_dim:  # type: ignore[union-attr]
                    vals = np.pad(vals, (0, self._config.stock_dim - len(vals)))  # type: ignore[union-attr]
                state_parts.extend(vals)
            else:
                state_parts.extend([0.0] * self._config.stock_dim)  # type: ignore[union-attr]

        # Turbulence
        if self._config.use_turbulence:  # type: ignore[union-attr]
            turbulence = latest.get("turbulence", 0.0) if hasattr(latest, "get") else 0.0
            state_parts.append(turbulence)

        return np.array(state_parts, dtype=np.float32).reshape(1, -1)

    def _action_to_signal(self, action: Any, df: Any) -> Signal | None:
        """Convert RL action to LIBRA Signal."""
        if not NUMPY_AVAILABLE:
            return None

        # Get the dominant action
        if len(action) == 0:
            return None

        # Sum of actions indicates overall direction
        total_action = float(np.sum(action))

        # Get symbol (use first if multiple)
        symbol = self._symbols[0] if self._symbols else "UNKNOWN"

        # Get latest price
        price = Decimal(str(df["close"].iloc[-1])) if "close" in df.columns else Decimal("0")

        # Determine signal type based on action
        if total_action > 0.1:  # Buy threshold
            return Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                price=price,
                quantity=Decimal(str(abs(total_action) * self._config.hmax)),  # type: ignore[union-attr]
                confidence=min(abs(total_action), 1.0),
                metadata={
                    "source": "finrl",
                    "algorithm": self._config.algorithm.value,  # type: ignore[union-attr]
                    "raw_action": action.tolist(),
                },
            )
        elif total_action < -0.1:  # Sell threshold
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                price=price,
                quantity=Decimal(str(abs(total_action) * self._config.hmax)),  # type: ignore[union-attr]
                confidence=min(abs(total_action), 1.0),
                metadata={
                    "source": "finrl",
                    "algorithm": self._config.algorithm.value,  # type: ignore[union-attr]
                    "raw_action": action.tolist(),
                },
            )

        # No significant action
        return None

    async def on_tick(self, tick: dict[str, Any]) -> Signal | None:
        """
        Process real-time tick data.

        RL strategies are primarily bar-based, so tick processing is limited.

        Args:
            tick: Tick data dictionary.

        Returns:
            None (RL models work on bars, not ticks).
        """
        logger.debug("Tick received but RL is bar-based: %s", tick)
        return None

    async def on_fill(self, order_result: dict[str, Any]) -> None:
        """
        Handle order fill notification.

        Args:
            order_result: Order fill details.
        """
        logger.debug("Order filled: %s", order_result)

    async def train(
        self,
        train_data: pl.DataFrame,
        eval_data: pl.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Train an RL model on historical data.

        Args:
            train_data: Training DataFrame with OHLCV data.
            eval_data: Optional evaluation DataFrame.

        Returns:
            Dictionary with training results and metrics.

        Raises:
            RuntimeError: If adapter not initialized.
            FinRLNotInstalledError: If SB3 not available.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        if not SB3_AVAILABLE:
            raise FinRLNotInstalledError("stable-baselines3 is required for training")

        from libra.plugins.finrl_adapter.training import RLTrainer

        logger.info(
            "Starting training: algorithm=%s, timesteps=%d",
            self._config.algorithm.value,
            self._config.total_timesteps,
        )

        # Convert to pandas and engineer features
        train_df = train_data.to_pandas()
        if self._feature_engineer:
            train_df = self._feature_engineer.process(train_df, fit=True)

        eval_df = None
        if eval_data is not None:
            eval_df = eval_data.to_pandas()
            if self._feature_engineer:
                eval_df = self._feature_engineer.process(eval_df, fit=False)

        # Train
        trainer = RLTrainer(self._config, self._training_config)
        self._model = trainer.train(train_df, eval_df)

        # Save model
        if self._config.model_path:
            model_path = trainer.save()
            logger.info("Model saved to %s", model_path)

        return {
            "algorithm": self._config.algorithm.value,
            "total_timesteps": self._config.total_timesteps,
            "model_path": str(self._config.model_path) if self._config.model_path else None,
            "status": "completed",
        }

    async def backtest(
        self,
        data: pl.DataFrame,
        initial_capital: Decimal,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Historical OHLCV DataFrame.
            initial_capital: Starting capital.

        Returns:
            BacktestResult with performance metrics.

        Raises:
            RuntimeError: If adapter not initialized or no model loaded.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        if self._model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")

        from libra.plugins.finrl_adapter.environment import TradingEnvironment

        # Prepare data
        pandas_df = data.to_pandas()
        if self._feature_engineer:
            pandas_df = self._feature_engineer.process(pandas_df, fit=False)

        # Create environment
        env = TradingEnvironment(
            df=pandas_df,
            config=self._config,
            mode="test",
        )

        # Run backtest
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Get final metrics
        terminal_info = info

        return BacktestResult(
            total_return=Decimal(str(terminal_info.get("total_return", 0))),
            sharpe_ratio=terminal_info.get("sharpe_ratio", 0),
            max_drawdown=Decimal(str(terminal_info.get("max_drawdown", 0))),
            win_rate=0.5,  # Not tracked in basic env
            total_trades=terminal_info.get("total_trades", 0),
            profit_factor=Decimal("1"),
            start_date=pandas_df["date"].iloc[0] if "date" in pandas_df.columns else None,
            end_date=pandas_df["date"].iloc[-1] if "date" in pandas_df.columns else None,
        )

    async def optimize(
        self,
        data: pl.DataFrame,
        param_space: dict[str, Any],
        metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """
        Optimize strategy hyperparameters.

        Args:
            data: Historical data for optimization.
            param_space: Parameter search space.
            metric: Metric to optimize.

        Returns:
            Dictionary with optimized parameters.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        try:
            from libra.plugins.finrl_adapter.training import HyperparameterTuner
        except ImportError:
            raise FinRLNotInstalledError(
                "Optuna is required for optimization. Install with: pip install optuna"
            )

        # Split data
        pandas_df = data.to_pandas()
        split_idx = int(len(pandas_df) * 0.8)
        train_df = pandas_df.iloc[:split_idx]
        eval_df = pandas_df.iloc[split_idx:]

        if self._feature_engineer:
            train_df = self._feature_engineer.process(train_df, fit=True)
            eval_df = self._feature_engineer.process(eval_df, fit=False)

        # Run optimization
        tuner = HyperparameterTuner(
            base_config=self._config,
            train_df=train_df,
            eval_df=eval_df,
            training_config=self._training_config,
        )

        n_trials = param_space.get("n_trials", 20)
        best_params = tuner.tune(n_trials=n_trials)

        return {
            "best_params": best_params,
            "best_value": tuner.best_value,
            "metric": metric,
        }

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._model = None
        self._feature_engineer = None
        self._initialized = False
        logger.info("FinRLAdapter shut down")

    # -------------------------------------------------------------------------
    # Additional Methods
    # -------------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        if self._model is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "algorithm": self._config.algorithm.value if self._config else None,
            "policy_type": type(self._model.policy).__name__ if hasattr(self._model, "policy") else None,
        }

    def get_supported_algorithms(self) -> list[str]:
        """Get list of supported RL algorithms."""
        return [algo.value for algo in RLAlgorithm]

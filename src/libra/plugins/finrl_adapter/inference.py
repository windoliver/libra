"""
Inference Engine for FinRL Adapter.

Provides inference functionality for deployed RL trading models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from libra.plugins.finrl_adapter.config import (
    FinRLAdapterConfig,
    InferenceConfig,
    RLAlgorithm,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

# Try to import stable-baselines3
try:
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.vec_env import VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class InferenceEngine:
    """
    Inference engine for RL trading models.

    Provides:
    - Model loading and caching
    - Real-time action prediction
    - Risk management integration
    - Shadow mode for testing
    - Batch inference for backtesting

    Usage:
        engine = InferenceEngine(config)
        engine.load_model("path/to/model.zip")
        action = engine.predict(observation)
    """

    # Algorithm mapping
    ALGORITHMS = {
        RLAlgorithm.PPO: PPO if SB3_AVAILABLE else None,
        RLAlgorithm.A2C: A2C if SB3_AVAILABLE else None,
        RLAlgorithm.SAC: SAC if SB3_AVAILABLE else None,
        RLAlgorithm.TD3: TD3 if SB3_AVAILABLE else None,
        RLAlgorithm.DDPG: DDPG if SB3_AVAILABLE else None,
    }

    def __init__(
        self,
        config: FinRLAdapterConfig,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        """
        Initialize the inference engine.

        Args:
            config: FinRL adapter configuration.
            inference_config: Inference-specific configuration.
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for inference. "
                "Install with: pip install stable-baselines3"
            )
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for inference. "
                "Install with: pip install numpy"
            )

        self.config = config
        self.inference_config = inference_config or InferenceConfig()
        self.model: Any = None
        self.vec_normalize: Any = None
        self._model_path: Path | None = None

    def load_model(
        self,
        model_path: Path | str,
        algorithm: RLAlgorithm | None = None,
    ) -> None:
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model (.zip file).
            algorithm: Algorithm type (auto-detected from config if not specified).
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        algorithm = algorithm or self.config.algorithm
        AlgoClass = self.ALGORITHMS.get(algorithm)
        if AlgoClass is None:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.model = AlgoClass.load(str(model_path))
        self._model_path = model_path

        # Load VecNormalize stats if available
        stats_path = model_path.with_suffix(".pkl")
        if stats_path.exists():
            try:
                # Load normalization stats
                # Note: Full loading requires env, we store stats for manual normalization
                import pickle

                with open(stats_path, "rb") as f:
                    self.vec_normalize = pickle.load(f)
                logger.info("Loaded normalization stats from %s", stats_path)
            except Exception as e:
                logger.warning("Could not load normalization stats: %s", e)

        logger.info("Model loaded from %s", model_path)

    def predict(
        self,
        observation: NDArray[np.float32] | list[float],
        state: Any | None = None,
        deterministic: bool | None = None,
    ) -> tuple[NDArray[np.float32], Any]:
        """
        Predict action for a given observation.

        Args:
            observation: Current state observation.
            state: RNN hidden state (for recurrent policies).
            deterministic: Whether to use deterministic actions.

        Returns:
            Tuple of (action, new_state).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        deterministic = (
            deterministic
            if deterministic is not None
            else self.inference_config.deterministic
        )

        # Ensure observation is numpy array
        if isinstance(observation, list):
            observation = np.array(observation, dtype=np.float32)

        # Normalize observation if stats available
        if self.vec_normalize is not None:
            observation = self._normalize_observation(observation)

        # Add batch dimension if needed
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # Get prediction
        action, state = self.model.predict(
            observation,
            state=state,
            deterministic=deterministic,
        )

        return action, state

    def predict_with_risk_management(
        self,
        observation: NDArray[np.float32] | list[float],
        turbulence: float = 0.0,
        current_holdings: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """
        Predict action with risk management constraints.

        Args:
            observation: Current state observation.
            turbulence: Current turbulence index.
            current_holdings: Current portfolio holdings.

        Returns:
            Risk-adjusted action array.
        """
        # Get base prediction
        action, _ = self.predict(observation)
        action = action.flatten()

        # Apply turbulence-based risk reduction
        if turbulence > self.inference_config.risk_threshold:
            # High turbulence: reduce position sizes
            risk_factor = self.inference_config.risk_threshold / (turbulence + 1e-8)
            risk_factor = max(0.0, min(1.0, risk_factor))
            action = action * risk_factor
            logger.debug(
                "Turbulence risk adjustment: factor=%.2f, turbulence=%.2f",
                risk_factor,
                turbulence,
            )

        # Apply position size limits
        max_position = self.inference_config.max_position_pct
        action = np.clip(action, -max_position, max_position)

        # Shadow mode: return zeros (no actual trades)
        if self.inference_config.enable_shadow_mode:
            logger.debug("Shadow mode: returning zero actions")
            return np.zeros_like(action)

        return action

    def _normalize_observation(
        self,
        observation: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Normalize observation using saved statistics."""
        if self.vec_normalize is None:
            return observation

        try:
            if hasattr(self.vec_normalize, "obs_rms"):
                # VecNormalize object
                mean = self.vec_normalize.obs_rms.mean
                var = self.vec_normalize.obs_rms.var
                observation = (observation - mean) / np.sqrt(var + 1e-8)
                observation = np.clip(observation, -10.0, 10.0)
        except Exception as e:
            logger.warning("Normalization failed: %s", e)

        return observation

    def batch_predict(
        self,
        observations: NDArray[np.float32],
        deterministic: bool = True,
    ) -> NDArray[np.float32]:
        """
        Predict actions for a batch of observations.

        Args:
            observations: Batch of observations (N, obs_dim).
            deterministic: Whether to use deterministic actions.

        Returns:
            Batch of actions (N, action_dim).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Normalize if stats available
        if self.vec_normalize is not None:
            observations = np.array(
                [self._normalize_observation(obs) for obs in observations]
            )

        actions, _ = self.model.predict(observations, deterministic=deterministic)
        return actions

    def get_action_distribution(
        self,
        observation: NDArray[np.float32],
    ) -> dict[str, Any]:
        """
        Get the action distribution for an observation.

        Useful for understanding model uncertainty.

        Args:
            observation: Current state observation.

        Returns:
            Dictionary with distribution parameters.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Ensure observation is properly shaped
        if isinstance(observation, list):
            observation = np.array(observation, dtype=np.float32)
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # Normalize
        if self.vec_normalize is not None:
            observation = self._normalize_observation(observation)

        # Get distribution from policy
        try:
            # This works for PPO/A2C
            if hasattr(self.model.policy, "get_distribution"):
                import torch

                obs_tensor = torch.as_tensor(observation).float()
                distribution = self.model.policy.get_distribution(obs_tensor)
                return {
                    "mean": distribution.distribution.mean.detach().numpy(),
                    "std": distribution.distribution.stddev.detach().numpy(),
                }
        except Exception as e:
            logger.debug("Could not get distribution: %s", e)

        # Fallback: return deterministic and stochastic actions
        action_det, _ = self.model.predict(observation, deterministic=True)
        action_stoch, _ = self.model.predict(observation, deterministic=False)

        return {
            "deterministic": action_det,
            "stochastic": action_stoch,
        }

    @property
    def is_loaded(self) -> bool:
        """Whether a model is loaded."""
        return self.model is not None

    @property
    def model_path(self) -> Path | None:
        """Path to the loaded model."""
        return self._model_path


class LiveTradingEngine:
    """
    Live trading engine for deploying RL models.

    Wraps the inference engine with additional functionality for live trading:
    - State management across time steps
    - Order generation
    - Position tracking
    - Performance monitoring
    """

    def __init__(
        self,
        config: FinRLAdapterConfig,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        """
        Initialize the live trading engine.

        Args:
            config: FinRL adapter configuration.
            inference_config: Inference configuration.
        """
        self.config = config
        self.inference = InferenceEngine(config, inference_config)

        # State tracking
        self.current_holdings: NDArray[np.float32] | None = None
        self.current_balance: float = config.initial_amount
        self.trade_history: list[dict[str, Any]] = []
        self.performance_history: list[dict[str, Any]] = []

    def initialize(
        self,
        model_path: Path | str,
        initial_holdings: NDArray[np.float32] | None = None,
        initial_balance: float | None = None,
    ) -> None:
        """
        Initialize the trading engine with a model.

        Args:
            model_path: Path to the trained model.
            initial_holdings: Starting portfolio holdings.
            initial_balance: Starting cash balance.
        """
        self.inference.load_model(model_path)

        if initial_holdings is not None:
            self.current_holdings = initial_holdings
        else:
            self.current_holdings = np.zeros(
                self.config.stock_dim, dtype=np.float32
            )

        if initial_balance is not None:
            self.current_balance = initial_balance

        logger.info(
            "Live trading engine initialized: balance=%.2f, holdings=%s",
            self.current_balance,
            self.current_holdings,
        )

    def generate_orders(
        self,
        observation: NDArray[np.float32],
        prices: NDArray[np.float32],
        turbulence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Generate trading orders based on current state.

        Args:
            observation: Current market observation.
            prices: Current asset prices.
            turbulence: Current turbulence index.

        Returns:
            List of order dictionaries.
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for order generation")

        if self.current_holdings is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Get action from model
        action = self.inference.predict_with_risk_management(
            observation=observation,
            turbulence=turbulence,
            current_holdings=self.current_holdings,
        )

        # Convert actions to orders
        orders = []
        for i, (act, price) in enumerate(zip(action, prices)):
            if abs(act) < 0.01:  # Skip tiny actions
                continue

            shares = int(act * self.config.hmax)
            if shares == 0:
                continue

            if shares > 0:
                # Buy order
                max_affordable = int(
                    self.current_balance
                    / (price * (1 + self.config.transaction_cost_pct))
                )
                shares = min(shares, max_affordable)
                if shares > 0:
                    orders.append(
                        {
                            "type": "buy",
                            "symbol_idx": i,
                            "shares": shares,
                            "price": float(price),
                            "action": float(act),
                        }
                    )
            else:
                # Sell order
                shares = min(abs(shares), int(self.current_holdings[i]))
                if shares > 0:
                    orders.append(
                        {
                            "type": "sell",
                            "symbol_idx": i,
                            "shares": shares,
                            "price": float(price),
                            "action": float(act),
                        }
                    )

        return orders

    def execute_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """
        Execute an order and update state.

        Args:
            order: Order dictionary from generate_orders.

        Returns:
            Execution result dictionary.
        """
        if self.current_holdings is None:
            raise RuntimeError("Engine not initialized")

        idx = order["symbol_idx"]
        shares = order["shares"]
        price = order["price"]
        cost_pct = self.config.transaction_cost_pct

        if order["type"] == "buy":
            total_cost = shares * price * (1 + cost_pct)
            if total_cost > self.current_balance:
                return {"success": False, "reason": "Insufficient balance"}

            self.current_balance -= total_cost
            self.current_holdings[idx] += shares
            transaction_cost = shares * price * cost_pct

        else:  # sell
            if shares > self.current_holdings[idx]:
                return {"success": False, "reason": "Insufficient holdings"}

            proceeds = shares * price * (1 - cost_pct)
            self.current_balance += proceeds
            self.current_holdings[idx] -= shares
            transaction_cost = shares * price * cost_pct

        result = {
            "success": True,
            "type": order["type"],
            "symbol_idx": idx,
            "shares": shares,
            "price": price,
            "transaction_cost": transaction_cost,
            "balance": self.current_balance,
        }

        self.trade_history.append(result)
        return result

    def get_portfolio_value(self, prices: NDArray[np.float32]) -> float:
        """Calculate current portfolio value."""
        if self.current_holdings is None:
            return self.current_balance

        holdings_value = float(np.sum(self.current_holdings * prices))
        return self.current_balance + holdings_value

    def record_performance(
        self,
        timestamp: Any,
        prices: NDArray[np.float32],
    ) -> dict[str, Any]:
        """
        Record current performance metrics.

        Args:
            timestamp: Current timestamp.
            prices: Current asset prices.

        Returns:
            Performance snapshot dictionary.
        """
        portfolio_value = self.get_portfolio_value(prices)

        snapshot = {
            "timestamp": timestamp,
            "portfolio_value": portfolio_value,
            "balance": self.current_balance,
            "holdings": self.current_holdings.copy() if self.current_holdings is not None else None,
            "n_trades": len(self.trade_history),
        }

        self.performance_history.append(snapshot)
        return snapshot

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of trading performance."""
        if not self.performance_history:
            return {}

        values = [p["portfolio_value"] for p in self.performance_history]
        initial = self.config.initial_amount
        final = values[-1] if values else initial

        # Calculate metrics
        total_return = (final - initial) / initial
        max_value = max(values) if values else initial
        max_drawdown = max(
            (max_value - v) / max_value for v in values
        ) if values else 0.0

        # Calculate Sharpe (simplified)
        if len(values) > 1:
            returns = [
                (values[i] - values[i - 1]) / values[i - 1]
                for i in range(1, len(values))
            ]
            if returns:
                mean_ret = sum(returns) / len(returns)
                std_ret = (
                    sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                ) ** 0.5
                sharpe = mean_ret / std_ret * (252 ** 0.5) if std_ret > 0 else 0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return {
            "total_return": total_return,
            "final_value": final,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "n_trades": len(self.trade_history),
            "n_periods": len(self.performance_history),
        }

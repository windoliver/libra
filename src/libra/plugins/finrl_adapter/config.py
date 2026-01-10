"""
Configuration for FinRL Adapter.

Defines the configuration schema for FinRL reinforcement learning strategies.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import msgspec


class RLAlgorithm(str, Enum):
    """Supported reinforcement learning algorithms."""

    PPO = "ppo"
    A2C = "a2c"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"


class RewardType(str, Enum):
    """Reward function types."""

    BASIC = "basic"  # Simple portfolio value change
    SHARPE = "sharpe"  # Sharpe ratio based
    DIFFERENTIAL_SHARPE = "differential_sharpe"  # Online Sharpe
    COMPOSITE = "composite"  # Multi-objective


class FinRLAdapterConfig(msgspec.Struct, frozen=True, kw_only=True):
    """
    Configuration for the FinRL adapter.

    Attributes:
        stock_dim: Number of stocks/assets in the portfolio.
        hmax: Maximum number of shares to trade per action.
        initial_amount: Initial portfolio value.
        transaction_cost_pct: Transaction cost as percentage.
        reward_scaling: Scaling factor for rewards.
        state_space_dim: Dimension of observation space (auto-calculated if None).
        action_space_dim: Dimension of action space (defaults to stock_dim).
        tech_indicators: List of technical indicators to use.
        use_turbulence: Whether to include turbulence index.
        turbulence_threshold: Threshold for turbulence-based risk management.
        algorithm: RL algorithm to use.
        total_timesteps: Total training timesteps.
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size for training.
        buffer_size: Replay buffer size (for off-policy algorithms).
        gamma: Discount factor.
        ent_coef: Entropy coefficient (for PPO/A2C).
        n_steps: Number of steps per update (for on-policy algorithms).
        reward_type: Type of reward function to use.
        model_path: Path to save/load models.
        model_name: Name for the model.
        enable_tensorboard: Whether to enable TensorBoard logging.
        log_dir: Directory for logs.
        seed: Random seed for reproducibility.
        device: Device to use (auto, cpu, cuda).
        pair_whitelist: List of trading pairs.
        timeframe: Trading timeframe.
    """

    # Environment configuration
    stock_dim: int = 30
    hmax: int = 100
    initial_amount: float = 1_000_000.0
    transaction_cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    state_space_dim: int | None = None
    action_space_dim: int | None = None

    # Feature configuration
    tech_indicators: tuple[str, ...] = (
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    )
    use_turbulence: bool = True
    turbulence_threshold: float = 140.0

    # Training configuration
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    gamma: float = 0.99
    ent_coef: float = 0.01
    n_steps: int = 2048

    # Reward configuration
    reward_type: RewardType = RewardType.SHARPE

    # Model management
    model_path: Path | None = None
    model_name: str = "finrl_model"
    enable_tensorboard: bool = True
    log_dir: Path | None = None
    seed: int | None = None
    device: str = "auto"

    # Trading configuration
    pair_whitelist: tuple[str, ...] = ()
    timeframe: str = "1d"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> FinRLAdapterConfig:
        """
        Create configuration from dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            FinRLAdapterConfig instance.
        """
        # Convert lists to tuples for frozen struct
        if "tech_indicators" in config and isinstance(
            config["tech_indicators"], list
        ):
            config["tech_indicators"] = tuple(config["tech_indicators"])
        if "pair_whitelist" in config and isinstance(
            config["pair_whitelist"], list
        ):
            config["pair_whitelist"] = tuple(config["pair_whitelist"])

        # Convert string enums
        if "algorithm" in config and isinstance(config["algorithm"], str):
            config["algorithm"] = RLAlgorithm(config["algorithm"])
        if "reward_type" in config and isinstance(config["reward_type"], str):
            config["reward_type"] = RewardType(config["reward_type"])

        # Convert paths
        if "model_path" in config and config["model_path"] is not None:
            config["model_path"] = Path(config["model_path"])
        if "log_dir" in config and config["log_dir"] is not None:
            config["log_dir"] = Path(config["log_dir"])

        # Filter to known fields
        known_fields = {f.name for f in msgspec.structs.fields(cls)}
        filtered = {k: v for k, v in config.items() if k in known_fields}

        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field in msgspec.structs.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Path):
                result[field.name] = str(value)
            elif isinstance(value, Enum):
                result[field.name] = value.value
            elif isinstance(value, tuple):
                result[field.name] = list(value)
            else:
                result[field.name] = value
        return result

    def get_state_space_dim(self) -> int:
        """
        Calculate the state/observation space dimension.

        State includes:
        - Balance (1)
        - Stock prices (stock_dim)
        - Stock holdings (stock_dim)
        - Technical indicators (stock_dim * num_indicators)
        - Turbulence (1 if enabled)
        """
        if self.state_space_dim is not None:
            return self.state_space_dim

        dim = 1  # Balance
        dim += self.stock_dim  # Stock prices
        dim += self.stock_dim  # Holdings
        dim += self.stock_dim * len(self.tech_indicators)  # Indicators
        if self.use_turbulence:
            dim += 1  # Turbulence index
        return dim

    def get_action_space_dim(self) -> int:
        """Get action space dimension (number of stocks)."""
        if self.action_space_dim is not None:
            return self.action_space_dim
        return self.stock_dim

    def get_algorithm_kwargs(self) -> dict[str, Any]:
        """
        Get algorithm-specific keyword arguments.

        Returns:
            Dictionary of kwargs for the SB3 algorithm.
        """
        base_kwargs: dict[str, Any] = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "device": self.device,
            "seed": self.seed,
        }

        if self.algorithm in (RLAlgorithm.PPO, RLAlgorithm.A2C):
            base_kwargs.update(
                {
                    "n_steps": self.n_steps,
                    "batch_size": self.batch_size,
                    "ent_coef": self.ent_coef,
                }
            )
        elif self.algorithm in (RLAlgorithm.SAC, RLAlgorithm.TD3, RLAlgorithm.DDPG):
            base_kwargs.update(
                {
                    "buffer_size": self.buffer_size,
                    "batch_size": self.batch_size,
                }
            )

        # Remove None values
        return {k: v for k, v in base_kwargs.items() if v is not None}


class TrainingConfig(msgspec.Struct, frozen=True, kw_only=True):
    """
    Configuration for model training.

    Attributes:
        n_trials: Number of Optuna trials for hyperparameter tuning.
        study_name: Name for the Optuna study.
        optimization_metric: Metric to optimize.
        early_stopping_patience: Number of evaluations without improvement.
        eval_freq: Evaluation frequency in timesteps.
        n_eval_episodes: Number of episodes for evaluation.
        save_freq: Model checkpoint frequency.
        verbose: Verbosity level.
    """

    n_trials: int = 100
    study_name: str = "finrl_optimization"
    optimization_metric: str = "sharpe_ratio"
    early_stopping_patience: int = 10
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    save_freq: int = 10_000
    verbose: int = 1


class InferenceConfig(msgspec.Struct, frozen=True, kw_only=True):
    """
    Configuration for model inference.

    Attributes:
        deterministic: Whether to use deterministic actions.
        risk_threshold: Turbulence threshold for risk management.
        max_position_pct: Maximum position size as percentage.
        enable_shadow_mode: Run predictions without execution.
    """

    deterministic: bool = True
    risk_threshold: float = 140.0
    max_position_pct: float = 0.2
    enable_shadow_mode: bool = False

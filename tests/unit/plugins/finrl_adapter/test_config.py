"""Tests for FinRL adapter configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from libra.plugins.finrl_adapter.config import (
    FinRLAdapterConfig,
    InferenceConfig,
    RewardType,
    RLAlgorithm,
    TrainingConfig,
)


class TestFinRLAdapterConfig:
    """Tests for FinRLAdapterConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FinRLAdapterConfig()

        assert config.stock_dim == 30
        assert config.hmax == 100
        assert config.initial_amount == 1_000_000.0
        assert config.algorithm == RLAlgorithm.PPO
        assert config.reward_type == RewardType.SHARPE
        assert config.total_timesteps == 100_000

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config_dict = {
            "stock_dim": 10,
            "algorithm": "sac",
            "total_timesteps": 50000,
            "pair_whitelist": ["AAPL", "GOOGL"],
            "tech_indicators": ["macd", "rsi"],
        }

        config = FinRLAdapterConfig.from_dict(config_dict)

        assert config.stock_dim == 10
        assert config.algorithm == RLAlgorithm.SAC
        assert config.total_timesteps == 50000
        assert config.pair_whitelist == ("AAPL", "GOOGL")
        assert config.tech_indicators == ("macd", "rsi")

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = FinRLAdapterConfig(
            stock_dim=15,
            algorithm=RLAlgorithm.TD3,
        )

        result = config.to_dict()

        assert result["stock_dim"] == 15
        assert result["algorithm"] == "td3"
        assert isinstance(result["tech_indicators"], list)

    def test_from_dict_with_path(self) -> None:
        """Test config with path fields."""
        config_dict = {
            "model_path": "/tmp/models",
            "log_dir": "/tmp/logs",
        }

        config = FinRLAdapterConfig.from_dict(config_dict)

        assert config.model_path == Path("/tmp/models")
        assert config.log_dir == Path("/tmp/logs")

    def test_get_state_space_dim(self) -> None:
        """Test state space dimension calculation."""
        config = FinRLAdapterConfig(
            stock_dim=5,
            tech_indicators=("macd", "rsi"),
            use_turbulence=True,
        )

        # 1 (balance) + 5 (prices) + 5 (holdings) + 5*2 (indicators) + 1 (turbulence)
        expected = 1 + 5 + 5 + 10 + 1
        assert config.get_state_space_dim() == expected

    def test_get_action_space_dim(self) -> None:
        """Test action space dimension."""
        config = FinRLAdapterConfig(stock_dim=10)
        assert config.get_action_space_dim() == 10

    def test_get_algorithm_kwargs_ppo(self) -> None:
        """Test algorithm kwargs for PPO."""
        config = FinRLAdapterConfig(
            algorithm=RLAlgorithm.PPO,
            learning_rate=1e-4,
            batch_size=128,
        )

        kwargs = config.get_algorithm_kwargs()

        assert kwargs["learning_rate"] == 1e-4
        assert kwargs["batch_size"] == 128
        assert "n_steps" in kwargs
        assert "ent_coef" in kwargs

    def test_get_algorithm_kwargs_sac(self) -> None:
        """Test algorithm kwargs for SAC."""
        config = FinRLAdapterConfig(
            algorithm=RLAlgorithm.SAC,
            buffer_size=50000,
        )

        kwargs = config.get_algorithm_kwargs()

        assert kwargs["buffer_size"] == 50000
        assert "n_steps" not in kwargs  # SAC doesn't use n_steps


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self) -> None:
        """Test default training config."""
        config = TrainingConfig()

        assert config.n_trials == 100
        assert config.optimization_metric == "sharpe_ratio"
        assert config.eval_freq == 10_000

    def test_custom_config(self) -> None:
        """Test custom training config."""
        config = TrainingConfig(
            n_trials=50,
            study_name="my_study",
            early_stopping_patience=5,
        )

        assert config.n_trials == 50
        assert config.study_name == "my_study"
        assert config.early_stopping_patience == 5


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_config(self) -> None:
        """Test default inference config."""
        config = InferenceConfig()

        assert config.deterministic is True
        assert config.risk_threshold == 140.0
        assert config.enable_shadow_mode is False

    def test_shadow_mode(self) -> None:
        """Test shadow mode configuration."""
        config = InferenceConfig(enable_shadow_mode=True)

        assert config.enable_shadow_mode is True


class TestRLAlgorithm:
    """Tests for RLAlgorithm enum."""

    def test_algorithm_values(self) -> None:
        """Test algorithm enum values."""
        assert RLAlgorithm.PPO.value == "ppo"
        assert RLAlgorithm.SAC.value == "sac"
        assert RLAlgorithm.A2C.value == "a2c"
        assert RLAlgorithm.TD3.value == "td3"
        assert RLAlgorithm.DDPG.value == "ddpg"

    def test_algorithm_from_string(self) -> None:
        """Test creating algorithm from string."""
        algo = RLAlgorithm("ppo")
        assert algo == RLAlgorithm.PPO


class TestRewardType:
    """Tests for RewardType enum."""

    def test_reward_type_values(self) -> None:
        """Test reward type enum values."""
        assert RewardType.BASIC.value == "basic"
        assert RewardType.SHARPE.value == "sharpe"
        assert RewardType.DIFFERENTIAL_SHARPE.value == "differential_sharpe"
        assert RewardType.COMPOSITE.value == "composite"

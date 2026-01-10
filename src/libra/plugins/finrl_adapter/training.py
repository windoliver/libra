"""
Training Pipeline for FinRL Adapter.

Provides training functionality for RL trading agents using Stable-Baselines3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from libra.plugins.finrl_adapter.config import (
    FinRLAdapterConfig,
    RLAlgorithm,
    TrainingConfig,
)
from libra.plugins.finrl_adapter.environment import TradingEnvironment

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Try to import stable-baselines3
try:
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# Try to import optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class TrainingProgressCallback(BaseCallback if SB3_AVAILABLE else object):  # type: ignore[misc]
    """
    Callback for tracking training progress.

    Logs training metrics and tracks best performance.
    """

    def __init__(
        self,
        check_freq: int = 1000,
        verbose: int = 1,
    ) -> None:
        """
        Initialize the callback.

        Args:
            check_freq: Frequency of logging (in timesteps).
            verbose: Verbosity level.
        """
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = float("-inf")
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.check_freq == 0:
            # Log current progress
            if len(self.episode_rewards) > 0:
                mean_reward = sum(self.episode_rewards[-100:]) / min(
                    len(self.episode_rewards), 100
                )
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                logger.info(
                    "Step %d: Mean reward (last 100): %.2f, Best: %.2f",
                    self.n_calls,
                    mean_reward,
                    self.best_mean_reward,
                )
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        # Track episode rewards from info
        if hasattr(self, "locals") and "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])


class RLTrainer:
    """
    Trainer for RL trading agents.

    Provides:
    - Training with various SB3 algorithms
    - Hyperparameter tuning with Optuna
    - Model checkpointing and evaluation
    - TensorBoard logging
    """

    # Algorithm mapping
    ALGORITHMS = {
        RLAlgorithm.PPO: "PPO",
        RLAlgorithm.A2C: "A2C",
        RLAlgorithm.SAC: "SAC",
        RLAlgorithm.TD3: "TD3",
        RLAlgorithm.DDPG: "DDPG",
    }

    def __init__(
        self,
        config: FinRLAdapterConfig,
        training_config: TrainingConfig | None = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            config: FinRL adapter configuration.
            training_config: Training-specific configuration.
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for training. "
                "Install with: pip install stable-baselines3"
            )

        self.config = config
        self.training_config = training_config or TrainingConfig()
        self.model: Any = None
        self.env: Any = None
        self.vec_env: Any = None

    def create_env(
        self,
        df: Any,
        mode: str = "train",
    ) -> Any:
        """
        Create a vectorized training environment.

        Args:
            df: DataFrame with OHLCV and indicator data.
            mode: Environment mode ('train', 'test', 'trade').

        Returns:
            Vectorized and normalized environment.
        """
        # Create base environment
        env = TradingEnvironment(df=df, config=self.config, mode=mode)

        # Wrap in vectorized environment
        vec_env = DummyVecEnv([lambda: env])

        # Add normalization wrapper
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        return vec_env

    def get_algorithm_class(self) -> type:
        """Get the SB3 algorithm class."""
        algorithm_map = {
            RLAlgorithm.PPO: PPO,
            RLAlgorithm.A2C: A2C,
            RLAlgorithm.SAC: SAC,
            RLAlgorithm.TD3: TD3,
            RLAlgorithm.DDPG: DDPG,
        }
        return algorithm_map[self.config.algorithm]

    def train(
        self,
        train_df: Any,
        eval_df: Any | None = None,
        callbacks: list[Any] | None = None,
    ) -> Any:
        """
        Train an RL agent.

        Args:
            train_df: Training DataFrame.
            eval_df: Evaluation DataFrame (optional).
            callbacks: Additional callbacks.

        Returns:
            Trained model.
        """
        logger.info(
            "Starting training with %s for %d timesteps",
            self.config.algorithm.value,
            self.config.total_timesteps,
        )

        # Create training environment
        self.vec_env = self.create_env(train_df, mode="train")

        # Get algorithm kwargs
        algo_kwargs = self.config.get_algorithm_kwargs()

        # Add TensorBoard logging
        tensorboard_log = None
        if self.config.enable_tensorboard and self.config.log_dir:
            tensorboard_log = str(self.config.log_dir / "tensorboard")
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

        # Create model
        AlgoClass = self.get_algorithm_class()
        self.model = AlgoClass(
            "MlpPolicy",
            self.vec_env,
            tensorboard_log=tensorboard_log,
            verbose=self.training_config.verbose,
            **algo_kwargs,
        )

        # Setup callbacks
        callback_list = []

        # Progress callback
        progress_callback = TrainingProgressCallback(
            check_freq=self.training_config.eval_freq,
            verbose=self.training_config.verbose,
        )
        callback_list.append(progress_callback)

        # Checkpoint callback
        if self.config.model_path:
            checkpoint_path = self.config.model_path / "checkpoints"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=self.training_config.save_freq,
                save_path=str(checkpoint_path),
                name_prefix=self.config.model_name,
            )
            callback_list.append(checkpoint_callback)

        # Evaluation callback
        if eval_df is not None:
            eval_env = self.create_env(eval_df, mode="test")
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.config.model_path) if self.config.model_path else None,
                log_path=str(self.config.log_dir) if self.config.log_dir else None,
                eval_freq=self.training_config.eval_freq,
                n_eval_episodes=self.training_config.n_eval_episodes,
                deterministic=True,
            )
            callback_list.append(eval_callback)

        # Add user callbacks
        if callbacks:
            callback_list.extend(callbacks)

        # Train
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=CallbackList(callback_list) if callback_list else None,
            progress_bar=self.training_config.verbose > 0,
        )

        logger.info("Training complete")
        return self.model

    def save(self, path: Path | str | None = None) -> Path:
        """
        Save the trained model.

        Args:
            path: Path to save the model. Uses config path if not specified.

        Returns:
            Path where the model was saved.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")

        if path is None:
            if self.config.model_path is None:
                raise ValueError("No save path specified")
            path = self.config.model_path / f"{self.config.model_name}.zip"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

        # Save VecNormalize stats
        if self.vec_env is not None:
            stats_path = path.with_suffix(".pkl")
            self.vec_env.save(str(stats_path))
            logger.info("Saved normalization stats to %s", stats_path)

        logger.info("Model saved to %s", path)
        return path

    def load(self, path: Path | str) -> Any:
        """
        Load a trained model.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded model.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        AlgoClass = self.get_algorithm_class()
        self.model = AlgoClass.load(str(path))

        # Load VecNormalize stats if available
        stats_path = path.with_suffix(".pkl")
        if stats_path.exists():
            # Need to create env to load stats
            logger.info("Found normalization stats at %s", stats_path)

        logger.info("Model loaded from %s", path)
        return self.model


class HyperparameterTuner:
    """
    Hyperparameter tuner using Optuna.

    Tunes RL algorithm hyperparameters to maximize trading performance.
    """

    # Hyperparameter search spaces by algorithm
    SEARCH_SPACES = {
        RLAlgorithm.PPO: {
            "learning_rate": (1e-5, 1e-2, "log"),
            "n_steps": [256, 512, 1024, 2048],
            "batch_size": [32, 64, 128, 256],
            "gamma": (0.9, 0.9999, "log"),
            "ent_coef": (1e-8, 0.1, "log"),
            "clip_range": (0.1, 0.4),
            "n_epochs": [3, 5, 10, 20],
        },
        RLAlgorithm.SAC: {
            "learning_rate": (1e-5, 1e-2, "log"),
            "buffer_size": [10000, 50000, 100000, 500000],
            "batch_size": [64, 128, 256, 512],
            "gamma": (0.9, 0.9999, "log"),
            "tau": (0.001, 0.1, "log"),
            "train_freq": [1, 4, 8, 16],
        },
        RLAlgorithm.A2C: {
            "learning_rate": (1e-5, 1e-2, "log"),
            "n_steps": [5, 16, 32, 64],
            "gamma": (0.9, 0.9999, "log"),
            "ent_coef": (1e-8, 0.1, "log"),
        },
    }

    def __init__(
        self,
        base_config: FinRLAdapterConfig,
        train_df: Any,
        eval_df: Any,
        training_config: TrainingConfig | None = None,
    ) -> None:
        """
        Initialize the tuner.

        Args:
            base_config: Base FinRL configuration.
            train_df: Training DataFrame.
            eval_df: Evaluation DataFrame.
            training_config: Training configuration.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )

        self.base_config = base_config
        self.train_df = train_df
        self.eval_df = eval_df
        self.training_config = training_config or TrainingConfig()
        self.best_params: dict[str, Any] = {}
        self.best_value: float = float("-inf")

    def _sample_params(self, trial: Any) -> dict[str, Any]:
        """Sample hyperparameters for a trial."""
        search_space = self.SEARCH_SPACES.get(
            self.base_config.algorithm,
            self.SEARCH_SPACES[RLAlgorithm.PPO],
        )

        params: dict[str, Any] = {}
        for name, space in search_space.items():
            if isinstance(space, list):
                params[name] = trial.suggest_categorical(name, space)
            elif isinstance(space, tuple):
                if len(space) == 3 and space[2] == "log":
                    params[name] = trial.suggest_float(
                        name, space[0], space[1], log=True
                    )
                else:
                    params[name] = trial.suggest_float(name, space[0], space[1])

        return params

    def _objective(self, trial: Any) -> float:
        """Objective function for Optuna."""
        # Sample hyperparameters
        params = self._sample_params(trial)

        # Create config with sampled params
        config_dict = self.base_config.to_dict()
        config_dict.update(params)
        # Reduce timesteps for tuning
        config_dict["total_timesteps"] = min(
            self.base_config.total_timesteps // 5,
            50000,
        )
        config = FinRLAdapterConfig.from_dict(config_dict)

        try:
            # Train with sampled parameters
            trainer = RLTrainer(config, self.training_config)
            model = trainer.train(self.train_df)

            # Evaluate
            eval_env = trainer.create_env(self.eval_df, mode="test")

            # Run evaluation episodes
            episode_rewards = []
            episode_returns = []

            for _ in range(self.training_config.n_eval_episodes):
                obs = eval_env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward[0]

                episode_rewards.append(episode_reward)
                if "terminal_observation" in info[0]:
                    terminal_info = info[0].get("terminal_info", {})
                    if "sharpe_ratio" in terminal_info:
                        episode_returns.append(terminal_info["sharpe_ratio"])

            # Calculate objective (Sharpe ratio or mean reward)
            if episode_returns:
                import numpy as np
                objective_value = float(np.mean(episode_returns))
            else:
                import numpy as np
                objective_value = float(np.mean(episode_rewards))

            # Report intermediate value for pruning
            trial.report(objective_value, 0)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return objective_value

        except Exception as e:
            logger.warning("Trial failed: %s", e)
            return float("-inf")

    def tune(
        self,
        n_trials: int | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            n_trials: Number of trials (defaults to config).
            timeout: Maximum time in seconds.

        Returns:
            Best hyperparameters found.
        """
        n_trials = n_trials or self.training_config.n_trials

        logger.info(
            "Starting hyperparameter tuning: %d trials for %s",
            n_trials,
            self.base_config.algorithm.value,
        )

        study = optuna.create_study(
            study_name=self.training_config.study_name,
            direction="maximize",
            sampler=TPESampler(seed=self.base_config.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        self.best_value = study.best_value

        logger.info("Best params: %s", self.best_params)
        logger.info("Best value: %.4f", self.best_value)

        return self.best_params

    def get_best_config(self) -> FinRLAdapterConfig:
        """
        Get configuration with best hyperparameters.

        Returns:
            FinRLAdapterConfig with best params.
        """
        if not self.best_params:
            raise RuntimeError("No tuning results. Run tune() first.")

        config_dict = self.base_config.to_dict()
        config_dict.update(self.best_params)
        return FinRLAdapterConfig.from_dict(config_dict)


def train_agent(
    train_df: Any,
    config: FinRLAdapterConfig | None = None,
    eval_df: Any | None = None,
    save_path: Path | str | None = None,
) -> Any:
    """
    Convenience function to train an RL agent.

    Args:
        train_df: Training DataFrame with OHLCV and indicators.
        config: FinRL adapter configuration.
        eval_df: Evaluation DataFrame (optional).
        save_path: Path to save the trained model.

    Returns:
        Trained model.
    """
    config = config or FinRLAdapterConfig()
    trainer = RLTrainer(config)
    model = trainer.train(train_df, eval_df)

    if save_path:
        trainer.save(save_path)

    return model

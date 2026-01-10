"""
Trading Environment for FinRL Adapter.

Gymnasium-compatible environment for training RL trading agents.
Based on FinRL's StockTradingEnv with enhancements for LIBRA.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from libra.plugins.finrl_adapter.config import FinRLAdapterConfig, RewardType

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Try to import gymnasium, fall back to gym
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces

        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        gym = None  # type: ignore[assignment]
        spaces = None  # type: ignore[assignment]


class TradingEnvironment:
    """
    Gymnasium-compatible trading environment for RL agents.

    This environment simulates stock trading with:
    - Continuous action space for portfolio allocation
    - Configurable observation space with technical indicators
    - Multiple reward function options
    - Transaction costs and slippage modeling
    - Turbulence-based risk management

    The environment follows the standard Gymnasium interface:
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info

    Observation Space:
        [balance, stock_prices, holdings, technical_indicators, turbulence]

    Action Space:
        Continuous values in [-1, 1] for each stock, representing:
        - Negative: Sell proportion of holdings
        - Zero: Hold
        - Positive: Buy proportion of available cash
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: Any,  # pandas or polars DataFrame
        config: FinRLAdapterConfig | None = None,
        reward_function: Any | None = None,
        mode: str = "train",
    ) -> None:
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with OHLCV data and technical indicators.
                Expected columns: date, tic (ticker), open, high, low, close, volume,
                plus any configured technical indicators.
            config: FinRL adapter configuration.
            reward_function: Custom reward function (optional).
            mode: Environment mode ('train', 'test', 'trade').
        """
        if not GYM_AVAILABLE:
            raise ImportError(
                "gymnasium or gym is required. Install with: pip install gymnasium"
            )

        self.config = config or FinRLAdapterConfig()
        self.mode = mode
        self.reward_function = reward_function

        # Convert to numpy for faster operations
        self._setup_data(df)

        # Environment dimensions
        self.stock_dim = self.config.stock_dim
        self.hmax = self.config.hmax
        self.initial_amount = self.config.initial_amount
        self.transaction_cost_pct = self.config.transaction_cost_pct
        self.reward_scaling = self.config.reward_scaling

        # Define spaces
        self.state_space_dim = self.config.get_state_space_dim()
        self.action_space_dim = self.config.get_action_space_dim()

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_space_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space_dim,),
            dtype=np.float32,
        )

        # State tracking
        self.day = 0
        self.terminal = False
        self.data = self._get_day_data(self.day)

        # Portfolio state
        self.state: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.balance = self.initial_amount
        self.holdings: NDArray[np.float32] = np.zeros(self.stock_dim, dtype=np.float32)
        self.cost = 0.0
        self.trades = 0

        # History tracking
        self.asset_memory: list[float] = [self.initial_amount]
        self.rewards_memory: list[float] = []
        self.actions_memory: list[NDArray[np.float32]] = []
        self.date_memory: list[Any] = []
        self.state_memory: list[NDArray[np.float32]] = []

        # For Sharpe calculation
        self.returns_memory: list[float] = []

        # Turbulence
        self.turbulence = 0.0
        self.turbulence_threshold = self.config.turbulence_threshold

        # Initialize state
        self._init_state()

    def _setup_data(self, df: Any) -> None:
        """Convert DataFrame to numpy arrays for faster access."""
        # Convert polars to pandas if needed
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        self.df = df

        # Get unique dates and tickers
        self.dates = df["date"].unique()
        self.n_days = len(self.dates)

        # Get tickers
        if "tic" in df.columns:
            self.tickers = sorted(df["tic"].unique())
        else:
            self.tickers = [f"STOCK_{i}" for i in range(self.config.stock_dim)]

        # Extract technical indicator columns
        base_cols = {"date", "tic", "open", "high", "low", "close", "volume"}
        self.indicator_cols = [c for c in df.columns if c not in base_cols]

        logger.debug(
            "Environment setup: %d days, %d tickers, %d indicators",
            self.n_days,
            len(self.tickers),
            len(self.indicator_cols),
        )

    def _get_day_data(self, day: int) -> Any:
        """Get data for a specific day."""
        if day >= self.n_days:
            return None
        date = self.dates[day]
        return self.df[self.df["date"] == date]

    def _init_state(self) -> None:
        """Initialize the state vector."""
        if self.data is None or len(self.data) == 0:
            self.state = np.zeros(self.state_space_dim, dtype=np.float32)
            return

        # Build state vector
        state_parts = []

        # Balance (normalized)
        state_parts.append(self.balance / self.initial_amount)

        # Stock prices
        prices = self.data["close"].values
        if len(prices) < self.stock_dim:
            prices = np.pad(prices, (0, self.stock_dim - len(prices)))
        state_parts.extend(prices[: self.stock_dim])

        # Holdings
        state_parts.extend(self.holdings)

        # Technical indicators
        for indicator in self.config.tech_indicators:
            if indicator in self.data.columns:
                vals = self.data[indicator].values
                if len(vals) < self.stock_dim:
                    vals = np.pad(vals, (0, self.stock_dim - len(vals)))
                state_parts.extend(vals[: self.stock_dim])
            else:
                state_parts.extend([0.0] * self.stock_dim)

        # Turbulence
        if self.config.use_turbulence:
            if "turbulence" in self.data.columns:
                self.turbulence = float(self.data["turbulence"].iloc[0])
            state_parts.append(self.turbulence)

        self.state = np.array(state_parts, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info dict).
        """
        if seed is not None:
            np.random.seed(seed)

        self.day = 0
        self.data = self._get_day_data(self.day)
        self.terminal = False

        # Reset portfolio
        self.balance = self.initial_amount
        self.holdings = np.zeros(self.stock_dim, dtype=np.float32)
        self.cost = 0.0
        self.trades = 0

        # Reset history
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.dates[0] if self.n_days > 0 else None]
        self.state_memory = []
        self.returns_memory = []

        # Reset turbulence
        self.turbulence = 0.0

        # Initialize state
        self._init_state()

        info = {
            "balance": self.balance,
            "holdings": self.holdings.copy(),
            "portfolio_value": self._get_portfolio_value(),
        }

        return self.state.copy(), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Array of actions for each stock in [-1, 1].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self.terminal = self.day >= self.n_days - 1

        if self.terminal:
            # Episode ended
            info = self._get_terminal_info()
            return self.state.copy(), 0.0, True, False, info

        # Record previous portfolio value
        prev_portfolio_value = self._get_portfolio_value()

        # Execute trades
        self._execute_trades(action)

        # Move to next day
        self.day += 1
        self.data = self._get_day_data(self.day)

        # Update state
        self._init_state()

        # Calculate new portfolio value
        new_portfolio_value = self._get_portfolio_value()

        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, new_portfolio_value)

        # Store history
        self.asset_memory.append(new_portfolio_value)
        self.rewards_memory.append(reward)
        self.actions_memory.append(action.copy())
        self.state_memory.append(self.state.copy())
        if self.n_days > 0 and self.day < self.n_days:
            self.date_memory.append(self.dates[self.day])

        # Calculate return for Sharpe
        if prev_portfolio_value > 0:
            ret = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.returns_memory.append(ret)

        # Check turbulence-based termination
        truncated = False
        if (
            self.config.use_turbulence
            and self.turbulence > self.turbulence_threshold
            and self.mode == "trade"
        ):
            # In trading mode, high turbulence triggers early stop
            truncated = True
            logger.warning(
                "High turbulence detected: %.2f > %.2f",
                self.turbulence,
                self.turbulence_threshold,
            )

        info = {
            "balance": self.balance,
            "holdings": self.holdings.copy(),
            "portfolio_value": new_portfolio_value,
            "cost": self.cost,
            "trades": self.trades,
            "turbulence": self.turbulence,
            "day": self.day,
        }

        return self.state.copy(), reward, self.terminal, truncated, info

    def _execute_trades(self, action: NDArray[np.float32]) -> None:
        """
        Execute trading actions.

        Args:
            action: Normalized actions in [-1, 1] for each stock.
        """
        if self.data is None or len(self.data) == 0:
            return

        # Get current prices
        prices = self.data["close"].values
        if len(prices) < self.stock_dim:
            prices = np.pad(prices, (0, self.stock_dim - len(prices)), constant_values=1)
        prices = prices[: self.stock_dim]

        # Denormalize actions to share amounts
        # action > 0: buy, action < 0: sell
        actions = action * self.hmax

        for i in range(self.stock_dim):
            price = prices[i]
            if price <= 0:
                continue

            shares_to_trade = int(actions[i])

            if shares_to_trade > 0:
                # Buy
                max_affordable = int(self.balance / (price * (1 + self.transaction_cost_pct)))
                shares_to_buy = min(shares_to_trade, max_affordable)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.transaction_cost_pct)
                    self.balance -= cost
                    self.holdings[i] += shares_to_buy
                    self.cost += shares_to_buy * price * self.transaction_cost_pct
                    self.trades += 1

            elif shares_to_trade < 0:
                # Sell
                shares_to_sell = min(abs(shares_to_trade), int(self.holdings[i]))
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price * (1 - self.transaction_cost_pct)
                    self.balance += proceeds
                    self.holdings[i] -= shares_to_sell
                    self.cost += shares_to_sell * price * self.transaction_cost_pct
                    self.trades += 1

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        if self.data is None or len(self.data) == 0:
            return self.balance

        prices = self.data["close"].values
        if len(prices) < self.stock_dim:
            prices = np.pad(prices, (0, self.stock_dim - len(prices)), constant_values=0)
        prices = prices[: self.stock_dim]

        holdings_value = float(np.sum(self.holdings * prices))
        return self.balance + holdings_value

    def _calculate_reward(
        self, prev_value: float, new_value: float
    ) -> float:
        """
        Calculate reward based on configured reward function.

        Args:
            prev_value: Previous portfolio value.
            new_value: New portfolio value.

        Returns:
            Scaled reward value.
        """
        if self.reward_function is not None:
            return float(self.reward_function(prev_value, new_value, self))

        reward_type = self.config.reward_type

        if reward_type == RewardType.BASIC:
            # Simple value change
            reward = new_value - prev_value

        elif reward_type == RewardType.SHARPE:
            # Sharpe-based reward using recent returns
            reward = self._calculate_sharpe_reward(prev_value, new_value)

        elif reward_type == RewardType.DIFFERENTIAL_SHARPE:
            # Differential Sharpe ratio for online learning
            reward = self._calculate_differential_sharpe(prev_value, new_value)

        elif reward_type == RewardType.COMPOSITE:
            # Multi-objective composite reward
            reward = self._calculate_composite_reward(prev_value, new_value)

        else:
            reward = new_value - prev_value

        return reward * self.reward_scaling

    def _calculate_sharpe_reward(
        self, prev_value: float, new_value: float
    ) -> float:
        """Calculate Sharpe ratio-based reward."""
        if prev_value <= 0:
            return 0.0

        ret = (new_value - prev_value) / prev_value

        # Use rolling window for Sharpe approximation
        window = 20
        if len(self.returns_memory) >= window:
            recent_returns = self.returns_memory[-window:] + [ret]
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns)
            if std_ret > 0:
                sharpe = mean_ret / std_ret * np.sqrt(252)  # Annualized
                return float(sharpe)

        return ret * 100  # Scale up for early training

    def _calculate_differential_sharpe(
        self, prev_value: float, new_value: float
    ) -> float:
        """
        Calculate differential Sharpe ratio for online learning.

        Based on Moody & Saffell's differential Sharpe ratio.
        """
        if prev_value <= 0:
            return 0.0

        ret = (new_value - prev_value) / prev_value

        # Exponential moving averages
        eta = 0.01  # Adaptation rate

        if not hasattr(self, "_A"):
            self._A = 0.0
            self._B = 0.0

        # Update exponential moving statistics
        delta_A = ret - self._A
        delta_B = ret * ret - self._B

        self._A += eta * delta_A
        self._B += eta * delta_B

        # Differential Sharpe ratio
        denom = (self._B - self._A * self._A) ** 0.5
        if denom > 1e-8:
            dSR = (self._B * delta_A - 0.5 * self._A * delta_B) / (denom ** 3)
            return float(dSR)

        return ret

    def _calculate_composite_reward(
        self, prev_value: float, new_value: float
    ) -> float:
        """
        Calculate multi-objective composite reward.

        Combines: return, downside risk, and portfolio concentration.
        """
        if prev_value <= 0:
            return 0.0

        ret = (new_value - prev_value) / prev_value

        # Downside risk penalty
        downside_penalty = 0.0
        if ret < 0:
            downside_penalty = ret * 2  # Double penalty for losses

        # Concentration penalty (prefer diversification)
        if np.sum(self.holdings) > 0:
            weights = self.holdings / np.sum(self.holdings)
            concentration = np.sum(weights ** 2)  # Herfindahl index
            concentration_penalty = concentration * 0.01
        else:
            concentration_penalty = 0.0

        return ret + downside_penalty - concentration_penalty

    def _get_terminal_info(self) -> dict[str, Any]:
        """Get information at episode termination."""
        # Calculate final metrics
        total_return = (
            self._get_portfolio_value() - self.initial_amount
        ) / self.initial_amount

        # Sharpe ratio
        if len(self.returns_memory) > 1:
            mean_ret = np.mean(self.returns_memory)
            std_ret = np.std(self.returns_memory)
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe = 0.0

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()

        return {
            "balance": self.balance,
            "holdings": self.holdings.copy(),
            "portfolio_value": self._get_portfolio_value(),
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_cost": self.cost,
            "total_trades": self.trades,
            "asset_memory": self.asset_memory.copy(),
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from asset history."""
        if len(self.asset_memory) < 2:
            return 0.0

        values = np.array(self.asset_memory)
        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max
        return float(np.max(drawdowns))

    def render(self, mode: str = "human") -> None:
        """Render the environment (print portfolio status)."""
        portfolio_value = self._get_portfolio_value()
        print(f"Day {self.day}: Balance=${self.balance:.2f}, "
              f"Holdings Value=${portfolio_value - self.balance:.2f}, "
              f"Total=${portfolio_value:.2f}")

    def get_sb3_env(self) -> Any:
        """
        Get environment wrapped for Stable-Baselines3.

        Returns:
            DummyVecEnv wrapped environment.
        """
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            return DummyVecEnv([lambda: self])
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )


def make_trading_env(
    df: Any,
    config: FinRLAdapterConfig | None = None,
    **kwargs: Any,
) -> TradingEnvironment:
    """
    Factory function to create a trading environment.

    Args:
        df: DataFrame with OHLCV and indicator data.
        config: FinRL adapter configuration.
        **kwargs: Additional arguments passed to TradingEnvironment.

    Returns:
        Configured TradingEnvironment instance.
    """
    return TradingEnvironment(df=df, config=config, **kwargs)

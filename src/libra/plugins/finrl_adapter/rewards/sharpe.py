"""
Sharpe Ratio Based Reward Functions for FinRL Adapter.

Provides risk-adjusted reward functions for RL trading.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from libra.plugins.finrl_adapter.rewards.base import RewardFunction

if TYPE_CHECKING:
    pass

# Try to import numpy
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]


class SharpeReward(RewardFunction):
    """
    Sharpe ratio-based reward function.

    Uses a rolling window to estimate the Sharpe ratio of recent returns.
    This encourages agents to maximize risk-adjusted returns.

    The Sharpe ratio is calculated as:
        Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(annualization)

    This implementation uses online updates for efficiency.
    """

    def __init__(
        self,
        window_size: int = 20,
        risk_free_rate: float = 0.0,
        annualization: int = 252,
        scaling: float = 1.0,
    ) -> None:
        """
        Initialize the Sharpe reward function.

        Args:
            window_size: Rolling window size for Sharpe calculation.
            risk_free_rate: Daily risk-free rate.
            annualization: Annualization factor (252 for daily trading).
            scaling: Scaling factor for the reward.
        """
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.annualization = annualization
        self.scaling = scaling

        # State for rolling calculation
        self._returns: list[float] = []

    def __call__(
        self,
        prev_value: float,
        new_value: float,
        env: Any,
    ) -> float:
        """Calculate Sharpe ratio-based reward."""
        if not NUMPY_AVAILABLE:
            # Fallback to simple return
            if prev_value <= 0:
                return 0.0
            return (new_value - prev_value) / prev_value * self.scaling

        # Calculate return
        if prev_value <= 0:
            return 0.0
        ret = (new_value - prev_value) / prev_value

        # Update rolling window
        self._returns.append(ret)
        if len(self._returns) > self.window_size:
            self._returns.pop(0)

        # Calculate Sharpe ratio
        if len(self._returns) < 2:
            return ret * self.scaling * 100  # Early episodes use scaled return

        returns_array = np.array(self._returns)
        mean_ret = float(np.mean(returns_array)) - self.risk_free_rate
        std_ret = float(np.std(returns_array, ddof=1))

        if std_ret < 1e-8:
            return ret * self.scaling * 100

        sharpe = mean_ret / std_ret * math.sqrt(self.annualization)
        return sharpe * self.scaling

    def reset(self) -> None:
        """Reset rolling window."""
        self._returns = []

    @property
    def name(self) -> str:
        return "sharpe"


class DifferentialSharpeReward(RewardFunction):
    """
    Differential Sharpe ratio reward function.

    Based on Moody & Saffell's differential Sharpe ratio for online learning.
    This provides a gradient estimate of the Sharpe ratio that can be
    computed incrementally without storing a window of returns.

    The differential Sharpe ratio is:
        dSR/dR_t = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / (B_{t-1} - A_{t-1}^2)^(3/2)

    where:
        A_t = exponential moving average of returns
        B_t = exponential moving average of squared returns
    """

    def __init__(
        self,
        eta: float = 0.01,
        scaling: float = 1.0,
    ) -> None:
        """
        Initialize the differential Sharpe reward function.

        Args:
            eta: Adaptation rate for exponential moving averages.
            scaling: Scaling factor for the reward.
        """
        self.eta = eta
        self.scaling = scaling

        # Exponential moving averages
        self._A = 0.0  # EMA of returns
        self._B = 0.0  # EMA of squared returns
        self._initialized = False

    def __call__(
        self,
        prev_value: float,
        new_value: float,
        env: Any,
    ) -> float:
        """Calculate differential Sharpe ratio reward."""
        if prev_value <= 0:
            return 0.0

        ret = (new_value - prev_value) / prev_value

        if not self._initialized:
            self._A = ret
            self._B = ret * ret
            self._initialized = True
            return ret * self.scaling * 100

        # Calculate deltas
        delta_A = ret - self._A
        delta_B = ret * ret - self._B

        # Calculate differential Sharpe ratio
        variance = self._B - self._A * self._A
        if variance > 1e-8:
            denom = variance ** 1.5
            dSR = (self._B * delta_A - 0.5 * self._A * delta_B) / denom
        else:
            dSR = ret

        # Update EMAs
        self._A += self.eta * delta_A
        self._B += self.eta * delta_B

        return dSR * self.scaling

    def reset(self) -> None:
        """Reset exponential moving averages."""
        self._A = 0.0
        self._B = 0.0
        self._initialized = False

    @property
    def name(self) -> str:
        return "differential_sharpe"


class SortinoReward(RewardFunction):
    """
    Sortino ratio-based reward function.

    Similar to Sharpe ratio but only penalizes downside volatility.
    This is more appropriate for trading where upside volatility is desirable.

    Sortino = (mean_return - target) / downside_deviation
    """

    def __init__(
        self,
        window_size: int = 20,
        target_return: float = 0.0,
        annualization: int = 252,
        scaling: float = 1.0,
    ) -> None:
        """
        Initialize the Sortino reward function.

        Args:
            window_size: Rolling window size.
            target_return: Target return (defaults to 0).
            annualization: Annualization factor.
            scaling: Scaling factor for the reward.
        """
        self.window_size = window_size
        self.target_return = target_return
        self.annualization = annualization
        self.scaling = scaling

        self._returns: list[float] = []

    def __call__(
        self,
        prev_value: float,
        new_value: float,
        env: Any,
    ) -> float:
        """Calculate Sortino ratio-based reward."""
        if not NUMPY_AVAILABLE:
            if prev_value <= 0:
                return 0.0
            return (new_value - prev_value) / prev_value * self.scaling

        if prev_value <= 0:
            return 0.0
        ret = (new_value - prev_value) / prev_value

        self._returns.append(ret)
        if len(self._returns) > self.window_size:
            self._returns.pop(0)

        if len(self._returns) < 2:
            return ret * self.scaling * 100

        returns_array = np.array(self._returns)
        mean_ret = float(np.mean(returns_array))

        # Calculate downside deviation
        downside_returns = returns_array[returns_array < self.target_return]
        if len(downside_returns) < 2:
            downside_std = 1e-8
        else:
            downside_std = float(np.std(downside_returns, ddof=1))

        if downside_std < 1e-8:
            downside_std = 1e-8

        sortino = (mean_ret - self.target_return) / downside_std * math.sqrt(self.annualization)
        return sortino * self.scaling

    def reset(self) -> None:
        """Reset rolling window."""
        self._returns = []

    @property
    def name(self) -> str:
        return "sortino"


class CompositeReward(RewardFunction):
    """
    Multi-objective composite reward function.

    Combines multiple reward components:
    - Return component: Rewards positive returns
    - Risk component: Penalizes volatility
    - Drawdown component: Penalizes drawdowns
    - Diversification component: Rewards portfolio diversification

    Reward = w1 * return + w2 * sharpe + w3 * drawdown_penalty + w4 * diversification
    """

    def __init__(
        self,
        return_weight: float = 0.4,
        sharpe_weight: float = 0.3,
        drawdown_weight: float = 0.2,
        diversification_weight: float = 0.1,
        window_size: int = 20,
        scaling: float = 1.0,
    ) -> None:
        """
        Initialize the composite reward function.

        Args:
            return_weight: Weight for return component.
            sharpe_weight: Weight for Sharpe component.
            drawdown_weight: Weight for drawdown penalty.
            diversification_weight: Weight for diversification bonus.
            window_size: Rolling window size for calculations.
            scaling: Scaling factor for the reward.
        """
        self.return_weight = return_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.diversification_weight = diversification_weight
        self.window_size = window_size
        self.scaling = scaling

        self._returns: list[float] = []
        self._max_value = 0.0

    def __call__(
        self,
        prev_value: float,
        new_value: float,
        env: Any,
    ) -> float:
        """Calculate composite reward."""
        if prev_value <= 0:
            return 0.0

        ret = (new_value - prev_value) / prev_value

        # Update max value for drawdown
        self._max_value = max(self._max_value, new_value)

        # Update returns window
        self._returns.append(ret)
        if len(self._returns) > self.window_size:
            self._returns.pop(0)

        # Component 1: Return
        return_component = ret * 100

        # Component 2: Sharpe approximation
        sharpe_component = 0.0
        if NUMPY_AVAILABLE and len(self._returns) >= 2:
            returns_array = np.array(self._returns)
            mean_ret = float(np.mean(returns_array))
            std_ret = float(np.std(returns_array, ddof=1))
            if std_ret > 1e-8:
                sharpe_component = mean_ret / std_ret * math.sqrt(252)

        # Component 3: Drawdown penalty
        drawdown_component = 0.0
        if self._max_value > 0:
            drawdown = (self._max_value - new_value) / self._max_value
            drawdown_component = -drawdown * 100  # Negative penalty

        # Component 4: Diversification bonus
        diversification_component = 0.0
        if hasattr(env, "holdings"):
            holdings = env.holdings
            if NUMPY_AVAILABLE and holdings is not None:
                holdings_sum = np.sum(np.abs(holdings))
                if holdings_sum > 0:
                    weights = np.abs(holdings) / holdings_sum
                    # Use inverse Herfindahl index (higher is more diversified)
                    hhi = np.sum(weights ** 2)
                    diversification_component = (1 - hhi) * 10

        # Combine components
        reward = (
            self.return_weight * return_component
            + self.sharpe_weight * sharpe_component
            + self.drawdown_weight * drawdown_component
            + self.diversification_weight * diversification_component
        )

        return reward * self.scaling

    def reset(self) -> None:
        """Reset internal state."""
        self._returns = []
        self._max_value = 0.0

    @property
    def name(self) -> str:
        return "composite"


def get_reward_function(
    name: str,
    **kwargs: Any,
) -> RewardFunction:
    """
    Factory function to get a reward function by name.

    Args:
        name: Name of the reward function.
        **kwargs: Keyword arguments for the reward function.

    Returns:
        Configured reward function instance.

    Raises:
        ValueError: If reward function name is unknown.
    """
    from libra.plugins.finrl_adapter.rewards.base import BasicReward, LogReturnReward, ReturnReward

    reward_functions: dict[str, type[RewardFunction]] = {
        "basic": BasicReward,
        "return": ReturnReward,
        "log_return": LogReturnReward,
        "sharpe": SharpeReward,
        "differential_sharpe": DifferentialSharpeReward,
        "sortino": SortinoReward,
        "composite": CompositeReward,
    }

    if name not in reward_functions:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(reward_functions.keys())}"
        )

    return reward_functions[name](**kwargs)

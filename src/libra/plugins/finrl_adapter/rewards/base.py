"""
Base Reward Function for FinRL Adapter.

Defines the interface for reward functions used in RL trading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    Reward functions calculate the reward signal for RL agents based on
    portfolio performance and other factors. Different reward functions
    emphasize different aspects of trading performance:

    - Basic: Simple portfolio value change
    - Risk-adjusted: Sharpe ratio, Sortino ratio
    - Multi-objective: Combination of return and risk metrics

    Usage:
        reward_fn = SharpeReward()
        reward = reward_fn(prev_value, new_value, env)
    """

    @abstractmethod
    def __call__(
        self,
        prev_value: float,
        new_value: float,
        env: Any,
    ) -> float:
        """
        Calculate reward for a transition.

        Args:
            prev_value: Portfolio value before action.
            new_value: Portfolio value after action.
            env: Trading environment instance (for access to state).

        Returns:
            Reward value (should be scaled appropriately).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (called at episode start)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the reward function."""
        ...


class BasicReward(RewardFunction):
    """
    Basic reward function using portfolio value change.

    This is the simplest reward function that just uses the change
    in portfolio value as the reward signal.

    Reward = (new_value - prev_value) * scaling
    """

    def __init__(self, scaling: float = 1e-4) -> None:
        """
        Initialize the basic reward function.

        Args:
            scaling: Scaling factor for the reward.
        """
        self.scaling = scaling

    def __call__(
        self,
        prev_value: float,
        new_value: float,
        env: Any,
    ) -> float:
        """Calculate basic reward."""
        return (new_value - prev_value) * self.scaling

    def reset(self) -> None:
        """No state to reset."""
        pass

    @property
    def name(self) -> str:
        return "basic"


class ReturnReward(RewardFunction):
    """
    Reward function based on percentage return.

    This normalizes the reward by the previous portfolio value,
    making it scale-invariant.

    Reward = (new_value - prev_value) / prev_value * scaling
    """

    def __init__(self, scaling: float = 100.0) -> None:
        """
        Initialize the return reward function.

        Args:
            scaling: Scaling factor for the reward.
        """
        self.scaling = scaling

    def __call__(
        self,
        prev_value: float,
        new_value: float,
        env: Any,
    ) -> float:
        """Calculate return-based reward."""
        if prev_value <= 0:
            return 0.0
        return (new_value - prev_value) / prev_value * self.scaling

    def reset(self) -> None:
        """No state to reset."""
        pass

    @property
    def name(self) -> str:
        return "return"


class LogReturnReward(RewardFunction):
    """
    Reward function based on log return.

    Log returns are preferred for their additive property and
    better behavior for compounding returns.

    Reward = log(new_value / prev_value) * scaling
    """

    def __init__(self, scaling: float = 100.0) -> None:
        """
        Initialize the log return reward function.

        Args:
            scaling: Scaling factor for the reward.
        """
        self.scaling = scaling

    def __call__(
        self,
        prev_value: float,
        new_value: float,
        env: Any,
    ) -> float:
        """Calculate log return reward."""
        import math

        if prev_value <= 0 or new_value <= 0:
            return 0.0
        return math.log(new_value / prev_value) * self.scaling

    def reset(self) -> None:
        """No state to reset."""
        pass

    @property
    def name(self) -> str:
        return "log_return"

"""Tests for FinRL reward functions."""

from __future__ import annotations

import pytest

from libra.plugins.finrl_adapter.rewards.base import (
    BasicReward,
    LogReturnReward,
    ReturnReward,
)
from libra.plugins.finrl_adapter.rewards.sharpe import (
    CompositeReward,
    DifferentialSharpeReward,
    SharpeReward,
    SortinoReward,
    get_reward_function,
)


class TestBasicReward:
    """Tests for BasicReward."""

    def test_positive_change(self) -> None:
        """Test reward for positive portfolio change."""
        reward_fn = BasicReward(scaling=1.0)
        reward = reward_fn(100.0, 110.0, None)
        assert reward == 10.0

    def test_negative_change(self) -> None:
        """Test reward for negative portfolio change."""
        reward_fn = BasicReward(scaling=1.0)
        reward = reward_fn(100.0, 90.0, None)
        assert reward == -10.0

    def test_no_change(self) -> None:
        """Test reward for no change."""
        reward_fn = BasicReward(scaling=1.0)
        reward = reward_fn(100.0, 100.0, None)
        assert reward == 0.0

    def test_scaling(self) -> None:
        """Test reward scaling."""
        reward_fn = BasicReward(scaling=0.01)
        reward = reward_fn(100.0, 110.0, None)
        assert reward == 0.1

    def test_name(self) -> None:
        """Test reward function name."""
        reward_fn = BasicReward()
        assert reward_fn.name == "basic"

    def test_reset(self) -> None:
        """Test reset does nothing."""
        reward_fn = BasicReward()
        reward_fn.reset()  # Should not raise


class TestReturnReward:
    """Tests for ReturnReward."""

    def test_positive_return(self) -> None:
        """Test reward for positive return."""
        reward_fn = ReturnReward(scaling=100.0)
        reward = reward_fn(100.0, 110.0, None)
        assert reward == 10.0  # 10% return * 100 scaling

    def test_negative_return(self) -> None:
        """Test reward for negative return."""
        reward_fn = ReturnReward(scaling=100.0)
        reward = reward_fn(100.0, 90.0, None)
        assert reward == -10.0

    def test_zero_prev_value(self) -> None:
        """Test reward with zero previous value."""
        reward_fn = ReturnReward()
        reward = reward_fn(0.0, 100.0, None)
        assert reward == 0.0

    def test_name(self) -> None:
        """Test reward function name."""
        reward_fn = ReturnReward()
        assert reward_fn.name == "return"


class TestLogReturnReward:
    """Tests for LogReturnReward."""

    def test_positive_return(self) -> None:
        """Test log return reward."""
        import math

        reward_fn = LogReturnReward(scaling=100.0)
        reward = reward_fn(100.0, 110.0, None)
        expected = math.log(110.0 / 100.0) * 100.0
        assert abs(reward - expected) < 1e-6

    def test_zero_values(self) -> None:
        """Test with zero values."""
        reward_fn = LogReturnReward()
        assert reward_fn(0.0, 100.0, None) == 0.0
        assert reward_fn(100.0, 0.0, None) == 0.0

    def test_name(self) -> None:
        """Test reward function name."""
        reward_fn = LogReturnReward()
        assert reward_fn.name == "log_return"


class TestSharpeReward:
    """Tests for SharpeReward."""

    def test_initial_returns(self) -> None:
        """Test reward with few returns (uses simple return)."""
        reward_fn = SharpeReward(window_size=5)
        reward = reward_fn(100.0, 110.0, None)
        # Early returns are scaled differently
        assert reward != 0.0

    def test_rolling_window(self) -> None:
        """Test reward with full rolling window."""
        reward_fn = SharpeReward(window_size=3)

        # Fill window
        reward_fn(100.0, 101.0, None)
        reward_fn(101.0, 102.0, None)
        reward = reward_fn(102.0, 103.0, None)

        # Should calculate Sharpe
        assert isinstance(reward, float)

    def test_reset(self) -> None:
        """Test reset clears window."""
        reward_fn = SharpeReward(window_size=5)
        reward_fn(100.0, 110.0, None)
        reward_fn.reset()
        assert len(reward_fn._returns) == 0

    def test_name(self) -> None:
        """Test reward function name."""
        reward_fn = SharpeReward()
        assert reward_fn.name == "sharpe"


class TestDifferentialSharpeReward:
    """Tests for DifferentialSharpeReward."""

    def test_first_return(self) -> None:
        """Test reward for first return."""
        reward_fn = DifferentialSharpeReward()
        reward = reward_fn(100.0, 110.0, None)
        assert reward != 0.0

    def test_subsequent_returns(self) -> None:
        """Test reward after first return."""
        reward_fn = DifferentialSharpeReward()
        reward_fn(100.0, 110.0, None)
        reward = reward_fn(110.0, 115.0, None)
        assert isinstance(reward, float)

    def test_reset(self) -> None:
        """Test reset clears state."""
        reward_fn = DifferentialSharpeReward()
        reward_fn(100.0, 110.0, None)
        reward_fn.reset()
        assert reward_fn._A == 0.0
        assert reward_fn._B == 0.0

    def test_name(self) -> None:
        """Test reward function name."""
        reward_fn = DifferentialSharpeReward()
        assert reward_fn.name == "differential_sharpe"


class TestSortinoReward:
    """Tests for SortinoReward."""

    def test_positive_returns(self) -> None:
        """Test with positive returns."""
        reward_fn = SortinoReward(window_size=3)
        reward_fn(100.0, 101.0, None)
        reward_fn(101.0, 102.0, None)
        reward = reward_fn(102.0, 103.0, None)
        assert isinstance(reward, float)

    def test_name(self) -> None:
        """Test reward function name."""
        reward_fn = SortinoReward()
        assert reward_fn.name == "sortino"


class TestCompositeReward:
    """Tests for CompositeReward."""

    def test_reward_calculation(self) -> None:
        """Test composite reward calculation."""
        reward_fn = CompositeReward()
        reward = reward_fn(100.0, 110.0, None)
        assert isinstance(reward, float)
        assert reward != 0.0

    def test_reset(self) -> None:
        """Test reset clears state."""
        reward_fn = CompositeReward()
        reward_fn(100.0, 110.0, None)
        reward_fn.reset()
        assert len(reward_fn._returns) == 0
        assert reward_fn._max_value == 0.0

    def test_name(self) -> None:
        """Test reward function name."""
        reward_fn = CompositeReward()
        assert reward_fn.name == "composite"


class TestGetRewardFunction:
    """Tests for get_reward_function factory."""

    def test_get_basic(self) -> None:
        """Test getting basic reward."""
        reward_fn = get_reward_function("basic")
        assert reward_fn.name == "basic"

    def test_get_sharpe(self) -> None:
        """Test getting Sharpe reward."""
        reward_fn = get_reward_function("sharpe", window_size=10)
        assert reward_fn.name == "sharpe"

    def test_get_unknown(self) -> None:
        """Test getting unknown reward raises error."""
        with pytest.raises(ValueError, match="Unknown reward function"):
            get_reward_function("unknown")

    def test_get_all_types(self) -> None:
        """Test getting all reward types."""
        types = [
            "basic",
            "return",
            "log_return",
            "sharpe",
            "differential_sharpe",
            "sortino",
            "composite",
        ]
        for reward_type in types:
            reward_fn = get_reward_function(reward_type)
            assert reward_fn.name == reward_type

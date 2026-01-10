"""
FinRL Adapter Plugin for LIBRA.

This plugin enables training and deploying reinforcement learning
strategies using the FinRL framework with Stable-Baselines3.
"""

from libra.plugins.finrl_adapter.adapter import FinRLAdapter
from libra.plugins.finrl_adapter.config import FinRLAdapterConfig

__all__ = [
    "FinRLAdapter",
    "FinRLAdapterConfig",
]

"""
Plugin system for LIBRA.

Uses Python's entry_points for plugin discovery, following ADR-007.

Usage:
    # Register a plugin in pyproject.toml:
    [project.entry-points."libra.strategies"]
    my_strategy = "my_package.strategies:MyStrategy"

    # Discover and load plugins:
    from libra.plugins import discover_strategies, load_strategy

    strategies = discover_strategies()
    my_strat = load_strategy("my_strategy")
"""

from libra.plugins.base import PluginMetadata, StrategyPlugin
from libra.plugins.loader import (
    discover_gateways,
    discover_strategies,
    list_strategy_plugins,
    load_gateway,
    load_strategy,
)

__all__ = [
    # Base classes
    "PluginMetadata",
    "StrategyPlugin",
    # Strategy plugins
    "discover_strategies",
    "list_strategy_plugins",
    "load_strategy",
    # Gateway plugins (Issue #29)
    "discover_gateways",
    "load_gateway",
]

"""
Plugin loader for LIBRA.

Uses Python's importlib.metadata entry_points for plugin discovery.
This follows the pattern recommended in PEP 621 and ADR-007.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libra.plugins.base import PluginMetadata, StrategyPlugin

logger = logging.getLogger(__name__)

# Entry point group names
STRATEGY_PLUGINS_GROUP = "libra.strategies"
GATEWAY_PLUGINS_GROUP = "libra.gateways"


def _get_entry_points(group: str) -> dict[str, type]:
    """
    Load entry points for a group.

    Uses importlib.metadata which works with Python 3.10+.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        # Python 3.9 fallback (though we require 3.12+)
        from importlib_metadata import entry_points  # type: ignore[import-not-found]

    plugins: dict[str, type] = {}

    # entry_points() returns a SelectableGroups in Python 3.10+
    eps = entry_points(group=group)

    for ep in eps:
        try:
            plugin_class = ep.load()
            plugins[ep.name] = plugin_class
            logger.debug("Loaded plugin %s from %s", ep.name, ep.value)
        except Exception as e:
            logger.warning("Failed to load plugin %s: %s", ep.name, e)

    return plugins


def discover_strategies() -> dict[str, type[StrategyPlugin]]:
    """
    Discover all registered strategy plugins.

    Returns:
        Dictionary mapping plugin names to their classes.

    Example:
        strategies = discover_strategies()
        # {'freqtrade': <class 'FreqtradeAdapter'>, ...}
    """
    return _get_entry_points(STRATEGY_PLUGINS_GROUP)


def load_strategy(name: str) -> type[StrategyPlugin]:
    """
    Load a specific strategy plugin by name.

    Args:
        name: The plugin name as registered in entry_points.

    Returns:
        The strategy plugin class.

    Raises:
        KeyError: If plugin not found.

    Example:
        FreqtradeAdapter = load_strategy("freqtrade")
        adapter = FreqtradeAdapter()
    """
    strategies = discover_strategies()
    if name not in strategies:
        available = list(strategies.keys())
        raise KeyError(
            f"Strategy plugin '{name}' not found. Available: {available}"
        )
    return strategies[name]


def list_strategy_plugins() -> list[PluginMetadata]:
    """
    List metadata for all registered strategy plugins.

    Returns:
        List of PluginMetadata for each registered plugin.

    Example:
        for meta in list_strategy_plugins():
            print(f"{meta.name} v{meta.version}: {meta.description}")
    """
    from libra.plugins.base import PluginMetadata

    strategies = discover_strategies()
    metadata_list: list[PluginMetadata] = []

    for name, plugin_class in strategies.items():
        try:
            if hasattr(plugin_class, "metadata"):
                meta = plugin_class.metadata()
                metadata_list.append(meta)
            else:
                # Create basic metadata if not provided
                metadata_list.append(
                    PluginMetadata.create(
                        name=name,
                        version="0.0.0",
                        description=f"Plugin: {name}",
                    )
                )
        except Exception as e:
            logger.warning("Failed to get metadata for %s: %s", name, e)

    return metadata_list


def discover_gateways() -> dict[str, type]:
    """
    Discover all registered gateway plugins.

    Returns:
        Dictionary mapping plugin names to their classes.
    """
    return _get_entry_points(GATEWAY_PLUGINS_GROUP)


def load_gateway(name: str) -> type:
    """
    Load a specific gateway plugin by name.

    Args:
        name: The plugin name as registered in entry_points.

    Returns:
        The gateway plugin class.

    Raises:
        KeyError: If plugin not found.
    """
    gateways = discover_gateways()
    if name not in gateways:
        available = list(gateways.keys())
        raise KeyError(
            f"Gateway plugin '{name}' not found. Available: {available}"
        )
    return gateways[name]

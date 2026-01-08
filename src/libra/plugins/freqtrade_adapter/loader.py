"""
Freqtrade strategy loader for LIBRA.

Handles discovery, loading, and instantiation of Freqtrade IStrategy classes.
Provides a bridge between Freqtrade's strategy resolution system and LIBRA.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from types import ModuleType


logger = logging.getLogger(__name__)


class FreqtradeNotInstalledError(ImportError):
    """Raised when Freqtrade is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "Freqtrade is not installed. Install with: pip install libra[freqtrade]"
        )


class StrategyNotFoundError(ValueError):
    """Raised when a strategy cannot be found."""

    def __init__(self, strategy_name: str, searched_paths: list[Path]) -> None:
        paths_str = ", ".join(str(p) for p in searched_paths)
        super().__init__(
            f"Strategy '{strategy_name}' not found. Searched: {paths_str}"
        )
        self.strategy_name = strategy_name
        self.searched_paths = searched_paths


class FreqtradeStrategyLoader:
    """
    Loads and manages Freqtrade strategies.

    This loader provides methods to:
    - Discover available strategies in directories
    - Load strategies by name or path
    - Validate strategy implementations
    - Access strategy metadata

    The loader works with or without full Freqtrade installation,
    falling back to direct module loading when Freqtrade is not available.

    Examples:
        loader = FreqtradeStrategyLoader()

        # List available strategies
        strategies = loader.list_strategies(Path("./strategies"))
        print(strategies)  # ['SampleStrategy', 'MyRSIStrategy', ...]

        # Load a specific strategy
        strategy_class = loader.load_strategy(
            strategy_name="SampleStrategy",
            strategy_path=Path("./strategies"),
        )

        # Instantiate with config
        strategy = strategy_class({})
    """

    # Freqtrade's base strategy class name
    BASE_STRATEGY_CLASS = "IStrategy"

    def __init__(
        self,
        strategy_paths: list[Path] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the strategy loader.

        Args:
            strategy_paths: Default directories to search for strategies.
            config: Default Freqtrade configuration.
        """
        self.strategy_paths = strategy_paths or []
        self.config = config or {}
        self._freqtrade_available = self._check_freqtrade()
        self._strategy_cache: dict[str, type] = {}

    def _check_freqtrade(self) -> bool:
        """Check if Freqtrade is installed."""
        try:
            import freqtrade  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def freqtrade_available(self) -> bool:
        """Whether Freqtrade is installed."""
        return self._freqtrade_available

    def load_strategy(
        self,
        strategy_name: str,
        strategy_path: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> type:
        """
        Load a Freqtrade strategy class by name.

        Args:
            strategy_name: Name of the strategy class.
            strategy_path: Directory containing strategy files.
            config: Freqtrade configuration (optional).

        Returns:
            The strategy class (not instantiated).

        Raises:
            StrategyNotFoundError: If strategy cannot be found.
            FreqtradeNotInstalledError: If Freqtrade is required but not installed.
        """
        # Check cache first
        cache_key = f"{strategy_path}:{strategy_name}"
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]

        search_paths = self._get_search_paths(strategy_path)

        # Try Freqtrade's native resolver first
        if self._freqtrade_available:
            try:
                strategy_class = self._load_via_freqtrade(
                    strategy_name, search_paths, config
                )
                self._strategy_cache[cache_key] = strategy_class
                return strategy_class
            except Exception as e:
                logger.debug(
                    "Freqtrade resolver failed, falling back to direct load: %s", e
                )

        # Fallback to direct module loading
        strategy_class = self._load_direct(strategy_name, search_paths)
        self._strategy_cache[cache_key] = strategy_class
        return strategy_class

    def load_strategy_from_file(self, file_path: Path) -> type:
        """
        Load a strategy directly from a Python file.

        Args:
            file_path: Path to the strategy Python file.

        Returns:
            The strategy class.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If no valid strategy found in file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {file_path}")

        module = self._load_module_from_file(file_path)
        return self._find_strategy_in_module(module, file_path.stem)

    def list_strategies(
        self,
        strategy_path: Path | None = None,
        include_builtin: bool = False,
    ) -> list[str]:
        """
        List available strategies in the given path.

        Args:
            strategy_path: Directory to search.
            include_builtin: Include Freqtrade's built-in sample strategies.

        Returns:
            List of strategy class names.
        """
        strategies: list[str] = []
        search_paths = self._get_search_paths(strategy_path)

        for path in search_paths:
            if not path.exists():
                continue

            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    module = self._load_module_from_file(py_file)
                    for name in dir(module):
                        obj = getattr(module, name)
                        if self._is_strategy_class(obj) and name not in strategies:
                            strategies.append(name)
                except Exception as e:
                    logger.debug("Failed to load %s: %s", py_file, e)

        # Add built-in strategies if requested
        if include_builtin and self._freqtrade_available:
            try:
                from freqtrade.resolvers import StrategyResolver

                builtin = StrategyResolver.search_all_objects(
                    {"recursive_strategy_search": True},
                    enum_failed=False,
                )
                for name, _ in builtin:
                    if name not in strategies:
                        strategies.append(name)
            except Exception as e:
                logger.debug("Failed to list built-in strategies: %s", e)

        return sorted(strategies)

    def get_strategy_info(
        self,
        strategy_name: str,
        strategy_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Get information about a strategy.

        Args:
            strategy_name: Name of the strategy.
            strategy_path: Directory containing strategy.

        Returns:
            Dictionary with strategy metadata.
        """
        strategy_class = self.load_strategy(strategy_name, strategy_path)

        info: dict[str, Any] = {
            "name": strategy_name,
            "module": strategy_class.__module__,
            "docstring": strategy_class.__doc__ or "",
        }

        # Extract common Freqtrade attributes
        for attr in [
            "timeframe",
            "stoploss",
            "minimal_roi",
            "startup_candle_count",
            "can_short",
            "use_exit_signal",
            "process_only_new_candles",
        ]:
            if hasattr(strategy_class, attr):
                value = getattr(strategy_class, attr)
                # Convert to JSON-serializable
                if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    info[attr] = value
                else:
                    info[attr] = str(value)

        return info

    def load_config(self, config_path: Path) -> dict[str, Any]:
        """
        Load Freqtrade configuration from JSON file.

        Args:
            config_path: Path to config.json file.

        Returns:
            Configuration dictionary.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open() as f:
            return json.load(f)

    def _get_search_paths(self, strategy_path: Path | None) -> list[Path]:
        """Get list of paths to search for strategies."""
        paths: list[Path] = []

        if strategy_path:
            paths.append(strategy_path)

        paths.extend(self.strategy_paths)

        # Add default Freqtrade user_data path
        user_data = Path.home() / ".freqtrade" / "strategies"
        if user_data.exists() and user_data not in paths:
            paths.append(user_data)

        return paths

    def _load_via_freqtrade(
        self,
        strategy_name: str,
        search_paths: list[Path],
        config: dict[str, Any] | None,
    ) -> type:
        """Load strategy using Freqtrade's native resolver."""
        from freqtrade.resolvers import StrategyResolver

        ft_config = config or self.config.copy()
        ft_config["strategy"] = strategy_name

        if search_paths:
            ft_config["strategy_path"] = str(search_paths[0])

        strategy = StrategyResolver.load_strategy(ft_config)
        return type(strategy)

    def _load_direct(
        self,
        strategy_name: str,
        search_paths: list[Path],
    ) -> type:
        """Load strategy by directly importing the module."""
        for path in search_paths:
            if not path.exists():
                continue

            # Search for .py files
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    module = self._load_module_from_file(py_file)
                    if hasattr(module, strategy_name):
                        obj = getattr(module, strategy_name)
                        if self._is_strategy_class(obj):
                            return obj
                except Exception as e:
                    logger.debug("Failed to load %s: %s", py_file, e)

        raise StrategyNotFoundError(strategy_name, search_paths)

    def _load_module_from_file(self, file_path: Path) -> ModuleType:
        """Load a Python module from file path."""
        module_name = f"libra_ft_strategy_{file_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[module_name]
            raise ImportError(f"Error executing module {file_path}: {e}") from e

        return module

    def _find_strategy_in_module(self, module: ModuleType, hint: str) -> type:
        """Find a strategy class in a loaded module."""
        # Try exact match first
        if hasattr(module, hint):
            obj = getattr(module, hint)
            if self._is_strategy_class(obj):
                return obj

        # Search for any strategy class
        candidates: list[type] = []
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if self._is_strategy_class(obj):
                candidates.append(obj)

        if not candidates:
            raise ValueError(f"No strategy class found in module: {module.__name__}")

        if len(candidates) == 1:
            return candidates[0]

        # Multiple candidates - try to find one matching the hint
        for candidate in candidates:
            if candidate.__name__.lower() == hint.lower():
                return candidate

        # Return first candidate
        logger.warning(
            "Multiple strategies found in %s, using %s",
            module.__name__,
            candidates[0].__name__,
        )
        return candidates[0]

    def _is_strategy_class(self, obj: Any) -> bool:
        """Check if an object is a valid Freqtrade strategy class."""
        if not isinstance(obj, type):
            return False

        # Check for IStrategy inheritance if Freqtrade is available
        if self._freqtrade_available:
            try:
                from freqtrade.strategy import IStrategy

                return issubclass(obj, IStrategy) and obj is not IStrategy
            except ImportError:
                pass

        # Fallback: check for required methods
        required_methods = [
            "populate_indicators",
            "populate_entry_trend",
            "populate_exit_trend",
        ]
        return all(hasattr(obj, method) for method in required_methods)

    def clear_cache(self) -> None:
        """Clear the strategy cache."""
        self._strategy_cache.clear()

"""
Path management for LIBRA local storage.

Directory structure:
    ~/.libra/
    ├── config/
    │   └── libra.yaml
    ├── strategies/
    │   └── {name}/
    │       ├── config.yaml
    │       └── state.json
    ├── results/
    │   ├── backtests/
    │   └── live/
    ├── logs/
    └── cache/
"""

from __future__ import annotations

import os
from pathlib import Path

import msgspec


class LibraPaths(msgspec.Struct, frozen=True):
    """
    Paths for LIBRA local storage.

    All paths are resolved relative to a root directory,
    defaulting to ~/.libra/ or $LIBRA_HOME if set.

    Example:
        paths = LibraPaths.default()
        paths.ensure_dirs()

        # Access paths
        config_file = paths.config / "libra.yaml"
        strategy_dir = paths.strategies / "my_strategy"
    """

    root: Path

    @classmethod
    def default(cls) -> LibraPaths:
        """
        Create paths with default root.

        Uses $LIBRA_HOME if set, otherwise ~/.libra/
        """
        if env_home := os.environ.get("LIBRA_HOME"):
            root = Path(env_home)
        else:
            root = Path.home() / ".libra"
        return cls(root=root)

    @classmethod
    def from_root(cls, root: Path | str) -> LibraPaths:
        """Create paths with custom root."""
        return cls(root=Path(root))

    @property
    def config(self) -> Path:
        """Config directory: ~/.libra/config/"""
        return self.root / "config"

    @property
    def strategies(self) -> Path:
        """Strategies directory: ~/.libra/strategies/"""
        return self.root / "strategies"

    @property
    def results(self) -> Path:
        """Results directory: ~/.libra/results/"""
        return self.root / "results"

    @property
    def backtests(self) -> Path:
        """Backtest results: ~/.libra/results/backtests/"""
        return self.results / "backtests"

    @property
    def live_results(self) -> Path:
        """Live trading results: ~/.libra/results/live/"""
        return self.results / "live"

    @property
    def logs(self) -> Path:
        """Logs directory: ~/.libra/logs/"""
        return self.root / "logs"

    @property
    def cache(self) -> Path:
        """Cache directory: ~/.libra/cache/"""
        return self.root / "cache"

    def strategy_dir(self, name: str) -> Path:
        """Get directory for a specific strategy."""
        return self.strategies / name

    def strategy_config(self, name: str) -> Path:
        """Get config file path for a strategy."""
        return self.strategy_dir(name) / "config.yaml"

    def strategy_state(self, name: str) -> Path:
        """Get state file path for a strategy."""
        return self.strategy_dir(name) / "state.json"

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        dirs = [
            self.config,
            self.strategies,
            self.backtests,
            self.live_results,
            self.logs,
            self.cache,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def all_strategies(self) -> list[str]:
        """List all strategy names."""
        if not self.strategies.exists():
            return []
        return [
            d.name
            for d in self.strategies.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

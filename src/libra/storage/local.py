"""
Local storage implementation for LIBRA.

Provides simple file-based storage for:
- Strategy configurations (YAML)
- Backtest results (Parquet via Polars)
- Runtime state (JSON via msgspec)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from libra.storage.paths import LibraPaths

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class LocalStorage:
    """
    Simple local storage for LIBRA.

    Stores data under ~/.libra/ (or $LIBRA_HOME):
    - configs: YAML files
    - results: Parquet files
    - state: JSON files

    Example:
        storage = LocalStorage()

        # Save/load strategy config
        storage.save_strategy_config("momentum", {"period": 20})
        config = storage.load_strategy_config("momentum")

        # Save backtest results
        storage.save_backtest("momentum_btc", results_df)

        # List strategies
        strategies = storage.list_strategies()
    """

    def __init__(self, paths: LibraPaths | None = None) -> None:
        """
        Initialize storage.

        Args:
            paths: Custom paths. If None, uses LibraPaths.default()
        """
        self.paths = paths or LibraPaths.default()
        self.paths.ensure_dirs()

    # =========================================================================
    # Strategy Configuration (YAML)
    # =========================================================================

    def save_strategy_config(self, name: str, config: dict[str, Any]) -> Path:
        """
        Save strategy configuration as YAML.

        Args:
            name: Strategy name (used as directory name)
            config: Configuration dictionary

        Returns:
            Path to saved config file
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML required for config storage. "
                "Install with: pip install pyyaml"
            ) from e

        strategy_dir = self.paths.strategy_dir(name)
        strategy_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.paths.strategy_config(name)
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        logger.debug("Saved strategy config: %s", config_path)
        return config_path

    def load_strategy_config(self, name: str) -> dict[str, Any]:
        """
        Load strategy configuration from YAML.

        Args:
            name: Strategy name

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If strategy config doesn't exist
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML required for config storage. "
                "Install with: pip install pyyaml"
            ) from e

        config_path = self.paths.strategy_config(name)
        if not config_path.exists():
            raise FileNotFoundError(f"Strategy config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config or {}

    def strategy_exists(self, name: str) -> bool:
        """Check if a strategy configuration exists."""
        return self.paths.strategy_config(name).exists()

    def list_strategies(self) -> list[str]:
        """List all strategy names with configurations."""
        return [
            name
            for name in self.paths.all_strategies()
            if self.paths.strategy_config(name).exists()
        ]

    def delete_strategy(self, name: str) -> bool:
        """
        Delete a strategy and all its files.

        Returns:
            True if deleted, False if didn't exist
        """
        import shutil

        strategy_dir = self.paths.strategy_dir(name)
        if not strategy_dir.exists():
            return False

        shutil.rmtree(strategy_dir)
        logger.info("Deleted strategy: %s", name)
        return True

    # =========================================================================
    # Strategy State (JSON)
    # =========================================================================

    def save_strategy_state(self, name: str, state: dict[str, Any]) -> Path:
        """
        Save strategy runtime state as JSON.

        Uses msgspec for fast serialization.
        """
        strategy_dir = self.paths.strategy_dir(name)
        strategy_dir.mkdir(parents=True, exist_ok=True)

        state_path = self.paths.strategy_state(name)
        data = msgspec.json.encode(state)
        state_path.write_bytes(data)

        return state_path

    def load_strategy_state(self, name: str) -> dict[str, Any]:
        """
        Load strategy runtime state from JSON.

        Returns empty dict if no state file exists.
        """
        state_path = self.paths.strategy_state(name)
        if not state_path.exists():
            return {}

        data = state_path.read_bytes()
        return msgspec.json.decode(data)

    # =========================================================================
    # Backtest Results (Parquet)
    # =========================================================================

    def save_backtest(
        self,
        name: str,
        results: pl.DataFrame,
        timestamp: datetime | None = None,
    ) -> Path:
        """
        Save backtest results as Parquet.

        Args:
            name: Backtest name (e.g., "momentum_btc")
            results: Results DataFrame
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Path to saved Parquet file
        """
        if timestamp is None:
            timestamp = datetime.now()

        date_str = timestamp.strftime("%Y-%m-%d_%H%M%S")
        filename = f"{date_str}_{name}.parquet"
        path = self.paths.backtests / filename

        results.write_parquet(path)
        logger.info("Saved backtest results: %s", path)
        return path

    def load_backtest(self, filename: str) -> pl.DataFrame:
        """
        Load backtest results from Parquet.

        Args:
            filename: Parquet filename (with or without .parquet extension)

        Returns:
            Results DataFrame
        """
        import polars as pl

        if not filename.endswith(".parquet"):
            filename = f"{filename}.parquet"

        path = self.paths.backtests / filename
        if not path.exists():
            raise FileNotFoundError(f"Backtest results not found: {path}")

        return pl.read_parquet(path)

    def list_backtests(self) -> list[str]:
        """List all backtest result filenames."""
        if not self.paths.backtests.exists():
            return []
        return sorted(
            f.name for f in self.paths.backtests.glob("*.parquet")
        )

    # =========================================================================
    # Live Trading Results (Parquet)
    # =========================================================================

    def save_live_trades(
        self,
        trades: pl.DataFrame,
        period: str | None = None,
    ) -> Path:
        """
        Save live trading results as Parquet.

        Args:
            trades: Trades DataFrame
            period: Period identifier (e.g., "2026-01"). Defaults to current month.

        Returns:
            Path to saved Parquet file
        """
        if period is None:
            period = datetime.now().strftime("%Y-%m")

        filename = f"trades_{period}.parquet"
        path = self.paths.live_results / filename

        # Append to existing file if it exists
        if path.exists():
            import polars as pl

            existing = pl.read_parquet(path)
            trades = pl.concat([existing, trades])

        trades.write_parquet(path)
        logger.info("Saved live trades: %s", path)
        return path

    def load_live_trades(self, period: str) -> pl.DataFrame:
        """
        Load live trading results for a period.

        Args:
            period: Period identifier (e.g., "2026-01")

        Returns:
            Trades DataFrame
        """
        import polars as pl

        filename = f"trades_{period}.parquet"
        path = self.paths.live_results / filename

        if not path.exists():
            raise FileNotFoundError(f"Live trades not found: {path}")

        return pl.read_parquet(path)

    # =========================================================================
    # Main Config
    # =========================================================================

    def save_config(self, config: dict[str, Any]) -> Path:
        """Save main LIBRA configuration."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML required for config storage. "
                "Install with: pip install pyyaml"
            ) from e

        config_path = self.paths.config / "libra.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        return config_path

    def load_config(self) -> dict[str, Any]:
        """Load main LIBRA configuration."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML required for config storage. "
                "Install with: pip install pyyaml"
            ) from e

        config_path = self.paths.config / "libra.yaml"
        if not config_path.exists():
            return {}

        with open(config_path) as f:
            return yaml.safe_load(f) or {}

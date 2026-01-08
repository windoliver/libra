"""Tests for local storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from libra.storage.local import LocalStorage
from libra.storage.paths import LibraPaths


@pytest.fixture
def temp_storage(tmp_path: Path) -> LocalStorage:
    """Create a LocalStorage with temporary directory."""
    paths = LibraPaths.from_root(tmp_path)
    return LocalStorage(paths)


class TestLibraPaths:
    """Tests for LibraPaths."""

    def test_default_paths(self) -> None:
        """Test default path structure."""
        paths = LibraPaths.default()
        assert paths.root == Path.home() / ".libra"
        assert paths.config == paths.root / "config"
        assert paths.strategies == paths.root / "strategies"
        assert paths.backtests == paths.root / "results" / "backtests"

    def test_custom_root(self, tmp_path: Path) -> None:
        """Test custom root path."""
        paths = LibraPaths.from_root(tmp_path)
        assert paths.root == tmp_path
        assert paths.config == tmp_path / "config"

    def test_libra_home_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test LIBRA_HOME environment variable."""
        monkeypatch.setenv("LIBRA_HOME", str(tmp_path))
        paths = LibraPaths.default()
        assert paths.root == tmp_path

    def test_ensure_dirs(self, tmp_path: Path) -> None:
        """Test directory creation."""
        paths = LibraPaths.from_root(tmp_path)
        paths.ensure_dirs()

        assert paths.config.exists()
        assert paths.strategies.exists()
        assert paths.backtests.exists()
        assert paths.live_results.exists()
        assert paths.logs.exists()
        assert paths.cache.exists()

    def test_strategy_paths(self, tmp_path: Path) -> None:
        """Test strategy path helpers."""
        paths = LibraPaths.from_root(tmp_path)
        assert paths.strategy_dir("momentum") == tmp_path / "strategies" / "momentum"
        assert paths.strategy_config("momentum") == tmp_path / "strategies" / "momentum" / "config.yaml"
        assert paths.strategy_state("momentum") == tmp_path / "strategies" / "momentum" / "state.json"

    def test_all_strategies_empty(self, tmp_path: Path) -> None:
        """Test listing strategies when none exist."""
        paths = LibraPaths.from_root(tmp_path)
        assert paths.all_strategies() == []

    def test_all_strategies(self, tmp_path: Path) -> None:
        """Test listing strategies."""
        paths = LibraPaths.from_root(tmp_path)
        paths.ensure_dirs()

        # Create strategy directories
        (paths.strategies / "momentum").mkdir()
        (paths.strategies / "mean_reversion").mkdir()
        (paths.strategies / ".hidden").mkdir()  # Should be ignored

        strategies = paths.all_strategies()
        assert set(strategies) == {"momentum", "mean_reversion"}


class TestLocalStorageConfig:
    """Tests for strategy configuration storage."""

    def test_save_and_load_strategy_config(self, temp_storage: LocalStorage) -> None:
        """Test saving and loading strategy config."""
        config = {
            "name": "momentum",
            "period": 20,
            "symbols": ["BTC/USDT", "ETH/USDT"],
        }

        path = temp_storage.save_strategy_config("momentum", config)
        assert path.exists()
        assert path.suffix == ".yaml"

        loaded = temp_storage.load_strategy_config("momentum")
        assert loaded == config

    def test_load_missing_config(self, temp_storage: LocalStorage) -> None:
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            temp_storage.load_strategy_config("nonexistent")

    def test_strategy_exists(self, temp_storage: LocalStorage) -> None:
        """Test strategy existence check."""
        assert not temp_storage.strategy_exists("momentum")

        temp_storage.save_strategy_config("momentum", {"period": 20})
        assert temp_storage.strategy_exists("momentum")

    def test_list_strategies(self, temp_storage: LocalStorage) -> None:
        """Test listing strategies."""
        assert temp_storage.list_strategies() == []

        temp_storage.save_strategy_config("momentum", {})
        temp_storage.save_strategy_config("mean_reversion", {})

        strategies = temp_storage.list_strategies()
        assert set(strategies) == {"momentum", "mean_reversion"}

    def test_delete_strategy(self, temp_storage: LocalStorage) -> None:
        """Test deleting a strategy."""
        temp_storage.save_strategy_config("momentum", {"period": 20})
        assert temp_storage.strategy_exists("momentum")

        result = temp_storage.delete_strategy("momentum")
        assert result is True
        assert not temp_storage.strategy_exists("momentum")

        # Delete non-existent returns False
        result = temp_storage.delete_strategy("nonexistent")
        assert result is False


class TestLocalStorageState:
    """Tests for strategy state storage."""

    def test_save_and_load_state(self, temp_storage: LocalStorage) -> None:
        """Test saving and loading strategy state."""
        state = {
            "position": "long",
            "entry_price": 50000.0,
            "quantity": 0.1,
        }

        path = temp_storage.save_strategy_state("momentum", state)
        assert path.exists()
        assert path.suffix == ".json"

        loaded = temp_storage.load_strategy_state("momentum")
        assert loaded == state

    def test_load_missing_state_returns_empty(self, temp_storage: LocalStorage) -> None:
        """Test loading non-existent state returns empty dict."""
        state = temp_storage.load_strategy_state("nonexistent")
        assert state == {}


class TestLocalStorageBacktest:
    """Tests for backtest results storage."""

    def test_save_and_load_backtest(self, temp_storage: LocalStorage) -> None:
        """Test saving and loading backtest results."""
        import polars as pl
        from datetime import datetime

        results = pl.DataFrame({
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "pnl": [100.0, -50.0],
            "cumulative_pnl": [100.0, 50.0],
        })

        path = temp_storage.save_backtest("momentum_btc", results)
        assert path.exists()
        assert path.suffix == ".parquet"
        assert "momentum_btc" in path.name

        # Load by filename
        loaded = temp_storage.load_backtest(path.name)
        assert loaded.shape == results.shape

    def test_list_backtests(self, temp_storage: LocalStorage) -> None:
        """Test listing backtest results."""
        import polars as pl

        assert temp_storage.list_backtests() == []

        results = pl.DataFrame({"pnl": [1.0, 2.0]})
        temp_storage.save_backtest("test1", results)
        temp_storage.save_backtest("test2", results)

        backtests = temp_storage.list_backtests()
        assert len(backtests) == 2
        assert all(".parquet" in b for b in backtests)

    def test_load_missing_backtest(self, temp_storage: LocalStorage) -> None:
        """Test loading non-existent backtest raises error."""
        with pytest.raises(FileNotFoundError):
            temp_storage.load_backtest("nonexistent")


class TestLocalStorageLiveTrades:
    """Tests for live trading results storage."""

    def test_save_and_load_live_trades(self, temp_storage: LocalStorage) -> None:
        """Test saving and loading live trades."""
        import polars as pl

        trades = pl.DataFrame({
            "symbol": ["BTC/USDT", "ETH/USDT"],
            "side": ["buy", "sell"],
            "amount": [0.1, 1.0],
        })

        path = temp_storage.save_live_trades(trades, "2026-01")
        assert path.exists()

        loaded = temp_storage.load_live_trades("2026-01")
        assert loaded.shape == trades.shape

    def test_append_live_trades(self, temp_storage: LocalStorage) -> None:
        """Test appending to existing live trades."""
        import polars as pl

        trades1 = pl.DataFrame({"amount": [1.0]})
        trades2 = pl.DataFrame({"amount": [2.0]})

        temp_storage.save_live_trades(trades1, "2026-01")
        temp_storage.save_live_trades(trades2, "2026-01")

        loaded = temp_storage.load_live_trades("2026-01")
        assert loaded.shape[0] == 2  # Both rows


class TestLocalStorageMainConfig:
    """Tests for main LIBRA configuration."""

    def test_save_and_load_config(self, temp_storage: LocalStorage) -> None:
        """Test saving and loading main config."""
        config = {
            "mode": "paper",
            "default_exchange": "binance",
        }

        path = temp_storage.save_config(config)
        assert path.exists()
        assert path.name == "libra.yaml"

        loaded = temp_storage.load_config()
        assert loaded == config

    def test_load_missing_config_returns_empty(self, temp_storage: LocalStorage) -> None:
        """Test loading non-existent config returns empty dict."""
        config = temp_storage.load_config()
        assert config == {}

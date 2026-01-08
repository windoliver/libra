"""Tests for plugin loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from libra.plugins.base import PluginMetadata, StrategyPlugin
from libra.plugins.loader import (
    STRATEGY_PLUGINS_GROUP,
    discover_strategies,
    list_strategy_plugins,
    load_strategy,
)


class MockStrategyPlugin(StrategyPlugin):
    """Mock strategy plugin for testing."""

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata.create(
            name="mock-strategy",
            version="1.0.0",
            description="A mock strategy for testing",
            author="Test",
            requires=["mock-dep>=1.0"],
        )

    @property
    def name(self) -> str:
        return "mock"

    @property
    def symbols(self) -> list[str]:
        return ["BTC/USDT"]

    async def initialize(self, config: dict) -> None:
        pass

    async def on_data(self, data) -> None:
        return None


class TestPluginMetadata:
    """Tests for PluginMetadata."""

    def test_create_minimal(self) -> None:
        """Test creating metadata with minimal fields."""
        meta = PluginMetadata.create(
            name="test",
            version="1.0.0",
            description="Test plugin",
        )
        assert meta.name == "test"
        assert meta.version == "1.0.0"
        assert meta.description == "Test plugin"
        assert meta.author == "LIBRA"
        assert meta.requires == ()

    def test_create_full(self) -> None:
        """Test creating metadata with all fields."""
        meta = PluginMetadata.create(
            name="test",
            version="2.0.0",
            description="Full test plugin",
            author="Test Author",
            requires=["dep1>=1.0", "dep2>=2.0"],
        )
        assert meta.name == "test"
        assert meta.version == "2.0.0"
        assert meta.author == "Test Author"
        assert meta.requires == ("dep1>=1.0", "dep2>=2.0")

    def test_metadata_is_frozen(self) -> None:
        """Test that metadata is immutable."""
        meta = PluginMetadata.create(name="test", version="1.0.0", description="Test")
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]


class TestDiscoverStrategies:
    """Tests for discover_strategies."""

    def test_discover_no_plugins(self) -> None:
        """Test discovery with no plugins registered."""
        mock_entry_points = MagicMock(return_value=[])
        with patch(
            "importlib.metadata.entry_points", mock_entry_points
        ):
            strategies = discover_strategies()
            assert strategies == {}
            mock_entry_points.assert_called_once_with(group=STRATEGY_PLUGINS_GROUP)

    def test_discover_with_plugins(self) -> None:
        """Test discovery with registered plugins."""
        mock_ep = MagicMock()
        mock_ep.name = "mock-strategy"
        mock_ep.value = "test.module:MockStrategy"
        mock_ep.load.return_value = MockStrategyPlugin

        mock_entry_points = MagicMock(return_value=[mock_ep])
        with patch(
            "importlib.metadata.entry_points", mock_entry_points
        ):
            strategies = discover_strategies()
            assert "mock-strategy" in strategies
            assert strategies["mock-strategy"] == MockStrategyPlugin

    def test_discover_handles_load_error(self) -> None:
        """Test that failed plugin loads are handled gracefully."""
        mock_ep = MagicMock()
        mock_ep.name = "broken-plugin"
        mock_ep.load.side_effect = ImportError("Module not found")

        mock_entry_points = MagicMock(return_value=[mock_ep])
        with patch(
            "importlib.metadata.entry_points", mock_entry_points
        ):
            strategies = discover_strategies()
            assert "broken-plugin" not in strategies


class TestLoadStrategy:
    """Tests for load_strategy."""

    def test_load_existing_plugin(self) -> None:
        """Test loading an existing plugin."""
        mock_ep = MagicMock()
        mock_ep.name = "test-strategy"
        mock_ep.load.return_value = MockStrategyPlugin

        mock_entry_points = MagicMock(return_value=[mock_ep])
        with patch(
            "importlib.metadata.entry_points", mock_entry_points
        ):
            plugin_class = load_strategy("test-strategy")
            assert plugin_class == MockStrategyPlugin

    def test_load_nonexistent_plugin(self) -> None:
        """Test loading a non-existent plugin raises KeyError."""
        mock_entry_points = MagicMock(return_value=[])
        with patch(
            "importlib.metadata.entry_points", mock_entry_points
        ):
            with pytest.raises(KeyError) as exc_info:
                load_strategy("nonexistent")
            assert "nonexistent" in str(exc_info.value)
            assert "not found" in str(exc_info.value)


class TestListStrategyPlugins:
    """Tests for list_strategy_plugins."""

    def test_list_plugins_with_metadata(self) -> None:
        """Test listing plugins that have metadata."""
        mock_ep = MagicMock()
        mock_ep.name = "mock-strategy"
        mock_ep.load.return_value = MockStrategyPlugin

        mock_entry_points = MagicMock(return_value=[mock_ep])
        with patch(
            "importlib.metadata.entry_points", mock_entry_points
        ):
            metadata_list = list_strategy_plugins()
            assert len(metadata_list) == 1
            assert metadata_list[0].name == "mock-strategy"
            assert metadata_list[0].version == "1.0.0"

    def test_list_plugins_without_metadata(self) -> None:
        """Test listing plugins that don't have metadata method."""

        class NoMetadataPlugin:
            pass

        mock_ep = MagicMock()
        mock_ep.name = "no-meta"
        mock_ep.load.return_value = NoMetadataPlugin

        mock_entry_points = MagicMock(return_value=[mock_ep])
        with patch(
            "importlib.metadata.entry_points", mock_entry_points
        ):
            metadata_list = list_strategy_plugins()
            assert len(metadata_list) == 1
            assert metadata_list[0].name == "no-meta"
            assert metadata_list[0].version == "0.0.0"


class TestStrategyPluginBase:
    """Tests for StrategyPlugin base class."""

    def test_plugin_metadata(self) -> None:
        """Test that plugin metadata is accessible."""
        meta = MockStrategyPlugin.metadata()
        assert meta.name == "mock-strategy"
        assert meta.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_default_on_tick_returns_none(self) -> None:
        """Test default on_tick implementation."""
        plugin = MockStrategyPlugin()
        result = await plugin.on_tick({"price": 100})
        assert result is None

    @pytest.mark.asyncio
    async def test_default_backtest_raises(self) -> None:
        """Test default backtest raises NotImplementedError."""
        from decimal import Decimal

        plugin = MockStrategyPlugin()
        with pytest.raises(NotImplementedError):
            await plugin.backtest(data=None, initial_capital=Decimal("10000"))

    @pytest.mark.asyncio
    async def test_default_optimize_raises(self) -> None:
        """Test default optimize raises NotImplementedError."""
        plugin = MockStrategyPlugin()
        with pytest.raises(NotImplementedError):
            await plugin.optimize(data=None, param_space={})

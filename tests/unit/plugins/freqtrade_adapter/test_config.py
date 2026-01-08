"""Unit tests for FreqtradeAdapterConfig."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from libra.plugins.freqtrade_adapter.config import FreqtradeAdapterConfig


class TestFreqtradeAdapterConfig:
    """Tests for FreqtradeAdapterConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FreqtradeAdapterConfig(strategy_name="TestStrategy")

        assert config.strategy_name == "TestStrategy"
        assert config.timeframe == "1h"
        assert config.stake_currency == "USDT"
        assert config.dry_run is True
        assert config.stoploss == Decimal("-0.10")
        assert config.max_open_trades == 3
        assert config.trailing_stop is False
        assert config.use_exit_signal is True
        assert config.freqai_enabled is False

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = FreqtradeAdapterConfig(
            strategy_name="MyStrategy",
            timeframe="5m",
            pair_whitelist=["BTC/USDT", "ETH/USDT"],
            stake_currency="USDC",
            stake_amount=Decimal("500"),
            dry_run=False,
            stoploss=Decimal("-0.05"),
            max_open_trades=5,
            trailing_stop=True,
            trailing_stop_positive=Decimal("0.01"),
        )

        assert config.strategy_name == "MyStrategy"
        assert config.timeframe == "5m"
        assert config.pair_whitelist == ["BTC/USDT", "ETH/USDT"]
        assert config.stake_currency == "USDC"
        assert config.stake_amount == Decimal("500")
        assert config.dry_run is False
        assert config.stoploss == Decimal("-0.05")
        assert config.max_open_trades == 5
        assert config.trailing_stop is True
        assert config.trailing_stop_positive == Decimal("0.01")

    def test_to_freqtrade_config(self) -> None:
        """Test conversion to Freqtrade config format."""
        config = FreqtradeAdapterConfig(
            strategy_name="TestStrategy",
            timeframe="1h",
            pair_whitelist=["BTC/USDT"],
            stake_currency="USDT",
            stake_amount=Decimal("100"),
            dry_run=True,
            stoploss=Decimal("-0.10"),
            max_open_trades=3,
        )

        ft_config = config.to_freqtrade_config()

        assert ft_config["strategy"] == "TestStrategy"
        assert ft_config["timeframe"] == "1h"
        assert ft_config["exchange"]["pair_whitelist"] == ["BTC/USDT"]
        assert ft_config["stake_currency"] == "USDT"
        assert ft_config["stake_amount"] == "100"
        assert ft_config["dry_run"] is True
        assert ft_config["stoploss"] == -0.10
        assert ft_config["max_open_trades"] == 3

    def test_to_freqtrade_config_with_trailing_stop(self) -> None:
        """Test conversion with trailing stop configuration."""
        config = FreqtradeAdapterConfig(
            strategy_name="TestStrategy",
            trailing_stop=True,
            trailing_stop_positive=Decimal("0.02"),
            trailing_stop_positive_offset=Decimal("0.03"),
        )

        ft_config = config.to_freqtrade_config()

        assert ft_config["trailing_stop"] is True
        assert ft_config["trailing_stop_positive"] == 0.02
        assert ft_config["trailing_stop_positive_offset"] == 0.03

    def test_to_freqtrade_config_with_freqai(self) -> None:
        """Test conversion with FreqAI configuration."""
        config = FreqtradeAdapterConfig(
            strategy_name="TestStrategy",
            freqai_enabled=True,
            freqai_config={
                "model_training_parameters": {"n_estimators": 100},
            },
        )

        ft_config = config.to_freqtrade_config()

        assert ft_config["freqai"]["enabled"] is True
        assert ft_config["freqai"]["model_training_parameters"]["n_estimators"] == 100

    def test_to_freqtrade_config_with_strategy_path(self) -> None:
        """Test conversion with strategy path."""
        config = FreqtradeAdapterConfig(
            strategy_name="TestStrategy",
            strategy_path=Path("/path/to/strategies"),
        )

        ft_config = config.to_freqtrade_config()

        assert ft_config["strategy_path"] == "/path/to/strategies"

    def test_from_dict_basic(self) -> None:
        """Test creation from dictionary."""
        data = {
            "strategy_name": "TestStrategy",
            "timeframe": "15m",
            "pair_whitelist": ["BTC/USDT"],
        }

        config = FreqtradeAdapterConfig.from_dict(data)

        assert config.strategy_name == "TestStrategy"
        assert config.timeframe == "15m"
        assert config.pair_whitelist == ["BTC/USDT"]

    def test_from_dict_with_paths(self) -> None:
        """Test creation from dictionary with path strings."""
        data = {
            "strategy_name": "TestStrategy",
            "config_path": "/path/to/config.json",
            "strategy_path": "/path/to/strategies",
        }

        config = FreqtradeAdapterConfig.from_dict(data)

        assert config.config_path == Path("/path/to/config.json")
        assert config.strategy_path == Path("/path/to/strategies")

    def test_from_dict_with_decimals(self) -> None:
        """Test creation from dictionary with decimal strings."""
        data = {
            "strategy_name": "TestStrategy",
            "stoploss": "-0.05",
            "stake_amount": "250.50",
            "trailing_stop_positive": "0.015",
            "trailing_stop_positive_offset": "0.02",
        }

        config = FreqtradeAdapterConfig.from_dict(data)

        assert config.stoploss == Decimal("-0.05")
        assert config.stake_amount == Decimal("250.50")
        assert config.trailing_stop_positive == Decimal("0.015")
        assert config.trailing_stop_positive_offset == Decimal("0.02")

    def test_from_dict_unlimited_stake(self) -> None:
        """Test creation with 'unlimited' stake amount."""
        data = {
            "strategy_name": "TestStrategy",
            "stake_amount": "unlimited",
        }

        config = FreqtradeAdapterConfig.from_dict(data)

        assert config.stake_amount == "unlimited"

    def test_from_dict_extra_config(self) -> None:
        """Test that unknown fields go to extra_config."""
        data = {
            "strategy_name": "TestStrategy",
            "custom_field": "custom_value",
            "another_field": 123,
        }

        config = FreqtradeAdapterConfig.from_dict(data)

        assert config.extra_config["custom_field"] == "custom_value"
        assert config.extra_config["another_field"] == 123

    def test_from_freqtrade_config(self) -> None:
        """Test creation from native Freqtrade config."""
        ft_config = {
            "strategy": "SampleStrategy",
            "timeframe": "5m",
            "stake_currency": "USDT",
            "stake_amount": 100,
            "dry_run": True,
            "max_open_trades": 5,
            "stoploss": -0.10,
            "trailing_stop": True,
            "trailing_stop_positive": 0.01,
            "trailing_stop_positive_offset": 0.02,
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exchange": {
                "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
            },
        }

        config = FreqtradeAdapterConfig.from_freqtrade_config(ft_config)

        assert config.strategy_name == "SampleStrategy"
        assert config.timeframe == "5m"
        assert config.stake_currency == "USDT"
        assert config.stake_amount == Decimal("100")
        assert config.dry_run is True
        assert config.max_open_trades == 5
        assert config.stoploss == Decimal("-0.10")
        assert config.trailing_stop is True
        assert config.trailing_stop_positive == Decimal("0.01")
        assert config.trailing_stop_positive_offset == Decimal("0.02")
        assert config.pair_whitelist == ["BTC/USDT", "ETH/USDT"]

    def test_from_freqtrade_config_with_override(self) -> None:
        """Test creation from Freqtrade config with strategy name override."""
        ft_config = {
            "strategy": "OriginalStrategy",
            "timeframe": "1h",
            "exchange": {},
        }

        config = FreqtradeAdapterConfig.from_freqtrade_config(
            ft_config,
            strategy_name="OverriddenStrategy",
        )

        assert config.strategy_name == "OverriddenStrategy"

    def test_from_freqtrade_config_with_freqai(self) -> None:
        """Test creation from Freqtrade config with FreqAI."""
        ft_config = {
            "strategy": "FreqAIStrategy",
            "timeframe": "1h",
            "exchange": {},
            "freqai": {
                "enabled": True,
                "model_training_parameters": {"n_estimators": 100},
            },
        }

        config = FreqtradeAdapterConfig.from_freqtrade_config(ft_config)

        assert config.freqai_enabled is True
        assert config.freqai_config["enabled"] is True
        assert config.freqai_config["model_training_parameters"]["n_estimators"] == 100

    def test_roundtrip_conversion(self) -> None:
        """Test that config survives roundtrip conversion."""
        original = FreqtradeAdapterConfig(
            strategy_name="TestStrategy",
            timeframe="15m",
            pair_whitelist=["BTC/USDT", "ETH/USDT"],
            stake_currency="USDT",
            stake_amount=Decimal("500"),
            stoploss=Decimal("-0.08"),
            max_open_trades=4,
            trailing_stop=True,
            trailing_stop_positive=Decimal("0.02"),
        )

        # Convert to FT config and back
        ft_config = original.to_freqtrade_config()
        restored = FreqtradeAdapterConfig.from_freqtrade_config(ft_config)

        assert restored.strategy_name == original.strategy_name
        assert restored.timeframe == original.timeframe
        assert restored.pair_whitelist == original.pair_whitelist
        assert restored.stake_currency == original.stake_currency
        assert restored.stoploss == original.stoploss
        assert restored.max_open_trades == original.max_open_trades
        assert restored.trailing_stop == original.trailing_stop

"""
Configuration schema for Freqtrade Adapter.

Defines the configuration structure for loading and running Freqtrade strategies
within LIBRA. Handles mapping between LIBRA config format and Freqtrade's
native configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass
class FreqtradeAdapterConfig:
    """
    Configuration for the Freqtrade adapter.

    This configuration maps LIBRA's settings to Freqtrade's expected format.

    Attributes:
        strategy_name: Name of the Freqtrade strategy class to load.
        config_path: Path to Freqtrade JSON configuration file.
        strategy_path: Directory containing strategy Python files.
        timeframe: Trading timeframe (e.g., "5m", "1h", "1d").
        pair_whitelist: List of trading pairs to trade.
        stake_currency: Quote currency for trading (e.g., "USDT").
        stake_amount: Amount to stake per trade (or "unlimited").
        dry_run: Whether to run in paper trading mode.
        startup_candle_count: Number of candles needed for indicator warmup.
        max_open_trades: Maximum concurrent open trades.
        stoploss: Default stoploss as decimal (e.g., -0.10 for 10%).
        trailing_stop: Enable trailing stoploss.
        trailing_stop_positive: Profit level to start trailing.
        trailing_stop_positive_offset: Offset for trailing activation.
        use_exit_signal: Whether to use strategy exit signals.
        exit_profit_only: Only exit on profitable trades.
        freqai_enabled: Enable FreqAI machine learning.
        freqai_config: FreqAI-specific configuration.

    Examples:
        config = FreqtradeAdapterConfig(
            strategy_name="SampleStrategy",
            config_path=Path("/path/to/config.json"),
            timeframe="1h",
            pair_whitelist=["BTC/USDT", "ETH/USDT"],
        )
    """

    # Required fields
    strategy_name: str
    config_path: Path | None = None

    # Strategy location
    strategy_path: Path | None = None

    # Trading parameters
    timeframe: str = "1h"
    pair_whitelist: list[str] = field(default_factory=list)
    stake_currency: str = "USDT"
    stake_amount: Decimal | str = Decimal("100")
    dry_run: bool = True

    # Strategy settings
    startup_candle_count: int = 0
    max_open_trades: int = 3

    # Risk management
    stoploss: Decimal = Decimal("-0.10")
    trailing_stop: bool = False
    trailing_stop_positive: Decimal | None = None
    trailing_stop_positive_offset: Decimal = Decimal("0")

    # Exit configuration
    use_exit_signal: bool = True
    exit_profit_only: bool = False

    # FreqAI configuration
    freqai_enabled: bool = False
    freqai_config: dict[str, Any] = field(default_factory=dict)

    # Extra Freqtrade config options
    extra_config: dict[str, Any] = field(default_factory=dict)

    def to_freqtrade_config(self) -> dict[str, Any]:
        """
        Convert to Freqtrade-native configuration format.

        Returns:
            Dictionary in Freqtrade's expected configuration format.
        """
        config: dict[str, Any] = {
            "strategy": self.strategy_name,
            "timeframe": self.timeframe,
            "dry_run": self.dry_run,
            "stake_currency": self.stake_currency,
            "stake_amount": (
                str(self.stake_amount)
                if isinstance(self.stake_amount, Decimal)
                else self.stake_amount
            ),
            "max_open_trades": self.max_open_trades,
            "stoploss": float(self.stoploss),
            "trailing_stop": self.trailing_stop,
            "use_exit_signal": self.use_exit_signal,
            "exit_profit_only": self.exit_profit_only,
            "exchange": {
                "pair_whitelist": self.pair_whitelist,
            },
        }

        if self.strategy_path:
            config["strategy_path"] = str(self.strategy_path)

        if self.trailing_stop_positive is not None:
            config["trailing_stop_positive"] = float(self.trailing_stop_positive)
            config["trailing_stop_positive_offset"] = float(
                self.trailing_stop_positive_offset
            )

        if self.freqai_enabled:
            config["freqai"] = {
                "enabled": True,
                **self.freqai_config,
            }

        # Merge extra config
        config.update(self.extra_config)

        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FreqtradeAdapterConfig:
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            FreqtradeAdapterConfig instance.
        """
        # Handle path conversions
        if "config_path" in data and data["config_path"]:
            data["config_path"] = Path(data["config_path"])
        if "strategy_path" in data and data["strategy_path"]:
            data["strategy_path"] = Path(data["strategy_path"])

        # Handle decimal conversions
        if "stoploss" in data:
            data["stoploss"] = Decimal(str(data["stoploss"]))
        if "stake_amount" in data and data["stake_amount"] != "unlimited":
            data["stake_amount"] = Decimal(str(data["stake_amount"]))
        if "trailing_stop_positive" in data and data["trailing_stop_positive"]:
            data["trailing_stop_positive"] = Decimal(str(data["trailing_stop_positive"]))
        if "trailing_stop_positive_offset" in data:
            data["trailing_stop_positive_offset"] = Decimal(
                str(data["trailing_stop_positive_offset"])
            )

        # Extract known fields, put rest in extra_config
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {k: v for k, v in data.items() if k not in known_fields}

        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        if extra:
            filtered_data["extra_config"] = extra

        return cls(**filtered_data)

    @classmethod
    def from_freqtrade_config(
        cls,
        ft_config: dict[str, Any],
        strategy_name: str | None = None,
    ) -> FreqtradeAdapterConfig:
        """
        Create configuration from native Freqtrade config.

        Args:
            ft_config: Freqtrade configuration dictionary.
            strategy_name: Override strategy name (optional).

        Returns:
            FreqtradeAdapterConfig instance.
        """
        exchange_config = ft_config.get("exchange", {})

        return cls(
            strategy_name=strategy_name or ft_config.get("strategy", ""),
            timeframe=ft_config.get("timeframe", "1h"),
            pair_whitelist=exchange_config.get("pair_whitelist", []),
            stake_currency=ft_config.get("stake_currency", "USDT"),
            stake_amount=(
                Decimal(str(ft_config["stake_amount"]))
                if ft_config.get("stake_amount") not in (None, "unlimited")
                else "unlimited"
            ),
            dry_run=ft_config.get("dry_run", True),
            max_open_trades=ft_config.get("max_open_trades", 3),
            stoploss=Decimal(str(ft_config.get("stoploss", -0.10))),
            trailing_stop=ft_config.get("trailing_stop", False),
            trailing_stop_positive=(
                Decimal(str(ft_config["trailing_stop_positive"]))
                if ft_config.get("trailing_stop_positive")
                else None
            ),
            trailing_stop_positive_offset=Decimal(
                str(ft_config.get("trailing_stop_positive_offset", 0))
            ),
            use_exit_signal=ft_config.get("use_exit_signal", True),
            exit_profit_only=ft_config.get("exit_profit_only", False),
            freqai_enabled=ft_config.get("freqai", {}).get("enabled", False),
            freqai_config=ft_config.get("freqai", {}),
            extra_config={
                k: v
                for k, v in ft_config.items()
                if k
                not in {
                    "strategy",
                    "timeframe",
                    "exchange",
                    "stake_currency",
                    "stake_amount",
                    "dry_run",
                    "max_open_trades",
                    "stoploss",
                    "trailing_stop",
                    "trailing_stop_positive",
                    "trailing_stop_positive_offset",
                    "use_exit_signal",
                    "exit_profit_only",
                    "freqai",
                }
            },
        )

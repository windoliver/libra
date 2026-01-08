"""
Base classes for LIBRA plugins.

This module defines the base abstractions for plugins following ADR-007.
Plugins are discovered via Python's entry_points mechanism.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from decimal import Decimal

    import pandas as pd

    from libra.strategies.protocol import BacktestResult, Signal


class PluginMetadata(msgspec.Struct, frozen=True):
    """Metadata describing a plugin."""

    name: str
    version: str
    description: str
    author: str
    requires: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        name: str,
        version: str,
        description: str,
        author: str = "LIBRA",
        requires: list[str] | None = None,
    ) -> PluginMetadata:
        """Create plugin metadata."""
        return cls(
            name=name,
            version=version,
            description=description,
            author=author,
            requires=tuple(requires or []),
        )


class StrategyPlugin(ABC):
    """
    Base class for strategy plugins.

    Strategy plugins adapt external frameworks (Freqtrade, Hummingbot, etc.)
    to the LIBRA Strategy protocol.

    Example:
        class FreqtradeAdapter(StrategyPlugin):
            @classmethod
            def metadata(cls) -> PluginMetadata:
                return PluginMetadata.create(
                    name="freqtrade-adapter",
                    version="0.1.0",
                    description="Freqtrade strategy adapter",
                    requires=["freqtrade>=2024.1"],
                )

            @property
            def name(self) -> str:
                return "freqtrade"

            @property
            def symbols(self) -> list[str]:
                return self._symbols

            async def initialize(self, config: dict) -> None:
                # Load Freqtrade strategy
                ...
    """

    @classmethod
    @abstractmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Symbols this strategy trades."""
        ...

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize strategy with configuration."""
        ...

    @abstractmethod
    async def on_data(self, data: pd.DataFrame) -> Signal | None:
        """
        Process new market data and generate signal.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Signal if action needed, None otherwise
        """
        ...

    async def on_tick(self, tick: dict) -> Signal | None:
        """
        Process real-time tick (optional, for HFT strategies).

        Default implementation returns None (no signal).
        """
        return None

    async def on_fill(self, order_result: dict) -> None:
        """Called when an order is filled (optional)."""
        pass

    async def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: Decimal,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Default implementation raises NotImplementedError.
        Override in subclass if backtesting is supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support backtesting"
        )

    async def optimize(
        self,
        data: pd.DataFrame,
        param_space: dict,
        metric: str = "sharpe_ratio",
    ) -> dict:
        """
        Optimize strategy parameters.

        Default implementation raises NotImplementedError.
        Override in subclass if optimization is supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support optimization"
        )

    async def shutdown(self) -> None:
        """Clean up resources (optional)."""
        pass

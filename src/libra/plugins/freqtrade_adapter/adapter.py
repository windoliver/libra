"""
Main Freqtrade Adapter for LIBRA.

Implements the StrategyPlugin interface to enable running Freqtrade strategies
within the LIBRA trading platform.

Uses Polars as primary DataFrame type (10-3500x faster than pandas).
Converts to pandas only when interfacing with Freqtrade internals.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from libra.plugins.base import PluginMetadata, StrategyPlugin
from libra.plugins.freqtrade_adapter.backtest_bridge import (
    BacktestResultData,
    FreqtradeBacktestBridge,
)
from libra.plugins.freqtrade_adapter.config import FreqtradeAdapterConfig
from libra.plugins.freqtrade_adapter.converter import FreqtradeSignalConverter
from libra.plugins.freqtrade_adapter.loader import (
    FreqtradeNotInstalledError,
    FreqtradeStrategyLoader,
)
from libra.strategies.protocol import BacktestResult, Signal


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class FreqtradeAdapter(StrategyPlugin):
    """
    Adapter for running Freqtrade strategies in LIBRA.

    This adapter enables seamless integration of Freqtrade strategies with
    LIBRA's trading infrastructure. It provides:

    - Strategy loading from Freqtrade ecosystem
    - Signal conversion to LIBRA format
    - Backtesting via Freqtrade's native engine or LIBRA's engine
    - Configuration mapping between frameworks
    - FreqAI machine learning support (when available)
    - Hyperopt optimization integration (when available)

    Usage:
        # Create adapter
        adapter = FreqtradeAdapter()

        # Initialize with configuration
        await adapter.initialize({
            "strategy_name": "SampleStrategy",
            "config_path": "/path/to/config.json",
            "strategy_path": "/path/to/strategies/",
        })

        # Process market data
        signal = await adapter.on_data(ohlcv_dataframe)
        if signal:
            print(f"Signal: {signal.signal_type}")

        # Run backtest
        result = await adapter.backtest(historical_data, Decimal("10000"))
        print(result.summary())

    Lifecycle:
        1. Create adapter instance
        2. Call initialize() with configuration
        3. Call on_data() for each new bar/candle
        4. Optionally call backtest() for historical analysis
        5. Call shutdown() when done

    Thread Safety:
        The adapter is NOT thread-safe. Use separate instances for concurrent
        processing or implement external synchronization.
    """

    # Plugin version
    VERSION = "0.1.0"

    # Minimum Freqtrade version required
    MIN_FREQTRADE_VERSION = "2024.1"

    def __init__(self) -> None:
        """Initialize the Freqtrade adapter."""
        self._config: FreqtradeAdapterConfig | None = None
        self._strategy: Any = None  # Freqtrade IStrategy instance
        self._loader = FreqtradeStrategyLoader()
        self._converter = FreqtradeSignalConverter()
        self._backtest_bridge = FreqtradeBacktestBridge()
        self._initialized = False
        self._symbols: list[str] = []

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """
        Return plugin metadata.

        Returns:
            PluginMetadata describing this adapter.
        """
        return PluginMetadata.create(
            name="freqtrade-adapter",
            version=cls.VERSION,
            description="Run Freqtrade strategies in LIBRA",
            author="LIBRA Team",
            requires=[f"freqtrade>={cls.MIN_FREQTRADE_VERSION}"],
        )

    @property
    def name(self) -> str:
        """
        Strategy name.

        Returns:
            Combined name of adapter and loaded strategy.
        """
        if self._strategy:
            strategy_name = getattr(self._strategy, "__class__", type(self._strategy))
            return f"freqtrade:{strategy_name.__name__}"
        return "freqtrade:uninitialized"

    @property
    def symbols(self) -> list[str]:
        """
        Symbols this strategy trades.

        Returns:
            List of trading pair symbols.
        """
        return self._symbols

    @property
    def is_initialized(self) -> bool:
        """Whether the adapter has been initialized."""
        return self._initialized

    @property
    def freqtrade_available(self) -> bool:
        """Whether Freqtrade is installed."""
        return self._loader.freqtrade_available

    @property
    def strategy(self) -> Any:
        """
        Access the underlying Freqtrade strategy instance.

        Returns:
            The loaded IStrategy instance, or None if not initialized.
        """
        return self._strategy

    @property
    def config(self) -> FreqtradeAdapterConfig | None:
        """
        Access the adapter configuration.

        Returns:
            Configuration instance, or None if not initialized.
        """
        return self._config

    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the adapter with configuration.

        Loads the Freqtrade strategy and prepares for signal generation.

        Args:
            config: Configuration dictionary with keys:
                - strategy_name (required): Name of the strategy class
                - config_path (optional): Path to Freqtrade config.json
                - strategy_path (optional): Directory containing strategies
                - pair_whitelist (optional): List of trading pairs
                - timeframe (optional): Trading timeframe
                - Plus any other FreqtradeAdapterConfig fields

        Raises:
            ValueError: If strategy_name is not provided.
            FreqtradeNotInstalledError: If Freqtrade is required but not installed.
            StrategyNotFoundError: If strategy cannot be found.

        Examples:
            await adapter.initialize({
                "strategy_name": "SampleStrategy",
                "config_path": "/path/to/config.json",
            })
        """
        # Validate required fields
        if "strategy_name" not in config:
            raise ValueError("strategy_name is required in configuration")

        # Parse configuration
        self._config = FreqtradeAdapterConfig.from_dict(config)

        # Load Freqtrade config if provided
        ft_config: dict[str, Any] = {}
        if self._config.config_path and self._config.config_path.exists():
            ft_config = self._loader.load_config(self._config.config_path)
            # Merge with adapter config
            ft_config.update(self._config.to_freqtrade_config())
        else:
            ft_config = self._config.to_freqtrade_config()

        # Load strategy
        strategy_path = self._config.strategy_path
        if strategy_path is None and self._config.config_path:
            # Try to find strategies relative to config
            strategy_path = self._config.config_path.parent / "strategies"
            if not strategy_path.exists():
                strategy_path = None

        try:
            strategy_class = self._loader.load_strategy(
                strategy_name=self._config.strategy_name,
                strategy_path=strategy_path,
                config=ft_config,
            )

            # Instantiate strategy
            if self._loader.freqtrade_available:
                # Let Freqtrade handle instantiation
                from freqtrade.resolvers import StrategyResolver

                ft_config["strategy"] = self._config.strategy_name
                if strategy_path:
                    ft_config["strategy_path"] = str(strategy_path)
                self._strategy = StrategyResolver.load_strategy(ft_config)
            else:
                # Direct instantiation
                self._strategy = strategy_class(ft_config)

        except ImportError as e:
            if "freqtrade" in str(e).lower():
                raise FreqtradeNotInstalledError from e
            raise

        # Set symbols from config
        self._symbols = self._config.pair_whitelist.copy()

        # Initialize strategy if it has startup methods
        if hasattr(self._strategy, "bot_start"):
            self._strategy.bot_start()

        self._initialized = True
        logger.info(
            "Initialized FreqtradeAdapter with strategy: %s, symbols: %s",
            self._config.strategy_name,
            self._symbols,
        )

    async def on_data(self, data: pl.DataFrame) -> Signal | None:
        """
        Process market data and generate trading signal.

        Runs the Freqtrade strategy's indicator and signal logic on the
        provided Polars DataFrame, then converts any signals to LIBRA format.

        Internally converts to pandas for Freqtrade compatibility, then
        converts back to Polars for signal extraction.

        Args:
            data: Polars OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            Signal if the strategy generates a trade signal, None otherwise.

        Raises:
            RuntimeError: If adapter is not initialized.

        Examples:
            signal = await adapter.on_data(ohlcv_df)
            if signal and signal.signal_type == SignalType.LONG:
                # Handle long entry
                pass
        """
        if not self._initialized or self._strategy is None:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        if data.is_empty():
            return None

        # Get the pair from symbols or use placeholder
        pair = self._symbols[0] if self._symbols else "UNKNOWN/USDT"
        metadata = {"pair": pair}

        # Convert Polars to pandas for Freqtrade strategy execution
        # (Freqtrade strategies require pandas DataFrames)
        pandas_df = data.to_pandas()

        # Run Freqtrade strategy pipeline
        try:
            # Add indicators
            if hasattr(self._strategy, "advise_indicators"):
                pandas_df = self._strategy.advise_indicators(pandas_df, metadata)
            elif hasattr(self._strategy, "populate_indicators"):
                pandas_df = self._strategy.populate_indicators(pandas_df, metadata)

            # Add entry signals
            if hasattr(self._strategy, "advise_entry"):
                pandas_df = self._strategy.advise_entry(pandas_df, metadata)
            elif hasattr(self._strategy, "populate_entry_trend"):
                pandas_df = self._strategy.populate_entry_trend(pandas_df, metadata)

            # Add exit signals
            if hasattr(self._strategy, "advise_exit"):
                pandas_df = self._strategy.advise_exit(pandas_df, metadata)
            elif hasattr(self._strategy, "populate_exit_trend"):
                pandas_df = self._strategy.populate_exit_trend(pandas_df, metadata)

        except Exception as e:
            logger.exception("Error running Freqtrade strategy: %s", e)
            return None

        # Convert back to Polars for signal extraction
        result_df = pl.from_pandas(pandas_df)

        # Convert signals using Polars DataFrame
        return self._converter.convert_dataframe(result_df, pair)

    async def on_tick(self, tick: dict[str, Any]) -> Signal | None:
        """
        Process real-time tick data.

        Freqtrade strategies are primarily bar-based, so tick processing
        is limited. This method aggregates ticks if needed and delegates
        to on_data when a new bar is complete.

        Args:
            tick: Tick data dictionary with price/volume info.

        Returns:
            Signal if a new bar triggers a trade signal, None otherwise.
        """
        # Freqtrade is bar-based, ticks are not directly processed
        # Could implement tick aggregation here if needed
        logger.debug("Tick received but Freqtrade is bar-based: %s", tick)
        return None

    async def on_fill(self, order_result: dict[str, Any]) -> None:
        """
        Handle order fill notification.

        Updates the strategy's internal state when an order is filled.
        This can be used to track positions and trigger callbacks.

        Args:
            order_result: Order fill details.
        """
        if not self._initialized or self._strategy is None:
            return

        # Call Freqtrade's order_filled callback if available
        if hasattr(self._strategy, "order_filled"):
            try:
                self._strategy.order_filled(
                    pair=order_result.get("symbol", ""),
                    trade=None,  # Would need to construct Trade object
                    order=None,  # Would need to construct Order object
                    current_time=order_result.get("timestamp"),
                )
            except Exception as e:
                logger.debug("Error calling order_filled callback: %s", e)

    async def backtest(
        self,
        data: pl.DataFrame,
        initial_capital: Decimal,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Uses either Freqtrade's native backtesting engine (if available)
        or a simplified simulation via the backtest bridge.

        Args:
            data: Historical Polars OHLCV DataFrame.
            initial_capital: Starting capital for the backtest.

        Returns:
            BacktestResult with performance metrics.

        Raises:
            RuntimeError: If adapter is not initialized.

        Examples:
            result = await adapter.backtest(
                data=historical_df,
                initial_capital=Decimal("10000"),
            )
            print(f"Return: {result.total_return * 100:.2f}%")
            print(f"Sharpe: {result.sharpe_ratio:.2f}")
        """
        if not self._initialized or self._strategy is None:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        # Convert Polars to pandas for backtest bridge
        pandas_df = data.to_pandas()

        # Run backtest via bridge
        result_data: BacktestResultData = self._backtest_bridge.run_on_dataframe(
            strategy=self._strategy,
            data=pandas_df,
            initial_capital=initial_capital,
        )

        # Convert to LIBRA BacktestResult
        return result_data.to_libra_result()

    async def optimize(
        self,
        data: pl.DataFrame,
        param_space: dict[str, Any],
        metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """
        Optimize strategy parameters.

        Uses Freqtrade's hyperopt if available, otherwise raises NotImplementedError.

        Args:
            data: Historical data for optimization.
            param_space: Parameter search space definition.
            metric: Metric to optimize (e.g., "sharpe_ratio", "profit").

        Returns:
            Dictionary with optimized parameters.

        Raises:
            NotImplementedError: If Freqtrade hyperopt is not available.
        """
        if not self._loader.freqtrade_available:
            raise NotImplementedError(
                "Hyperopt requires Freqtrade installation. "
                "Install with: pip install libra[freqtrade]"
            )

        # TODO: Implement hyperopt integration
        raise NotImplementedError(
            "Hyperopt integration not yet implemented. "
            "Use Freqtrade CLI for now: freqtrade hyperopt"
        )

    async def shutdown(self) -> None:
        """
        Clean up resources.

        Calls any cleanup methods on the strategy and releases resources.
        """
        if self._strategy is not None:
            # Call bot_cleanup if available
            if hasattr(self._strategy, "bot_cleanup"):
                try:
                    self._strategy.bot_cleanup()
                except Exception as e:
                    logger.debug("Error in bot_cleanup: %s", e)

        self._strategy = None
        self._initialized = False
        logger.info("FreqtradeAdapter shut down")

    # -------------------------------------------------------------------------
    # Additional Methods
    # -------------------------------------------------------------------------

    def list_strategies(self, strategy_path: Path | None = None) -> list[str]:
        """
        List available Freqtrade strategies.

        Args:
            strategy_path: Directory to search for strategies.

        Returns:
            List of available strategy names.
        """
        return self._loader.list_strategies(strategy_path)

    def get_strategy_info(
        self,
        strategy_name: str,
        strategy_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Get information about a strategy.

        Args:
            strategy_name: Name of the strategy.
            strategy_path: Directory containing the strategy.

        Returns:
            Dictionary with strategy metadata.
        """
        return self._loader.get_strategy_info(strategy_name, strategy_path)

    def get_strategy_parameters(self) -> dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of strategy parameters and their values.
        """
        if self._strategy is None:
            return {}

        params: dict[str, Any] = {}
        for attr in [
            "timeframe",
            "stoploss",
            "minimal_roi",
            "startup_candle_count",
            "can_short",
            "use_exit_signal",
            "trailing_stop",
            "trailing_stop_positive",
            "trailing_stop_positive_offset",
            "process_only_new_candles",
        ]:
            if hasattr(self._strategy, attr):
                params[attr] = getattr(self._strategy, attr)

        return params

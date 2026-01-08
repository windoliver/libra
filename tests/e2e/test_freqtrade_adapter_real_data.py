"""
E2E tests for Freqtrade Adapter using REAL market data from Binance.

Demonstrates the Freqtrade Adapter Plugin (Phase 2.1) working with:
- Real OHLCV data from Binance public API
- Polars DataFrames (10-3500x faster than pandas)
- Signal conversion to LIBRA format
- Backtesting with real market data

Uses direct Binance API calls (no CCXT dependency) for maximum portability.
"""

from __future__ import annotations

import json
import urllib.request
from decimal import Decimal
from typing import Any

import polars as pl
import pytest

from libra.plugins.freqtrade_adapter.adapter import FreqtradeAdapter
from libra.plugins.freqtrade_adapter.config import FreqtradeAdapterConfig
from libra.plugins.freqtrade_adapter.converter import FreqtradeSignalConverter
from libra.strategies.protocol import SignalType


# =============================================================================
# Real Data Fetchers (Direct Binance API)
# =============================================================================


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 100,
) -> pl.DataFrame:
    """
    Fetch real OHLCV data from Binance public API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT").
        interval: Candle interval (e.g., "1h", "4h", "1d").
        limit: Number of candles to fetch.

    Returns:
        Polars DataFrame with OHLCV data.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw_data = json.loads(response.read().decode())
    except Exception as e:
        pytest.skip(f"Could not fetch Binance data: {e}")

    # Convert to Polars DataFrame
    # Binance kline format: [open_time, open, high, low, close, volume, ...]
    return pl.DataFrame({
        "timestamp": [int(k[0]) for k in raw_data],
        "open": [float(k[1]) for k in raw_data],
        "high": [float(k[2]) for k in raw_data],
        "low": [float(k[3]) for k in raw_data],
        "close": [float(k[4]) for k in raw_data],
        "volume": [float(k[5]) for k in raw_data],
    })


# =============================================================================
# Sample Freqtrade-style Strategy (Pure Python, no FT dependency)
# =============================================================================


class SampleSMACrossStrategy:
    """
    Sample SMA Crossover strategy in Freqtrade style.

    This strategy demonstrates Freqtrade's interface without requiring
    the actual Freqtrade package to be installed.

    Uses:
    - 10-period fast SMA
    - 30-period slow SMA
    - Long when fast crosses above slow
    - Exit when fast crosses below slow
    """

    # Freqtrade-style attributes
    timeframe = "1h"
    stoploss = -0.10
    minimal_roi = {"0": 0.05}
    startup_candle_count = 30
    can_short = False
    use_exit_signal = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize strategy."""
        self.config = config or {}
        self.fast_period = 10
        self.slow_period = 30

    def populate_indicators(
        self,
        dataframe: Any,  # pandas DataFrame
        metadata: dict[str, Any],
    ) -> Any:
        """Add SMA indicators to dataframe."""
        # Calculate SMAs using pandas (Freqtrade style)
        dataframe["sma_fast"] = dataframe["close"].rolling(self.fast_period).mean()
        dataframe["sma_slow"] = dataframe["close"].rolling(self.slow_period).mean()

        # Previous values for crossover detection
        dataframe["sma_fast_prev"] = dataframe["sma_fast"].shift(1)
        dataframe["sma_slow_prev"] = dataframe["sma_slow"].shift(1)

        return dataframe

    def populate_entry_trend(
        self,
        dataframe: Any,  # pandas DataFrame
        metadata: dict[str, Any],
    ) -> Any:
        """Add entry signals."""
        # Initialize signal columns
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        # Golden cross: fast SMA crosses above slow SMA
        conditions = (
            (dataframe["sma_fast"] > dataframe["sma_slow"])
            & (dataframe["sma_fast_prev"] <= dataframe["sma_slow_prev"])
            & (dataframe["volume"] > 0)  # Ensure there's volume
        )

        dataframe.loc[conditions, "enter_long"] = 1
        dataframe.loc[conditions, "enter_tag"] = "golden_cross"

        return dataframe

    def populate_exit_trend(
        self,
        dataframe: Any,  # pandas DataFrame
        metadata: dict[str, Any],
    ) -> Any:
        """Add exit signals."""
        # Initialize signal columns
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        # Death cross: fast SMA crosses below slow SMA
        conditions = (
            (dataframe["sma_fast"] < dataframe["sma_slow"])
            & (dataframe["sma_fast_prev"] >= dataframe["sma_slow_prev"])
        )

        dataframe.loc[conditions, "exit_long"] = 1
        dataframe.loc[conditions, "exit_tag"] = "death_cross"

        return dataframe


class SampleRSIStrategy:
    """
    Sample RSI-based strategy in Freqtrade style.

    Uses:
    - 14-period RSI
    - Long when RSI < 30 (oversold)
    - Exit when RSI > 70 (overbought)
    """

    timeframe = "1h"
    stoploss = -0.08
    minimal_roi = {"0": 0.03}
    startup_candle_count = 20
    can_short = True
    use_exit_signal = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize strategy."""
        self.config = config or {}
        self.rsi_period = 14
        self.rsi_buy = 30
        self.rsi_sell = 70

    def _calculate_rsi(self, series: Any, period: int = 14) -> Any:
        """Calculate RSI using pandas."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def populate_indicators(
        self,
        dataframe: Any,
        metadata: dict[str, Any],
    ) -> Any:
        """Add RSI indicator."""
        dataframe["rsi"] = self._calculate_rsi(dataframe["close"], self.rsi_period)
        return dataframe

    def populate_entry_trend(
        self,
        dataframe: Any,
        metadata: dict[str, Any],
    ) -> Any:
        """Add entry signals."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # Long on oversold
        long_conditions = (dataframe["rsi"] < self.rsi_buy) & (dataframe["volume"] > 0)
        dataframe.loc[long_conditions, "enter_long"] = 1
        dataframe.loc[long_conditions, "enter_tag"] = "rsi_oversold"

        # Short on overbought (if enabled)
        short_conditions = (dataframe["rsi"] > self.rsi_sell) & (dataframe["volume"] > 0)
        dataframe.loc[short_conditions, "enter_short"] = 1
        dataframe.loc[short_conditions, "enter_tag"] = "rsi_overbought"

        return dataframe

    def populate_exit_trend(
        self,
        dataframe: Any,
        metadata: dict[str, Any],
    ) -> Any:
        """Add exit signals."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        # Exit long on overbought
        exit_long_conditions = dataframe["rsi"] > self.rsi_sell
        dataframe.loc[exit_long_conditions, "exit_long"] = 1
        dataframe.loc[exit_long_conditions, "exit_tag"] = "rsi_overbought"

        # Exit short on oversold
        exit_short_conditions = dataframe["rsi"] < self.rsi_buy
        dataframe.loc[exit_short_conditions, "exit_short"] = 1
        dataframe.loc[exit_short_conditions, "exit_tag"] = "rsi_oversold"

        return dataframe


# =============================================================================
# Test: Signal Converter with Real Data
# =============================================================================


class TestSignalConverterRealData:
    """Test FreqtradeSignalConverter with real market data."""

    @pytest.fixture
    def converter(self) -> FreqtradeSignalConverter:
        """Create converter instance."""
        return FreqtradeSignalConverter()

    @pytest.fixture
    def real_data(self) -> pl.DataFrame:
        """Fetch real market data."""
        return fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=100)

    def test_convert_real_data_no_signals(
        self,
        converter: FreqtradeSignalConverter,
        real_data: pl.DataFrame,
    ) -> None:
        """Test conversion with real data but no signals."""
        # Raw data has no signal columns
        result = converter.convert_dataframe(real_data, "BTC/USDT")
        assert result is None  # No signals in raw data

    def test_convert_with_added_signals(
        self,
        converter: FreqtradeSignalConverter,
        real_data: pl.DataFrame,
    ) -> None:
        """Test conversion with manually added signals."""
        # Add signal columns
        df = real_data.with_columns([
            pl.lit(0).alias("enter_long"),
            pl.lit(0).alias("exit_long"),
        ])

        # Set last row to have enter_long signal
        df = df.with_columns(
            pl.when(pl.arange(0, df.height) == df.height - 1)
            .then(1)
            .otherwise(pl.col("enter_long"))
            .alias("enter_long")
        )

        result = converter.convert_dataframe(df, "BTC/USDT")

        assert result is not None
        assert result.signal_type == SignalType.LONG
        assert result.symbol == "BTC/USDT"
        assert result.price is not None
        assert result.metadata["source"] == "freqtrade"

    def test_data_integrity(self, real_data: pl.DataFrame) -> None:
        """Verify real data has expected structure."""
        assert real_data.height >= 50  # At least 50 rows
        assert "open" in real_data.columns
        assert "high" in real_data.columns
        assert "low" in real_data.columns
        assert "close" in real_data.columns
        assert "volume" in real_data.columns

        # Verify data is valid
        close_min = real_data["close"].min()
        volume_min = real_data["volume"].min()
        assert close_min is not None and float(close_min) > 0
        assert volume_min is not None and float(volume_min) >= 0
        # High >= Low for all rows
        assert (real_data["high"] >= real_data["low"]).all()


# =============================================================================
# Test: Full Adapter Workflow with Real Data
# =============================================================================


class TestFreqtradeAdapterRealData:
    """Test FreqtradeAdapter with real market data."""

    @pytest.fixture
    def real_data(self) -> pl.DataFrame:
        """Fetch real market data for testing."""
        return fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=200)

    @pytest.fixture
    def adapter_with_sma_strategy(self) -> FreqtradeAdapter:
        """Create adapter with mocked SMA strategy."""
        adapter = FreqtradeAdapter()
        # Manually set up the adapter without full initialization
        adapter._config = FreqtradeAdapterConfig(
            strategy_name="SampleSMACrossStrategy",
            pair_whitelist=["BTC/USDT"],
            timeframe="1h",
        )
        adapter._strategy = SampleSMACrossStrategy()
        adapter._converter = FreqtradeSignalConverter()
        adapter._symbols = ["BTC/USDT"]
        adapter._initialized = True
        return adapter

    @pytest.fixture
    def adapter_with_rsi_strategy(self) -> FreqtradeAdapter:
        """Create adapter with mocked RSI strategy."""
        adapter = FreqtradeAdapter()
        adapter._config = FreqtradeAdapterConfig(
            strategy_name="SampleRSIStrategy",
            pair_whitelist=["BTC/USDT"],
            timeframe="1h",
        )
        adapter._strategy = SampleRSIStrategy()
        adapter._converter = FreqtradeSignalConverter()
        adapter._symbols = ["BTC/USDT"]
        adapter._initialized = True
        return adapter

    @pytest.mark.asyncio
    async def test_on_data_with_real_market_data(
        self,
        adapter_with_sma_strategy: FreqtradeAdapter,
        real_data: pl.DataFrame,
    ) -> None:
        """Test on_data processing with real market data."""
        # Process real data through the adapter
        signal = await adapter_with_sma_strategy.on_data(real_data)

        # Signal may or may not be generated depending on market conditions
        if signal is not None:
            assert signal.symbol == "BTC/USDT"
            assert signal.signal_type in [
                SignalType.LONG,
                SignalType.CLOSE_LONG,
            ]
            assert signal.price is not None
            assert signal.metadata["source"] == "freqtrade"
            print(f"Signal generated: {signal.signal_type} at {signal.price}")
        else:
            print("No signal generated (market conditions don't match strategy)")

    @pytest.mark.asyncio
    async def test_rsi_strategy_with_real_data(
        self,
        adapter_with_rsi_strategy: FreqtradeAdapter,
        real_data: pl.DataFrame,
    ) -> None:
        """Test RSI strategy with real market data."""
        signal = await adapter_with_rsi_strategy.on_data(real_data)

        if signal is not None:
            assert signal.symbol == "BTC/USDT"
            assert signal.signal_type in [
                SignalType.LONG,
                SignalType.SHORT,
                SignalType.CLOSE_LONG,
                SignalType.CLOSE_SHORT,
            ]
            print(f"RSI Signal: {signal.signal_type}, tag: {signal.metadata.get('enter_tag', signal.metadata.get('exit_tag'))}")

    @pytest.mark.asyncio
    async def test_backtest_with_real_data(
        self,
        adapter_with_sma_strategy: FreqtradeAdapter,
        real_data: pl.DataFrame,
    ) -> None:
        """Test backtesting with real market data."""
        result = await adapter_with_sma_strategy.backtest(
            data=real_data,
            initial_capital=Decimal("10000"),
        )

        # Verify result structure
        assert result is not None
        assert result.initial_capital == Decimal("10000")
        assert result.final_capital >= Decimal("0")

        # Print summary
        print("\n=== Backtest Results (Real BTC Data) ===")
        print(f"Total Return: {result.total_return * 100:.2f}%")
        print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate * 100:.1f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Final Capital: ${result.final_capital:.2f}")

    @pytest.mark.asyncio
    async def test_multiple_symbols_real_data(self) -> None:
        """Test with multiple trading pairs."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        results = []

        for symbol in symbols:
            try:
                data = fetch_binance_klines(symbol=symbol, interval="1h", limit=100)

                adapter = FreqtradeAdapter()
                adapter._config = FreqtradeAdapterConfig(
                    strategy_name="SampleSMACrossStrategy",
                    pair_whitelist=[symbol.replace("USDT", "/USDT")],
                )
                adapter._strategy = SampleSMACrossStrategy()
                adapter._converter = FreqtradeSignalConverter()
                adapter._symbols = [symbol.replace("USDT", "/USDT")]
                adapter._initialized = True

                signal = await adapter.on_data(data)
                results.append({
                    "symbol": symbol,
                    "signal": signal.signal_type.value if signal else "NONE",
                    "rows": data.height,
                })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "signal": f"ERROR: {e}",
                    "rows": 0,
                })

        print("\n=== Multi-Symbol Test Results ===")
        for r in results:
            print(f"{r['symbol']}: {r['signal']} ({r['rows']} rows)")

        # At least one symbol should have processed successfully
        successful = [r for r in results if "ERROR" not in str(r["signal"])]
        assert len(successful) >= 1


# =============================================================================
# Test: Performance with Real Data
# =============================================================================


class TestPerformanceRealData:
    """Performance tests with real market data."""

    @pytest.fixture
    def large_dataset(self) -> pl.DataFrame:
        """Fetch larger dataset for performance testing."""
        return fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=500)

    def test_polars_performance(self, large_dataset: pl.DataFrame) -> None:
        """Verify Polars operations are fast."""
        import time

        # Time DataFrame operations
        start = time.perf_counter()

        # Simulate typical operations
        for _ in range(100):
            _ = large_dataset.with_columns([
                pl.col("close").rolling_mean(20).alias("sma_20"),
                pl.col("close").rolling_mean(50).alias("sma_50"),
            ])
            _ = large_dataset.row(-1, named=True)

        elapsed = time.perf_counter() - start

        print(f"\n100 iterations of Polars operations: {elapsed*1000:.2f}ms")
        assert elapsed < 2.0  # Should complete in under 2 seconds

    @pytest.mark.asyncio
    async def test_signal_processing_performance(
        self,
        large_dataset: pl.DataFrame,
    ) -> None:
        """Test signal processing performance."""
        import time

        adapter = FreqtradeAdapter()
        adapter._config = FreqtradeAdapterConfig(
            strategy_name="SampleSMACrossStrategy",
            pair_whitelist=["BTC/USDT"],
        )
        adapter._strategy = SampleSMACrossStrategy()
        adapter._converter = FreqtradeSignalConverter()
        adapter._symbols = ["BTC/USDT"]
        adapter._initialized = True

        # Time signal processing
        start = time.perf_counter()

        for _ in range(10):
            _ = await adapter.on_data(large_dataset)

        elapsed = time.perf_counter() - start

        print(f"\n10 signal processing iterations: {elapsed*1000:.2f}ms")
        print(f"Average per iteration: {elapsed*100:.2f}ms")
        assert elapsed < 5.0  # Should complete in under 5 seconds


# =============================================================================
# Test: Configuration Integration
# =============================================================================


class TestConfigurationIntegration:
    """Test configuration with real scenarios."""

    def test_config_creation(self) -> None:
        """Test creating configuration for real trading pair."""
        config = FreqtradeAdapterConfig(
            strategy_name="SampleSMACrossStrategy",
            timeframe="1h",
            pair_whitelist=["BTC/USDT", "ETH/USDT"],
            stake_currency="USDT",
            stake_amount=Decimal("100"),
            stoploss=Decimal("-0.10"),
            max_open_trades=3,
        )

        ft_config = config.to_freqtrade_config()

        assert ft_config["strategy"] == "SampleSMACrossStrategy"
        assert ft_config["timeframe"] == "1h"
        assert ft_config["exchange"]["pair_whitelist"] == ["BTC/USDT", "ETH/USDT"]
        assert ft_config["stoploss"] == -0.10

    def test_adapter_metadata(self) -> None:
        """Test adapter metadata is correct."""
        metadata = FreqtradeAdapter.metadata()

        assert metadata.name == "freqtrade-adapter"
        assert "0.1.0" in metadata.version
        assert "freqtrade" in metadata.requires[0].lower()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

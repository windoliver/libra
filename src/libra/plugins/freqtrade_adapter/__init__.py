"""
Freqtrade Adapter Plugin for LIBRA.

This plugin enables running Freqtrade strategies within the LIBRA trading platform,
providing seamless integration with the Freqtrade ecosystem including:
- Strategy loading and execution
- Signal conversion to LIBRA format
- Backtesting bridge to native Freqtrade engine
- FreqAI machine learning support
- Hyperopt optimization integration

Usage:
    from libra.plugins.freqtrade_adapter import FreqtradeAdapter

    adapter = FreqtradeAdapter()
    await adapter.initialize({
        "strategy_name": "SampleStrategy",
        "config_path": "/path/to/config.json",
    })

    signal = await adapter.on_data(dataframe)
"""

from libra.plugins.freqtrade_adapter.adapter import FreqtradeAdapter
from libra.plugins.freqtrade_adapter.config import FreqtradeAdapterConfig
from libra.plugins.freqtrade_adapter.converter import FreqtradeSignalConverter
from libra.plugins.freqtrade_adapter.loader import FreqtradeStrategyLoader


__all__ = [
    "FreqtradeAdapter",
    "FreqtradeAdapterConfig",
    "FreqtradeSignalConverter",
    "FreqtradeStrategyLoader",
]

"""
Data layer for LIBRA trading platform.

Provides time-series database integration for:
- Tick data storage and retrieval
- OHLCV bar data
- Trade history and audit logs
- Backtest data with ASOF JOIN support

Primary implementation: QuestDB (ADR-002)
- 5M+ rows/sec ingestion via ILP
- Sub-25ms OHLCV aggregation queries
- ASOF JOIN for backtest accuracy (prevents look-ahead bias)

See: https://github.com/windoliver/libra/issues/21
"""

from libra.data.config import QuestDBConfig
from libra.data.protocol import TimeSeriesDB
from libra.data.questdb import AsyncQuestDBClient


__all__ = [
    "AsyncQuestDBClient",
    "QuestDBConfig",
    "TimeSeriesDB",
]

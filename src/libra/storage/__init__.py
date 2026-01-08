"""
Simple local storage for LIBRA.

Provides file-based storage under ~/.libra/ for:
- Strategy configurations (YAML)
- Backtest results (Parquet)
- Trade logs (Parquet)
- Runtime state (JSON)

See ADR-007 for design rationale.
"""

from libra.storage.local import LocalStorage
from libra.storage.paths import LibraPaths

__all__ = ["LocalStorage", "LibraPaths"]

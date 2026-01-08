# ADR-007: Storage Architecture - Simple Local Storage

**Status:** Accepted
**Date:** 2026-01-08
**Decision Makers:** Libra Team

## Context

The original LIBRA design document referenced NexusFS for storage. After researching Nexus and how Aquarius uses it, we found:

**What NexusFS provides:**
- Virtual filesystem server for AI agents
- Multi-tenant isolation (per-user path scoping)
- Content-addressable storage (SHA-256 deduplication)
- Versioning system
- ReBAC permissions (Zanzibar-style)
- E2B sandbox execution integration
- Multi-backend mounting (S3, GCS, Google Drive)

**What LIBRA actually needs:**
- Strategy configuration storage (YAML/JSON)
- Backtest results storage (efficient columnar format)
- Trade history (already in QuestDB)
- Time-series data (already in QuestDB)

## Decision

**Use simple local storage (`~/.libra/`) instead of NexusFS.**

LIBRA is a single-user trading platform. NexusFS features like multi-tenant isolation, ReBAC permissions, and cloud backend mounting are unnecessary overhead.

## Implementation

### Directory Structure

```
~/.libra/
├── config/
│   └── libra.yaml           # Main configuration
├── strategies/
│   ├── my_strategy/
│   │   ├── config.yaml      # Strategy parameters
│   │   └── state.json       # Runtime state (optional)
│   └── ...
├── results/
│   ├── backtests/
│   │   ├── 2026-01-08_btc_momentum.parquet
│   │   └── ...
│   └── live/
│       └── trades_2026-01.parquet
├── logs/
│   └── libra.log
└── cache/
    └── ...
```

### Storage Formats

| Data Type | Format | Rationale |
|-----------|--------|-----------|
| Configuration | YAML | Human-readable, easy to edit |
| Backtest results | Parquet | Fast, columnar, good compression |
| Trade logs | Parquet | Efficient for time-series analysis |
| Runtime state | JSON | Simple, debuggable |

### Code Design

```python
from pathlib import Path
import yaml
import polars as pl

class LocalStorage:
    """Simple local storage for LIBRA."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path.home() / ".libra"
        self._ensure_dirs()

    def save_backtest(self, name: str, results: pl.DataFrame) -> Path:
        """Save backtest results as Parquet."""
        path = self.root / "results" / "backtests" / f"{name}.parquet"
        results.write_parquet(path)
        return path

    def load_strategy_config(self, name: str) -> dict:
        """Load strategy configuration."""
        path = self.root / "strategies" / name / "config.yaml"
        return yaml.safe_load(path.read_text())
```

## Consequences

### Positive
- Simple, no external dependencies
- Easy to debug (just look at files)
- Fast (local filesystem)
- Portable (copy `~/.libra/` to backup)

### Negative
- No multi-user support (not needed for LIBRA)
- No built-in versioning (use git if needed)
- No cloud sync (can add later if needed)

### Migration Path

If LIBRA later needs multi-tenant or cloud storage:
1. Abstract storage behind a protocol
2. Add NexusFS or S3 backend
3. Existing local storage continues to work

## References

- [QuantStart - Securities Master Databases](https://www.quantstart.com/articles/Securities-Master-Databases-for-Algorithmic-Trading/)
- [Optimizing Data Handling for Backtesting](https://medium.com/@twkim323/optimizing-data-handling-for-backtesting-b62be848a314)
- Parquet format: Columnar, 10x smaller than CSV, fast random access

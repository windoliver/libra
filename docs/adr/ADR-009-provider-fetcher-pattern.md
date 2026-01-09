# ADR-009: Provider/Fetcher Pattern for Gateway Layer

**Status:** Accepted
**Date:** 2026-01-09
**Decision Makers:** Libra Team
**Issue:** [#27](https://github.com/windoliver/libra/issues/27)

## Context

LIBRA needs a consistent, testable approach for fetching data from multiple sources:
- Exchange data via CCXT (bars, quotes, orderbooks, positions, orders, trades)
- Historical data from data providers
- Alternative data sources (sentiment, on-chain)

We evaluated several patterns:

1. **Direct API calls** - Simple but leads to duplicated transformation code
2. **Repository pattern** - Focused on persistence, not well-suited for external APIs
3. **OpenBB's TET Pipeline** - Proven 3-stage pattern used by 56K+ star project

## Decision

**Adopt OpenBB's TET (Transform-Extract-Transform) Pipeline pattern.**

The pattern splits data fetching into three stages:

```
1. transform_query(params: dict) -> Query    # Validate & type params
2. extract_data(query: Query) -> raw         # Fetch from provider
3. transform_data(query, raw) -> Response    # Normalize to standard format
```

### Core Design

```python
class GatewayFetcher(ABC, Generic[Q, R]):
    """Abstract base for all data fetchers."""

    @abstractmethod
    def transform_query(self, params: dict[str, Any]) -> Q:
        """Convert params dict to typed query object."""
        ...

    @abstractmethod
    async def extract_data(self, query: Q, **kwargs: Any) -> Any:
        """Fetch raw data from provider."""
        ...

    @abstractmethod
    def transform_data(self, query: Q, raw: Any) -> R:
        """Normalize raw data to standard format."""
        ...

    async def fetch(self, **params: Any) -> R:
        """Execute the full TET pipeline."""
        query = self.transform_query(params)
        raw = await self.extract_data(query)
        return self.transform_data(query, raw)
```

### Query Types

Immutable, typed query objects for each data type:

| Query Type | Purpose | Key Fields |
|------------|---------|------------|
| `BarQuery` | OHLCV data | symbol, interval, limit, start, end |
| `TickQuery` | Quote/ticker | symbol |
| `OrderBookQuery` | Order book | symbol, depth |
| `BalanceQuery` | Account balance | currency (optional) |
| `PositionQuery` | Open positions | symbol (optional) |
| `OrderQuery` | Order history | symbol, status, order_id |
| `TradeQuery` | Trade history | symbol, limit, since |

### Response Types

High-performance response types using `msgspec.Struct`:

| Response Type | Purpose | Performance |
|---------------|---------|-------------|
| `Bar` | OHLCV candle | 4x faster serialization than dataclass |
| `Quote` | Market quote | Includes spread_bps property |
| `OrderBookSnapshot` | L2 data | With best_bid/ask/mid helpers |
| `AccountBalance` | Balance | total/available/locked |
| `AccountPosition` | Position | With notional_value, pnl_percent |
| `AccountOrder` | Order record | With is_open, fill_percent |
| `TradeRecord` | Trade/fill | Individual execution |

### Fetcher Registry

Discovery system for fetchers by gateway and data type:

```python
# Register fetchers
fetcher_registry.register("ccxt", "bar", CCXTBarFetcher)
fetcher_registry.register("ccxt", "position", CCXTPositionFetcher)
fetcher_registry.register("openbb", "bar", OpenBBBarFetcher)

# Get fetcher class
fetcher_class = fetcher_registry.get("ccxt", "bar")
fetcher = fetcher_class(exchange)
bars = await fetcher.fetch(symbol="BTC/USDT")
```

## Implementation

### CCXT Fetchers (Complete)

| Fetcher | Data Type | CCXT Method |
|---------|-----------|-------------|
| `CCXTBarFetcher` | Bar/OHLCV | `fetch_ohlcv` |
| `CCXTQuoteFetcher` | Quote/Ticker | `fetch_ticker` |
| `CCXTOrderBookFetcher` | Order Book | `fetch_order_book` |
| `CCXTBalanceFetcher` | Balance | `fetch_balance` |
| `CCXTPositionFetcher` | Positions | `fetch_positions` |
| `CCXTOrderFetcher` | Orders | `fetch_open_orders`, `fetch_closed_orders` |
| `CCXTTradeFetcher` | Trades | `fetch_my_trades` |

### File Structure

```
src/libra/gateways/
├── fetcher.py         # Base classes, query types, response types
├── ccxt_fetchers.py   # CCXT implementations
└── protocol.py        # Gateway protocol (existing)
```

### Usage Example

```python
from ccxt.pro import binance
from libra.gateways.ccxt_fetchers import (
    CCXTBarFetcher,
    CCXTPositionFetcher,
    CCXTTradeFetcher,
)

exchange = binance({"apiKey": "...", "secret": "..."})
await exchange.load_markets()

# Fetch OHLCV bars
bar_fetcher = CCXTBarFetcher(exchange)
bars = await bar_fetcher.fetch(
    symbol="BTC/USDT",
    interval="1h",
    limit=100,
)

# Fetch positions
position_fetcher = CCXTPositionFetcher(exchange)
positions = await position_fetcher.fetch()

# Fetch trade history
trade_fetcher = CCXTTradeFetcher(exchange)
trades = await trade_fetcher.fetch(symbol="BTC/USDT", limit=50)
```

## Consequences

### Positive

1. **Separation of Concerns**: Each stage handles one responsibility
2. **Testability**: Each stage can be unit tested independently
3. **Consistency**: Same pattern across all data sources
4. **Type Safety**: Typed queries and responses with validation
5. **Performance**: msgspec.Struct provides 4x faster serialization
6. **Extensibility**: Easy to add new fetchers and gateways
7. **Proven Pattern**: Battle-tested in OpenBB (56K+ GitHub stars)

### Negative

1. **Boilerplate**: Three methods per fetcher (mitigated by clear structure)
2. **Learning Curve**: Developers must understand TET pipeline
3. **Indirection**: Additional layer between caller and exchange API

### Trade-offs

| Aspect | Choice | Alternative |
|--------|--------|-------------|
| Response types | msgspec.Struct | Pydantic (slower), dataclass (less performant) |
| Query validation | dataclass frozen | Pydantic model (heavier) |
| Registry | Dict-based | Metaclass magic (harder to debug) |
| Async | Required | Sync wrapper (limits concurrency) |

## Future Work

1. **OpenBB Integration**: Add OpenBB as historical data gateway
2. **Error Handling**: Add retry logic and circuit breaker middleware
3. **Caching**: Add caching layer for frequently accessed data
4. **REST API**: Expose fetchers via FastAPI endpoints

## References

- [OpenBB Fetcher Pattern](https://github.com/OpenBB-finance/OpenBB/blob/develop/openbb_platform/core/openbb_core/provider/abstract/fetcher.py)
- [OpenBB Provider Architecture](https://docs.openbb.co/platform/development/provider)
- [NautilusTrader Adapter Pattern](https://nautilustrader.io/)
- [msgspec Performance](https://jcristharif.com/msgspec/benchmarks.html)
- Issue #27: Provider/Fetcher Pattern for Gateway Layer

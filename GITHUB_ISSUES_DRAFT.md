# Draft GitHub Issues for LIBRA

> Issues derived from OpenBB research. Ready to be created via `gh issue create`.

---

## Issue #1: Implement Provider Pattern for Gateway Layer

**Labels**: `enhancement`, `gateway`, `architecture`

### Title
[Feature] Implement Provider/Fetcher Pattern for Gateway Layer

### Description

Adopt OpenBB's proven Fetcher pattern for standardizing gateway implementations. This pattern uses a 3-stage pipeline (transform_query -> extract_data -> transform_data) that cleanly separates concerns.

### Motivation

OpenBB (56K+ stars) uses this pattern successfully for 30+ data providers. Benefits:
- Consistent interface across all gateways
- Clear separation of query building, data fetching, and normalization
- Easy testing of each stage independently
- Proven at scale

### Proposed Implementation

```python
# src/libra/gateways/base.py
from typing import Protocol, TypeVar, Generic, Any

Q = TypeVar("Q")  # Query type
R = TypeVar("R")  # Response type

class GatewayFetcher(Protocol[Q, R]):
    """Base protocol for all gateway data fetchers."""

    def transform_query(self, params: dict[str, Any]) -> Q:
        """Convert generic params to gateway-specific query."""
        ...

    async def extract_data(self, query: Q) -> Any:
        """Fetch raw data from the gateway."""
        ...

    def transform_data(self, query: Q, raw: Any) -> R:
        """Normalize raw data to standard format."""
        ...

    async def fetch(self, **params: Any) -> R:
        """Execute full pipeline."""
        query = self.transform_query(params)
        raw = await self.extract_data(query)
        return self.transform_data(query, raw)
```

### Tasks

- [ ] Define `GatewayFetcher` protocol in `src/libra/gateways/base.py`
- [ ] Create standard query types (BarQuery, TickQuery, OrderQuery)
- [ ] Create standard response types (aligned with existing Event types)
- [ ] Add unit tests for the protocol
- [ ] Update CCXT gateway to use this pattern
- [ ] Document the pattern in `ARCHITECTURE_DECISIONS.md`

### References

- OpenBB Fetcher: `openbb_platform/core/openbb_core/provider/abstract/fetcher.py`
- Current LIBRA gateway stubs: `src/libra/gateways/`

---

## Issue #2: Integrate OpenBB as Data Gateway

**Labels**: `enhancement`, `gateway`, `data`

### Title
[Feature] Add OpenBB Data Gateway for Market Data

### Description

Integrate OpenBB as a market data gateway to leverage its 30+ data provider ecosystem. This allows LIBRA to access historical bars, fundamentals, options data, and more without implementing individual provider connectors.

### Motivation

- OpenBB normalizes data from 30+ providers (FMP, Polygon, Yahoo Finance, etc.)
- Single integration = access to entire ecosystem
- Well-tested and maintained (56K+ stars)
- Complements LIBRA's focus on execution

### Proposed Implementation

```python
# src/libra/gateways/openbb/gateway.py
from openbb import obb
from libra.core.events import Event, EventType
from libra.gateways.base import GatewayFetcher

class OpenBBBarFetcher(GatewayFetcher[BarQuery, list[BarEvent]]):
    """Fetches historical bar data via OpenBB."""

    def transform_query(self, params: dict) -> BarQuery:
        return BarQuery(
            symbol=params["symbol"],
            start=params.get("start"),
            end=params.get("end"),
            interval=params.get("interval", "1d"),
            provider=params.get("provider", "yfinance"),
        )

    async def extract_data(self, query: BarQuery) -> Any:
        return obb.equity.price.historical(
            symbol=query.symbol,
            start_date=query.start,
            end_date=query.end,
            interval=query.interval,
            provider=query.provider,
        )

    def transform_data(self, query: BarQuery, raw: Any) -> list[BarEvent]:
        df = raw.to_df()
        return [self._row_to_bar_event(row) for row in df.itertuples()]
```

### Tasks

- [ ] Add `openbb` as optional dependency in `pyproject.toml`
- [ ] Create `src/libra/gateways/openbb/` package
- [ ] Implement `OpenBBBarFetcher` for historical data
- [ ] Implement `OpenBBQuoteFetcher` for current prices
- [ ] Add integration tests with mock OpenBB responses
- [ ] Document usage in README

### Dependencies

- Requires Issue #1 (Provider Pattern) to be completed first
- Requires `openbb>=4.0` package

---

## Issue #3: Add FastAPI REST API

**Labels**: `enhancement`, `api`

### Title
[Feature] Add FastAPI REST API for External Integrations

### Description

Add a REST API layer following OpenBB's pattern for external integrations. This enables web dashboards, mobile apps, and third-party tools to interact with LIBRA.

### Motivation

OpenBB exposes all functionality via REST API at `/api/v1/`. This pattern:
- Enables non-Python integrations
- Supports web dashboards
- Allows mobile app development
- Facilitates testing and debugging

### Proposed Implementation

```python
# src/libra/api/main.py
from fastapi import FastAPI
from libra.api.routers import strategies, positions, orders

app = FastAPI(
    title="LIBRA API",
    version="0.1.0",
    description="High-performance AI trading platform API",
)

app.include_router(strategies.router, prefix="/api/v1/strategies")
app.include_router(positions.router, prefix="/api/v1/positions")
app.include_router(orders.router, prefix="/api/v1/orders")
```

### Tasks

- [ ] Add `fastapi>=0.100` and `uvicorn>=0.24` to dependencies
- [ ] Create `src/libra/api/` package structure
- [ ] Implement strategy management endpoints
- [ ] Implement position query endpoints
- [ ] Implement order management endpoints
- [ ] Add OpenAPI documentation
- [ ] Add authentication middleware
- [ ] Add rate limiting

### API Endpoints (Initial)

```
GET  /api/v1/strategies          # List strategies
POST /api/v1/strategies          # Create strategy
GET  /api/v1/strategies/{id}     # Get strategy
PUT  /api/v1/strategies/{id}     # Update strategy

GET  /api/v1/positions           # List positions
GET  /api/v1/positions/{symbol}  # Get position

POST /api/v1/orders              # Submit order
GET  /api/v1/orders              # List orders
GET  /api/v1/orders/{id}         # Get order
DELETE /api/v1/orders/{id}       # Cancel order
```

---

## Issue #4: Add MCP Server for AI Agent Integration

**Labels**: `enhancement`, `ai`, `mcp`

### Title
[Feature] Add MCP Server for AI Agent Integration

### Description

Implement Model Context Protocol (MCP) server following OpenBB's pattern to enable Claude, GPT, and other AI agents to interact with LIBRA for trading analysis and execution.

### Motivation

OpenBB provides MCP server integration for AI agents. This enables:
- Claude Code integration for trading analysis
- Natural language trading commands
- AI-powered strategy development
- Automated research and analysis

### Proposed Implementation

```python
# src/libra/mcp/server.py
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("libra")

@server.tool("get_positions")
async def get_positions() -> list[TextContent]:
    """Get current portfolio positions."""
    positions = await position_manager.get_all()
    return [TextContent(text=format_positions(positions))]

@server.tool("submit_order")
async def submit_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
) -> list[TextContent]:
    """Submit a trading order."""
    order = await order_manager.submit(...)
    return [TextContent(text=f"Order submitted: {order.id}")]
```

### Tasks

- [ ] Add `mcp>=1.0` to dependencies
- [ ] Create `src/libra/mcp/` package
- [ ] Implement core tools (positions, orders, market data)
- [ ] Add risk check integration
- [ ] Document MCP server setup
- [ ] Add integration tests

### MCP Tools (Initial)

| Tool | Description |
|------|-------------|
| `get_positions` | Get current portfolio positions |
| `get_market_data` | Get current/historical market data |
| `submit_order` | Submit a trading order |
| `cancel_order` | Cancel pending order |
| `get_strategy_status` | Get strategy execution status |
| `analyze_symbol` | Run analysis on a symbol |

---

## Issue #5: Plugin Architecture for Extensions

**Labels**: `enhancement`, `architecture`

### Title
[Feature] Implement Plugin Architecture for Extensions

### Description

Create a plugin/extension system following OpenBB's pattern to allow optional features to be installed and loaded dynamically.

### Motivation

OpenBB separates core functionality from optional extensions:
- **Standard extensions**: Included with core
- **Optional extensions**: Separate install
- **Community extensions**: Third-party

This allows:
- Smaller core package
- Optional heavy dependencies
- Community contributions
- Custom enterprise extensions

### Proposed Structure

```
src/libra/
├── core/               # Core (always included)
├── gateways/
│   ├── base.py        # Gateway protocol (core)
│   ├── paper/         # Paper trading (core)
│   └── ccxt/          # Optional extension
├── extensions/
│   ├── openbb/        # OpenBB integration (optional)
│   ├── charting/      # Visualization (optional)
│   └── ai/            # AI agents (optional)
```

### Tasks

- [ ] Define extension protocol/interface
- [ ] Implement extension discovery mechanism
- [ ] Create entry points in `pyproject.toml`
- [ ] Add extension loading on startup
- [ ] Document extension development
- [ ] Migrate CCXT gateway to extension

### pyproject.toml Changes

```toml
[project.optional-dependencies]
ccxt = ["ccxt>=4.0"]
openbb = ["openbb>=4.0"]
ai = ["langgraph>=0.2", "langchain>=0.2"]
charting = ["plotly>=5.0"]
all = ["libra[ccxt,openbb,ai,charting]"]

[project.entry-points."libra.gateways"]
ccxt = "libra.gateways.ccxt:CCXTGateway"
paper = "libra.gateways.paper:PaperGateway"

[project.entry-points."libra.extensions"]
openbb = "libra.extensions.openbb:OpenBBExtension"
```

---

## Issue #6: Add CONTRIBUTING.md

**Labels**: `documentation`

### Title
[Docs] Add CONTRIBUTING.md Guide

### Description

Create a comprehensive contribution guide following OpenBB's practices.

### Tasks

- [ ] Create `CONTRIBUTING.md`
- [ ] Document code standards
- [ ] Document PR process
- [ ] Document testing requirements
- [ ] Document commit message format
- [ ] Add development setup instructions

---

## Quick Create Commands

```bash
# Issue #1: Provider Pattern
gh issue create \
  --title "[Feature] Implement Provider/Fetcher Pattern for Gateway Layer" \
  --label "enhancement,gateway,architecture" \
  --body-file .github/issues/provider-pattern.md

# Issue #2: OpenBB Gateway
gh issue create \
  --title "[Feature] Add OpenBB Data Gateway for Market Data" \
  --label "enhancement,gateway,data" \
  --body-file .github/issues/openbb-gateway.md

# Issue #3: FastAPI
gh issue create \
  --title "[Feature] Add FastAPI REST API for External Integrations" \
  --label "enhancement,api" \
  --body-file .github/issues/fastapi.md

# Issue #4: MCP Server
gh issue create \
  --title "[Feature] Add MCP Server for AI Agent Integration" \
  --label "enhancement,ai,mcp" \
  --body-file .github/issues/mcp-server.md

# Issue #5: Plugin Architecture
gh issue create \
  --title "[Feature] Implement Plugin Architecture for Extensions" \
  --label "enhancement,architecture" \
  --body-file .github/issues/plugin-architecture.md

# Issue #6: CONTRIBUTING.md
gh issue create \
  --title "[Docs] Add CONTRIBUTING.md Guide" \
  --label "documentation" \
  --body-file .github/issues/contributing.md
```

---

## Priority Order

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 1 | Provider Pattern | Medium | Foundation for all gateways |
| 2 | OpenBB Gateway | Low | Instant 30+ data providers |
| 3 | Plugin Architecture | Medium | Extensibility |
| 4 | FastAPI REST API | Medium | External integrations |
| 5 | MCP Server | Medium | AI agent support |
| 6 | CONTRIBUTING.md | Low | Community |

# LIBRA

High-performance AI trading platform with LMAX Disruptor-inspired message bus.

## Features

- **Event-Driven Architecture**: Priority-based message bus (Risk > Orders > Signals > Market Data)
- **High Performance**: Target <100μs dispatch latency, 100K+ events/sec
- **Multi-Exchange**: 100+ exchanges via CCXT
- **Time-Series Database**: QuestDB for tick data, OHLCV, and trade history
- **AI-Powered**: Multi-agent LLM intelligence for strategy analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/windoliver/libra.git
cd libra

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with all dependencies
pip install -e ".[dev,database,exchange,tui]"
```

## Quick Start

### 1. Start the Database

```bash
# Start QuestDB (requires Docker)
./scripts/start-db.sh

# Verify it's running
open http://localhost:9000  # Web console
```

### 2. Run the TUI

```bash
python -m libra.tui
```

### 3. Use Programmatically

```python
import asyncio
from libra.core import MessageBus, Event, EventType
from libra.data import AsyncQuestDBClient, QuestDBConfig

async def main():
    # Connect to database
    config = QuestDBConfig.docker()
    async with AsyncQuestDBClient(config) as db:
        await db.create_tables()

        # Query historical data
        bars = await db.get_bars("BTC/USDT", "1h", start, end)

        # Get DataFrame for analysis
        df = await db.get_bars_df("BTC/USDT", "1h", start, end)

asyncio.run(main())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LIBRA PLATFORM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         MESSAGE BUS                                  │    │
│  │     Priority: RISK(0) > ORDERS(1) > SIGNALS(2) > MARKET_DATA(3)     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│           │              │              │              │                     │
│      ┌────▼────┐   ┌─────▼────┐   ┌────▼────┐   ┌────▼────┐                 │
│      │  Risk   │   │ Executor │   │Strategy │   │ Gateway │                 │
│      │ Manager │   │          │   │         │   │ (CCXT)  │                 │
│      └─────────┘   └──────────┘   └─────────┘   └─────────┘                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         DATA LAYER                                   │    │
│  │                                                                      │    │
│  │   QuestDB (Time-Series)     │    Async Python Client                │    │
│  │   ├─ Ticks (ILP ingestion)  │    ├─ asyncpg (queries)               │    │
│  │   ├─ OHLCV bars             │    ├─ questdb.Sender (ingestion)      │    │
│  │   ├─ Trade history          │    └─ Polars DataFrames               │    │
│  │   └─ ASOF JOIN (backtest)   │                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Database Setup

LIBRA uses [QuestDB](https://questdb.com) for high-performance time-series storage.

### Using Docker (Recommended)

```bash
# Start database
./scripts/start-db.sh

# Stop database
./scripts/start-db.sh --stop

# Reset all data
./scripts/start-db.sh --reset
```

### Connection Details

| Service | Port | URL |
|---------|------|-----|
| Web Console | 9000 | http://localhost:9000 |
| ILP Ingestion | 9009 | High-throughput writes |
| PostgreSQL | 8812 | SQL queries |

**Credentials**: `libra` / `libra`

### Configuration Options

```python
from libra.data import QuestDBConfig

# Docker Compose (default)
config = QuestDBConfig.docker()

# From environment variables
config = QuestDBConfig.from_env()

# Custom configuration
config = QuestDBConfig(
    host="questdb.example.com",
    username="libra",
    password="secret",
    use_tls=True,
)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QUESTDB_HOST` | localhost | Database host |
| `QUESTDB_PG_PORT` | 8812 | PostgreSQL port |
| `QUESTDB_ILP_PORT` | 9009 | ILP ingestion port |
| `QUESTDB_USERNAME` | None | Username |
| `QUESTDB_PASSWORD` | None | Password |
| `QUESTDB_USE_TLS` | false | Enable TLS |

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=libra --cov-report=html
```

### Project Structure

```
libra/
├── src/libra/
│   ├── core/           # Message bus, events, kernel
│   ├── clients/        # DataClient, ExecutionClient
│   ├── data/           # QuestDB integration
│   ├── gateways/       # Exchange connectivity
│   ├── strategies/     # Strategy base classes
│   ├── risk/           # Risk management
│   └── tui/            # Terminal UI
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   ├── start-db.sh     # Database management
│   └── init-questdb.sql
├── docker-compose.yml
└── pyproject.toml
```

### Optional Dependencies

```bash
# Development tools
pip install -e ".[dev]"

# Database (QuestDB)
pip install -e ".[database]"

# Exchange connectivity
pip install -e ".[exchange]"

# Terminal UI
pip install -e ".[tui]"

# All dependencies
pip install -e ".[dev,database,exchange,tui]"
```

## Performance

| Component | Target | Notes |
|-----------|--------|-------|
| Message Bus | <100μs dispatch | Priority queue |
| DB Ingestion | 5M+ rows/sec | ILP protocol |
| DB Queries | <10ms | OHLCV aggregation |
| ASOF JOIN | <25ms | Backtest accuracy |

## License

MIT

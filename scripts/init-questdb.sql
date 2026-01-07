-- =============================================================================
-- LIBRA QuestDB Schema Initialization
--
-- Run via QuestDB Web Console (http://localhost:9000) or:
--   curl -G --data-urlencode "query=$(cat scripts/init-questdb.sql)" http://localhost:9000/exec
-- =============================================================================

-- Tick data (real-time quotes)
CREATE TABLE IF NOT EXISTS ticks (
    timestamp TIMESTAMP,
    symbol SYMBOL,
    exchange SYMBOL,
    bid DOUBLE,
    ask DOUBLE,
    last DOUBLE,
    bid_size DOUBLE,
    ask_size DOUBLE,
    volume_24h DOUBLE
) TIMESTAMP(timestamp) PARTITION BY DAY WAL
DEDUP UPSERT KEYS(timestamp, symbol);

-- OHLCV bars (candlesticks)
CREATE TABLE IF NOT EXISTS ohlcv (
    timestamp TIMESTAMP,
    symbol SYMBOL,
    timeframe SYMBOL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    trades INT
) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
DEDUP UPSERT KEYS(timestamp, symbol, timeframe);

-- Trade history (executed trades)
CREATE TABLE IF NOT EXISTS trades (
    timestamp TIMESTAMP,
    trade_id SYMBOL,
    order_id SYMBOL,
    symbol SYMBOL,
    exchange SYMBOL,
    side SYMBOL,
    amount DOUBLE,
    price DOUBLE,
    fee DOUBLE,
    fee_currency SYMBOL,
    strategy SYMBOL,
    realized_pnl DOUBLE
) TIMESTAMP(timestamp) PARTITION BY MONTH WAL;

-- Trading signals (for backtest replay)
CREATE TABLE IF NOT EXISTS signals (
    timestamp TIMESTAMP,
    symbol SYMBOL,
    signal_type SYMBOL,
    strength DOUBLE,
    price DOUBLE,
    strategy SYMBOL
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;

-- Verify tables exist
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public';

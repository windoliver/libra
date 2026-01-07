"""
Libra Clients - DataClient and ExecutionClient split architecture.

This package implements the separation of market data and order execution
operations, following NautilusTrader's proven pattern.

Architecture:
    DataClient              ExecutionClient
    ┌────────────────┐      ┌────────────────┐
    │ subscribe_*()  │      │ submit_order() │
    │ stream_*()     │      │ cancel_order() │
    │ request_*()    │      │ get_positions()│
    │ get_orderbook()│      │ get_balances() │
    └────────────────┘      └────────────────┘
           ↑                        ↑
           └────── TradingKernel ───┘

Protocol Classes:
    - DataClient: Market data subscriptions and historical requests
    - ExecutionClient: Order submission, cancellation, and account queries

Implementations:
    - CCXTDataClient/CCXTExecutionClient: 100+ exchanges via CCXT (live trading)
    - BacktestDataClient/BacktestExecutionClient: Historical backtesting with
      realistic fill/slippage models

Backtest Models:
    Fill Models:
    - ImmediateFillModel: Market orders fill at bid/ask, limits check price
    - QueuePositionFillModel: Simulates limit order queue position

    Slippage Models:
    - NoSlippage: No price impact (testing only)
    - FixedSlippage: Fixed basis points (e.g., 10 bps)
    - VolumeSlippage: Price impact based on order size / volume
    - StochasticSlippage: Random slippage with gaussian distribution

Design Benefits:
    - Independent scaling (data vs execution)
    - Cleaner testing (mock one without the other)
    - Same strategy code works in backtest and live
    - Clear responsibility boundaries
    - Unified data structures (Bar, Tick, Order, Position, etc.)

Usage Examples:

    # Backtest usage
    from libra.clients import (
        BacktestDataClient, BacktestExecutionClient,
        InMemoryDataSource, ImmediateFillModel, FixedSlippage,
    )

    data_source = InMemoryDataSource(bars=historical_bars)
    data_client = BacktestDataClient(data_source)
    exec_client = BacktestExecutionClient(
        initial_balances={"USDT": Decimal("10000")},
        fill_model=ImmediateFillModel(FixedSlippage(bps=10)),
    )

    # Live trading usage
    from libra.clients import CCXTDataClient, CCXTExecutionClient

    config = {"api_key": "...", "secret": "..."}
    data_client = CCXTDataClient("binance", config)
    exec_client = CCXTExecutionClient("binance", config)

    # With TradingKernel
    from libra.core import TradingKernel, KernelConfig

    kernel = TradingKernel(KernelConfig(environment="live"))
    kernel.set_clients(data_client, exec_client)
    kernel.add_strategy(my_strategy)

    async with kernel:
        await asyncio.sleep(3600)

See: https://github.com/windoliver/libra/issues/33
"""

# Protocols
from libra.clients.data_client import (
    BaseDataClient,
    DataClient,
    DataClientError,
    DataNotAvailableError,
    Instrument,
    SubscriptionError,
)
from libra.clients.execution_client import (
    BaseExecutionClient,
    ExecutionClient,
    ExecutionClientError,
    InsufficientFundsError,
    OrderError,
    OrderNotFoundError,
    OrderRejectedError,
    ReconciliationError,
)

# Backtest implementations
from libra.clients.backtest_data_client import (
    BacktestDataClient,
    CSVDataSource,
    DataSource,
    InMemoryDataSource,
)
from libra.clients.backtest_execution_client import (
    BacktestExecutionClient,
    BaseSlippageModel,
    FillModel,
    FixedSlippage,
    ImmediateFillModel,
    NoSlippage,
    QueuePositionFillModel,
    SlippageModel,
    StochasticSlippage,
    VolumeSlippage,
)

# CCXT implementations (live trading)
from libra.clients.ccxt_data_client import CCXTDataClient
from libra.clients.ccxt_execution_client import CCXTExecutionClient

# Re-export data structures from gateway protocol
from libra.gateways.protocol import (
    Balance,
    Order,
    OrderBook,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Tick,
    TimeInForce,
)

__all__ = [
    # Protocols
    "DataClient",
    "ExecutionClient",
    "BaseDataClient",
    "BaseExecutionClient",
    # CCXT implementations (live trading)
    "CCXTDataClient",
    "CCXTExecutionClient",
    # Backtest implementations
    "BacktestDataClient",
    "BacktestExecutionClient",
    # Data sources
    "DataSource",
    "CSVDataSource",
    "InMemoryDataSource",
    # Fill models
    "FillModel",
    "ImmediateFillModel",
    "QueuePositionFillModel",
    # Slippage models
    "SlippageModel",
    "BaseSlippageModel",
    "NoSlippage",
    "FixedSlippage",
    "VolumeSlippage",
    "StochasticSlippage",
    # Data structures
    "Instrument",
    "Order",
    "OrderResult",
    "Position",
    "Tick",
    "OrderBook",
    "Balance",
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "PositionSide",
    # Exceptions
    "DataClientError",
    "DataNotAvailableError",
    "SubscriptionError",
    "ExecutionClientError",
    "OrderError",
    "OrderNotFoundError",
    "InsufficientFundsError",
    "OrderRejectedError",
    "ReconciliationError",
]

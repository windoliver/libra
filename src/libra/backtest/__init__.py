"""
LIBRA Backtest: Event-driven backtesting engine.

This module provides high-performance backtesting capabilities:

- BacktestEngine: Main orchestrator for running backtests
- BacktestConfig: Complete backtest configuration
- BacktestResult: Immutable result container with metrics
- MetricsCollector: Real-time performance metrics calculation

Key features:
- Event-driven architecture (same as live trading)
- Look-ahead bias prevention
- Multiple fill models (immediate, queue position)
- Multiple slippage models (fixed, volume, stochastic)
- Comprehensive performance metrics

Usage:
    from libra.backtest import BacktestEngine, BacktestConfig
    from libra.strategies import MyStrategy

    config = BacktestConfig(
        start_time="2024-01-01",
        end_time="2024-12-31",
        initial_capital=Decimal("100000"),
    )

    engine = BacktestEngine(config)
    engine.add_strategy(MyStrategy())
    engine.add_data_source(bars)

    result = await engine.run()

    print(f"Total Return: {result.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                       BacktestEngine                         │
    │                                                              │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │                    TradingKernel                        ││
    │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ││
    │  │  │BacktestData  │  │ MessageBus   │  │BacktestExec  │  ││
    │  │  │   Client     ├──┤  (events)    ├──┤   Client     │  ││
    │  │  └──────────────┘  └──────────────┘  └──────────────┘  ││
    │  │          │                │                  │          ││
    │  │          ▼                ▼                  ▼          ││
    │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ││
    │  │  │  Strategy    │  │ RiskEngine   │  │    Cache     │  ││
    │  │  │  (signals)   │  │ (validation) │  │  (state)     │  ││
    │  │  └──────────────┘  └──────────────┘  └──────────────┘  ││
    │  └─────────────────────────────────────────────────────────┘│
    │                              │                               │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │                   MetricsCollector                      ││
    │  │  equity_curve, returns, Sharpe, drawdown, trade_log    ││
    │  └─────────────────────────────────────────────────────────┘│
    │                              │                               │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │                   BacktestResult                        ││
    │  │  immutable msgspec.Struct with all results              ││
    │  └─────────────────────────────────────────────────────────┘│
    └──────────────────────────────────────────────────────────────┘

References:
    - NautilusTrader BacktestEngine pattern
    - hftbacktest queue position model
    - Issue #7: Backtest Engine Implementation
"""

from libra.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
)
from libra.backtest.fill_model import (
    FillModel,
    FillModelType,
    FillResult,
    SlippageType,
    create_conservative_model,
    create_immediate_model,
    create_order_book_model,
    create_realistic_model,
)
from libra.backtest.factors import (
    BollingerBandFactor,
    Factor,
    FactorMeta,
    LogReturnsFactor,
    MeanReversionFactor,
    MomentumFactor,
    RankFactor,
    ReturnsFactor,
    RSIFactor,
    VolatilityFactor,
    VWAPDeviationFactor,
    get_factor,
    list_factors,
    register_factor,
)
from libra.backtest.gateway import BacktestGateway
from libra.backtest.metrics import (
    MetricsCollector,
    TradeRecord,
)
from libra.backtest.pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineResult,
    compute_factor,
    run_factors,
)
from libra.backtest.result import (
    BacktestResult,
    BacktestSummary,
)
from libra.backtest.vectorized import (
    VectorizedBacktest,
    VectorizedConfig,
    VectorizedResult,
    generate_moving_average_signals,
    generate_rsi_signals,
)


__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    # Fill Model (Issue #107)
    "FillModel",
    "FillModelType",
    "FillResult",
    "SlippageType",
    "create_immediate_model",
    "create_realistic_model",
    "create_conservative_model",
    "create_order_book_model",
    # Gateway (for unified strategies)
    "BacktestGateway",
    # Metrics
    "MetricsCollector",
    "TradeRecord",
    # Result
    "BacktestResult",
    "BacktestSummary",
    # Factors (Issue #93)
    "Factor",
    "FactorMeta",
    "MomentumFactor",
    "MeanReversionFactor",
    "VolatilityFactor",
    "RSIFactor",
    "BollingerBandFactor",
    "VWAPDeviationFactor",
    "ReturnsFactor",
    "LogReturnsFactor",
    "RankFactor",
    "get_factor",
    "list_factors",
    "register_factor",
    # Pipeline (Issue #93)
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "compute_factor",
    "run_factors",
    # Vectorized (Issue #104)
    "VectorizedBacktest",
    "VectorizedConfig",
    "VectorizedResult",
    "generate_moving_average_signals",
    "generate_rsi_signals",
]

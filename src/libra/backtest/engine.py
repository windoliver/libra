"""
BacktestEngine: Main orchestrator for running backtests.

Coordinates:
- BacktestDataClient for data replay
- BacktestExecutionClient for fill simulation
- Strategy for signal generation
- MetricsCollector for performance tracking
- TradingKernel for component orchestration

Follows NautilusTrader's BacktestEngine pattern.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from libra.backtest.metrics import MetricsCollector
from libra.backtest.result import BacktestResult
from libra.clients.backtest_data_client import BacktestDataClient, InMemoryDataSource
from libra.clients.backtest_execution_client import (
    BacktestExecutionClient,
    SlippageModel,
)
from libra.core.clock import Clock, ClockType
from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus, MessageBusConfig
from libra.strategies.base import BaseStrategy
from libra.strategies.protocol import Bar


if TYPE_CHECKING:
    from libra.risk.engine import RiskEngine


logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.

    All monetary values should use the quote currency of the traded symbol.
    """

    # Capital
    initial_capital: Decimal = Decimal("100000")

    # Time range (optional - if None, uses full data range)
    start_time: datetime | int | str | None = None  # datetime, timestamp_ns, or ISO string
    end_time: datetime | int | str | None = None

    # Execution
    slippage_model: SlippageModel = SlippageModel.FIXED
    slippage_bps: Decimal = Decimal("5")  # Basis points for fixed slippage

    # Fees
    maker_fee_rate: Decimal = Decimal("0.001")  # 0.1%
    taker_fee_rate: Decimal = Decimal("0.001")  # 0.1%

    # Risk (optional)
    enable_risk_engine: bool = False

    # Message bus config
    bus_config: MessageBusConfig = field(default_factory=MessageBusConfig)

    # Output
    verbose: bool = True
    save_results: bool = False
    results_path: str | None = None

    # Instance ID for tracking
    instance_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "initial_capital": str(self.initial_capital),
            "start_time": str(self.start_time) if self.start_time else None,
            "end_time": str(self.end_time) if self.end_time else None,
            "slippage_model": self.slippage_model.value,
            "slippage_bps": str(self.slippage_bps),
            "maker_fee_rate": str(self.maker_fee_rate),
            "taker_fee_rate": str(self.taker_fee_rate),
            "enable_risk_engine": self.enable_risk_engine,
            "instance_id": self.instance_id,
        }


class BacktestEngine:
    """
    Main orchestrator for running backtests.

    Coordinates all components and executes the backtest simulation
    in a single-threaded event loop (no async required for basic use).

    Usage:
        # Create engine with config
        config = BacktestConfig(initial_capital=Decimal("100000"))
        engine = BacktestEngine(config)

        # Add strategy
        engine.add_strategy(MyStrategy(strategy_config))

        # Add data (bars)
        engine.add_bars(symbol="BTC/USDT", bars=bars_list)

        # Run backtest
        result = await engine.run()

        # Analyze results
        result.print_summary()

    For advanced use with async context manager:
        async with BacktestEngine(config) as engine:
            engine.add_strategy(strategy)
            engine.add_bars(symbol, bars)
            result = await engine.run()
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()

        # Core components
        self._bus = MessageBus(self.config.bus_config)
        self._clock = Clock(clock_type=ClockType.BACKTEST)

        # Clients
        self._data_client: BacktestDataClient | None = None
        self._execution_client: BacktestExecutionClient | None = None

        # Strategy
        self._strategy: BaseStrategy | None = None

        # Risk engine (optional)
        self._risk_engine: RiskEngine | None = None

        # Metrics
        self._metrics = MetricsCollector(initial_capital=self.config.initial_capital)

        # State
        self._bars: dict[str, list[Bar]] = {}  # symbol -> bars
        self._is_running = False
        self._run_id: str | None = None

        # Position tracking for P&L
        self._positions: dict[str, dict[str, Any]] = {}  # symbol -> position info
        self._cash = self.config.initial_capital

        logger.info(
            "BacktestEngine initialized: instance_id=%s capital=%s",
            self.config.instance_id,
            self.config.initial_capital,
        )

    # =========================================================================
    # Configuration
    # =========================================================================

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Add a strategy to the backtest.

        Currently supports only one strategy per backtest.
        For multi-strategy testing, run multiple backtests.

        Args:
            strategy: Strategy instance to test

        Raises:
            ValueError: If strategy already set
        """
        if self._strategy is not None:
            raise ValueError(
                "Strategy already set. Create new engine for different strategy."
            )
        self._strategy = strategy
        logger.info("Strategy added: %s", strategy.name)

    def add_bars(self, symbol: str, bars: list[Bar]) -> None:
        """
        Add historical bar data for a symbol.

        Bars must be sorted by timestamp in ascending order.
        The engine will validate this on run().

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            bars: List of Bar objects, sorted by timestamp
        """
        if not bars:
            raise ValueError("bars list cannot be empty")

        # Validate sorting
        for i in range(1, len(bars)):
            if bars[i].timestamp_ns <= bars[i - 1].timestamp_ns:
                raise ValueError(
                    f"Bars must be sorted by timestamp. "
                    f"Bar {i} has timestamp <= bar {i-1}"
                )

        self._bars[symbol] = bars
        logger.info(
            "Added %d bars for %s: %s to %s",
            len(bars),
            symbol,
            datetime.fromtimestamp(bars[0].timestamp_ns / 1e9),
            datetime.fromtimestamp(bars[-1].timestamp_ns / 1e9),
        )

    def set_risk_engine(self, risk_engine: RiskEngine) -> None:
        """
        Set optional risk engine for pre-trade validation.

        Args:
            risk_engine: RiskEngine instance
        """
        self._risk_engine = risk_engine
        logger.info("RiskEngine set")

    # =========================================================================
    # Execution
    # =========================================================================

    async def run(self) -> BacktestResult:
        """
        Execute the backtest simulation.

        Returns:
            BacktestResult with complete metrics and analysis

        Raises:
            ValueError: If no strategy or data configured
            RuntimeError: If already running
        """
        if self._is_running:
            raise RuntimeError("Backtest already running")

        if self._strategy is None:
            raise ValueError("No strategy configured. Call add_strategy() first.")

        if not self._bars:
            raise ValueError("No bar data configured. Call add_bars() first.")

        self._is_running = True
        self._run_id = uuid4().hex[:12]

        logger.info(
            "Starting backtest: run_id=%s strategy=%s symbols=%s",
            self._run_id,
            self._strategy.name,
            list(self._bars.keys()),
        )

        start_time = time.perf_counter()

        try:
            # Initialize components
            await self._initialize()

            # Run simulation
            await self._run_simulation()

            # Calculate results
            result = self._build_result()

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Backtest complete: run_id=%s elapsed=%.2fs bars=%d",
                self._run_id,
                elapsed,
                self._metrics.bars_processed,
            )

            if self.config.verbose:
                result.print_summary()

            if self.config.save_results and self.config.results_path:
                result.save(self.config.results_path)

            return result

        finally:
            await self._cleanup()
            self._is_running = False

    async def _initialize(self) -> None:
        """Initialize all components for the backtest."""
        # Create data client
        symbol = next(iter(self._bars.keys()))
        bars = self._bars[symbol]

        # Create in-memory data source with bars
        data_source = InMemoryDataSource()
        data_source.add_bars(symbol, self._strategy.timeframe, bars)

        self._data_client = BacktestDataClient(
            data_source=data_source,
            clock=self._clock,
        )

        # Create execution client
        # Parse quote currency from symbol (e.g., "BTC/USDT" -> "USDT")
        quote_currency = symbol.split("/")[1] if "/" in symbol else "USDT"
        initial_balance = {quote_currency: self.config.initial_capital}

        self._execution_client = BacktestExecutionClient(
            clock=self._clock,
            initial_balance=initial_balance,
            slippage_model=self.config.slippage_model,
            slippage_bps=self.config.slippage_bps,
            maker_fee=self.config.maker_fee_rate,
            taker_fee=self.config.taker_fee_rate,
        )

        # Connect clients
        await self._data_client.connect()
        await self._execution_client.connect()

        # Configure data range from the bars
        start_time = datetime.fromtimestamp(bars[0].timestamp_ns / 1_000_000_000)
        end_time = datetime.fromtimestamp(bars[-1].timestamp_ns / 1_000_000_000)
        self._data_client.configure_range(start=start_time, end=end_time)

        # Subscribe to bar data
        await self._data_client.subscribe_bars(
            symbol=symbol,
            timeframe=self._strategy.timeframe,
        )

        # Subscribe to events
        self._bus.subscribe(EventType.BAR, self._on_bar)
        self._bus.subscribe(EventType.ORDER_FILLED, self._on_fill)

        # Start clock and set initial time to first bar
        self._clock.start()
        self._clock.set_time(start_time)

        # Initialize strategy
        self._strategy.on_start()

        # Initialize position tracking
        self._cash = self.config.initial_capital
        self._positions.clear()

        # Reset metrics
        self._metrics.reset()

        logger.debug("Backtest initialized")

    async def _run_simulation(self) -> None:
        """Run the main simulation loop."""
        assert self._data_client is not None
        assert self._execution_client is not None
        assert self._strategy is not None

        # Get bars directly (bypass stream_bars which has clock synchronization)
        bars_list = self._data_client._bar_events

        # Track position changes for trade recording
        prev_position: dict[str, Decimal] = {}

        for bar in bars_list:
            # Update clock to bar time
            bar_datetime = datetime.fromtimestamp(bar.timestamp_ns / 1_000_000_000)
            self._clock.set_time(bar_datetime)

            # Process pending orders at this bar
            await self._execution_client.process_bar(bar)

            # Let strategy process bar
            signal = self._strategy._process_bar(bar)

            # Handle signal
            if signal is not None:
                await self._handle_signal(signal, bar)

            # Track position changes for trade recording
            current_position = self._execution_client._positions.get(bar.symbol)
            current_qty = current_position.amount if current_position else Decimal("0")
            prev_qty = prev_position.get(bar.symbol, Decimal("0"))

            if current_qty != prev_qty:
                # Position changed - record trade
                if current_qty > prev_qty:
                    # Opened or added to position
                    if prev_qty == 0:
                        # New position opened
                        self._metrics.open_trade(
                            trade_id=f"trade_{uuid4().hex[:8]}",
                            symbol=bar.symbol,
                            side="long",
                            entry_time_ns=bar.timestamp_ns,
                            entry_price=bar.close,
                            quantity=current_qty,
                            fees=Decimal("0"),  # Fees handled in equity calculation
                            entry_reason="signal",
                        )
                else:
                    # Closed or reduced position
                    if current_qty == 0:
                        # Position fully closed
                        self._metrics.close_trade(
                            symbol=bar.symbol,
                            exit_time_ns=bar.timestamp_ns,
                            exit_price=bar.close,
                            fees=Decimal("0"),
                            exit_reason="signal",
                        )

                prev_position[bar.symbol] = current_qty

            # Record equity snapshot
            equity = self._calculate_equity(bar)
            quote_balance = self._execution_client._balances.get(
                bar.symbol.split("/")[1] if "/" in bar.symbol else "USDT"
            )
            cash = quote_balance.total if quote_balance else Decimal("0")
            position_value = equity - cash

            self._metrics.record_equity(
                timestamp_ns=bar.timestamp_ns,
                equity=equity,
                cash=cash,
                position_value=position_value,
            )

            # Dispatch any pending events (allow async handlers to run)
            while self._bus.total_pending > 0:
                await self._bus._dispatch_batch()

    async def _handle_signal(self, signal: Any, bar: Bar) -> None:
        """
        Handle a trading signal from the strategy.

        Converts signals to orders and submits to execution client.
        """
        assert self._execution_client is not None

        from libra.gateways.protocol import Order, OrderSide, OrderType
        from libra.strategies.protocol import SignalType

        # Determine order side
        if signal.signal_type in (SignalType.LONG, SignalType.CLOSE_SHORT):
            side = OrderSide.BUY
        elif signal.signal_type in (SignalType.SHORT, SignalType.CLOSE_LONG):
            side = OrderSide.SELL
        else:
            return  # Unknown signal type

        # Get current position from execution client (not engine's stale state)
        exec_position = self._execution_client._positions.get(signal.symbol)
        current_qty = exec_position.amount if exec_position else Decimal("0")

        # Get available cash from execution client
        quote_currency = signal.symbol.split("/")[1] if "/" in signal.symbol else "USDT"
        quote_balance = self._execution_client._balances.get(quote_currency)
        available_cash = quote_balance.available if quote_balance else Decimal("0")

        amount: Decimal

        if signal.signal_type == SignalType.LONG:
            # Open long position - use available capital
            order_value = available_cash * Decimal("0.95")  # Leave 5% buffer
            amount = order_value / bar.close
        elif signal.signal_type == SignalType.SHORT:
            # Open short position
            order_value = available_cash * Decimal("0.95")
            amount = order_value / bar.close
        elif signal.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
            # Close existing position
            amount = abs(current_qty)
            if amount == 0:
                return  # No position to close
        else:
            return  # Unknown signal type

        # Create order
        order = Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
            client_order_id=f"bt_{uuid4().hex[:8]}",
        )

        # Risk validation (if enabled)
        if self._risk_engine is not None:
            result = self._risk_engine.validate_order(order, bar.close, None)
            if not result.passed:
                logger.warning("Order rejected by risk engine: %s", result.reason)
                return

        # Submit order
        await self._execution_client.submit_order(order)

    async def _on_bar(self, _event: Event) -> None:
        """Handle bar event from message bus."""
        # Bar processing is done in _run_simulation directly
        pass

    async def _on_fill(self, event: Event) -> None:
        """Handle fill event - update positions and metrics."""
        payload = event.payload
        symbol = payload.get("symbol", "")
        side = payload.get("side", "")
        filled_qty = Decimal(str(payload.get("filled_amount", 0)))
        fill_price = Decimal(str(payload.get("price", 0)))
        fee = Decimal(str(payload.get("fee", 0)))

        if filled_qty == 0:
            return

        # Update position
        position = self._positions.setdefault(
            symbol,
            {"quantity": Decimal("0"), "avg_price": Decimal("0"), "entry_time_ns": 0},
        )

        old_qty = position["quantity"]

        if side == "buy":
            # Adding to long position or closing short
            if old_qty < 0:
                # Closing short - calculate P&L
                close_qty = min(filled_qty, abs(old_qty))
                pnl = (position["avg_price"] - fill_price) * close_qty - fee
                self._cash += close_qty * fill_price + pnl

                # Record trade
                self._metrics.close_trade(
                    symbol=symbol,
                    exit_time_ns=self._clock.timestamp_ns(),
                    exit_price=fill_price,
                    fees=fee,
                    exit_reason="signal",
                )

                filled_qty -= close_qty
                position["quantity"] += close_qty

            if filled_qty > 0:
                # Opening/adding to long
                cost = filled_qty * fill_price + fee
                self._cash -= cost
                new_qty = position["quantity"] + filled_qty

                # Update average price
                if position["quantity"] > 0:
                    total_cost = position["avg_price"] * position["quantity"] + cost
                    position["avg_price"] = total_cost / new_qty
                else:
                    position["avg_price"] = fill_price
                    position["entry_time_ns"] = self._clock.timestamp_ns()

                    # Open trade record
                    self._metrics.open_trade(
                        trade_id=f"trade_{uuid4().hex[:8]}",
                        symbol=symbol,
                        side="long",
                        entry_time_ns=self._clock.timestamp_ns(),
                        entry_price=fill_price,
                        quantity=filled_qty,
                        fees=fee,
                        entry_reason="signal",
                    )

                position["quantity"] = new_qty

        else:  # sell
            # Closing long or opening short
            if old_qty > 0:
                # Closing long - calculate P&L
                close_qty = min(filled_qty, old_qty)
                pnl = (fill_price - position["avg_price"]) * close_qty - fee
                self._cash += close_qty * fill_price - fee

                # Record trade
                self._metrics.close_trade(
                    symbol=symbol,
                    exit_time_ns=self._clock.timestamp_ns(),
                    exit_price=fill_price,
                    fees=fee,
                    exit_reason="signal",
                )

                filled_qty -= close_qty
                position["quantity"] -= close_qty

            if filled_qty > 0:
                # Opening short
                position["quantity"] -= filled_qty
                position["avg_price"] = fill_price
                position["entry_time_ns"] = self._clock.timestamp_ns()
                self._cash += filled_qty * fill_price - fee

                # Open trade record
                self._metrics.open_trade(
                    trade_id=f"trade_{uuid4().hex[:8]}",
                    symbol=symbol,
                    side="short",
                    entry_time_ns=self._clock.timestamp_ns(),
                    entry_price=fill_price,
                    quantity=filled_qty,
                    fees=fee,
                    entry_reason="signal",
                )

    def _calculate_equity(self, bar: Bar) -> Decimal:
        """Calculate total equity at current bar using execution client state."""
        assert self._execution_client is not None

        # Get balances from execution client
        balances = self._execution_client._balances
        positions = self._execution_client._positions

        # Start with quote currency balance
        quote_currency = bar.symbol.split("/")[1] if "/" in bar.symbol else "USDT"
        quote_balance = balances.get(quote_currency)
        equity = quote_balance.total if quote_balance else Decimal("0")

        # Add position value
        position = positions.get(bar.symbol)
        if position and position.amount > 0:
            # Mark to market
            equity += position.amount * bar.close

        return equity

    def _build_result(self) -> BacktestResult:
        """Build final backtest result."""
        assert self._strategy is not None

        symbol = next(iter(self._bars.keys()))

        # Calculate summary
        summary = self._metrics.calculate_summary(
            strategy_name=self._strategy.name,
            symbol=symbol,
            timeframe=self._strategy.timeframe,
        )

        # Build equity curve
        equity_curve = self._metrics.get_equity_curve()

        return BacktestResult(
            summary=summary,
            equity_curve=equity_curve,
            trades=list(self._metrics.trades),
            daily_returns=list(self._metrics.daily_returns),
            config=self.config.to_dict(),
            run_timestamp_ns=time.time_ns(),
        )

    async def _cleanup(self) -> None:
        """Clean up after backtest."""
        # Stop strategy
        if self._strategy is not None:
            self._strategy.on_stop()

        # Disconnect clients
        if self._data_client is not None:
            await self._data_client.disconnect()

        if self._execution_client is not None:
            await self._execution_client.disconnect()

        # Stop clock
        self._clock.stop()

        logger.debug("Backtest cleanup complete")

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> BacktestEngine:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        _exc_type: Any,  # noqa: PYI036
        _exc_val: Any,  # noqa: PYI036
        _exc_tb: Any,  # noqa: PYI036
    ) -> None:
        """Async context manager exit."""
        if self._is_running:
            await self._cleanup()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if backtest is running."""
        return self._is_running

    @property
    def strategy(self) -> BaseStrategy | None:
        """Get the configured strategy."""
        return self._strategy

    @property
    def symbols(self) -> list[str]:
        """Get list of configured symbols."""
        return list(self._bars.keys())

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BacktestEngine("
            f"instance_id={self.config.instance_id!r}, "
            f"capital={self.config.initial_capital}, "
            f"strategy={self._strategy.name if self._strategy else None!r}, "
            f"symbols={self.symbols})"
        )

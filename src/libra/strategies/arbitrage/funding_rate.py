"""
Funding Rate Arbitrage Strategy.

A delta-neutral strategy that profits from perpetual futures funding payments
without exposure to price direction.

Strategy Logic:
1. Monitor funding rates across exchanges in real-time
2. When funding rate is positive: short perpetual + long spot (collect funding)
3. When funding rate is negative: long perpetual + short spot (collect funding)
4. Maintain delta-neutral positions to eliminate directional risk
5. Exit when funding rate normalizes or reverses

Key Features:
- Multi-exchange funding rate monitoring
- Delta-neutral position management
- Automatic rebalancing for position drift
- Risk limits and emergency exits
- Backtesting support with historical funding data

See: https://github.com/windoliver/libra/issues/13
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from libra.strategies.base import BaseStrategy
from libra.strategies.protocol import (
    Bar,
    Signal,
    SignalType,
    StrategyConfig,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Funding periods per day (most exchanges use 8-hour intervals)
FUNDING_PERIODS_PER_DAY = 3
FUNDING_PERIODS_PER_YEAR = FUNDING_PERIODS_PER_DAY * 365  # 1095

# Default thresholds
DEFAULT_MIN_FUNDING_RATE = Decimal("0.0001")  # 0.01% per period
DEFAULT_MIN_APR = Decimal("0.10")  # 10% annualized
DEFAULT_MAX_LEVERAGE = 3
DEFAULT_MAX_BASIS_DEVIATION = Decimal("0.02")  # 2%


# =============================================================================
# Enums
# =============================================================================


class ArbitrageDirection(str, Enum):
    """Direction of the funding arbitrage position."""

    LONG_SPOT_SHORT_PERP = "long_spot_short_perp"  # Collect positive funding
    SHORT_SPOT_LONG_PERP = "short_spot_long_perp"  # Collect negative funding
    FLAT = "flat"  # No position


class FundingArbitrageState(str, Enum):
    """State of the arbitrage strategy."""

    IDLE = "idle"  # Waiting for opportunity
    ENTERING = "entering"  # Opening positions
    ACTIVE = "active"  # Position open, collecting funding
    EXITING = "exiting"  # Closing positions
    EMERGENCY = "emergency"  # Emergency exit in progress


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class FundingRateData:
    """
    Funding rate data from an exchange.

    Attributes:
        symbol: Trading pair (e.g., "BTC/USDT:USDT")
        exchange: Exchange name
        funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)
        next_funding_time_ns: Unix nanoseconds of next funding settlement
        mark_price: Current mark price
        index_price: Current index/spot price
        predicted_rate: Predicted next funding rate (if available)
        timestamp_ns: Data timestamp
    """

    symbol: str
    exchange: str
    funding_rate: Decimal
    next_funding_time_ns: int
    mark_price: Decimal
    index_price: Decimal
    predicted_rate: Decimal | None = None
    timestamp_ns: int = field(default_factory=time.time_ns)

    @property
    def annualized_rate(self) -> Decimal:
        """Annualized funding rate (APR)."""
        return self.funding_rate * FUNDING_PERIODS_PER_YEAR

    @property
    def basis(self) -> Decimal:
        """Basis: (mark - index) / index."""
        if self.index_price == 0:
            return Decimal("0")
        return (self.mark_price - self.index_price) / self.index_price

    @property
    def basis_bps(self) -> Decimal:
        """Basis in basis points."""
        return self.basis * 10000

    @property
    def time_to_funding_secs(self) -> float:
        """Seconds until next funding settlement."""
        now_ns = time.time_ns()
        return max(0, (self.next_funding_time_ns - now_ns) / 1_000_000_000)

    @property
    def is_positive(self) -> bool:
        """Check if funding rate is positive (longs pay shorts)."""
        return self.funding_rate > 0

    @property
    def is_negative(self) -> bool:
        """Check if funding rate is negative (shorts pay longs)."""
        return self.funding_rate < 0


@dataclass
class FundingArbitragePosition:
    """
    Represents an active funding arbitrage position.

    Tracks both legs (spot and perpetual) of the delta-neutral position.
    """

    symbol: str
    direction: ArbitrageDirection

    # Position sizes (always positive)
    spot_size: Decimal
    perp_size: Decimal

    # Entry information
    entry_funding_rate: Decimal
    entry_spot_price: Decimal
    entry_perp_price: Decimal
    entry_time_ns: int

    # Tracking
    cumulative_funding: Decimal = Decimal("0")
    funding_payments_count: int = 0
    last_funding_time_ns: int | None = None

    # P&L tracking
    spot_pnl: Decimal = Decimal("0")
    perp_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")

    @property
    def net_delta(self) -> Decimal:
        """Net delta exposure (should be ~0 for delta-neutral)."""
        if self.direction == ArbitrageDirection.LONG_SPOT_SHORT_PERP:
            return self.spot_size - self.perp_size
        elif self.direction == ArbitrageDirection.SHORT_SPOT_LONG_PERP:
            return self.perp_size - self.spot_size
        return Decimal("0")

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L including funding and trading P&L."""
        return self.cumulative_funding + self.spot_pnl + self.perp_pnl - self.fees_paid

    @property
    def holding_time_hours(self) -> float:
        """Hours since position was opened."""
        now_ns = time.time_ns()
        return (now_ns - self.entry_time_ns) / (1_000_000_000 * 3600)

    @property
    def avg_funding_per_period(self) -> Decimal:
        """Average funding collected per period."""
        if self.funding_payments_count == 0:
            return Decimal("0")
        return self.cumulative_funding / self.funding_payments_count

    def add_funding_payment(self, amount: Decimal) -> None:
        """Record a funding payment."""
        self.cumulative_funding += amount
        self.funding_payments_count += 1
        self.last_funding_time_ns = time.time_ns()

    def update_pnl(
        self,
        current_spot_price: Decimal,
        current_perp_price: Decimal,
    ) -> None:
        """Update P&L based on current prices."""
        if self.direction == ArbitrageDirection.LONG_SPOT_SHORT_PERP:
            # Long spot: profit when price goes up
            self.spot_pnl = (current_spot_price - self.entry_spot_price) * self.spot_size
            # Short perp: profit when price goes down
            self.perp_pnl = (self.entry_perp_price - current_perp_price) * self.perp_size
        elif self.direction == ArbitrageDirection.SHORT_SPOT_LONG_PERP:
            # Short spot: profit when price goes down
            self.spot_pnl = (self.entry_spot_price - current_spot_price) * self.spot_size
            # Long perp: profit when price goes up
            self.perp_pnl = (current_perp_price - self.entry_perp_price) * self.perp_size


@dataclass
class FundingArbitrageConfig(StrategyConfig):
    """
    Configuration for funding rate arbitrage strategy.

    Extends base StrategyConfig with arbitrage-specific parameters.
    """

    # Entry thresholds
    min_funding_rate: Decimal = DEFAULT_MIN_FUNDING_RATE  # 0.01% minimum
    min_annualized_return: Decimal = DEFAULT_MIN_APR  # 10% APR minimum

    # Position sizing
    max_position_size_usd: Decimal = Decimal("100000")  # Max $100k per position
    position_size_pct: Decimal = Decimal("0.10")  # 10% of equity per position
    max_leverage: int = DEFAULT_MAX_LEVERAGE  # Max 3x leverage

    # Risk limits
    max_basis_deviation: Decimal = DEFAULT_MAX_BASIS_DEVIATION  # 2% max basis
    max_delta_drift: Decimal = Decimal("0.05")  # 5% delta drift before rebalance
    stop_loss_pct: Decimal = Decimal("0.01")  # 1% stop loss
    max_positions: int = 5  # Max concurrent positions

    # Exit criteria
    min_hold_periods: int = 3  # Hold at least 3 funding periods
    exit_on_rate_reversal: bool = True  # Exit when rate reverses sign
    rate_reversal_threshold: Decimal = Decimal("-0.00005")  # -0.005% triggers exit

    # Exchange configuration
    spot_exchange: str = "binance"
    perp_exchange: str = "binance"

    # Timing
    check_interval_seconds: int = 60  # Check rates every minute
    rebalance_interval_seconds: int = 300  # Check rebalance every 5 min

    # Fees (conservative estimates)
    taker_fee_pct: Decimal = Decimal("0.001")  # 0.1% taker fee
    maker_fee_pct: Decimal = Decimal("0.0002")  # 0.02% maker fee

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_leverage < 1:
            raise ValueError("max_leverage must be >= 1")
        if self.min_funding_rate < 0:
            raise ValueError("min_funding_rate must be >= 0")
        if self.max_position_size_usd <= 0:
            raise ValueError("max_position_size_usd must be > 0")


# =============================================================================
# Funding Rate Monitor
# =============================================================================


@runtime_checkable
class FundingRateProvider(Protocol):
    """Protocol for funding rate data providers."""

    async def fetch_funding_rate(self, symbol: str) -> FundingRateData | None:
        """Fetch current funding rate for symbol."""
        ...

    async def fetch_funding_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 100,
    ) -> list[FundingRateData]:
        """Fetch historical funding rates."""
        ...


class FundingRateMonitor:
    """
    Monitors funding rates across exchanges and symbols.

    Features:
    - Real-time funding rate tracking
    - Multi-exchange support
    - Historical funding data caching
    - Opportunity detection
    """

    def __init__(
        self,
        providers: dict[str, FundingRateProvider] | None = None,
        symbols: list[str] | None = None,
        poll_interval_seconds: int = 60,
    ) -> None:
        """
        Initialize funding rate monitor.

        Args:
            providers: Dict of exchange name -> provider
            symbols: List of symbols to monitor
            poll_interval_seconds: Polling interval
        """
        self.providers = providers or {}
        self.symbols = symbols or []
        self.poll_interval = poll_interval_seconds

        # Current rates: {symbol: {exchange: FundingRateData}}
        self._rates: dict[str, dict[str, FundingRateData]] = {}

        # Historical rates for analysis
        self._history: dict[str, list[FundingRateData]] = {}

        # Running state
        self._running = False
        self._poll_task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    def get_rate(self, symbol: str, exchange: str | None = None) -> FundingRateData | None:
        """
        Get current funding rate for symbol.

        Args:
            symbol: Trading symbol
            exchange: Specific exchange (optional)

        Returns:
            FundingRateData or None if not available
        """
        if symbol not in self._rates:
            return None

        if exchange:
            return self._rates[symbol].get(exchange)

        # Return highest absolute rate if no exchange specified
        rates = list(self._rates[symbol].values())
        if not rates:
            return None
        return max(rates, key=lambda r: abs(r.funding_rate))

    def get_best_opportunity(
        self,
        min_rate: Decimal = DEFAULT_MIN_FUNDING_RATE,
    ) -> FundingRateData | None:
        """
        Find the best funding rate opportunity across all monitored symbols.

        Args:
            min_rate: Minimum absolute funding rate

        Returns:
            Best FundingRateData or None
        """
        best: FundingRateData | None = None
        best_abs_rate = Decimal("0")

        for symbol_rates in self._rates.values():
            for rate_data in symbol_rates.values():
                abs_rate = abs(rate_data.funding_rate)
                if abs_rate >= min_rate and abs_rate > best_abs_rate:
                    best = rate_data
                    best_abs_rate = abs_rate

        return best

    def get_all_opportunities(
        self,
        min_rate: Decimal = DEFAULT_MIN_FUNDING_RATE,
        max_basis: Decimal = DEFAULT_MAX_BASIS_DEVIATION,
    ) -> list[FundingRateData]:
        """
        Get all symbols with attractive funding rates.

        Args:
            min_rate: Minimum absolute funding rate
            max_basis: Maximum basis deviation

        Returns:
            List of FundingRateData sorted by rate (highest first)
        """
        opportunities: list[FundingRateData] = []

        for symbol_rates in self._rates.values():
            for rate_data in symbol_rates.values():
                abs_rate = abs(rate_data.funding_rate)
                abs_basis = abs(rate_data.basis)

                if abs_rate >= min_rate and abs_basis <= max_basis:
                    opportunities.append(rate_data)

        # Sort by absolute rate descending
        opportunities.sort(key=lambda r: abs(r.funding_rate), reverse=True)
        return opportunities

    async def poll_rates(self) -> dict[str, dict[str, FundingRateData]]:
        """
        Poll all providers for current funding rates.

        Returns:
            Updated rates dict
        """
        for symbol in self.symbols:
            if symbol not in self._rates:
                self._rates[symbol] = {}

            for exchange, provider in self.providers.items():
                try:
                    rate_data = await provider.fetch_funding_rate(symbol)
                    if rate_data:
                        self._rates[symbol][exchange] = rate_data

                        # Add to history
                        if symbol not in self._history:
                            self._history[symbol] = []
                        self._history[symbol].append(rate_data)

                        # Limit history size
                        if len(self._history[symbol]) > 1000:
                            self._history[symbol] = self._history[symbol][-500:]

                except Exception as e:
                    logger.warning(
                        "Failed to fetch funding rate for %s on %s: %s",
                        symbol, exchange, e
                    )

        return self._rates

    async def start(self) -> None:
        """Start the funding rate monitor."""
        if self._running:
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Funding rate monitor started")

    async def stop(self) -> None:
        """Stop the funding rate monitor."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("Funding rate monitor stopped")

    async def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                await self.poll_rates()
            except Exception as e:
                logger.error("Error in funding rate poll loop: %s", e)

            await asyncio.sleep(self.poll_interval)


# =============================================================================
# CCXT Funding Rate Provider
# =============================================================================


class CCXTFundingRateProvider:
    """
    Funding rate provider using CCXT.

    Fetches funding rates from supported exchanges via CCXT.
    """

    def __init__(
        self,
        exchange_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize CCXT funding rate provider.

        Args:
            exchange_id: CCXT exchange ID (e.g., "binance", "bybit")
            config: Exchange configuration
        """
        self.exchange_id = exchange_id
        self.config = config or {}
        self._exchange: Any = None

    async def _get_exchange(self) -> Any:
        """Get or create CCXT exchange instance."""
        if self._exchange is None:
            import ccxt.async_support as ccxt

            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class({
                **self.config,
                "enableRateLimit": True,
            })
        return self._exchange

    async def fetch_funding_rate(self, symbol: str) -> FundingRateData | None:
        """Fetch current funding rate for symbol."""
        try:
            exchange = await self._get_exchange()

            # Fetch funding rate
            funding = await exchange.fetch_funding_rate(symbol)

            # Fetch ticker for prices
            ticker = await exchange.fetch_ticker(symbol)

            # Get mark and index prices
            mark_price = Decimal(str(funding.get("markPrice", ticker["last"])))
            index_price = Decimal(str(funding.get("indexPrice", ticker["last"])))

            # Calculate next funding time
            next_funding_ts = funding.get("fundingTimestamp")
            if next_funding_ts:
                next_funding_ns = int(next_funding_ts * 1_000_000)
            else:
                # Estimate next 8-hour mark
                now = time.time()
                hours_since_midnight = (now % 86400) / 3600
                next_8h = ((hours_since_midnight // 8) + 1) * 8
                next_funding_ns = int((now - (now % 86400) + next_8h * 3600) * 1_000_000_000)

            return FundingRateData(
                symbol=symbol,
                exchange=self.exchange_id,
                funding_rate=Decimal(str(funding["fundingRate"])),
                next_funding_time_ns=next_funding_ns,
                mark_price=mark_price,
                index_price=index_price,
                predicted_rate=Decimal(str(funding["fundingRate"])) if "nextFundingRate" not in funding else Decimal(str(funding["nextFundingRate"])),
                timestamp_ns=time.time_ns(),
            )

        except Exception as e:
            logger.error("Failed to fetch funding rate for %s: %s", symbol, e)
            return None

    async def fetch_funding_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 100,
    ) -> list[FundingRateData]:
        """Fetch historical funding rates."""
        try:
            exchange = await self._get_exchange()

            history = await exchange.fetch_funding_rate_history(
                symbol,
                since=since,
                limit=limit,
            )

            result: list[FundingRateData] = []
            for item in history:
                result.append(FundingRateData(
                    symbol=symbol,
                    exchange=self.exchange_id,
                    funding_rate=Decimal(str(item["fundingRate"])),
                    next_funding_time_ns=int(item["timestamp"] * 1_000_000),
                    mark_price=Decimal(str(item.get("markPrice", 0))),
                    index_price=Decimal(str(item.get("indexPrice", 0))),
                    timestamp_ns=int(item["timestamp"] * 1_000_000),
                ))

            return result

        except Exception as e:
            logger.error("Failed to fetch funding history for %s: %s", symbol, e)
            return []

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None


# =============================================================================
# Funding Rate Arbitrage Strategy
# =============================================================================


class FundingRateArbitrageStrategy(BaseStrategy):
    """
    Delta-neutral funding rate arbitrage strategy.

    Profits from perpetual futures funding payments by maintaining
    offsetting positions in spot and perpetual markets.

    Strategy Flow:
    1. Monitor funding rates across exchanges
    2. When rate exceeds threshold:
       - Positive rate: Long spot + Short perp (collect from longs)
       - Negative rate: Short spot + Long perp (collect from shorts)
    3. Maintain delta-neutral hedge
    4. Collect funding payments every 8 hours
    5. Exit when:
       - Funding rate reverses
       - Basis deviation exceeds threshold
       - Stop loss triggered
       - Minimum holding period met and better opportunity exists

    Example:
        config = FundingArbitrageConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            min_funding_rate=Decimal("0.0001"),
            max_position_size_usd=Decimal("50000"),
        )
        strategy = FundingRateArbitrageStrategy(config)
        await strategy.start()
    """

    def __init__(
        self,
        config: FundingArbitrageConfig,
        funding_monitor: FundingRateMonitor | None = None,
    ) -> None:
        """
        Initialize the funding rate arbitrage strategy.

        Args:
            config: Strategy configuration
            funding_monitor: Optional funding rate monitor (created if not provided)
        """
        super().__init__(config)
        self._arb_config = config

        # Funding rate monitor
        self._monitor = funding_monitor

        # Active positions
        self._positions: dict[str, FundingArbitragePosition] = {}

        # Strategy state
        self._state = FundingArbitrageState.IDLE

        # Metrics
        self._total_funding_collected = Decimal("0")
        self._total_trades = 0
        self._total_pnl = Decimal("0")

        # Last check times
        self._last_rate_check_ns = 0
        self._last_rebalance_check_ns = 0

    @property
    def name(self) -> str:
        """Strategy name."""
        return "funding_rate_arbitrage"

    @property
    def state(self) -> FundingArbitrageState:
        """Current strategy state."""
        return self._state

    @property
    def positions(self) -> dict[str, FundingArbitragePosition]:
        """Active arbitrage positions."""
        return self._positions.copy()

    @property
    def total_funding_collected(self) -> Decimal:
        """Total funding collected across all positions."""
        return self._total_funding_collected

    @property
    def arb_config(self) -> FundingArbitrageConfig:
        """Arbitrage-specific configuration."""
        return self._arb_config

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """Start the strategy."""
        super().on_start()
        self._state = FundingArbitrageState.IDLE
        logger.info("Funding rate arbitrage strategy started")

    def on_stop(self) -> None:
        """Stop the strategy."""
        super().on_stop()
        self._state = FundingArbitrageState.IDLE
        logger.info(
            "Funding rate arbitrage strategy stopped. "
            "Total funding collected: %s, Total P&L: %s",
            self._total_funding_collected,
            self._total_pnl,
        )

    def on_reset(self) -> None:
        """Reset strategy state."""
        super().on_reset()
        self._positions.clear()
        self._state = FundingArbitrageState.IDLE
        self._total_funding_collected = Decimal("0")
        self._total_trades = 0
        self._total_pnl = Decimal("0")

    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> Signal | None:
        """
        Process bar data and generate signals.

        For funding rate arbitrage, we primarily react to funding rate
        changes rather than price bars. This method handles position
        updates and rebalancing checks.
        """
        self._bar_count += 1

        # Update P&L for existing positions
        self._update_position_pnl(bar.symbol, bar.close)

        # Check for exit conditions
        exit_signal = self._check_exit_conditions(bar.symbol, bar)
        if exit_signal:
            return exit_signal

        # Check for rebalance needs
        rebalance_signal = self._check_rebalance(bar.symbol)
        if rebalance_signal:
            return rebalance_signal

        return None

    def on_funding_rate(self, rate_data: FundingRateData) -> Signal | None:
        """
        Process funding rate update and generate entry/exit signals.

        This is the primary signal generation method for this strategy.

        Args:
            rate_data: Current funding rate data

        Returns:
            Trading signal or None
        """
        symbol = rate_data.symbol

        # Check if we have an existing position
        if symbol in self._positions:
            # Check for exit conditions
            return self._evaluate_exit(symbol, rate_data)
        else:
            # Check for entry opportunity
            return self._evaluate_entry(rate_data)

    def on_funding_payment(
        self,
        symbol: str,
        amount: Decimal,
    ) -> None:
        """
        Record a funding payment receipt.

        Called when a funding settlement occurs.

        Args:
            symbol: Symbol that received funding
            amount: Funding amount (positive = received, negative = paid)
        """
        if symbol in self._positions:
            self._positions[symbol].add_funding_payment(amount)
            self._total_funding_collected += amount

            logger.info(
                "Funding payment for %s: %s (cumulative: %s)",
                symbol,
                amount,
                self._positions[symbol].cumulative_funding,
            )

    # -------------------------------------------------------------------------
    # Entry Logic
    # -------------------------------------------------------------------------

    def _evaluate_entry(self, rate_data: FundingRateData) -> Signal | None:
        """Evaluate if we should enter a new position."""
        # Check if we have capacity for new positions
        if len(self._positions) >= self._arb_config.max_positions:
            return None

        # Check minimum funding rate
        abs_rate = abs(rate_data.funding_rate)
        if abs_rate < self._arb_config.min_funding_rate:
            return None

        # Check annualized return
        apr = abs_rate * FUNDING_PERIODS_PER_YEAR
        if apr < self._arb_config.min_annualized_return:
            return None

        # Check basis deviation
        if abs(rate_data.basis) > self._arb_config.max_basis_deviation:
            logger.debug(
                "Skipping %s: basis %.4f%% exceeds max %.4f%%",
                rate_data.symbol,
                float(rate_data.basis * 100),
                float(self._arb_config.max_basis_deviation * 100),
            )
            return None

        # Determine direction
        if rate_data.is_positive:
            # Positive funding: short perp to collect from longs
            direction = ArbitrageDirection.LONG_SPOT_SHORT_PERP
            signal_type = SignalType.SHORT  # Primary signal is to short perp
        else:
            # Negative funding: long perp to collect from shorts
            direction = ArbitrageDirection.SHORT_SPOT_LONG_PERP
            signal_type = SignalType.LONG  # Primary signal is to long perp

        # Calculate position size
        position_size = self._calculate_position_size(rate_data)

        # Generate entry signal
        signal = Signal.create(
            signal_type=signal_type,
            symbol=rate_data.symbol,
            strength=float(min(Decimal("1.0"), apr / Decimal("0.5"))),  # 50% APR = max strength
            price=rate_data.mark_price,
            metadata={
                "strategy": self.name,
                "action": "open_arbitrage",
                "direction": direction.value,
                "funding_rate": str(rate_data.funding_rate),
                "annualized_rate": str(apr),
                "basis_bps": str(rate_data.basis_bps),
                "position_size": str(position_size),
                "spot_exchange": self._arb_config.spot_exchange,
                "perp_exchange": self._arb_config.perp_exchange,
            },
        )

        self._signal_count += 1
        logger.info(
            "Entry signal for %s: %s, funding=%.4f%%, APR=%.2f%%",
            rate_data.symbol,
            direction.value,
            float(rate_data.funding_rate * 100),
            float(apr * 100),
        )

        return signal

    def _calculate_position_size(self, rate_data: FundingRateData) -> Decimal:
        """Calculate position size based on config and risk limits."""
        # Base size from config percentage (would need equity access)
        base_size = self._arb_config.max_position_size_usd

        # Adjust for funding rate strength (higher rate = larger position)
        rate_multiplier = min(
            Decimal("1.0"),
            abs(rate_data.funding_rate) / (self._arb_config.min_funding_rate * 5),
        )

        # Convert to asset units
        if rate_data.index_price > 0:
            size_in_units = (base_size * rate_multiplier) / rate_data.index_price
        else:
            size_in_units = Decimal("0")

        return size_in_units.quantize(Decimal("0.0001"))

    # -------------------------------------------------------------------------
    # Exit Logic
    # -------------------------------------------------------------------------

    def _evaluate_exit(
        self,
        symbol: str,
        rate_data: FundingRateData,
    ) -> Signal | None:
        """Evaluate if we should exit an existing position."""
        position = self._positions.get(symbol)
        if not position:
            return None

        exit_reason: str | None = None

        # Check for rate reversal
        if self._arb_config.exit_on_rate_reversal:
            if position.direction == ArbitrageDirection.LONG_SPOT_SHORT_PERP:
                # Was collecting positive funding, now rate is negative
                if rate_data.funding_rate < self._arb_config.rate_reversal_threshold:
                    exit_reason = "rate_reversal_to_negative"
            else:
                # Was collecting negative funding, now rate is positive
                if rate_data.funding_rate > -self._arb_config.rate_reversal_threshold:
                    exit_reason = "rate_reversal_to_positive"

        # Check minimum holding period
        if exit_reason and position.funding_payments_count < self._arb_config.min_hold_periods:
            # Only exit for rate reversal if we haven't held long enough
            # unless it's a significant reversal
            if abs(rate_data.funding_rate) < self._arb_config.min_funding_rate * 2:
                exit_reason = None  # Don't exit yet

        # Check basis deviation
        if abs(rate_data.basis) > self._arb_config.max_basis_deviation * Decimal("1.5"):
            exit_reason = "excessive_basis_deviation"

        # Check stop loss
        if position.total_pnl < -(position.spot_size * position.entry_spot_price * self._arb_config.stop_loss_pct):
            exit_reason = "stop_loss"

        if exit_reason:
            # Generate exit signal
            signal_type = (
                SignalType.CLOSE_SHORT
                if position.direction == ArbitrageDirection.LONG_SPOT_SHORT_PERP
                else SignalType.CLOSE_LONG
            )

            signal = Signal.create(
                signal_type=signal_type,
                symbol=symbol,
                strength=1.0,
                price=rate_data.mark_price,
                metadata={
                    "strategy": self.name,
                    "action": "close_arbitrage",
                    "exit_reason": exit_reason,
                    "position_pnl": str(position.total_pnl),
                    "funding_collected": str(position.cumulative_funding),
                    "holding_hours": str(position.holding_time_hours),
                },
            )

            self._signal_count += 1
            logger.info(
                "Exit signal for %s: reason=%s, P&L=%s, funding=%s",
                symbol,
                exit_reason,
                position.total_pnl,
                position.cumulative_funding,
            )

            return signal

        return None

    def _check_exit_conditions(self, symbol: str, bar: Bar) -> Signal | None:
        """Check exit conditions based on price data."""
        position = self._positions.get(symbol)
        if not position:
            return None

        # Check stop loss based on current price
        entry_value = position.spot_size * position.entry_spot_price

        if position.total_pnl < -(entry_value * self._arb_config.stop_loss_pct):
            signal_type = (
                SignalType.CLOSE_SHORT
                if position.direction == ArbitrageDirection.LONG_SPOT_SHORT_PERP
                else SignalType.CLOSE_LONG
            )

            return Signal.create(
                signal_type=signal_type,
                symbol=symbol,
                strength=1.0,
                price=bar.close,
                metadata={
                    "strategy": self.name,
                    "action": "close_arbitrage",
                    "exit_reason": "stop_loss",
                    "position_pnl": str(position.total_pnl),
                },
            )

        return None

    # -------------------------------------------------------------------------
    # Rebalancing
    # -------------------------------------------------------------------------

    def _check_rebalance(self, symbol: str) -> Signal | None:
        """Check if position needs rebalancing to maintain delta neutrality."""
        position = self._positions.get(symbol)
        if not position:
            return None

        # Check delta drift
        delta_pct = abs(position.net_delta / position.spot_size) if position.spot_size > 0 else Decimal("0")

        if delta_pct > self._arb_config.max_delta_drift:
            logger.info(
                "Rebalance needed for %s: delta drift %.2f%%",
                symbol,
                float(delta_pct * 100),
            )

            # Generate rebalance signal
            return Signal.create(
                signal_type=SignalType.HOLD,  # Special signal for rebalance
                symbol=symbol,
                strength=float(delta_pct),
                metadata={
                    "strategy": self.name,
                    "action": "rebalance",
                    "delta_drift_pct": str(delta_pct * 100),
                    "net_delta": str(position.net_delta),
                },
            )

        return None

    def _update_position_pnl(self, symbol: str, current_price: Decimal) -> None:
        """Update P&L for a position."""
        if symbol in self._positions:
            self._positions[symbol].update_pnl(current_price, current_price)

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------

    def register_position(
        self,
        symbol: str,
        direction: ArbitrageDirection,
        spot_size: Decimal,
        perp_size: Decimal,
        spot_price: Decimal,
        perp_price: Decimal,
        funding_rate: Decimal,
    ) -> FundingArbitragePosition:
        """
        Register a new arbitrage position.

        Called after successful order execution to track the position.
        """
        position = FundingArbitragePosition(
            symbol=symbol,
            direction=direction,
            spot_size=spot_size,
            perp_size=perp_size,
            entry_funding_rate=funding_rate,
            entry_spot_price=spot_price,
            entry_perp_price=perp_price,
            entry_time_ns=time.time_ns(),
        )

        self._positions[symbol] = position
        self._total_trades += 2  # Spot + perp trades
        self._state = FundingArbitrageState.ACTIVE

        logger.info(
            "Registered arbitrage position: %s %s, size=%s, funding=%.4f%%",
            symbol,
            direction.value,
            spot_size,
            float(funding_rate * 100),
        )

        return position

    def close_position(self, symbol: str) -> FundingArbitragePosition | None:
        """
        Close and remove an arbitrage position.

        Called after successful exit order execution.
        """
        position = self._positions.pop(symbol, None)
        if position:
            self._total_pnl += position.total_pnl
            self._total_trades += 2  # Spot + perp close trades

            logger.info(
                "Closed arbitrage position: %s, P&L=%s, funding=%s",
                symbol,
                position.total_pnl,
                position.cumulative_funding,
            )

            if not self._positions:
                self._state = FundingArbitrageState.IDLE

        return position

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def on_save(self) -> dict[str, Any]:
        """Save strategy state."""
        return {
            "positions": {
                symbol: {
                    "direction": pos.direction.value,
                    "spot_size": str(pos.spot_size),
                    "perp_size": str(pos.perp_size),
                    "entry_funding_rate": str(pos.entry_funding_rate),
                    "entry_spot_price": str(pos.entry_spot_price),
                    "entry_perp_price": str(pos.entry_perp_price),
                    "entry_time_ns": pos.entry_time_ns,
                    "cumulative_funding": str(pos.cumulative_funding),
                    "funding_payments_count": pos.funding_payments_count,
                }
                for symbol, pos in self._positions.items()
            },
            "total_funding_collected": str(self._total_funding_collected),
            "total_trades": self._total_trades,
            "total_pnl": str(self._total_pnl),
            "state": self._state.value,
        }

    def on_load(self, data: dict[str, Any]) -> None:
        """Load strategy state."""
        self._positions.clear()

        for symbol, pos_data in data.get("positions", {}).items():
            self._positions[symbol] = FundingArbitragePosition(
                symbol=symbol,
                direction=ArbitrageDirection(pos_data["direction"]),
                spot_size=Decimal(pos_data["spot_size"]),
                perp_size=Decimal(pos_data["perp_size"]),
                entry_funding_rate=Decimal(pos_data["entry_funding_rate"]),
                entry_spot_price=Decimal(pos_data["entry_spot_price"]),
                entry_perp_price=Decimal(pos_data["entry_perp_price"]),
                entry_time_ns=pos_data["entry_time_ns"],
                cumulative_funding=Decimal(pos_data["cumulative_funding"]),
                funding_payments_count=pos_data["funding_payments_count"],
            )

        self._total_funding_collected = Decimal(data.get("total_funding_collected", "0"))
        self._total_trades = data.get("total_trades", 0)
        self._total_pnl = Decimal(data.get("total_pnl", "0"))
        self._state = FundingArbitrageState(data.get("state", "idle"))


# =============================================================================
# Demo/Mock Data
# =============================================================================


def create_demo_funding_rates() -> list[FundingRateData]:
    """Create demo funding rate data for testing."""
    now = time.time_ns()
    next_funding = now + 4 * 3600 * 1_000_000_000  # 4 hours from now

    return [
        FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0003"),  # 0.03%
            next_funding_time_ns=next_funding,
            mark_price=Decimal("95000.00"),
            index_price=Decimal("94950.00"),
            predicted_rate=Decimal("0.00025"),
        ),
        FundingRateData(
            symbol="ETH/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0002"),  # 0.02%
            next_funding_time_ns=next_funding,
            mark_price=Decimal("3400.00"),
            index_price=Decimal("3395.00"),
            predicted_rate=Decimal("0.00018"),
        ),
        FundingRateData(
            symbol="SOL/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("-0.0001"),  # -0.01% (shorts pay longs)
            next_funding_time_ns=next_funding,
            mark_price=Decimal("180.00"),
            index_price=Decimal("180.50"),
            predicted_rate=Decimal("-0.00008"),
        ),
        FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="bybit",
            funding_rate=Decimal("0.00035"),  # 0.035% (higher than Binance)
            next_funding_time_ns=next_funding,
            mark_price=Decimal("95010.00"),
            index_price=Decimal("94950.00"),
        ),
    ]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ArbitrageDirection",
    "FundingArbitrageState",
    # Data structures
    "FundingRateData",
    "FundingArbitragePosition",
    "FundingArbitrageConfig",
    # Providers
    "FundingRateProvider",
    "FundingRateMonitor",
    "CCXTFundingRateProvider",
    # Strategy
    "FundingRateArbitrageStrategy",
    # Demo
    "create_demo_funding_rates",
]

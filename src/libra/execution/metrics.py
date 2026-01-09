"""
Transaction Cost Analysis (TCA) Metrics for Execution Quality.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

Provides comprehensive metrics for measuring execution quality,
including implementation shortfall, slippage, and benchmark comparisons.

Key Metrics:
- Implementation Shortfall: Cost vs arrival price
- VWAP Slippage: Deviation from VWAP benchmark
- TWAP Slippage: Deviation from TWAP benchmark
- Market Impact: Price movement caused by execution
- Timing Slippage: Delay costs
- Fill Rate: Order completion quality

Use Cases:
- Post-trade analysis of execution quality
- Algorithm performance comparison
- Execution cost attribution
- Compliance reporting

Example:
    tca = ExecutionTCA(
        arrival_price=Decimal("50000"),
        arrival_timestamp_ns=start_time,
    )

    # Record fills during execution
    tca.record_fill(Decimal("10"), Decimal("50010"), timestamp)
    tca.record_fill(Decimal("15"), Decimal("50005"), timestamp)

    # Calculate final metrics
    tca.finalize(benchmark_vwap, benchmark_twap)
    print(f"Implementation shortfall: {tca.implementation_shortfall_bps} bps")

References:
- QuestDB Execution Slippage: https://questdb.com/glossary/execution-slippage-measurement/
- Talos Market Impact Model: https://www.talos.com/insights/understanding-market-impact
- Anboto Labs TCA: https://medium.com/@anboto_labs/slippage-benchmarks-and-beyond
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from libra.gateways.protocol import OrderSide


# =============================================================================
# Fill Record
# =============================================================================


@dataclass
class FillRecord:
    """
    Record of a single fill during execution.

    Tracks individual fills for detailed TCA analysis.
    """

    quantity: Decimal
    price: Decimal
    timestamp_ns: int
    fee: Decimal = Decimal("0")
    venue: str | None = None
    child_order_id: str | None = None


# =============================================================================
# TCA Metrics
# =============================================================================


@dataclass
class ExecutionTCA:
    """
    Transaction Cost Analysis for execution quality measurement.

    Provides comprehensive metrics for evaluating how well an
    execution was performed relative to various benchmarks.

    Attributes:
        arrival_price: Price when execution started
        arrival_timestamp_ns: Time when execution started
        side: Order side (buy/sell) for directional metrics

    Usage:
        tca = ExecutionTCA(
            arrival_price=Decimal("100"),
            arrival_timestamp_ns=time.time_ns(),
            side=OrderSide.BUY,
        )

        # During execution
        tca.record_fill(qty, price, timestamp)

        # After execution
        tca.finalize(vwap_benchmark, twap_benchmark)
        print(f"Shortfall: {tca.implementation_shortfall_bps:.2f} bps")
    """

    # Arrival benchmarks (set at start)
    arrival_price: Decimal
    arrival_timestamp_ns: int
    side: OrderSide | None = None

    # Execution results (updated during/after execution)
    total_quantity: Decimal = Decimal("0")
    total_value: Decimal = Decimal("0")  # Sum of qty * price
    avg_execution_price: Decimal = Decimal("0")
    completion_timestamp_ns: int | None = None

    # Individual fills
    fills: list[FillRecord] = field(default_factory=list)

    # Cost breakdown (basis points)
    spread_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0
    total_cost_bps: float = 0.0

    # Benchmark slippage (basis points)
    implementation_shortfall_bps: float = 0.0
    vwap_slippage_bps: float = 0.0
    twap_slippage_bps: float = 0.0
    arrival_slippage_bps: float = 0.0

    # Fill quality metrics
    num_fills: int = 0
    fill_rate: float = 0.0  # Fills vs orders submitted
    avg_fill_time_ms: float = 0.0
    total_fees: Decimal = Decimal("0")

    # Timing metrics
    execution_duration_ns: int = 0
    first_fill_delay_ns: int = 0
    last_fill_delay_ns: int = 0

    # Finalized flag
    _finalized: bool = False

    # -------------------------------------------------------------------------
    # Fill Recording
    # -------------------------------------------------------------------------

    def record_fill(
        self,
        quantity: Decimal,
        price: Decimal,
        timestamp_ns: int | None = None,
        fee: Decimal = Decimal("0"),
        venue: str | None = None,
        child_order_id: str | None = None,
    ) -> None:
        """
        Record a fill during execution.

        Call this for each partial or complete fill to build
        up the execution record.

        Args:
            quantity: Fill quantity
            price: Fill price
            timestamp_ns: Fill timestamp (uses current time if None)
            fee: Transaction fee
            venue: Execution venue
            child_order_id: Associated child order ID
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        fill = FillRecord(
            quantity=quantity,
            price=price,
            timestamp_ns=timestamp_ns,
            fee=fee,
            venue=venue,
            child_order_id=child_order_id,
        )
        self.fills.append(fill)

        # Update running totals
        self.total_quantity += quantity
        self.total_value += quantity * price
        self.total_fees += fee
        self.num_fills += 1

        # Update average price
        if self.total_quantity > 0:
            self.avg_execution_price = self.total_value / self.total_quantity

        # Track timing
        if self.num_fills == 1:
            self.first_fill_delay_ns = timestamp_ns - self.arrival_timestamp_ns
        self.last_fill_delay_ns = timestamp_ns - self.arrival_timestamp_ns

    # -------------------------------------------------------------------------
    # Finalization
    # -------------------------------------------------------------------------

    def finalize(
        self,
        benchmark_vwap: Decimal | None = None,
        benchmark_twap: Decimal | None = None,
        orders_submitted: int | None = None,
        completion_timestamp_ns: int | None = None,
    ) -> None:
        """
        Finalize TCA calculations after execution.

        Call this after all fills are recorded to compute
        final metrics.

        Args:
            benchmark_vwap: Market VWAP during execution period
            benchmark_twap: Market TWAP during execution period
            orders_submitted: Total orders submitted (for fill rate)
            completion_timestamp_ns: Execution completion time
        """
        if completion_timestamp_ns is None:
            completion_timestamp_ns = time.time_ns()

        self.completion_timestamp_ns = completion_timestamp_ns
        self.execution_duration_ns = completion_timestamp_ns - self.arrival_timestamp_ns

        # Calculate fill rate
        if orders_submitted and orders_submitted > 0:
            self.fill_rate = self.num_fills / orders_submitted

        # Calculate average fill time
        if self.num_fills > 0 and self.execution_duration_ns > 0:
            self.avg_fill_time_ms = (
                self.execution_duration_ns / self.num_fills / 1_000_000
            )

        # Calculate slippage metrics
        self._calculate_implementation_shortfall()

        if benchmark_vwap is not None:
            self._calculate_vwap_slippage(benchmark_vwap)

        if benchmark_twap is not None:
            self._calculate_twap_slippage(benchmark_twap)

        # Calculate total cost
        self.total_cost_bps = (
            self.spread_cost_bps + self.market_impact_bps + self.timing_cost_bps
        )

        self._finalized = True

    def _calculate_implementation_shortfall(self) -> None:
        """Calculate implementation shortfall vs arrival price."""
        if self.arrival_price == 0 or self.total_quantity == 0:
            return

        # Import here to avoid circular import
        from libra.gateways.protocol import OrderSide

        if self.side == OrderSide.BUY:
            # For buys: positive shortfall = paid more than arrival
            price_diff = self.avg_execution_price - self.arrival_price
        elif self.side == OrderSide.SELL:
            # For sells: positive shortfall = received less than arrival
            price_diff = self.arrival_price - self.avg_execution_price
        else:
            # No side specified, use absolute difference
            price_diff = abs(self.avg_execution_price - self.arrival_price)

        self.implementation_shortfall_bps = float(
            price_diff / self.arrival_price * Decimal("10000")
        )
        self.arrival_slippage_bps = self.implementation_shortfall_bps

    def _calculate_vwap_slippage(self, benchmark_vwap: Decimal) -> None:
        """Calculate slippage vs VWAP benchmark."""
        if benchmark_vwap == 0 or self.total_quantity == 0:
            return

        from libra.gateways.protocol import OrderSide

        if self.side == OrderSide.BUY:
            price_diff = self.avg_execution_price - benchmark_vwap
        elif self.side == OrderSide.SELL:
            price_diff = benchmark_vwap - self.avg_execution_price
        else:
            price_diff = abs(self.avg_execution_price - benchmark_vwap)

        self.vwap_slippage_bps = float(
            price_diff / benchmark_vwap * Decimal("10000")
        )

    def _calculate_twap_slippage(self, benchmark_twap: Decimal) -> None:
        """Calculate slippage vs TWAP benchmark."""
        if benchmark_twap == 0 or self.total_quantity == 0:
            return

        from libra.gateways.protocol import OrderSide

        if self.side == OrderSide.BUY:
            price_diff = self.avg_execution_price - benchmark_twap
        elif self.side == OrderSide.SELL:
            price_diff = benchmark_twap - self.avg_execution_price
        else:
            price_diff = abs(self.avg_execution_price - benchmark_twap)

        self.twap_slippage_bps = float(
            price_diff / benchmark_twap * Decimal("10000")
        )

    # -------------------------------------------------------------------------
    # Cost Attribution
    # -------------------------------------------------------------------------

    def attribute_costs(
        self,
        bid_ask_spread_bps: float,
        estimated_impact_bps: float | None = None,
    ) -> None:
        """
        Attribute execution costs to components.

        Breaks down total costs into spread, impact, and timing
        components for analysis.

        Args:
            bid_ask_spread_bps: Average bid-ask spread during execution
            estimated_impact_bps: Pre-trade estimated market impact
        """
        # Spread cost: Half the spread for crossing
        self.spread_cost_bps = bid_ask_spread_bps / 2

        # Market impact: Remaining shortfall beyond spread
        if self.implementation_shortfall_bps > self.spread_cost_bps:
            self.market_impact_bps = (
                self.implementation_shortfall_bps - self.spread_cost_bps
            )
        else:
            self.market_impact_bps = 0.0

        # Timing cost: If we have estimated impact, difference is timing
        if estimated_impact_bps is not None:
            self.timing_cost_bps = max(
                0.0, self.market_impact_bps - estimated_impact_bps
            )
            self.market_impact_bps = min(
                self.market_impact_bps, estimated_impact_bps
            )

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Convert metrics to dictionary for reporting."""
        return {
            "arrival_price": str(self.arrival_price),
            "avg_execution_price": str(self.avg_execution_price),
            "total_quantity": str(self.total_quantity),
            "total_value": str(self.total_value),
            "num_fills": self.num_fills,
            "fill_rate": self.fill_rate,
            "implementation_shortfall_bps": self.implementation_shortfall_bps,
            "vwap_slippage_bps": self.vwap_slippage_bps,
            "twap_slippage_bps": self.twap_slippage_bps,
            "spread_cost_bps": self.spread_cost_bps,
            "market_impact_bps": self.market_impact_bps,
            "timing_cost_bps": self.timing_cost_bps,
            "total_cost_bps": self.total_cost_bps,
            "execution_duration_ms": self.execution_duration_ns / 1_000_000,
            "avg_fill_time_ms": self.avg_fill_time_ms,
            "total_fees": str(self.total_fees),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Execution TCA Summary ===",
            f"Arrival Price: {self.arrival_price}",
            f"Avg Execution Price: {self.avg_execution_price}",
            f"Total Quantity: {self.total_quantity}",
            f"Num Fills: {self.num_fills}",
            "",
            "--- Slippage (bps) ---",
            f"Implementation Shortfall: {self.implementation_shortfall_bps:+.2f}",
            f"VWAP Slippage: {self.vwap_slippage_bps:+.2f}",
            f"TWAP Slippage: {self.twap_slippage_bps:+.2f}",
            "",
            "--- Cost Attribution (bps) ---",
            f"Spread Cost: {self.spread_cost_bps:.2f}",
            f"Market Impact: {self.market_impact_bps:.2f}",
            f"Timing Cost: {self.timing_cost_bps:.2f}",
            f"Total Cost: {self.total_cost_bps:.2f}",
            "",
            "--- Timing ---",
            f"Duration: {self.execution_duration_ns / 1e9:.2f}s",
            f"Avg Fill Time: {self.avg_fill_time_ms:.1f}ms",
        ]
        return "\n".join(lines)


# =============================================================================
# Aggregated TCA
# =============================================================================


@dataclass
class AggregatedTCA:
    """
    Aggregated TCA metrics across multiple executions.

    Useful for analyzing algorithm performance over time
    or comparing different algorithms.
    """

    num_executions: int = 0
    total_quantity: Decimal = Decimal("0")
    total_value: Decimal = Decimal("0")

    # Averages (weighted by quantity)
    avg_implementation_shortfall_bps: float = 0.0
    avg_vwap_slippage_bps: float = 0.0
    avg_twap_slippage_bps: float = 0.0
    avg_total_cost_bps: float = 0.0

    # Best/Worst
    best_shortfall_bps: float = float("inf")
    worst_shortfall_bps: float = float("-inf")

    # Consistency
    shortfall_std_dev: float = 0.0

    # Running sums for weighted average calculation
    _weighted_shortfall_sum: float = 0.0
    _weighted_vwap_sum: float = 0.0
    _weighted_twap_sum: float = 0.0
    _weighted_cost_sum: float = 0.0
    _shortfall_values: list[float] = field(default_factory=list)

    def add_execution(self, tca: ExecutionTCA) -> None:
        """
        Add an execution's TCA to the aggregate.

        Args:
            tca: TCA metrics from a single execution
        """
        if not tca._finalized:
            raise ValueError("TCA must be finalized before aggregation")

        qty_float = float(tca.total_quantity)

        self.num_executions += 1
        self.total_quantity += tca.total_quantity
        self.total_value += tca.total_value

        # Weighted sums
        self._weighted_shortfall_sum += tca.implementation_shortfall_bps * qty_float
        self._weighted_vwap_sum += tca.vwap_slippage_bps * qty_float
        self._weighted_twap_sum += tca.twap_slippage_bps * qty_float
        self._weighted_cost_sum += tca.total_cost_bps * qty_float

        # Track for std dev
        self._shortfall_values.append(tca.implementation_shortfall_bps)

        # Best/worst
        self.best_shortfall_bps = min(
            self.best_shortfall_bps, tca.implementation_shortfall_bps
        )
        self.worst_shortfall_bps = max(
            self.worst_shortfall_bps, tca.implementation_shortfall_bps
        )

        # Recalculate averages
        total_qty = float(self.total_quantity)
        if total_qty > 0:
            self.avg_implementation_shortfall_bps = (
                self._weighted_shortfall_sum / total_qty
            )
            self.avg_vwap_slippage_bps = self._weighted_vwap_sum / total_qty
            self.avg_twap_slippage_bps = self._weighted_twap_sum / total_qty
            self.avg_total_cost_bps = self._weighted_cost_sum / total_qty

        # Calculate std dev
        if len(self._shortfall_values) > 1:
            mean = sum(self._shortfall_values) / len(self._shortfall_values)
            variance = sum(
                (x - mean) ** 2 for x in self._shortfall_values
            ) / len(self._shortfall_values)
            self.shortfall_std_dev = variance ** 0.5

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"=== Aggregated TCA ({self.num_executions} executions) ===\n"
            f"Total Quantity: {self.total_quantity}\n"
            f"Avg Shortfall: {self.avg_implementation_shortfall_bps:+.2f} bps\n"
            f"Avg VWAP Slip: {self.avg_vwap_slippage_bps:+.2f} bps\n"
            f"Avg Total Cost: {self.avg_total_cost_bps:.2f} bps\n"
            f"Best: {self.best_shortfall_bps:+.2f} bps\n"
            f"Worst: {self.worst_shortfall_bps:+.2f} bps\n"
            f"Std Dev: {self.shortfall_std_dev:.2f} bps"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_tca(
    arrival_price: Decimal,
    side: OrderSide | None = None,
) -> ExecutionTCA:
    """
    Create a new TCA tracker for an execution.

    Args:
        arrival_price: Price at execution start
        side: Order side for directional metrics

    Returns:
        ExecutionTCA instance ready for fill recording
    """
    return ExecutionTCA(
        arrival_price=arrival_price,
        arrival_timestamp_ns=time.time_ns(),
        side=side,
    )

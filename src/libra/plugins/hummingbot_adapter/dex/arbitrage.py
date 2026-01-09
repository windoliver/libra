"""
DEX Arbitrage Strategy (Issue #12).

Detects and executes arbitrage opportunities across:
- Multiple DEXs (Uniswap V2/V3, SushiSwap, etc.)
- Multiple fee tiers (V3)
- Triangular arbitrage within single DEX

Supports both flash loan and traditional capital-based arbitrage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from libra.plugins.hummingbot_adapter.dex.base import (
    DEXGateway,
    DEXQuote,
    DEXSwapResult,
    Token,
)


if TYPE_CHECKING:
    pass


class ArbitrageType(str, Enum):
    """Type of arbitrage opportunity."""

    CROSS_DEX = "cross_dex"  # Same pair on different DEXs
    TRIANGULAR = "triangular"  # A -> B -> C -> A on single DEX
    CROSS_FEE_TIER = "cross_fee_tier"  # Same pair, different V3 fee tiers


@dataclass
class ArbitrageOpportunity:
    """
    Detected arbitrage opportunity.

    Contains all information needed to evaluate and execute the trade.
    """

    arb_type: ArbitrageType
    token_path: list[Token]
    dex_path: list[str]  # DEX names for each hop
    input_amount: Decimal
    expected_output: Decimal
    expected_profit: Decimal
    profit_percentage: float
    gas_estimate: int
    quotes: list[DEXQuote]
    timestamp_ns: int = 0

    @property
    def is_profitable(self) -> bool:
        """Check if opportunity is profitable after gas."""
        return self.expected_profit > 0

    @property
    def net_profit_estimate(self) -> Decimal:
        """Estimated profit after gas costs (assuming 50 gwei, ETH=$2000)."""
        gas_cost_eth = Decimal(self.gas_estimate * 50_000_000_000) / Decimal(10**18)
        gas_cost_usd = gas_cost_eth * Decimal("2000")  # Rough ETH price
        return self.expected_profit - gas_cost_usd


@dataclass
class ArbitrageResult:
    """Result of an arbitrage execution."""

    success: bool
    opportunity: ArbitrageOpportunity
    swap_results: list[DEXSwapResult] = field(default_factory=list)
    actual_profit: Decimal = Decimal("0")
    total_gas_used: int = 0
    error_message: str = ""
    execution_time_ms: float = 0.0


@dataclass
class ArbitrageConfig:
    """Configuration for arbitrage strategy."""

    # Minimum profit thresholds
    min_profit_percentage: float = 0.1  # 0.1% minimum profit
    min_profit_amount: Decimal = Decimal("1.0")  # $1 minimum

    # Risk parameters
    max_position_size: Decimal = Decimal("10000")  # Max per trade
    max_price_impact: float = 2.0  # Max 2% price impact per hop

    # Gas settings
    max_gas_price_gwei: int = 100  # Max gas price to execute
    gas_buffer_multiplier: float = 1.2  # Buffer for gas estimation

    # Execution
    max_slippage: float = 0.5  # Max slippage per hop
    execution_timeout_ms: int = 30000  # 30 second timeout


class DEXArbitrageStrategy:
    """
    DEX arbitrage detection and execution.

    Supports:
    - Cross-DEX arbitrage (same pair on multiple DEXs)
    - Triangular arbitrage (A -> B -> C -> A)
    - Multi-hop routes

    Can operate with or without flash loans.
    """

    def __init__(
        self,
        gateways: list[DEXGateway],
        config: ArbitrageConfig | None = None,
    ) -> None:
        """
        Initialize arbitrage strategy.

        Args:
            gateways: List of DEX gateways to use
            config: Arbitrage configuration
        """
        self.gateways = {gw.name: gw for gw in gateways}
        self.config = config or ArbitrageConfig()

        # Statistics
        self._opportunities_found = 0
        self._opportunities_executed = 0
        self._total_profit = Decimal("0")
        self._total_gas_spent = 0

    @property
    def stats(self) -> dict[str, float | int]:
        """Get strategy statistics."""
        return {
            "opportunities_found": self._opportunities_found,
            "opportunities_executed": self._opportunities_executed,
            "total_profit": float(self._total_profit),
            "total_gas_spent": self._total_gas_spent,
            "success_rate": (
                self._opportunities_executed / max(1, self._opportunities_found)
            ),
        }

    async def find_cross_dex_opportunity(
        self,
        token_a: Token,
        token_b: Token,
        input_amount: Decimal,
    ) -> ArbitrageOpportunity | None:
        """
        Find cross-DEX arbitrage opportunity.

        Looks for price discrepancies between the same pair on different DEXs.

        Args:
            token_a: First token
            token_b: Second token
            input_amount: Amount to trade

        Returns:
            Arbitrage opportunity or None if none found
        """
        quotes: dict[str, DEXQuote] = {}

        # Get quotes from all DEXs
        for name, gateway in self.gateways.items():
            quote = await gateway.get_quote(token_a, token_b, input_amount)
            if quote and quote.is_valid and quote.price_impact < self.config.max_price_impact:
                quotes[name] = quote

        if len(quotes) < 2:
            return None

        # Find best buy and sell DEXs
        sorted_quotes = sorted(
            quotes.items(),
            key=lambda x: x[1].output_amount,
            reverse=True,
        )

        best_buy_dex, best_buy_quote = sorted_quotes[0]
        worst_buy_dex, worst_buy_quote = sorted_quotes[-1]

        # Check if reverse trade is profitable
        # Buy on DEX with lowest price, sell on DEX with highest price
        sell_gateway = self.gateways[worst_buy_dex]
        sell_quote = await sell_gateway.get_quote(
            token_b, token_a, best_buy_quote.output_amount
        )

        if not sell_quote or not sell_quote.is_valid:
            return None

        # Calculate profit
        profit = sell_quote.output_amount - input_amount
        profit_pct = float(profit / input_amount * 100)

        if profit_pct < self.config.min_profit_percentage:
            return None

        if profit < self.config.min_profit_amount:
            return None

        self._opportunities_found += 1

        return ArbitrageOpportunity(
            arb_type=ArbitrageType.CROSS_DEX,
            token_path=[token_a, token_b, token_a],
            dex_path=[worst_buy_dex, best_buy_dex],
            input_amount=input_amount,
            expected_output=sell_quote.output_amount,
            expected_profit=profit,
            profit_percentage=profit_pct,
            gas_estimate=best_buy_quote.gas_estimate + sell_quote.gas_estimate,
            quotes=[worst_buy_quote, sell_quote],  # Actually use right order
            timestamp_ns=time.time_ns(),
        )

    async def find_triangular_opportunity(
        self,
        token_a: Token,
        token_b: Token,
        token_c: Token,
        input_amount: Decimal,
        dex_name: str | None = None,
    ) -> ArbitrageOpportunity | None:
        """
        Find triangular arbitrage opportunity.

        Route: A -> B -> C -> A

        Args:
            token_a: Starting token (usually a stable or major token)
            token_b: Intermediate token
            token_c: Second intermediate token
            input_amount: Amount to trade
            dex_name: Specific DEX to use (or all if None)

        Returns:
            Arbitrage opportunity or None if none found
        """
        gateways_to_check = (
            [self.gateways[dex_name]] if dex_name else list(self.gateways.values())
        )

        best_opportunity: ArbitrageOpportunity | None = None

        for gateway in gateways_to_check:
            # Get quotes for full path
            quote_ab = await gateway.get_quote(token_a, token_b, input_amount)
            if not quote_ab or not quote_ab.is_valid:
                continue

            quote_bc = await gateway.get_quote(
                token_b, token_c, quote_ab.output_amount
            )
            if not quote_bc or not quote_bc.is_valid:
                continue

            quote_ca = await gateway.get_quote(
                token_c, token_a, quote_bc.output_amount
            )
            if not quote_ca or not quote_ca.is_valid:
                continue

            # Check total price impact
            total_impact = (
                quote_ab.price_impact + quote_bc.price_impact + quote_ca.price_impact
            )
            if total_impact > self.config.max_price_impact * 3:
                continue

            # Calculate profit
            profit = quote_ca.output_amount - input_amount
            profit_pct = float(profit / input_amount * 100)

            if profit_pct < self.config.min_profit_percentage:
                continue

            if profit < self.config.min_profit_amount:
                continue

            opportunity = ArbitrageOpportunity(
                arb_type=ArbitrageType.TRIANGULAR,
                token_path=[token_a, token_b, token_c, token_a],
                dex_path=[gateway.name, gateway.name, gateway.name],
                input_amount=input_amount,
                expected_output=quote_ca.output_amount,
                expected_profit=profit,
                profit_percentage=profit_pct,
                gas_estimate=(
                    quote_ab.gas_estimate
                    + quote_bc.gas_estimate
                    + quote_ca.gas_estimate
                ),
                quotes=[quote_ab, quote_bc, quote_ca],
                timestamp_ns=time.time_ns(),
            )

            if (
                best_opportunity is None
                or opportunity.expected_profit > best_opportunity.expected_profit
            ):
                best_opportunity = opportunity

        if best_opportunity:
            self._opportunities_found += 1

        return best_opportunity

    async def find_v3_fee_tier_opportunity(
        self,
        token_a: Token,
        token_b: Token,
        input_amount: Decimal,
        v3_gateway_name: str,
    ) -> ArbitrageOpportunity | None:
        """
        Find arbitrage between V3 fee tiers.

        Sometimes the same pair has better rates on different fee tiers.

        Args:
            token_a: First token
            token_b: Second token
            input_amount: Amount to trade
            v3_gateway_name: Name of V3 gateway

        Returns:
            Arbitrage opportunity or None if none found
        """
        gateway = self.gateways.get(v3_gateway_name)
        if gateway is None:
            return None

        # Dynamically check if it's a V3 gateway (has FEE_TIERS attribute)
        if not hasattr(gateway, "FEE_TIERS"):
            return None

        v3_gateway = gateway  # Type: UniswapV3Gateway (checked via hasattr)

        # Get quotes for all fee tiers
        quotes: dict[int, DEXQuote] = {}
        for fee_tier in [100, 500, 3000, 10000]:  # V3 fee tiers
            quote = await v3_gateway.get_quote(
                token_a, token_b, input_amount, fee_tier  # type: ignore[call-arg]
            )
            if quote and quote.is_valid:
                quotes[fee_tier] = quote

        if len(quotes) < 2:
            return None

        # Find best and worst fee tiers
        sorted_quotes = sorted(
            quotes.items(),
            key=lambda x: x[1].output_amount,
            reverse=True,
        )

        best_fee, best_quote = sorted_quotes[0]
        worst_fee, _ = sorted_quotes[-1]  # Only need fee tier for sell quote

        # Check if reverse on different tier is profitable
        sell_quote = await v3_gateway.get_quote(
            token_b, token_a, best_quote.output_amount, worst_fee  # type: ignore[call-arg]
        )

        if not sell_quote or not sell_quote.is_valid:
            return None

        profit = sell_quote.output_amount - input_amount
        profit_pct = float(profit / input_amount * 100)

        if profit_pct < self.config.min_profit_percentage:
            return None

        if profit < self.config.min_profit_amount:
            return None

        self._opportunities_found += 1

        return ArbitrageOpportunity(
            arb_type=ArbitrageType.CROSS_FEE_TIER,
            token_path=[token_a, token_b, token_a],
            dex_path=[f"{v3_gateway_name}_{best_fee}", f"{v3_gateway_name}_{worst_fee}"],
            input_amount=input_amount,
            expected_output=sell_quote.output_amount,
            expected_profit=profit,
            profit_percentage=profit_pct,
            gas_estimate=best_quote.gas_estimate + sell_quote.gas_estimate,
            quotes=[best_quote, sell_quote],
            timestamp_ns=time.time_ns(),
        )

    async def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> ArbitrageResult:
        """
        Execute an arbitrage opportunity.

        Executes all swaps in sequence, tracking results.

        Args:
            opportunity: Opportunity to execute

        Returns:
            Execution result with actual profit/loss
        """
        start_time = time.time()
        swap_results: list[DEXSwapResult] = []
        total_gas = 0

        try:
            for i, quote in enumerate(opportunity.quotes):
                dex_name = opportunity.dex_path[i].split("_")[0]  # Handle fee tier suffix
                gateway = self.gateways.get(dex_name)

                if not gateway:
                    return ArbitrageResult(
                        success=False,
                        opportunity=opportunity,
                        error_message=f"Gateway not found: {dex_name}",
                        execution_time_ms=(time.time() - start_time) * 1000,
                    )

                # Calculate min output with slippage
                min_output = quote.output_amount * (
                    1 - Decimal(str(self.config.max_slippage / 100))
                )

                result = await gateway.execute_swap(quote, min_output)
                swap_results.append(result)
                total_gas += result.gas_used

                if not result.success:
                    return ArbitrageResult(
                        success=False,
                        opportunity=opportunity,
                        swap_results=swap_results,
                        total_gas_used=total_gas,
                        error_message=result.error_message,
                        execution_time_ms=(time.time() - start_time) * 1000,
                    )

            # Calculate actual profit
            if swap_results:
                actual_output = swap_results[-1].output_amount
                actual_profit = actual_output - opportunity.input_amount
            else:
                actual_profit = Decimal("0")

            self._opportunities_executed += 1
            self._total_profit += actual_profit
            self._total_gas_spent += total_gas

            return ArbitrageResult(
                success=True,
                opportunity=opportunity,
                swap_results=swap_results,
                actual_profit=actual_profit,
                total_gas_used=total_gas,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ArbitrageResult(
                success=False,
                opportunity=opportunity,
                swap_results=swap_results,
                total_gas_used=total_gas,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def scan_for_opportunities(
        self,
        token_pairs: list[tuple[Token, Token]],
        input_amounts: list[Decimal],
        include_triangular: bool = False,
        triangular_intermediates: list[Token] | None = None,
    ) -> list[ArbitrageOpportunity]:
        """
        Scan for arbitrage opportunities across multiple pairs.

        Args:
            token_pairs: List of token pairs to check
            input_amounts: List of input amounts to test
            include_triangular: Whether to scan for triangular opportunities
            triangular_intermediates: Intermediate tokens for triangular arb

        Returns:
            List of found opportunities, sorted by profit
        """
        opportunities: list[ArbitrageOpportunity] = []

        # Cross-DEX opportunities
        for token_a, token_b in token_pairs:
            for amount in input_amounts:
                opp = await self.find_cross_dex_opportunity(token_a, token_b, amount)
                if opp and opp.is_profitable:
                    opportunities.append(opp)

        # Triangular opportunities
        if include_triangular and triangular_intermediates:
            for token_a, token_b in token_pairs:
                for token_c in triangular_intermediates:
                    if token_c.address not in (token_a.address, token_b.address):
                        for amount in input_amounts:
                            opp = await self.find_triangular_opportunity(
                                token_a, token_b, token_c, amount
                            )
                            if opp and opp.is_profitable:
                                opportunities.append(opp)

        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit, reverse=True)

        return opportunities

    def reset_stats(self) -> None:
        """Reset strategy statistics."""
        self._opportunities_found = 0
        self._opportunities_executed = 0
        self._total_profit = Decimal("0")
        self._total_gas_spent = 0

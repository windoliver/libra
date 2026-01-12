"""
Arbitrage Strategies Module.

Contains delta-neutral and cross-market arbitrage strategies:
- Funding Rate Arbitrage: Profit from perpetual funding payments
- (Future) Basis Arbitrage: Spot-futures basis trades
- (Future) Cross-Exchange Arbitrage: Price discrepancies across exchanges

See: https://github.com/windoliver/libra/issues/13
"""

from libra.strategies.arbitrage.funding_rate import (
    FundingArbitrageConfig,
    FundingArbitragePosition,
    FundingRateArbitrageStrategy,
    FundingRateData,
    FundingRateMonitor,
)

__all__ = [
    "FundingArbitrageConfig",
    "FundingArbitragePosition",
    "FundingRateArbitrageStrategy",
    "FundingRateData",
    "FundingRateMonitor",
]

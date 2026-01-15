"""
Options Data Models.

Comprehensive data models for options trading including:
- Core contract and position models
- Greeks (Delta, Gamma, Theta, Vega, Rho)
- Option chains with filtering
- Multi-leg strategy definitions
- Exercise/assignment handling

Issue #63: Options Data Models

Examples:
    from libra.core.options import (
        OptionContract, OptionType, OptionStyle,
        Greeks, GreeksSnapshot,
        OptionChain, OptionChainExpiry,
        OptionStrategy, create_vertical_spread,
    )

    # Create a contract
    contract = OptionContract(
        symbol="AAPL250117C00150000",
        underlying="AAPL",
        option_type=OptionType.CALL,
        strike=Decimal("150"),
        expiration=date(2025, 1, 17),
    )

    # Check moneyness
    if contract.is_itm(Decimal("160.00")):
        print("In the money!")

    # Create a strategy
    spread = create_vertical_spread(
        underlying="AAPL",
        expiration=date(2025, 1, 17),
        long_strike=Decimal("150"),
        short_strike=Decimal("155"),
        option_type=OptionType.CALL,
    )
"""

from libra.core.options.chains import (
    OptionChain,
    OptionChainEntry,
    OptionChainExpiry,
    build_option_chain_expiry,
)
from libra.core.options.exercise import (
    ExerciseEvent,
    ExerciseType,
    ExpirationAction,
    create_exercise_event,
)
from libra.core.options.greeks import (
    Greeks,
    GreeksSnapshot,
    greeks_from_dict,
    zero_greeks,
)
from libra.core.options.models import (
    OptionContract,
    OptionPosition,
    OptionStyle,
    OptionType,
    decode_option_contract,
    decode_option_position,
    encode_option_contract,
    encode_option_position,
)
from libra.core.options.strategies import (
    OptionStrategy,
    StrategyLeg,
    StrategyType,
    create_butterfly,
    create_iron_condor,
    create_straddle,
    create_strangle,
    create_vertical_spread,
)


__all__ = [
    # Enums
    "OptionType",
    "OptionStyle",
    "StrategyType",
    "ExerciseType",
    "ExpirationAction",
    # Core models
    "OptionContract",
    "OptionPosition",
    # Greeks
    "Greeks",
    "GreeksSnapshot",
    "greeks_from_dict",
    "zero_greeks",
    # Chains
    "OptionChainEntry",
    "OptionChainExpiry",
    "OptionChain",
    "build_option_chain_expiry",
    # Strategies
    "StrategyLeg",
    "OptionStrategy",
    "create_vertical_spread",
    "create_straddle",
    "create_strangle",
    "create_iron_condor",
    "create_butterfly",
    # Exercise
    "ExerciseEvent",
    "create_exercise_event",
    # Serialization
    "encode_option_contract",
    "decode_option_contract",
    "encode_option_position",
    "decode_option_position",
]

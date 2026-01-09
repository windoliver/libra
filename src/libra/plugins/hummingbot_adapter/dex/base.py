"""
DEX Gateway Base Classes (Issue #12).

Provides abstract interfaces for DEX integrations:
- Token representation
- Pool/pair information
- Quote fetching
- Swap execution

Supports both AMM (Uniswap V2 style) and CLMM (Uniswap V3 style) DEXs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class DEXType(str, Enum):
    """Type of decentralized exchange."""

    AMM_V2 = "amm_v2"  # Constant product (x*y=k) - Uniswap V2 style
    AMM_V3 = "amm_v3"  # Concentrated liquidity - Uniswap V3 style
    ORDER_BOOK = "order_book"  # On-chain order book - dYdX, Serum style


class ChainId(int, Enum):
    """Supported blockchain networks."""

    ETHEREUM = 1
    POLYGON = 137
    ARBITRUM = 42161
    OPTIMISM = 10
    BSC = 56
    AVALANCHE = 43114
    BASE = 8453
    # Testnets
    GOERLI = 5
    SEPOLIA = 11155111


@dataclass(frozen=True)
class Token:
    """
    ERC-20 token representation.

    Immutable to ensure consistency across the system.
    """

    address: str
    symbol: str
    decimals: int
    chain_id: int
    name: str = ""

    def __post_init__(self) -> None:
        """Validate token address."""
        # Normalize address to checksum format
        if not self.address.startswith("0x"):
            object.__setattr__(self, "address", f"0x{self.address}")

    @property
    def is_native(self) -> bool:
        """Check if this is the native token (ETH, MATIC, etc.)."""
        return self.address.lower() == "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

    def to_wei(self, amount: Decimal) -> int:
        """Convert token amount to wei (smallest unit)."""
        return int(amount * Decimal(10 ** self.decimals))

    def from_wei(self, wei_amount: int) -> Decimal:
        """Convert wei amount to token amount."""
        return Decimal(wei_amount) / Decimal(10 ** self.decimals)


# Common token definitions
WETH = Token(
    address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    symbol="WETH",
    decimals=18,
    chain_id=ChainId.ETHEREUM,
    name="Wrapped Ether",
)

USDC = Token(
    address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    symbol="USDC",
    decimals=6,
    chain_id=ChainId.ETHEREUM,
    name="USD Coin",
)

USDT = Token(
    address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
    symbol="USDT",
    decimals=6,
    chain_id=ChainId.ETHEREUM,
    name="Tether USD",
)


@dataclass
class DEXPool:
    """
    Liquidity pool information.

    Represents a trading pair on a DEX.
    """

    address: str
    token0: Token
    token1: Token
    reserve0: Decimal
    reserve1: Decimal
    fee: Decimal  # Fee as decimal (e.g., 0.003 for 0.3%)
    dex_type: DEXType = DEXType.AMM_V2

    # V3-specific fields
    tick: int | None = None
    liquidity: int | None = None
    sqrt_price_x96: int | None = None

    @property
    def price_0_to_1(self) -> Decimal:
        """Price of token0 in terms of token1."""
        if self.reserve0 == 0:
            return Decimal("0")
        return self.reserve1 / self.reserve0

    @property
    def price_1_to_0(self) -> Decimal:
        """Price of token1 in terms of token0."""
        if self.reserve1 == 0:
            return Decimal("0")
        return self.reserve0 / self.reserve1

    @property
    def total_liquidity_usd(self) -> Decimal | None:
        """Total liquidity in USD (requires price oracle)."""
        # This would need external price data
        return None


@dataclass
class DEXQuote:
    """
    Quote for a potential swap.

    Contains expected output, price impact, and route information.
    """

    input_token: Token
    output_token: Token
    input_amount: Decimal
    output_amount: Decimal
    price_impact: float  # As percentage (e.g., 0.5 for 0.5%)
    route: list[str] = field(default_factory=list)  # Pool addresses in route
    gas_estimate: int = 0
    timestamp_ns: int = 0

    @property
    def execution_price(self) -> Decimal:
        """Effective execution price."""
        if self.input_amount == 0:
            return Decimal("0")
        return self.output_amount / self.input_amount

    @property
    def is_valid(self) -> bool:
        """Check if quote is valid for execution."""
        return self.output_amount > 0 and self.price_impact < 50.0  # Max 50% slippage


@dataclass
class DEXSwapResult:
    """
    Result of a swap execution.

    Contains transaction details and actual amounts.
    """

    success: bool
    tx_hash: str | None
    input_token: Token
    output_token: Token
    input_amount: Decimal
    output_amount: Decimal
    gas_used: int = 0
    gas_price: int = 0  # In wei
    block_number: int = 0
    error_message: str = ""

    @property
    def gas_cost_wei(self) -> int:
        """Total gas cost in wei."""
        return self.gas_used * self.gas_price

    @property
    def slippage(self) -> float:
        """Actual slippage compared to quote (requires quote)."""
        return 0.0  # Would need to compare with original quote


@dataclass
class DEXGatewayConfig:
    """Configuration for DEX gateway."""

    chain_id: int = ChainId.ETHEREUM
    rpc_url: str = ""
    private_key: str = ""  # For signing transactions
    max_slippage: float = 0.5  # Maximum slippage percentage
    gas_limit: int = 500000
    gas_price_multiplier: float = 1.1  # Multiply estimated gas price

    # Wallet settings
    wallet_address: str = ""

    # DEX-specific
    router_address: str = ""
    factory_address: str = ""

    def validate(self) -> None:
        """Validate configuration."""
        if not self.rpc_url:
            raise ValueError("rpc_url is required")
        if self.max_slippage < 0 or self.max_slippage > 100:
            raise ValueError("max_slippage must be between 0 and 100")


class DEXGateway(ABC):
    """
    Abstract base class for DEX gateways.

    Provides interface for:
    - Getting quotes
    - Executing swaps
    - Fetching pool information
    - Monitoring prices
    """

    def __init__(self, config: DEXGatewayConfig) -> None:
        """Initialize DEX gateway."""
        self.config = config
        self._connected = False
        self._web3 = None

    @property
    @abstractmethod
    def name(self) -> str:
        """DEX name (e.g., 'uniswap_v2', 'sushiswap')."""
        ...

    @property
    @abstractmethod
    def dex_type(self) -> DEXType:
        """Type of DEX (AMM_V2, AMM_V3, ORDER_BOOK)."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to blockchain."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Connect to blockchain RPC."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from blockchain."""
        ...

    @abstractmethod
    async def get_pool(self, token0: Token, token1: Token) -> DEXPool | None:
        """
        Get pool information for a token pair.

        Args:
            token0: First token
            token1: Second token

        Returns:
            Pool information or None if pool doesn't exist
        """
        ...

    @abstractmethod
    async def get_quote(
        self,
        input_token: Token,
        output_token: Token,
        input_amount: Decimal,
    ) -> DEXQuote | None:
        """
        Get quote for a swap.

        Args:
            input_token: Token to swap from
            output_token: Token to swap to
            input_amount: Amount of input token

        Returns:
            Quote with expected output or None if route not found
        """
        ...

    @abstractmethod
    async def execute_swap(
        self,
        quote: DEXQuote,
        min_output: Decimal | None = None,
    ) -> DEXSwapResult:
        """
        Execute a swap based on quote.

        Args:
            quote: Quote to execute
            min_output: Minimum output amount (slippage protection)

        Returns:
            Swap result with transaction details
        """
        ...

    async def get_token_balance(self, token: Token, address: str | None = None) -> Decimal:
        """
        Get token balance for an address.

        Args:
            token: Token to check
            address: Address to check (defaults to wallet address)

        Returns:
            Token balance
        """
        # Default implementation - subclasses should override
        return Decimal("0")

    async def approve_token(
        self,
        token: Token,
        spender: str,
        amount: Decimal | None = None,
    ) -> str | None:
        """
        Approve token spending.

        Args:
            token: Token to approve
            spender: Address to approve
            amount: Amount to approve (None for unlimited)

        Returns:
            Transaction hash or None if already approved
        """
        # Default implementation - subclasses should override
        return None

    def calculate_price_impact(
        self,
        pool: DEXPool,
        input_amount: Decimal,
        is_token0_input: bool,
    ) -> float:
        """
        Calculate price impact for a swap.

        Uses constant product formula: (x + dx)(y - dy) = xy

        Args:
            pool: Pool to swap in
            input_amount: Amount to swap
            is_token0_input: True if swapping token0 for token1

        Returns:
            Price impact as percentage
        """
        if pool.dex_type != DEXType.AMM_V2:
            # V3 calculation is more complex
            return 0.0

        if is_token0_input:
            reserve_in = pool.reserve0
            reserve_out = pool.reserve1
        else:
            reserve_in = pool.reserve1
            reserve_out = pool.reserve0

        if reserve_in == 0 or reserve_out == 0:
            return 100.0

        # Constant product: k = x * y
        k = reserve_in * reserve_out

        # After swap: (x + dx) * (y - dy) = k
        new_reserve_in = reserve_in + input_amount
        new_reserve_out = k / new_reserve_in

        output_amount = reserve_out - new_reserve_out

        # Price before
        price_before = reserve_out / reserve_in

        # Price after (marginal)
        price_after = new_reserve_out / new_reserve_in

        # Price impact
        impact = float((price_before - price_after) / price_before * 100)

        return abs(impact)

    def calculate_output_amount(
        self,
        pool: DEXPool,
        input_amount: Decimal,
        is_token0_input: bool,
    ) -> Decimal:
        """
        Calculate output amount for a swap (V2 AMM).

        Uses formula: dy = y * dx / (x + dx) * (1 - fee)

        Args:
            pool: Pool to swap in
            input_amount: Amount to swap
            is_token0_input: True if swapping token0 for token1

        Returns:
            Expected output amount
        """
        if is_token0_input:
            reserve_in = pool.reserve0
            reserve_out = pool.reserve1
        else:
            reserve_in = pool.reserve1
            reserve_out = pool.reserve0

        if reserve_in == 0:
            return Decimal("0")

        # Apply fee
        input_with_fee = input_amount * (1 - pool.fee)

        # Constant product formula
        numerator = input_with_fee * reserve_out
        denominator = reserve_in + input_with_fee

        return numerator / denominator

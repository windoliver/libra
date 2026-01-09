"""
Uniswap V2/V3 Gateway Implementation (Issue #12).

Provides DEX integration for:
- Uniswap V2 (constant product AMM)
- Uniswap V3 (concentrated liquidity AMM)

Note: Requires web3 optional dependency for actual blockchain interaction.
This module provides simulation/calculation without web3 for backtesting.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.plugins.hummingbot_adapter.dex.base import (
    DEXGateway,
    DEXGatewayConfig,
    DEXPool,
    DEXQuote,
    DEXSwapResult,
    DEXType,
    Token,
)


if TYPE_CHECKING:
    pass


# Uniswap V2 Router and Factory addresses (Ethereum mainnet)
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
UNISWAP_V2_FACTORY = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"

# Uniswap V3 addresses (Ethereum mainnet)
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
UNISWAP_V3_QUOTER = "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"


class UniswapV2Gateway(DEXGateway):
    """
    Uniswap V2 Gateway.

    Implements constant product AMM (x * y = k) for:
    - Quote calculation
    - Swap execution
    - Pool information

    Can operate in simulation mode without web3 for backtesting.
    """

    def __init__(self, config: DEXGatewayConfig) -> None:
        """Initialize Uniswap V2 gateway."""
        super().__init__(config)

        # Set default addresses if not provided
        if not config.router_address:
            self.config.router_address = UNISWAP_V2_ROUTER
        if not config.factory_address:
            self.config.factory_address = UNISWAP_V2_FACTORY

        # Pool cache for simulation
        self._pool_cache: dict[str, DEXPool] = {}

        # Simulation mode (no web3 required)
        self._simulation_mode = True

    @property
    def name(self) -> str:
        """DEX name."""
        return "uniswap_v2"

    @property
    def dex_type(self) -> DEXType:
        """DEX type."""
        return DEXType.AMM_V2

    async def connect(self) -> None:
        """Connect to blockchain RPC."""
        if self.config.rpc_url:
            try:
                # Optional web3 import
                from web3 import Web3

                self._web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
                self._connected = self._web3.is_connected()
                self._simulation_mode = False
            except ImportError:
                # Run in simulation mode without web3
                self._connected = True
                self._simulation_mode = True
        else:
            # Simulation mode
            self._connected = True
            self._simulation_mode = True

    async def disconnect(self) -> None:
        """Disconnect from blockchain."""
        self._connected = False
        self._web3 = None

    def add_simulated_pool(self, pool: DEXPool) -> None:
        """
        Add a pool for simulation/backtesting.

        Args:
            pool: Pool to add to simulation
        """
        key = self._pool_key(pool.token0, pool.token1)
        self._pool_cache[key] = pool

    def _pool_key(self, token0: Token, token1: Token) -> str:
        """Generate cache key for token pair."""
        # Sort by address to ensure consistent key
        if token0.address.lower() < token1.address.lower():
            return f"{token0.address.lower()}_{token1.address.lower()}"
        return f"{token1.address.lower()}_{token0.address.lower()}"

    async def get_pool(self, token0: Token, token1: Token) -> DEXPool | None:
        """
        Get pool information for a token pair.

        Args:
            token0: First token
            token1: Second token

        Returns:
            Pool information or None if pool doesn't exist
        """
        key = self._pool_key(token0, token1)

        # Check cache first
        if key in self._pool_cache:
            return self._pool_cache[key]

        if self._simulation_mode:
            return None

        # Fetch from blockchain (requires web3)
        try:
            pool = await self._fetch_pool_from_chain(token0, token1)
            if pool:
                self._pool_cache[key] = pool
            return pool
        except Exception:
            return None

    async def _fetch_pool_from_chain(
        self, token0: Token, token1: Token
    ) -> DEXPool | None:
        """Fetch pool data from blockchain."""
        if not self._web3:
            return None

        # Factory ABI for getPair
        factory_abi = [
            {
                "constant": True,
                "inputs": [
                    {"name": "tokenA", "type": "address"},
                    {"name": "tokenB", "type": "address"},
                ],
                "name": "getPair",
                "outputs": [{"name": "pair", "type": "address"}],
                "type": "function",
            }
        ]

        # Pair ABI for getReserves
        pair_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "getReserves",
                "outputs": [
                    {"name": "reserve0", "type": "uint112"},
                    {"name": "reserve1", "type": "uint112"},
                    {"name": "blockTimestampLast", "type": "uint32"},
                ],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [],
                "name": "token0",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [],
                "name": "token1",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function",
            },
        ]

        factory = self._web3.eth.contract(
            address=self._web3.to_checksum_address(self.config.factory_address),
            abi=factory_abi,
        )

        # Get pair address
        pair_address = factory.functions.getPair(
            self._web3.to_checksum_address(token0.address),
            self._web3.to_checksum_address(token1.address),
        ).call()

        # Check if pair exists
        if pair_address == "0x0000000000000000000000000000000000000000":
            return None

        pair = self._web3.eth.contract(
            address=self._web3.to_checksum_address(pair_address),
            abi=pair_abi,
        )

        # Get reserves
        reserves = pair.functions.getReserves().call()

        # Get token order in pair
        pair_token0 = pair.functions.token0().call()

        # Determine correct token order
        if pair_token0.lower() == token0.address.lower():
            reserve0 = Decimal(reserves[0]) / Decimal(10 ** token0.decimals)
            reserve1 = Decimal(reserves[1]) / Decimal(10 ** token1.decimals)
            t0, t1 = token0, token1
        else:
            reserve0 = Decimal(reserves[1]) / Decimal(10 ** token0.decimals)
            reserve1 = Decimal(reserves[0]) / Decimal(10 ** token1.decimals)
            t0, t1 = token1, token0

        return DEXPool(
            address=pair_address,
            token0=t0,
            token1=t1,
            reserve0=reserve0,
            reserve1=reserve1,
            fee=Decimal("0.003"),  # Uniswap V2 standard 0.3% fee
            dex_type=DEXType.AMM_V2,
        )

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
        pool = await self.get_pool(input_token, output_token)
        if not pool:
            return None

        # Determine token order
        is_token0_input = input_token.address.lower() == pool.token0.address.lower()

        # Calculate output using constant product formula
        output_amount = self.calculate_output_amount(pool, input_amount, is_token0_input)

        # Calculate price impact
        price_impact = self.calculate_price_impact(pool, input_amount, is_token0_input)

        # Estimate gas (typical V2 swap)
        gas_estimate = 150000

        return DEXQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=input_amount,
            output_amount=output_amount,
            price_impact=price_impact,
            route=[pool.address],
            gas_estimate=gas_estimate,
            timestamp_ns=time.time_ns(),
        )

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
        if self._simulation_mode:
            return await self._simulate_swap(quote, min_output)

        # Real execution requires web3
        if not self._web3 or not self.config.private_key:
            return DEXSwapResult(
                success=False,
                tx_hash=None,
                input_token=quote.input_token,
                output_token=quote.output_token,
                input_amount=quote.input_amount,
                output_amount=Decimal("0"),
                error_message="Web3 not connected or private key not provided",
            )

        try:
            return await self._execute_real_swap(quote, min_output)
        except Exception as e:
            return DEXSwapResult(
                success=False,
                tx_hash=None,
                input_token=quote.input_token,
                output_token=quote.output_token,
                input_amount=quote.input_amount,
                output_amount=Decimal("0"),
                error_message=str(e),
            )

    async def _simulate_swap(
        self,
        quote: DEXQuote,
        min_output: Decimal | None = None,
    ) -> DEXSwapResult:
        """Simulate a swap for backtesting."""
        # Check slippage
        actual_min = min_output or quote.output_amount * (
            1 - Decimal(str(self.config.max_slippage / 100))
        )

        if quote.output_amount < actual_min:
            return DEXSwapResult(
                success=False,
                tx_hash=None,
                input_token=quote.input_token,
                output_token=quote.output_token,
                input_amount=quote.input_amount,
                output_amount=Decimal("0"),
                error_message="Slippage exceeded",
            )

        # Update pool reserves for simulation
        pool = await self.get_pool(quote.input_token, quote.output_token)
        if pool:
            key = self._pool_key(quote.input_token, quote.output_token)
            is_token0_input = (
                quote.input_token.address.lower() == pool.token0.address.lower()
            )

            if is_token0_input:
                new_reserve0 = pool.reserve0 + quote.input_amount
                new_reserve1 = pool.reserve1 - quote.output_amount
            else:
                new_reserve0 = pool.reserve0 - quote.output_amount
                new_reserve1 = pool.reserve1 + quote.input_amount

            self._pool_cache[key] = DEXPool(
                address=pool.address,
                token0=pool.token0,
                token1=pool.token1,
                reserve0=new_reserve0,
                reserve1=new_reserve1,
                fee=pool.fee,
                dex_type=pool.dex_type,
            )

        return DEXSwapResult(
            success=True,
            tx_hash="0x" + "0" * 64,  # Simulated tx hash
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            gas_used=quote.gas_estimate,
            gas_price=50_000_000_000,  # 50 gwei simulated
            block_number=0,
        )

    async def _execute_real_swap(
        self,
        quote: DEXQuote,
        min_output: Decimal | None = None,
    ) -> DEXSwapResult:
        """Execute real swap on blockchain."""
        if not self._web3:
            raise RuntimeError("Web3 not connected")

        # Router ABI for swapExactTokensForTokens
        router_abi = [
            {
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMin", "type": "uint256"},
                    {"name": "path", "type": "address[]"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"},
                ],
                "name": "swapExactTokensForTokens",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "type": "function",
            }
        ]

        router = self._web3.eth.contract(
            address=self._web3.to_checksum_address(self.config.router_address),
            abi=router_abi,
        )

        # Calculate amounts
        amount_in = quote.input_token.to_wei(quote.input_amount)
        min_out = min_output or quote.output_amount * (
            1 - Decimal(str(self.config.max_slippage / 100))
        )
        amount_out_min = quote.output_token.to_wei(min_out)

        # Build path
        path = [
            self._web3.to_checksum_address(quote.input_token.address),
            self._web3.to_checksum_address(quote.output_token.address),
        ]

        # Deadline (30 minutes from now)
        deadline = int(time.time()) + 1800

        # Build transaction
        account = self._web3.eth.account.from_key(self.config.private_key)

        tx = router.functions.swapExactTokensForTokens(
            amount_in,
            amount_out_min,
            path,
            account.address,
            deadline,
        ).build_transaction(
            {
                "from": account.address,
                "gas": self.config.gas_limit,
                "gasPrice": int(
                    self._web3.eth.gas_price * self.config.gas_price_multiplier
                ),
                "nonce": self._web3.eth.get_transaction_count(account.address),
            }
        )

        # Sign and send
        signed_tx = self._web3.eth.account.sign_transaction(
            tx, self.config.private_key
        )
        tx_hash = self._web3.eth.send_raw_transaction(signed_tx.raw_transaction)

        # Wait for receipt
        receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        return DEXSwapResult(
            success=receipt["status"] == 1,
            tx_hash=tx_hash.hex(),
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,  # Actual amount from logs
            gas_used=receipt["gasUsed"],
            gas_price=tx["gasPrice"],
            block_number=receipt["blockNumber"],
        )


class UniswapV3Gateway(DEXGateway):
    """
    Uniswap V3 Gateway.

    Implements concentrated liquidity AMM for:
    - Quote calculation via Quoter contract
    - Swap execution via SwapRouter
    - Pool information with tick data

    Supports multiple fee tiers: 0.01%, 0.05%, 0.3%, 1%
    """

    # Fee tiers in basis points
    FEE_TIERS = [100, 500, 3000, 10000]  # 0.01%, 0.05%, 0.3%, 1%

    def __init__(self, config: DEXGatewayConfig) -> None:
        """Initialize Uniswap V3 gateway."""
        super().__init__(config)

        if not config.router_address:
            self.config.router_address = UNISWAP_V3_ROUTER
        if not config.factory_address:
            self.config.factory_address = UNISWAP_V3_FACTORY

        self._quoter_address = UNISWAP_V3_QUOTER
        self._pool_cache: dict[str, DEXPool] = {}
        self._simulation_mode = True

    @property
    def name(self) -> str:
        """DEX name."""
        return "uniswap_v3"

    @property
    def dex_type(self) -> DEXType:
        """DEX type."""
        return DEXType.AMM_V3

    async def connect(self) -> None:
        """Connect to blockchain RPC."""
        if self.config.rpc_url:
            try:
                from web3 import Web3

                self._web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
                self._connected = self._web3.is_connected()
                self._simulation_mode = False
            except ImportError:
                self._connected = True
                self._simulation_mode = True
        else:
            self._connected = True
            self._simulation_mode = True

    async def disconnect(self) -> None:
        """Disconnect from blockchain."""
        self._connected = False
        self._web3 = None

    def add_simulated_pool(
        self,
        pool: DEXPool,
    ) -> None:
        """Add a pool for simulation/backtesting."""
        key = self._pool_key(pool.token0, pool.token1, pool.fee)
        self._pool_cache[key] = pool

    def _pool_key(self, token0: Token, token1: Token, fee: Decimal) -> str:
        """Generate cache key for token pair with fee tier."""
        if token0.address.lower() < token1.address.lower():
            return f"{token0.address.lower()}_{token1.address.lower()}_{fee}"
        return f"{token1.address.lower()}_{token0.address.lower()}_{fee}"

    async def get_pool(
        self,
        token0: Token,
        token1: Token,
        fee_tier: int = 3000,  # Default 0.3%
    ) -> DEXPool | None:
        """
        Get pool information for a token pair.

        Args:
            token0: First token
            token1: Second token
            fee_tier: Fee tier in basis points (100, 500, 3000, 10000)

        Returns:
            Pool information or None if pool doesn't exist
        """
        fee = Decimal(fee_tier) / Decimal(1_000_000)
        key = self._pool_key(token0, token1, fee)

        if key in self._pool_cache:
            return self._pool_cache[key]

        if self._simulation_mode:
            return None

        try:
            pool = await self._fetch_pool_from_chain(token0, token1, fee_tier)
            if pool:
                self._pool_cache[key] = pool
            return pool
        except Exception:
            return None

    async def _fetch_pool_from_chain(
        self,
        token0: Token,
        token1: Token,
        fee_tier: int,
    ) -> DEXPool | None:
        """Fetch V3 pool data from blockchain."""
        if not self._web3:
            return None

        # Factory ABI for getPool
        factory_abi = [
            {
                "inputs": [
                    {"name": "tokenA", "type": "address"},
                    {"name": "tokenB", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                ],
                "name": "getPool",
                "outputs": [{"name": "pool", "type": "address"}],
                "type": "function",
            }
        ]

        # Pool ABI for slot0 and liquidity
        pool_abi = [
            {
                "inputs": [],
                "name": "slot0",
                "outputs": [
                    {"name": "sqrtPriceX96", "type": "uint160"},
                    {"name": "tick", "type": "int24"},
                    {"name": "observationIndex", "type": "uint16"},
                    {"name": "observationCardinality", "type": "uint16"},
                    {"name": "observationCardinalityNext", "type": "uint16"},
                    {"name": "feeProtocol", "type": "uint8"},
                    {"name": "unlocked", "type": "bool"},
                ],
                "type": "function",
            },
            {
                "inputs": [],
                "name": "liquidity",
                "outputs": [{"name": "", "type": "uint128"}],
                "type": "function",
            },
        ]

        factory = self._web3.eth.contract(
            address=self._web3.to_checksum_address(self.config.factory_address),
            abi=factory_abi,
        )

        pool_address = factory.functions.getPool(
            self._web3.to_checksum_address(token0.address),
            self._web3.to_checksum_address(token1.address),
            fee_tier,
        ).call()

        if pool_address == "0x0000000000000000000000000000000000000000":
            return None

        pool_contract = self._web3.eth.contract(
            address=self._web3.to_checksum_address(pool_address),
            abi=pool_abi,
        )

        slot0 = pool_contract.functions.slot0().call()
        liquidity = pool_contract.functions.liquidity().call()

        sqrt_price_x96 = slot0[0]
        tick = slot0[1]

        # Calculate price from sqrtPriceX96
        # price = (sqrtPriceX96 / 2^96)^2
        price = (Decimal(sqrt_price_x96) / Decimal(2**96)) ** 2

        # Adjust for decimals
        decimal_adjustment = Decimal(10 ** (token0.decimals - token1.decimals))
        adjusted_price = price * decimal_adjustment

        # For V3, reserves are virtual based on liquidity and price
        # This is a simplification - actual reserves depend on tick range
        reserve0 = Decimal(liquidity) / Decimal(10**token0.decimals)
        reserve1 = reserve0 * adjusted_price

        return DEXPool(
            address=pool_address,
            token0=token0,
            token1=token1,
            reserve0=reserve0,
            reserve1=reserve1,
            fee=Decimal(fee_tier) / Decimal(1_000_000),
            dex_type=DEXType.AMM_V3,
            tick=tick,
            liquidity=liquidity,
            sqrt_price_x96=sqrt_price_x96,
        )

    async def get_quote(
        self,
        input_token: Token,
        output_token: Token,
        input_amount: Decimal,
        fee_tier: int = 3000,
    ) -> DEXQuote | None:
        """
        Get quote for a V3 swap.

        Args:
            input_token: Token to swap from
            output_token: Token to swap to
            input_amount: Amount of input token
            fee_tier: Fee tier in basis points

        Returns:
            Quote with expected output or None if route not found
        """
        pool = await self.get_pool(input_token, output_token, fee_tier)
        if not pool:
            return None

        if self._simulation_mode:
            # Simplified V3 calculation (uses V2-style for simulation)
            is_token0_input = (
                input_token.address.lower() == pool.token0.address.lower()
            )
            output_amount = self.calculate_output_amount(
                pool, input_amount, is_token0_input
            )
            price_impact = self.calculate_price_impact(
                pool, input_amount, is_token0_input
            )
        else:
            # Use Quoter contract for accurate V3 quote
            output_amount, price_impact = await self._get_quoter_quote(
                input_token, output_token, input_amount, fee_tier
            )

        return DEXQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=input_amount,
            output_amount=output_amount,
            price_impact=price_impact,
            route=[pool.address],
            gas_estimate=180000,  # V3 swaps use more gas
            timestamp_ns=time.time_ns(),
        )

    async def _get_quoter_quote(
        self,
        input_token: Token,
        output_token: Token,
        input_amount: Decimal,
        fee_tier: int,
    ) -> tuple[Decimal, float]:
        """Get quote from Quoter contract."""
        if not self._web3:
            return Decimal("0"), 0.0

        quoter_abi = [
            {
                "inputs": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
                "name": "quoteExactInputSingle",
                "outputs": [{"name": "amountOut", "type": "uint256"}],
                "type": "function",
            }
        ]

        quoter = self._web3.eth.contract(
            address=self._web3.to_checksum_address(self._quoter_address),
            abi=quoter_abi,
        )

        amount_in_wei = input_token.to_wei(input_amount)

        try:
            amount_out = quoter.functions.quoteExactInputSingle(
                self._web3.to_checksum_address(input_token.address),
                self._web3.to_checksum_address(output_token.address),
                fee_tier,
                amount_in_wei,
                0,  # No price limit
            ).call()

            output_amount = output_token.from_wei(amount_out)

            # Calculate price impact from market price
            pool = await self.get_pool(input_token, output_token, fee_tier)
            if pool:
                is_token0_input = (
                    input_token.address.lower() == pool.token0.address.lower()
                )
                market_price = pool.price_0_to_1 if is_token0_input else pool.price_1_to_0
                if market_price > 0:
                    exec_price = output_amount / input_amount
                    price_impact = float(
                        abs(market_price - exec_price) / market_price * 100
                    )
                else:
                    price_impact = 0.0
            else:
                price_impact = 0.0

            return output_amount, price_impact
        except Exception:
            return Decimal("0"), 100.0

    async def execute_swap(
        self,
        quote: DEXQuote,
        min_output: Decimal | None = None,
    ) -> DEXSwapResult:
        """Execute a V3 swap."""
        if self._simulation_mode:
            return await self._simulate_swap(quote, min_output)

        if not self._web3 or not self.config.private_key:
            return DEXSwapResult(
                success=False,
                tx_hash=None,
                input_token=quote.input_token,
                output_token=quote.output_token,
                input_amount=quote.input_amount,
                output_amount=Decimal("0"),
                error_message="Web3 not connected or private key not provided",
            )

        try:
            return await self._execute_real_swap(quote, min_output)
        except Exception as e:
            return DEXSwapResult(
                success=False,
                tx_hash=None,
                input_token=quote.input_token,
                output_token=quote.output_token,
                input_amount=quote.input_amount,
                output_amount=Decimal("0"),
                error_message=str(e),
            )

    async def _simulate_swap(
        self,
        quote: DEXQuote,
        min_output: Decimal | None = None,
    ) -> DEXSwapResult:
        """Simulate a V3 swap for backtesting."""
        actual_min = min_output or quote.output_amount * (
            1 - Decimal(str(self.config.max_slippage / 100))
        )

        if quote.output_amount < actual_min:
            return DEXSwapResult(
                success=False,
                tx_hash=None,
                input_token=quote.input_token,
                output_token=quote.output_token,
                input_amount=quote.input_amount,
                output_amount=Decimal("0"),
                error_message="Slippage exceeded",
            )

        return DEXSwapResult(
            success=True,
            tx_hash="0x" + "0" * 64,
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            gas_used=quote.gas_estimate,
            gas_price=50_000_000_000,
            block_number=0,
        )

    async def _execute_real_swap(
        self,
        quote: DEXQuote,
        min_output: Decimal | None = None,
        fee_tier: int = 3000,
    ) -> DEXSwapResult:
        """Execute real V3 swap on blockchain."""
        if not self._web3:
            raise RuntimeError("Web3 not connected")

        # SwapRouter ABI for exactInputSingle
        router_abi = [
            {
                "inputs": [
                    {
                        "components": [
                            {"name": "tokenIn", "type": "address"},
                            {"name": "tokenOut", "type": "address"},
                            {"name": "fee", "type": "uint24"},
                            {"name": "recipient", "type": "address"},
                            {"name": "deadline", "type": "uint256"},
                            {"name": "amountIn", "type": "uint256"},
                            {"name": "amountOutMinimum", "type": "uint256"},
                            {"name": "sqrtPriceLimitX96", "type": "uint160"},
                        ],
                        "name": "params",
                        "type": "tuple",
                    }
                ],
                "name": "exactInputSingle",
                "outputs": [{"name": "amountOut", "type": "uint256"}],
                "type": "function",
            }
        ]

        router = self._web3.eth.contract(
            address=self._web3.to_checksum_address(self.config.router_address),
            abi=router_abi,
        )

        amount_in = quote.input_token.to_wei(quote.input_amount)
        min_out = min_output or quote.output_amount * (
            1 - Decimal(str(self.config.max_slippage / 100))
        )
        amount_out_min = quote.output_token.to_wei(min_out)

        deadline = int(time.time()) + 1800
        account = self._web3.eth.account.from_key(self.config.private_key)

        params = (
            self._web3.to_checksum_address(quote.input_token.address),
            self._web3.to_checksum_address(quote.output_token.address),
            fee_tier,
            account.address,
            deadline,
            amount_in,
            amount_out_min,
            0,  # No price limit
        )

        tx = router.functions.exactInputSingle(params).build_transaction(
            {
                "from": account.address,
                "gas": self.config.gas_limit,
                "gasPrice": int(
                    self._web3.eth.gas_price * self.config.gas_price_multiplier
                ),
                "nonce": self._web3.eth.get_transaction_count(account.address),
            }
        )

        signed_tx = self._web3.eth.account.sign_transaction(
            tx, self.config.private_key
        )
        tx_hash = self._web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        return DEXSwapResult(
            success=receipt["status"] == 1,
            tx_hash=tx_hash.hex(),
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            gas_used=receipt["gasUsed"],
            gas_price=tx["gasPrice"],
            block_number=receipt["blockNumber"],
        )

    async def find_best_fee_tier(
        self,
        input_token: Token,
        output_token: Token,
        input_amount: Decimal,
    ) -> tuple[int, DEXQuote] | None:
        """
        Find the best fee tier for a swap.

        Args:
            input_token: Token to swap from
            output_token: Token to swap to
            input_amount: Amount to swap

        Returns:
            Tuple of (fee_tier, quote) or None if no pool found
        """
        best_quote: DEXQuote | None = None
        best_fee_tier = 0

        for fee_tier in self.FEE_TIERS:
            quote = await self.get_quote(
                input_token, output_token, input_amount, fee_tier
            )
            if quote and quote.output_amount > 0:
                if best_quote is None or quote.output_amount > best_quote.output_amount:
                    best_quote = quote
                    best_fee_tier = fee_tier

        if best_quote:
            return best_fee_tier, best_quote
        return None

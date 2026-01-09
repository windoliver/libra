"""Tests for DEX Gateway Support (Issue #12)."""

from decimal import Decimal

import pytest

from libra.plugins.hummingbot_adapter.dex.base import (
    ChainId,
    DEXGatewayConfig,
    DEXPool,
    DEXQuote,
    DEXSwapResult,
    DEXType,
    Token,
    USDC,
    WETH,
)
from libra.plugins.hummingbot_adapter.dex.uniswap import (
    UniswapV2Gateway,
    UniswapV3Gateway,
)
from libra.plugins.hummingbot_adapter.dex.arbitrage import (
    ArbitrageConfig,
    ArbitrageOpportunity,
    ArbitrageType,
    DEXArbitrageStrategy,
)


class TestToken:
    """Tests for Token dataclass."""

    def test_token_creation(self) -> None:
        """Test basic token creation."""
        token = Token(
            address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            symbol="WETH",
            decimals=18,
            chain_id=ChainId.ETHEREUM,
            name="Wrapped Ether",
        )

        assert token.symbol == "WETH"
        assert token.decimals == 18
        assert token.chain_id == ChainId.ETHEREUM

    def test_token_address_normalization(self) -> None:
        """Test address normalization adds 0x prefix."""
        token = Token(
            address="C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            symbol="WETH",
            decimals=18,
            chain_id=ChainId.ETHEREUM,
        )

        assert token.address.startswith("0x")

    def test_token_is_native(self) -> None:
        """Test native token detection."""
        native = Token(
            address="0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
            symbol="ETH",
            decimals=18,
            chain_id=ChainId.ETHEREUM,
        )
        assert native.is_native

        assert not WETH.is_native

    def test_token_to_wei(self) -> None:
        """Test conversion to wei."""
        # WETH has 18 decimals
        wei = WETH.to_wei(Decimal("1.5"))
        assert wei == 1_500_000_000_000_000_000

        # USDC has 6 decimals
        usdc_wei = USDC.to_wei(Decimal("100"))
        assert usdc_wei == 100_000_000

    def test_token_from_wei(self) -> None:
        """Test conversion from wei."""
        amount = WETH.from_wei(1_500_000_000_000_000_000)
        assert amount == Decimal("1.5")

        usdc_amount = USDC.from_wei(100_000_000)
        assert usdc_amount == Decimal("100")

    def test_predefined_tokens(self) -> None:
        """Test predefined token constants."""
        assert WETH.symbol == "WETH"
        assert WETH.decimals == 18

        assert USDC.symbol == "USDC"
        assert USDC.decimals == 6


class TestDEXPool:
    """Tests for DEXPool dataclass."""

    def test_pool_creation(self) -> None:
        """Test pool creation."""
        pool = DEXPool(
            address="0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("1000"),
            reserve1=Decimal("2000000"),
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        )

        assert pool.fee == Decimal("0.003")
        assert pool.dex_type == DEXType.AMM_V2

    def test_pool_price_calculations(self) -> None:
        """Test pool price calculations."""
        pool = DEXPool(
            address="0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("1000"),  # 1000 WETH
            reserve1=Decimal("2000000"),  # 2M USDC
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        )

        # Price of WETH in USDC = 2000
        assert pool.price_0_to_1 == Decimal("2000")

        # Price of USDC in WETH = 0.0005
        assert pool.price_1_to_0 == Decimal("0.0005")

    def test_pool_zero_reserve_handling(self) -> None:
        """Test price calculation with zero reserves."""
        pool = DEXPool(
            address="0x0000000000000000000000000000000000000000",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("0"),
            reserve1=Decimal("100"),
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        )

        assert pool.price_0_to_1 == Decimal("0")


class TestDEXQuote:
    """Tests for DEXQuote dataclass."""

    def test_quote_creation(self) -> None:
        """Test quote creation."""
        quote = DEXQuote(
            input_token=WETH,
            output_token=USDC,
            input_amount=Decimal("1"),
            output_amount=Decimal("1990"),  # Accounting for fees/slippage
            price_impact=0.5,
            route=["0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852"],
            gas_estimate=150000,
        )

        assert quote.execution_price == Decimal("1990")
        assert quote.is_valid

    def test_quote_validity_check(self) -> None:
        """Test quote validity for extreme price impact."""
        bad_quote = DEXQuote(
            input_token=WETH,
            output_token=USDC,
            input_amount=Decimal("1"),
            output_amount=Decimal("100"),
            price_impact=75.0,  # 75% slippage
            route=[],
        )

        assert not bad_quote.is_valid

    def test_quote_zero_output(self) -> None:
        """Test quote with zero output is invalid."""
        quote = DEXQuote(
            input_token=WETH,
            output_token=USDC,
            input_amount=Decimal("1"),
            output_amount=Decimal("0"),
            price_impact=0.0,
            route=[],
        )

        assert not quote.is_valid


class TestUniswapV2Gateway:
    """Tests for UniswapV2Gateway."""

    @pytest.fixture
    def config(self) -> DEXGatewayConfig:
        """Create test config."""
        return DEXGatewayConfig(
            chain_id=ChainId.ETHEREUM,
            max_slippage=0.5,
        )

    @pytest.fixture
    def gateway(self, config: DEXGatewayConfig) -> UniswapV2Gateway:
        """Create gateway instance."""
        return UniswapV2Gateway(config)

    @pytest.fixture
    def pool(self) -> DEXPool:
        """Create test pool."""
        return DEXPool(
            address="0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("10000"),  # 10k WETH
            reserve1=Decimal("20000000"),  # 20M USDC
            fee=Decimal("0.003"),
            dex_type=DEXType.AMM_V2,
        )

    def test_gateway_properties(self, gateway: UniswapV2Gateway) -> None:
        """Test gateway property methods."""
        assert gateway.name == "uniswap_v2"
        assert gateway.dex_type == DEXType.AMM_V2
        assert not gateway.is_connected

    async def test_gateway_connect_simulation(
        self, gateway: UniswapV2Gateway
    ) -> None:
        """Test connecting in simulation mode."""
        await gateway.connect()
        assert gateway.is_connected
        assert gateway._simulation_mode

    async def test_add_simulated_pool(
        self, gateway: UniswapV2Gateway, pool: DEXPool
    ) -> None:
        """Test adding simulated pool."""
        await gateway.connect()
        gateway.add_simulated_pool(pool)

        fetched = await gateway.get_pool(WETH, USDC)
        assert fetched is not None
        assert fetched.address == pool.address

    async def test_get_quote_simulation(
        self, gateway: UniswapV2Gateway, pool: DEXPool
    ) -> None:
        """Test getting quote in simulation mode."""
        await gateway.connect()
        gateway.add_simulated_pool(pool)

        quote = await gateway.get_quote(WETH, USDC, Decimal("1"))

        assert quote is not None
        assert quote.input_amount == Decimal("1")
        assert quote.output_amount > 0
        assert quote.price_impact >= 0
        assert len(quote.route) == 1

    async def test_execute_swap_simulation(
        self, gateway: UniswapV2Gateway, pool: DEXPool
    ) -> None:
        """Test executing swap in simulation mode."""
        await gateway.connect()
        gateway.add_simulated_pool(pool)

        quote = await gateway.get_quote(WETH, USDC, Decimal("1"))
        assert quote is not None

        result = await gateway.execute_swap(quote)

        assert result.success
        assert result.tx_hash is not None
        assert result.output_amount > 0

    async def test_slippage_protection(
        self, gateway: UniswapV2Gateway, pool: DEXPool
    ) -> None:
        """Test slippage protection rejects excessive slippage."""
        await gateway.connect()
        gateway.add_simulated_pool(pool)

        quote = await gateway.get_quote(WETH, USDC, Decimal("1"))
        assert quote is not None

        # Request unreasonably high min output
        result = await gateway.execute_swap(
            quote, min_output=quote.output_amount * Decimal("2")
        )

        assert not result.success
        assert "Slippage" in result.error_message

    def test_calculate_output_amount(
        self, gateway: UniswapV2Gateway, pool: DEXPool
    ) -> None:
        """Test output amount calculation."""
        output = gateway.calculate_output_amount(pool, Decimal("1"), is_token0_input=True)

        # Should get approximately 1990-2000 USDC for 1 WETH (with 0.3% fee)
        assert output > Decimal("1990")
        assert output < Decimal("2000")

    def test_calculate_price_impact(
        self, gateway: UniswapV2Gateway, pool: DEXPool
    ) -> None:
        """Test price impact calculation."""
        # Small trade - low impact
        small_impact = gateway.calculate_price_impact(
            pool, Decimal("1"), is_token0_input=True
        )
        assert small_impact < 0.1  # Less than 0.1%

        # Large trade - higher impact
        large_impact = gateway.calculate_price_impact(
            pool, Decimal("1000"), is_token0_input=True
        )
        assert large_impact > small_impact


class TestUniswapV3Gateway:
    """Tests for UniswapV3Gateway."""

    @pytest.fixture
    def config(self) -> DEXGatewayConfig:
        """Create test config."""
        return DEXGatewayConfig(
            chain_id=ChainId.ETHEREUM,
            max_slippage=0.5,
        )

    @pytest.fixture
    def gateway(self, config: DEXGatewayConfig) -> UniswapV3Gateway:
        """Create gateway instance."""
        return UniswapV3Gateway(config)

    @pytest.fixture
    def pool(self) -> DEXPool:
        """Create test V3 pool."""
        return DEXPool(
            address="0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",
            token0=WETH,
            token1=USDC,
            reserve0=Decimal("10000"),
            reserve1=Decimal("20000000"),
            fee=Decimal("0.003"),  # 0.3% fee tier
            dex_type=DEXType.AMM_V3,
            tick=-201234,
            liquidity=1000000000000,
        )

    def test_gateway_properties(self, gateway: UniswapV3Gateway) -> None:
        """Test gateway properties."""
        assert gateway.name == "uniswap_v3"
        assert gateway.dex_type == DEXType.AMM_V3
        assert gateway.FEE_TIERS == [100, 500, 3000, 10000]

    async def test_gateway_connect_simulation(
        self, gateway: UniswapV3Gateway
    ) -> None:
        """Test connecting in simulation mode."""
        await gateway.connect()
        assert gateway.is_connected
        assert gateway._simulation_mode

    async def test_add_simulated_pool_with_fee(
        self, gateway: UniswapV3Gateway, pool: DEXPool
    ) -> None:
        """Test adding simulated pool with specific fee tier."""
        await gateway.connect()
        gateway.add_simulated_pool(pool)

        fetched = await gateway.get_pool(WETH, USDC, 3000)
        assert fetched is not None

    async def test_get_quote_simulation(
        self, gateway: UniswapV3Gateway, pool: DEXPool
    ) -> None:
        """Test getting quote in simulation mode."""
        await gateway.connect()
        gateway.add_simulated_pool(pool)

        quote = await gateway.get_quote(WETH, USDC, Decimal("1"), fee_tier=3000)

        assert quote is not None
        assert quote.output_amount > 0
        assert quote.gas_estimate > 0


class TestDEXArbitrageStrategy:
    """Tests for DEX arbitrage detection and execution."""

    @pytest.fixture
    def gateways(self) -> list[UniswapV2Gateway]:
        """Create multiple gateways for arbitrage testing."""
        config = DEXGatewayConfig(chain_id=ChainId.ETHEREUM)

        # Create two "different" DEXs (using V2 gateway with different configs)
        gw1 = UniswapV2Gateway(config)
        gw2 = UniswapV2Gateway(config)

        # Override names to simulate different DEXs
        object.__setattr__(gw1, "_name", "dex1")
        object.__setattr__(gw2, "_name", "dex2")

        return [gw1, gw2]

    @pytest.fixture
    def config(self) -> ArbitrageConfig:
        """Create arbitrage config."""
        return ArbitrageConfig(
            min_profit_percentage=0.1,
            min_profit_amount=Decimal("1"),
            max_price_impact=5.0,
        )

    @pytest.fixture
    def strategy(
        self, gateways: list[UniswapV2Gateway], config: ArbitrageConfig
    ) -> DEXArbitrageStrategy:
        """Create arbitrage strategy."""
        return DEXArbitrageStrategy(gateways, config)

    def test_arbitrage_opportunity_dataclass(self) -> None:
        """Test ArbitrageOpportunity properties."""
        opportunity = ArbitrageOpportunity(
            arb_type=ArbitrageType.CROSS_DEX,
            token_path=[WETH, USDC, WETH],
            dex_path=["dex1", "dex2"],
            input_amount=Decimal("1"),
            expected_output=Decimal("1.005"),
            expected_profit=Decimal("0.005"),
            profit_percentage=0.5,
            gas_estimate=300000,
            quotes=[],
        )

        assert opportunity.is_profitable
        assert opportunity.arb_type == ArbitrageType.CROSS_DEX

    def test_arbitrage_config_defaults(self) -> None:
        """Test ArbitrageConfig default values."""
        config = ArbitrageConfig()

        assert config.min_profit_percentage == 0.1
        assert config.max_position_size == Decimal("10000")
        assert config.max_price_impact == 2.0

    async def test_strategy_stats(
        self, strategy: DEXArbitrageStrategy
    ) -> None:
        """Test strategy statistics tracking."""
        stats = strategy.stats

        assert stats["opportunities_found"] == 0
        assert stats["opportunities_executed"] == 0
        assert stats["total_profit"] == 0.0

    async def test_strategy_reset_stats(
        self, strategy: DEXArbitrageStrategy
    ) -> None:
        """Test resetting strategy statistics."""
        strategy._opportunities_found = 5
        strategy._total_profit = Decimal("100")

        strategy.reset_stats()

        assert strategy._opportunities_found == 0
        assert strategy._total_profit == Decimal("0")


class TestDEXSwapResult:
    """Tests for DEXSwapResult dataclass."""

    def test_successful_swap_result(self) -> None:
        """Test successful swap result."""
        result = DEXSwapResult(
            success=True,
            tx_hash="0x" + "a" * 64,
            input_token=WETH,
            output_token=USDC,
            input_amount=Decimal("1"),
            output_amount=Decimal("1990"),
            gas_used=150000,
            gas_price=50_000_000_000,
            block_number=12345678,
        )

        assert result.success
        assert result.gas_cost_wei == 150000 * 50_000_000_000

    def test_failed_swap_result(self) -> None:
        """Test failed swap result."""
        result = DEXSwapResult(
            success=False,
            tx_hash=None,
            input_token=WETH,
            output_token=USDC,
            input_amount=Decimal("1"),
            output_amount=Decimal("0"),
            error_message="Transaction reverted",
        )

        assert not result.success
        assert result.error_message == "Transaction reverted"

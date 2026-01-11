"""
Polymarket Provider.

Provider for Polymarket - a crypto-native prediction market on Polygon.

API Documentation: https://docs.polymarket.com/

Features:
- CLOB (Central Limit Order Book) trading via py-clob-client
- USDC settlement on Polygon
- Conditional Token Framework (CTF)
- Real-time WebSocket streams
- L1 (Private Key) and L2 (API Key) authentication
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.gateways.prediction_market.protocol import (
    MarketStatus,
    MarketType,
    Outcome,
    OutcomeType,
    POLYMARKET_CAPABILITIES,
    PredictionMarket,
    PredictionMarketCapabilities,
    PredictionOrder,
    PredictionOrderBook,
    PredictionOrderBookLevel,
    PredictionOrderResult,
    PredictionOrderSide,
    PredictionOrderStatus,
    PredictionOrderType,
    PredictionPosition,
    PredictionQuote,
)
from libra.gateways.prediction_market.providers.base import (
    BasePredictionProvider,
    ProviderConfig,
)


if TYPE_CHECKING:
    from py_clob_client.client import ClobClient


logger = logging.getLogger(__name__)


# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"

# Polygon chain ID
POLYGON_CHAIN_ID = 137

# Contract addresses for allowances
POLYMARKET_CONTRACTS = {
    "usdc": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # USDC.e on Polygon
    "ctf": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",  # Conditional Tokens
    "exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",  # Exchange
    "neg_risk_exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",  # Neg Risk Exchange
    "neg_risk_adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",  # Neg Risk Adapter
}


class PolymarketTradingError(Exception):
    """Error during Polymarket trading operations."""


class PolymarketAuthError(Exception):
    """Error during Polymarket authentication."""


class PolymarketProvider(BasePredictionProvider):
    """
    Polymarket prediction market provider.

    Supports:
    - Market discovery via Gamma API
    - Order book and trading via CLOB API (py-clob-client)
    - Positions and trades via Data API

    Example:
        # Read-only (no credentials needed)
        provider = PolymarketProvider()
        await provider.connect()
        markets = await provider.get_markets(category="crypto")

        # Trading (requires private key)
        config = ProviderConfig(
            private_key="your_polygon_private_key",
        )
        provider = PolymarketProvider(config)
        await provider.connect()

        # Check if trading is ready
        if provider.is_trading_ready:
            quote = await provider.get_quote(market_id, "yes")
            order = PredictionOrder(...)
            result = await provider.submit_order(order)
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """
        Initialize Polymarket provider.

        Args:
            config: Provider configuration with optional private_key for trading
        """
        super().__init__(config)
        self._clob_client: ClobClient | None = None
        self._trading_ready = False
        self._wallet_address: str | None = None

    @property
    def name(self) -> str:
        return "polymarket"

    @property
    def base_url(self) -> str:
        return GAMMA_API_URL

    @property
    def capabilities(self) -> PredictionMarketCapabilities:
        return POLYMARKET_CAPABILITIES

    @property
    def is_trading_ready(self) -> bool:
        """Check if trading is enabled and authenticated."""
        return self._trading_ready and self._clob_client is not None

    @property
    def wallet_address(self) -> str | None:
        """Get the wallet address (if trading is configured)."""
        return self._wallet_address

    @property
    def clob_client(self) -> ClobClient | None:
        """Get the underlying CLOB client for advanced operations."""
        return self._clob_client

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """
        Connect to Polymarket.

        Sets up:
        - HTTP client for Gamma API (market data)
        - CLOB client for trading (if private key provided)
        """
        # Connect HTTP client for market data (from base class)
        await super().connect()

        # Initialize CLOB client for trading if private key is provided
        if self._config.private_key:
            await self._init_clob_client()

    async def disconnect(self) -> None:
        """Disconnect from Polymarket."""
        self._clob_client = None
        self._trading_ready = False
        self._wallet_address = None
        await super().disconnect()

    async def _init_clob_client(self) -> None:
        """Initialize the py-clob-client for trading."""
        try:
            from py_clob_client.client import ClobClient
        except ImportError as e:
            logger.warning(
                "py-clob-client not installed. Trading disabled. "
                "Install with: pip install py-clob-client"
            )
            raise ImportError(
                "py-clob-client is required for Polymarket trading. "
                "Install with: pip install py-clob-client"
            ) from e

        try:
            # Determine signature type
            # 0 = EOA (direct private key)
            # 1 = Poly Proxy (email/Magic wallet)
            # 2 = Gnosis Safe
            signature_type = self._config.extra.get("signature_type", 0)
            funder = self._config.extra.get("funder_address")

            # Create client with credentials
            self._clob_client = ClobClient(
                CLOB_API_URL,
                key=self._config.private_key,
                chain_id=POLYGON_CHAIN_ID,
                signature_type=signature_type,
                funder=funder,
            )

            # Derive or create API credentials (L2 auth)
            # This is done in a thread since py-clob-client is synchronous
            loop = asyncio.get_event_loop()
            api_creds = await loop.run_in_executor(
                None, self._clob_client.create_or_derive_api_creds
            )
            self._clob_client.set_api_creds(api_creds)

            # Get wallet address
            from eth_account import Account
            account = Account.from_key(self._config.private_key)
            self._wallet_address = account.address

            self._trading_ready = True
            logger.info(
                f"Polymarket CLOB client initialized for wallet {self._wallet_address[:10]}..."
            )

        except Exception as e:
            logger.error(f"Failed to initialize CLOB client: {e}")
            self._clob_client = None
            self._trading_ready = False
            raise PolymarketAuthError(f"Failed to authenticate with Polymarket: {e}") from e

    # =========================================================================
    # Market Data (Gamma API)
    # =========================================================================

    async def get_markets(
        self,
        category: str | None = None,
        status: MarketStatus | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PredictionMarket]:
        """Get markets from Polymarket."""
        self._ensure_connected()

        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "offset": offset,
        }

        if status == MarketStatus.OPEN:
            params["active"] = True
            params["closed"] = False
        elif status == MarketStatus.CLOSED:
            params["active"] = False
            params["closed"] = True

        if search:
            params["_q"] = search

        try:
            response = await self._get("/markets", params=params)
            markets = []
            for item in response:
                market = self._parse_market(item)
                if market:
                    if category and market.category and category.lower() not in market.category.lower():
                        continue
                    markets.append(market)
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch Polymarket markets: {e}")
            return []

    async def get_market(self, market_id: str) -> PredictionMarket | None:
        """Get a specific market by ID."""
        self._ensure_connected()

        try:
            response = await self._get(f"/markets/{market_id}")
            if isinstance(response, list) and response:
                return self._parse_market(response[0])
            elif isinstance(response, dict):
                return self._parse_market(response)
            return None
        except Exception as e:
            logger.warning(f"Market {market_id} not found: {e}")
            return None

    async def get_quote(
        self, market_id: str, outcome_id: str
    ) -> PredictionQuote | None:
        """Get quote for a market outcome."""
        self._ensure_connected()

        try:
            orderbook = await self.get_orderbook(market_id, outcome_id)
            if not orderbook:
                return None

            bid = orderbook.best_bid or Decimal("0")
            ask = orderbook.best_ask or Decimal("1")
            mid = orderbook.mid or ((bid + ask) / 2)

            return PredictionQuote(
                market_id=market_id,
                outcome_id=outcome_id,
                platform=self.name,
                bid=bid,
                ask=ask,
                mid=mid,
                timestamp_ns=time.time_ns(),
                bid_size=orderbook.bids[0].size if orderbook.bids else None,
                ask_size=orderbook.asks[0].size if orderbook.asks else None,
            )
        except Exception as e:
            logger.warning(f"Failed to get quote for {market_id}/{outcome_id}: {e}")
            return None

    async def get_orderbook(
        self, market_id: str, outcome_id: str, depth: int = 20
    ) -> PredictionOrderBook | None:
        """Get order book from CLOB API."""
        self._ensure_connected()

        try:
            token_id = outcome_id

            # Use CLOB API for orderbook (via httpx, not py-clob-client)
            params = {"token_id": token_id}
            response = await self._clob_request("GET", "/book", params=params)

            bids = []
            asks = []

            for bid in response.get("bids", [])[:depth]:
                bids.append(
                    PredictionOrderBookLevel(
                        price=Decimal(str(bid.get("price", 0))),
                        size=Decimal(str(bid.get("size", 0))),
                    )
                )

            for ask in response.get("asks", [])[:depth]:
                asks.append(
                    PredictionOrderBookLevel(
                        price=Decimal(str(ask.get("price", 0))),
                        size=Decimal(str(ask.get("size", 0))),
                    )
                )

            return PredictionOrderBook(
                market_id=market_id,
                outcome_id=outcome_id,
                platform=self.name,
                bids=tuple(bids),
                asks=tuple(asks),
                timestamp_ns=time.time_ns(),
            )
        except Exception as e:
            logger.warning(f"Failed to get orderbook for {market_id}/{outcome_id}: {e}")
            return None

    # =========================================================================
    # Trading (py-clob-client)
    # =========================================================================

    async def get_positions(
        self, market_id: str | None = None
    ) -> list[PredictionPosition]:
        """Get user positions."""
        if not self.is_trading_ready:
            logger.warning("Trading not enabled. Provide private_key to enable.")
            return []

        try:
            loop = asyncio.get_event_loop()
            positions_data = await loop.run_in_executor(
                None, self._clob_client.get_positions  # type: ignore
            )

            positions = []
            for item in positions_data or []:
                if market_id and item.get("market_id") != market_id:
                    continue

                positions.append(
                    PredictionPosition(
                        market_id=item.get("market_id", ""),
                        outcome_id=item.get("asset", item.get("token_id", "")),
                        platform=self.name,
                        size=Decimal(str(item.get("size", 0))),
                        avg_price=Decimal(str(item.get("avg_price", 0))),
                        current_price=Decimal(str(item.get("current_price", 0))),
                        unrealized_pnl=Decimal(str(item.get("unrealized_pnl", 0))),
                        realized_pnl=Decimal(str(item.get("realized_pnl", 0))),
                        timestamp_ns=time.time_ns(),
                    )
                )
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    async def submit_order(self, order: PredictionOrder) -> PredictionOrderResult:
        """
        Submit order to Polymarket via py-clob-client.

        Args:
            order: Order to submit

        Returns:
            Order result with status and fill information

        Raises:
            PolymarketTradingError: If order submission fails
            PolymarketAuthError: If trading is not enabled
        """
        if not self.is_trading_ready:
            raise PolymarketAuthError(
                "Trading not enabled. Provide private_key in config to enable trading."
            )

        try:
            from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            side = BUY if order.side == PredictionOrderSide.BUY else SELL

            loop = asyncio.get_event_loop()

            # Determine order type and create appropriate order
            if order.order_type == PredictionOrderType.MARKET:
                # Market order - specify amount in dollars
                order_args = MarketOrderArgs(
                    token_id=order.outcome_id,
                    amount=float(order.size * (order.price or Decimal("0.5"))),
                    side=side,
                )
                signed_order = await loop.run_in_executor(
                    None,
                    self._clob_client.create_market_order,  # type: ignore
                    order_args,
                )
                clob_order_type = OrderType.FOK
            else:
                # Limit order - specify size in shares and price
                if order.price is None:
                    raise PolymarketTradingError("Price required for limit orders")

                order_args = OrderArgs(
                    token_id=order.outcome_id,
                    price=float(order.price),
                    size=float(order.size),
                    side=side,
                )
                signed_order = await loop.run_in_executor(
                    None,
                    self._clob_client.create_order,  # type: ignore
                    order_args,
                )
                # Map order type
                if order.order_type == PredictionOrderType.IOC:
                    clob_order_type = OrderType.IOC
                elif order.order_type == PredictionOrderType.FOK:
                    clob_order_type = OrderType.FOK
                else:
                    clob_order_type = OrderType.GTC

            # Post the order
            response = await loop.run_in_executor(
                None,
                lambda: self._clob_client.post_order(signed_order, clob_order_type),  # type: ignore
            )

            return PredictionOrderResult(
                order_id=response.get("orderID", response.get("id", "")),
                market_id=order.market_id,
                outcome_id=order.outcome_id,
                platform=self.name,
                status=self._parse_order_status(response.get("status", "pending")),
                side=order.side,
                size=order.size,
                filled_size=Decimal(str(response.get("filledSize", 0))),
                timestamp_ns=time.time_ns(),
                average_price=Decimal(str(response.get("averagePrice", 0)))
                if response.get("averagePrice")
                else None,
                price=order.price,
            )

        except ImportError as e:
            raise PolymarketTradingError(
                "py-clob-client not installed. Install with: pip install py-clob-client"
            ) from e
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise PolymarketTradingError(f"Order submission failed: {e}") from e

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.is_trading_ready:
            raise PolymarketAuthError("Trading not enabled")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._clob_client.cancel,  # type: ignore
                order_id,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if not self.is_trading_ready:
            raise PolymarketAuthError("Trading not enabled")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._clob_client.cancel_all,  # type: ignore
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    async def get_open_orders(
        self, market_id: str | None = None
    ) -> list[PredictionOrderResult]:
        """Get open orders."""
        if not self.is_trading_ready:
            return []

        try:
            from py_clob_client.clob_types import OpenOrderParams

            loop = asyncio.get_event_loop()
            params = OpenOrderParams(market=market_id) if market_id else OpenOrderParams()
            orders_data = await loop.run_in_executor(
                None,
                lambda: self._clob_client.get_orders(params),  # type: ignore
            )

            orders = []
            for item in orders_data or []:
                orders.append(
                    PredictionOrderResult(
                        order_id=item.get("id", ""),
                        market_id=item.get("market", ""),
                        outcome_id=item.get("asset_id", item.get("token_id", "")),
                        platform=self.name,
                        status=self._parse_order_status(item.get("status", "open")),
                        side=PredictionOrderSide.BUY if item.get("side") == "BUY" else PredictionOrderSide.SELL,
                        size=Decimal(str(item.get("original_size", item.get("size", 0)))),
                        filled_size=Decimal(str(item.get("size_matched", 0))),
                        price=Decimal(str(item.get("price", 0))),
                        timestamp_ns=time.time_ns(),
                    )
                )
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    async def get_balance(self) -> dict[str, Decimal]:
        """Get USDC balance on Polygon."""
        if not self.is_trading_ready or not self._wallet_address:
            return {}

        try:
            from web3 import Web3

            # Connect to Polygon RPC
            rpc_url = self._config.extra.get(
                "polygon_rpc_url",
                "https://polygon-rpc.com"
            )
            w3 = Web3(Web3.HTTPProvider(rpc_url))

            # USDC.e ABI (just balanceOf)
            usdc_abi = [
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function",
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function",
                },
            ]

            usdc_contract = w3.eth.contract(
                address=Web3.to_checksum_address(POLYMARKET_CONTRACTS["usdc"]),
                abi=usdc_abi,
            )

            loop = asyncio.get_event_loop()
            balance_wei = await loop.run_in_executor(
                None,
                lambda: usdc_contract.functions.balanceOf(self._wallet_address).call(),
            )

            # USDC has 6 decimals
            balance = Decimal(str(balance_wei)) / Decimal("1000000")

            return {"USDC": balance}

        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    # =========================================================================
    # Allowance Management
    # =========================================================================

    async def check_allowances(self) -> dict[str, bool]:
        """
        Check if token allowances are set for trading.

        Returns:
            Dict with keys 'usdc' and 'ctf' indicating if allowances are sufficient
        """
        if not self._wallet_address:
            return {"usdc": False, "ctf": False}

        try:
            from web3 import Web3

            rpc_url = self._config.extra.get("polygon_rpc_url", "https://polygon-rpc.com")
            w3 = Web3(Web3.HTTPProvider(rpc_url))

            allowance_abi = [
                {
                    "constant": True,
                    "inputs": [
                        {"name": "_owner", "type": "address"},
                        {"name": "_spender", "type": "address"},
                    ],
                    "name": "allowance",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "type": "function",
                },
            ]

            # Check USDC allowance for exchange contracts
            usdc_contract = w3.eth.contract(
                address=Web3.to_checksum_address(POLYMARKET_CONTRACTS["usdc"]),
                abi=allowance_abi,
            )

            ctf_contract = w3.eth.contract(
                address=Web3.to_checksum_address(POLYMARKET_CONTRACTS["ctf"]),
                abi=allowance_abi,
            )

            loop = asyncio.get_event_loop()

            # Check allowance for main exchange
            usdc_allowance = await loop.run_in_executor(
                None,
                lambda: usdc_contract.functions.allowance(
                    self._wallet_address,
                    Web3.to_checksum_address(POLYMARKET_CONTRACTS["exchange"]),
                ).call(),
            )

            ctf_allowance = await loop.run_in_executor(
                None,
                lambda: ctf_contract.functions.allowance(
                    self._wallet_address,
                    Web3.to_checksum_address(POLYMARKET_CONTRACTS["exchange"]),
                ).call(),
            )

            # Consider allowance sufficient if > 1M (arbitrary threshold)
            min_allowance = 10**24  # Large number

            return {
                "usdc": usdc_allowance >= min_allowance,
                "ctf": ctf_allowance >= min_allowance,
            }

        except Exception as e:
            logger.error(f"Failed to check allowances: {e}")
            return {"usdc": False, "ctf": False}

    async def set_allowances(self) -> dict[str, str]:
        """
        Set token allowances for trading (approve max).

        Returns:
            Dict mapping token to transaction hash

        Raises:
            PolymarketAuthError: If private key not configured
        """
        if not self._config.private_key or not self._wallet_address:
            raise PolymarketAuthError("Private key required to set allowances")

        try:
            from web3 import Web3
            from eth_account import Account

            rpc_url = self._config.extra.get("polygon_rpc_url", "https://polygon-rpc.com")
            w3 = Web3(Web3.HTTPProvider(rpc_url))

            approve_abi = [
                {
                    "constant": False,
                    "inputs": [
                        {"name": "_spender", "type": "address"},
                        {"name": "_value", "type": "uint256"},
                    ],
                    "name": "approve",
                    "outputs": [{"name": "", "type": "bool"}],
                    "type": "function",
                },
            ]

            account = Account.from_key(self._config.private_key)
            max_approval = 2**256 - 1

            results = {}

            # Approve USDC for all exchange contracts
            usdc_contract = w3.eth.contract(
                address=Web3.to_checksum_address(POLYMARKET_CONTRACTS["usdc"]),
                abi=approve_abi,
            )

            for contract_name in ["exchange", "neg_risk_exchange", "neg_risk_adapter"]:
                spender = POLYMARKET_CONTRACTS[contract_name]

                loop = asyncio.get_event_loop()
                nonce = await loop.run_in_executor(
                    None,
                    lambda: w3.eth.get_transaction_count(account.address),
                )

                tx = usdc_contract.functions.approve(
                    Web3.to_checksum_address(spender),
                    max_approval,
                ).build_transaction({
                    "from": account.address,
                    "nonce": nonce,
                    "gas": 100000,
                    "gasPrice": w3.eth.gas_price,
                    "chainId": POLYGON_CHAIN_ID,
                })

                signed_tx = account.sign_transaction(tx)
                tx_hash = await loop.run_in_executor(
                    None,
                    lambda: w3.eth.send_raw_transaction(signed_tx.raw_transaction),
                )
                results[f"usdc_{contract_name}"] = tx_hash.hex()

            # Approve CTF for exchange contracts
            ctf_contract = w3.eth.contract(
                address=Web3.to_checksum_address(POLYMARKET_CONTRACTS["ctf"]),
                abi=approve_abi,
            )

            for contract_name in ["exchange", "neg_risk_exchange", "neg_risk_adapter"]:
                spender = POLYMARKET_CONTRACTS[contract_name]

                loop = asyncio.get_event_loop()
                nonce = await loop.run_in_executor(
                    None,
                    lambda: w3.eth.get_transaction_count(account.address),
                )

                tx = ctf_contract.functions.approve(
                    Web3.to_checksum_address(spender),
                    max_approval,
                ).build_transaction({
                    "from": account.address,
                    "nonce": nonce,
                    "gas": 100000,
                    "gasPrice": w3.eth.gas_price,
                    "chainId": POLYGON_CHAIN_ID,
                })

                signed_tx = account.sign_transaction(tx)
                tx_hash = await loop.run_in_executor(
                    None,
                    lambda: w3.eth.send_raw_transaction(signed_tx.raw_transaction),
                )
                results[f"ctf_{contract_name}"] = tx_hash.hex()

            logger.info(f"Set allowances: {list(results.keys())}")
            return results

        except Exception as e:
            logger.error(f"Failed to set allowances: {e}")
            raise PolymarketTradingError(f"Failed to set allowances: {e}") from e

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def get_midpoint(self, token_id: str) -> Decimal | None:
        """Get midpoint price for a token."""
        if not self.is_trading_ready:
            # Fall back to HTTP request
            try:
                response = await self._clob_request("GET", f"/midpoint", params={"token_id": token_id})
                return Decimal(str(response.get("mid", 0)))
            except Exception:
                return None

        try:
            loop = asyncio.get_event_loop()
            mid = await loop.run_in_executor(
                None,
                lambda: self._clob_client.get_midpoint(token_id),  # type: ignore
            )
            return Decimal(str(mid)) if mid else None
        except Exception as e:
            logger.warning(f"Failed to get midpoint: {e}")
            return None

    async def get_spread(self, token_id: str) -> dict[str, Decimal] | None:
        """Get bid/ask spread for a token."""
        if not self.is_trading_ready:
            return None

        try:
            loop = asyncio.get_event_loop()
            spread = await loop.run_in_executor(
                None,
                lambda: self._clob_client.get_spread(token_id),  # type: ignore
            )
            if spread:
                return {
                    "bid": Decimal(str(spread.get("bid", 0))),
                    "ask": Decimal(str(spread.get("ask", 0))),
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to get spread: {e}")
            return None

    async def get_last_trade_price(self, token_id: str) -> Decimal | None:
        """Get last trade price for a token."""
        if not self.is_trading_ready:
            return None

        try:
            loop = asyncio.get_event_loop()
            price = await loop.run_in_executor(
                None,
                lambda: self._clob_client.get_last_trade_price(token_id),  # type: ignore
            )
            return Decimal(str(price)) if price else None
        except Exception as e:
            logger.warning(f"Failed to get last trade price: {e}")
            return None

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _parse_market(self, data: dict[str, Any]) -> PredictionMarket | None:
        """Parse market data from Gamma API response."""
        try:
            outcomes = []
            tokens = data.get("tokens", [])

            if tokens:
                for token in tokens:
                    outcome = Outcome(
                        outcome_id=token.get("token_id", ""),
                        name=token.get("outcome", ""),
                        probability=Decimal(str(token.get("price", 0))),
                        price=Decimal(str(token.get("price", 0))),
                        volume=Decimal(str(token.get("volume", 0))),
                        token_id=token.get("token_id"),
                        winner=token.get("winner"),
                    )
                    outcomes.append(outcome)
            else:
                yes_price = Decimal(str(data.get("outcomePrices", ["0.5", "0.5"])[0]))
                outcomes = [
                    Outcome(
                        outcome_id="yes",
                        name="Yes",
                        probability=yes_price,
                        price=yes_price,
                        volume=Decimal(str(data.get("volume", 0))) / 2,
                    ),
                    Outcome(
                        outcome_id="no",
                        name="No",
                        probability=Decimal("1") - yes_price,
                        price=Decimal("1") - yes_price,
                        volume=Decimal(str(data.get("volume", 0))) / 2,
                    ),
                ]

            status = MarketStatus.OPEN
            if data.get("closed"):
                status = MarketStatus.CLOSED
            if data.get("resolved"):
                status = MarketStatus.RESOLVED

            end_date_iso = data.get("endDateIso")
            close_date = None
            if end_date_iso:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
                    close_date = int(dt.timestamp() * 1_000_000_000)
                except Exception:
                    pass

            return PredictionMarket(
                market_id=data.get("condition_id") or data.get("id", ""),
                platform=self.name,
                title=data.get("question", data.get("title", "")),
                description=data.get("description"),
                category=data.get("category"),
                tags=tuple(data.get("tags", [])),
                slug=data.get("slug"),
                url=f"https://polymarket.com/event/{data.get('slug', '')}",
                outcomes=tuple(outcomes),
                status=status,
                outcome_type=OutcomeType.BINARY if len(outcomes) == 2 else OutcomeType.MULTIPLE,
                market_type=MarketType.CLOB,
                volume=Decimal(str(data.get("volume", 0))),
                volume_24h=Decimal(str(data.get("volume24hr", 0))),
                liquidity=Decimal(str(data.get("liquidity", 0))),
                close_date=close_date,
                created_at=None,
                updated_at=time.time_ns(),
            )
        except Exception as e:
            logger.error(f"Failed to parse market: {e}")
            return None

    def _parse_order_status(self, status: str) -> PredictionOrderStatus:
        """Parse order status string."""
        status_map = {
            "pending": PredictionOrderStatus.PENDING,
            "open": PredictionOrderStatus.OPEN,
            "live": PredictionOrderStatus.OPEN,
            "filled": PredictionOrderStatus.FILLED,
            "matched": PredictionOrderStatus.FILLED,
            "partially_filled": PredictionOrderStatus.PARTIALLY_FILLED,
            "cancelled": PredictionOrderStatus.CANCELLED,
            "canceled": PredictionOrderStatus.CANCELLED,
            "expired": PredictionOrderStatus.EXPIRED,
        }
        return status_map.get(status.lower(), PredictionOrderStatus.PENDING)

    async def _clob_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make request to CLOB API."""
        if not self._client:
            raise RuntimeError("Provider not connected")

        url = f"{CLOB_API_URL}{path}"
        response = await self._client.request(
            method=method,
            url=url,
            params=params,
            json=json,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def _data_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make request to Data API."""
        if not self._client:
            raise RuntimeError("Provider not connected")

        url = f"{DATA_API_URL}{path}"
        response = await self._client.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def _verify_connection(self) -> None:
        """Verify connection by fetching one market."""
        await self.get_markets(limit=1)

"""
Polymarket Provider.

Provider for Polymarket - a crypto-native prediction market on Polygon.

API Documentation: https://docs.polymarket.com/

Features:
- CLOB (Central Limit Order Book) trading
- USDC settlement on Polygon
- Conditional Token Framework (CTF)
- Real-time WebSocket streams
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from decimal import Decimal
from typing import Any

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
    PredictionPosition,
    PredictionQuote,
)
from libra.gateways.prediction_market.providers.base import (
    BasePredictionProvider,
    ProviderConfig,
)


logger = logging.getLogger(__name__)


# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"


class PolymarketProvider(BasePredictionProvider):
    """
    Polymarket prediction market provider.

    Supports:
    - Market discovery via Gamma API
    - Order book and trading via CLOB API
    - Positions and trades via Data API

    Example:
        config = ProviderConfig(
            api_key="your_api_key",
            api_secret="your_api_secret",
            private_key="your_polygon_private_key",
        )
        provider = PolymarketProvider(config)
        await provider.connect()

        markets = await provider.get_markets(category="crypto")
        quote = await provider.get_quote(market_id, "yes")
    """

    @property
    def name(self) -> str:
        return "polymarket"

    @property
    def base_url(self) -> str:
        return GAMMA_API_URL

    @property
    def capabilities(self) -> PredictionMarketCapabilities:
        return POLYMARKET_CAPABILITIES

    # =========================================================================
    # Market Data
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
            # Use search endpoint
            params["_q"] = search

        try:
            response = await self._get("/markets", params=params)
            markets = []
            for item in response:
                market = self._parse_market(item)
                if market:
                    # Filter by category if specified
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
            # Try condition_id first
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
            # Get orderbook from CLOB API
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
            # The outcome_id should be the token_id for CLOB
            token_id = outcome_id

            # Use CLOB API for orderbook
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
    # Trading (Phase 3)
    # =========================================================================

    async def get_positions(
        self, market_id: str | None = None
    ) -> list[PredictionPosition]:
        """Get user positions from Data API."""
        self._ensure_connected()

        if not self._config.api_key:
            logger.warning("API key required for positions")
            return []

        try:
            headers = self._get_auth_headers()
            response = await self._data_request("GET", "/positions", headers=headers)

            positions = []
            for item in response:
                if market_id and item.get("market_id") != market_id:
                    continue

                positions.append(
                    PredictionPosition(
                        market_id=item.get("market_id", ""),
                        outcome_id=item.get("outcome_id", ""),
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
        """Submit order to CLOB API."""
        self._ensure_connected()

        if not self._config.api_key or not self._config.api_secret:
            raise ValueError("API key and secret required for trading")

        try:
            headers = self._get_auth_headers()
            order_data = {
                "token_id": order.outcome_id,
                "side": "BUY" if order.side == PredictionOrderSide.BUY else "SELL",
                "size": str(order.size),
                "type": order.order_type.value.upper(),
            }

            if order.price is not None:
                order_data["price"] = str(order.price)

            response = await self._clob_request(
                "POST", "/order", json=order_data, headers=headers
            )

            return PredictionOrderResult(
                order_id=response.get("order_id", ""),
                market_id=order.market_id,
                outcome_id=order.outcome_id,
                platform=self.name,
                status=self._parse_order_status(response.get("status", "pending")),
                side=order.side,
                size=order.size,
                filled_size=Decimal(str(response.get("filled_size", 0))),
                timestamp_ns=time.time_ns(),
                average_price=Decimal(str(response.get("average_price", 0)))
                if response.get("average_price")
                else None,
                price=order.price,
            )
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        self._ensure_connected()

        if not self._config.api_key:
            raise ValueError("API key required for trading")

        try:
            headers = self._get_auth_headers()
            await self._clob_request(
                "DELETE", f"/order/{order_id}", headers=headers
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_balance(self) -> dict[str, Decimal]:
        """Get USDC balance."""
        self._ensure_connected()

        if not self._config.api_key:
            return {}

        try:
            headers = self._get_auth_headers()
            response = await self._data_request("GET", "/balance", headers=headers)
            return {
                "USDC": Decimal(str(response.get("balance", 0))),
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _parse_market(self, data: dict[str, Any]) -> PredictionMarket | None:
        """Parse market data from Gamma API response."""
        try:
            # Parse outcomes
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
                # Binary market without explicit tokens
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

            # Parse status
            status = MarketStatus.OPEN
            if data.get("closed"):
                status = MarketStatus.CLOSED
            if data.get("resolved"):
                status = MarketStatus.RESOLVED

            # Parse timestamps
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

    def _get_auth_headers(self) -> dict[str, str]:
        """Generate authentication headers for API requests."""
        if not self._config.api_key or not self._config.api_secret:
            return {}

        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}GET/auth"
        signature = hmac.new(
            self._config.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return {
            "POLY_ADDRESS": self._config.api_key,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": timestamp,
        }

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

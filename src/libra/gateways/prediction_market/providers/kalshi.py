"""
Kalshi Provider.

Provider for Kalshi - a CFTC-regulated prediction market.

API Documentation: https://docs.kalshi.com/

Features:
- Regulated event contracts
- USD settlement
- Binary and ranged outcomes
- REST and WebSocket APIs
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

from libra.gateways.prediction_market.protocol import (
    KALSHI_CAPABILITIES,
    MarketStatus,
    MarketType,
    Outcome,
    OutcomeType,
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
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiProvider(BasePredictionProvider):
    """
    Kalshi prediction market provider.

    Kalshi is a CFTC-regulated prediction market offering event contracts
    on economics, politics, climate, technology, and more.

    Example:
        config = ProviderConfig(
            api_key="your_api_key",
            private_key="your_rsa_private_key",
        )
        provider = KalshiProvider(config)
        await provider.connect()

        markets = await provider.get_markets(category="Politics")
        quote = await provider.get_quote(market_id, "yes")
    """

    @property
    def name(self) -> str:
        return "kalshi"

    @property
    def base_url(self) -> str:
        if self._config.testnet:
            return KALSHI_DEMO_URL
        return KALSHI_API_URL

    @property
    def capabilities(self) -> PredictionMarketCapabilities:
        return KALSHI_CAPABILITIES

    # =========================================================================
    # Market Data (Public - No Auth Required)
    # =========================================================================

    async def get_markets(
        self,
        category: str | None = None,
        status: MarketStatus | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PredictionMarket]:
        """Get markets from Kalshi."""
        self._ensure_connected()

        params: dict[str, Any] = {
            "limit": min(limit, 200),
        }

        if status == MarketStatus.OPEN:
            params["status"] = "open"
        elif status == MarketStatus.CLOSED:
            params["status"] = "closed"
        elif status == MarketStatus.RESOLVED:
            params["status"] = "settled"

        # Kalshi uses series for categories
        if category:
            params["series_ticker"] = category.upper()

        try:
            response = await self._get("/markets", params=params)
            markets_data = response.get("markets", [])

            markets = []
            for item in markets_data[offset : offset + limit]:
                market = self._parse_market(item)
                if market:
                    if search and search.lower() not in market.title.lower():
                        continue
                    markets.append(market)
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch Kalshi markets: {e}")
            return []

    async def get_market(self, market_id: str) -> PredictionMarket | None:
        """Get a specific market by ticker."""
        self._ensure_connected()

        try:
            response = await self._get(f"/markets/{market_id}")
            market_data = response.get("market")
            if market_data:
                return self._parse_market(market_data)
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
            logger.warning(f"Failed to get quote for {market_id}: {e}")
            return None

    async def get_orderbook(
        self, market_id: str, outcome_id: str, depth: int = 20
    ) -> PredictionOrderBook | None:
        """Get order book for a market."""
        self._ensure_connected()

        try:
            response = await self._get(f"/markets/{market_id}/orderbook")
            orderbook_data = response.get("orderbook", {})

            # Kalshi returns YES side orderbook
            # For binary markets: NO price = 1 - YES price
            bids = []
            asks = []

            # YES bids (someone wants to buy YES)
            for bid in orderbook_data.get("yes", [])[:depth]:
                price = Decimal(str(bid[0])) / 100  # Kalshi uses cents
                size = Decimal(str(bid[1]))
                if outcome_id.lower() == "yes":
                    bids.append(PredictionOrderBookLevel(price=price, size=size))
                else:
                    # For NO, this becomes an ask at (1 - price)
                    asks.append(
                        PredictionOrderBookLevel(price=Decimal("1") - price, size=size)
                    )

            # YES asks (someone wants to sell YES)
            for ask in orderbook_data.get("no", [])[:depth]:
                price = Decimal(str(ask[0])) / 100
                size = Decimal(str(ask[1]))
                if outcome_id.lower() == "yes":
                    asks.append(PredictionOrderBookLevel(price=price, size=size))
                else:
                    bids.append(
                        PredictionOrderBookLevel(price=Decimal("1") - price, size=size)
                    )

            # Sort appropriately
            bids = sorted(bids, key=lambda x: x.price, reverse=True)
            asks = sorted(asks, key=lambda x: x.price)

            return PredictionOrderBook(
                market_id=market_id,
                outcome_id=outcome_id,
                platform=self.name,
                bids=tuple(bids),
                asks=tuple(asks),
                timestamp_ns=time.time_ns(),
            )
        except Exception as e:
            logger.warning(f"Failed to get orderbook for {market_id}: {e}")
            return None

    # =========================================================================
    # Trading (Phase 3 - Requires Auth)
    # =========================================================================

    async def get_positions(
        self, market_id: str | None = None
    ) -> list[PredictionPosition]:
        """Get user positions."""
        self._ensure_connected()

        if not self._config.api_key:
            logger.warning("API key required for positions")
            return []

        try:
            headers = await self._get_auth_headers("GET", "/portfolio/positions")
            params = {}
            if market_id:
                params["ticker"] = market_id

            response = await self._get(
                "/portfolio/positions", params=params, headers=headers
            )

            positions = []
            for item in response.get("market_positions", []):
                # Calculate unrealized P&L
                size = Decimal(str(item.get("position", 0)))
                avg_price = Decimal(str(item.get("average_price", 0))) / 100
                # Would need current price for accurate P&L
                current_price = avg_price  # Placeholder

                positions.append(
                    PredictionPosition(
                        market_id=item.get("ticker", ""),
                        outcome_id="yes" if size > 0 else "no",
                        platform=self.name,
                        size=abs(size),
                        avg_price=avg_price,
                        current_price=current_price,
                        unrealized_pnl=Decimal("0"),  # Would calculate with current price
                        realized_pnl=Decimal(str(item.get("realized_pnl", 0))) / 100,
                        timestamp_ns=time.time_ns(),
                    )
                )
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    async def submit_order(self, order: PredictionOrder) -> PredictionOrderResult:
        """Submit order to Kalshi."""
        self._ensure_connected()

        if not self._config.api_key:
            raise ValueError("API key required for trading")

        try:
            headers = await self._get_auth_headers("POST", "/portfolio/orders")

            # Kalshi uses cents for prices
            price_cents = int(float(order.price or 0) * 100) if order.price else None

            order_data = {
                "ticker": order.market_id,
                "side": "yes" if order.side == PredictionOrderSide.BUY else "no",
                "count": int(order.size),
                "type": "limit" if order.price else "market",
                "action": "buy" if order.side == PredictionOrderSide.BUY else "sell",
            }

            if price_cents:
                order_data["yes_price"] = price_cents

            response = await self._post(
                "/portfolio/orders", json=order_data, headers=headers
            )
            order_result = response.get("order", {})

            return PredictionOrderResult(
                order_id=order_result.get("order_id", ""),
                market_id=order.market_id,
                outcome_id=order.outcome_id,
                platform=self.name,
                status=self._parse_order_status(order_result.get("status", "pending")),
                side=order.side,
                size=order.size,
                filled_size=Decimal(str(order_result.get("count_filled", 0))),
                timestamp_ns=time.time_ns(),
                average_price=Decimal(str(order_result.get("average_price", 0))) / 100
                if order_result.get("average_price")
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
            headers = await self._get_auth_headers(
                "DELETE", f"/portfolio/orders/{order_id}"
            )
            await self._delete(f"/portfolio/orders/{order_id}", headers=headers)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_balance(self) -> dict[str, Decimal]:
        """Get USD balance."""
        self._ensure_connected()

        if not self._config.api_key:
            return {}

        try:
            headers = await self._get_auth_headers("GET", "/portfolio/balance")
            response = await self._get("/portfolio/balance", headers=headers)

            # Kalshi returns cents
            balance_cents = response.get("balance", 0)
            return {
                "USD": Decimal(str(balance_cents)) / 100,
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _parse_market(self, data: dict[str, Any]) -> PredictionMarket | None:
        """Parse market data from Kalshi API response."""
        try:
            ticker = data.get("ticker", "")
            title = data.get("title", data.get("subtitle", ""))

            # Parse status
            status_str = data.get("status", "open")
            status_map = {
                "open": MarketStatus.OPEN,
                "closed": MarketStatus.CLOSED,
                "settled": MarketStatus.RESOLVED,
            }
            status = status_map.get(status_str.lower(), MarketStatus.OPEN)

            # Parse prices (Kalshi uses cents 0-100)
            yes_price = Decimal(str(data.get("yes_ask", 50))) / 100
            no_price = Decimal("1") - yes_price

            outcomes = [
                Outcome(
                    outcome_id="yes",
                    name="Yes",
                    probability=yes_price,
                    price=yes_price,
                    volume=Decimal(str(data.get("volume", 0))),
                    winner=data.get("result") == "yes" if data.get("result") else None,
                ),
                Outcome(
                    outcome_id="no",
                    name="No",
                    probability=no_price,
                    price=no_price,
                    volume=Decimal(str(data.get("volume", 0))),
                    winner=data.get("result") == "no" if data.get("result") else None,
                ),
            ]

            # Parse timestamps
            close_time = data.get("close_time")
            close_date = None
            if close_time:
                try:
                    dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    close_date = int(dt.timestamp() * 1_000_000_000)
                except Exception:
                    pass

            # Determine outcome type
            outcome_type = OutcomeType.BINARY
            if data.get("floor_strike") or data.get("cap_strike"):
                outcome_type = OutcomeType.SCALAR

            return PredictionMarket(
                market_id=ticker,
                platform=self.name,
                title=title,
                description=data.get("rules_primary"),
                category=data.get("category"),
                tags=tuple(data.get("tags", [])) if data.get("tags") else (),
                slug=ticker.lower(),
                url=f"https://kalshi.com/markets/{ticker}",
                outcomes=tuple(outcomes),
                status=status,
                outcome_type=outcome_type,
                market_type=MarketType.CLOB,
                volume=Decimal(str(data.get("volume", 0))),
                volume_24h=Decimal(str(data.get("volume_24h", 0))),
                liquidity=Decimal(str(data.get("open_interest", 0))),
                close_date=close_date,
                resolution_source="Kalshi Resolution Source",
                updated_at=time.time_ns(),
            )
        except Exception as e:
            logger.error(f"Failed to parse market: {e}")
            return None

    def _parse_order_status(self, status: str) -> PredictionOrderStatus:
        """Parse order status string."""
        status_map = {
            "pending": PredictionOrderStatus.PENDING,
            "resting": PredictionOrderStatus.OPEN,
            "executed": PredictionOrderStatus.FILLED,
            "canceled": PredictionOrderStatus.CANCELLED,
            "cancelled": PredictionOrderStatus.CANCELLED,
        }
        return status_map.get(status.lower(), PredictionOrderStatus.PENDING)

    async def _get_auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authentication headers using RSA signature."""
        if not self._config.api_key:
            return {}

        timestamp = str(int(time.time() * 1000))

        # Kalshi uses RSA-based authentication
        # For simplicity, using API key header if no private key
        if self._config.private_key:
            # RSA signature would go here
            # message = f"{timestamp}{method}{path}"
            # signature = sign_with_rsa(message, private_key)
            pass

        return {
            "Authorization": f"Bearer {self._config.api_key}",
        }

    async def _verify_connection(self) -> None:
        """Verify connection by fetching exchange status."""
        # Kalshi has a status endpoint
        try:
            await self._get("/exchange/status")
        except Exception:
            # Fallback to markets
            await self.get_markets(limit=1)

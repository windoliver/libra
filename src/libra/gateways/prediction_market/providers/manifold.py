"""
Manifold Markets Provider.

Provider for Manifold Markets - a play-money prediction market.

API Documentation: https://docs.manifold.markets/api

Features:
- Play-money (Mana) trading
- AMM-based markets (CPMM)
- User-created markets
- Multiple outcome types

Note: Manifold uses play money (Mana), not real currency.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

from libra.gateways.prediction_market.protocol import (
    MANIFOLD_CAPABILITIES,
    MarketStatus,
    MarketType,
    Outcome,
    OutcomeType,
    PredictionMarket,
    PredictionMarketCapabilities,
    PredictionOrder,
    PredictionOrderResult,
    PredictionOrderSide,
    PredictionOrderStatus,
    PredictionPosition,
    PredictionQuote,
)
from libra.gateways.prediction_market.providers.base import (
    BasePredictionProvider,
)


logger = logging.getLogger(__name__)


# API endpoints
MANIFOLD_API_URL = "https://api.manifold.markets/v0"


class ManifoldProvider(BasePredictionProvider):
    """
    Manifold Markets prediction market provider.

    Manifold Markets is a play-money prediction market where users
    trade with Mana (M$), the platform's virtual currency.

    Example:
        config = ProviderConfig(api_key="your_api_key")
        provider = ManifoldProvider(config)
        await provider.connect()

        markets = await provider.get_markets(category="technology")
        quote = await provider.get_quote(market_id, "yes")
    """

    @property
    def name(self) -> str:
        return "manifold"

    @property
    def base_url(self) -> str:
        return MANIFOLD_API_URL

    @property
    def capabilities(self) -> PredictionMarketCapabilities:
        return MANIFOLD_CAPABILITIES

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
        """Get markets from Manifold."""
        self._ensure_connected()

        params: dict[str, Any] = {
            "limit": min(limit, 1000),
        }

        if search:
            # Use search endpoint
            params["term"] = search
            endpoint = "/search-markets"
        else:
            endpoint = "/markets"

        try:
            response = await self._get(endpoint, params=params)

            # Handle different response formats
            if isinstance(response, dict):
                markets_data = response.get("markets", response.get("data", []))
            else:
                markets_data = response

            markets = []
            for item in markets_data[offset : offset + limit]:
                # Filter by status
                is_resolved = item.get("isResolved", False)
                close_time = item.get("closeTime")
                is_closed = False
                if close_time:
                    try:
                        is_closed = close_time < time.time() * 1000
                    except Exception:
                        pass

                if status == MarketStatus.OPEN and (is_resolved or is_closed):
                    continue
                if status == MarketStatus.RESOLVED and not is_resolved:
                    continue
                if status == MarketStatus.CLOSED and not is_closed:
                    continue

                market = self._parse_market(item)
                if market:
                    # Filter by category (using group)
                    if category:
                        groups = item.get("groupSlugs", [])
                        if not any(category.lower() in g.lower() for g in groups):
                            continue
                    markets.append(market)

            return markets
        except Exception as e:
            logger.error(f"Failed to fetch Manifold markets: {e}")
            return []

    async def get_market(self, market_id: str) -> PredictionMarket | None:
        """Get a specific market by ID or slug."""
        self._ensure_connected()

        try:
            # Try by slug first
            response = await self._get(f"/slug/{market_id}")
            return self._parse_market(response)
        except Exception:
            try:
                # Try by ID
                response = await self._get(f"/market/{market_id}")
                return self._parse_market(response)
            except Exception as e:
                logger.warning(f"Market {market_id} not found: {e}")
                return None

    async def get_quote(
        self, market_id: str, outcome_id: str
    ) -> PredictionQuote | None:
        """
        Get quote for a market outcome.

        Manifold uses AMM, so there's no traditional order book.
        We return the current probability as the price.
        """
        self._ensure_connected()

        try:
            market = await self.get_market(market_id)
            if not market:
                return None

            for outcome in market.outcomes:
                if outcome.outcome_id == outcome_id:
                    price = outcome.price
                    return PredictionQuote(
                        market_id=market_id,
                        outcome_id=outcome_id,
                        platform=self.name,
                        bid=price,  # AMM - no spread
                        ask=price,
                        mid=price,
                        timestamp_ns=time.time_ns(),
                    )

            return None
        except Exception as e:
            logger.warning(f"Failed to get quote for {market_id}: {e}")
            return None

    # =========================================================================
    # Trading
    # =========================================================================

    async def get_positions(
        self, market_id: str | None = None
    ) -> list[PredictionPosition]:
        """Get user positions (bets)."""
        self._ensure_connected()

        if not self._config.api_key:
            logger.warning("API key required for positions")
            return []

        try:
            headers = {"Authorization": f"Key {self._config.api_key}"}
            response = await self._get("/me", headers=headers)
            user_id = response.get("id")

            if not user_id:
                return []

            # Get user's bets
            params: dict[str, Any] = {"userId": user_id}
            if market_id:
                params["contractId"] = market_id

            bets_response = await self._get("/bets", params=params)

            # Aggregate bets into positions
            positions_map: dict[str, dict[str, Any]] = {}

            for bet in bets_response:
                contract_id = bet.get("contractId", "")
                outcome = bet.get("outcome", "")
                key = f"{contract_id}_{outcome}"

                if key not in positions_map:
                    positions_map[key] = {
                        "market_id": contract_id,
                        "outcome_id": outcome.lower(),
                        "size": Decimal("0"),
                        "cost": Decimal("0"),
                        "payout": Decimal("0"),
                    }

                amount = Decimal(str(bet.get("amount", 0)))
                shares = Decimal(str(bet.get("shares", 0)))

                positions_map[key]["size"] += shares
                positions_map[key]["cost"] += amount

            positions = []
            for pos_data in positions_map.values():
                if pos_data["size"] > 0:
                    avg_price = (
                        pos_data["cost"] / pos_data["size"]
                        if pos_data["size"] > 0
                        else Decimal("0")
                    )
                    positions.append(
                        PredictionPosition(
                            market_id=pos_data["market_id"],
                            outcome_id=pos_data["outcome_id"],
                            platform=self.name,
                            size=pos_data["size"],
                            avg_price=avg_price,
                            current_price=avg_price,  # Would need to fetch current
                            unrealized_pnl=Decimal("0"),
                            cost_basis=pos_data["cost"],
                            timestamp_ns=time.time_ns(),
                        )
                    )

            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    async def submit_order(self, order: PredictionOrder) -> PredictionOrderResult:
        """Submit a bet (order) to Manifold."""
        self._ensure_connected()

        if not self._config.api_key:
            raise ValueError("API key required for trading")

        try:
            headers = {"Authorization": f"Key {self._config.api_key}"}

            bet_data = {
                "contractId": order.market_id,
                "outcome": order.outcome_id.upper(),
                "amount": float(order.size),  # Mana amount
            }

            if order.price:
                # Limit order
                bet_data["limitProb"] = float(order.price)

            response = await self._post("/bet", json=bet_data, headers=headers)

            return PredictionOrderResult(
                order_id=response.get("betId", str(time.time_ns())),
                market_id=order.market_id,
                outcome_id=order.outcome_id,
                platform=self.name,
                status=PredictionOrderStatus.FILLED,  # AMM fills instantly
                side=order.side,
                size=order.size,
                filled_size=Decimal(str(response.get("amount", order.size))),
                timestamp_ns=time.time_ns(),
                average_price=Decimal(str(response.get("probAfter", 0))),
            )
        except Exception as e:
            logger.error(f"Failed to submit bet: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a limit order (bet)."""
        self._ensure_connected()

        if not self._config.api_key:
            raise ValueError("API key required for trading")

        try:
            headers = {"Authorization": f"Key {self._config.api_key}"}
            await self._post(f"/bet/cancel/{order_id}", headers=headers)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel bet {order_id}: {e}")
            return False

    async def get_balance(self) -> dict[str, Decimal]:
        """Get Mana balance."""
        self._ensure_connected()

        if not self._config.api_key:
            return {}

        try:
            headers = {"Authorization": f"Key {self._config.api_key}"}
            response = await self._get("/me", headers=headers)

            return {
                "MANA": Decimal(str(response.get("balance", 0))),
                "M$": Decimal(str(response.get("balance", 0))),
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _parse_market(self, data: dict[str, Any]) -> PredictionMarket | None:
        """Parse market data from Manifold API response."""
        try:
            market_id = data.get("id", "")
            title = data.get("question", "")

            # Parse status
            is_resolved = data.get("isResolved", False)
            close_time = data.get("closeTime")

            if is_resolved:
                status = MarketStatus.RESOLVED
            elif close_time and close_time < time.time() * 1000:
                status = MarketStatus.CLOSED
            else:
                status = MarketStatus.OPEN

            # Parse outcome type
            outcome_type_str = data.get("outcomeType", "BINARY")
            if outcome_type_str == "BINARY":
                outcome_type = OutcomeType.BINARY
            elif outcome_type_str in ("NUMERIC", "PSEUDO_NUMERIC"):
                outcome_type = OutcomeType.SCALAR
            else:
                outcome_type = OutcomeType.MULTIPLE

            # Parse probability/outcomes
            prob = Decimal(str(data.get("probability", 0.5)))

            if outcome_type == OutcomeType.BINARY:
                resolution = data.get("resolution")
                outcomes = (
                    Outcome(
                        outcome_id="yes",
                        name="Yes",
                        probability=prob,
                        price=prob,
                        volume=Decimal(str(data.get("volume", 0))),
                        winner=resolution == "YES" if resolution else None,
                    ),
                    Outcome(
                        outcome_id="no",
                        name="No",
                        probability=Decimal("1") - prob,
                        price=Decimal("1") - prob,
                        volume=Decimal(str(data.get("volume", 0))),
                        winner=resolution == "NO" if resolution else None,
                    ),
                )
            elif outcome_type == OutcomeType.MULTIPLE:
                # Multiple choice
                answers = data.get("answers", [])
                outcomes = tuple(
                    Outcome(
                        outcome_id=str(ans.get("id", "")),
                        name=ans.get("text", ""),
                        probability=Decimal(str(ans.get("probability", 0))),
                        price=Decimal(str(ans.get("probability", 0))),
                        volume=Decimal("0"),
                    )
                    for ans in answers
                )
            else:
                # Scalar
                outcomes = (
                    Outcome(
                        outcome_id="value",
                        name="Value",
                        probability=prob,
                        price=prob,
                        volume=Decimal(str(data.get("volume", 0))),
                    ),
                )

            # Parse timestamps
            close_date = None
            if close_time:
                try:
                    close_date = int(close_time * 1_000_000)  # ms to ns
                except Exception:
                    pass

            created_time = data.get("createdTime")
            created_at = None
            if created_time:
                try:
                    created_at = int(created_time * 1_000_000)  # ms to ns
                except Exception:
                    pass

            # Market mechanism
            mechanism = data.get("mechanism", "cpmm-1")
            if mechanism in ("cpmm-1", "cpmm-2"):
                market_type = MarketType.CPMM
            elif mechanism == "dpm-2":
                market_type = MarketType.AMM
            else:
                market_type = MarketType.AMM

            return PredictionMarket(
                market_id=market_id,
                platform=self.name,
                title=title,
                description=data.get("description"),
                category=data.get("groupSlugs", [""])[0] if data.get("groupSlugs") else None,
                tags=tuple(data.get("groupSlugs", [])),
                slug=data.get("slug"),
                url=data.get("url") or f"https://manifold.markets/{data.get('creatorUsername', '')}/{data.get('slug', '')}",
                outcomes=outcomes,
                status=status,
                outcome_type=outcome_type,
                market_type=market_type,
                volume=Decimal(str(data.get("volume", 0))),
                volume_24h=Decimal(str(data.get("volume24Hours", 0))),
                liquidity=Decimal(str(data.get("totalLiquidity", 0))),
                num_traders=data.get("uniqueBettorCount", 0),
                close_date=close_date,
                created_at=created_at,
                updated_at=time.time_ns(),
            )
        except Exception as e:
            logger.error(f"Failed to parse market: {e}")
            return None

    async def _verify_connection(self) -> None:
        """Verify connection by fetching markets."""
        await self.get_markets(limit=1)

"""
Metaculus Provider.

Provider for Metaculus - a reputation-based forecasting platform.

API Documentation: https://www.metaculus.com/api/

Features:
- Community forecasts
- AI predictions (Metaculus prediction)
- Research questions
- Tournaments and competitions

Note: Metaculus is NOT a trading platform - it's reputation-based forecasting.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

from libra.gateways.prediction_market.protocol import (
    MarketStatus,
    MarketType,
    METACULUS_CAPABILITIES,
    Outcome,
    OutcomeType,
    PredictionMarket,
    PredictionMarketCapabilities,
    PredictionQuote,
)
from libra.gateways.prediction_market.providers.base import (
    BasePredictionProvider,
)


logger = logging.getLogger(__name__)


# API endpoints
METACULUS_API_URL = "https://www.metaculus.com/api2"


class MetaculusProvider(BasePredictionProvider):
    """
    Metaculus forecasting platform provider.

    Metaculus is a reputation-based forecasting platform, not a trading market.
    It provides community predictions and AI forecasts.

    Example:
        provider = MetaculusProvider()
        await provider.connect()

        questions = await provider.get_markets(category="ai")
        # Get community prediction
        quote = await provider.get_quote(question_id, "yes")
    """

    @property
    def name(self) -> str:
        return "metaculus"

    @property
    def base_url(self) -> str:
        return METACULUS_API_URL

    @property
    def capabilities(self) -> PredictionMarketCapabilities:
        return METACULUS_CAPABILITIES

    # =========================================================================
    # Question Data (Markets in Metaculus are "Questions")
    # =========================================================================

    async def get_markets(
        self,
        category: str | None = None,
        status: MarketStatus | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PredictionMarket]:
        """Get questions from Metaculus."""
        self._ensure_connected()

        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "offset": offset,
            "order_by": "-activity",
        }

        # Status mapping
        if status == MarketStatus.OPEN:
            params["status"] = "open"
        elif status == MarketStatus.RESOLVED:
            params["status"] = "resolved"
        elif status == MarketStatus.CLOSED:
            params["status"] = "closed"

        if search:
            params["search"] = search

        if category:
            params["categories"] = category

        try:
            response = await self._get("/questions/", params=params)
            results = response.get("results", [])

            markets = []
            for item in results:
                market = self._parse_question(item)
                if market:
                    markets.append(market)
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch Metaculus questions: {e}")
            return []

    async def get_market(self, market_id: str) -> PredictionMarket | None:
        """Get a specific question by ID."""
        self._ensure_connected()

        try:
            response = await self._get(f"/questions/{market_id}/")
            return self._parse_question(response)
        except Exception as e:
            logger.warning(f"Question {market_id} not found: {e}")
            return None

    async def get_quote(
        self, market_id: str, outcome_id: str
    ) -> PredictionQuote | None:
        """
        Get community/Metaculus prediction for a question.

        Note: Metaculus doesn't have bid/ask - we return the community prediction.
        """
        self._ensure_connected()

        try:
            response = await self._get(f"/questions/{market_id}/")

            # Get predictions
            community_pred = response.get("community_prediction", {})
            metaculus_pred = response.get("metaculus_prediction", {})

            # For binary questions
            if response.get("type") == "binary":
                prob = None

                # Try community prediction first
                if community_pred:
                    prob = community_pred.get("full", {}).get("q2")  # Median

                # Fall back to Metaculus prediction
                if prob is None and metaculus_pred:
                    prob = metaculus_pred.get("full", {}).get("q2")

                if prob is not None:
                    prob_decimal = Decimal(str(prob))
                    if outcome_id.lower() in ("yes", "true"):
                        price = prob_decimal
                    else:
                        price = Decimal("1") - prob_decimal

                    return PredictionQuote(
                        market_id=market_id,
                        outcome_id=outcome_id,
                        platform=self.name,
                        bid=price,  # No real bid/ask in Metaculus
                        ask=price,
                        mid=price,
                        timestamp_ns=time.time_ns(),
                    )

            return None
        except Exception as e:
            logger.warning(f"Failed to get prediction for {market_id}: {e}")
            return None

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _parse_question(self, data: dict[str, Any]) -> PredictionMarket | None:
        """Parse question data from Metaculus API response."""
        try:
            question_id = str(data.get("id", ""))
            title = data.get("title", "")

            # Parse status
            status_str = data.get("status", "open")
            status_map = {
                "open": MarketStatus.OPEN,
                "closed": MarketStatus.CLOSED,
                "resolved": MarketStatus.RESOLVED,
            }
            status = status_map.get(status_str.lower(), MarketStatus.OPEN)

            # Parse question type
            q_type = data.get("type", "binary")
            if q_type == "binary":
                outcome_type = OutcomeType.BINARY
            elif q_type in ("numeric", "date"):
                outcome_type = OutcomeType.SCALAR
            else:
                outcome_type = OutcomeType.MULTIPLE

            # Get community prediction
            community_pred = data.get("community_prediction", {})
            prob = Decimal("0.5")  # Default

            if community_pred and outcome_type == OutcomeType.BINARY:
                full_pred = community_pred.get("full", {})
                if full_pred and full_pred.get("q2"):
                    prob = Decimal(str(full_pred["q2"]))

            # Create outcomes
            if outcome_type == OutcomeType.BINARY:
                outcomes = (
                    Outcome(
                        outcome_id="yes",
                        name="Yes",
                        probability=prob,
                        price=prob,
                        volume=Decimal(str(data.get("number_of_predictions", 0))),
                    ),
                    Outcome(
                        outcome_id="no",
                        name="No",
                        probability=Decimal("1") - prob,
                        price=Decimal("1") - prob,
                        volume=Decimal(str(data.get("number_of_predictions", 0))),
                    ),
                )
            else:
                # For non-binary, create single outcome
                outcomes = (
                    Outcome(
                        outcome_id="prediction",
                        name="Prediction",
                        probability=prob,
                        price=prob,
                        volume=Decimal(str(data.get("number_of_predictions", 0))),
                    ),
                )

            # Parse timestamps
            close_time = data.get("close_time")
            close_date = None
            if close_time:
                try:
                    dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    close_date = int(dt.timestamp() * 1_000_000_000)
                except Exception:
                    pass

            created_time = data.get("created_time")
            created_at = None
            if created_time:
                try:
                    dt = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
                    created_at = int(dt.timestamp() * 1_000_000_000)
                except Exception:
                    pass

            return PredictionMarket(
                market_id=question_id,
                platform=self.name,
                title=title,
                description=data.get("description"),
                category=data.get("categories", [""])[0] if data.get("categories") else None,
                tags=tuple(data.get("tags", [])) if data.get("tags") else (),
                slug=data.get("slug"),
                url=data.get("page_url") or f"https://www.metaculus.com/questions/{question_id}/",
                outcomes=outcomes,
                status=status,
                outcome_type=outcome_type,
                market_type=MarketType.REPUTATION,
                volume=Decimal(str(data.get("number_of_predictions", 0))),
                num_traders=data.get("number_of_forecasters", 0),
                close_date=close_date,
                resolution_source=data.get("resolution_criteria"),
                created_at=created_at,
                updated_at=time.time_ns(),
            )
        except Exception as e:
            logger.error(f"Failed to parse question: {e}")
            return None

    async def _verify_connection(self) -> None:
        """Verify connection by fetching one question."""
        await self.get_markets(limit=1)

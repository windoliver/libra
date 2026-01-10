"""Tests for Prediction Market Protocol data structures."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.gateways.prediction_market.protocol import (
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
    PredictionOrderType,
    PredictionPosition,
    PredictionQuote,
    POLYMARKET_CAPABILITIES,
    KALSHI_CAPABILITIES,
    METACULUS_CAPABILITIES,
    MANIFOLD_CAPABILITIES,
)


class TestOutcome:
    """Tests for Outcome data structure."""

    def test_create_outcome(self) -> None:
        """Test creating an outcome."""
        outcome = Outcome(
            outcome_id="yes",
            name="Yes",
            probability=Decimal("0.65"),
            price=Decimal("0.65"),
            volume=Decimal("150000"),
        )

        assert outcome.outcome_id == "yes"
        assert outcome.name == "Yes"
        assert outcome.probability == Decimal("0.65")
        assert outcome.price == Decimal("0.65")
        assert outcome.volume == Decimal("150000")

    def test_outcome_with_optional_fields(self) -> None:
        """Test outcome with optional fields."""
        outcome = Outcome(
            outcome_id="yes",
            name="Yes",
            probability=Decimal("0.65"),
            price=Decimal("0.65"),
            token_id="0x123abc",
            open_interest=Decimal("50000"),
            winner=True,
        )

        assert outcome.token_id == "0x123abc"
        assert outcome.open_interest == Decimal("50000")
        assert outcome.winner is True

    def test_outcome_is_frozen(self) -> None:
        """Test that outcome is immutable."""
        outcome = Outcome(
            outcome_id="yes",
            name="Yes",
            probability=Decimal("0.65"),
            price=Decimal("0.65"),
        )

        with pytest.raises(AttributeError):
            outcome.probability = Decimal("0.70")  # type: ignore


class TestPredictionMarket:
    """Tests for PredictionMarket data structure."""

    def test_create_binary_market(self) -> None:
        """Test creating a binary prediction market."""
        outcomes = (
            Outcome(
                outcome_id="yes",
                name="Yes",
                probability=Decimal("0.65"),
                price=Decimal("0.65"),
            ),
            Outcome(
                outcome_id="no",
                name="No",
                probability=Decimal("0.35"),
                price=Decimal("0.35"),
            ),
        )

        market = PredictionMarket(
            market_id="0x123abc",
            platform="polymarket",
            title="Will BTC exceed $100k in 2024?",
            outcomes=outcomes,
            status=MarketStatus.OPEN,
            outcome_type=OutcomeType.BINARY,
            market_type=MarketType.CLOB,
            volume=Decimal("500000"),
        )

        assert market.market_id == "0x123abc"
        assert market.platform == "polymarket"
        assert market.title == "Will BTC exceed $100k in 2024?"
        assert len(market.outcomes) == 2
        assert market.is_open is True
        assert market.is_resolved is False

    def test_market_best_prices(self) -> None:
        """Test best_yes_price and best_no_price properties."""
        outcomes = (
            Outcome(
                outcome_id="yes",
                name="Yes",
                probability=Decimal("0.65"),
                price=Decimal("0.65"),
            ),
            Outcome(
                outcome_id="no",
                name="No",
                probability=Decimal("0.35"),
                price=Decimal("0.35"),
            ),
        )

        market = PredictionMarket(
            market_id="test",
            platform="test",
            title="Test",
            outcomes=outcomes,
        )

        assert market.best_yes_price == Decimal("0.65")
        assert market.best_no_price == Decimal("0.35")

    def test_market_status_properties(self) -> None:
        """Test is_open and is_resolved properties."""
        # Open market
        open_market = PredictionMarket(
            market_id="test",
            platform="test",
            title="Test",
            outcomes=(),
            status=MarketStatus.OPEN,
        )
        assert open_market.is_open is True
        assert open_market.is_resolved is False

        # Resolved market
        resolved_market = PredictionMarket(
            market_id="test",
            platform="test",
            title="Test",
            outcomes=(),
            status=MarketStatus.RESOLVED,
        )
        assert resolved_market.is_open is False
        assert resolved_market.is_resolved is True


class TestPredictionQuote:
    """Tests for PredictionQuote data structure."""

    def test_create_quote(self) -> None:
        """Test creating a quote."""
        import time

        ts = time.time_ns()
        quote = PredictionQuote(
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            bid=Decimal("0.64"),
            ask=Decimal("0.66"),
            mid=Decimal("0.65"),
            timestamp_ns=ts,
        )

        assert quote.market_id == "0x123abc"
        assert quote.outcome_id == "yes"
        assert quote.bid == Decimal("0.64")
        assert quote.ask == Decimal("0.66")
        assert quote.mid == Decimal("0.65")

    def test_quote_spread(self) -> None:
        """Test spread calculation."""
        quote = PredictionQuote(
            market_id="test",
            outcome_id="yes",
            platform="test",
            bid=Decimal("0.64"),
            ask=Decimal("0.66"),
            mid=Decimal("0.65"),
            timestamp_ns=0,
        )

        assert quote.spread == Decimal("0.02")

    def test_quote_spread_bps(self) -> None:
        """Test spread in basis points."""
        quote = PredictionQuote(
            market_id="test",
            outcome_id="yes",
            platform="test",
            bid=Decimal("0.64"),
            ask=Decimal("0.66"),
            mid=Decimal("0.65"),
            timestamp_ns=0,
        )

        # 0.02 / 0.65 * 10000 ≈ 307.69 bps
        assert quote.spread_bps > Decimal("300")
        assert quote.spread_bps < Decimal("310")


class TestPredictionOrderBook:
    """Tests for PredictionOrderBook data structure."""

    def test_create_orderbook(self) -> None:
        """Test creating an order book."""
        bids = (
            PredictionOrderBookLevel(price=Decimal("0.64"), size=Decimal("1000")),
            PredictionOrderBookLevel(price=Decimal("0.63"), size=Decimal("2000")),
        )
        asks = (
            PredictionOrderBookLevel(price=Decimal("0.66"), size=Decimal("500")),
            PredictionOrderBookLevel(price=Decimal("0.67"), size=Decimal("1500")),
        )

        orderbook = PredictionOrderBook(
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            bids=bids,
            asks=asks,
            timestamp_ns=0,
        )

        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.best_bid == Decimal("0.64")
        assert orderbook.best_ask == Decimal("0.66")

    def test_orderbook_mid_and_spread(self) -> None:
        """Test mid and spread calculations."""
        bids = (PredictionOrderBookLevel(price=Decimal("0.64"), size=Decimal("1000")),)
        asks = (PredictionOrderBookLevel(price=Decimal("0.66"), size=Decimal("500")),)

        orderbook = PredictionOrderBook(
            market_id="test",
            outcome_id="yes",
            platform="test",
            bids=bids,
            asks=asks,
            timestamp_ns=0,
        )

        assert orderbook.mid == Decimal("0.65")
        assert orderbook.spread == Decimal("0.02")

    def test_empty_orderbook(self) -> None:
        """Test empty order book."""
        orderbook = PredictionOrderBook(
            market_id="test",
            outcome_id="yes",
            platform="test",
            bids=(),
            asks=(),
            timestamp_ns=0,
        )

        assert orderbook.best_bid is None
        assert orderbook.best_ask is None
        assert orderbook.mid is None
        assert orderbook.spread is None


class TestPredictionOrder:
    """Tests for PredictionOrder data structure."""

    def test_create_limit_order(self) -> None:
        """Test creating a limit order."""
        order = PredictionOrder(
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            side=PredictionOrderSide.BUY,
            order_type=PredictionOrderType.LIMIT,
            size=Decimal("100"),
            price=Decimal("0.65"),
        )

        assert order.market_id == "0x123abc"
        assert order.side == PredictionOrderSide.BUY
        assert order.order_type == PredictionOrderType.LIMIT
        assert order.size == Decimal("100")
        assert order.price == Decimal("0.65")

    def test_order_with_id(self) -> None:
        """Test with_id method."""
        order = PredictionOrder(
            market_id="test",
            outcome_id="yes",
            platform="test",
            side=PredictionOrderSide.BUY,
            order_type=PredictionOrderType.MARKET,
            size=Decimal("100"),
        )

        order_with_id = order.with_id("order_123")

        assert order_with_id.order_id == "order_123"
        assert order_with_id.market_id == order.market_id
        assert order_with_id.size == order.size

    def test_order_with_timestamp(self) -> None:
        """Test with_timestamp method."""
        order = PredictionOrder(
            market_id="test",
            outcome_id="yes",
            platform="test",
            side=PredictionOrderSide.SELL,
            order_type=PredictionOrderType.LIMIT,
            size=Decimal("50"),
            price=Decimal("0.70"),
        )

        order_with_ts = order.with_timestamp()

        assert order_with_ts.timestamp_ns is not None
        assert order_with_ts.timestamp_ns > 0


class TestPredictionOrderResult:
    """Tests for PredictionOrderResult data structure."""

    def test_create_filled_result(self) -> None:
        """Test creating a filled order result."""
        result = PredictionOrderResult(
            order_id="ord_123",
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            status=PredictionOrderStatus.FILLED,
            side=PredictionOrderSide.BUY,
            size=Decimal("100"),
            filled_size=Decimal("100"),
            timestamp_ns=0,
            average_price=Decimal("0.65"),
        )

        assert result.order_id == "ord_123"
        assert result.status == PredictionOrderStatus.FILLED
        assert result.is_open is False
        assert result.fill_percent == Decimal("100")

    def test_partially_filled_result(self) -> None:
        """Test partially filled order result."""
        result = PredictionOrderResult(
            order_id="ord_123",
            market_id="test",
            outcome_id="yes",
            platform="test",
            status=PredictionOrderStatus.PARTIALLY_FILLED,
            side=PredictionOrderSide.BUY,
            size=Decimal("100"),
            filled_size=Decimal("50"),
            timestamp_ns=0,
        )

        assert result.is_open is True
        assert result.fill_percent == Decimal("50")


class TestPredictionPosition:
    """Tests for PredictionPosition data structure."""

    def test_create_position(self) -> None:
        """Test creating a position."""
        position = PredictionPosition(
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            size=Decimal("500"),
            avg_price=Decimal("0.60"),
            current_price=Decimal("0.65"),
            unrealized_pnl=Decimal("25"),
        )

        assert position.size == Decimal("500")
        assert position.avg_price == Decimal("0.60")
        assert position.current_price == Decimal("0.65")
        assert position.unrealized_pnl == Decimal("25")

    def test_position_total_pnl(self) -> None:
        """Test total P&L calculation."""
        position = PredictionPosition(
            market_id="test",
            outcome_id="yes",
            platform="test",
            size=Decimal("100"),
            avg_price=Decimal("0.50"),
            current_price=Decimal("0.60"),
            unrealized_pnl=Decimal("10"),
            realized_pnl=Decimal("5"),
        )

        assert position.total_pnl == Decimal("15")

    def test_position_pnl_percent(self) -> None:
        """Test P&L percentage calculation."""
        position = PredictionPosition(
            market_id="test",
            outcome_id="yes",
            platform="test",
            size=Decimal("100"),
            avg_price=Decimal("0.50"),
            current_price=Decimal("0.60"),
            unrealized_pnl=Decimal("10"),
        )

        # 10 / (100 * 0.50) * 100 = 20%
        assert position.pnl_percent == Decimal("20")


class TestPredictionMarketCapabilities:
    """Tests for capability presets."""

    def test_polymarket_capabilities(self) -> None:
        """Test Polymarket capability preset."""
        caps = POLYMARKET_CAPABILITIES

        assert caps.supports_trading is True
        assert caps.supports_limit_orders is True
        assert caps.supports_orderbook is True
        assert caps.is_real_money is True
        assert caps.settlement_currency == "USDC"

    def test_kalshi_capabilities(self) -> None:
        """Test Kalshi capability preset."""
        caps = KALSHI_CAPABILITIES

        assert caps.supports_trading is True
        assert caps.is_real_money is True
        assert caps.is_regulated is True
        assert caps.settlement_currency == "USD"

    def test_metaculus_capabilities(self) -> None:
        """Test Metaculus capability preset."""
        caps = METACULUS_CAPABILITIES

        assert caps.supports_trading is False
        assert caps.is_real_money is False
        assert caps.supports_orderbook is False

    def test_manifold_capabilities(self) -> None:
        """Test Manifold capability preset."""
        caps = MANIFOLD_CAPABILITIES

        assert caps.supports_trading is True
        assert caps.is_real_money is False
        assert caps.settlement_currency == "M$"


class TestEnums:
    """Tests for enum values."""

    def test_market_status_values(self) -> None:
        """Test MarketStatus enum values."""
        assert MarketStatus.OPEN.value == "open"
        assert MarketStatus.CLOSED.value == "closed"
        assert MarketStatus.RESOLVED.value == "resolved"
        assert MarketStatus.CANCELLED.value == "cancelled"

    def test_outcome_type_values(self) -> None:
        """Test OutcomeType enum values."""
        assert OutcomeType.BINARY.value == "binary"
        assert OutcomeType.MULTIPLE.value == "multiple"
        assert OutcomeType.SCALAR.value == "scalar"

    def test_market_type_values(self) -> None:
        """Test MarketType enum values."""
        assert MarketType.CLOB.value == "clob"
        assert MarketType.AMM.value == "amm"
        assert MarketType.LMSR.value == "lmsr"
        assert MarketType.CPMM.value == "cpmm"
        assert MarketType.REPUTATION.value == "reputation"

    def test_order_side_values(self) -> None:
        """Test PredictionOrderSide enum values."""
        assert PredictionOrderSide.BUY.value == "buy"
        assert PredictionOrderSide.SELL.value == "sell"

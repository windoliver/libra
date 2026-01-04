"""
Tests for Gateway Protocol and data structures.

Tests:
- Order, OrderResult, Position, Tick creation and properties
- Protocol runtime_checkable behavior
- Serialization/deserialization
- Edge cases and validation
"""

import time
from decimal import Decimal

import pytest

from libra.gateways import (
    Balance,
    Gateway,
    Order,
    OrderBook,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperGateway,
    Position,
    PositionSide,
    Tick,
    TimeInForce,
    decode_order,
    encode_order,
)


class TestOrder:
    """Tests for Order struct."""

    def test_create_market_order(self) -> None:
        """Test creating a market order."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.amount == Decimal("0.1")
        assert order.price is None
        assert order.time_in_force == TimeInForce.GTC

    def test_create_limit_order(self) -> None:
        """Test creating a limit order with price."""
        order = Order(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("2000.00"),
            time_in_force=TimeInForce.IOC,
        )

        assert order.symbol == "ETH/USDT"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.amount == Decimal("1.0")
        assert order.price == Decimal("2000.00")
        assert order.time_in_force == TimeInForce.IOC

    def test_create_stop_order(self) -> None:
        """Test creating a stop order."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("0.5"),
            stop_price=Decimal("45000.00"),
        )

        assert order.order_type == OrderType.STOP
        assert order.stop_price == Decimal("45000.00")

    def test_order_with_id(self) -> None:
        """Test adding order ID."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        order_with_id = order.with_id("12345")

        assert order.id is None
        assert order_with_id.id == "12345"
        assert order_with_id.symbol == order.symbol

    def test_order_with_timestamp(self) -> None:
        """Test adding timestamp."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        before = time.time_ns()
        order_with_ts = order.with_timestamp()
        after = time.time_ns()

        assert order.timestamp_ns is None
        assert order_with_ts.timestamp_ns is not None
        assert before <= order_with_ts.timestamp_ns <= after

    def test_order_immutable(self) -> None:
        """Test that Order is immutable (frozen)."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        with pytest.raises(AttributeError):
            order.amount = Decimal("0.2")  # type: ignore

    def test_order_serialization(self) -> None:
        """Test Order JSON serialization."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("50000.00"),
            client_order_id="test-123",
        )

        encoded = encode_order(order)
        decoded = decode_order(encoded)

        assert decoded.symbol == order.symbol
        assert decoded.side == order.side
        assert decoded.order_type == order.order_type
        assert decoded.amount == order.amount
        assert decoded.price == order.price
        assert decoded.client_order_id == order.client_order_id


class TestOrderResult:
    """Tests for OrderResult struct."""

    def test_create_filled_result(self) -> None:
        """Test creating a filled order result."""
        result = OrderResult(
            order_id="12345",
            symbol="BTC/USDT",
            status=OrderStatus.FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
            filled_amount=Decimal("0.1"),
            remaining_amount=Decimal("0"),
            average_price=Decimal("50000.00"),
            fee=Decimal("0.00005"),
            fee_currency="BTC",
            timestamp_ns=time.time_ns(),
        )

        assert result.status == OrderStatus.FILLED
        assert result.is_closed
        assert not result.is_open
        assert result.fill_percent == Decimal("100")

    def test_create_partial_fill(self) -> None:
        """Test partially filled order."""
        result = OrderResult(
            order_id="12345",
            symbol="BTC/USDT",
            status=OrderStatus.PARTIALLY_FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            filled_amount=Decimal("0.3"),
            remaining_amount=Decimal("0.7"),
            average_price=Decimal("50000.00"),
            fee=Decimal("0.00015"),
            fee_currency="BTC",
            timestamp_ns=time.time_ns(),
        )

        assert result.is_open
        assert not result.is_closed
        assert result.fill_percent == Decimal("30")

    def test_open_order(self) -> None:
        """Test open order status."""
        result = OrderResult(
            order_id="12345",
            symbol="BTC/USDT",
            status=OrderStatus.OPEN,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            filled_amount=Decimal("0"),
            remaining_amount=Decimal("1.0"),
            average_price=None,
            fee=Decimal("0"),
            fee_currency="BTC",
            timestamp_ns=time.time_ns(),
        )

        assert result.is_open
        assert result.fill_percent == Decimal("0")


class TestPosition:
    """Tests for Position struct."""

    def test_create_long_position(self) -> None:
        """Test creating a long position."""
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("48000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("1000.00"),
            realized_pnl=Decimal("0"),
        )

        assert position.side == PositionSide.LONG
        assert position.notional_value == Decimal("25000.00")
        assert position.total_pnl == Decimal("1000.00")

    def test_position_pnl_percent(self) -> None:
        """Test P&L percentage calculation."""
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("1.0"),
            entry_price=Decimal("100.00"),
            current_price=Decimal("110.00"),
            unrealized_pnl=Decimal("10.00"),
            realized_pnl=Decimal("0"),
        )

        assert position.pnl_percent == Decimal("10")

    def test_short_position(self) -> None:
        """Test short position."""
        position = Position(
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            amount=Decimal("10.0"),
            entry_price=Decimal("2000.00"),
            current_price=Decimal("1900.00"),
            unrealized_pnl=Decimal("1000.00"),  # Profit on short
            realized_pnl=Decimal("0"),
        )

        assert position.side == PositionSide.SHORT
        assert position.notional_value == Decimal("19000.00")


class TestTick:
    """Tests for Tick struct."""

    def test_create_tick(self) -> None:
        """Test creating a tick."""
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("49999.00"),
            ask=Decimal("50001.00"),
            last=Decimal("50000.00"),
            timestamp_ns=time.time_ns(),
        )

        assert tick.symbol == "BTC/USDT"
        assert tick.mid == Decimal("50000.00")
        assert tick.spread == Decimal("2.00")

    def test_tick_spread_bps(self) -> None:
        """Test spread in basis points."""
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("10000.00"),
            ask=Decimal("10001.00"),
            last=Decimal("10000.50"),
            timestamp_ns=time.time_ns(),
        )

        # Spread is 1, mid is 10000.50
        # spread_bps = (1 / 10000.50) * 10000 ≈ 1
        assert tick.spread_bps < Decimal("2")

    def test_tick_timestamp_conversion(self) -> None:
        """Test timestamp conversion to seconds."""
        ts_ns = 1704067200_000_000_000  # 2024-01-01 00:00:00
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000.50"),
            timestamp_ns=ts_ns,
        )

        assert tick.timestamp_sec == 1704067200.0


class TestOrderBook:
    """Tests for OrderBook struct."""

    def test_create_orderbook(self) -> None:
        """Test creating an order book."""
        orderbook = OrderBook(
            symbol="BTC/USDT",
            bids=[
                (Decimal("49999"), Decimal("1.5")),
                (Decimal("49998"), Decimal("2.0")),
            ],
            asks=[
                (Decimal("50001"), Decimal("1.0")),
                (Decimal("50002"), Decimal("3.0")),
            ],
            timestamp_ns=time.time_ns(),
        )

        assert orderbook.best_bid == Decimal("49999")
        assert orderbook.best_ask == Decimal("50001")
        assert orderbook.mid == Decimal("50000")
        assert orderbook.spread == Decimal("2")

    def test_empty_orderbook(self) -> None:
        """Test empty order book."""
        orderbook = OrderBook(
            symbol="BTC/USDT",
            bids=[],
            asks=[],
            timestamp_ns=time.time_ns(),
        )

        assert orderbook.best_bid is None
        assert orderbook.best_ask is None
        assert orderbook.mid is None
        assert orderbook.spread is None


class TestBalance:
    """Tests for Balance struct."""

    def test_create_balance(self) -> None:
        """Test creating a balance."""
        balance = Balance(
            currency="USDT",
            total=Decimal("10000.00"),
            available=Decimal("8000.00"),
            locked=Decimal("2000.00"),
        )

        assert balance.currency == "USDT"
        assert balance.used_percent == Decimal("20")

    def test_zero_balance(self) -> None:
        """Test zero balance doesn't divide by zero."""
        balance = Balance(
            currency="BTC",
            total=Decimal("0"),
            available=Decimal("0"),
            locked=Decimal("0"),
        )

        assert balance.used_percent == Decimal("0")


class TestGatewayProtocol:
    """Tests for Gateway Protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that Gateway protocol is runtime checkable."""
        gateway = PaperGateway()

        # Should pass isinstance check
        assert isinstance(gateway, Gateway)

    def test_protocol_check_non_gateway(self) -> None:
        """Test that non-gateway objects fail isinstance check."""

        class NotAGateway:
            pass

        obj = NotAGateway()
        assert not isinstance(obj, Gateway)


class TestEnums:
    """Tests for enum values."""

    def test_order_side_values(self) -> None:
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type_values(self) -> None:
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_order_status_values(self) -> None:
        """Test OrderStatus enum values."""
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"

    def test_time_in_force_values(self) -> None:
        """Test TimeInForce enum values."""
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"

"""Tests for TimeSeriesDB protocol and TradeRecord."""

from decimal import Decimal

import pytest

from libra.data.protocol import TradeRecord


class TestTradeRecord:
    """Tests for TradeRecord struct."""

    def test_create_trade_record(self) -> None:
        """Test creating a trade record."""
        trade = TradeRecord(
            trade_id="trade-123",
            order_id="order-456",
            symbol="BTC/USDT",
            exchange="binance",
            side="buy",
            amount=Decimal("0.5"),
            price=Decimal("50000.00"),
            fee=Decimal("0.0005"),
            fee_currency="BTC",
            timestamp_ns=1704067200000000000,
        )

        assert trade.trade_id == "trade-123"
        assert trade.order_id == "order-456"
        assert trade.symbol == "BTC/USDT"
        assert trade.exchange == "binance"
        assert trade.side == "buy"
        assert trade.amount == Decimal("0.5")
        assert trade.price == Decimal("50000.00")
        assert trade.fee == Decimal("0.0005")
        assert trade.fee_currency == "BTC"
        assert trade.timestamp_ns == 1704067200000000000

    def test_trade_record_optional_fields(self) -> None:
        """Test trade record with optional fields."""
        trade = TradeRecord(
            trade_id="trade-123",
            order_id="order-456",
            symbol="BTC/USDT",
            exchange="binance",
            side="sell",
            amount=Decimal("1.0"),
            price=Decimal("52000.00"),
            fee=Decimal("0.001"),
            fee_currency="BTC",
            timestamp_ns=1704067200000000000,
            strategy="sma_cross",
            signal_id="signal-789",
            realized_pnl=Decimal("100.00"),
            position_after=Decimal("0.0"),
            metadata={"slippage": 0.01},
        )

        assert trade.strategy == "sma_cross"
        assert trade.signal_id == "signal-789"
        assert trade.realized_pnl == Decimal("100.00")
        assert trade.position_after == Decimal("0.0")
        assert trade.metadata == {"slippage": 0.01}

    def test_trade_record_is_frozen(self) -> None:
        """Test that trade record is immutable."""
        trade = TradeRecord(
            trade_id="trade-123",
            order_id="order-456",
            symbol="BTC/USDT",
            exchange="binance",
            side="buy",
            amount=Decimal("0.5"),
            price=Decimal("50000.00"),
            fee=Decimal("0.0005"),
            fee_currency="BTC",
            timestamp_ns=1704067200000000000,
        )

        with pytest.raises(AttributeError):
            trade.price = Decimal("60000.00")  # type: ignore[misc]

    def test_trade_record_default_optional_fields(self) -> None:
        """Test default values for optional fields."""
        trade = TradeRecord(
            trade_id="trade-123",
            order_id="order-456",
            symbol="BTC/USDT",
            exchange="binance",
            side="buy",
            amount=Decimal("0.5"),
            price=Decimal("50000.00"),
            fee=Decimal("0.0005"),
            fee_currency="BTC",
            timestamp_ns=1704067200000000000,
        )

        assert trade.strategy is None
        assert trade.signal_id is None
        assert trade.realized_pnl is None
        assert trade.position_after is None
        assert trade.metadata is None
        assert trade.order_timestamp_ns is None

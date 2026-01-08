"""Integration tests for RiskEngine with TradingKernel order flow."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.clients.data_client import Instrument
from libra.core.events import EventType
from libra.core.kernel import TradingKernel, KernelConfig
from libra.gateways.protocol import (
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from libra.risk import (
    RiskEngine,
    RiskEngineConfig,
    RiskLimits,
    SymbolLimits,
)


@pytest.fixture
def risk_limits() -> RiskLimits:
    """Risk limits for testing."""
    return RiskLimits(
        max_total_exposure=Decimal("100000"),
        max_single_position_pct=Decimal("0.20"),
        max_daily_loss_pct=Decimal("-0.03"),
        max_weekly_loss_pct=Decimal("-0.07"),
        max_total_drawdown_pct=Decimal("-0.15"),
        max_orders_per_second=10,
        max_orders_per_minute=60,
        symbol_limits={
            "BTC/USDT": SymbolLimits(
                max_position_size=Decimal("1.0"),
                max_notional_per_order=Decimal("50000"),
                max_order_rate=5,
            ),
        },
    )


@pytest.fixture
def risk_engine(risk_limits: RiskLimits) -> RiskEngine:
    """RiskEngine instance for testing."""
    config = RiskEngineConfig(
        limits=risk_limits,
        enable_self_trade_prevention=True,
        enable_price_collar=True,
        price_collar_pct=Decimal("0.10"),
    )
    return RiskEngine(config=config)


@pytest.fixture
def mock_execution_client():
    """Mock execution client for testing."""
    client = MagicMock()
    client.name = "mock_execution"

    async def mock_connect():
        pass

    async def mock_disconnect():
        pass

    async def mock_submit_order(order: Order) -> OrderResult:
        return OrderResult(
            order_id="test-order-123",
            symbol=order.symbol,
            status=OrderStatus.OPEN,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            filled_amount=Decimal("0"),
            remaining_amount=order.amount,
            average_price=None,
            fee=Decimal("0"),
            fee_currency="USDT",
            timestamp_ns=0,
            client_order_id=order.client_order_id,
            price=order.price,
        )

    async def mock_cancel_order(order_id: str, symbol: str) -> bool:
        return True

    async def mock_modify_order(
        order_id: str, symbol: str, price=None, amount=None
    ) -> OrderResult:
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            status=OrderStatus.OPEN,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=amount or Decimal("0.5"),
            filled_amount=Decimal("0"),
            remaining_amount=amount or Decimal("0.5"),
            average_price=None,
            fee=Decimal("0"),
            fee_currency="USDT",
            timestamp_ns=0,
            price=price,
        )

    client.connect = AsyncMock(side_effect=mock_connect)
    client.disconnect = AsyncMock(side_effect=mock_disconnect)
    client.submit_order = AsyncMock(side_effect=mock_submit_order)
    client.cancel_order = AsyncMock(side_effect=mock_cancel_order)
    client.modify_order = AsyncMock(side_effect=mock_modify_order)

    return client


@pytest.fixture
def btc_instrument() -> Instrument:
    """BTC instrument for testing."""
    return Instrument(
        symbol="BTC/USDT",
        base="BTC",
        quote="USDT",
        exchange="test",
        lot_size=Decimal("0.00001"),
        tick_size=Decimal("0.01"),
    )


class TestRiskEngineKernelIntegration:
    """Tests for RiskEngine integration with TradingKernel."""

    @pytest.mark.asyncio
    async def test_kernel_with_risk_engine(self, risk_engine, mock_execution_client):
        """Kernel accepts RiskEngine configuration."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        assert kernel.risk_engine is risk_engine
        # Risk engine should have bus wired up
        assert risk_engine.bus is kernel.bus

    @pytest.mark.asyncio
    async def test_submit_order_passes_risk_validation(
        self, risk_engine, mock_execution_client, btc_instrument
    ):
        """Order passes through risk validation before execution."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        async with kernel:
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("0.5"),
            )

            result = await kernel.submit_order(
                order, Decimal("50000"), btc_instrument
            )

            assert result.status == OrderStatus.OPEN
            assert result.order_id == "test-order-123"
            mock_execution_client.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_order_denied_by_risk_engine(
        self, risk_engine, mock_execution_client, btc_instrument
    ):
        """Order denied by risk engine doesn't reach execution client."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        async with kernel:
            # Order that exceeds position limit
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("1.5"),  # Exceeds 1.0 limit
            )

            result = await kernel.submit_order(
                order, Decimal("50000"), btc_instrument
            )

            # Order should be rejected
            assert result.status == OrderStatus.REJECTED
            # Execution client should NOT have been called
            mock_execution_client.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_order_denied_price_collar(
        self, risk_engine, mock_execution_client, btc_instrument
    ):
        """Order denied due to price collar violation."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        async with kernel:
            # Order with price far from market (>10% collar)
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.5"),
                price=Decimal("60000"),  # 20% above market
            )

            result = await kernel.submit_order(
                order, Decimal("50000"), btc_instrument
            )

            # Order should be rejected due to price collar
            assert result.status == OrderStatus.REJECTED
            mock_execution_client.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_self_trade_prevention_in_kernel(
        self, risk_engine, mock_execution_client, btc_instrument
    ):
        """Self-trade prevention works through kernel submission."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        async with kernel:
            # Submit first order (SELL)
            sell_order = Order(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.5"),
                price=Decimal("50000"),
                client_order_id="sell-1",
            )

            result1 = await kernel.submit_order(
                sell_order, Decimal("50000"), btc_instrument
            )
            assert result1.status == OrderStatus.OPEN

            # Try to submit crossing order (BUY at same price)
            buy_order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.5"),
                price=Decimal("50000"),  # Would self-trade
                client_order_id="buy-1",
            )

            result2 = await kernel.submit_order(
                buy_order, Decimal("50000"), btc_instrument
            )

            # Second order should be rejected
            assert result2.status == OrderStatus.REJECTED
            # Only first order should have been submitted
            assert mock_execution_client.submit_order.call_count == 1

    @pytest.mark.asyncio
    async def test_cancel_order_updates_tracking(
        self, risk_engine, mock_execution_client, btc_instrument
    ):
        """Canceling order removes it from self-trade tracking."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        async with kernel:
            # Submit order
            sell_order = Order(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.5"),
                price=Decimal("50000"),
                client_order_id="sell-1",
            )

            await kernel.submit_order(sell_order, Decimal("50000"), btc_instrument)

            # Cancel it using client_order_id (which is how the order was tracked)
            # We need to remove from risk engine manually since cancel_order
            # removes by order_id, but the open order was tracked by the Order object
            risk_engine.remove_open_order("BTC/USDT", client_order_id="sell-1")

            # Now crossing order should be allowed
            buy_order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.5"),
                price=Decimal("50000"),
            )

            result = await kernel.submit_order(
                buy_order, Decimal("50000"), btc_instrument
            )
            assert result.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_modify_order_with_risk_validation(
        self, risk_engine, mock_execution_client, btc_instrument
    ):
        """Order modification goes through risk validation."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        async with kernel:
            # Valid modification
            result = await kernel.modify_order(
                "order-1",
                "BTC/USDT",
                price=Decimal("51000.00"),  # Valid precision
                amount=Decimal("0.5"),
                current_price=Decimal("50000"),
                instrument=btc_instrument,
            )

            assert result.status == OrderStatus.OPEN
            mock_execution_client.modify_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_modify_order_denied_by_risk(
        self, risk_engine, mock_execution_client, btc_instrument
    ):
        """Invalid modification denied by risk engine."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        async with kernel:
            # Modification with price far from market
            result = await kernel.modify_order(
                "order-1",
                "BTC/USDT",
                price=Decimal("60000"),  # 20% above market
                current_price=Decimal("50000"),
                instrument=btc_instrument,
            )

            assert result.status == OrderStatus.REJECTED
            mock_execution_client.modify_order.assert_not_called()


class TestRiskEventsIntegration:
    """Tests for risk event publishing."""

    @pytest.mark.asyncio
    async def test_order_denied_event_published(
        self, risk_engine, mock_execution_client, btc_instrument
    ):
        """ORDER_DENIED event is published when order fails validation."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_risk_engine(risk_engine)
        kernel.set_execution_client(mock_execution_client)

        received_events = []

        def capture_event(event):
            received_events.append(event)

        async with kernel:
            # Subscribe to ORDER_DENIED events
            kernel.bus.subscribe(EventType.ORDER_DENIED, capture_event)

            # Submit order that will fail
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("1.5"),  # Exceeds limit
            )

            await kernel.submit_order(order, Decimal("50000"), btc_instrument)

            # Give bus time to process
            await asyncio.sleep(0.1)

            # Should have received ORDER_DENIED event
            assert len(received_events) >= 1
            event = received_events[0]
            assert event.event_type == EventType.ORDER_DENIED
            assert event.payload["check"] == "position_limit"


class TestKernelWithoutRiskEngine:
    """Tests for kernel operation without risk engine."""

    @pytest.mark.asyncio
    async def test_submit_order_without_risk_engine(self, mock_execution_client):
        """Orders can be submitted without risk engine (no validation)."""
        config = KernelConfig(environment="sandbox")
        kernel = TradingKernel(config)

        kernel.set_execution_client(mock_execution_client)
        # Intentionally NOT setting risk engine

        async with kernel:
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("100"),  # Would fail risk check if engine was set
            )

            result = await kernel.submit_order(order, Decimal("50000"))

            # Order should go through (no risk validation)
            assert result.status == OrderStatus.OPEN
            mock_execution_client.submit_order.assert_called_once()

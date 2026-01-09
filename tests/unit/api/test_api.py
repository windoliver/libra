"""
Unit tests for LIBRA REST API (Issue #30).

Tests cover:
- Authentication (JWT and API key)
- Strategy CRUD and control
- Position queries
- Order management
- Market data endpoints
- System health and metrics
"""

from __future__ import annotations

from decimal import Decimal

import pytest


# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None


pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed. Install with: pip install fastapi[standard]",
)


@pytest.fixture
def client():
    """Create test client."""
    from libra.api.main import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Get authentication headers."""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestRoot:
    """Test root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "LIBRA Trading Platform API"
        assert "version" in data
        assert "docs" in data


class TestAuthentication:
    """Test authentication endpoints."""

    def test_login_success(self, client):
        """Test successful login."""
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_login_wrong_password(self, client):
        """Test login with wrong password."""
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "wrong"},
        )
        assert response.status_code == 401
        assert "detail" in response.json()

    def test_login_unknown_user(self, client):
        """Test login with unknown user."""
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "unknown", "password": "password"},
        )
        assert response.status_code == 401

    def test_get_current_user(self, client, auth_headers):
        """Test get current user endpoint."""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert "id" in data
        assert "email" in data

    def test_get_current_user_no_auth(self, client):
        """Test get current user without auth fails."""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

    def test_create_api_key(self, client, auth_headers):
        """Test API key creation."""
        response = client.post(
            "/api/v1/auth/api-key?name=test_key",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert data["name"] == "test_key"


class TestStrategies:
    """Test strategy endpoints."""

    def test_list_strategies(self, client, auth_headers):
        """Test listing strategies."""
        response = client.get("/api/v1/strategies", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert "total" in data
        assert isinstance(data["strategies"], list)

    def test_list_strategies_no_auth(self, client):
        """Test listing strategies without auth fails."""
        response = client.get("/api/v1/strategies")
        assert response.status_code == 401

    def test_create_strategy(self, client, auth_headers):
        """Test creating a strategy."""
        strategy_data = {
            "name": "Test Strategy",
            "description": "A test strategy",
            "strategy_type": "sma_cross",
            "symbols": ["BTC/USDT"],
            "parameters": {"fast_period": 10, "slow_period": 20},
        }
        response = client.post(
            "/api/v1/strategies",
            json=strategy_data,
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Strategy"
        assert data["strategy_type"] == "sma_cross"
        assert data["status"] == "stopped"
        assert "id" in data

    def test_get_strategy(self, client, auth_headers):
        """Test getting strategy details."""
        # First list to get an ID
        list_response = client.get("/api/v1/strategies", headers=auth_headers)
        strategies = list_response.json()["strategies"]

        if strategies:
            strategy_id = strategies[0]["id"]
            response = client.get(
                f"/api/v1/strategies/{strategy_id}",
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert response.json()["id"] == strategy_id

    def test_get_strategy_not_found(self, client, auth_headers):
        """Test getting non-existent strategy."""
        response = client.get(
            "/api/v1/strategies/nonexistent",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_update_strategy(self, client, auth_headers):
        """Test updating a strategy."""
        # Create a strategy first
        create_response = client.post(
            "/api/v1/strategies",
            json={
                "name": "Update Test",
                "strategy_type": "test",
                "symbols": [],
                "parameters": {},
            },
            headers=auth_headers,
        )
        strategy_id = create_response.json()["id"]

        # Update it
        response = client.put(
            f"/api/v1/strategies/{strategy_id}",
            json={"name": "Updated Name"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"

    def test_start_stop_strategy(self, client, auth_headers):
        """Test starting and stopping a strategy."""
        # Create a strategy
        create_response = client.post(
            "/api/v1/strategies",
            json={
                "name": "Start Stop Test",
                "strategy_type": "test",
                "symbols": [],
                "parameters": {},
            },
            headers=auth_headers,
        )
        strategy_id = create_response.json()["id"]

        # Start it
        start_response = client.post(
            f"/api/v1/strategies/{strategy_id}/start",
            headers=auth_headers,
        )
        assert start_response.status_code == 200
        assert start_response.json()["status"] == "running"

        # Stop it
        stop_response = client.post(
            f"/api/v1/strategies/{strategy_id}/stop",
            headers=auth_headers,
        )
        assert stop_response.status_code == 200
        assert stop_response.json()["status"] == "stopped"

    def test_delete_strategy(self, client, auth_headers):
        """Test deleting a strategy."""
        # Create a strategy
        create_response = client.post(
            "/api/v1/strategies",
            json={
                "name": "Delete Test",
                "strategy_type": "test",
                "symbols": [],
                "parameters": {},
            },
            headers=auth_headers,
        )
        strategy_id = create_response.json()["id"]

        # Delete it
        response = client.delete(
            f"/api/v1/strategies/{strategy_id}",
            headers=auth_headers,
        )
        assert response.status_code == 204

        # Verify it's gone
        get_response = client.get(
            f"/api/v1/strategies/{strategy_id}",
            headers=auth_headers,
        )
        assert get_response.status_code == 404


class TestPositions:
    """Test position endpoints."""

    def test_list_positions(self, client, auth_headers):
        """Test listing positions."""
        response = client.get("/api/v1/positions", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert "total_unrealized_pnl" in data
        assert "total_realized_pnl" in data

    def test_get_position(self, client, auth_headers):
        """Test getting a specific position."""
        response = client.get("/api/v1/positions/BTC-USDT", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert "size" in data
        assert "entry_price" in data

    def test_get_position_not_found(self, client, auth_headers):
        """Test getting non-existent position."""
        response = client.get("/api/v1/positions/UNKNOWN", headers=auth_headers)
        assert response.status_code == 404


class TestOrders:
    """Test order endpoints."""

    def test_list_orders(self, client, auth_headers):
        """Test listing orders."""
        response = client.get("/api/v1/orders", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "orders" in data
        assert "total" in data

    def test_create_market_order(self, client, auth_headers):
        """Test creating a market order."""
        order_data = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "market",
            "quantity": "0.1",
        }
        response = client.post(
            "/api/v1/orders",
            json=order_data,
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert data["side"] == "buy"
        assert data["order_type"] == "market"
        assert "id" in data

    def test_create_limit_order(self, client, auth_headers):
        """Test creating a limit order."""
        order_data = {
            "symbol": "ETH/USDT",
            "side": "sell",
            "order_type": "limit",
            "quantity": "1.0",
            "price": "2500.00",
        }
        response = client.post(
            "/api/v1/orders",
            json=order_data,
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["order_type"] == "limit"
        assert data["price"] is not None

    def test_create_limit_order_without_price_fails(self, client, auth_headers):
        """Test creating limit order without price fails."""
        order_data = {
            "symbol": "ETH/USDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": "1.0",
        }
        response = client.post(
            "/api/v1/orders",
            json=order_data,
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_cancel_order(self, client, auth_headers):
        """Test cancelling an order."""
        # Create an order
        order_data = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": "0.1",
            "price": "40000.00",
        }
        create_response = client.post(
            "/api/v1/orders",
            json=order_data,
            headers=auth_headers,
        )
        order_id = create_response.json()["id"]

        # Cancel it
        response = client.delete(
            f"/api/v1/orders/{order_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is True


class TestMarketData:
    """Test market data endpoints."""

    def test_list_symbols(self, client):
        """Test listing available symbols (public endpoint)."""
        response = client.get("/api/v1/market/symbols")
        assert response.status_code == 200
        data = response.json()
        assert "symbols" in data
        assert "total" in data
        assert len(data["symbols"]) > 0

    def test_get_quote(self, client):
        """Test getting a quote (public endpoint)."""
        response = client.get("/api/v1/market/quotes/BTC-USDT")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert "bid" in data
        assert "ask" in data
        assert "last" in data

    def test_get_quote_not_found(self, client):
        """Test getting quote for unknown symbol."""
        response = client.get("/api/v1/market/quotes/UNKNOWN")
        assert response.status_code == 404

    def test_get_bars(self, client):
        """Test getting OHLCV bars (public endpoint)."""
        response = client.get("/api/v1/market/bars/BTC-USDT?timeframe=1h&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert len(data["bars"]) == 10
        # Check bar structure
        bar = data["bars"][0]
        assert "open" in bar
        assert "high" in bar
        assert "low" in bar
        assert "close" in bar
        assert "volume" in bar


class TestSystemEndpoints:
    """Test system endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint (public)."""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_metrics(self, client):
        """Test metrics endpoint (public)."""
        response = client.get("/api/v1/system/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "metrics" in data

    def test_system_info(self, client):
        """Test system info endpoint (public)."""
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "LIBRA Trading Platform API"
        assert "version" in data
        assert "uptime_seconds" in data


class TestWebSocket:
    """Test WebSocket endpoint."""

    def test_websocket_connect(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive welcome message
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert "timestamp" in data

    def test_websocket_subscribe(self, client):
        """Test WebSocket subscription."""
        with client.websocket_connect("/ws") as websocket:
            # Receive welcome
            websocket.receive_json()

            # Subscribe to a channel
            websocket.send_json({
                "action": "subscribe",
                "channels": ["quotes:BTC/USDT"],
            })

            # Should receive subscription confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert "quotes:BTC/USDT" in data["channels"]

    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong."""
        with client.websocket_connect("/ws") as websocket:
            # Receive welcome
            websocket.receive_json()

            # Send ping
            websocket.send_json({"action": "ping"})

            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"


class TestRoleBasedAccess:
    """Test role-based access control."""

    def test_viewer_cannot_create_strategy(self, client):
        """Test that viewer role cannot create strategies."""
        # Login as viewer
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "viewer", "password": "viewer123"},
        )
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Try to create strategy
        response = client.post(
            "/api/v1/strategies",
            json={
                "name": "Test",
                "strategy_type": "test",
                "symbols": [],
                "parameters": {},
            },
            headers=headers,
        )
        assert response.status_code == 403

    def test_viewer_can_list_strategies(self, client):
        """Test that viewer role can list strategies."""
        # Login as viewer
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "viewer", "password": "viewer123"},
        )
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # List strategies should work
        response = client.get("/api/v1/strategies", headers=headers)
        assert response.status_code == 200

    def test_trader_can_create_order(self, client):
        """Test that trader role can create orders."""
        # Login as trader
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "trader", "password": "trader123"},
        )
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create order should work
        response = client.post(
            "/api/v1/orders",
            json={
                "symbol": "BTC/USDT",
                "side": "buy",
                "order_type": "market",
                "quantity": "0.1",
            },
            headers=headers,
        )
        assert response.status_code == 201

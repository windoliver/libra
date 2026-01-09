"""
FastAPI Application for LIBRA Trading Platform (Issue #30).

Main entry point for the REST API server.

Usage:
    uvicorn libra.api.main:app --reload
    # or
    python -m libra.api.main
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from libra.api.deps import check_rate_limit, get_optional_user
from libra.api.routers import (
    auth_router,
    market_router,
    orders_router,
    positions_router,
    strategies_router,
    system_router,
)


# =============================================================================
# WebSocket Connection Manager
# =============================================================================


class ConnectionManager:
    """Manage WebSocket connections and subscriptions."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self.subscriptions: dict[WebSocket, set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]

    async def subscribe(self, websocket: WebSocket, channels: list[str]) -> None:
        """Subscribe a connection to channels."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(channels)

    async def unsubscribe(self, websocket: WebSocket, channels: list[str]) -> None:
        """Unsubscribe a connection from channels."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].difference_update(channels)

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Broadcast message to all connections subscribed to channel."""
        message = json.dumps(
            {
                "type": "update",
                "channel": channel,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        for ws, channels in list(self.subscriptions.items()):
            if channel in channels:
                try:
                    await ws.send_text(message)
                except Exception:
                    self.disconnect(ws)


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("LIBRA API starting up...")

    # Start background tasks (e.g., market data simulation)
    task = asyncio.create_task(market_data_broadcaster())

    yield

    # Shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    print("LIBRA API shutting down...")


async def market_data_broadcaster() -> None:
    """Background task to broadcast simulated market data."""
    import random

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    while True:
        try:
            await asyncio.sleep(1)  # Broadcast every second

            for symbol in symbols:
                if any("market:" in ch or f"quotes:{symbol}" in ch for ch in
                       [ch for channels in manager.subscriptions.values() for ch in channels]):
                    # Generate simulated price update
                    data = {
                        "symbol": symbol,
                        "price": round(random.uniform(40000, 50000) if symbol == "BTC/USDT"
                                       else random.uniform(2000, 3000) if symbol == "ETH/USDT"
                                       else random.uniform(80, 120), 2),
                        "change_24h": round(random.uniform(-5, 5), 2),
                    }
                    await manager.broadcast(f"quotes:{symbol}", data)
                    await manager.broadcast("market:all", data)

        except asyncio.CancelledError:
            break
        except Exception:
            pass  # Ignore broadcast errors


# =============================================================================
# Create FastAPI Application
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LIBRA Trading Platform API",
        description="""
## LIBRA REST API

A comprehensive API for algorithmic trading operations.

### Features
- **Strategies**: Manage and control trading strategies
- **Positions**: Query portfolio positions
- **Orders**: Create and manage orders
- **Market Data**: Real-time quotes and historical bars
- **Authentication**: JWT tokens and API keys

### Authentication

Use one of these methods:
1. **Bearer Token**: Get a token via POST /api/v1/auth/token
2. **API Key**: Pass key in X-API-Key header

### Demo Credentials
- admin / admin123 (full access)
- trader / trader123 (read/write)
- viewer / viewer123 (read only)
        """,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(
        auth_router,
        prefix="/api/v1/auth",
        tags=["Authentication"],
    )

    app.include_router(
        strategies_router,
        prefix="/api/v1/strategies",
        tags=["Strategies"],
        dependencies=[Depends(check_rate_limit)],
    )

    app.include_router(
        positions_router,
        prefix="/api/v1/positions",
        tags=["Positions"],
        dependencies=[Depends(check_rate_limit)],
    )

    app.include_router(
        orders_router,
        prefix="/api/v1/orders",
        tags=["Orders"],
        dependencies=[Depends(check_rate_limit)],
    )

    app.include_router(
        market_router,
        prefix="/api/v1/market",
        tags=["Market Data"],
        dependencies=[Depends(check_rate_limit)],
    )

    app.include_router(
        system_router,
        prefix="/api/v1/system",
        tags=["System"],
    )

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            {
                "name": "LIBRA Trading Platform API",
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/api/v1/system/health",
            }
        )

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
    ):
        """
        WebSocket endpoint for real-time updates.

        **Subscription format:**
        ```json
        {"action": "subscribe", "channels": ["quotes:BTC/USDT", "orders", "positions"]}
        {"action": "unsubscribe", "channels": ["quotes:BTC/USDT"]}
        ```

        **Available channels:**
        - quotes:{symbol} - Price updates for a symbol
        - market:all - All market updates
        - orders - Order updates
        - positions - Position updates
        - strategies - Strategy status updates
        """
        await manager.connect(websocket)

        try:
            # Send welcome message
            await websocket.send_json(
                {
                    "type": "connected",
                    "message": "Connected to LIBRA WebSocket",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            while True:
                # Receive and process messages
                data = await websocket.receive_json()

                action = data.get("action")
                channels = data.get("channels", [])

                if action == "subscribe":
                    await manager.subscribe(websocket, channels)
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "channels": channels,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                elif action == "unsubscribe":
                    await manager.unsubscribe(websocket, channels)
                    await websocket.send_json(
                        {
                            "type": "unsubscribed",
                            "channels": channels,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                elif action == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    return app


# Create the app instance
app = create_app()


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "libra.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

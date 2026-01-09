"""
API Routers for LIBRA REST API (Issue #30).

Each router handles a specific domain:
- auth: Authentication and user management
- strategies: Strategy CRUD and control
- positions: Position queries
- orders: Order management
- market: Market data
- system: Health and metrics
"""

from libra.api.routers.auth import router as auth_router
from libra.api.routers.market import router as market_router
from libra.api.routers.orders import router as orders_router
from libra.api.routers.positions import router as positions_router
from libra.api.routers.strategies import router as strategies_router
from libra.api.routers.system import router as system_router

__all__ = [
    "auth_router",
    "strategies_router",
    "positions_router",
    "orders_router",
    "market_router",
    "system_router",
]

"""
LIBRA REST API for External Integrations (Issue #30).

Provides a FastAPI-based REST API following OpenBB patterns:
- /api/v1/strategies - Strategy management
- /api/v1/positions - Position queries
- /api/v1/orders - Order management
- /api/v1/market - Market data
- /api/v1/ws - WebSocket real-time updates

Usage:
    # Run with uvicorn
    uvicorn libra.api.main:app --reload

    # Or programmatically
    from libra.api import create_app
    app = create_app()
"""

from libra.api.main import app, create_app

__all__ = ["app", "create_app"]

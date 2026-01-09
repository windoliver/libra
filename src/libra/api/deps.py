"""
FastAPI Dependencies for LIBRA API (Issue #30).

Provides dependency injection for:
- Authentication (JWT/API Key)
- Rate limiting
- Database sessions (future)
- Service instances
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Header, Security, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer

from libra.api.auth import decode_access_token, validate_api_key
from libra.api.schemas import TokenData


# =============================================================================
# Authentication Dependencies
# =============================================================================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> dict[str, Any]:
    """
    Get current authenticated user from JWT token or API key.

    Supports both authentication methods:
    - Bearer token (OAuth2)
    - API key header (X-API-Key)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Try JWT token first
    if token:
        token_data = decode_access_token(token)
        if token_data and token_data.username:
            from libra.api.auth import get_user

            user = get_user(token_data.username)
            if user:
                return user

    # Try API key
    if api_key:
        key_data = validate_api_key(api_key)
        if key_data:
            from libra.api.auth import get_user

            # API keys are associated with users
            # For simplicity, return admin user for valid API keys
            return get_user("admin") or {"id": "api_user", "scopes": ["read", "write"]}

    raise credentials_exception


async def get_current_active_user(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """Get current user and verify they are active."""
    if not current_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


def require_scope(required_scope: str):
    """
    Dependency factory for scope-based authorization.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_scope("admin"))])
    """

    async def check_scope(
        current_user: Annotated[dict[str, Any], Depends(get_current_active_user)],
    ) -> dict[str, Any]:
        user_scopes = current_user.get("scopes", [])
        if required_scope not in user_scopes and "admin" not in user_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required scope: {required_scope}",
            )
        return current_user

    return check_scope


# =============================================================================
# Optional Authentication (for public endpoints with enhanced features)
# =============================================================================


async def get_optional_user(
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> dict[str, Any] | None:
    """Get current user if authenticated, None otherwise."""
    try:
        return await get_current_user(token, api_key)
    except HTTPException:
        return None


# =============================================================================
# Rate Limiting
# =============================================================================

# In-memory rate limit tracking
_rate_limits: dict[str, list[float]] = defaultdict(list)
_rate_limit_window = 60  # seconds
_rate_limit_max_requests = 100  # requests per window


async def check_rate_limit(
    x_forwarded_for: Annotated[str | None, Header()] = None,
    x_real_ip: Annotated[str | None, Header()] = None,
) -> None:
    """
    Simple rate limiting based on IP address.

    Limits to 100 requests per minute per IP.
    """
    # Get client IP
    client_ip = x_forwarded_for or x_real_ip or "unknown"
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    current_time = time.time()
    window_start = current_time - _rate_limit_window

    # Clean old entries
    _rate_limits[client_ip] = [
        t for t in _rate_limits[client_ip] if t > window_start
    ]

    # Check limit
    if len(_rate_limits[client_ip]) >= _rate_limit_max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(_rate_limit_window)},
        )

    # Record request
    _rate_limits[client_ip].append(current_time)


# =============================================================================
# Service Dependencies
# =============================================================================

# Global service instances (initialized on startup)
_services: dict[str, Any] = {}


def get_message_bus():
    """Get MessageBus instance."""
    if "message_bus" not in _services:
        from libra.core.message_bus import MessageBus

        _services["message_bus"] = MessageBus()
    return _services["message_bus"]


def get_metrics_collector():
    """Get MetricsCollector instance."""
    if "metrics_collector" not in _services:
        from libra.observability import get_collector

        _services["metrics_collector"] = get_collector()
    return _services["metrics_collector"]


def get_health_monitor():
    """Get HealthMonitor instance."""
    if "health_monitor" not in _services:
        from libra.observability import get_monitor

        _services["health_monitor"] = get_monitor()
    return _services["health_monitor"]


def set_service(name: str, service: Any) -> None:
    """Set a service instance (for testing/configuration)."""
    _services[name] = service


def clear_services() -> None:
    """Clear all service instances (for testing)."""
    _services.clear()

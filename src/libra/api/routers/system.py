"""
System Router for LIBRA API (Issue #30).

Endpoints:
- GET /health - Health check
- GET /metrics - System metrics
- GET /info - System information
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from libra.api.deps import get_health_monitor, get_metrics_collector, get_optional_user


router = APIRouter()

# Track API start time
_start_time = datetime.now(timezone.utc)


@router.get("/health")
async def health_check(
    _current_user: Annotated[dict | None, Depends(get_optional_user)] = None,
) -> dict[str, Any]:
    """
    Health check endpoint.

    Returns system health status. This endpoint is public.
    """
    try:
        health_monitor = get_health_monitor()
        status = health_monitor.check_health()
        return {
            "status": "healthy" if status.get("healthy", True) else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": status.get("components", {}),
        }
    except Exception:
        # Fallback if health monitor not available
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {},
        }


@router.get("/metrics")
async def get_metrics(
    _current_user: Annotated[dict | None, Depends(get_optional_user)] = None,
) -> dict[str, Any]:
    """
    Get system metrics.

    Returns current metrics snapshot. This endpoint is public.
    """
    try:
        collector = get_metrics_collector()
        metrics = collector.get_snapshot()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
    except Exception:
        # Fallback if metrics collector not available
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "api_requests_total": 0,
                "api_errors_total": 0,
            },
        }


@router.get("/info")
async def system_info(
    _current_user: Annotated[dict | None, Depends(get_optional_user)] = None,
) -> dict[str, Any]:
    """
    Get system information.

    Returns API version, Python version, and uptime. This endpoint is public.
    """
    uptime = datetime.now(timezone.utc) - _start_time

    return {
        "name": "LIBRA Trading Platform API",
        "version": "0.1.0",
        "python_version": sys.version,
        "platform": platform.platform(),
        "uptime_seconds": int(uptime.total_seconds()),
        "started_at": _start_time.isoformat(),
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }

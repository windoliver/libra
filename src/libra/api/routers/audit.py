"""
Audit Router for LIBRA API (Issue #16).

Endpoints:
- GET /audit/events - List audit events with filtering
- GET /audit/events/{id} - Get event details
- GET /audit/orders - List order audit trails
- GET /audit/orders/{id} - Get order audit details
- GET /audit/risk - List risk audit events
- GET /audit/export - Export audit logs
- GET /audit/statistics - Get audit statistics
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from libra.api.deps import get_current_active_user, require_scope


router = APIRouter()


# === Schemas ===


class AuditEventTypeSchema(str, Enum):
    """Audit event types."""

    ORDER_CREATED = "order.created"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_ACCEPTED = "order.accepted"
    ORDER_REJECTED = "order.rejected"
    ORDER_FILLED = "order.filled"
    ORDER_PARTIALLY_FILLED = "order.partially_filled"
    ORDER_CANCELLED = "order.cancelled"
    RISK_CHECK_PASSED = "risk.check_passed"
    RISK_CHECK_FAILED = "risk.check_failed"
    RISK_LIMIT_BREACH = "risk.limit_breach"
    CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker"
    AGENT_DECISION = "agent.decision"
    AGENT_SIGNAL = "agent.signal"
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    ERROR = "system.error"


class AuditSeveritySchema(str, Enum):
    """Audit severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ExportFormatSchema(str, Enum):
    """Export formats."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    SEC = "sec"
    CFTC = "cftc"
    FCA = "fca"


class AuditEventResponse(BaseModel):
    """Response schema for audit event."""

    audit_id: str
    event_type: str
    severity: str
    timestamp: datetime
    sequence_number: int = 0
    source: str = ""
    actor: str = ""
    session_id: str = ""
    trace_id: str = ""
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    checksum: str = ""


class AuditEventListResponse(BaseModel):
    """Response schema for audit event list."""

    events: list[AuditEventResponse]
    total: int
    limit: int
    offset: int


class OrderAuditResponse(BaseModel):
    """Response schema for order audit trail."""

    order_id: str
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: str = "0"
    price: str | None = None
    status: str = "pending"
    strategy_name: str = ""
    risk_check_passed: bool = True
    filled_quantity: str = "0"
    filled_price: str = "0"
    commission: str = "0"
    slippage: str = "0"
    created_at: datetime
    filled_at: datetime | None = None
    session_id: str = ""
    trace_id: str = ""


class OrderAuditListResponse(BaseModel):
    """Response schema for order audit list."""

    orders: list[OrderAuditResponse]
    total: int


class RiskAuditResponse(BaseModel):
    """Response schema for risk audit trail."""

    risk_event_id: str
    order_id: str = ""
    check_name: str
    check_passed: bool
    check_reason: str = ""
    current_value: str = "0"
    limit_value: str = "0"
    utilization_pct: float = 0.0
    symbol: str = ""
    strategy_id: str = ""
    timestamp: datetime
    trace_id: str = ""


class RiskAuditListResponse(BaseModel):
    """Response schema for risk audit list."""

    risk_events: list[RiskAuditResponse]
    total: int


class ExportResponse(BaseModel):
    """Response schema for export operation."""

    success: bool
    format: str
    record_count: int
    exported_at: datetime


class AuditStatisticsResponse(BaseModel):
    """Response schema for audit statistics."""

    total_events: int
    total_orders: int
    events_by_type: dict[str, int]
    events_by_severity: dict[str, int]
    hot_files: int
    warm_files: int
    cold_files: int
    storage_size_mb: float


# === Demo Data ===


def _generate_demo_events() -> list[dict[str, Any]]:
    """Generate demo audit events."""
    now = datetime.now(timezone.utc)
    events = []

    event_types = [
        ("order.created", "info", "Order ORD-001 created for BTC/USDT"),
        ("risk.check_passed", "info", "Position limit check passed"),
        ("order.submitted", "info", "Order ORD-001 submitted to exchange"),
        ("order.filled", "info", "Order ORD-001 filled at 42500.00"),
        ("agent.decision", "info", "Agent decided to buy based on momentum signal"),
        ("order.created", "info", "Order ORD-002 created for ETH/USDT"),
        ("risk.check_failed", "warning", "Notional limit check failed for ORD-003"),
        ("order.rejected", "warning", "Order ORD-003 rejected: risk limit exceeded"),
        ("system.error", "error", "Exchange connection timeout"),
        ("risk.circuit_breaker", "critical", "Circuit breaker triggered: max drawdown"),
    ]

    for i, (event_type, severity, message) in enumerate(event_types):
        events.append({
            "audit_id": str(uuid4()),
            "event_type": event_type,
            "severity": severity,
            "timestamp": (now - timedelta(minutes=i * 5)).isoformat(),
            "sequence_number": i + 1,
            "source": "trading_engine",
            "actor": "system",
            "session_id": "session_001",
            "trace_id": f"trace_{i:03d}",
            "message": message,
            "details": {},
            "checksum": f"check_{i:08x}",
        })

    return events


def _generate_demo_orders() -> list[dict[str, Any]]:
    """Generate demo order audit trails."""
    now = datetime.now(timezone.utc)

    orders = [
        {
            "order_id": "ORD-001",
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": "0.5",
            "price": "42500.00",
            "status": "filled",
            "strategy_name": "momentum_v1",
            "risk_check_passed": True,
            "filled_quantity": "0.5",
            "filled_price": "42485.00",
            "commission": "10.62",
            "slippage": "-15.00",
            "created_at": now - timedelta(hours=2),
            "filled_at": now - timedelta(hours=1, minutes=55),
            "session_id": "session_001",
            "trace_id": "trace_001",
        },
        {
            "order_id": "ORD-002",
            "symbol": "ETH/USDT",
            "side": "buy",
            "order_type": "market",
            "quantity": "5.0",
            "price": None,
            "status": "filled",
            "strategy_name": "mean_reversion",
            "risk_check_passed": True,
            "filled_quantity": "5.0",
            "filled_price": "2205.50",
            "commission": "11.03",
            "slippage": "5.50",
            "created_at": now - timedelta(hours=1),
            "filled_at": now - timedelta(minutes=58),
            "session_id": "session_001",
            "trace_id": "trace_002",
        },
        {
            "order_id": "ORD-003",
            "symbol": "SOL/USDT",
            "side": "sell",
            "order_type": "limit",
            "quantity": "100.0",
            "price": "95.00",
            "status": "rejected",
            "strategy_name": "momentum_v1",
            "risk_check_passed": False,
            "filled_quantity": "0",
            "filled_price": "0",
            "commission": "0",
            "slippage": "0",
            "created_at": now - timedelta(minutes=30),
            "filled_at": None,
            "session_id": "session_001",
            "trace_id": "trace_003",
        },
    ]

    return orders


def _generate_demo_risk_events() -> list[dict[str, Any]]:
    """Generate demo risk audit events."""
    now = datetime.now(timezone.utc)

    risk_events = [
        {
            "risk_event_id": str(uuid4()),
            "order_id": "ORD-001",
            "check_name": "position_limit",
            "check_passed": True,
            "check_reason": "Within position limits",
            "current_value": "0.5",
            "limit_value": "2.0",
            "utilization_pct": 25.0,
            "symbol": "BTC/USDT",
            "strategy_id": "momentum_v1",
            "timestamp": now - timedelta(hours=2),
            "trace_id": "trace_001",
        },
        {
            "risk_event_id": str(uuid4()),
            "order_id": "ORD-002",
            "check_name": "notional_limit",
            "check_passed": True,
            "check_reason": "Within notional limits",
            "current_value": "11027.50",
            "limit_value": "50000.00",
            "utilization_pct": 22.05,
            "symbol": "ETH/USDT",
            "strategy_id": "mean_reversion",
            "timestamp": now - timedelta(hours=1),
            "trace_id": "trace_002",
        },
        {
            "risk_event_id": str(uuid4()),
            "order_id": "ORD-003",
            "check_name": "notional_limit",
            "check_passed": False,
            "check_reason": "Order would exceed notional limit",
            "current_value": "59500.00",
            "limit_value": "50000.00",
            "utilization_pct": 119.0,
            "symbol": "SOL/USDT",
            "strategy_id": "momentum_v1",
            "timestamp": now - timedelta(minutes=30),
            "trace_id": "trace_003",
        },
    ]

    return risk_events


# === Endpoints ===


@router.get("/events", response_model=AuditEventListResponse)
async def list_audit_events(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    start_time: datetime | None = Query(None, description="Filter events after this time"),
    end_time: datetime | None = Query(None, description="Filter events before this time"),
    event_type: AuditEventTypeSchema | None = Query(None, description="Filter by event type"),
    severity: AuditSeveritySchema | None = Query(None, description="Filter by severity"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    trace_id: str | None = Query(None, description="Filter by trace ID"),
    source: str | None = Query(None, description="Filter by source"),
    search: str | None = Query(None, description="Search in message"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> AuditEventListResponse:
    """
    List audit events with filtering.

    Supports filtering by time range, event type, severity, and more.
    Requires read access.
    """
    events = _generate_demo_events()

    # Apply filters
    if event_type:
        events = [e for e in events if e["event_type"] == event_type.value]

    if severity:
        events = [e for e in events if e["severity"] == severity.value]

    if session_id:
        events = [e for e in events if e["session_id"] == session_id]

    if trace_id:
        events = [e for e in events if e["trace_id"] == trace_id]

    if source:
        events = [e for e in events if e["source"] == source]

    if search:
        events = [e for e in events if search.lower() in e["message"].lower()]

    if start_time:
        events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= start_time]

    if end_time:
        events = [e for e in events if datetime.fromisoformat(e["timestamp"]) <= end_time]

    total = len(events)
    events = events[offset : offset + limit]

    # Convert to response objects
    event_responses = []
    for e in events:
        event_responses.append(AuditEventResponse(
            audit_id=e["audit_id"],
            event_type=e["event_type"],
            severity=e["severity"],
            timestamp=datetime.fromisoformat(e["timestamp"]),
            sequence_number=e["sequence_number"],
            source=e["source"],
            actor=e["actor"],
            session_id=e["session_id"],
            trace_id=e["trace_id"],
            message=e["message"],
            details=e["details"],
            checksum=e["checksum"],
        ))

    return AuditEventListResponse(
        events=event_responses,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/events/{audit_id}", response_model=AuditEventResponse)
async def get_audit_event(
    audit_id: str,
    current_user: Annotated[dict, Depends(get_current_active_user)],
) -> AuditEventResponse:
    """Get audit event details by ID."""
    events = _generate_demo_events()

    for e in events:
        if e["audit_id"] == audit_id:
            return AuditEventResponse(
                audit_id=e["audit_id"],
                event_type=e["event_type"],
                severity=e["severity"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                sequence_number=e["sequence_number"],
                source=e["source"],
                actor=e["actor"],
                session_id=e["session_id"],
                trace_id=e["trace_id"],
                message=e["message"],
                details=e["details"],
                checksum=e["checksum"],
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Audit event {audit_id} not found",
    )


@router.get("/orders", response_model=OrderAuditListResponse)
async def list_order_audits(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    symbol: str | None = Query(None, description="Filter by symbol"),
    status_filter: str | None = Query(None, alias="status", description="Filter by status"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    limit: int = Query(50, ge=1, le=1000),
) -> OrderAuditListResponse:
    """
    List order audit trails.

    Returns complete audit trail for each order including risk checks,
    execution details, and timing.
    """
    orders = _generate_demo_orders()

    if symbol:
        symbol_upper = symbol.upper()
        orders = [o for o in orders if o["symbol"].upper() == symbol_upper]

    if status_filter:
        orders = [o for o in orders if o["status"] == status_filter.lower()]

    if session_id:
        orders = [o for o in orders if o["session_id"] == session_id]

    total = len(orders)
    orders = orders[:limit]

    order_responses = [OrderAuditResponse(**o) for o in orders]

    return OrderAuditListResponse(orders=order_responses, total=total)


@router.get("/orders/{order_id}", response_model=OrderAuditResponse)
async def get_order_audit(
    order_id: str,
    current_user: Annotated[dict, Depends(get_current_active_user)],
) -> OrderAuditResponse:
    """Get order audit trail by order ID."""
    orders = _generate_demo_orders()

    for o in orders:
        if o["order_id"] == order_id:
            return OrderAuditResponse(**o)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Order audit trail for {order_id} not found",
    )


@router.get("/risk", response_model=RiskAuditListResponse)
async def list_risk_audits(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    check_name: str | None = Query(None, description="Filter by check name"),
    passed: bool | None = Query(None, description="Filter by pass/fail"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    limit: int = Query(50, ge=1, le=1000),
) -> RiskAuditListResponse:
    """
    List risk audit events.

    Returns risk check results for compliance and analysis.
    """
    risk_events = _generate_demo_risk_events()

    if check_name:
        risk_events = [r for r in risk_events if r["check_name"] == check_name]

    if passed is not None:
        risk_events = [r for r in risk_events if r["check_passed"] == passed]

    if symbol:
        symbol_upper = symbol.upper()
        risk_events = [r for r in risk_events if r["symbol"].upper() == symbol_upper]

    total = len(risk_events)
    risk_events = risk_events[:limit]

    risk_responses = [RiskAuditResponse(**r) for r in risk_events]

    return RiskAuditListResponse(risk_events=risk_responses, total=total)


@router.get("/export")
async def export_audit_logs(
    current_user: Annotated[dict, Depends(require_scope("admin"))],
    format: ExportFormatSchema = Query(ExportFormatSchema.CSV, description="Export format"),
    start_time: datetime | None = Query(None, description="Start of export period"),
    end_time: datetime | None = Query(None, description="End of export period"),
    include_events: bool = Query(True, description="Include audit events"),
    include_orders: bool = Query(True, description="Include order trails"),
    include_risk: bool = Query(True, description="Include risk events"),
) -> Response:
    """
    Export audit logs.

    Requires admin scope. Supports multiple formats including
    regulatory formats (SEC, CFTC, FCA).
    """
    import csv
    import io
    import json

    events = _generate_demo_events() if include_events else []
    orders = _generate_demo_orders() if include_orders else []
    risk_events = _generate_demo_risk_events() if include_risk else []

    content: str
    media_type: str

    if format == ExportFormatSchema.CSV:
        output = io.StringIO()
        writer = csv.writer(output)

        # Events
        if events:
            writer.writerow(["# AUDIT EVENTS"])
            writer.writerow(["audit_id", "event_type", "severity", "timestamp", "message"])
            for e in events:
                writer.writerow([e["audit_id"], e["event_type"], e["severity"], e["timestamp"], e["message"]])

        # Orders
        if orders:
            writer.writerow([])
            writer.writerow(["# ORDER AUDIT TRAILS"])
            writer.writerow(["order_id", "symbol", "side", "status", "quantity", "filled_price", "created_at"])
            for o in orders:
                writer.writerow([o["order_id"], o["symbol"], o["side"], o["status"], o["quantity"], o["filled_price"], str(o["created_at"])])

        # Risk
        if risk_events:
            writer.writerow([])
            writer.writerow(["# RISK AUDIT EVENTS"])
            writer.writerow(["risk_event_id", "order_id", "check_name", "passed", "utilization_pct"])
            for r in risk_events:
                writer.writerow([r["risk_event_id"], r["order_id"], r["check_name"], r["check_passed"], r["utilization_pct"]])

        content = output.getvalue()
        media_type = "text/csv"

    elif format in (ExportFormatSchema.JSON, ExportFormatSchema.SEC, ExportFormatSchema.CFTC, ExportFormatSchema.FCA):
        export_data = {
            "export_format": format.value,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
        }

        if include_events:
            export_data["events"] = events
        if include_orders:
            # Convert datetime to string for JSON
            for o in orders:
                o["created_at"] = str(o["created_at"])
                o["filled_at"] = str(o["filled_at"]) if o["filled_at"] else None
            export_data["orders"] = orders
        if include_risk:
            for r in risk_events:
                r["timestamp"] = str(r["timestamp"])
            export_data["risk_events"] = risk_events

        content = json.dumps(export_data, indent=2, default=str)
        media_type = "application/json"

    else:  # JSONL
        lines = []
        if include_events:
            for e in events:
                lines.append(json.dumps({"type": "event", **e}))
        if include_orders:
            for o in orders:
                o["created_at"] = str(o["created_at"])
                o["filled_at"] = str(o["filled_at"]) if o["filled_at"] else None
                lines.append(json.dumps({"type": "order", **o}))
        if include_risk:
            for r in risk_events:
                r["timestamp"] = str(r["timestamp"])
                lines.append(json.dumps({"type": "risk", **r}))
        content = "\n".join(lines)
        media_type = "application/jsonl"

    filename = f"audit_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.{format.value}"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/statistics", response_model=AuditStatisticsResponse)
async def get_audit_statistics(
    current_user: Annotated[dict, Depends(get_current_active_user)],
) -> AuditStatisticsResponse:
    """
    Get audit statistics.

    Returns summary statistics about audit logs including
    event counts, storage usage, and distribution by type.
    """
    events = _generate_demo_events()
    orders = _generate_demo_orders()

    # Count by type
    events_by_type: dict[str, int] = {}
    events_by_severity: dict[str, int] = {}

    for e in events:
        events_by_type[e["event_type"]] = events_by_type.get(e["event_type"], 0) + 1
        events_by_severity[e["severity"]] = events_by_severity.get(e["severity"], 0) + 1

    return AuditStatisticsResponse(
        total_events=len(events),
        total_orders=len(orders),
        events_by_type=events_by_type,
        events_by_severity=events_by_severity,
        hot_files=3,
        warm_files=12,
        cold_files=45,
        storage_size_mb=156.7,
    )

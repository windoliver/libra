"""
Audit Trail Models.

Defines the core data structures for audit logging:
- AuditEvent: Base audit event
- OrderAuditTrail: Complete order lifecycle audit
- RiskAuditTrail: Risk decision audit
- AgentAuditTrail: Agent reasoning audit

All audit events are immutable and timestamped for compliance.

Issue #16: Audit Logging System
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4


class AuditEventType(Enum):
    """Types of audit events."""

    # Order lifecycle
    ORDER_CREATED = "order.created"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_ACCEPTED = "order.accepted"
    ORDER_REJECTED = "order.rejected"
    ORDER_FILLED = "order.filled"
    ORDER_PARTIALLY_FILLED = "order.partially_filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_EXPIRED = "order.expired"

    # Risk events
    RISK_CHECK_PASSED = "risk.check_passed"
    RISK_CHECK_FAILED = "risk.check_failed"
    RISK_LIMIT_BREACH = "risk.limit_breach"
    RISK_STATE_CHANGE = "risk.state_change"
    CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker"

    # Agent events
    AGENT_DECISION = "agent.decision"
    AGENT_SIGNAL = "agent.signal"
    AGENT_OVERRIDE = "agent.override"

    # Position events
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_MODIFIED = "position.modified"

    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGE = "system.config_change"
    ERROR = "system.error"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AuditEvent:
    """
    Base audit event.

    Immutable record of an auditable action with full context.
    All fields are frozen to ensure audit integrity.
    """

    # Identity
    audit_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: AuditEventType = AuditEventType.SYSTEM_START
    severity: AuditSeverity = AuditSeverity.INFO

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: int = 0

    # Context
    source: str = ""  # Component that generated the event
    actor: str = ""  # User/agent/system that initiated
    session_id: str = ""  # Trading session ID
    trace_id: str = ""  # Distributed trace ID
    span_id: str = ""  # Span ID within trace

    # Payload
    message: str = ""
    details: tuple = field(default_factory=tuple)  # Immutable dict alternative
    tags: tuple = field(default_factory=tuple)  # Immutable list alternative

    # Integrity
    checksum: str = ""

    def __post_init__(self) -> None:
        """Calculate checksum if not provided."""
        if not self.checksum:
            # Use object.__setattr__ since dataclass is frozen
            object.__setattr__(self, "checksum", self._calculate_checksum())

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data."""
        data = {
            "audit_id": self.audit_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "actor": self.actor,
            "message": self.message,
            "details": dict(self.details) if self.details else {},
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "audit_id": self.audit_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "source": self.source,
            "actor": self.actor,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "message": self.message,
            "details": dict(self.details) if self.details else {},
            "tags": list(self.tags) if self.tags else [],
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Create from dictionary."""
        # Handle details that may be a JSON string from database
        details_raw = data.get("details", {})
        if isinstance(details_raw, str):
            try:
                details_raw = json.loads(details_raw)
            except json.JSONDecodeError:
                details_raw = {}
        details = tuple(details_raw.items()) if isinstance(details_raw, dict) else ()

        # Handle tags that may be a JSON string from database
        tags_raw = data.get("tags", [])
        if isinstance(tags_raw, str):
            try:
                tags_raw = json.loads(tags_raw)
            except json.JSONDecodeError:
                tags_raw = []
        tags = tuple(tags_raw) if isinstance(tags_raw, list) else ()

        return cls(
            audit_id=data.get("audit_id", str(uuid4())),
            event_type=AuditEventType(data.get("event_type", "system.start")),
            severity=AuditSeverity(data.get("severity", "info")),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else data.get("timestamp", datetime.now(timezone.utc)),
            sequence_number=data.get("sequence_number", 0),
            source=data.get("source", ""),
            actor=data.get("actor", ""),
            session_id=data.get("session_id", ""),
            trace_id=data.get("trace_id", ""),
            span_id=data.get("span_id", ""),
            message=data.get("message", ""),
            details=details,
            tags=tags,
            checksum=data.get("checksum", ""),
        )


@dataclass(frozen=True)
class OrderAuditTrail:
    """
    Complete audit trail for an order lifecycle.

    Captures every stage from signal generation to execution.
    """

    # Order identity
    order_id: str
    client_order_id: str = ""
    exchange_order_id: str = ""

    # Order details
    symbol: str = ""
    side: str = ""  # "buy" or "sell"
    order_type: str = ""  # "market", "limit", etc.
    quantity: Decimal = Decimal("0")
    price: Decimal | None = None
    time_in_force: str = "GTC"

    # Signal source
    strategy_id: str = ""
    strategy_name: str = ""
    signal_strength: float = 0.0
    signal_reason: str = ""

    # Risk check
    risk_check_passed: bool = True
    risk_check_details: tuple = field(default_factory=tuple)
    risk_check_timestamp: datetime | None = None

    # Execution
    submitted_at: datetime | None = None
    accepted_at: datetime | None = None
    filled_at: datetime | None = None
    filled_quantity: Decimal = Decimal("0")
    filled_price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")

    # Status
    status: str = "pending"  # pending, submitted, filled, rejected, cancelled
    rejection_reason: str = ""
    error_message: str = ""

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Trace context
    trace_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "time_in_force": self.time_in_force,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "signal_strength": self.signal_strength,
            "signal_reason": self.signal_reason,
            "risk_check_passed": self.risk_check_passed,
            "risk_check_details": dict(self.risk_check_details),
            "risk_check_timestamp": self.risk_check_timestamp.isoformat()
            if self.risk_check_timestamp
            else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_quantity": str(self.filled_quantity),
            "filled_price": str(self.filled_price),
            "commission": str(self.commission),
            "slippage": str(self.slippage),
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "trace_id": self.trace_id,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class RiskAuditTrail:
    """
    Audit trail for risk decisions.

    Documents every risk check and state change.
    """

    # Identity
    risk_event_id: str = field(default_factory=lambda: str(uuid4()))
    order_id: str = ""

    # Check details
    check_name: str = ""  # e.g., "position_limit", "notional_limit"
    check_passed: bool = True
    check_reason: str = ""

    # Values
    current_value: Decimal = Decimal("0")
    limit_value: Decimal = Decimal("0")
    utilization_pct: float = 0.0

    # State
    previous_state: str = ""  # ACTIVE, REDUCING, HALTED
    new_state: str = ""
    state_change_reason: str = ""

    # Context
    symbol: str = ""
    strategy_id: str = ""

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_event_id": self.risk_event_id,
            "order_id": self.order_id,
            "check_name": self.check_name,
            "check_passed": self.check_passed,
            "check_reason": self.check_reason,
            "current_value": str(self.current_value),
            "limit_value": str(self.limit_value),
            "utilization_pct": self.utilization_pct,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "state_change_reason": self.state_change_reason,
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class AgentAuditTrail:
    """
    Audit trail for agent decisions.

    Captures reasoning and context for agent-generated actions.
    """

    # Identity
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    agent_id: str = ""
    agent_name: str = ""

    # Decision
    action_type: str = ""  # "signal", "order", "cancel", "modify"
    action_target: str = ""  # symbol or order_id

    # Reasoning
    reasoning: str = ""  # Natural language explanation
    confidence: float = 0.0  # 0.0 to 1.0
    model_version: str = ""

    # Inputs
    market_state: tuple = field(default_factory=tuple)  # Key observations
    signals_received: tuple = field(default_factory=tuple)
    rules_triggered: tuple = field(default_factory=tuple)

    # Output
    decision_output: str = ""  # JSON of the decision
    order_id: str = ""  # If decision resulted in order

    # Timing
    decision_time_ms: float = 0.0  # Time to make decision
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "action_target": self.action_target,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "market_state": dict(self.market_state) if self.market_state else {},
            "signals_received": list(self.signals_received),
            "rules_triggered": list(self.rules_triggered),
            "decision_output": self.decision_output,
            "order_id": self.order_id,
            "decision_time_ms": self.decision_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
        }


def create_order_audit(
    order_id: str,
    symbol: str,
    side: str,
    quantity: Decimal,
    strategy_id: str = "",
    strategy_name: str = "",
    signal_reason: str = "",
    price: Decimal | None = None,
    order_type: str = "market",
    trace_id: str = "",
    session_id: str = "",
) -> OrderAuditTrail:
    """Factory function to create order audit trail."""
    return OrderAuditTrail(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_type=order_type,
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        signal_reason=signal_reason,
        trace_id=trace_id,
        session_id=session_id,
    )


def create_risk_audit(
    check_name: str,
    check_passed: bool,
    check_reason: str = "",
    current_value: Decimal = Decimal("0"),
    limit_value: Decimal = Decimal("0"),
    order_id: str = "",
    trace_id: str = "",
) -> RiskAuditTrail:
    """Factory function to create risk audit trail."""
    utilization = (
        float(current_value / limit_value) * 100 if limit_value > 0 else 0.0
    )
    return RiskAuditTrail(
        check_name=check_name,
        check_passed=check_passed,
        check_reason=check_reason,
        current_value=current_value,
        limit_value=limit_value,
        utilization_pct=utilization,
        order_id=order_id,
        trace_id=trace_id,
    )


def create_agent_audit(
    agent_id: str,
    agent_name: str,
    action_type: str,
    reasoning: str,
    confidence: float = 0.0,
    action_target: str = "",
    trace_id: str = "",
) -> AgentAuditTrail:
    """Factory function to create agent audit trail."""
    return AgentAuditTrail(
        agent_id=agent_id,
        agent_name=agent_name,
        action_type=action_type,
        action_target=action_target,
        reasoning=reasoning,
        confidence=confidence,
        trace_id=trace_id,
    )

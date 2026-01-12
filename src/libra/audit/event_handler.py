"""
Audit Event Handler.

Integrates the audit logging system with the core event bus:
- Subscribes to relevant events
- Translates core events to audit events
- Provides automatic audit logging for order/risk events

Issue #16: Audit Logging System
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.audit.logger import AuditLogger, AuditLoggerConfig
from libra.audit.persistence import AuditPersistence
from libra.audit.trail import AuditEventType, AuditSeverity

if TYPE_CHECKING:
    from libra.core.events import Event, EventType

logger = logging.getLogger(__name__)


# Event type mappings from core events to audit events
EVENT_TYPE_MAP: dict[str, AuditEventType] = {
    "ORDER_NEW": AuditEventType.ORDER_CREATED,
    "ORDER_SUBMITTED": AuditEventType.ORDER_SUBMITTED,
    "ORDER_ACCEPTED": AuditEventType.ORDER_ACCEPTED,
    "ORDER_FILLED": AuditEventType.ORDER_FILLED,
    "ORDER_PARTIALLY_FILLED": AuditEventType.ORDER_PARTIALLY_FILLED,
    "ORDER_CANCELLED": AuditEventType.ORDER_CANCELLED,
    "ORDER_REJECTED": AuditEventType.ORDER_REJECTED,
    "ORDER_DENIED": AuditEventType.ORDER_REJECTED,
    "ORDER_EXPIRED": AuditEventType.ORDER_EXPIRED,
    "POSITION_OPENED": AuditEventType.POSITION_OPENED,
    "POSITION_CLOSED": AuditEventType.POSITION_CLOSED,
    "POSITION_UPDATED": AuditEventType.POSITION_MODIFIED,
    "RISK_LIMIT_BREACH": AuditEventType.RISK_LIMIT_BREACH,
    "DRAWDOWN_WARNING": AuditEventType.RISK_STATE_CHANGE,
    "CIRCUIT_BREAKER": AuditEventType.CIRCUIT_BREAKER_TRIGGERED,
    "SIGNAL": AuditEventType.AGENT_SIGNAL,
}


class AuditEventHandler:
    """
    Bridges core event bus with audit logging system.

    Automatically logs audit trails for:
    - Order lifecycle events
    - Risk events and limit breaches
    - Position changes
    - Trading signals

    Example:
        # Create handler
        handler = AuditEventHandler()

        # Register with message bus
        message_bus.subscribe(EventType.ORDER_FILLED, handler.handle_event)
        message_bus.subscribe(EventType.RISK_LIMIT_BREACH, handler.handle_event)

        # Or handle events directly
        handler.handle_event(event)
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
        persistence: AuditPersistence | None = None,
        persist_immediately: bool = False,
    ) -> None:
        """
        Initialize audit event handler.

        Args:
            audit_logger: Custom audit logger (creates default if not provided)
            persistence: Persistence backend for audit logs
            persist_immediately: Write each event immediately to persistence
        """
        self._logger = audit_logger or AuditLogger(
            config=AuditLoggerConfig(persist_immediately=persist_immediately),
            persistence=persistence,
        )
        self._persistence = persistence

        # Statistics
        self._events_processed = 0
        self._events_audited = 0

    @property
    def audit_logger(self) -> AuditLogger:
        """Get the underlying audit logger."""
        return self._logger

    def handle_event(self, event: Event) -> None:
        """
        Handle a core event and create audit trail.

        Args:
            event: Core event from message bus
        """
        self._events_processed += 1

        event_name = event.event_type.name
        audit_type = EVENT_TYPE_MAP.get(event_name)

        if audit_type is None:
            # Event type not mapped for auditing
            return

        self._events_audited += 1

        # Determine severity based on event type
        severity = self._determine_severity(event_name)

        # Create audit event
        audit_event = self._logger.log_event(
            event_type=audit_type,
            message=self._format_message(event),
            severity=severity,
            source=event.source,
            trace_id=event.trace_id,
            span_id=event.span_id,
            details=event.payload,
        )

        # Handle specific event types with additional logging
        if event_name.startswith("ORDER_"):
            self._handle_order_event(event)
        elif event_name.startswith("RISK_") or event_name in ("DRAWDOWN_WARNING", "CIRCUIT_BREAKER"):
            self._handle_risk_event(event)
        elif event_name == "SIGNAL":
            self._handle_signal_event(event)

        # Persist if configured
        if audit_event and self._persistence:
            try:
                self._persistence.write_event(audit_event)
            except Exception as e:
                logger.error("Failed to persist audit event: %s", e)

    def _determine_severity(self, event_name: str) -> AuditSeverity:
        """Determine audit severity from event type."""
        if event_name in ("CIRCUIT_BREAKER", "RISK_LIMIT_BREACH"):
            return AuditSeverity.CRITICAL
        elif event_name in ("ORDER_REJECTED", "ORDER_DENIED", "DRAWDOWN_WARNING"):
            return AuditSeverity.WARNING
        elif event_name in ("ORDER_FILLED", "ORDER_PARTIALLY_FILLED"):
            return AuditSeverity.INFO
        else:
            return AuditSeverity.INFO

    def _format_message(self, event: Event) -> str:
        """Format a human-readable message from event."""
        payload = event.payload
        event_name = event.event_type.name

        if event_name == "ORDER_NEW":
            return f"Order created: {payload.get('side', '?')} {payload.get('quantity', '?')} {payload.get('symbol', '?')}"
        elif event_name == "ORDER_FILLED":
            return f"Order filled: {payload.get('order_id', '?')} @ {payload.get('price', '?')}"
        elif event_name == "ORDER_REJECTED":
            return f"Order rejected: {payload.get('order_id', '?')} - {payload.get('reason', 'unknown')}"
        elif event_name == "RISK_LIMIT_BREACH":
            return f"Risk limit breach: {payload.get('limit_name', '?')} ({payload.get('utilization', 0):.1f}%)"
        elif event_name == "CIRCUIT_BREAKER":
            return f"Circuit breaker triggered: {payload.get('reason', 'unknown')}"
        elif event_name == "SIGNAL":
            return f"Signal: {payload.get('action', '?')} {payload.get('symbol', '?')} (confidence: {payload.get('confidence', 0):.2f})"
        else:
            return f"{event_name.replace('_', ' ').title()}"

    def _handle_order_event(self, event: Event) -> None:
        """Handle order-specific audit logging."""
        payload = event.payload
        event_name = event.event_type.name

        order_id = payload.get("order_id", payload.get("id", ""))
        if not order_id:
            return

        if event_name == "ORDER_NEW":
            # Log order creation
            self._logger.log_order(
                order_id=order_id,
                symbol=payload.get("symbol", ""),
                side=payload.get("side", ""),
                quantity=Decimal(str(payload.get("quantity", 0))),
                price=Decimal(str(payload.get("price", 0))) if payload.get("price") else None,
                order_type=payload.get("order_type", "market"),
                strategy_id=payload.get("strategy_id", ""),
                strategy_name=payload.get("strategy_name", ""),
                signal_reason=payload.get("signal_reason", ""),
                trace_id=event.trace_id,
            )

        elif event_name == "ORDER_SUBMITTED":
            self._logger.log_order_submitted(
                order_id=order_id,
                exchange_order_id=payload.get("exchange_order_id", ""),
                trace_id=event.trace_id,
            )

        elif event_name == "ORDER_FILLED":
            self._logger.log_order_filled(
                order_id=order_id,
                filled_quantity=Decimal(str(payload.get("filled_quantity", 0))),
                filled_price=Decimal(str(payload.get("price", 0))),
                commission=Decimal(str(payload.get("commission", 0))),
                trace_id=event.trace_id,
            )

        elif event_name in ("ORDER_REJECTED", "ORDER_DENIED"):
            self._logger.log_order_rejected(
                order_id=order_id,
                reason=payload.get("reason", "Unknown rejection"),
                trace_id=event.trace_id,
            )

    def _handle_risk_event(self, event: Event) -> None:
        """Handle risk-specific audit logging."""
        payload = event.payload
        event_name = event.event_type.name

        if event_name == "RISK_LIMIT_BREACH":
            self._logger.log_risk_check(
                check_name=payload.get("limit_name", "unknown"),
                passed=False,
                reason=payload.get("reason", "Limit exceeded"),
                current_value=Decimal(str(payload.get("current_value", 0))),
                limit_value=Decimal(str(payload.get("limit_value", 0))),
                order_id=payload.get("order_id", ""),
                symbol=payload.get("symbol", ""),
                trace_id=event.trace_id,
            )

        elif event_name == "CIRCUIT_BREAKER":
            self._logger.log_risk_state_change(
                previous_state=payload.get("previous_state", "ACTIVE"),
                new_state="HALTED",
                reason=payload.get("reason", "Circuit breaker triggered"),
                trace_id=event.trace_id,
            )

        elif event_name == "DRAWDOWN_WARNING":
            self._logger.log_risk_check(
                check_name="drawdown",
                passed=True,  # Warning, not breach
                reason=payload.get("reason", "Drawdown warning"),
                current_value=Decimal(str(payload.get("drawdown", 0))),
                limit_value=Decimal(str(payload.get("max_drawdown", 0))),
                trace_id=event.trace_id,
            )

    def _handle_signal_event(self, event: Event) -> None:
        """Handle signal/agent audit logging."""
        payload = event.payload

        self._logger.log_agent_decision(
            agent_id=payload.get("agent_id", payload.get("strategy_id", "unknown")),
            agent_name=payload.get("agent_name", payload.get("strategy_name", "unknown")),
            action_type="signal",
            reasoning=payload.get("reasoning", payload.get("signal_reason", "")),
            confidence=payload.get("confidence", 0.0),
            action_target=payload.get("symbol", ""),
            trace_id=event.trace_id,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "events_processed": self._events_processed,
            "events_audited": self._events_audited,
            "audit_rate": self._events_audited / max(1, self._events_processed),
            "logger_stats": self._logger.get_statistics(),
        }


def create_audit_handler(
    session_id: str | None = None,
    persistence_path: str | None = None,
) -> AuditEventHandler:
    """
    Factory function to create configured audit handler.

    Args:
        session_id: Optional session identifier
        persistence_path: Optional path for audit storage

    Returns:
        Configured AuditEventHandler
    """
    from libra.audit.persistence import PersistenceConfig, create_audit_persistence

    persistence = None
    if persistence_path:
        persistence = create_audit_persistence(base_path=persistence_path)
        persistence.initialize()

    config = AuditLoggerConfig()
    if session_id:
        config = AuditLoggerConfig(session_id=session_id)

    logger = AuditLogger(config=config, persistence=persistence)

    return AuditEventHandler(audit_logger=logger, persistence=persistence)

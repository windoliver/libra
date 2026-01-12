"""
Audit Logger.

Central audit logging facility that:
- Records all audit events
- Maintains sequence ordering
- Integrates with persistence layer
- Provides structured logging interface

Issue #16: Audit Logging System
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from libra.audit.trail import (
    AgentAuditTrail,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    OrderAuditTrail,
    RiskAuditTrail,
    create_agent_audit,
    create_order_audit,
    create_risk_audit,
)

if TYPE_CHECKING:
    from libra.audit.persistence import AuditPersistence

logger = logging.getLogger(__name__)


@dataclass
class AuditLoggerConfig:
    """Configuration for audit logger."""

    # Session
    session_id: str = field(default_factory=lambda: str(uuid4()))

    # Buffer settings
    buffer_size: int = 10000  # Max events in memory buffer
    flush_interval: float = 5.0  # Seconds between flushes
    flush_batch_size: int = 100  # Events per flush batch

    # Persistence
    persist_immediately: bool = False  # Persist each event immediately
    enable_async_flush: bool = True  # Async background flushing

    # Filtering
    min_severity: AuditSeverity = AuditSeverity.DEBUG
    enabled_event_types: set[AuditEventType] | None = None  # None = all

    # Integrity
    enable_checksums: bool = True


class AuditLogger:
    """
    Central audit logging facility.

    Thread-safe logger that records all audit events with:
    - Automatic sequence numbering
    - In-memory buffering
    - Background persistence
    - Event filtering
    - Integrity checksums

    Example:
        logger = AuditLogger(config=AuditLoggerConfig())

        # Log an order
        order_audit = logger.log_order(
            order_id="ORD-123",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
            strategy_name="momentum",
        )

        # Log a risk event
        logger.log_risk_check(
            check_name="position_limit",
            passed=True,
            order_id="ORD-123",
        )

        # Log agent reasoning
        logger.log_agent_decision(
            agent_id="agent-1",
            agent_name="TradingAgent",
            action_type="signal",
            reasoning="Strong bullish momentum detected",
        )
    """

    def __init__(
        self,
        config: AuditLoggerConfig | None = None,
        persistence: AuditPersistence | None = None,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            config: Logger configuration
            persistence: Optional persistence backend
        """
        self.config = config or AuditLoggerConfig()
        self._persistence = persistence

        # Thread-safe sequence counter
        self._sequence = 0
        self._sequence_lock = threading.Lock()

        # Event buffer
        self._buffer: deque[AuditEvent] = deque(maxlen=self.config.buffer_size)
        self._buffer_lock = threading.Lock()

        # Order tracking
        self._order_trails: dict[str, OrderAuditTrail] = {}
        self._order_lock = threading.Lock()

        # Risk tracking
        self._risk_trails: deque[RiskAuditTrail] = deque(maxlen=1000)

        # Agent tracking
        self._agent_trails: deque[AgentAuditTrail] = deque(maxlen=1000)

        # Async flush task
        self._flush_task: asyncio.Task | None = None
        self._running = False

    def _next_sequence(self) -> int:
        """Get next sequence number (thread-safe)."""
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _should_log(self, event_type: AuditEventType, severity: AuditSeverity) -> bool:
        """Check if event should be logged based on filters."""
        # Check severity
        severity_order = list(AuditSeverity)
        if severity_order.index(severity) < severity_order.index(self.config.min_severity):
            return False

        # Check event type filter
        if self.config.enabled_event_types is not None:
            if event_type not in self.config.enabled_event_types:
                return False

        return True

    def log_event(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        source: str = "",
        actor: str = "",
        trace_id: str = "",
        span_id: str = "",
        details: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> AuditEvent | None:
        """
        Log a generic audit event.

        Args:
            event_type: Type of event
            message: Human-readable message
            severity: Event severity
            source: Component that generated the event
            actor: User/agent that initiated
            trace_id: Distributed trace ID
            span_id: Span ID
            details: Additional details
            tags: Event tags

        Returns:
            AuditEvent if logged, None if filtered
        """
        if not self._should_log(event_type, severity):
            return None

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            sequence_number=self._next_sequence(),
            source=source,
            actor=actor,
            session_id=self.config.session_id,
            trace_id=trace_id,
            span_id=span_id,
            message=message,
            details=tuple((details or {}).items()),
            tags=tuple(tags or []),
        )

        with self._buffer_lock:
            self._buffer.append(event)

        if self.config.persist_immediately and self._persistence:
            try:
                self._persistence.write_event(event)
            except Exception as e:
                logger.error("Failed to persist audit event: %s", e)

        return event

    def log_order(
        self,
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
    ) -> OrderAuditTrail:
        """
        Log order creation.

        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            strategy_id: Strategy that generated signal
            strategy_name: Strategy name
            signal_reason: Why signal was generated
            price: Limit price (if applicable)
            order_type: Order type
            trace_id: Trace ID

        Returns:
            OrderAuditTrail for tracking
        """
        trail = create_order_audit(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            signal_reason=signal_reason,
            price=price,
            order_type=order_type,
            trace_id=trace_id,
            session_id=self.config.session_id,
        )

        with self._order_lock:
            self._order_trails[order_id] = trail

        # Log event
        self.log_event(
            event_type=AuditEventType.ORDER_CREATED,
            message=f"Order created: {side} {quantity} {symbol}",
            source="order_manager",
            trace_id=trace_id,
            details={
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": str(quantity),
                "strategy": strategy_name,
            },
        )

        return trail

    def log_order_submitted(
        self,
        order_id: str,
        exchange_order_id: str = "",
        trace_id: str = "",
    ) -> None:
        """Log order submission."""
        now = datetime.now(timezone.utc)

        with self._order_lock:
            if order_id in self._order_trails:
                old = self._order_trails[order_id]
                # Create new immutable trail with updated fields
                self._order_trails[order_id] = OrderAuditTrail(
                    order_id=old.order_id,
                    client_order_id=old.client_order_id,
                    exchange_order_id=exchange_order_id,
                    symbol=old.symbol,
                    side=old.side,
                    order_type=old.order_type,
                    quantity=old.quantity,
                    price=old.price,
                    time_in_force=old.time_in_force,
                    strategy_id=old.strategy_id,
                    strategy_name=old.strategy_name,
                    signal_strength=old.signal_strength,
                    signal_reason=old.signal_reason,
                    risk_check_passed=old.risk_check_passed,
                    risk_check_details=old.risk_check_details,
                    risk_check_timestamp=old.risk_check_timestamp,
                    submitted_at=now,
                    status="submitted",
                    created_at=old.created_at,
                    updated_at=now,
                    trace_id=old.trace_id or trace_id,
                    session_id=old.session_id,
                )

        self.log_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            message=f"Order submitted: {order_id}",
            source="execution",
            trace_id=trace_id,
            details={"order_id": order_id, "exchange_order_id": exchange_order_id},
        )

    def log_order_filled(
        self,
        order_id: str,
        filled_quantity: Decimal,
        filled_price: Decimal,
        commission: Decimal = Decimal("0"),
        trace_id: str = "",
    ) -> None:
        """Log order fill."""
        now = datetime.now(timezone.utc)

        with self._order_lock:
            if order_id in self._order_trails:
                old = self._order_trails[order_id]
                # Calculate slippage
                slippage = Decimal("0")
                if old.price and old.price > 0:
                    slippage = filled_price - old.price

                self._order_trails[order_id] = OrderAuditTrail(
                    order_id=old.order_id,
                    client_order_id=old.client_order_id,
                    exchange_order_id=old.exchange_order_id,
                    symbol=old.symbol,
                    side=old.side,
                    order_type=old.order_type,
                    quantity=old.quantity,
                    price=old.price,
                    time_in_force=old.time_in_force,
                    strategy_id=old.strategy_id,
                    strategy_name=old.strategy_name,
                    signal_strength=old.signal_strength,
                    signal_reason=old.signal_reason,
                    risk_check_passed=old.risk_check_passed,
                    risk_check_details=old.risk_check_details,
                    risk_check_timestamp=old.risk_check_timestamp,
                    submitted_at=old.submitted_at,
                    accepted_at=old.accepted_at,
                    filled_at=now,
                    filled_quantity=filled_quantity,
                    filled_price=filled_price,
                    commission=commission,
                    slippage=slippage,
                    status="filled",
                    created_at=old.created_at,
                    updated_at=now,
                    trace_id=old.trace_id or trace_id,
                    session_id=old.session_id,
                )

        self.log_event(
            event_type=AuditEventType.ORDER_FILLED,
            message=f"Order filled: {order_id} @ {filled_price}",
            source="execution",
            trace_id=trace_id,
            details={
                "order_id": order_id,
                "filled_quantity": str(filled_quantity),
                "filled_price": str(filled_price),
                "commission": str(commission),
            },
        )

    def log_order_rejected(
        self,
        order_id: str,
        reason: str,
        trace_id: str = "",
    ) -> None:
        """Log order rejection."""
        now = datetime.now(timezone.utc)

        with self._order_lock:
            if order_id in self._order_trails:
                old = self._order_trails[order_id]
                self._order_trails[order_id] = OrderAuditTrail(
                    order_id=old.order_id,
                    client_order_id=old.client_order_id,
                    exchange_order_id=old.exchange_order_id,
                    symbol=old.symbol,
                    side=old.side,
                    order_type=old.order_type,
                    quantity=old.quantity,
                    price=old.price,
                    time_in_force=old.time_in_force,
                    strategy_id=old.strategy_id,
                    strategy_name=old.strategy_name,
                    signal_strength=old.signal_strength,
                    signal_reason=old.signal_reason,
                    risk_check_passed=old.risk_check_passed,
                    risk_check_details=old.risk_check_details,
                    risk_check_timestamp=old.risk_check_timestamp,
                    submitted_at=old.submitted_at,
                    status="rejected",
                    rejection_reason=reason,
                    created_at=old.created_at,
                    updated_at=now,
                    trace_id=old.trace_id or trace_id,
                    session_id=old.session_id,
                )

        self.log_event(
            event_type=AuditEventType.ORDER_REJECTED,
            message=f"Order rejected: {order_id} - {reason}",
            severity=AuditSeverity.WARNING,
            source="execution",
            trace_id=trace_id,
            details={"order_id": order_id, "reason": reason},
        )

    def log_risk_check(
        self,
        check_name: str,
        passed: bool,
        reason: str = "",
        current_value: Decimal = Decimal("0"),
        limit_value: Decimal = Decimal("0"),
        order_id: str = "",
        symbol: str = "",
        trace_id: str = "",
    ) -> RiskAuditTrail:
        """
        Log a risk check result.

        Args:
            check_name: Name of the risk check
            passed: Whether check passed
            reason: Reason for result
            current_value: Current value being checked
            limit_value: Limit value
            order_id: Related order ID
            symbol: Trading symbol
            trace_id: Trace ID

        Returns:
            RiskAuditTrail
        """
        trail = create_risk_audit(
            check_name=check_name,
            check_passed=passed,
            check_reason=reason,
            current_value=current_value,
            limit_value=limit_value,
            order_id=order_id,
            trace_id=trace_id,
        )

        self._risk_trails.append(trail)

        # Update order trail if applicable
        if order_id:
            with self._order_lock:
                if order_id in self._order_trails:
                    old = self._order_trails[order_id]
                    now = datetime.now(timezone.utc)
                    self._order_trails[order_id] = OrderAuditTrail(
                        order_id=old.order_id,
                        client_order_id=old.client_order_id,
                        exchange_order_id=old.exchange_order_id,
                        symbol=old.symbol,
                        side=old.side,
                        order_type=old.order_type,
                        quantity=old.quantity,
                        price=old.price,
                        time_in_force=old.time_in_force,
                        strategy_id=old.strategy_id,
                        strategy_name=old.strategy_name,
                        signal_strength=old.signal_strength,
                        signal_reason=old.signal_reason,
                        risk_check_passed=passed,
                        risk_check_details=tuple(
                            list(old.risk_check_details) + [(check_name, passed)]
                        ),
                        risk_check_timestamp=now,
                        submitted_at=old.submitted_at,
                        status=old.status,
                        created_at=old.created_at,
                        updated_at=now,
                        trace_id=old.trace_id,
                        session_id=old.session_id,
                    )

        event_type = (
            AuditEventType.RISK_CHECK_PASSED
            if passed
            else AuditEventType.RISK_CHECK_FAILED
        )
        severity = AuditSeverity.INFO if passed else AuditSeverity.WARNING

        self.log_event(
            event_type=event_type,
            message=f"Risk check {check_name}: {'passed' if passed else 'failed'}",
            severity=severity,
            source="risk_engine",
            trace_id=trace_id,
            details={
                "check_name": check_name,
                "passed": passed,
                "reason": reason,
                "order_id": order_id,
            },
        )

        return trail

    def log_risk_state_change(
        self,
        previous_state: str,
        new_state: str,
        reason: str = "",
        trace_id: str = "",
    ) -> None:
        """Log trading state change."""
        self.log_event(
            event_type=AuditEventType.RISK_STATE_CHANGE,
            message=f"Trading state: {previous_state} -> {new_state}",
            severity=AuditSeverity.WARNING,
            source="risk_engine",
            trace_id=trace_id,
            details={
                "previous_state": previous_state,
                "new_state": new_state,
                "reason": reason,
            },
        )

    def log_agent_decision(
        self,
        agent_id: str,
        agent_name: str,
        action_type: str,
        reasoning: str,
        confidence: float = 0.0,
        action_target: str = "",
        market_state: dict[str, Any] | None = None,
        decision_output: str = "",
        decision_time_ms: float = 0.0,
        trace_id: str = "",
    ) -> AgentAuditTrail:
        """
        Log an agent decision with reasoning.

        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            action_type: Type of action (signal, order, cancel)
            reasoning: Natural language reasoning
            confidence: Confidence score (0-1)
            action_target: Target symbol or order ID
            market_state: Key market observations
            decision_output: JSON of decision
            decision_time_ms: Time to make decision
            trace_id: Trace ID

        Returns:
            AgentAuditTrail
        """
        trail = create_agent_audit(
            agent_id=agent_id,
            agent_name=agent_name,
            action_type=action_type,
            reasoning=reasoning,
            confidence=confidence,
            action_target=action_target,
            trace_id=trace_id,
        )

        self._agent_trails.append(trail)

        self.log_event(
            event_type=AuditEventType.AGENT_DECISION,
            message=f"Agent {agent_name}: {action_type} on {action_target}",
            source=agent_id,
            actor=agent_name,
            trace_id=trace_id,
            details={
                "agent_id": agent_id,
                "action_type": action_type,
                "reasoning": reasoning[:200],  # Truncate for event
                "confidence": confidence,
                "decision_time_ms": decision_time_ms,
            },
        )

        return trail

    def get_order_trail(self, order_id: str) -> OrderAuditTrail | None:
        """Get order audit trail by ID."""
        with self._order_lock:
            return self._order_trails.get(order_id)

    def get_recent_events(self, limit: int = 100) -> list[AuditEvent]:
        """Get recent audit events."""
        with self._buffer_lock:
            return list(self._buffer)[-limit:]

    def get_recent_orders(self, limit: int = 50) -> list[OrderAuditTrail]:
        """Get recent order trails."""
        with self._order_lock:
            trails = list(self._order_trails.values())
            return sorted(trails, key=lambda t: t.created_at, reverse=True)[:limit]

    def get_recent_risk_events(self, limit: int = 50) -> list[RiskAuditTrail]:
        """Get recent risk audit trails."""
        return list(self._risk_trails)[-limit:]

    def get_recent_agent_decisions(self, limit: int = 50) -> list[AgentAuditTrail]:
        """Get recent agent audit trails."""
        return list(self._agent_trails)[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get audit logger statistics."""
        with self._buffer_lock:
            buffer_size = len(self._buffer)

        with self._order_lock:
            order_count = len(self._order_trails)

        return {
            "session_id": self.config.session_id,
            "sequence_number": self._sequence,
            "buffer_size": buffer_size,
            "buffer_capacity": self.config.buffer_size,
            "order_trails": order_count,
            "risk_trails": len(self._risk_trails),
            "agent_trails": len(self._agent_trails),
        }

    async def start(self) -> None:
        """Start async flush task."""
        if self.config.enable_async_flush and not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._flush_loop())
            logger.info("Audit logger started")

    async def stop(self) -> None:
        """Stop async flush task and flush remaining events."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()
        logger.info("Audit logger stopped")

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Flush error: %s", e)

    async def _flush(self) -> None:
        """Flush buffered events to persistence."""
        if not self._persistence:
            return

        with self._buffer_lock:
            events = list(self._buffer)

        if events:
            try:
                await self._persistence.write_events_async(events)
                logger.debug("Flushed %d audit events", len(events))
            except Exception as e:
                logger.error("Failed to flush audit events: %s", e)


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def set_audit_logger(logger: AuditLogger) -> None:
    """Set global audit logger."""
    global _audit_logger
    _audit_logger = logger

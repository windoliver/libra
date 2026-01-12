"""
Audit Logging System.

Provides comprehensive audit trail logging for trading operations:
- Order lifecycle tracking
- Risk event logging
- Agent decision auditing
- Compliance export formats

Issue #16: Audit Logging System
"""

from libra.audit.event_handler import AuditEventHandler, create_audit_handler
from libra.audit.exporters import AuditExporter, ExportFormat, ExportResult
from libra.audit.logger import AuditLogger, AuditLoggerConfig
from libra.audit.persistence import (
    AuditPersistence,
    PersistenceConfig,
    RetentionPolicy,
    create_audit_persistence,
)
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

__all__ = [
    # Core models
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "OrderAuditTrail",
    "RiskAuditTrail",
    "AgentAuditTrail",
    # Factory functions
    "create_order_audit",
    "create_risk_audit",
    "create_agent_audit",
    # Logger
    "AuditLogger",
    "AuditLoggerConfig",
    # Event Handler
    "AuditEventHandler",
    "create_audit_handler",
    # Persistence
    "AuditPersistence",
    "PersistenceConfig",
    "RetentionPolicy",
    "create_audit_persistence",
    # Exporters
    "AuditExporter",
    "ExportFormat",
    "ExportResult",
]

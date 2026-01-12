"""
Unit tests for Audit Logging System (Issue #16).

Tests:
- AuditEvent creation and serialization
- OrderAuditTrail lifecycle
- RiskAuditTrail events
- AgentAuditTrail decisions
- AuditLogger functionality
- Persistence layer
- Exporters
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

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
from libra.audit.logger import AuditLogger, AuditLoggerConfig
from libra.audit.persistence import (
    AuditPersistence,
    PersistenceConfig,
    RetentionPolicy,
)
from libra.audit.exporters import (
    AuditExporter,
    ExportFormat,
    ExportOptions,
)


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_create_audit_event(self):
        """Test basic event creation."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_CREATED,
            severity=AuditSeverity.INFO,
            source="test_source",
            message="Test order created",
        )

        assert event.event_type == AuditEventType.ORDER_CREATED
        assert event.severity == AuditSeverity.INFO
        assert event.source == "test_source"
        assert event.message == "Test order created"
        assert event.audit_id  # Should have auto-generated ID
        assert event.checksum  # Should have auto-calculated checksum

    def test_event_checksum_calculated(self):
        """Test that checksum is calculated on creation."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_FILLED,
            message="Order filled",
        )

        assert event.checksum
        assert len(event.checksum) == 16  # SHA-256 truncated to 16 chars

    def test_event_immutability(self):
        """Test that event is frozen/immutable."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_CREATED,
            message="Test",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            event.message = "Modified"  # type: ignore

    def test_event_to_dict(self):
        """Test event serialization to dict."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_SUBMITTED,
            severity=AuditSeverity.INFO,
            source="test",
            message="Submitted",
            session_id="session_001",
            trace_id="trace_001",
        )

        data = event.to_dict()

        assert data["event_type"] == "order.submitted"
        assert data["severity"] == "info"
        assert data["source"] == "test"
        assert data["message"] == "Submitted"
        assert data["session_id"] == "session_001"
        assert "timestamp" in data
        assert "audit_id" in data
        assert "checksum" in data

    def test_event_from_dict(self):
        """Test event deserialization from dict."""
        original = AuditEvent(
            event_type=AuditEventType.RISK_CHECK_PASSED,
            message="Risk check passed",
        )

        data = original.to_dict()
        restored = AuditEvent.from_dict(data)

        assert restored.event_type == original.event_type
        assert restored.message == original.message
        assert restored.audit_id == original.audit_id

    def test_event_with_details(self):
        """Test event with additional details."""
        details = (("order_id", "ORD-001"), ("symbol", "BTC/USDT"))
        event = AuditEvent(
            event_type=AuditEventType.ORDER_FILLED,
            message="Order filled",
            details=details,
        )

        assert event.details == details
        data = event.to_dict()
        assert data["details"] == {"order_id": "ORD-001", "symbol": "BTC/USDT"}


class TestOrderAuditTrail:
    """Tests for OrderAuditTrail dataclass."""

    def test_create_order_audit_trail(self):
        """Test order audit trail creation."""
        trail = OrderAuditTrail(
            order_id="ORD-001",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.5"),
            price=Decimal("42500"),
        )

        assert trail.order_id == "ORD-001"
        assert trail.symbol == "BTC/USDT"
        assert trail.side == "buy"
        assert trail.quantity == Decimal("0.5")
        assert trail.price == Decimal("42500")
        assert trail.status == "pending"

    def test_order_audit_factory(self):
        """Test create_order_audit factory function."""
        trail = create_order_audit(
            order_id="ORD-002",
            symbol="ETH/USDT",
            side="sell",
            quantity=Decimal("5.0"),
            strategy_id="strat_001",
            strategy_name="momentum",
            signal_reason="Strong uptrend signal",
        )

        assert trail.order_id == "ORD-002"
        assert trail.strategy_name == "momentum"
        assert trail.signal_reason == "Strong uptrend signal"

    def test_order_audit_to_dict(self):
        """Test order audit serialization."""
        trail = OrderAuditTrail(
            order_id="ORD-003",
            symbol="SOL/USDT",
            side="buy",
            quantity=Decimal("100"),
            status="filled",
            filled_quantity=Decimal("100"),
            filled_price=Decimal("95.50"),
            commission=Decimal("9.55"),
        )

        data = trail.to_dict()

        assert data["order_id"] == "ORD-003"
        assert data["symbol"] == "SOL/USDT"
        assert data["quantity"] == "100"
        assert data["filled_price"] == "95.50"
        assert data["commission"] == "9.55"


class TestRiskAuditTrail:
    """Tests for RiskAuditTrail dataclass."""

    def test_create_risk_audit_trail(self):
        """Test risk audit trail creation."""
        trail = RiskAuditTrail(
            order_id="ORD-001",
            check_name="position_limit",
            check_passed=True,
            check_reason="Within limits",
            current_value=Decimal("0.5"),
            limit_value=Decimal("2.0"),
            utilization_pct=25.0,
        )

        assert trail.check_name == "position_limit"
        assert trail.check_passed is True
        assert trail.utilization_pct == 25.0

    def test_risk_audit_factory(self):
        """Test create_risk_audit factory function."""
        trail = create_risk_audit(
            check_name="notional_limit",
            check_passed=False,
            check_reason="Exceeds notional limit",
            current_value=Decimal("55000"),
            limit_value=Decimal("50000"),
            order_id="ORD-002",
        )

        assert trail.check_passed is False
        assert abs(trail.utilization_pct - 110.0) < 0.01  # 55000/50000 * 100

    def test_risk_audit_to_dict(self):
        """Test risk audit serialization."""
        trail = RiskAuditTrail(
            check_name="drawdown_limit",
            check_passed=False,
            current_value=Decimal("15"),
            limit_value=Decimal("10"),
            new_state="HALTED",
            state_change_reason="Max drawdown exceeded",
        )

        data = trail.to_dict()

        assert data["check_name"] == "drawdown_limit"
        assert data["new_state"] == "HALTED"
        assert data["current_value"] == "15"


class TestAgentAuditTrail:
    """Tests for AgentAuditTrail dataclass."""

    def test_create_agent_audit_trail(self):
        """Test agent audit trail creation."""
        trail = AgentAuditTrail(
            agent_id="agent_001",
            agent_name="momentum_agent",
            action_type="signal",
            action_target="BTC/USDT",
            reasoning="Strong momentum detected with RSI > 70",
            confidence=0.85,
        )

        assert trail.agent_id == "agent_001"
        assert trail.action_type == "signal"
        assert trail.confidence == 0.85

    def test_agent_audit_factory(self):
        """Test create_agent_audit factory function."""
        trail = create_agent_audit(
            agent_id="agent_002",
            agent_name="mean_reversion_agent",
            action_type="order",
            reasoning="Price deviation from mean suggests reversal",
            confidence=0.72,
            action_target="ETH/USDT",
        )

        assert trail.agent_name == "mean_reversion_agent"
        assert trail.action_target == "ETH/USDT"

    def test_agent_audit_to_dict(self):
        """Test agent audit serialization."""
        market_state = (("price", "42500"), ("volume", "1000"))
        trail = AgentAuditTrail(
            agent_id="agent_003",
            agent_name="test_agent",
            action_type="decision",
            reasoning="Test reasoning",
            market_state=market_state,
            decision_time_ms=15.5,
        )

        data = trail.to_dict()

        assert data["agent_id"] == "agent_003"
        assert data["market_state"] == {"price": "42500", "volume": "1000"}
        assert data["decision_time_ms"] == 15.5


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_create_audit_logger(self):
        """Test logger creation."""
        config = AuditLoggerConfig(session_id="test_session")
        logger = AuditLogger(config=config)

        assert logger.config.session_id == "test_session"
        assert logger._sequence == 0

    def test_log_event(self):
        """Test logging an event."""
        logger = AuditLogger()
        event = logger.log_event(
            event_type=AuditEventType.SYSTEM_START,
            message="System started",
            source="test",
        )

        assert event.event_type == AuditEventType.SYSTEM_START
        assert event.message == "System started"
        assert event.sequence_number == 1

    def test_sequence_number_increments(self):
        """Test that sequence numbers increment."""
        logger = AuditLogger()

        event1 = logger.log_event(AuditEventType.SYSTEM_START, "Start")
        event2 = logger.log_event(AuditEventType.ORDER_CREATED, "Order")
        event3 = logger.log_event(AuditEventType.ORDER_FILLED, "Filled")

        assert event1.sequence_number == 1
        assert event2.sequence_number == 2
        assert event3.sequence_number == 3

    def test_log_order(self):
        """Test logging order creation."""
        logger = AuditLogger()
        trail = logger.log_order(
            order_id="ORD-001",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
            strategy_id="strat_001",
        )

        assert trail.order_id == "ORD-001"
        assert trail.symbol == "BTC/USDT"
        assert "ORD-001" in logger._order_trails

    def test_log_order_filled(self):
        """Test logging order fill."""
        logger = AuditLogger()

        # Create order first
        logger.log_order(
            order_id="ORD-001",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
        )

        # Log fill
        logger.log_order_filled(
            order_id="ORD-001",
            filled_quantity=Decimal("0.5"),
            filled_price=Decimal("42500"),
            commission=Decimal("10.63"),
        )

        trail = logger._order_trails["ORD-001"]
        assert trail.status == "filled"
        assert trail.filled_price == Decimal("42500")

    def test_log_risk_check(self):
        """Test logging risk check."""
        logger = AuditLogger()
        trail = logger.log_risk_check(
            check_name="position_limit",
            passed=True,
            current_value=Decimal("1.0"),
            limit_value=Decimal("5.0"),
            order_id="ORD-001",
        )

        assert trail.check_passed is True
        assert trail.utilization_pct == 20.0

    def test_log_agent_decision(self):
        """Test logging agent decision."""
        logger = AuditLogger()
        trail = logger.log_agent_decision(
            agent_id="agent_001",
            agent_name="test_agent",
            action_type="signal",
            reasoning="Strong buy signal detected",
            confidence=0.9,
        )

        assert trail.agent_name == "test_agent"
        assert trail.confidence == 0.9


class TestAuditPersistence:
    """Tests for AuditPersistence class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_persistence(self, temp_storage):
        """Test persistence creation."""
        config = PersistenceConfig(
            base_path=temp_storage,
            hot_path=temp_storage / "hot",
            warm_path=temp_storage / "warm",
            cold_path=temp_storage / "cold",
            db_path=temp_storage / "audit.db",
        )

        persistence = AuditPersistence(config)
        persistence.initialize()

        assert persistence._initialized
        assert (temp_storage / "hot").exists()
        assert (temp_storage / "warm").exists()
        assert (temp_storage / "cold").exists()
        assert (temp_storage / "audit.db").exists()

        persistence.close()

    def test_write_event(self, temp_storage):
        """Test writing an event."""
        config = PersistenceConfig(
            base_path=temp_storage,
            hot_path=temp_storage / "hot",
            warm_path=temp_storage / "warm",
            cold_path=temp_storage / "cold",
            db_path=temp_storage / "audit.db",
        )

        persistence = AuditPersistence(config)
        persistence.initialize()

        event = AuditEvent(
            event_type=AuditEventType.ORDER_CREATED,
            message="Test order",
            source="test",
        )

        persistence.write_event(event)

        # Check file was written
        hot_files = list((temp_storage / "hot").glob("*.jsonl"))
        assert len(hot_files) == 1

        # Check content
        with open(hot_files[0], "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["event_type"] == "order.created"
            assert data["message"] == "Test order"

        persistence.close()

    def test_query_events(self, temp_storage):
        """Test querying events."""
        config = PersistenceConfig(
            base_path=temp_storage,
            hot_path=temp_storage / "hot",
            warm_path=temp_storage / "warm",
            cold_path=temp_storage / "cold",
            db_path=temp_storage / "audit.db",
        )

        persistence = AuditPersistence(config)
        persistence.initialize()

        # Write some events
        for i in range(5):
            event = AuditEvent(
                event_type=AuditEventType.ORDER_CREATED if i % 2 == 0 else AuditEventType.ORDER_FILLED,
                message=f"Event {i}",
            )
            persistence.write_event(event)

        # Query all events
        results = persistence.query_events()
        assert len(results) == 5

        # Query by type
        created_events = persistence.query_events(
            event_types=[AuditEventType.ORDER_CREATED]
        )
        assert len(created_events) == 3

        persistence.close()

    def test_write_order(self, temp_storage):
        """Test writing order audit trail."""
        config = PersistenceConfig(
            base_path=temp_storage,
            hot_path=temp_storage / "hot",
            warm_path=temp_storage / "warm",
            cold_path=temp_storage / "cold",
            db_path=temp_storage / "audit.db",
        )

        persistence = AuditPersistence(config)
        persistence.initialize()

        order = OrderAuditTrail(
            order_id="ORD-001",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
            status="filled",
        )

        persistence.write_order(order)

        # Query order
        orders = persistence.query_orders(symbol="BTC/USDT")
        assert len(orders) == 1
        assert orders[0]["order_id"] == "ORD-001"

        persistence.close()

    def test_get_statistics(self, temp_storage):
        """Test getting statistics."""
        config = PersistenceConfig(
            base_path=temp_storage,
            hot_path=temp_storage / "hot",
            warm_path=temp_storage / "warm",
            cold_path=temp_storage / "cold",
            db_path=temp_storage / "audit.db",
        )

        persistence = AuditPersistence(config)
        persistence.initialize()

        # Write some events
        for i in range(3):
            event = AuditEvent(
                event_type=AuditEventType.ORDER_CREATED,
                message=f"Event {i}",
            )
            persistence.write_event(event)

        stats = persistence.get_statistics()

        assert stats["initialized"] is True
        assert stats["indexed_events"] == 3
        assert stats["hot_files"] >= 1

        persistence.close()


class TestAuditExporter:
    """Tests for AuditExporter class."""

    @pytest.fixture
    def sample_events(self):
        """Create sample events for testing."""
        return [
            AuditEvent(
                event_type=AuditEventType.ORDER_CREATED,
                severity=AuditSeverity.INFO,
                message="Order created",
                source="test",
            ),
            AuditEvent(
                event_type=AuditEventType.ORDER_FILLED,
                severity=AuditSeverity.INFO,
                message="Order filled",
                source="test",
            ),
        ]

    @pytest.fixture
    def sample_orders(self):
        """Create sample orders for testing."""
        now = datetime.now(timezone.utc)
        return [
            OrderAuditTrail(
                order_id="ORD-001",
                symbol="BTC/USDT",
                side="buy",
                order_type="limit",
                quantity=Decimal("0.5"),
                price=Decimal("42500"),
                status="filled",
                filled_quantity=Decimal("0.5"),
                filled_price=Decimal("42485"),
                created_at=now,
                filled_at=now,
            ),
        ]

    def test_create_exporter(self):
        """Test exporter creation."""
        exporter = AuditExporter()
        assert exporter.options.format == ExportFormat.CSV

    def test_export_events_csv(self, sample_events):
        """Test exporting events to CSV."""
        exporter = AuditExporter(ExportOptions(format=ExportFormat.CSV))
        result = exporter.export_events(sample_events)

        assert result.success
        assert result.format == ExportFormat.CSV
        assert result.record_count == 2
        assert result.content is not None
        assert "audit_id" in result.content  # Header
        assert "Order created" in result.content

    def test_export_events_json(self, sample_events):
        """Test exporting events to JSON."""
        exporter = AuditExporter(ExportOptions(format=ExportFormat.JSON))
        result = exporter.export_events(sample_events)

        assert result.success
        assert result.format == ExportFormat.JSON
        assert result.content is not None

        data = json.loads(result.content)
        assert "events" in data
        assert len(data["events"]) == 2

    def test_export_events_jsonl(self, sample_events):
        """Test exporting events to JSON Lines."""
        exporter = AuditExporter(ExportOptions(format=ExportFormat.JSONL))
        result = exporter.export_events(sample_events)

        assert result.success
        assert result.format == ExportFormat.JSONL
        assert result.content is not None

        lines = result.content.strip().split("\n")
        assert len(lines) == 2

    def test_export_orders_csv(self, sample_orders):
        """Test exporting orders to CSV."""
        exporter = AuditExporter()
        result = exporter.export_orders(sample_orders)

        assert result.success
        assert "ORD-001" in result.content
        assert "BTC/USDT" in result.content

    def test_export_orders_sec_format(self, sample_orders):
        """Test exporting orders to SEC CAT format."""
        exporter = AuditExporter()
        result = exporter.export_orders(sample_orders, format=ExportFormat.SEC)

        assert result.success
        assert result.format == ExportFormat.SEC

        data = json.loads(result.content)
        assert data["catSubmissionType"] == "ORDER"
        assert "records" in data
        assert data["records"][0]["orderID"] == "ORD-001"

    def test_export_orders_cftc_format(self, sample_orders):
        """Test exporting orders to CFTC format."""
        exporter = AuditExporter()
        result = exporter.export_orders(sample_orders, format=ExportFormat.CFTC)

        assert result.success
        assert result.format == ExportFormat.CFTC

        data = json.loads(result.content)
        assert data["reportType"] == "TRADE"
        assert "trades" in data

    def test_export_orders_fca_format(self, sample_orders):
        """Test exporting orders to FCA MiFID II format."""
        exporter = AuditExporter()
        result = exporter.export_orders(sample_orders, format=ExportFormat.FCA)

        assert result.success
        assert result.format == ExportFormat.FCA

        data = json.loads(result.content)
        assert data["reportType"] == "TRANSACTION"
        assert "transactions" in data

    def test_export_to_file(self, sample_events, tmp_path):
        """Test exporting to file."""
        exporter = AuditExporter()
        output_path = tmp_path / "export.csv"

        result = exporter.export_events(sample_events, output_path=output_path)

        assert result.success
        assert result.file_path == output_path
        assert output_path.exists()

    def test_generate_compliance_report(self, sample_events, sample_orders):
        """Test generating compliance report."""
        exporter = AuditExporter()

        risk_events = [
            RiskAuditTrail(
                check_name="position_limit",
                check_passed=True,
                current_value=Decimal("0.5"),
                limit_value=Decimal("2.0"),
            ),
        ]

        result = exporter.generate_compliance_report(
            events=sample_events,
            orders=sample_orders,
            risk_events=risk_events,
        )

        assert result.success
        assert result.content is not None

        report = json.loads(result.content)
        assert report["report_type"] == "COMPLIANCE_AUDIT"
        assert "summary" in report
        assert report["summary"]["total_events"] == 2
        assert report["summary"]["total_orders"] == 1


class TestRetentionPolicy:
    """Tests for RetentionPolicy configuration."""

    def test_default_retention_policy(self):
        """Test default retention settings."""
        policy = RetentionPolicy()

        assert policy.hot_retention_days == 7
        assert policy.warm_retention_days == 30
        assert policy.cold_retention_days == 365
        assert policy.delete_after_days == 730  # 2 years
        assert policy.max_file_size_mb == 100

    def test_custom_retention_policy(self):
        """Test custom retention settings."""
        policy = RetentionPolicy(
            hot_retention_days=3,
            warm_retention_days=14,
            cold_retention_days=180,
        )

        assert policy.hot_retention_days == 3
        assert policy.warm_retention_days == 14
        assert policy.cold_retention_days == 180


class TestIntegration:
    """Integration tests for the audit system."""

    @pytest.fixture
    def audit_system(self):
        """Create complete audit system for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            config = PersistenceConfig(
                base_path=storage_path,
                hot_path=storage_path / "hot",
                warm_path=storage_path / "warm",
                cold_path=storage_path / "cold",
                db_path=storage_path / "audit.db",
            )

            persistence = AuditPersistence(config)
            persistence.initialize()

            logger_config = AuditLoggerConfig(session_id="integration_test")
            logger = AuditLogger(config=logger_config)

            yield {
                "logger": logger,
                "persistence": persistence,
                "exporter": AuditExporter(),
            }

            persistence.close()

    def test_full_order_lifecycle(self, audit_system):
        """Test logging a complete order lifecycle."""
        logger = audit_system["logger"]
        persistence = audit_system["persistence"]

        # 1. Create order
        trail = logger.log_order(
            order_id="ORD-INT-001",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
            strategy_id="strat_001",
            strategy_name="momentum",
            signal_reason="Strong uptrend",
            price=Decimal("42500"),
        )

        # Persist order created event
        event = logger.log_event(
            AuditEventType.ORDER_CREATED,
            f"Order {trail.order_id} created",
            source="order_manager",
        )
        persistence.write_event(event)

        # 2. Risk check
        risk_trail = logger.log_risk_check(
            check_name="position_limit",
            passed=True,
            current_value=Decimal("0.5"),
            limit_value=Decimal("5.0"),
            order_id="ORD-INT-001",
        )

        event = logger.log_event(
            AuditEventType.RISK_CHECK_PASSED,
            "Risk check passed",
            source="risk_manager",
        )
        persistence.write_event(event)

        # 3. Order submitted
        logger.log_order_submitted(
            order_id="ORD-INT-001",
            exchange_order_id="EX-123",
        )

        event = logger.log_event(
            AuditEventType.ORDER_SUBMITTED,
            "Order submitted to exchange",
            source="order_manager",
        )
        persistence.write_event(event)

        # 4. Order filled
        logger.log_order_filled(
            order_id="ORD-INT-001",
            filled_quantity=Decimal("0.5"),
            filled_price=Decimal("42485"),
            commission=Decimal("10.62"),
        )

        event = logger.log_event(
            AuditEventType.ORDER_FILLED,
            "Order fully filled",
            source="order_manager",
        )
        persistence.write_event(event)

        # Verify persistence
        events = persistence.query_events()
        assert len(events) == 4

        # Get final order trail
        final_trail = logger._order_trails["ORD-INT-001"]
        assert final_trail.status == "filled"
        assert final_trail.filled_price == Decimal("42485")

    def test_export_after_logging(self, audit_system):
        """Test exporting logs after creating them."""
        logger = audit_system["logger"]
        persistence = audit_system["persistence"]
        exporter = audit_system["exporter"]

        # Create some events
        for i in range(5):
            event = logger.log_event(
                AuditEventType.ORDER_CREATED,
                f"Order {i} created",
                source="test",
            )
            persistence.write_event(event)

        # Query and export
        events_data = persistence.query_events()

        # Convert to AuditEvent objects
        events = [AuditEvent.from_dict(e) for e in events_data]

        result = exporter.export_events(events, format=ExportFormat.JSON)

        assert result.success
        assert result.record_count == 5

        data = json.loads(result.content)
        assert len(data["events"]) == 5

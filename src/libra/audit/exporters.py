"""
Audit Log Exporters.

Export audit logs in various formats for compliance:
- CSV for spreadsheet analysis
- JSON for API consumption
- Regulatory formats (SEC, CFTC, FCA)
- Custom report formats

Issue #16: Audit Logging System
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libra.audit.trail import (
        AuditEvent,
        OrderAuditTrail,
        RiskAuditTrail,
    )

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines
    SEC = "sec"  # SEC regulatory format
    CFTC = "cftc"  # CFTC regulatory format
    FCA = "fca"  # UK FCA format
    CUSTOM = "custom"


@dataclass
class ExportOptions:
    """Options for audit export."""

    format: ExportFormat = ExportFormat.CSV
    include_headers: bool = True
    include_metadata: bool = True
    pretty_print: bool = False
    date_format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
    decimal_places: int = 8
    timezone: str = "UTC"


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    format: ExportFormat
    record_count: int
    file_path: Path | None = None
    content: str | None = None
    error: str | None = None
    exported_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AuditExporter:
    """
    Export audit logs to various formats.

    Example:
        exporter = AuditExporter()

        # Export to CSV
        result = exporter.export_events(
            events=events,
            format=ExportFormat.CSV,
            output_path=Path("audit_export.csv"),
        )

        # Export orders to JSON
        json_content = exporter.export_orders_to_json(orders)

        # Generate compliance report
        report = exporter.generate_compliance_report(
            events=events,
            orders=orders,
            format=ExportFormat.SEC,
        )
    """

    def __init__(self, options: ExportOptions | None = None) -> None:
        """Initialize exporter."""
        self.options = options or ExportOptions()

    def export_events(
        self,
        events: list[AuditEvent],
        output_path: Path | None = None,
        format: ExportFormat | None = None,
    ) -> ExportResult:
        """
        Export audit events.

        Args:
            events: Events to export
            output_path: Optional file path
            format: Override format

        Returns:
            ExportResult
        """
        fmt = format or self.options.format

        try:
            if fmt == ExportFormat.CSV:
                content = self._events_to_csv(events)
            elif fmt == ExportFormat.JSON:
                content = self._events_to_json(events)
            elif fmt == ExportFormat.JSONL:
                content = self._events_to_jsonl(events)
            else:
                content = self._events_to_json(events)

            if output_path:
                output_path.write_text(content, encoding="utf-8")

            return ExportResult(
                success=True,
                format=fmt,
                record_count=len(events),
                file_path=output_path,
                content=content if not output_path else None,
            )
        except Exception as e:
            logger.error("Export failed: %s", e)
            return ExportResult(
                success=False,
                format=fmt,
                record_count=0,
                error=str(e),
            )

    def export_orders(
        self,
        orders: list[OrderAuditTrail],
        output_path: Path | None = None,
        format: ExportFormat | None = None,
    ) -> ExportResult:
        """
        Export order audit trails.

        Args:
            orders: Orders to export
            output_path: Optional file path
            format: Override format

        Returns:
            ExportResult
        """
        fmt = format or self.options.format

        try:
            if fmt == ExportFormat.CSV:
                content = self._orders_to_csv(orders)
            elif fmt == ExportFormat.JSON:
                content = self._orders_to_json(orders)
            elif fmt in (ExportFormat.SEC, ExportFormat.CFTC, ExportFormat.FCA):
                content = self._orders_to_regulatory(orders, fmt)
            else:
                content = self._orders_to_json(orders)

            if output_path:
                output_path.write_text(content, encoding="utf-8")

            return ExportResult(
                success=True,
                format=fmt,
                record_count=len(orders),
                file_path=output_path,
                content=content if not output_path else None,
            )
        except Exception as e:
            logger.error("Export failed: %s", e)
            return ExportResult(
                success=False,
                format=fmt,
                record_count=0,
                error=str(e),
            )

    def _events_to_csv(self, events: list[AuditEvent]) -> str:
        """Convert events to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)

        if self.options.include_headers:
            writer.writerow([
                "audit_id",
                "event_type",
                "severity",
                "timestamp",
                "sequence_number",
                "source",
                "actor",
                "session_id",
                "trace_id",
                "message",
                "checksum",
            ])

        for event in events:
            writer.writerow([
                event.audit_id,
                event.event_type.value,
                event.severity.value,
                event.timestamp.strftime(self.options.date_format),
                event.sequence_number,
                event.source,
                event.actor,
                event.session_id,
                event.trace_id,
                event.message,
                event.checksum,
            ])

        return output.getvalue()

    def _events_to_json(self, events: list[AuditEvent]) -> str:
        """Convert events to JSON format."""
        data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "record_count": len(events),
            "events": [event.to_dict() for event in events],
        }

        if self.options.pretty_print:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)

    def _events_to_jsonl(self, events: list[AuditEvent]) -> str:
        """Convert events to JSON Lines format."""
        lines = [json.dumps(event.to_dict(), default=str) for event in events]
        return "\n".join(lines)

    def _orders_to_csv(self, orders: list[OrderAuditTrail]) -> str:
        """Convert orders to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)

        if self.options.include_headers:
            writer.writerow([
                "order_id",
                "symbol",
                "side",
                "order_type",
                "quantity",
                "price",
                "status",
                "strategy_name",
                "risk_check_passed",
                "filled_quantity",
                "filled_price",
                "commission",
                "slippage",
                "created_at",
                "filled_at",
                "session_id",
                "trace_id",
            ])

        for order in orders:
            writer.writerow([
                order.order_id,
                order.symbol,
                order.side,
                order.order_type,
                str(order.quantity),
                str(order.price) if order.price else "",
                order.status,
                order.strategy_name,
                "Yes" if order.risk_check_passed else "No",
                str(order.filled_quantity),
                str(order.filled_price),
                str(order.commission),
                str(order.slippage),
                order.created_at.strftime(self.options.date_format),
                order.filled_at.strftime(self.options.date_format) if order.filled_at else "",
                order.session_id,
                order.trace_id,
            ])

        return output.getvalue()

    def _orders_to_json(self, orders: list[OrderAuditTrail]) -> str:
        """Convert orders to JSON format."""
        data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "record_count": len(orders),
            "orders": [order.to_dict() for order in orders],
        }

        if self.options.pretty_print:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)

    def _orders_to_regulatory(
        self,
        orders: list[OrderAuditTrail],
        format: ExportFormat,
    ) -> str:
        """
        Convert orders to regulatory format.

        Different regulators require different fields and formats.
        """
        if format == ExportFormat.SEC:
            return self._orders_to_sec_format(orders)
        elif format == ExportFormat.CFTC:
            return self._orders_to_cftc_format(orders)
        elif format == ExportFormat.FCA:
            return self._orders_to_fca_format(orders)
        else:
            return self._orders_to_json(orders)

    def _orders_to_sec_format(self, orders: list[OrderAuditTrail]) -> str:
        """
        SEC CAT (Consolidated Audit Trail) format.

        Required fields per SEC Rule 613:
        - Unique order identifier
        - Symbol/CUSIP
        - Event type (new, modified, cancelled, executed)
        - Timestamp (to millisecond)
        - Firm identification
        - Customer information (if applicable)
        - Order details (side, quantity, price, type)
        """
        records = []

        for order in orders:
            record = {
                # CAT required fields
                "catReporterIMID": "LIBRA",  # Firm identifier
                "orderID": order.order_id,
                "symbol": order.symbol,
                "eventTimestamp": order.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "manualFlag": False,
                "electronicDupFlag": False,
                "electronicTimestamp": order.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",

                # Order details
                "side": "B" if order.side.lower() == "buy" else "S",
                "quantity": str(order.quantity),
                "orderType": self._map_order_type_sec(order.order_type),
                "price": str(order.price) if order.price else None,
                "timeInForce": self._map_tif_sec(order.time_in_force),

                # Execution
                "orderStatus": self._map_status_sec(order.status),
                "leavesQty": str(order.quantity - order.filled_quantity),
                "execQty": str(order.filled_quantity),
                "execPrice": str(order.filled_price) if order.filled_price else None,

                # Internal tracking
                "firmOrderID": order.client_order_id or order.order_id,
                "senderIMID": "LIBRA",
            }
            records.append(record)

        output = {
            "catSubmissionType": "ORDER",
            "catReporterIMID": "LIBRA",
            "submissionTimestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            "records": records,
        }

        return json.dumps(output, indent=2)

    def _orders_to_cftc_format(self, orders: list[OrderAuditTrail]) -> str:
        """
        CFTC swap data repository format.

        Required for derivatives trading under Dodd-Frank.
        """
        records = []

        for order in orders:
            record = {
                # CFTC required fields
                "uniqueTransactionIdentifier": order.order_id,
                "reportingTimestamp": order.created_at.isoformat(),
                "executionTimestamp": order.filled_at.isoformat() if order.filled_at else None,

                # Product identification
                "productIdentifier": order.symbol,
                "underlyingAsset": order.symbol.split("/")[0] if "/" in order.symbol else order.symbol,

                # Trade details
                "buySellIndicator": "B" if order.side.lower() == "buy" else "S",
                "notionalAmount": str(order.quantity * (order.filled_price or order.price or 0)),
                "notionalCurrency": "USD",
                "priceNotation": str(order.filled_price or order.price or 0),

                # Counterparty
                "reportingParty": "LIBRA",
                "nonReportingParty": "EXCHANGE",

                # Status
                "actionType": self._map_status_cftc(order.status),
            }
            records.append(record)

        output = {
            "reportType": "TRADE",
            "reportingEntity": "LIBRA",
            "submissionDate": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "trades": records,
        }

        return json.dumps(output, indent=2)

    def _orders_to_fca_format(self, orders: list[OrderAuditTrail]) -> str:
        """
        UK FCA MiFID II transaction reporting format.

        Required under MiFID II Article 26.
        """
        records = []

        for order in orders:
            record = {
                # MiFID II required fields
                "transactionReferenceNumber": order.order_id,
                "tradingVenue": "OTC",
                "executingEntity": "LIBRA",
                "tradingDateTime": order.filled_at.isoformat() if order.filled_at else order.created_at.isoformat(),

                # Instrument
                "instrumentIdentification": order.symbol,
                "instrumentClassification": "CRYPTO",

                # Trade details
                "buySellIndicator": "BUYI" if order.side.lower() == "buy" else "SELL",
                "quantity": str(order.quantity),
                "price": str(order.filled_price or order.price or 0),
                "priceCurrency": "USD",

                # Execution
                "netAmount": str(
                    (order.filled_quantity * order.filled_price) - order.commission
                    if order.filled_price else 0
                ),

                # Flags
                "waiver": "NONE",
                "shortSellingIndicator": "UNDI",
            }
            records.append(record)

        output = {
            "reportType": "TRANSACTION",
            "reportingFirm": "LIBRA",
            "reportDate": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "transactions": records,
        }

        return json.dumps(output, indent=2)

    def _map_order_type_sec(self, order_type: str) -> str:
        """Map order type to SEC code."""
        mapping = {
            "market": "MKT",
            "limit": "LMT",
            "stop": "STP",
            "stop_limit": "STL",
        }
        return mapping.get(order_type.lower(), "OTH")

    def _map_tif_sec(self, tif: str) -> str:
        """Map time in force to SEC code."""
        mapping = {
            "gtc": "GTC",
            "day": "DAY",
            "ioc": "IOC",
            "fok": "FOK",
        }
        return mapping.get(tif.lower(), "GTD")

    def _map_status_sec(self, status: str) -> str:
        """Map status to SEC code."""
        mapping = {
            "pending": "NEW",
            "submitted": "NEW",
            "filled": "FILL",
            "partially_filled": "PFIL",
            "cancelled": "CANC",
            "rejected": "REJ",
        }
        return mapping.get(status.lower(), "NEW")

    def _map_status_cftc(self, status: str) -> str:
        """Map status to CFTC action type."""
        mapping = {
            "pending": "NEWT",
            "submitted": "NEWT",
            "filled": "NEWT",
            "cancelled": "CANC",
            "rejected": "RJCT",
        }
        return mapping.get(status.lower(), "NEWT")

    def generate_compliance_report(
        self,
        events: list[AuditEvent] | None = None,
        orders: list[OrderAuditTrail] | None = None,
        risk_events: list[RiskAuditTrail] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        output_path: Path | None = None,
    ) -> ExportResult:
        """
        Generate comprehensive compliance report.

        Includes:
        - Executive summary
        - Order activity
        - Risk events
        - Exceptions and alerts

        Args:
            events: Audit events
            orders: Order trails
            risk_events: Risk audit trails
            start_time: Report period start
            end_time: Report period end
            output_path: Optional output file

        Returns:
            ExportResult
        """
        now = datetime.now(timezone.utc)

        report = {
            "report_type": "COMPLIANCE_AUDIT",
            "generated_at": now.isoformat(),
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "summary": {
                "total_events": len(events) if events else 0,
                "total_orders": len(orders) if orders else 0,
                "total_risk_events": len(risk_events) if risk_events else 0,
            },
        }

        # Order summary
        if orders:
            filled = [o for o in orders if o.status == "filled"]
            rejected = [o for o in orders if o.status == "rejected"]

            report["orders"] = {
                "total": len(orders),
                "filled": len(filled),
                "rejected": len(rejected),
                "fill_rate": len(filled) / len(orders) * 100 if orders else 0,
                "total_volume": sum(float(o.filled_quantity) for o in filled),
                "total_commission": sum(float(o.commission) for o in filled),
                "by_symbol": self._group_orders_by_symbol(orders),
                "by_strategy": self._group_orders_by_strategy(orders),
            }

        # Risk summary
        if risk_events:
            failed_checks = [r for r in risk_events if not r.check_passed]
            report["risk"] = {
                "total_checks": len(risk_events),
                "failed_checks": len(failed_checks),
                "pass_rate": (len(risk_events) - len(failed_checks)) / len(risk_events) * 100
                if risk_events else 0,
                "by_check_type": self._group_risk_by_type(risk_events),
            }

        # Exceptions
        if events:
            errors = [e for e in events if e.severity.value in ("error", "critical")]
            report["exceptions"] = {
                "total": len(errors),
                "events": [e.to_dict() for e in errors[:100]],  # First 100
            }

        content = json.dumps(report, indent=2, default=str)

        if output_path:
            output_path.write_text(content, encoding="utf-8")

        return ExportResult(
            success=True,
            format=ExportFormat.JSON,
            record_count=(len(events or []) + len(orders or []) + len(risk_events or [])),
            file_path=output_path,
            content=content if not output_path else None,
        )

    def _group_orders_by_symbol(self, orders: list[OrderAuditTrail]) -> dict[str, int]:
        """Group orders by symbol."""
        groups: dict[str, int] = {}
        for order in orders:
            groups[order.symbol] = groups.get(order.symbol, 0) + 1
        return groups

    def _group_orders_by_strategy(self, orders: list[OrderAuditTrail]) -> dict[str, int]:
        """Group orders by strategy."""
        groups: dict[str, int] = {}
        for order in orders:
            key = order.strategy_name or "unknown"
            groups[key] = groups.get(key, 0) + 1
        return groups

    def _group_risk_by_type(self, risk_events: list[RiskAuditTrail]) -> dict[str, dict]:
        """Group risk events by check type."""
        groups: dict[str, dict] = {}
        for event in risk_events:
            key = event.check_name
            if key not in groups:
                groups[key] = {"total": 0, "passed": 0, "failed": 0}
            groups[key]["total"] += 1
            if event.check_passed:
                groups[key]["passed"] += 1
            else:
                groups[key]["failed"] += 1
        return groups


def create_exporter(
    format: str = "csv",
    pretty_print: bool = False,
) -> AuditExporter:
    """
    Factory function to create exporter.

    Args:
        format: Export format (csv, json, jsonl, sec, cftc, fca)
        pretty_print: Pretty print JSON output

    Returns:
        Configured AuditExporter
    """
    options = ExportOptions(
        format=ExportFormat(format.lower()),
        pretty_print=pretty_print,
    )
    return AuditExporter(options)

"""
Audit Persistence Layer.

Handles storage, rotation, and retention of audit logs:
- File-based storage with rotation
- SQLite for indexed queries
- Retention policy enforcement
- Compression for archived logs

Issue #16: Audit Logging System
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import shutil
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from libra.audit.trail import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    OrderAuditTrail,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Audit log retention policy."""

    # Retention periods
    hot_retention_days: int = 7  # Keep in active storage
    warm_retention_days: int = 30  # Keep in indexed DB
    cold_retention_days: int = 365  # Keep in compressed archive
    delete_after_days: int = 730  # Delete entirely (2 years)

    # Rotation settings
    max_file_size_mb: int = 100  # Rotate when file exceeds this
    max_files_per_day: int = 24  # Max rotated files per day

    # Compression
    compress_after_days: int = 1  # Compress after this many days


@dataclass
class PersistenceConfig:
    """Configuration for audit persistence."""

    # Storage paths
    base_path: Path = field(default_factory=lambda: Path("data/audit"))
    hot_path: Path = field(default_factory=lambda: Path("data/audit/hot"))
    warm_path: Path = field(default_factory=lambda: Path("data/audit/warm"))
    cold_path: Path = field(default_factory=lambda: Path("data/audit/cold"))

    # Database
    db_path: Path = field(default_factory=lambda: Path("data/audit/audit.db"))

    # Retention
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)

    # Performance
    write_buffer_size: int = 100
    sync_interval: float = 1.0  # Seconds

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        self.base_path = Path(self.base_path)
        self.hot_path = Path(self.hot_path)
        self.warm_path = Path(self.warm_path)
        self.cold_path = Path(self.cold_path)
        self.db_path = Path(self.db_path)


class AuditPersistence:
    """
    Audit log persistence with rotation and retention.

    Provides:
    - File-based storage with automatic rotation
    - SQLite indexing for fast queries
    - Compression of old logs
    - Retention policy enforcement

    Example:
        persistence = AuditPersistence(config=PersistenceConfig())
        persistence.initialize()

        # Write events
        persistence.write_event(audit_event)

        # Query events
        events = persistence.query_events(
            start_time=datetime.now() - timedelta(hours=1),
            event_types=[AuditEventType.ORDER_FILLED],
        )

        # Run retention cleanup
        await persistence.enforce_retention()
    """

    def __init__(self, config: PersistenceConfig | None = None) -> None:
        """Initialize persistence layer."""
        self.config = config or PersistenceConfig()

        # File handles
        self._current_file: Path | None = None
        self._file_handle: Any = None
        self._file_lock = threading.Lock()

        # Write buffer
        self._buffer: list[AuditEvent] = []
        self._buffer_lock = threading.Lock()

        # SQLite connection
        self._db_conn: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()

        # State
        self._initialized = False
        self._current_file_size = 0

    def initialize(self) -> None:
        """Initialize storage directories and database."""
        # Create directories
        for path in [
            self.config.base_path,
            self.config.hot_path,
            self.config.warm_path,
            self.config.cold_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Open current log file
        self._open_log_file()

        self._initialized = True
        logger.info("Audit persistence initialized at %s", self.config.base_path)

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_conn = sqlite3.connect(
            str(self.config.db_path),
            check_same_thread=False,
        )
        self._db_conn.row_factory = sqlite3.Row

        cursor = self._db_conn.cursor()

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sequence_number INTEGER,
                source TEXT,
                actor TEXT,
                session_id TEXT,
                trace_id TEXT,
                message TEXT,
                details TEXT,
                checksum TEXT,
                file_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                symbol TEXT,
                side TEXT,
                quantity TEXT,
                price TEXT,
                status TEXT,
                strategy_name TEXT,
                risk_check_passed INTEGER,
                filled_quantity TEXT,
                filled_price TEXT,
                commission TEXT,
                slippage TEXT,
                created_at TEXT,
                filled_at TEXT,
                session_id TEXT,
                trace_id TEXT,
                data TEXT
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON audit_events(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON audit_events(event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_session
            ON audit_events(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_trace
            ON audit_events(trace_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_symbol
            ON audit_orders(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_status
            ON audit_orders(status)
        """)

        self._db_conn.commit()

    def _open_log_file(self) -> None:
        """Open or rotate log file."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%H")

        file_name = f"audit_{date_str}_{hour_str}.jsonl"
        file_path = self.config.hot_path / file_name

        with self._file_lock:
            if self._file_handle:
                self._file_handle.close()

            self._current_file = file_path
            self._file_handle = open(file_path, "a", encoding="utf-8")
            self._current_file_size = file_path.stat().st_size if file_path.exists() else 0

    def _check_rotation(self) -> None:
        """Check if log rotation is needed."""
        max_size = self.config.retention.max_file_size_mb * 1024 * 1024

        if self._current_file_size >= max_size:
            self._rotate_file()

    def _rotate_file(self) -> None:
        """Rotate current log file."""
        if not self._current_file:
            return

        with self._file_lock:
            if self._file_handle:
                self._file_handle.close()

            # Rename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            new_name = self._current_file.stem + f"_{timestamp}" + self._current_file.suffix
            new_path = self._current_file.parent / new_name

            if self._current_file.exists():
                self._current_file.rename(new_path)

            # Open new file
            self._open_log_file()

    def write_event(self, event: AuditEvent) -> None:
        """
        Write a single audit event.

        Args:
            event: Audit event to write
        """
        if not self._initialized:
            self.initialize()

        # Write to file
        with self._file_lock:
            if self._file_handle:
                line = json.dumps(event.to_dict(), default=str) + "\n"
                self._file_handle.write(line)
                self._file_handle.flush()
                self._current_file_size += len(line.encode())

        # Index in database
        self._index_event(event)

        # Check rotation
        self._check_rotation()

    def write_events(self, events: list[AuditEvent]) -> None:
        """
        Write multiple audit events.

        Args:
            events: List of events to write
        """
        for event in events:
            self.write_event(event)

    async def write_events_async(self, events: list[AuditEvent]) -> None:
        """
        Write events asynchronously.

        Args:
            events: List of events to write
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.write_events, events)

    def _index_event(self, event: AuditEvent) -> None:
        """Index event in SQLite database."""
        if not self._db_conn:
            return

        with self._db_lock:
            cursor = self._db_conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO audit_events
                    (audit_id, event_type, severity, timestamp, sequence_number,
                     source, actor, session_id, trace_id, message, details,
                     checksum, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.audit_id,
                        event.event_type.value,
                        event.severity.value,
                        event.timestamp.isoformat(),
                        event.sequence_number,
                        event.source,
                        event.actor,
                        event.session_id,
                        event.trace_id,
                        event.message,
                        json.dumps(dict(event.details)),
                        event.checksum,
                        str(self._current_file) if self._current_file else "",
                    ),
                )
                self._db_conn.commit()
            except Exception as e:
                logger.error("Failed to index event: %s", e)

    def write_order(self, order: OrderAuditTrail) -> None:
        """
        Write order audit trail.

        Args:
            order: Order audit trail to write
        """
        if not self._db_conn:
            return

        with self._db_lock:
            cursor = self._db_conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO audit_orders
                    (order_id, symbol, side, quantity, price, status,
                     strategy_name, risk_check_passed, filled_quantity,
                     filled_price, commission, slippage, created_at,
                     filled_at, session_id, trace_id, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order.order_id,
                        order.symbol,
                        order.side,
                        str(order.quantity),
                        str(order.price) if order.price else None,
                        order.status,
                        order.strategy_name,
                        1 if order.risk_check_passed else 0,
                        str(order.filled_quantity),
                        str(order.filled_price),
                        str(order.commission),
                        str(order.slippage),
                        order.created_at.isoformat(),
                        order.filled_at.isoformat() if order.filled_at else None,
                        order.session_id,
                        order.trace_id,
                        json.dumps(order.to_dict()),
                    ),
                )
                self._db_conn.commit()
            except Exception as e:
                logger.error("Failed to write order: %s", e)

    def query_events(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        severity: AuditSeverity | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        source: str | None = None,
        search_text: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Query audit events with filters.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event types
            severity: Filter by minimum severity
            session_id: Filter by session
            trace_id: Filter by trace
            source: Filter by source
            search_text: Search in message
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of event dictionaries
        """
        if not self._db_conn:
            return []

        conditions = []
        params: list[Any] = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(et.value for et in event_types)

        if severity:
            conditions.append("severity = ?")
            params.append(severity.value)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        if trace_id:
            conditions.append("trace_id = ?")
            params.append(trace_id)

        if source:
            conditions.append("source = ?")
            params.append(source)

        if search_text:
            conditions.append("message LIKE ?")
            params.append(f"%{search_text}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM audit_events
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def query_orders(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        symbol: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Query order audit trails.

        Args:
            start_time: Filter orders after this time
            end_time: Filter orders before this time
            symbol: Filter by symbol
            status: Filter by status
            session_id: Filter by session
            limit: Maximum results

        Returns:
            List of order dictionaries
        """
        if not self._db_conn:
            return []

        conditions = []
        params: list[Any] = []

        if start_time:
            conditions.append("created_at >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("created_at <= ?")
            params.append(end_time.isoformat())

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        if status:
            conditions.append("status = ?")
            params.append(status)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM audit_orders
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_event_count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Get count of events in time range."""
        if not self._db_conn:
            return 0

        conditions = []
        params: list[Any] = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM audit_events WHERE {where_clause}",
                params,
            )
            return cursor.fetchone()[0]

    async def enforce_retention(self) -> dict[str, int]:
        """
        Enforce retention policy.

        - Compress old hot files
        - Move to warm/cold storage
        - Delete expired files

        Returns:
            Statistics about actions taken
        """
        stats = {
            "compressed": 0,
            "archived": 0,
            "deleted": 0,
        }

        now = datetime.now(timezone.utc)
        policy = self.config.retention

        # Process hot files
        for file_path in self.config.hot_path.glob("*.jsonl"):
            file_age = now - datetime.fromtimestamp(
                file_path.stat().st_mtime, tz=timezone.utc
            )

            if file_age.days >= policy.compress_after_days:
                # Compress and move to warm
                compressed = await self._compress_file(file_path)
                if compressed:
                    dest = self.config.warm_path / (file_path.name + ".gz")
                    shutil.move(str(compressed), str(dest))
                    file_path.unlink(missing_ok=True)
                    stats["compressed"] += 1

        # Process warm files
        for file_path in self.config.warm_path.glob("*.gz"):
            file_age = now - datetime.fromtimestamp(
                file_path.stat().st_mtime, tz=timezone.utc
            )

            if file_age.days >= policy.warm_retention_days:
                # Move to cold
                dest = self.config.cold_path / file_path.name
                shutil.move(str(file_path), str(dest))
                stats["archived"] += 1

        # Process cold files
        for file_path in self.config.cold_path.glob("*.gz"):
            file_age = now - datetime.fromtimestamp(
                file_path.stat().st_mtime, tz=timezone.utc
            )

            if file_age.days >= policy.delete_after_days:
                file_path.unlink()
                stats["deleted"] += 1

        # Clean up database
        cutoff = now - timedelta(days=policy.warm_retention_days)
        with self._db_lock:
            if self._db_conn:
                cursor = self._db_conn.cursor()
                cursor.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff.isoformat(),),
                )
                self._db_conn.commit()

        logger.info(
            "Retention enforced: %d compressed, %d archived, %d deleted",
            stats["compressed"],
            stats["archived"],
            stats["deleted"],
        )

        return stats

    async def _compress_file(self, file_path: Path) -> Path | None:
        """Compress a file with gzip."""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + ".gz")

            with open(file_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            return compressed_path
        except Exception as e:
            logger.error("Failed to compress %s: %s", file_path, e)
            return None

    def read_log_file(self, file_path: Path) -> Iterator[dict[str, Any]]:
        """
        Read events from a log file.

        Args:
            file_path: Path to log file

        Yields:
            Event dictionaries
        """
        opener = gzip.open if file_path.suffix == ".gz" else open

        with opener(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def get_statistics(self) -> dict[str, Any]:
        """Get persistence statistics."""
        stats = {
            "initialized": self._initialized,
            "current_file": str(self._current_file) if self._current_file else None,
            "current_file_size_mb": self._current_file_size / (1024 * 1024),
            "hot_files": len(list(self.config.hot_path.glob("*.jsonl"))),
            "warm_files": len(list(self.config.warm_path.glob("*.gz"))),
            "cold_files": len(list(self.config.cold_path.glob("*.gz"))),
        }

        if self._db_conn:
            with self._db_lock:
                cursor = self._db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM audit_events")
                stats["indexed_events"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM audit_orders")
                stats["indexed_orders"] = cursor.fetchone()[0]

        return stats

    def close(self) -> None:
        """Close persistence layer."""
        with self._file_lock:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None

        with self._db_lock:
            if self._db_conn:
                self._db_conn.close()
                self._db_conn = None

        self._initialized = False


def create_audit_persistence(
    base_path: str | Path = "data/audit",
    hot_retention_days: int = 7,
    cold_retention_days: int = 365,
) -> AuditPersistence:
    """
    Factory function to create audit persistence.

    Args:
        base_path: Base storage path
        hot_retention_days: Days to keep in hot storage
        cold_retention_days: Days to keep in cold storage

    Returns:
        Configured AuditPersistence
    """
    base = Path(base_path)
    config = PersistenceConfig(
        base_path=base,
        hot_path=base / "hot",
        warm_path=base / "warm",
        cold_path=base / "cold",
        db_path=base / "audit.db",
        retention=RetentionPolicy(
            hot_retention_days=hot_retention_days,
            cold_retention_days=cold_retention_days,
        ),
    )
    return AuditPersistence(config)

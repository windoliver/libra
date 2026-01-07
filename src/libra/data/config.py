"""
Configuration for database connections.

Supports QuestDB with multiple connection methods:
- ILP (InfluxDB Line Protocol) for high-throughput ingestion
- PostgreSQL Wire Protocol for queries (asyncpg)
- HTTP endpoint for REST queries

See: https://questdb.com/docs/
"""

from __future__ import annotations

import msgspec


class QuestDBConfig(msgspec.Struct, frozen=True, gc=False):
    """
    QuestDB connection configuration.

    QuestDB exposes multiple protocols:
    - ILP (port 9009): High-throughput ingestion, append-only
    - PGWire (port 8812): PostgreSQL-compatible queries
    - HTTP (port 9000): REST API for queries and management

    Examples:
        # Local development (Docker Compose default)
        config = QuestDBConfig.docker()

        # Custom configuration
        config = QuestDBConfig(
            host="questdb.example.com",
            ilp_port=9009,
            pg_port=8812,
            username="libra",
            password="secret",
            use_tls=True,
        )

        # From environment variables
        config = QuestDBConfig.from_env()
    """

    # Connection settings
    host: str = "localhost"
    ilp_port: int = 9009  # ILP ingestion (TCP)
    pg_port: int = 8812  # PostgreSQL wire protocol
    http_port: int = 9000  # HTTP REST API

    # Authentication (optional)
    username: str | None = None
    password: str | None = None

    # Security
    use_tls: bool = False

    # Connection pool settings (for asyncpg)
    pool_min_size: int = 2
    pool_max_size: int = 10
    command_timeout: float = 30.0  # seconds

    # Ingestion settings
    auto_flush_rows: int = 75000  # Flush after N rows
    auto_flush_interval_ms: int = 1000  # Flush every N ms

    @property
    def ilp_conf(self) -> str:
        """
        Generate ILP connection string for questdb.Sender.

        Format: "http::addr=host:port;username=...;password=...;tls_verify=..."

        See: https://py-questdb-client.readthedocs.io/
        """
        protocol = "https" if self.use_tls else "http"
        conf = f"{protocol}::addr={self.host}:{self.http_port};"

        if self.username:
            conf += f"username={self.username};"
        if self.password:
            conf += f"password={self.password};"

        conf += f"auto_flush_rows={self.auto_flush_rows};"
        conf += f"auto_flush_interval={self.auto_flush_interval_ms};"

        return conf

    @property
    def pg_dsn(self) -> str:
        """
        Generate PostgreSQL DSN for asyncpg.

        Format: "postgresql://user:pass@host:port/qdb"
        """
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""

        return f"postgresql://{auth}{self.host}:{self.pg_port}/qdb"

    @property
    def http_url(self) -> str:
        """HTTP endpoint URL for REST queries."""
        protocol = "https" if self.use_tls else "http"
        return f"{protocol}://{self.host}:{self.http_port}"

    @classmethod
    def docker(cls) -> QuestDBConfig:
        """
        Create config for Docker Compose setup.

        Uses the default credentials from docker-compose.yml:
        - Username: libra
        - Password: libra

        Returns:
            QuestDBConfig configured for local Docker.
        """
        return cls(
            host="localhost",
            username="libra",
            password="libra",
        )

    @classmethod
    def from_env(cls) -> QuestDBConfig:
        """
        Create config from environment variables.

        Environment variables:
        - QUESTDB_HOST (default: localhost)
        - QUESTDB_ILP_PORT (default: 9009)
        - QUESTDB_PG_PORT (default: 8812)
        - QUESTDB_HTTP_PORT (default: 9000)
        - QUESTDB_USERNAME (default: None)
        - QUESTDB_PASSWORD (default: None)
        - QUESTDB_USE_TLS (default: false)

        Returns:
            QuestDBConfig from environment.
        """
        import os

        return cls(
            host=os.getenv("QUESTDB_HOST", "localhost"),
            ilp_port=int(os.getenv("QUESTDB_ILP_PORT", "9009")),
            pg_port=int(os.getenv("QUESTDB_PG_PORT", "8812")),
            http_port=int(os.getenv("QUESTDB_HTTP_PORT", "9000")),
            username=os.getenv("QUESTDB_USERNAME"),
            password=os.getenv("QUESTDB_PASSWORD"),
            use_tls=os.getenv("QUESTDB_USE_TLS", "false").lower() == "true",
        )

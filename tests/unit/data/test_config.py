"""Tests for QuestDB configuration."""

import pytest

from libra.data.config import QuestDBConfig


class TestQuestDBConfig:
    """Tests for QuestDBConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = QuestDBConfig()

        assert config.host == "localhost"
        assert config.ilp_port == 9009
        assert config.pg_port == 8812
        assert config.http_port == 9000
        assert config.username is None
        assert config.password is None
        assert config.use_tls is False
        assert config.pool_min_size == 2
        assert config.pool_max_size == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = QuestDBConfig(
            host="questdb.example.com",
            ilp_port=19009,
            pg_port=18812,
            username="libra",
            password="secret",
            use_tls=True,
        )

        assert config.host == "questdb.example.com"
        assert config.ilp_port == 19009
        assert config.pg_port == 18812
        assert config.username == "libra"
        assert config.password == "secret"
        assert config.use_tls is True

    def test_ilp_conf_without_auth(self) -> None:
        """Test ILP connection string without authentication."""
        config = QuestDBConfig(host="localhost", http_port=9000)

        ilp_conf = config.ilp_conf

        assert "http::addr=localhost:9000;" in ilp_conf
        assert "username=" not in ilp_conf
        assert "password=" not in ilp_conf
        assert "auto_flush_rows=75000;" in ilp_conf
        assert "auto_flush_interval=1000;" in ilp_conf

    def test_ilp_conf_with_auth(self) -> None:
        """Test ILP connection string with authentication."""
        config = QuestDBConfig(
            host="questdb.example.com",
            http_port=9000,
            username="libra",
            password="secret",
        )

        ilp_conf = config.ilp_conf

        assert "http::addr=questdb.example.com:9000;" in ilp_conf
        assert "username=libra;" in ilp_conf
        assert "password=secret;" in ilp_conf

    def test_ilp_conf_with_tls(self) -> None:
        """Test ILP connection string with TLS."""
        config = QuestDBConfig(host="localhost", use_tls=True)

        ilp_conf = config.ilp_conf

        assert ilp_conf.startswith("https::")

    def test_pg_dsn_without_auth(self) -> None:
        """Test PostgreSQL DSN without authentication."""
        config = QuestDBConfig(host="localhost", pg_port=8812)

        assert config.pg_dsn == "postgresql://localhost:8812/qdb"

    def test_pg_dsn_with_auth(self) -> None:
        """Test PostgreSQL DSN with authentication."""
        config = QuestDBConfig(
            host="questdb.example.com",
            pg_port=8812,
            username="libra",
            password="secret",
        )

        assert config.pg_dsn == "postgresql://libra:secret@questdb.example.com:8812/qdb"

    def test_http_url_without_tls(self) -> None:
        """Test HTTP URL without TLS."""
        config = QuestDBConfig(host="localhost", http_port=9000)

        assert config.http_url == "http://localhost:9000"

    def test_http_url_with_tls(self) -> None:
        """Test HTTP URL with TLS."""
        config = QuestDBConfig(host="localhost", http_port=9000, use_tls=True)

        assert config.http_url == "https://localhost:9000"

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable."""
        config = QuestDBConfig()

        with pytest.raises(AttributeError):
            config.host = "new-host"  # type: ignore[misc]

    def test_docker_factory(self) -> None:
        """Test Docker Compose factory method."""
        config = QuestDBConfig.docker()

        assert config.host == "localhost"
        assert config.username == "libra"
        assert config.password == "libra"
        assert config.pg_port == 8812
        assert config.ilp_port == 9009

    def test_from_env_defaults(self) -> None:
        """Test from_env with no environment variables set."""
        import os

        # Clear any existing env vars
        env_vars = [
            "QUESTDB_HOST",
            "QUESTDB_ILP_PORT",
            "QUESTDB_PG_PORT",
            "QUESTDB_HTTP_PORT",
            "QUESTDB_USERNAME",
            "QUESTDB_PASSWORD",
            "QUESTDB_USE_TLS",
        ]
        original = {k: os.environ.pop(k, None) for k in env_vars}

        try:
            config = QuestDBConfig.from_env()

            assert config.host == "localhost"
            assert config.ilp_port == 9009
            assert config.pg_port == 8812
            assert config.http_port == 9000
            assert config.username is None
            assert config.password is None
            assert config.use_tls is False
        finally:
            # Restore original env vars
            for k, v in original.items():
                if v is not None:
                    os.environ[k] = v

    def test_from_env_custom(self) -> None:
        """Test from_env with custom environment variables."""
        import os

        os.environ["QUESTDB_HOST"] = "custom-host"
        os.environ["QUESTDB_USERNAME"] = "custom-user"
        os.environ["QUESTDB_USE_TLS"] = "true"

        try:
            config = QuestDBConfig.from_env()

            assert config.host == "custom-host"
            assert config.username == "custom-user"
            assert config.use_tls is True
        finally:
            del os.environ["QUESTDB_HOST"]
            del os.environ["QUESTDB_USERNAME"]
            del os.environ["QUESTDB_USE_TLS"]

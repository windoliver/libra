"""
Unit tests for IBKR Gateway Configuration.

Tests IBKRConfig validation and factory methods.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from libra.gateways.ibkr.config import IBKRConfig, IBKRCredentials, IBKRPort


class TestIBKRPort:
    """Tests for IBKRPort enum."""

    def test_port_values(self) -> None:
        """Port enum has correct values."""
        assert IBKRPort.TWS_LIVE == 7496
        assert IBKRPort.TWS_PAPER == 7497
        assert IBKRPort.GATEWAY_LIVE == 4001
        assert IBKRPort.GATEWAY_PAPER == 4002


class TestIBKRCredentials:
    """Tests for IBKRCredentials."""

    def test_from_env(self) -> None:
        """from_env creates empty credentials."""
        creds = IBKRCredentials.from_env()
        assert creds is not None


class TestIBKRConfig:
    """Tests for IBKRConfig."""

    def test_default_config(self) -> None:
        """Default config uses paper trading."""
        config = IBKRConfig()
        assert config.host == "127.0.0.1"
        assert config.port == IBKRPort.TWS_PAPER
        assert config.client_id == 1
        assert config.account is None
        assert config.readonly is False
        assert config.is_paper is True

    def test_paper_factory(self) -> None:
        """paper() creates paper trading config."""
        config = IBKRConfig.paper(client_id=5)
        assert config.port == IBKRPort.TWS_PAPER
        assert config.client_id == 5
        assert config.is_paper is True

    def test_live_factory(self) -> None:
        """live() creates live trading config."""
        config = IBKRConfig.live()
        assert config.port == IBKRPort.TWS_LIVE
        assert config.is_paper is False

    def test_gateway_paper_factory(self) -> None:
        """gateway_paper() creates IB Gateway paper config."""
        config = IBKRConfig.gateway_paper()
        assert config.port == IBKRPort.GATEWAY_PAPER
        assert config.is_paper is True
        assert config.is_gateway is True

    def test_gateway_live_factory(self) -> None:
        """gateway_live() creates IB Gateway live config."""
        config = IBKRConfig.gateway_live()
        assert config.port == IBKRPort.GATEWAY_LIVE
        assert config.is_paper is False
        assert config.is_gateway is True

    def test_is_paper_property(self) -> None:
        """is_paper correctly identifies paper ports."""
        assert IBKRConfig(port=7497).is_paper is True
        assert IBKRConfig(port=4002).is_paper is True
        assert IBKRConfig(port=7496).is_paper is False
        assert IBKRConfig(port=4001).is_paper is False

    def test_is_gateway_property(self) -> None:
        """is_gateway correctly identifies gateway ports."""
        assert IBKRConfig(port=7497).is_gateway is False
        assert IBKRConfig(port=7496).is_gateway is False
        assert IBKRConfig(port=4001).is_gateway is True
        assert IBKRConfig(port=4002).is_gateway is True

    def test_invalid_client_id_negative(self) -> None:
        """Negative client_id raises ValueError."""
        with pytest.raises(ValueError, match="client_id must be >= 1"):
            IBKRConfig(client_id=0)

    def test_invalid_client_id_too_large(self) -> None:
        """client_id > 999 raises ValueError."""
        with pytest.raises(ValueError, match="client_id must be <= 999"):
            IBKRConfig(client_id=1000)

    def test_invalid_timeout(self) -> None:
        """Non-positive timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            IBKRConfig(timeout=0)

    def test_invalid_options_level(self) -> None:
        """Invalid options_level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid options_level"):
            IBKRConfig(options_level=5)

    def test_custom_port_warning(self) -> None:
        """Non-standard port triggers warning."""
        with pytest.warns(UserWarning, match="Non-standard port"):
            IBKRConfig(port=9999)

    def test_from_env_defaults(self) -> None:
        """from_env uses defaults when env vars not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = IBKRConfig.from_env()
            assert config.host == "127.0.0.1"
            assert config.port == IBKRPort.TWS_PAPER
            assert config.client_id == 1

    def test_from_env_custom_values(self) -> None:
        """from_env reads custom values from environment."""
        env = {
            "IBKR_HOST": "192.168.1.100",
            "IBKR_PORT": "4002",
            "IBKR_CLIENT_ID": "10",
            "IBKR_ACCOUNT": "DU123456",
            "IBKR_READONLY": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = IBKRConfig.from_env()
            assert config.host == "192.168.1.100"
            assert config.port == 4002
            assert config.client_id == 10
            assert config.account == "DU123456"
            assert config.readonly is True

    def test_from_env_live(self) -> None:
        """from_env with paper=False uses live port."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = IBKRConfig.from_env(paper=False)
            assert config.port == IBKRPort.TWS_LIVE

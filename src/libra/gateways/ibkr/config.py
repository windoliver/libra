"""
Interactive Brokers Gateway Configuration.

Handles TWS/IB Gateway connection settings.
Supports paper and live trading modes via port selection.

Issue #64: Interactive Brokers Gateway - Full Options Lifecycle
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class IBKRPort(int, Enum):
    """Standard IBKR API ports."""

    TWS_LIVE = 7496
    TWS_PAPER = 7497
    GATEWAY_LIVE = 4001
    GATEWAY_PAPER = 4002


@dataclass(frozen=True)
class IBKRCredentials:
    """
    Interactive Brokers credentials.

    Unlike cloud brokers, IBKR doesn't use API keys.
    Authentication is handled by TWS/IB Gateway which must be running.

    For future OAuth or IBKR Lite web API integration, this class
    can be extended with additional authentication methods.
    """

    # Reserved for future auth methods (e.g., IBKR Web API)
    # Currently empty as TWS handles authentication

    @classmethod
    def from_env(cls) -> IBKRCredentials:
        """Create credentials placeholder from environment."""
        return cls()


@dataclass
class IBKRConfig:
    """
    Interactive Brokers Gateway configuration.

    Attributes:
        host: TWS/IB Gateway host (default: localhost)
        port: API port - determines paper vs live trading
        client_id: Unique client ID (1-32 for TWS, 1-999 for Gateway)
        account: Account ID for multi-account setups (optional)
        readonly: Read-only mode - no trading allowed
        timeout: Connection timeout in seconds
        auto_reconnect: Automatically reconnect on disconnect
        max_reconnect_attempts: Maximum reconnection attempts
        reconnect_delay: Base delay between reconnection attempts
        options_level: Options trading level (1-4 for IBKR)

    Port Reference:
        7496 - TWS Live
        7497 - TWS Paper (default)
        4001 - IB Gateway Live
        4002 - IB Gateway Paper

    Example:
        # Paper trading with TWS
        config = IBKRConfig(port=IBKRPort.TWS_PAPER)

        # Live trading with IB Gateway
        config = IBKRConfig(
            port=IBKRPort.GATEWAY_LIVE,
            account="U1234567",
        )
    """

    host: str = "127.0.0.1"
    port: int = IBKRPort.TWS_PAPER
    client_id: int = 1
    account: str | None = None
    readonly: bool = False
    timeout: float = 30.0
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    options_level: int = 2  # 1=covered, 2=single-leg, 3=spreads, 4=uncovered

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_ports = {p.value for p in IBKRPort}
        if self.port not in valid_ports:
            # Allow custom ports but warn
            import warnings

            warnings.warn(
                f"Non-standard port {self.port}. "
                f"Standard ports: {', '.join(f'{p.name}={p.value}' for p in IBKRPort)}",
                UserWarning,
                stacklevel=2,
            )

        if self.client_id < 1:
            raise ValueError("client_id must be >= 1")
        if self.client_id > 999:
            raise ValueError("client_id must be <= 999 (TWS allows 1-32)")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        if self.options_level not in (0, 1, 2, 3, 4):
            raise ValueError(f"Invalid options_level: {self.options_level}. Must be 0-4")

    @property
    def is_paper(self) -> bool:
        """Check if configured for paper trading."""
        return self.port in (IBKRPort.TWS_PAPER, IBKRPort.GATEWAY_PAPER)

    @property
    def is_gateway(self) -> bool:
        """Check if using IB Gateway (vs TWS)."""
        return self.port in (IBKRPort.GATEWAY_LIVE, IBKRPort.GATEWAY_PAPER)

    @classmethod
    def from_env(cls, paper: bool = True) -> IBKRConfig:
        """
        Create config from environment variables.

        Environment Variables:
            IBKR_HOST: TWS/Gateway host (default: 127.0.0.1)
            IBKR_PORT: API port (default: 7497 for paper, 7496 for live)
            IBKR_CLIENT_ID: Client ID (default: 1)
            IBKR_ACCOUNT: Account ID for multi-account (optional)
            IBKR_READONLY: Set to "true" for read-only mode

        Args:
            paper: Use paper trading port (default: True)

        Returns:
            IBKRConfig instance
        """
        host = os.environ.get("IBKR_HOST", "127.0.0.1")

        # Determine port
        port_env = os.environ.get("IBKR_PORT")
        if port_env:
            port = int(port_env)
        else:
            port = IBKRPort.TWS_PAPER if paper else IBKRPort.TWS_LIVE

        client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))
        account = os.environ.get("IBKR_ACCOUNT")
        readonly = os.environ.get("IBKR_READONLY", "").lower() == "true"

        return cls(
            host=host,
            port=port,
            client_id=client_id,
            account=account,
            readonly=readonly,
        )

    @classmethod
    def paper(cls, **kwargs) -> IBKRConfig:
        """Create paper trading config (convenience method)."""
        return cls(port=IBKRPort.TWS_PAPER, **kwargs)

    @classmethod
    def live(cls, **kwargs) -> IBKRConfig:
        """Create live trading config (convenience method)."""
        return cls(port=IBKRPort.TWS_LIVE, **kwargs)

    @classmethod
    def gateway_paper(cls, **kwargs) -> IBKRConfig:
        """Create IB Gateway paper config (for servers)."""
        return cls(port=IBKRPort.GATEWAY_PAPER, **kwargs)

    @classmethod
    def gateway_live(cls, **kwargs) -> IBKRConfig:
        """Create IB Gateway live config (for servers)."""
        return cls(port=IBKRPort.GATEWAY_LIVE, **kwargs)

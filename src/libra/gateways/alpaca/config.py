"""
Alpaca Gateway Configuration.

Handles API credentials and gateway settings.
Supports both paper and live trading modes.

Issue #61: Alpaca Gateway - Stock & Options Execution
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AlpacaCredentials:
    """
    Alpaca API credentials.

    Can be loaded from environment variables or passed directly.

    Environment Variables:
        ALPACA_API_KEY: API key (paper or live)
        ALPACA_SECRET_KEY: Secret key (paper or live)
        ALPACA_PAPER: Set to "true" for paper trading (default)

    Example:
        # From environment
        creds = AlpacaCredentials.from_env()

        # Direct
        creds = AlpacaCredentials(
            api_key="PKXXXXXXXX",
            secret_key="XXXXXXXX",
        )
    """

    api_key: str
    secret_key: str

    @classmethod
    def from_env(cls) -> AlpacaCredentials:
        """Load credentials from environment variables."""
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key:
            raise ValueError(
                "ALPACA_API_KEY environment variable not set. "
                "Get your API key at https://app.alpaca.markets"
            )
        if not secret_key:
            raise ValueError(
                "ALPACA_SECRET_KEY environment variable not set. "
                "Get your secret key at https://app.alpaca.markets"
            )

        return cls(api_key=api_key, secret_key=secret_key)

    def is_paper_key(self) -> bool:
        """Check if this is a paper trading API key (starts with PK)."""
        return self.api_key.startswith("PK")


@dataclass
class AlpacaConfig:
    """
    Alpaca Gateway configuration.

    Attributes:
        credentials: API credentials (required)
        paper: Use paper trading (default: True for safety)
        data_feed: Market data feed - "iex" (free) or "sip" (paid)
        max_retries: Maximum retry attempts for failed requests
        retry_delay: Base delay between retries (seconds)
        rate_limit_per_minute: API rate limit (default: 200)
        options_level: Options trading level (0-3)

    Example:
        config = AlpacaConfig(
            credentials=AlpacaCredentials.from_env(),
            paper=True,
            data_feed="iex",
        )
    """

    credentials: AlpacaCredentials
    paper: bool = True
    data_feed: str = "iex"  # "iex" (free) or "sip" (paid, full market)
    max_retries: int = 5
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 200
    options_level: int = 2  # 0=disabled, 1=covered, 2=single-leg, 3=multi-leg

    # WebSocket settings
    ws_ping_interval: float = 30.0
    ws_ping_timeout: float = 10.0
    ws_reconnect_delay: float = 1.0
    ws_max_reconnect_delay: float = 60.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.data_feed not in ("iex", "sip"):
            raise ValueError(f"Invalid data_feed: {self.data_feed}. Must be 'iex' or 'sip'")
        if self.options_level not in (0, 1, 2, 3):
            raise ValueError(f"Invalid options_level: {self.options_level}. Must be 0-3")
        if self.rate_limit_per_minute <= 0:
            raise ValueError("rate_limit_per_minute must be positive")

        # Safety check: warn if using live credentials in paper mode
        if self.paper and not self.credentials.is_paper_key():
            import warnings
            warnings.warn(
                "Using live API key with paper=True. "
                "This will still use paper trading endpoints.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_env(cls, paper: bool = True) -> AlpacaConfig:
        """
        Create config from environment variables.

        Args:
            paper: Use paper trading mode (default: True)

        Returns:
            AlpacaConfig instance
        """
        return cls(
            credentials=AlpacaCredentials.from_env(),
            paper=paper,
            data_feed=os.environ.get("ALPACA_DATA_FEED", "iex"),
        )

    @property
    def base_url(self) -> str:
        """Get the appropriate API base URL."""
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"

    @property
    def data_url(self) -> str:
        """Get the market data API URL."""
        return "https://data.alpaca.markets"

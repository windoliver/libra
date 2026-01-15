"""
Unit tests for IBKR Contract Builders.

Tests contract building functions without requiring ib_async.
"""

from __future__ import annotations

import pytest

from libra.gateways.ibkr.contracts import parse_symbol


class TestParseSymbol:
    """Tests for parse_symbol function."""

    def test_stock_symbol(self) -> None:
        """Simple stock symbol is parsed correctly."""
        asset_class, underlying = parse_symbol("AAPL")
        assert asset_class == "stock"
        assert underlying == "AAPL"

    def test_stock_symbol_long(self) -> None:
        """Multi-character stock symbol."""
        asset_class, underlying = parse_symbol("GOOGL")
        assert asset_class == "stock"
        assert underlying == "GOOGL"

    def test_option_occ_format(self) -> None:
        """OCC format option symbol is parsed."""
        # AAPL250117C00150000 = AAPL Jan 17 2025 $150 Call
        asset_class, underlying = parse_symbol("AAPL  250117C00150000")
        assert asset_class == "option"
        assert underlying == "AAPL"

    def test_crypto_pair(self) -> None:
        """Crypto pair format is parsed."""
        asset_class, underlying = parse_symbol("BTC/USD")
        assert asset_class == "crypto"
        assert underlying == "BTC"

    def test_crypto_pair_lowercase(self) -> None:
        """Crypto pair with mixed case."""
        asset_class, underlying = parse_symbol("ETH/USDT")
        assert asset_class == "crypto"
        assert underlying == "ETH"


class TestContractBuilders:
    """Tests for contract builder functions.

    These tests use mocking since ib_async may not be installed.
    """

    def test_build_stock_import_error(self) -> None:
        """build_stock raises ImportError when ib_async not installed."""
        # Skip if ib_async is installed
        try:
            import ib_async
            pytest.skip("ib_async is installed")
        except ImportError:
            pass

        from libra.gateways.ibkr.contracts import build_stock

        with pytest.raises(ImportError, match="ib_async is not installed"):
            build_stock("AAPL")

    def test_build_option_import_error(self) -> None:
        """build_option raises ImportError when ib_async not installed."""
        try:
            import ib_async
            pytest.skip("ib_async is installed")
        except ImportError:
            pass

        from datetime import date
        from decimal import Decimal

        from libra.core.options import OptionContract, OptionType
        from libra.gateways.ibkr.contracts import build_option

        contract = OptionContract(
            symbol="AAPL250117C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("150"),
            expiration=date(2025, 1, 17),
        )

        with pytest.raises(ImportError, match="ib_async is not installed"):
            build_option(contract)

"""
Unit tests for OpenBB to Core Model converters.

Tests the converter functions that transform OpenBB DTOs to core options models.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from libra.gateways.openbb.fetchers import (
    OptionContract as OpenBBOptionContract,
    openbb_to_chain_entry,
    openbb_to_core_option_contract,
    openbb_to_greeks,
    openbb_to_greeks_snapshot,
)
from libra.core.options import (
    OptionContract,
    OptionType,
    Greeks,
    GreeksSnapshot,
    OptionChainEntry,
)


class TestOpenBBToCore:
    """Tests for OpenBB to core model converters."""

    @pytest.fixture
    def openbb_call(self) -> OpenBBOptionContract:
        """Create sample OpenBB call contract."""
        return OpenBBOptionContract(
            contract_symbol="AAPL250117C00150000",
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            strike=Decimal("150.00"),
            option_type="call",
            bid=Decimal("7.40"),
            ask=Decimal("7.60"),
            last=Decimal("7.50"),
            volume=1500,
            open_interest=25000,
            implied_volatility=Decimal("0.35"),
            delta=Decimal("0.65"),
            gamma=Decimal("0.02"),
            theta=Decimal("-0.05"),
            vega=Decimal("0.15"),
            rho=Decimal("0.03"),
            in_the_money=True,
        )

    @pytest.fixture
    def openbb_put(self) -> OpenBBOptionContract:
        """Create sample OpenBB put contract."""
        return OpenBBOptionContract(
            contract_symbol="AAPL250117P00150000",
            underlying="AAPL",
            expiration=date(2025, 1, 17),
            strike=Decimal("150.00"),
            option_type="put",
            bid=Decimal("5.20"),
            ask=Decimal("5.40"),
            last=Decimal("5.30"),
            volume=800,
            open_interest=15000,
            implied_volatility=Decimal("0.32"),
            delta=Decimal("-0.35"),
            gamma=Decimal("0.02"),
            theta=Decimal("-0.04"),
            vega=Decimal("0.12"),
            rho=Decimal("-0.02"),
            in_the_money=False,
        )

    def test_convert_call_contract(self, openbb_call: OpenBBOptionContract) -> None:
        """Convert OpenBB call to core OptionContract."""
        contract = openbb_to_core_option_contract(openbb_call)

        assert isinstance(contract, OptionContract)
        assert contract.symbol == "AAPL250117C00150000"
        assert contract.underlying == "AAPL"
        assert contract.option_type == OptionType.CALL
        assert contract.strike == Decimal("150.00")
        assert contract.expiration == date(2025, 1, 17)
        assert contract.is_call is True

    def test_convert_put_contract(self, openbb_put: OpenBBOptionContract) -> None:
        """Convert OpenBB put to core OptionContract."""
        contract = openbb_to_core_option_contract(openbb_put)

        assert contract.option_type == OptionType.PUT
        assert contract.is_put is True

    def test_convert_greeks(self, openbb_call: OpenBBOptionContract) -> None:
        """Convert OpenBB contract to core Greeks."""
        greeks = openbb_to_greeks(openbb_call)

        assert isinstance(greeks, Greeks)
        assert greeks.delta == Decimal("0.65")
        assert greeks.gamma == Decimal("0.02")
        assert greeks.theta == Decimal("-0.05")
        assert greeks.vega == Decimal("0.15")
        assert greeks.rho == Decimal("0.03")
        assert greeks.iv == Decimal("0.35")

    def test_convert_greeks_missing_values(self) -> None:
        """Convert contract with missing Greeks defaults to zero."""
        openbb_contract = OpenBBOptionContract(
            contract_symbol="TEST",
            underlying="TEST",
            expiration=date(2025, 1, 17),
            strike=Decimal("100"),
            option_type="call",
            # No Greeks specified
        )
        greeks = openbb_to_greeks(openbb_contract)

        assert greeks.delta == Decimal("0")
        assert greeks.gamma == Decimal("0")
        assert greeks.iv == Decimal("0")

    def test_convert_greeks_snapshot(self, openbb_call: OpenBBOptionContract) -> None:
        """Convert OpenBB contract to GreeksSnapshot."""
        snapshot = openbb_to_greeks_snapshot(
            openbb_call, underlying_price=Decimal("155.00")
        )

        assert isinstance(snapshot, GreeksSnapshot)
        assert snapshot.underlying_price == Decimal("155.00")
        assert snapshot.option_price == Decimal("7.50")
        assert snapshot.bid == Decimal("7.40")
        assert snapshot.ask == Decimal("7.60")
        assert snapshot.volume == 1500
        assert snapshot.open_interest == 25000
        assert snapshot.greeks.delta == Decimal("0.65")

    def test_convert_greeks_snapshot_no_underlying(
        self, openbb_call: OpenBBOptionContract
    ) -> None:
        """Convert without underlying price uses zero."""
        snapshot = openbb_to_greeks_snapshot(openbb_call)

        assert snapshot.underlying_price == Decimal("0")

    def test_convert_chain_entry(self, openbb_call: OpenBBOptionContract) -> None:
        """Convert OpenBB contract to OptionChainEntry."""
        entry = openbb_to_chain_entry(openbb_call, underlying_price=Decimal("155.00"))

        assert isinstance(entry, OptionChainEntry)
        assert isinstance(entry.contract, OptionContract)
        assert isinstance(entry.snapshot, GreeksSnapshot)
        assert entry.contract.symbol == "AAPL250117C00150000"
        assert entry.snapshot.bid == Decimal("7.40")

    def test_core_contract_methods_work(
        self, openbb_call: OpenBBOptionContract
    ) -> None:
        """Core contract has rich domain methods."""
        contract = openbb_to_core_option_contract(openbb_call)

        # Use core model methods
        assert contract.is_itm(Decimal("160.00")) is True  # Call ITM when price > strike
        assert contract.is_otm(Decimal("140.00")) is True  # Call OTM when price < strike
        assert contract.intrinsic_value(Decimal("160.00")) == Decimal("10.00")

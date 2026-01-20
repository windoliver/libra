"""
Order Reconciliation: Sync local state with venue state at startup.

Provides reconciliation functionality to:
- Detect orders placed before restart that are unknown to the system
- Sync positions that may be out of sync with actual venue positions
- Ensure risk limits are calculated correctly after recovery

Design references:
- NautilusTrader ExecutionEngine reconciliation
- Issue: https://github.com/windoliver/libra/issues/109
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libra.clients.execution_client import ExecutionClient
    from libra.core.cache import Cache
    from libra.gateways.protocol import OrderResult, Position
    from libra.risk.engine import RiskEngine


logger = logging.getLogger(__name__)


# =============================================================================
# Reconciliation Types
# =============================================================================


class ReconciliationAction(Enum):
    """Action taken during reconciliation."""

    NONE = "none"  # No action needed
    ADD_LOCAL = "add_local"  # Order/position exists on venue, not locally
    REMOVE_LOCAL = "remove_local"  # Order/position exists locally, not on venue
    UPDATE_LOCAL = "update_local"  # Exists both places, different state


@dataclass
class Discrepancy:
    """Record of a single reconciliation discrepancy."""

    action: ReconciliationAction
    entity_type: str  # "order" or "position"
    identifier: str  # Order ID or symbol
    details: str
    venue_state: str | None = None
    local_state: str | None = None


@dataclass
class ReconciliationResult:
    """Result of the reconciliation process."""

    # Counts
    orders_added: int = 0
    orders_removed: int = 0
    orders_updated: int = 0
    positions_added: int = 0
    positions_removed: int = 0
    positions_adjusted: int = 0

    # Details
    discrepancies: list[Discrepancy] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Status
    success: bool = True

    @property
    def total_discrepancies(self) -> int:
        """Total number of discrepancies found."""
        return len(self.discrepancies)

    @property
    def had_changes(self) -> bool:
        """Check if any reconciliation changes were made."""
        return (
            self.orders_added > 0
            or self.orders_removed > 0
            or self.orders_updated > 0
            or self.positions_added > 0
            or self.positions_removed > 0
            or self.positions_adjusted > 0
        )

    def summary(self) -> str:
        """Generate a summary string of reconciliation results."""
        parts = []

        if self.orders_added > 0:
            parts.append(f"orders_added={self.orders_added}")
        if self.orders_removed > 0:
            parts.append(f"orders_removed={self.orders_removed}")
        if self.orders_updated > 0:
            parts.append(f"orders_updated={self.orders_updated}")
        if self.positions_added > 0:
            parts.append(f"positions_added={self.positions_added}")
        if self.positions_removed > 0:
            parts.append(f"positions_removed={self.positions_removed}")
        if self.positions_adjusted > 0:
            parts.append(f"positions_adjusted={self.positions_adjusted}")

        if not parts:
            return "No discrepancies found"

        return ", ".join(parts)


# =============================================================================
# Order Reconciler
# =============================================================================


class OrderReconciler:
    """
    Reconciles local order/position state with venue state.

    Call at startup after connecting to venue to ensure:
    - All orders on venue are tracked locally
    - Stale local orders (filled/cancelled while down) are removed
    - Positions match venue positions
    - Risk engine has accurate open order tracking

    Example:
        reconciler = OrderReconciler(
            execution_client=client,
            cache=cache,
            risk_engine=risk_engine,  # optional
        )

        result = await reconciler.reconcile()

        if result.had_changes:
            logger.warning("Reconciliation found %d discrepancies", result.total_discrepancies)
            for d in result.discrepancies:
                logger.info("  %s %s: %s", d.action.value, d.entity_type, d.details)
    """

    def __init__(
        self,
        execution_client: ExecutionClient,
        cache: Cache,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        """
        Initialize the reconciler.

        Args:
            execution_client: Client for fetching venue state
            cache: Local cache to reconcile
            risk_engine: Optional risk engine for open order tracking
        """
        self._client = execution_client
        self._cache = cache
        self._risk_engine = risk_engine

    async def reconcile(
        self,
        reconcile_orders: bool = True,
        reconcile_positions: bool = True,
    ) -> ReconciliationResult:
        """
        Reconcile all orders and positions with venue.

        Call at startup after connecting to venue.

        Args:
            reconcile_orders: Whether to reconcile orders (default True)
            reconcile_positions: Whether to reconcile positions (default True)

        Returns:
            ReconciliationResult with details of all changes made
        """
        result = ReconciliationResult()

        logger.info("Starting reconciliation with venue: %s", self._client.name)

        try:
            if reconcile_orders:
                await self._reconcile_orders(result)

            if reconcile_positions:
                await self._reconcile_positions(result)

        except Exception as e:
            result.success = False
            result.errors.append(f"Reconciliation failed: {e}")
            logger.exception("Reconciliation failed")
            return result

        if result.had_changes:
            logger.warning(
                "Reconciliation completed with changes: %s",
                result.summary(),
            )
        else:
            logger.info("Reconciliation completed: no discrepancies found")

        return result

    async def _reconcile_orders(self, result: ReconciliationResult) -> None:
        """
        Sync order state with venue.

        - Orders on venue but not local → add to cache
        - Orders local but not on venue → remove from cache (filled/cancelled while down)
        - Orders on both with different state → update local to match venue
        """
        # Get venue state
        venue_orders = await self._client.get_open_orders()
        venue_order_map: dict[str, OrderResult] = {}

        for order in venue_orders:
            # Use client_order_id if available, otherwise order_id
            key = order.client_order_id or order.order_id
            venue_order_map[key] = order

        # Get local state
        local_orders = self._cache.open_orders()
        local_order_map: dict[str, OrderResult] = {}

        for order in local_orders:
            key = order.client_order_id or order.order_id
            local_order_map[key] = order

        venue_ids = set(venue_order_map.keys())
        local_ids = set(local_order_map.keys())

        # Orders on venue but not local → add
        for order_id in venue_ids - local_ids:
            order = venue_order_map[order_id]
            await self._add_order_locally(order, result)

        # Orders local but not on venue → remove (filled/cancelled while down)
        for order_id in local_ids - venue_ids:
            order = local_order_map[order_id]
            await self._remove_stale_order(order, result)

        # Orders on both → check for state differences
        for order_id in venue_ids & local_ids:
            venue_order = venue_order_map[order_id]
            local_order = local_order_map[order_id]
            await self._check_order_state(venue_order, local_order, result)

        logger.debug(
            "Order reconciliation: added=%d, removed=%d, updated=%d",
            result.orders_added,
            result.orders_removed,
            result.orders_updated,
        )

    async def _add_order_locally(
        self,
        order: OrderResult,
        result: ReconciliationResult,
    ) -> None:
        """Add a venue order that's missing locally."""
        # Add to cache
        await self._cache.add_order(order)

        # Add to risk engine open order tracking
        if self._risk_engine is not None:
            # Create a minimal Order object for risk tracking
            from libra.gateways.protocol import Order

            tracking_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                amount=order.amount,
                price=order.price,
                client_order_id=order.client_order_id,
            )
            self._risk_engine.add_open_order(tracking_order)

        result.orders_added += 1
        result.discrepancies.append(
            Discrepancy(
                action=ReconciliationAction.ADD_LOCAL,
                entity_type="order",
                identifier=order.order_id,
                details=f"Added missing order: {order.symbol} {order.side.value} "
                f"{order.amount} @ {order.price or 'MARKET'}",
                venue_state=order.status.value,
                local_state=None,
            )
        )

        logger.info(
            "Reconciliation: added missing order %s (%s %s %s)",
            order.order_id,
            order.symbol,
            order.side.value,
            order.amount,
        )

    async def _remove_stale_order(
        self,
        order: OrderResult,
        result: ReconciliationResult,
    ) -> None:
        """Remove a local order that no longer exists on venue."""
        # Note: Cache doesn't have a remove_order method currently
        # The order will naturally age out or we update its status
        # For now, log the discrepancy - actual removal would require cache enhancement

        # Remove from risk engine tracking
        if self._risk_engine is not None:
            self._risk_engine.remove_open_order(
                order.symbol,
                order_id=order.order_id,
                client_order_id=order.client_order_id,
            )

        result.orders_removed += 1
        result.discrepancies.append(
            Discrepancy(
                action=ReconciliationAction.REMOVE_LOCAL,
                entity_type="order",
                identifier=order.order_id,
                details=f"Removed stale order: {order.symbol} {order.side.value} "
                f"(likely filled/cancelled while down)",
                venue_state=None,
                local_state=order.status.value,
            )
        )

        logger.info(
            "Reconciliation: removed stale order %s (%s) - "
            "likely filled/cancelled while disconnected",
            order.order_id,
            order.symbol,
        )

    async def _check_order_state(
        self,
        venue_order: OrderResult,
        local_order: OrderResult,
        result: ReconciliationResult,
    ) -> None:
        """Check if order state differs between venue and local."""
        # Check for status differences
        if venue_order.status != local_order.status:
            # Update local to match venue
            await self._cache.add_order(venue_order)

            result.orders_updated += 1
            result.discrepancies.append(
                Discrepancy(
                    action=ReconciliationAction.UPDATE_LOCAL,
                    entity_type="order",
                    identifier=venue_order.order_id,
                    details=f"Updated order status: {local_order.status.value} → "
                    f"{venue_order.status.value}",
                    venue_state=venue_order.status.value,
                    local_state=local_order.status.value,
                )
            )

            logger.info(
                "Reconciliation: updated order %s status: %s → %s",
                venue_order.order_id,
                local_order.status.value,
                venue_order.status.value,
            )

        # Check for fill amount differences
        elif venue_order.filled_amount != local_order.filled_amount:
            await self._cache.add_order(venue_order)

            result.orders_updated += 1
            result.discrepancies.append(
                Discrepancy(
                    action=ReconciliationAction.UPDATE_LOCAL,
                    entity_type="order",
                    identifier=venue_order.order_id,
                    details=f"Updated order filled: {local_order.filled_amount} → "
                    f"{venue_order.filled_amount}",
                    venue_state=str(venue_order.filled_amount),
                    local_state=str(local_order.filled_amount),
                )
            )

            logger.info(
                "Reconciliation: updated order %s filled amount: %s → %s",
                venue_order.order_id,
                local_order.filled_amount,
                venue_order.filled_amount,
            )

    async def _reconcile_positions(self, result: ReconciliationResult) -> None:
        """
        Sync position state with venue.

        - Positions on venue but not local → add
        - Positions local but not on venue → remove (closed while down)
        - Positions on both with different amounts → adjust local
        """
        # Get venue positions
        venue_positions = await self._client.get_positions()
        venue_position_map: dict[str, Position] = {
            pos.symbol: pos for pos in venue_positions if pos.amount != Decimal("0")
        }

        # Get local positions
        local_positions = self._cache.positions()
        local_position_map: dict[str, Position] = {
            pos.symbol: pos for pos in local_positions
        }

        venue_symbols = set(venue_position_map.keys())
        local_symbols = set(local_position_map.keys())

        # Positions on venue but not local → add
        for symbol in venue_symbols - local_symbols:
            position = venue_position_map[symbol]
            await self._add_position_locally(position, result)

        # Positions local but not on venue → remove
        for symbol in local_symbols - venue_symbols:
            position = local_position_map[symbol]
            await self._remove_position_locally(position, result)

        # Positions on both → check for amount differences
        for symbol in venue_symbols & local_symbols:
            venue_pos = venue_position_map[symbol]
            local_pos = local_position_map[symbol]
            await self._check_position_state(venue_pos, local_pos, result)

        logger.debug(
            "Position reconciliation: added=%d, removed=%d, adjusted=%d",
            result.positions_added,
            result.positions_removed,
            result.positions_adjusted,
        )

    async def _add_position_locally(
        self,
        position: Position,
        result: ReconciliationResult,
    ) -> None:
        """Add a venue position that's missing locally."""
        await self._cache.update_position(position)

        result.positions_added += 1
        result.discrepancies.append(
            Discrepancy(
                action=ReconciliationAction.ADD_LOCAL,
                entity_type="position",
                identifier=position.symbol,
                details=f"Added missing position: {position.symbol} "
                f"{position.side.value} {position.amount}",
                venue_state=str(position.amount),
                local_state=None,
            )
        )

        logger.info(
            "Reconciliation: added missing position %s (%s %s)",
            position.symbol,
            position.side.value,
            position.amount,
        )

    async def _remove_position_locally(
        self,
        position: Position,
        result: ReconciliationResult,
    ) -> None:
        """Remove a local position that no longer exists on venue."""
        # Create a zero-amount position to trigger removal in cache
        from libra.gateways.protocol import Position as Pos, PositionSide

        zero_position = Pos(
            symbol=position.symbol,
            side=PositionSide.FLAT,
            amount=Decimal("0"),
            entry_price=Decimal("0"),
            current_price=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )
        await self._cache.update_position(zero_position)

        result.positions_removed += 1
        result.discrepancies.append(
            Discrepancy(
                action=ReconciliationAction.REMOVE_LOCAL,
                entity_type="position",
                identifier=position.symbol,
                details=f"Removed stale position: {position.symbol} "
                f"(likely closed while down)",
                venue_state=None,
                local_state=str(position.amount),
            )
        )

        logger.info(
            "Reconciliation: removed stale position %s - "
            "likely closed while disconnected",
            position.symbol,
        )

    async def _check_position_state(
        self,
        venue_pos: Position,
        local_pos: Position,
        result: ReconciliationResult,
    ) -> None:
        """Check if position state differs between venue and local."""
        # Check for amount differences
        if venue_pos.amount != local_pos.amount:
            await self._cache.update_position(venue_pos)

            result.positions_adjusted += 1
            result.discrepancies.append(
                Discrepancy(
                    action=ReconciliationAction.UPDATE_LOCAL,
                    entity_type="position",
                    identifier=venue_pos.symbol,
                    details=f"Adjusted position amount: {local_pos.amount} → "
                    f"{venue_pos.amount}",
                    venue_state=str(venue_pos.amount),
                    local_state=str(local_pos.amount),
                )
            )

            logger.info(
                "Reconciliation: adjusted position %s amount: %s → %s",
                venue_pos.symbol,
                local_pos.amount,
                venue_pos.amount,
            )

        # Check for side differences (long vs short)
        elif venue_pos.side != local_pos.side:
            await self._cache.update_position(venue_pos)

            result.positions_adjusted += 1
            result.discrepancies.append(
                Discrepancy(
                    action=ReconciliationAction.UPDATE_LOCAL,
                    entity_type="position",
                    identifier=venue_pos.symbol,
                    details=f"Adjusted position side: {local_pos.side.value} → "
                    f"{venue_pos.side.value}",
                    venue_state=venue_pos.side.value,
                    local_state=local_pos.side.value,
                )
            )

            logger.info(
                "Reconciliation: adjusted position %s side: %s → %s",
                venue_pos.symbol,
                local_pos.side.value,
                venue_pos.side.value,
            )

    async def reconcile_single_order(self, order_id: str, symbol: str) -> Discrepancy | None:
        """
        Reconcile a single order by ID.

        Useful for reconciling a specific order after a suspected miss.

        Args:
            order_id: Order ID to reconcile
            symbol: Order symbol (required by some exchanges)

        Returns:
            Discrepancy if any change was made, None otherwise
        """
        try:
            venue_order = await self._client.get_order(order_id, symbol)
        except Exception as e:
            logger.warning(
                "Failed to fetch order %s from venue: %s",
                order_id,
                e,
            )
            return None

        local_order = self._cache.order_by_exchange_id(order_id)

        # Order exists on venue but not local
        if local_order is None:
            await self._cache.add_order(venue_order)

            if self._risk_engine is not None and venue_order.is_open:
                from libra.gateways.protocol import Order

                tracking_order = Order(
                    symbol=venue_order.symbol,
                    side=venue_order.side,
                    order_type=venue_order.order_type,
                    amount=venue_order.amount,
                    price=venue_order.price,
                    client_order_id=venue_order.client_order_id,
                )
                self._risk_engine.add_open_order(tracking_order)

            return Discrepancy(
                action=ReconciliationAction.ADD_LOCAL,
                entity_type="order",
                identifier=order_id,
                details=f"Added missing order from single reconciliation",
                venue_state=venue_order.status.value,
                local_state=None,
            )

        # Check for differences
        if venue_order.status != local_order.status:
            await self._cache.add_order(venue_order)
            return Discrepancy(
                action=ReconciliationAction.UPDATE_LOCAL,
                entity_type="order",
                identifier=order_id,
                details=f"Updated from single reconciliation: "
                f"{local_order.status.value} → {venue_order.status.value}",
                venue_state=venue_order.status.value,
                local_state=local_order.status.value,
            )

        return None

    async def reconcile_single_position(self, symbol: str) -> Discrepancy | None:
        """
        Reconcile a single position by symbol.

        Args:
            symbol: Symbol to reconcile

        Returns:
            Discrepancy if any change was made, None otherwise
        """
        try:
            venue_pos = await self._client.get_position(symbol)
        except Exception as e:
            logger.warning(
                "Failed to fetch position %s from venue: %s",
                symbol,
                e,
            )
            return None

        local_pos = self._cache.position(symbol)

        # Position exists on venue but not local
        if venue_pos is not None and local_pos is None:
            if venue_pos.amount != Decimal("0"):
                await self._cache.update_position(venue_pos)
                return Discrepancy(
                    action=ReconciliationAction.ADD_LOCAL,
                    entity_type="position",
                    identifier=symbol,
                    details=f"Added missing position from single reconciliation",
                    venue_state=str(venue_pos.amount),
                    local_state=None,
                )

        # Position exists local but not on venue (or zero on venue)
        if local_pos is not None and (venue_pos is None or venue_pos.amount == Decimal("0")):
            from libra.gateways.protocol import Position as Pos, PositionSide

            zero_position = Pos(
                symbol=symbol,
                side=PositionSide.FLAT,
                amount=Decimal("0"),
                entry_price=Decimal("0"),
                current_price=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
            )
            await self._cache.update_position(zero_position)
            return Discrepancy(
                action=ReconciliationAction.REMOVE_LOCAL,
                entity_type="position",
                identifier=symbol,
                details=f"Removed stale position from single reconciliation",
                venue_state="0",
                local_state=str(local_pos.amount),
            )

        # Both exist - check for differences
        if venue_pos is not None and local_pos is not None:
            if venue_pos.amount != local_pos.amount:
                await self._cache.update_position(venue_pos)
                return Discrepancy(
                    action=ReconciliationAction.UPDATE_LOCAL,
                    entity_type="position",
                    identifier=symbol,
                    details=f"Adjusted from single reconciliation: "
                    f"{local_pos.amount} → {venue_pos.amount}",
                    venue_state=str(venue_pos.amount),
                    local_state=str(local_pos.amount),
                )

        return None

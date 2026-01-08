"""
Strategy Tree Widget.

Hierarchical tree view for strategies with expandable nodes.

Structure:
    ▼ Strategy Name
      ├─ Symbol 1
      │  └─ Position: LONG 0.1 @ $42,500
      ├─ Symbol 2
      │  └─ (no position)
      └─ Metrics
         ├─ P&L: +$1,250
         └─ Trades: 42

Features:
- Expandable/collapsible nodes
- Status icons at each level
- Position details inline
- Keyboard navigation (vim-style)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode


if TYPE_CHECKING:
    pass


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PositionInfo:
    """Position information for display."""

    symbol: str
    side: str  # LONG, SHORT, FLAT
    size: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    @property
    def display_text(self) -> str:
        """Format position for display."""
        if self.side == "FLAT" or self.size == 0:
            return "[dim](no position)[/dim]"
        pnl_color = "green" if self.unrealized_pnl >= 0 else "red"
        pnl_sign = "+" if self.unrealized_pnl >= 0 else ""
        return (
            f"{self.side} {self.size} @ ${self.entry_price:,.2f} "
            f"[{pnl_color}]{pnl_sign}${self.unrealized_pnl:,.2f}[/{pnl_color}]"
        )


@dataclass
class StrategyInfo:
    """Strategy information for tree display."""

    strategy_id: str
    name: str
    status: str = "STOPPED"
    symbols: list[str] = field(default_factory=list)
    positions: dict[str, PositionInfo] = field(default_factory=dict)
    total_pnl: Decimal = Decimal("0")
    total_trades: int = 0
    win_rate: float = 0.0

    @property
    def status_icon(self) -> str:
        """Get status icon."""
        icons = {
            "RUNNING": "[green]●[/green]",
            "STOPPED": "[dim]○[/dim]",
            "PAUSED": "[yellow]◐[/yellow]",
            "STARTING": "[yellow]◔[/yellow]",
            "STOPPING": "[yellow]◕[/yellow]",
            "ERROR": "[red]✗[/red]",
            "DEGRADED": "[yellow]![/yellow]",
        }
        return icons.get(self.status, "[dim]?[/dim]")

    @property
    def pnl_display(self) -> str:
        """Format P&L for display."""
        pnl_color = "green" if self.total_pnl >= 0 else "red"
        pnl_sign = "+" if self.total_pnl >= 0 else ""
        return f"[{pnl_color}]{pnl_sign}${self.total_pnl:,.2f}[/{pnl_color}]"


# =============================================================================
# Strategy Tree Widget
# =============================================================================


class StrategyTree(Tree[StrategyInfo | str]):
    """
    Hierarchical tree view of strategies.

    Displays strategies with their symbols, positions, and metrics
    in an expandable tree structure.
    """

    DEFAULT_CSS = """
    StrategyTree {
        height: 1fr;
        background: $surface;
        border: round $primary-darken-2;
        padding: 1;
    }

    StrategyTree:focus {
        border: round $primary;
    }

    StrategyTree > .tree--guides {
        color: $primary-darken-1;
    }

    StrategyTree > .tree--cursor {
        background: $secondary-darken-1;
    }

    StrategyTree:focus > .tree--cursor {
        background: $secondary;
    }
    """

    class StrategySelected(Message):
        """Emitted when a strategy node is selected."""

        def __init__(self, strategy_id: str, strategy_name: str) -> None:
            self.strategy_id = strategy_id
            self.strategy_name = strategy_name
            super().__init__()

    class SymbolSelected(Message):
        """Emitted when a symbol node is selected."""

        def __init__(self, strategy_id: str, symbol: str) -> None:
            self.strategy_id = strategy_id
            self.symbol = symbol
            super().__init__()

    def __init__(
        self,
        label: str = "Strategies",
        strategies: list[StrategyInfo] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(label, id=id)
        self._strategies: dict[str, StrategyInfo] = {}
        self._strategy_nodes: dict[str, TreeNode[StrategyInfo | str]] = {}

        if strategies:
            for strategy in strategies:
                self._strategies[strategy.strategy_id] = strategy

    def on_mount(self) -> None:
        """Build the tree when mounted."""
        self._build_tree()

    def _build_tree(self) -> None:
        """Build or rebuild the entire tree."""
        self.root.expand()

        for strategy in self._strategies.values():
            self._add_strategy_node(strategy)

    def _add_strategy_node(self, strategy: StrategyInfo) -> TreeNode[StrategyInfo | str]:
        """Add a strategy and its children to the tree."""
        # Strategy label with status icon and P&L
        label = f"{strategy.status_icon} {strategy.name}  {strategy.pnl_display}"

        # Add strategy node
        strategy_node = self.root.add(label, data=strategy)
        self._strategy_nodes[strategy.strategy_id] = strategy_node

        # Add symbol nodes
        for symbol in strategy.symbols:
            position = strategy.positions.get(symbol)
            symbol_node = strategy_node.add(f"📊 {symbol}", data=symbol)

            if position:
                symbol_node.add_leaf(f"  └─ {position.display_text}", data="position")
            else:
                symbol_node.add_leaf("  └─ [dim](no position)[/dim]", data="position")

        # Add metrics node
        metrics_node = strategy_node.add("📈 Metrics", data="metrics")
        metrics_node.add_leaf(f"  ├─ P&L: {strategy.pnl_display}", data="pnl")
        metrics_node.add_leaf(f"  ├─ Trades: {strategy.total_trades}", data="trades")
        metrics_node.add_leaf(f"  └─ Win Rate: {strategy.win_rate:.1f}%", data="winrate")

        return strategy_node

    def on_tree_node_selected(self, event: Tree.NodeSelected[StrategyInfo | str]) -> None:
        """Handle node selection."""
        node = event.node
        data = node.data

        # Find the parent strategy for any selected node
        strategy = self._find_parent_strategy(node)

        if isinstance(data, StrategyInfo):
            # Direct strategy node selection
            self.post_message(self.StrategySelected(data.strategy_id, data.name))
        elif strategy:
            # Child node selected - still notify about the strategy
            self.post_message(self.StrategySelected(strategy.strategy_id, strategy.name))

            # Also emit symbol-specific event if it's a symbol
            if isinstance(data, str) and data not in ("metrics", "position", "pnl", "trades", "winrate"):
                self.post_message(self.SymbolSelected(strategy.strategy_id, data))

    def _find_parent_strategy(self, node: TreeNode[StrategyInfo | str]) -> StrategyInfo | None:
        """Walk up the tree to find the parent StrategyInfo."""
        current = node
        while current is not None:
            if isinstance(current.data, StrategyInfo):
                return current.data
            current = current.parent
        return None

    def add_strategy(self, strategy: StrategyInfo) -> None:
        """Add a new strategy to the tree."""
        self._strategies[strategy.strategy_id] = strategy
        self._add_strategy_node(strategy)

    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the tree."""
        if strategy_id in self._strategy_nodes:
            node = self._strategy_nodes[strategy_id]
            node.remove()
            del self._strategy_nodes[strategy_id]
            del self._strategies[strategy_id]

    def update_strategy(self, strategy: StrategyInfo) -> None:
        """Update an existing strategy."""
        if strategy.strategy_id in self._strategy_nodes:
            # Remove old node and re-add
            self.remove_strategy(strategy.strategy_id)

        self._strategies[strategy.strategy_id] = strategy
        self._add_strategy_node(strategy)

    def refresh_all(self, strategies: list[StrategyInfo]) -> None:
        """Refresh all strategies."""
        # Clear existing
        self.root.remove_children()
        self._strategies.clear()
        self._strategy_nodes.clear()

        # Add new
        for strategy in strategies:
            self._strategies[strategy.strategy_id] = strategy
            self._add_strategy_node(strategy)

    def get_selected_strategy(self) -> StrategyInfo | None:
        """Get the currently selected strategy."""
        node = self.cursor_node
        while node:
            if isinstance(node.data, StrategyInfo):
                return node.data
            node = node.parent
        return None

    def expand_strategy(self, strategy_id: str) -> None:
        """Expand a strategy node."""
        if strategy_id in self._strategy_nodes:
            self._strategy_nodes[strategy_id].expand()

    def collapse_strategy(self, strategy_id: str) -> None:
        """Collapse a strategy node."""
        if strategy_id in self._strategy_nodes:
            self._strategy_nodes[strategy_id].collapse()

    def expand_all(self) -> None:
        """Expand all nodes."""
        self.root.expand_all()

    def collapse_all(self) -> None:
        """Collapse all nodes except root."""
        for node in self._strategy_nodes.values():
            node.collapse()


# =============================================================================
# Strategy List Widget (Alternative to Tree)
# =============================================================================


class StrategyListView(Tree[StrategyInfo]):
    """
    Simplified list view of strategies (flat, no hierarchy).

    Use this for simpler UI needs where hierarchical display isn't required.
    """

    DEFAULT_CSS = """
    StrategyListView {
        height: 1fr;
        background: $surface;
        border: round $primary-darken-2;
        padding: 1;
    }

    StrategyListView:focus {
        border: round $primary;
    }
    """

    class StrategySelected(Message):
        """Emitted when a strategy is selected."""

        def __init__(self, strategy_id: str, strategy_name: str) -> None:
            self.strategy_id = strategy_id
            self.strategy_name = strategy_name
            super().__init__()

    def __init__(
        self,
        label: str = "Strategies",
        strategies: list[StrategyInfo] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(label, id=id)
        self._strategies: dict[str, StrategyInfo] = {}
        self.show_root = False
        self.guide_depth = 0

        if strategies:
            for strategy in strategies:
                self._strategies[strategy.strategy_id] = strategy

    def on_mount(self) -> None:
        """Build list when mounted."""
        self.root.expand()
        for strategy in self._strategies.values():
            self._add_strategy_item(strategy)

    def _add_strategy_item(self, strategy: StrategyInfo) -> None:
        """Add a strategy as a leaf node."""
        label = f"{strategy.status_icon} {strategy.name}  {strategy.pnl_display}"
        self.root.add_leaf(label, data=strategy)

    def on_tree_node_selected(self, event: Tree.NodeSelected[StrategyInfo]) -> None:
        """Handle selection."""
        if event.node.data:
            self.post_message(
                self.StrategySelected(
                    event.node.data.strategy_id,
                    event.node.data.name,
                )
            )

    def refresh_strategies(self, strategies: list[StrategyInfo]) -> None:
        """Refresh the strategy list."""
        self.root.remove_children()
        self._strategies.clear()

        for strategy in strategies:
            self._strategies[strategy.strategy_id] = strategy
            self._add_strategy_item(strategy)

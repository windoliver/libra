"""Spatial map for efficient widget visibility culling (Issue #89).

Per Textual blog (https://textual.textualize.io/blog/2024/12/12/algorithms-for-high-performance-terminal-apps/):
A "Spatial Map" is an internal optimization that quickly determines which widgets are visible.

This implementation uses grid-based tile culling:
- Divides the screen into tiles (default 10x2 cells per tile)
- Widgets are registered to tiles they overlap
- Visibility check is O(tiles_in_viewport) instead of O(n_widgets)
- Enables smooth scrolling with many widgets
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from textual.geometry import Region
    from textual.widget import Widget

T = TypeVar("T")


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned bounding box for spatial queries."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        """Right edge (exclusive)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge (exclusive)."""
        return self.y + self.height

    def overlaps(self, other: BoundingBox) -> bool:
        """Check if this box overlaps another."""
        return (
            self.x < other.x2
            and self.x2 > other.x
            and self.y < other.y2
            and self.y2 > other.y
        )

    @classmethod
    def from_region(cls, region: Region) -> BoundingBox:
        """Create bounding box from Textual Region."""
        return cls(region.x, region.y, region.width, region.height)


class SpatialMap(Generic[T]):
    """Grid-based spatial map for efficient widget culling.

    Divides the coordinate space into fixed-size tiles. Each widget
    is registered to all tiles it overlaps. Visibility queries return
    only widgets in tiles that overlap the viewport.

    Time complexity:
    - add_item: O(tiles_covered)
    - remove_item: O(tiles_covered)
    - visible_items: O(viewport_tiles + visible_items)
    - vs naive O(n) scan of all widgets

    Example:
        >>> spatial = SpatialMap[Widget](tile_width=10, tile_height=2)
        >>> spatial.add_item(widget, BoundingBox(0, 0, 20, 4))
        >>> visible = spatial.visible_items(viewport_box)
    """

    __slots__ = ("_tile_width", "_tile_height", "_tiles", "_item_tiles")

    def __init__(self, tile_width: int = 10, tile_height: int = 2) -> None:
        """Initialize spatial map with tile dimensions.

        Args:
            tile_width: Width of each tile in cells (default 10)
            tile_height: Height of each tile in cells (default 2)
        """
        self._tile_width = tile_width
        self._tile_height = tile_height
        # Map from tile coordinate to items in that tile
        self._tiles: dict[tuple[int, int], set[T]] = defaultdict(set)
        # Map from item to tiles it occupies (for efficient removal)
        self._item_tiles: dict[int, set[tuple[int, int]]] = defaultdict(set)

    @property
    def tile_width(self) -> int:
        """Width of each tile in cells."""
        return self._tile_width

    @property
    def tile_height(self) -> int:
        """Height of each tile in cells."""
        return self._tile_height

    @property
    def tile_count(self) -> int:
        """Number of non-empty tiles."""
        return len(self._tiles)

    @property
    def item_count(self) -> int:
        """Number of registered items."""
        return len(self._item_tiles)

    def _get_tile_range(self, box: BoundingBox) -> tuple[int, int, int, int]:
        """Get tile coordinate range for a bounding box.

        Returns:
            (min_tx, min_ty, max_tx, max_ty) tile coordinates (inclusive)
        """
        min_tx = box.x // self._tile_width
        min_ty = box.y // self._tile_height
        # Use x2-1 and y2-1 to handle exclusive bounds correctly
        max_tx = (box.x2 - 1) // self._tile_width if box.width > 0 else min_tx
        max_ty = (box.y2 - 1) // self._tile_height if box.height > 0 else min_ty
        return min_tx, min_ty, max_tx, max_ty

    def _iter_tiles(self, box: BoundingBox):
        """Iterate over all tile coordinates that overlap a bounding box."""
        min_tx, min_ty, max_tx, max_ty = self._get_tile_range(box)
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                yield tx, ty

    def add_item(self, item: T, box: BoundingBox) -> None:
        """Add an item to the spatial map.

        Args:
            item: Item to add (typically a Widget)
            box: Bounding box of the item
        """
        item_id = id(item)
        for tile in self._iter_tiles(box):
            self._tiles[tile].add(item)
            self._item_tiles[item_id].add(tile)

    def remove_item(self, item: T) -> bool:
        """Remove an item from the spatial map.

        Args:
            item: Item to remove

        Returns:
            True if item was found and removed, False otherwise
        """
        item_id = id(item)
        tiles = self._item_tiles.pop(item_id, None)
        if tiles is None:
            return False

        for tile in tiles:
            tile_items = self._tiles.get(tile)
            if tile_items:
                tile_items.discard(item)
                if not tile_items:
                    del self._tiles[tile]
        return True

    def update_item(self, item: T, box: BoundingBox) -> None:
        """Update an item's position in the spatial map.

        This is equivalent to remove + add but slightly more efficient.

        Args:
            item: Item to update
            box: New bounding box
        """
        self.remove_item(item)
        self.add_item(item, box)

    def visible_items(self, viewport: BoundingBox) -> set[T]:
        """Get all items that may be visible in the viewport.

        Note: This returns items whose tiles overlap the viewport.
        Some items may extend slightly outside the viewport.
        For pixel-perfect culling, filter results with box.overlaps().

        Args:
            viewport: Viewport bounding box

        Returns:
            Set of potentially visible items
        """
        visible: set[T] = set()
        for tile in self._iter_tiles(viewport):
            tile_items = self._tiles.get(tile)
            if tile_items:
                visible.update(tile_items)
        return visible

    def items_at_point(self, x: int, y: int) -> set[T]:
        """Get all items at a specific point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Set of items at that point
        """
        tx = x // self._tile_width
        ty = y // self._tile_height
        return set(self._tiles.get((tx, ty), set()))

    def clear(self) -> None:
        """Remove all items from the spatial map."""
        self._tiles.clear()
        self._item_tiles.clear()

    def __len__(self) -> int:
        """Return number of registered items."""
        return len(self._item_tiles)

    def __contains__(self, item: T) -> bool:
        """Check if an item is registered."""
        return id(item) in self._item_tiles


class WidgetSpatialMap(SpatialMap["Widget"]):
    """Specialized spatial map for Textual widgets.

    Provides convenience methods that work directly with Widget and Region types.
    """

    def add_widget(self, widget: Widget, region: Region) -> None:
        """Add a widget using its Textual Region.

        Args:
            widget: Widget to add
            region: Widget's region on screen
        """
        self.add_item(widget, BoundingBox.from_region(region))

    def remove_widget(self, widget: Widget) -> bool:
        """Remove a widget from the spatial map.

        Args:
            widget: Widget to remove

        Returns:
            True if widget was found and removed
        """
        return self.remove_item(widget)

    def update_widget(self, widget: Widget, region: Region) -> None:
        """Update a widget's position.

        Args:
            widget: Widget to update
            region: New region
        """
        self.update_item(widget, BoundingBox.from_region(region))

    def visible_widgets(self, viewport: Region) -> set[Widget]:
        """Get widgets visible in viewport.

        Args:
            viewport: Viewport region

        Returns:
            Set of potentially visible widgets
        """
        return self.visible_items(BoundingBox.from_region(viewport))

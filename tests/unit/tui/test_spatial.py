"""Tests for spatial map widget culling (Issue #89)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from libra.tui.spatial import BoundingBox, SpatialMap, WidgetSpatialMap


class TestBoundingBox:
    """Test BoundingBox class."""

    def test_create_box(self):
        """Test creating a bounding box."""
        box = BoundingBox(10, 20, 30, 40)
        assert box.x == 10
        assert box.y == 20
        assert box.width == 30
        assert box.height == 40

    def test_x2_y2_properties(self):
        """Test computed right/bottom edge properties."""
        box = BoundingBox(10, 20, 30, 40)
        assert box.x2 == 40  # 10 + 30
        assert box.y2 == 60  # 20 + 40

    def test_overlaps_true(self):
        """Test overlapping boxes."""
        box1 = BoundingBox(0, 0, 20, 20)
        box2 = BoundingBox(10, 10, 20, 20)
        assert box1.overlaps(box2)
        assert box2.overlaps(box1)

    def test_overlaps_false_horizontal(self):
        """Test non-overlapping boxes (horizontal gap)."""
        box1 = BoundingBox(0, 0, 10, 10)
        box2 = BoundingBox(20, 0, 10, 10)
        assert not box1.overlaps(box2)
        assert not box2.overlaps(box1)

    def test_overlaps_false_vertical(self):
        """Test non-overlapping boxes (vertical gap)."""
        box1 = BoundingBox(0, 0, 10, 10)
        box2 = BoundingBox(0, 20, 10, 10)
        assert not box1.overlaps(box2)
        assert not box2.overlaps(box1)

    def test_overlaps_edge_touch(self):
        """Test boxes that touch at edge (exclusive bounds)."""
        box1 = BoundingBox(0, 0, 10, 10)
        box2 = BoundingBox(10, 0, 10, 10)  # Touches at x=10
        assert not box1.overlaps(box2)

    def test_overlaps_contained(self):
        """Test one box contained in another."""
        outer = BoundingBox(0, 0, 100, 100)
        inner = BoundingBox(25, 25, 50, 50)
        assert outer.overlaps(inner)
        assert inner.overlaps(outer)

    def test_from_region(self):
        """Test creating box from Textual Region."""
        # Mock a Region object
        region = MagicMock()
        region.x = 5
        region.y = 10
        region.width = 15
        region.height = 20

        box = BoundingBox.from_region(region)
        assert box.x == 5
        assert box.y == 10
        assert box.width == 15
        assert box.height == 20


class TestSpatialMap:
    """Test SpatialMap class."""

    def test_create_default(self):
        """Test creating spatial map with defaults."""
        spatial = SpatialMap()
        assert spatial.tile_width == 10
        assert spatial.tile_height == 2
        assert spatial.tile_count == 0
        assert spatial.item_count == 0
        assert len(spatial) == 0

    def test_create_custom_tiles(self):
        """Test creating spatial map with custom tile size."""
        spatial = SpatialMap(tile_width=20, tile_height=5)
        assert spatial.tile_width == 20
        assert spatial.tile_height == 5

    def test_add_single_item(self):
        """Test adding a single item."""
        spatial = SpatialMap()
        item = object()
        spatial.add_item(item, BoundingBox(0, 0, 5, 1))

        assert len(spatial) == 1
        assert item in spatial

    def test_add_item_spanning_tiles(self):
        """Test adding item that spans multiple tiles."""
        spatial = SpatialMap(tile_width=10, tile_height=2)
        item = object()
        # Spans 2x2 tiles
        spatial.add_item(item, BoundingBox(5, 1, 15, 4))

        assert len(spatial) == 1
        assert spatial.tile_count >= 2  # At least 2 tiles

    def test_remove_item(self):
        """Test removing an item."""
        spatial = SpatialMap()
        item = object()
        spatial.add_item(item, BoundingBox(0, 0, 10, 2))

        assert item in spatial
        result = spatial.remove_item(item)
        assert result is True
        assert item not in spatial
        assert len(spatial) == 0

    def test_remove_nonexistent_item(self):
        """Test removing an item that doesn't exist."""
        spatial = SpatialMap()
        item = object()
        result = spatial.remove_item(item)
        assert result is False

    def test_update_item(self):
        """Test updating an item's position."""
        spatial = SpatialMap(tile_width=10, tile_height=2)
        item = object()

        # Add at position (0,0)
        spatial.add_item(item, BoundingBox(0, 0, 5, 1))
        visible1 = spatial.visible_items(BoundingBox(0, 0, 10, 2))
        assert item in visible1

        # Move to position (100, 100)
        spatial.update_item(item, BoundingBox(100, 100, 5, 1))

        # Should no longer be visible at origin
        visible2 = spatial.visible_items(BoundingBox(0, 0, 10, 2))
        assert item not in visible2

        # Should be visible at new location
        visible3 = spatial.visible_items(BoundingBox(100, 100, 10, 2))
        assert item in visible3

    def test_visible_items_simple(self):
        """Test visibility query with simple case."""
        spatial = SpatialMap(tile_width=10, tile_height=2)

        item1 = object()  # In view
        item2 = object()  # Out of view

        spatial.add_item(item1, BoundingBox(5, 1, 10, 2))
        spatial.add_item(item2, BoundingBox(100, 100, 10, 2))

        visible = spatial.visible_items(BoundingBox(0, 0, 20, 4))
        assert item1 in visible
        assert item2 not in visible

    def test_visible_items_partial_overlap(self):
        """Test visibility with partial viewport overlap."""
        spatial = SpatialMap(tile_width=10, tile_height=2)
        item = object()
        spatial.add_item(item, BoundingBox(15, 0, 20, 4))  # Spans tiles 1-3

        # Viewport overlaps part of item's tiles
        visible = spatial.visible_items(BoundingBox(0, 0, 20, 4))
        assert item in visible

    def test_items_at_point(self):
        """Test querying items at a specific point."""
        spatial = SpatialMap(tile_width=10, tile_height=2)

        item1 = object()
        item2 = object()

        spatial.add_item(item1, BoundingBox(0, 0, 15, 3))
        spatial.add_item(item2, BoundingBox(20, 0, 10, 2))

        # Point inside item1's first tile
        items_at_5_1 = spatial.items_at_point(5, 1)
        assert item1 in items_at_5_1
        assert item2 not in items_at_5_1

        # Point inside item2's tile
        items_at_25_1 = spatial.items_at_point(25, 1)
        assert item1 not in items_at_25_1
        assert item2 in items_at_25_1

    def test_clear(self):
        """Test clearing all items."""
        spatial = SpatialMap()

        for i in range(10):
            spatial.add_item(object(), BoundingBox(i * 10, 0, 5, 2))

        assert len(spatial) == 10
        spatial.clear()
        assert len(spatial) == 0
        assert spatial.tile_count == 0

    def test_contains(self):
        """Test __contains__ method."""
        spatial = SpatialMap()
        item1 = object()
        item2 = object()

        spatial.add_item(item1, BoundingBox(0, 0, 10, 2))

        assert item1 in spatial
        assert item2 not in spatial


class TestWidgetSpatialMap:
    """Test WidgetSpatialMap specialized class."""

    def test_add_widget(self):
        """Test adding a widget with Region."""
        spatial = WidgetSpatialMap()

        widget = MagicMock()
        region = MagicMock()
        region.x = 10
        region.y = 20
        region.width = 30
        region.height = 10

        spatial.add_widget(widget, region)
        assert widget in spatial

    def test_remove_widget(self):
        """Test removing a widget."""
        spatial = WidgetSpatialMap()

        widget = MagicMock()
        region = MagicMock()
        region.x = 0
        region.y = 0
        region.width = 10
        region.height = 2

        spatial.add_widget(widget, region)
        assert widget in spatial

        result = spatial.remove_widget(widget)
        assert result is True
        assert widget not in spatial

    def test_update_widget(self):
        """Test updating a widget's position."""
        spatial = WidgetSpatialMap()

        widget = MagicMock()

        region1 = MagicMock()
        region1.x = 0
        region1.y = 0
        region1.width = 10
        region1.height = 2

        region2 = MagicMock()
        region2.x = 100
        region2.y = 100
        region2.width = 10
        region2.height = 2

        spatial.add_widget(widget, region1)
        spatial.update_widget(widget, region2)

        # Check it moved
        viewport1 = MagicMock()
        viewport1.x = 0
        viewport1.y = 0
        viewport1.width = 20
        viewport1.height = 4

        visible1 = spatial.visible_widgets(viewport1)
        assert widget not in visible1

        viewport2 = MagicMock()
        viewport2.x = 100
        viewport2.y = 100
        viewport2.width = 20
        viewport2.height = 4

        visible2 = spatial.visible_widgets(viewport2)
        assert widget in visible2

    def test_visible_widgets(self):
        """Test visible_widgets with Region viewport."""
        spatial = WidgetSpatialMap()

        widget1 = MagicMock()
        widget2 = MagicMock()

        region1 = MagicMock()
        region1.x = 5
        region1.y = 0
        region1.width = 10
        region1.height = 2

        region2 = MagicMock()
        region2.x = 200
        region2.y = 200
        region2.width = 10
        region2.height = 2

        spatial.add_widget(widget1, region1)
        spatial.add_widget(widget2, region2)

        viewport = MagicMock()
        viewport.x = 0
        viewport.y = 0
        viewport.width = 50
        viewport.height = 10

        visible = spatial.visible_widgets(viewport)
        assert widget1 in visible
        assert widget2 not in visible


class TestSpatialMapPerformance:
    """Performance tests for spatial map."""

    def test_add_many_items(self):
        """Test adding 1000 items is fast."""
        spatial = SpatialMap(tile_width=10, tile_height=2)
        items = [object() for _ in range(1000)]

        start = time.perf_counter()
        for i, item in enumerate(items):
            x = (i % 100) * 10
            y = (i // 100) * 2
            spatial.add_item(item, BoundingBox(x, y, 8, 1))
        elapsed = time.perf_counter() - start

        assert len(spatial) == 1000
        # Should complete in under 100ms
        assert elapsed < 0.1, f"Adding 1000 items took {elapsed:.3f}s"

    def test_visibility_query_performance(self):
        """Test visibility query with many items is O(viewport_tiles)."""
        spatial = SpatialMap(tile_width=10, tile_height=2)

        # Add 1000 items spread across a large area
        for i in range(1000):
            x = (i % 100) * 10  # 0-990
            y = (i // 100) * 4  # 0-36
            spatial.add_item(object(), BoundingBox(x, y, 8, 3))

        # Query small viewport (should be fast)
        start = time.perf_counter()
        for _ in range(1000):
            visible = spatial.visible_items(BoundingBox(0, 0, 50, 10))
        elapsed = time.perf_counter() - start

        # 1000 queries should complete in under 50ms
        assert elapsed < 0.05, f"1000 visibility queries took {elapsed:.3f}s"

    def test_visibility_vs_naive_scan(self):
        """Compare spatial map to naive O(n) scan."""
        spatial = SpatialMap(tile_width=10, tile_height=2)
        items = []

        # Add 500 items
        for i in range(500):
            item = object()
            items.append((item, BoundingBox((i % 50) * 20, (i // 50) * 4, 15, 3)))
            spatial.add_item(item, items[-1][1])

        viewport = BoundingBox(0, 0, 100, 20)

        # Spatial map query
        start = time.perf_counter()
        for _ in range(1000):
            visible_spatial = spatial.visible_items(viewport)
        spatial_time = time.perf_counter() - start

        # Naive scan
        start = time.perf_counter()
        for _ in range(1000):
            visible_naive = {item for item, box in items if box.overlaps(viewport)}
        naive_time = time.perf_counter() - start

        # Spatial should be significantly faster
        # (may not always be true for small datasets, but should be comparable)
        print(f"Spatial: {spatial_time:.4f}s, Naive: {naive_time:.4f}s")

        # Both should return same visible items (spatial may have some extras at tile edges)
        assert visible_naive.issubset(visible_spatial)

    def test_benchmark_100_widgets(self):
        """Benchmark with 100+ widgets as per acceptance criteria."""
        spatial = WidgetSpatialMap(tile_width=10, tile_height=2)

        # Create 150 mock widgets
        widgets = []
        for i in range(150):
            widget = MagicMock()
            region = MagicMock()
            region.x = (i % 15) * 20
            region.y = (i // 15) * 6
            region.width = 18
            region.height = 5
            widgets.append((widget, region))
            spatial.add_widget(widget, region)

        assert len(spatial) == 150

        # Benchmark visibility queries
        viewport = MagicMock()
        viewport.x = 0
        viewport.y = 0
        viewport.width = 100
        viewport.height = 30

        start = time.perf_counter()
        for _ in range(10000):
            visible = spatial.visible_widgets(viewport)
        elapsed = time.perf_counter() - start

        avg_time_us = (elapsed / 10000) * 1_000_000
        print(f"Average visibility query: {avg_time_us:.2f} microseconds")

        # Should be under 100 microseconds on average
        assert avg_time_us < 100, f"Query too slow: {avg_time_us:.2f}us"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_size_box(self):
        """Test handling of zero-size bounding boxes."""
        spatial = SpatialMap()
        item = object()
        spatial.add_item(item, BoundingBox(10, 10, 0, 0))

        # Should still be registered in a tile
        assert item in spatial

    def test_negative_coordinates(self):
        """Test handling of negative coordinates."""
        spatial = SpatialMap(tile_width=10, tile_height=2)
        item = object()
        spatial.add_item(item, BoundingBox(-20, -10, 15, 8))

        # Should be findable with negative viewport
        visible = spatial.visible_items(BoundingBox(-30, -15, 40, 20))
        assert item in visible

    def test_large_coordinates(self):
        """Test handling of very large coordinates."""
        spatial = SpatialMap()
        item = object()
        spatial.add_item(item, BoundingBox(1_000_000, 500_000, 100, 50))

        visible = spatial.visible_items(BoundingBox(1_000_000, 500_000, 200, 100))
        assert item in visible

    def test_many_items_same_tile(self):
        """Test many items in the same tile."""
        spatial = SpatialMap(tile_width=100, tile_height=100)

        # Add 100 items all in the same tile
        items = []
        for i in range(100):
            item = object()
            items.append(item)
            spatial.add_item(item, BoundingBox(i, i, 1, 1))

        # All should be visible
        visible = spatial.visible_items(BoundingBox(0, 0, 100, 100))
        assert len(visible) == 100
        for item in items:
            assert item in visible

    def test_item_at_tile_boundary(self):
        """Test item exactly at tile boundary."""
        spatial = SpatialMap(tile_width=10, tile_height=2)
        item = object()
        # Item starts exactly at tile boundary
        spatial.add_item(item, BoundingBox(10, 2, 5, 1))

        # Should be in tile (1, 1)
        items = spatial.items_at_point(12, 2)
        assert item in items

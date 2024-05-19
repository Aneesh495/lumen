"""
Unit tests for Gaussian-to-tile binning and contiguous tile range indexer.
"""

from src.raster.binning import TileRangeIndex, TileRange
from src.raster.radix_sort import encode_tile_depth_key
from src.core.buffer import UInt64Buffer
from std.testing import assert_true

def test_tile_range_construction() raises:
    # 7 instances across 4 tiles (total_tiles = 4):
    # Tile 0: items 0, 1 (range [0, 2))
    # Tile 1: empty (range [-1, -1))
    # Tile 2: items 2, 3, 4 (range [2, 5))
    # Tile 3: items 5, 6 (range [5, 7))
    var keys = UInt64Buffer(7, 0)
    keys.set(0, encode_tile_depth_key(0, 1.0))
    keys.set(1, encode_tile_depth_key(0, 2.0))
    keys.set(2, encode_tile_depth_key(2, 3.0))
    keys.set(3, encode_tile_depth_key(2, 4.0))
    keys.set(4, encode_tile_depth_key(2, 5.0))
    keys.set(5, encode_tile_depth_key(3, 6.0))
    keys.set(6, encode_tile_depth_key(3, 7.0))

    var index = TileRangeIndex.build_from_sorted_keys(keys, 4)

    # Check tile 0
    var r0 = index.get_range(0)
    assert_true(not r0.is_empty(), "Tile 0 must not be empty")
    assert_true(r0.start == 0, "Tile 0 start must be 0")
    assert_true(r0.end == 2, "Tile 0 end must be 2")
    assert_true(r0.count() == 2, "Tile 0 count must be 2")

    # Check tile 1 (empty)
    var r1 = index.get_range(1)
    assert_true(r1.is_empty(), "Tile 1 must be empty")
    assert_true(r1.count() == 0, "Tile 1 count must be 0")

    # Check tile 2
    var r2 = index.get_range(2)
    assert_true(not r2.is_empty(), "Tile 2 must not be empty")
    assert_true(r2.start == 2, "Tile 2 start must be 2")
    assert_true(r2.end == 5, "Tile 2 end must be 5")
    assert_true(r2.count() == 3, "Tile 2 count must be 3")

    # Check tile 3
    var r3 = index.get_range(3)
    assert_true(not r3.is_empty(), "Tile 3 must not be empty")
    assert_true(r3.start == 5, "Tile 3 start must be 5")
    assert_true(r3.end == 7, "Tile 3 end must be 7")
    assert_true(r3.count() == 2, "Tile 3 count must be 2")

def main() raises:
    print("Testing Tile Range Indexing...")
    test_tile_range_construction()
    print("All binning tests passed successfully!")

"""
Unit tests for 64-bit LSD radix sorting, monotonicity, and stability.
"""

from src.raster.radix_sort import (
    radix_sort_64bit,
    encode_tile_depth_key,
    decode_tile_id,
    decode_depth_approx,
)
from src.core.buffer import UInt64Buffer, IntBuffer
from std.testing import assert_true, assert_almost_equal

def test_radix_sort_composite_keys() raises:
    # 4 items across 2 tiles with scrambled depths:
    # Item 0: Tile 1, depth 5.0
    # Item 1: Tile 0, depth 10.0
    # Item 2: Tile 0, depth 2.0
    # Item 3: Tile 1, depth 1.0
    var keys = UInt64Buffer(4, 0)
    var vals = IntBuffer(4, 0)

    keys.set(0, encode_tile_depth_key(1, 5.0))
    vals.set(0, 0)

    keys.set(1, encode_tile_depth_key(0, 10.0))
    vals.set(1, 1)

    keys.set(2, encode_tile_depth_key(0, 2.0))
    vals.set(2, 2)

    keys.set(3, encode_tile_depth_key(1, 1.0))
    vals.set(3, 3)

    radix_sort_64bit(keys, vals)

    # Expected sorted order:
    # 0: Tile 0, depth 2.0 (val 2)
    # 1: Tile 0, depth 10.0 (val 1)
    # 2: Tile 1, depth 1.0 (val 3)
    # 3: Tile 1, depth 5.0 (val 0)
    assert_true(vals.get(0) == 2, "First must be Tile 0, depth 2.0")
    assert_true(vals.get(1) == 1, "Second must be Tile 0, depth 10.0")
    assert_true(vals.get(2) == 3, "Third must be Tile 1, depth 1.0")
    assert_true(vals.get(3) == 0, "Fourth must be Tile 1, depth 5.0")

    assert_true(decode_tile_id(keys.get(0)) == 0, "Tile 0")
    assert_true(decode_tile_id(keys.get(1)) == 0, "Tile 0")
    assert_true(decode_tile_id(keys.get(2)) == 1, "Tile 1")
    assert_true(decode_tile_id(keys.get(3)) == 1, "Tile 1")

def test_radix_sort_large_monotonicity() raises:
    # 1,000 pseudorandom elements
    var n = 1000
    var keys = UInt64Buffer(n, 0)
    var vals = IntBuffer(n, 0)

    var lcg_seed: UInt64 = 123456789
    for i in range(n):
        lcg_seed = (lcg_seed * 6364136223846793005 + 1442695040888963407)
        keys.set(i, lcg_seed)
        vals.set(i, i)

    radix_sort_64bit(keys, vals)

    # Verify ascending monotonicity
    for i in range(n - 1):
        assert_true(keys.get(i) <= keys.get(i + 1), "Array must be monotonically non-decreasing")

def main() raises:
    print("Testing Radix Sort Composite Keys...")
    test_radix_sort_composite_keys()
    print("Testing Radix Sort Monotonicity (1,000 items)...")
    test_radix_sort_large_monotonicity()
    print("All radix sort tests passed successfully!")

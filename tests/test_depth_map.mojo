"""
Unit tests for depth buffer accumulation and surface normal extraction.
"""

from src.raster.depth_map import DepthMap
from src.math.vector import Vec3f
from std.testing import assert_true, assert_almost_equal

def test_flat_surface_normals() raises:
    # 5x5 depth map with constant depth 10.0
    var depth = DepthMap(5, 5)
    for y in range(5):
        for x in range(5):
            depth.set_depth(x, y, 10.0)

    # Normal at center (2, 2) must be directly toward viewer: (0, 0, 1)
    var n = depth.extract_normal(2, 2)
    assert_almost_equal(n.x, Float32(0.0), atol=1e-5)
    assert_almost_equal(n.y, Float32(0.0), atol=1e-5)
    assert_almost_equal(n.z, Float32(1.0), atol=1e-5)

def test_slanted_surface_normals() raises:
    # Depth increases linearly with X: z = x
    var depth = DepthMap(5, 5)
    for y in range(5):
        for x in range(5):
            depth.set_depth(x, y, Float32(x))

    # dz/dx = (z[x+1] - z[x-1]) / 2 = (3 - 1) / 2 = 1.0
    # normal = (-1, 0, 1) / sqrt(2) approx (-0.7071, 0, 0.7071)
    var n = depth.extract_normal(2, 2)
    assert_almost_equal(n.x, Float32(-0.707106), atol=1e-4)
    assert_almost_equal(n.y, Float32(0.0), atol=1e-4)
    assert_almost_equal(n.z, Float32(0.707106), atol=1e-4)

def main() raises:
    print("Testing Flat Surface Normals...")
    test_flat_surface_normals()
    print("Testing Slanted Surface Normals...")
    test_slanted_surface_normals()
    print("All depth map tests passed successfully!")

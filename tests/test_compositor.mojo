"""
Unit tests for front-to-back alpha blending, transmittance decay, and early ray termination.
"""

from src.raster.compositor import PixelAccumulator, compute_gaussian_pixel_alpha
from src.math.projection import Covariance2D
from src.math.vector import Vec3f
from std.testing import assert_true, assert_almost_equal

def test_single_splat_compositing() raises:
    var acc = PixelAccumulator()

    # Red splat with alpha = 0.6 at depth = 5.0
    var c_red = Vec3f(1.0, 0.0, 0.0)
    var terminated = acc.blend(c_red, Float32(0.6), Float32(5.0))

    assert_true(not terminated, "Pixel should not be saturated yet")
    assert_almost_equal(acc.color.x, Float32(0.6), atol=1e-5)
    assert_almost_equal(acc.color.y, Float32(0.0), atol=1e-5)
    assert_almost_equal(acc.color.z, Float32(0.0), atol=1e-5)
    assert_almost_equal(acc.transmittance, Float32(0.4), atol=1e-5)
    assert_almost_equal(acc.alpha, Float32(0.6), atol=1e-5)
    assert_almost_equal(acc.depth, Float32(3.0), atol=1e-5) # 5.0 * 0.6 = 3.0

def test_two_splat_order() raises:
    var acc = PixelAccumulator()

    # Front splat: Green (0, 1, 0), alpha = 0.5
    _ = acc.blend(Vec3f(0.0, 1.0, 0.0), Float32(0.5), Float32(2.0))

    # Back splat: Blue (0, 0, 1), alpha = 0.8
    _ = acc.blend(Vec3f(0.0, 0.0, 1.0), Float32(0.8), Float32(4.0))

    # Expected:
    # Green contribution: 0.5 * 1.0 = 0.5
    # Blue contribution: 0.8 * 0.5 (remaining T) = 0.4
    # Total T = 0.5 * (1 - 0.8) = 0.1
    assert_almost_equal(acc.color.y, Float32(0.5), atol=1e-5)
    assert_almost_equal(acc.color.z, Float32(0.4), atol=1e-5)
    assert_almost_equal(acc.transmittance, Float32(0.1), atol=1e-5)

def test_early_ray_termination() raises:
    var acc = PixelAccumulator()

    # Blend 10 opaque splats
    for _ in range(10):
        _ = acc.blend(Vec3f(1.0, 1.0, 1.0), Float32(0.9), Float32(1.0))

    # Transmittance drops as 0.1^10 = 10^-10 < 1e-4
    assert_true(acc.is_saturated(), "Ray must be early terminated due to opacity saturation")

def main() raises:
    print("Testing Single Splat Compositing...")
    test_single_splat_compositing()
    print("Testing Two Splat Front-to-Back Ordering...")
    test_two_splat_order()
    print("Testing Early Ray Termination...")
    test_early_ray_termination()
    print("All compositor tests passed successfully!")

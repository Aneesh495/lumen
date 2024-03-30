"""
Unit tests for 3D Gaussian covariance construction and mathematical invariants.
"""

from src.math.covariance import Covariance3D
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from src.core.types import HALF_PI_F32
from std.testing import assert_true, assert_almost_equal

def test_covariance_axis_aligned() raises:
    # Scale = (2, 3, 4), rotation = identity
    # Sigma = diag(4, 9, 16)
    var scale = Vec3f(2.0, 3.0, 4.0)
    var rot = Quatf.identity()
    var cov = Covariance3D.from_scale_and_rotation(scale, rot)

    assert_almost_equal(cov.c00, Float32(4.0), atol=1e-5)
    assert_almost_equal(cov.c11, Float32(9.0), atol=1e-5)
    assert_almost_equal(cov.c22, Float32(16.0), atol=1e-5)
    assert_almost_equal(cov.c01, Float32(0.0), atol=1e-5)
    assert_almost_equal(cov.c02, Float32(0.0), atol=1e-5)
    assert_almost_equal(cov.c12, Float32(0.0), atol=1e-5)

    # Determinant should equal 4 * 9 * 16 = 576
    assert_almost_equal(cov.determinant(), Float32(576.0), atol=1e-3)
    assert_true(cov.is_positive_definite(), "Axis-aligned covariance must be positive definite")

def test_covariance_rotation_invariance() raises:
    # Rotation preserves determinant (volume) of covariance ellipsoid
    var scale = Vec3f(1.5, 2.5, 3.5)
    var rot = Quatf.from_axis_angle(Vec3f(0.0, 1.0, 0.0), HALF_PI_F32)
    var cov = Covariance3D.from_scale_and_rotation(scale, rot)

    var expected_det: Float32 = Float32(1.5 * 1.5 * 2.5 * 2.5 * 3.5 * 3.5)
    assert_almost_equal(cov.determinant(), expected_det, atol=1e-2)
    assert_true(cov.is_positive_definite(), "Rotated covariance must be positive definite")

def main() raises:
    print("Testing Axis-Aligned 3D Covariance...")
    test_covariance_axis_aligned()
    print("Testing Rotated 3D Covariance Invariance...")
    test_covariance_rotation_invariance()
    print("All covariance tests passed successfully!")

"""
Unit tests for projective Jacobian derivation and EWA 2D screen covariance splatting.
"""

from src.math.projection import ProjectiveJacobian, Covariance2D, compute_ewa_covariance
from src.math.covariance import Covariance3D
from src.math.vector import Vec3f
from src.math.matrix import Mat3x3f
from src.core.types import LOWPASS_FILTER_VARIANCE
from std.testing import assert_true, assert_almost_equal

def test_jacobian_analytical() raises:
    # On-axis test: camera at (0, 0, 10), fx = fy = 1000
    var t_center = Vec3f(0.0, 0.0, 10.0)
    var J_center = ProjectiveJacobian.compute(t_center, Float32(1000.0), Float32(1000.0))

    assert_almost_equal(J_center.j00, Float32(100.0), atol=1e-4)
    assert_almost_equal(J_center.j11, Float32(100.0), atol=1e-4)
    assert_almost_equal(J_center.j02, Float32(0.0), atol=1e-4)
    assert_almost_equal(J_center.j12, Float32(0.0), atol=1e-4)

    # Off-axis test: camera at (2, 3, 10)
    var t_off = Vec3f(2.0, 3.0, 10.0)
    var J_off = ProjectiveJacobian.compute(t_off, Float32(1000.0), Float32(1000.0))
    # j02 = -fx * tx / tz^2 = -1000 * 2 / 100 = -20.0
    # j12 = -fy * ty / tz^2 = -1000 * 3 / 100 = -30.0
    assert_almost_equal(J_off.j02, Float32(-20.0), atol=1e-4)
    assert_almost_equal(J_off.j12, Float32(-30.0), atol=1e-4)

def test_ewa_screen_covariance() raises:
    var cov3d = Covariance3D(1.0, 0.0, 0.0, 1.0, 0.0, 1.0) # Identity 3D sphere
    var cam_pos = Vec3f(0.0, 0.0, 10.0)
    var view_rot = Mat3x3f() # Identity orientation

    var cov2d = compute_ewa_covariance(
        cov3d,
        cam_pos,
        view_rot,
        Float32(100.0),
        Float32(100.0),
        LOWPASS_FILTER_VARIANCE
    )

    # On screen: scale = fx / tz = 100 / 10 = 10
    # Expected variance: 10^2 * 1.0 + 0.3 = 100.3
    assert_almost_equal(cov2d.c00, Float32(100.3), atol=1e-3)
    assert_almost_equal(cov2d.c11, Float32(100.3), atol=1e-3)
    assert_almost_equal(cov2d.c01, Float32(0.0), atol=1e-4)

    # Invertibility check
    var inv2d = cov2d.inverse()
    var mat = cov2d.to_mat2()
    var inv_mat = inv2d.to_mat2()

    # Product should be 2x2 identity
    var prod00 = mat.m00 * inv_mat.m00 + mat.m01 * inv_mat.m10
    var prod11 = mat.m10 * inv_mat.m01 + mat.m11 * inv_mat.m11
    assert_almost_equal(prod00, Float32(1.0), atol=1e-4)
    assert_almost_equal(prod11, Float32(1.0), atol=1e-4)

def main() raises:
    print("Testing Projective Jacobian...")
    test_jacobian_analytical()
    print("Testing EWA Screen Covariance...")
    test_ewa_screen_covariance()
    print("All projection tests passed successfully!")

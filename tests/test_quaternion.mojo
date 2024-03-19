"""
Unit tests for quaternion algebra, SO(3) rotations, and SLERP interpolation.
"""

from src.math.quaternion import Quatf
from src.math.vector import Vec3f
from src.core.types import PI_F32, HALF_PI_F32
from std.testing import assert_true, assert_almost_equal

def test_quaternion_hamilton_product() raises:
    # i = (0, 1, 0, 0), j = (0, 0, 1, 0), k = (0, 0, 0, 1)
    # i * j = k
    var qi = Quatf(0.0, 1.0, 0.0, 0.0)
    var qj = Quatf(0.0, 0.0, 1.0, 0.0)
    var qk = qi.mul(qj)

    assert_almost_equal(qk.w, 0.0, atol=1e-5)
    assert_almost_equal(qk.x, 0.0, atol=1e-5)
    assert_almost_equal(qk.y, 0.0, atol=1e-5)
    assert_almost_equal(qk.z, 1.0, atol=1e-5)

def test_quaternion_vector_rotation() raises:
    # Rotate (1, 0, 0) by 90 degrees around Z axis -> (0, 1, 0)
    var q_rot_z = Quatf.from_axis_angle(Vec3f(0.0, 0.0, 1.0), HALF_PI_F32)
    var v_in = Vec3f(1.0, 0.0, 0.0)
    var v_out = q_rot_z.rotate_vec(v_in)

    assert_almost_equal(v_out.x, 0.0, atol=1e-5)
    assert_almost_equal(v_out.y, 1.0, atol=1e-5)
    assert_almost_equal(v_out.z, 0.0, atol=1e-5)

    # Verify equivalent rotation matrix
    var rot_mat = q_rot_z.to_rotation_matrix()
    var v_mat_out = rot_mat.mul_vec(v_in)
    assert_almost_equal(v_mat_out.x, 0.0, atol=1e-5)
    assert_almost_equal(v_mat_out.y, 1.0, atol=1e-5)
    assert_almost_equal(v_mat_out.z, 0.0, atol=1e-5)

    # Det(R) must equal 1 for proper rotation
    assert_almost_equal(rot_mat.determinant(), 1.0, atol=1e-5)

def test_quaternion_slerp() raises:
    var q_start = Quatf.identity()
    var q_end = Quatf.from_axis_angle(Vec3f(0.0, 1.0, 0.0), HALF_PI_F32) # 90 deg around Y

    var q_mid = q_start.slerp(q_end, 0.5) # 45 deg around Y
    var v_in = Vec3f(1.0, 0.0, 0.0)
    var v_mid = q_mid.rotate_vec(v_in)

    # cos(45 deg) = sqrt(2)/2 approx 0.7071
    # sin(-45 deg) = -0.7071 in right-handed convention
    assert_almost_equal(v_mid.x, 0.707106, atol=1e-4)
    assert_almost_equal(v_mid.y, 0.0, atol=1e-4)
    assert_almost_equal(v_mid.z, -0.707106, atol=1e-4)

def main() raises:
    print("Testing Quaternion Hamilton Product...")
    test_quaternion_hamilton_product()
    print("Testing Quaternion Vector Rotation...")
    test_quaternion_vector_rotation()
    print("Testing Quaternion SLERP...")
    test_quaternion_slerp()
    print("All quaternion tests passed successfully!")

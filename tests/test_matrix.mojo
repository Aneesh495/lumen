"""
Unit tests for 2x2, 3x3, and 4x4 matrix algebra, determinants, and inverses.
"""

from src.math.matrix import Mat2x2f, Mat3x3f, Mat4x4f
from src.math.vector import Vec2f, Vec3f, Vec4f
from std.testing import assert_true, assert_almost_equal

def test_mat2_determinant_and_inverse() raises:
    # A = [4, 7; 2, 6] -> det(A) = 24 - 14 = 10
    var m = Mat2x2f(4.0, 7.0, 2.0, 6.0)
    assert_almost_equal(m.determinant(), 10.0, atol=1e-5)
    assert_almost_equal(m.trace(), 10.0, atol=1e-5)

    var inv = m.inverse()
    # A * inv should equal identity
    var v = Vec2f(3.0, 5.0)
    var v_trans = m.mul_vec(v)
    var v_back = inv.mul_vec(v_trans)
    assert_almost_equal(v_back.x, 3.0, atol=1e-4)
    assert_almost_equal(v_back.y, 5.0, atol=1e-4)

    # Singular matrix
    var singular = Mat2x2f(1.0, 2.0, 2.0, 4.0)
    assert_almost_equal(singular.determinant(), 0.0, atol=1e-5)
    var inv_sing = singular.inverse()
    assert_almost_equal(inv_sing.m00, 0.0, atol=1e-5)

def test_mat3_inversion() raises:
    var m = Mat3x3f(
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0
    )
    # det = 1*(0 - 24) - 2*(0 - 20) + 3*(0 - 5) = -24 + 40 - 15 = 1
    assert_almost_equal(m.determinant(), 1.0, atol=1e-4)

    var inv = m.inverse()
    var v = Vec3f(2.0, -1.0, 4.0)
    var v_t = m.mul_vec(v)
    var v_rec = inv.mul_vec(v_t)

    assert_almost_equal(v_rec.x, 2.0, atol=1e-4)
    assert_almost_equal(v_rec.y, -1.0, atol=1e-4)
    assert_almost_equal(v_rec.z, 4.0, atol=1e-4)

def test_mat4_view_and_proj() raises:
    var eye = Vec3f(0.0, 0.0, 5.0)
    var target = Vec3f(0.0, 0.0, 0.0)
    var up = Vec3f(0.0, 1.0, 0.0)
    var view = Mat4x4f.look_at(eye, target, up)

    # Target point in view space should be at (0, 0, -5)
    var p_world = Vec4f(0.0, 0.0, 0.0, 1.0)
    var p_view = view.mul_vec(p_world)
    assert_almost_equal(p_view.x, 0.0, atol=1e-4)
    assert_almost_equal(p_view.y, 0.0, atol=1e-4)
    assert_almost_equal(p_view.z, -5.0, atol=1e-4)

def main() raises:
    print("Testing Mat2x2...")
    test_mat2_determinant_and_inverse()
    print("Testing Mat3x3...")
    test_mat3_inversion()
    print("Testing Mat4x4...")
    test_mat4_view_and_proj()
    print("All matrix tests passed successfully!")

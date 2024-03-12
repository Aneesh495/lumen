"""
Unit tests for 2D, 3D, and 4D vector mathematics and SIMD conversion.
"""

from src.math.vector import Vec2f, Vec3f, Vec4f
from std.testing import assert_true, assert_almost_equal

def test_vec2_basic() raises:
    var v1 = Vec2f(3.0, 4.0)
    assert_almost_equal(v1.length_squared(), 25.0, atol=1e-5)
    assert_almost_equal(v1.length(), 5.0, atol=1e-5)

    var v2 = Vec2f(1.0, 2.0)
    var v3 = v1.add(v2)
    assert_almost_equal(v3.x, 4.0, atol=1e-5)
    assert_almost_equal(v3.y, 6.0, atol=1e-5)

    var norm = v1.normalized()
    assert_almost_equal(norm.length(), 1.0, atol=1e-5)
    assert_almost_equal(norm.x, 0.6, atol=1e-5)
    assert_almost_equal(norm.y, 0.8, atol=1e-5)

    # Edge case: zero vector normalization should not produce NaN or explode
    var zero_vec = Vec2f(0.0, 0.0)
    var zero_norm = zero_vec.normalized()
    assert_almost_equal(zero_norm.x, 0.0, atol=1e-5)
    assert_almost_equal(zero_norm.y, 0.0, atol=1e-5)

def test_vec3_cross_and_dot() raises:
    var x_axis = Vec3f(1.0, 0.0, 0.0)
    var y_axis = Vec3f(0.0, 1.0, 0.0)
    var z_axis = x_axis.cross(y_axis)

    assert_almost_equal(z_axis.x, 0.0, atol=1e-5)
    assert_almost_equal(z_axis.y, 0.0, atol=1e-5)
    assert_almost_equal(z_axis.z, 1.0, atol=1e-5)

    assert_almost_equal(x_axis.dot(y_axis), 0.0, atol=1e-5)
    assert_almost_equal(x_axis.dot(x_axis), 1.0, atol=1e-5)

    # Cross product of parallel vectors is zero
    var parallel = x_axis.mul(5.0)
    var cross_parallel = x_axis.cross(parallel)
    assert_almost_equal(cross_parallel.length(), 0.0, atol=1e-5)

    # Lerp test
    var mid = x_axis.lerp(y_axis, 0.5)
    assert_almost_equal(mid.x, 0.5, atol=1e-5)
    assert_almost_equal(mid.y, 0.5, atol=1e-5)
    assert_almost_equal(mid.z, 0.0, atol=1e-5)

def test_vec4_simd() raises:
    var v4 = Vec4f(1.0, 2.0, 3.0, 4.0)
    var v_xyz = v4.to_xyz()
    assert_almost_equal(v_xyz.x, 1.0, atol=1e-5)
    assert_almost_equal(v_xyz.y, 2.0, atol=1e-5)
    assert_almost_equal(v_xyz.z, 3.0, atol=1e-5)

    var simd_val = v4.to_simd()
    assert_almost_equal(simd_val[0], 1.0, atol=1e-5)
    assert_almost_equal(simd_val[3], 4.0, atol=1e-5)

def main() raises:
    print("Testing Vector2...")
    test_vec2_basic()
    print("Testing Vector3...")
    test_vec3_cross_and_dot()
    print("Testing Vector4...")
    test_vec4_simd()
    print("All vector tests passed successfully!")

"""
Unit tests for 3D Morton spatial code generation and bitwise interleaving.
"""

from src.spatial.morton import compute_morton_3d, expand_bits_10_to_30
from src.math.vector import Vec3f
from std.testing import assert_true

def test_morton_primitives() raises:
    # (0, 0, 0) must map to Morton code 0
    var m_zero = compute_morton_3d(Vec3f(0.0, 0.0, 0.0))
    assert_true(m_zero == 0, "Origin must have Morton code 0")

    # (1, 1, 1) must map to all 30 bits set: 0x3FFFFFFF
    var m_one = compute_morton_3d(Vec3f(1.0, 1.0, 1.0))
    assert_true(m_one == 0x3FFFFFFF, "Unit corner must have all 30 bits set")

    # Single bit expansion: 1 (binary 01) -> expanded should be 1
    # 2 (binary 10) -> expanded should be binary 1000 = 8
    var exp_1 = expand_bits_10_to_30(1)
    var exp_2 = expand_bits_10_to_30(2)
    assert_true(exp_1 == 1, "Expanded 1 must be 1")
    assert_true(exp_2 == 8, "Expanded 2 must be 8 (bit 1 shifted by 2 zeros to bit 3)")

def test_morton_spatial_locality() raises:
    var p_a = Vec3f(0.5, 0.5, 0.5)
    var p_b = Vec3f(0.501, 0.501, 0.501)
    var p_c = Vec3f(0.9, 0.9, 0.9)

    var ma = compute_morton_3d(p_a)
    var mb = compute_morton_3d(p_b)
    var mc = compute_morton_3d(p_c)

    var diff_close = Int(mb) - Int(ma)
    if diff_close < 0: diff_close = -diff_close

    var diff_far = Int(mc) - Int(ma)
    if diff_far < 0: diff_far = -diff_far

    assert_true(diff_close < diff_far, "Close spatial points must have closer Morton codes")

def main() raises:
    print("Testing Morton Code Primitives...")
    test_morton_primitives()
    print("Testing Morton Spatial Locality...")
    test_morton_spatial_locality()
    print("All Morton tests passed successfully!")

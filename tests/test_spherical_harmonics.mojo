"""
Unit tests for spherical harmonics basis polynomials and view-dependent color rendering.
"""

from src.math.spherical_harmonics import SHBasis, evaluate_sh_color
from src.math.vector import Vec3f
from std.collections import List
from std.testing import assert_true, assert_almost_equal

def test_sh_basis_degrees() raises:
    # Look along +Z axis: (0, 0, 1)
    var dir_z = Vec3f(0.0, 0.0, 1.0)
    var basis_z = SHBasis.evaluate(dir_z)

    # Degree 0 is constant everywhere
    assert_almost_equal(basis_z.y0, Float32(0.2820948), atol=1e-5)

    # Along Z: x=0, y=0, z=1 -> y1=0, y3=0, y2 = SH_C1 * 1.0
    assert_almost_equal(basis_z.y1, Float32(0.0), atol=1e-5)
    assert_almost_equal(basis_z.y3, Float32(0.0), atol=1e-5)
    assert_almost_equal(basis_z.y2, Float32(0.4886025), atol=1e-5)

    # Opposite direction: along -Z -> y2 must flip sign (dipole anti-symmetry)
    var dir_neg_z = Vec3f(0.0, 0.0, -1.0)
    var basis_neg_z = SHBasis.evaluate(dir_neg_z)
    assert_almost_equal(basis_neg_z.y2, Float32(-0.4886025), atol=1e-5)

def test_sh_color_evaluation() raises:
    # 48 coefficients initialized to zero except degree 0
    var coeffs = List[Float32](capacity=48)
    for _ in range(48):
        coeffs.append(0.0)

    # Set degree 0 RGB: we want base color (0.8, 0.4, 0.2)
    # Since Color = 0.5 + c0 * Y0, c0 = (Color - 0.5) / Y0
    # Y0 approx 0.2820948
    var inv_y0: Float32 = 1.0 / 0.28209479
    coeffs[0] = (0.8 - 0.5) * inv_y0 # R
    coeffs[1] = (0.4 - 0.5) * inv_y0 # G
    coeffs[2] = (0.2 - 0.5) * inv_y0 # B

    var view_dir = Vec3f(0.0, 1.0, 0.0)
    var color = evaluate_sh_color(coeffs, 0, view_dir, 0)

    assert_almost_equal(color.x, Float32(0.8), atol=1e-4)
    assert_almost_equal(color.y, Float32(0.4), atol=1e-4)
    assert_almost_equal(color.z, Float32(0.2), atol=1e-4)

def main() raises:
    print("Testing SH Basis Polynomials...")
    test_sh_basis_degrees()
    print("Testing SH Color Evaluation...")
    test_sh_color_evaluation()
    print("All spherical harmonics tests passed successfully!")

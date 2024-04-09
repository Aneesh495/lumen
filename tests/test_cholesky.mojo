"""
Unit tests for 2x2 eigenvalue decomposition, 3-sigma bounding radius, and Cholesky factoring.
"""

from src.math.cholesky import solve_eigenvalues_2x2, Cholesky3D
from src.math.projection import Covariance2D
from src.math.covariance import Covariance3D
from std.testing import assert_true, assert_almost_equal

def test_eigenvalues_diagonal() raises:
    # cov = [16, 0; 0, 9] -> lambda_max = 16, lambda_min = 9, radius = 3 * 4 = 12
    var cov = Covariance2D(16.0, 0.0, 9.0)
    var eigen = solve_eigenvalues_2x2(cov)

    assert_almost_equal(eigen.lambda_max, Float32(16.0), atol=1e-5)
    assert_almost_equal(eigen.lambda_min, Float32(9.0), atol=1e-5)
    assert_true(eigen.radius == 12, "Radius must equal 12 for lambda_max = 16")

def test_eigenvalues_coupled() raises:
    # cov = [2, 1; 1, 2] -> lambda_max = 3, lambda_min = 1
    # 3 * sqrt(3) approx 3 * 1.73205 = 5.196 -> ceil is 6
    var cov = Covariance2D(2.0, 1.0, 2.0)
    var eigen = solve_eigenvalues_2x2(cov)

    assert_almost_equal(eigen.lambda_max, Float32(3.0), atol=1e-5)
    assert_almost_equal(eigen.lambda_min, Float32(1.0), atol=1e-5)
    assert_true(eigen.radius == 6, "Radius must equal 6 for lambda_max = 3")

def test_cholesky_reconstruction() raises:
    # Sigma = [4, 2, 0; 2, 5, 1; 0, 1, 3]
    var cov = Covariance3D(4.0, 2.0, 0.0, 5.0, 1.0, 3.0)
    var chol = Cholesky3D.factor(cov)

    # Reconstruct L * L^T:
    # Row 0: c00 = l00^2 = 4 -> l00 = 2
    assert_almost_equal(chol.l00, Float32(2.0), atol=1e-4)

    # c01 = l10 * l00 = 2 -> l10 = 1
    assert_almost_equal(chol.l10, Float32(1.0), atol=1e-4)

    # c11 = l10^2 + l11^2 = 1 + l11^2 = 5 -> l11 = 2
    assert_almost_equal(chol.l11, Float32(2.0), atol=1e-4)

    var recon_c00 = chol.l00 * chol.l00
    var recon_c01 = chol.l10 * chol.l00
    var recon_c11 = chol.l10 * chol.l10 + chol.l11 * chol.l11

    assert_almost_equal(recon_c00, Float32(4.0), atol=1e-4)
    assert_almost_equal(recon_c01, Float32(2.0), atol=1e-4)
    assert_almost_equal(recon_c11, Float32(5.0), atol=1e-4)

def main() raises:
    print("Testing Diagonal Eigenvalues...")
    test_eigenvalues_diagonal()
    print("Testing Coupled 2x2 Eigenvalues...")
    test_eigenvalues_coupled()
    print("Testing 3D Cholesky Factorization...")
    test_cholesky_reconstruction()
    print("All eigenvalue and Cholesky tests passed successfully!")

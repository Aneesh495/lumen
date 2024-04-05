"""
Closed-form eigenvalue decomposition of 2x2 symmetric matrices and 3-sigma bounding radius.
"""

from .projection import Covariance2D
from .covariance import Covariance3D
from std.math import sqrt, ceil, max

struct Eigen2D(ImplicitlyCopyable):
    """
    Eigenvalues lambda_1 >= lambda_2 of a symmetric 2x2 matrix, and 3-sigma bounding radius.
    """
    var lambda_max: Float32
    var lambda_min: Float32
    var radius: Int

    def __init__(out self):
        self.lambda_max = 1.0
        self.lambda_min = 1.0
        self.radius = 3

    def __init__(out self, lambda_max: Float32, lambda_min: Float32, radius: Int):
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.radius = radius


def solve_eigenvalues_2x2(cov: Covariance2D) -> Eigen2D:
    """
    Analytically compute eigenvalues of [a, b; b, c]:
    lambda = (a + c) / 2 +/- sqrt(((a - c) / 2)^2 + b^2).
    """
    var a = cov.c00
    var b = cov.c01
    var c = cov.c11

    var mid = (a + c) * 0.5
    var diff = (a - c) * 0.5
    var discriminant = diff * diff + b * b
    var root = sqrt(discriminant)

    var l_max = mid + root
    var l_min = mid - root
    if l_min < 0.0:
        l_min = 0.0

    # 3-sigma bounding radius: 3 * sqrt(lambda_max)
    var r_float = 3.0 * sqrt(l_max)
    var r = Int(ceil(r_float))

    return Eigen2D(l_max, l_min, r)


struct Cholesky3D(ImplicitlyCopyable):
    """
    Lower triangular factor L of 3x3 positive-definite matrix Sigma = L * L^T.
    """
    var l00: Float32
    var l10: Float32
    var l11: Float32
    var l20: Float32
    var l21: Float32
    var l22: Float32

    def __init__(out self):
        self.l00 = 1.0
        self.l10 = 0.0
        self.l11 = 1.0
        self.l20 = 0.0
        self.l21 = 0.0
        self.l22 = 1.0

    @staticmethod
    def factor(cov: Covariance3D) -> Cholesky3D:
        var res = Cholesky3D()

        # Row 0
        var l00 = sqrt(max(cov.c00, Float32(0.000001)))
        var inv_l00 = 1.0 / l00
        res.l00 = l00

        # Row 1
        var l10 = cov.c01 * inv_l00
        var l11_sq = cov.c11 - l10 * l10
        var l11 = sqrt(max(l11_sq, Float32(0.000001)))
        var inv_l11 = 1.0 / l11
        res.l10 = l10
        res.l11 = l11

        # Row 2
        var l20 = cov.c02 * inv_l00
        var l21 = (cov.c12 - l20 * l10) * inv_l11
        var l22_sq = cov.c22 - l20 * l20 - l21 * l21
        var l22 = sqrt(max(l22_sq, Float32(0.000001)))
        res.l20 = l20
        res.l21 = l21
        res.l22 = l22

        return res

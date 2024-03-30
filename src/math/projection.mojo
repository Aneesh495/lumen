"""
Projective camera Jacobian computation and Elliptical Weighted Average (EWA) 2D screen covariance splatting.
"""

from .vector import Vec2f, Vec3f
from .matrix import Mat3x3f, Mat2x2f
from .covariance import Covariance3D
from src.core.types import LOWPASS_FILTER_VARIANCE

struct Covariance2D(ImplicitlyCopyable):
    """
    Symmetric positive-definite 2D screen covariance matrix:
    [c00, c01]
    [c01, c11]
    """
    var c00: Float32 # Sigma_2D[0,0]
    var c01: Float32 # Sigma_2D[0,1] == Sigma_2D[1,0]
    var c11: Float32 # Sigma_2D[1,1]

    def __init__(out self):
        self.c00 = 1.0
        self.c01 = 0.0
        self.c11 = 1.0

    def __init__(out self, c00: Float32, c01: Float32, c11: Float32):
        self.c00 = c00
        self.c01 = c01
        self.c11 = c11

    def determinant(self) -> Float32:
        return self.c00 * self.c11 - self.c01 * self.c01

    def inverse(self) -> Covariance2D:
        var det = self.determinant()
        if det > -0.0000001 and det < 0.0000001:
            return Covariance2D(0.0, 0.0, 0.0)
        var inv_det = 1.0 / det
        return Covariance2D(
            self.c11 * inv_det,
            -self.c01 * inv_det,
            self.c00 * inv_det
        )

    def to_mat2(self) -> Mat2x2f:
        return Mat2x2f(self.c00, self.c01, self.c01, self.c11)


struct ProjectiveJacobian(ImplicitlyCopyable):
    """
    Affine 2x3 Jacobian J of perspective projection at camera coordinate t = (tx, ty, tz).
    """
    var j00: Float32 # fx / tz
    var j01: Float32 # 0
    var j02: Float32 # -fx * tx / tz^2

    var j10: Float32 # 0
    var j11: Float32 # fy / tz
    var j12: Float32 # -fy * ty / tz^2

    def __init__(out self):
        self.j00 = 0.0
        self.j01 = 0.0
        self.j02 = 0.0
        self.j10 = 0.0
        self.j11 = 0.0
        self.j12 = 0.0

    def __init__(
        out self,
        j00: Float32, j01: Float32, j02: Float32,
        j10: Float32, j11: Float32, j12: Float32
    ):
        self.j00 = j00
        self.j01 = j01
        self.j02 = j02
        self.j10 = j10
        self.j11 = j11
        self.j12 = j12

    @staticmethod
    def compute(t: Vec3f, focal_x: Float32, focal_y: Float32) -> ProjectiveJacobian:
        var tz = t.z
        if tz < 0.0001:
            tz = 0.0001 # Clamp to prevent division by zero in camera near field
        var inv_tz = 1.0 / tz
        var inv_tz2 = inv_tz * inv_tz

        return ProjectiveJacobian(
            focal_x * inv_tz,
            0.0,
            -focal_x * t.x * inv_tz2,

            0.0,
            focal_y * inv_tz,
            -focal_y * t.y * inv_tz2
        )


def compute_ewa_covariance(
    cov3d: Covariance3D,
    cam_pos: Vec3f,
    view_rot: Mat3x3f,
    focal_x: Float32,
    focal_y: Float32,
    lowpass_variance: Float32 = LOWPASS_FILTER_VARIANCE
) -> Covariance2D:
    """
    Project 3D Gaussian covariance into 2D screen space:
    Sigma_2D = J * W * Sigma_3D * W^T * J^T + sigma_lowpass^2 * I_2x2.
    """
    var J = ProjectiveJacobian.compute(cam_pos, focal_x, focal_y)

    # Let T = J * W (2x3 matrix)
    # T0 = J00 * W0 + J02 * W2
    # T1 = J11 * W1 + J12 * W2
    var t00 = J.j00 * view_rot.m00 + J.j02 * view_rot.m20
    var t01 = J.j00 * view_rot.m01 + J.j02 * view_rot.m21
    var t02 = J.j00 * view_rot.m02 + J.j02 * view_rot.m22

    var t10 = J.j11 * view_rot.m10 + J.j12 * view_rot.m20
    var t11 = J.j11 * view_rot.m11 + J.j12 * view_rot.m21
    var t12 = J.j11 * view_rot.m12 + J.j12 * view_rot.m22

    # Sigma_3D full matrix
    var s00 = cov3d.c00
    var s01 = cov3d.c01
    var s02 = cov3d.c02
    var s11 = cov3d.c11
    var s12 = cov3d.c12
    var s22 = cov3d.c22

    # Let Q = T * Sigma_3D (2x3 matrix)
    # Row 0:
    var q00 = t00 * s00 + t01 * s01 + t02 * s02
    var q01 = t00 * s01 + t01 * s11 + t02 * s12
    var q02 = t00 * s02 + t01 * s12 + t02 * s22

    # Row 1:
    var q10 = t10 * s00 + t11 * s01 + t12 * s02
    var q11 = t10 * s01 + t11 * s11 + t12 * s12
    var q12 = t10 * s02 + t11 * s12 + t12 * s22

    # Sigma_2D = Q * T^T
    var c00 = q00 * t00 + q01 * t01 + q02 * t02 + lowpass_variance
    var c01 = q00 * t10 + q01 * t11 + q02 * t12
    var c11 = q10 * t10 + q11 * t11 + q12 * t12 + lowpass_variance

    return Covariance2D(c00, c01, c11)

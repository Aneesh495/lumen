"""
3D Gaussian spatial covariance matrix construction, symmetry enforcement, and properties.
"""

from .vector import Vec3f
from .matrix import Mat3x3f
from .quaternion import Quatf

struct Covariance3D(ImplicitlyCopyable):
    """
    Symmetric positive-definite 3D covariance matrix: Sigma = R * S * S^T * R^T.
    Stores the 6 unique elements: (c00, c01, c02, c11, c12, c22).
    """
    var c00: Float32 # Sigma[0,0]
    var c01: Float32 # Sigma[0,1] == Sigma[1,0]
    var c02: Float32 # Sigma[0,2] == Sigma[2,0]
    var c11: Float32 # Sigma[1,1]
    var c12: Float32 # Sigma[1,2] == Sigma[2,1]
    var c22: Float32 # Sigma[2,2]

    def __init__(out self):
        self.c00 = 1.0
        self.c01 = 0.0
        self.c02 = 0.0
        self.c11 = 1.0
        self.c12 = 0.0
        self.c22 = 1.0

    def __init__(
        out self,
        c00: Float32, c01: Float32, c02: Float32,
        c11: Float32, c12: Float32, c22: Float32
    ):
        self.c00 = c00
        self.c01 = c01
        self.c02 = c02
        self.c11 = c11
        self.c12 = c12
        self.c22 = c22

    @staticmethod
    def from_scale_and_rotation(scale: Vec3f, rotation: Quatf) -> Covariance3D:
        """
        Construct Sigma = R * S * S^T * R^T from scale vector and orientation quaternion.
        Clamps scale to prevent singular zero-volume degeneracies.
        """
        var sx = scale.x
        if sx < 0.00001:
            sx = 0.00001
        var sy = scale.y
        if sy < 0.00001:
            sy = 0.00001
        var sz = scale.z
        if sz < 0.00001:
            sz = 0.00001

        var R = rotation.to_rotation_matrix()

        # M = R * S where S is diagonal(sx, sy, sz)
        # Column j of M is column j of R scaled by s[j]
        var m00 = R.m00 * sx
        var m01 = R.m01 * sy
        var m02 = R.m02 * sz

        var m10 = R.m10 * sx
        var m11 = R.m11 * sy
        var m12 = R.m12 * sz

        var m20 = R.m20 * sx
        var m21 = R.m21 * sy
        var m22 = R.m22 * sz

        # Sigma = M * M^T
        var c00 = m00 * m00 + m01 * m01 + m02 * m02
        var c01 = m00 * m10 + m01 * m11 + m02 * m12
        var c02 = m00 * m20 + m01 * m21 + m02 * m22

        var c11 = m10 * m10 + m11 * m11 + m12 * m12
        var c12 = m10 * m20 + m11 * m21 + m12 * m22

        var c22 = m20 * m20 + m21 * m21 + m22 * m22

        return Covariance3D(c00, c01, c02, c11, c12, c22)

    def to_mat3(self) -> Mat3x3f:
        return Mat3x3f(
            self.c00, self.c01, self.c02,
            self.c01, self.c11, self.c12,
            self.c02, self.c12, self.c22
        )

    def trace(self) -> Float32:
        return self.c00 + self.c11 + self.c22

    def determinant(self) -> Float32:
        return self.to_mat3().determinant()

    def is_positive_definite(self) -> Bool:
        """
        Verify positive definiteness using Sylvester's criterion:
        1. c00 > 0
        2. c00 * c11 - c01^2 > 0
        3. det(Sigma) > 0
        """
        if self.c00 <= 0.0:
            return False
        var minor_2x2 = self.c00 * self.c11 - self.c01 * self.c01
        if minor_2x2 <= 0.0:
            return False
        var det = self.determinant()
        return det > 0.0

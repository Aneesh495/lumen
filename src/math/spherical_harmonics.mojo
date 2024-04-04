"""
Real spherical harmonics (SH) basis evaluation up to degree 3 for view-dependent radiance.
"""

from .vector import Vec3f
from src.core.types import (
    SH_DEGREE_0_COEFFS,
    SH_DEGREE_1_COEFFS,
    SH_DEGREE_2_COEFFS,
    SH_DEGREE_3_COEFFS,
    SH_CHANNELS_RGB,
    MAX_SH_COEFFS_PER_GAUSSIAN,
)
from std.collections import List

# Mathematical constants for spherical harmonics normalization factors
comptime SH_C0: Float32 = 0.28209479177387814  # 0.5 * sqrt(1/pi)
comptime SH_C1: Float32 = 0.4886025119029199   # 0.5 * sqrt(3/pi)
comptime SH_C2_0: Float32 = 1.0925484305920792 # 0.5 * sqrt(15/pi)
comptime SH_C2_1: Float32 = 0.31539156525252005# 0.25 * sqrt(5/pi)
comptime SH_C2_2: Float32 = 0.5462742152960396 # 0.25 * sqrt(15/pi)
comptime SH_C3_0: Float32 = 0.5900435899266435 # 0.25 * sqrt(35/pi)
comptime SH_C3_1: Float32 = 2.890611442640554  # 0.5 * sqrt(105/pi)
comptime SH_C3_2: Float32 = 0.4570457994644658 # 0.25 * sqrt(21/pi)
comptime SH_C3_3: Float32 = 0.3731763325901154 # 0.25 * sqrt(7/pi)
comptime SH_C3_4: Float32 = 1.445305721320277  # 0.25 * sqrt(105/pi)

struct SHBasis(ImplicitlyCopyable):
    """
    Evaluated real spherical harmonics basis coefficients Y_l^m for l in [0, 3].
    Contains 16 orthogonal basis elements.
    """
    var y0: Float32
    var y1: Float32
    var y2: Float32
    var y3: Float32
    var y4: Float32
    var y5: Float32
    var y6: Float32
    var y7: Float32
    var y8: Float32
    var y9: Float32
    var y10: Float32
    var y11: Float32
    var y12: Float32
    var y13: Float32
    var y14: Float32
    var y15: Float32

    def __init__(out self):
        self.y0 = SH_C0
        self.y1 = 0.0
        self.y2 = 0.0
        self.y3 = 0.0
        self.y4 = 0.0
        self.y5 = 0.0
        self.y6 = 0.0
        self.y7 = 0.0
        self.y8 = 0.0
        self.y9 = 0.0
        self.y10 = 0.0
        self.y11 = 0.0
        self.y12 = 0.0
        self.y13 = 0.0
        self.y14 = 0.0
        self.y15 = 0.0

    @staticmethod
    def evaluate(dir: Vec3f) -> SHBasis:
        """
        Evaluate all 16 SH basis polynomials for a normalized viewing direction.
        """
        var d = dir.normalized()
        var x = d.x
        var y = d.y
        var z = d.z

        var xx = x * x
        var yy = y * y
        var zz = z * z
        var xy = x * y
        var yz = y * z
        var xz = x * z

        var res = SHBasis()

        # Degree 0 (l = 0)
        res.y0 = SH_C0

        # Degree 1 (l = 1)
        res.y1 = -SH_C1 * y
        res.y2 = SH_C1 * z
        res.y3 = -SH_C1 * x

        # Degree 2 (l = 2)
        res.y4 = SH_C2_0 * xy
        res.y5 = -SH_C2_0 * yz
        res.y6 = SH_C2_1 * (2.0 * zz - xx - yy)
        res.y7 = -SH_C2_0 * xz
        res.y8 = SH_C2_2 * (xx - yy)

        # Degree 3 (l = 3)
        res.y9 = -SH_C3_0 * y * (3.0 * xx - yy)
        res.y10 = SH_C3_1 * xy * z
        res.y11 = -SH_C3_2 * y * (4.0 * zz - xx - yy)
        res.y12 = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
        res.y13 = -SH_C3_2 * x * (4.0 * zz - xx - yy)
        res.y14 = SH_C3_4 * z * (xx - yy)
        res.y15 = -SH_C3_0 * x * (xx - 3.0 * yy)

        return res

    def get(self, index: Int) -> Float32:
        if index == 0: return self.y0
        elif index == 1: return self.y1
        elif index == 2: return self.y2
        elif index == 3: return self.y3
        elif index == 4: return self.y4
        elif index == 5: return self.y5
        elif index == 6: return self.y6
        elif index == 7: return self.y7
        elif index == 8: return self.y8
        elif index == 9: return self.y9
        elif index == 10: return self.y10
        elif index == 11: return self.y11
        elif index == 12: return self.y12
        elif index == 13: return self.y13
        elif index == 14: return self.y14
        elif index == 15: return self.y15
        return 0.0


def evaluate_sh_color(
    sh_coeffs: List[Float32],
    sh_offset: Int,
    view_dir: Vec3f,
    active_degree: Int = 3
) -> Vec3f:
    """
    Evaluate RGB radiance from spherical harmonics coefficients along viewing direction.
    """
    var basis = SHBasis.evaluate(view_dir)

    var max_coeffs = 16
    if active_degree == 0:
        max_coeffs = 1
    elif active_degree == 1:
        max_coeffs = 4
    elif active_degree == 2:
        max_coeffs = 9

    var r: Float32 = 0.5
    var g: Float32 = 0.5
    var b: Float32 = 0.5

    for i in range(max_coeffs):
        var y_val = basis.get(i)
        r += sh_coeffs[sh_offset + i * 3 + 0] * y_val
        g += sh_coeffs[sh_offset + i * 3 + 1] * y_val
        b += sh_coeffs[sh_offset + i * 3 + 2] * y_val

    # Clamp to [0, 1]
    if r < 0.0: r = 0.0
    elif r > 1.0: r = 1.0

    if g < 0.0: g = 0.0
    elif g > 1.0: g = 1.0

    if b < 0.0: b = 0.0
    elif b > 1.0: b = 1.0

    return Vec3f(r, g, b)

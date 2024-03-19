"""
Unit quaternion algebra, SO(3) rotations, and spherical linear interpolation (SLERP).
"""

from .vector import Vec3f
from .matrix import Mat3x3f
from std.math import sqrt, sin, cos, acos

struct Quatf(ImplicitlyCopyable):
    """
    Unit quaternion representing 3D spatial rotation: q = w + x*i + y*j + z*k.
    """
    var w: Float32
    var x: Float32
    var y: Float32
    var z: Float32

    def __init__(out self):
        self.w = 1.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def __init__(out self, w: Float32, x: Float32, y: Float32, z: Float32):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def identity() -> Quatf:
        return Quatf(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis: Vec3f, angle_rad: Float32) -> Quatf:
        var norm_axis = axis.normalized()
        var half_angle = angle_rad * 0.5
        var s = sin(half_angle)
        return Quatf(
            cos(half_angle),
            norm_axis.x * s,
            norm_axis.y * s,
            norm_axis.z * s
        ).normalized()

    def norm_squared(self) -> Float32:
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> Float32:
        return sqrt(self.norm_squared())

    def normalized(self) -> Quatf:
        var n = self.norm()
        if n > 0.000001:
            var inv_n = 1.0 / n
            return Quatf(self.w * inv_n, self.x * inv_n, self.y * inv_n, self.z * inv_n)
        return Quatf.identity()

    def conjugate(self) -> Quatf:
        return Quatf(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> Quatf:
        var ns = self.norm_squared()
        if ns > 0.000001:
            var inv_ns = 1.0 / ns
            return Quatf(self.w * inv_ns, -self.x * inv_ns, -self.y * inv_ns, -self.z * inv_ns)
        return Quatf.identity()

    def mul(self, other: Quatf) -> Quatf:
        """
        Hamilton quaternion product: self * other.
        """
        return Quatf(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )

    def rotate_vec(self, v: Vec3f) -> Vec3f:
        """
        Rotate 3D vector v using v_rot = q * (0, v) * q^-1.
        """
        var p = Quatf(0.0, v.x, v.y, v.z)
        var q_inv = self.inverse()
        var rotated = self.mul(p).mul(q_inv)
        return Vec3f(rotated.x, rotated.y, rotated.z)

    def to_rotation_matrix(self) -> Mat3x3f:
        """
        Derive 3x3 orthonormal SO(3) rotation matrix from unit quaternion.
        """
        var q = self.normalized()
        var w = q.w
        var x = q.x
        var y = q.y
        var z = q.z

        var x2 = x + x
        var y2 = y + y
        var z2 = z + z

        var xx = x * x2
        var xy = x * y2
        var xz = x * z2
        var yy = y * y2
        var yz = y * z2
        var zz = z * z2
        var wx = w * x2
        var wy = w * y2
        var wz = w * z2

        return Mat3x3f(
            1.0 - (yy + zz), xy - wz, xz + wy,
            xy + wz, 1.0 - (xx + zz), yz - wx,
            xz - wy, yz + wx, 1.0 - (xx + yy)
        )

    def dot(self, other: Quatf) -> Float32:
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def slerp(self, other: Quatf, t: Float32) -> Quatf:
        """
        Spherical linear interpolation between two orientations on S^3 manifold.
        """
        var q1 = self.normalized()
        var q2 = other.normalized()

        var dot_prod = q1.dot(q2)

        # Ensure shortest path on 4D hypersphere
        if dot_prod < 0.0:
            q2 = Quatf(-q2.w, -q2.x, -q2.y, -q2.z)
            dot_prod = -dot_prod

        # If quaternions are extremely close, use normalized linear interpolation
        if dot_prod > 0.9995:
            var w = q1.w + t * (q2.w - q1.w)
            var x = q1.x + t * (q2.x - q1.x)
            var y = q1.y + t * (q2.y - q1.y)
            var z = q1.z + t * (q2.z - q1.z)
            return Quatf(w, x, y, z).normalized()

        var theta_0 = acos(dot_prod)
        var theta = theta_0 * t
        var sin_theta = sin(theta)
        var sin_theta_0 = sin(theta_0)

        var s0 = cos(theta) - dot_prod * sin_theta / sin_theta_0
        var s1 = sin_theta / sin_theta_0

        return Quatf(
            s0 * q1.w + s1 * q2.w,
            s0 * q1.x + s1 * q2.x,
            s0 * q1.y + s1 * q2.y,
            s0 * q1.z + s1 * q2.z
        ).normalized()

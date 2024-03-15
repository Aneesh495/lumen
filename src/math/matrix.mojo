"""
High-performance 2x2, 3x3, and 4x4 matrix representations and linear algebraic operations.
"""

from .vector import Vec2f, Vec3f, Vec4f
from std.math import tan

struct Mat2x2f(ImplicitlyCopyable):
    var m00: Float32
    var m01: Float32
    var m10: Float32
    var m11: Float32

    def __init__(out self):
        self.m00 = 1.0
        self.m01 = 0.0
        self.m10 = 0.0
        self.m11 = 1.0

    def __init__(out self, m00: Float32, m01: Float32, m10: Float32, m11: Float32):
        self.m00 = m00
        self.m01 = m01
        self.m10 = m10
        self.m11 = m11

    def determinant(self) -> Float32:
        return self.m00 * self.m11 - self.m01 * self.m10

    def trace(self) -> Float32:
        return self.m00 + self.m11

    def transpose(self) -> Mat2x2f:
        return Mat2x2f(self.m00, self.m10, self.m01, self.m11)

    def inverse(self) -> Mat2x2f:
        var det = self.determinant()
        if det > -0.0000001 and det < 0.0000001:
            return Mat2x2f(0.0, 0.0, 0.0, 0.0)
        var inv_det = 1.0 / det
        return Mat2x2f(
            self.m11 * inv_det,
            -self.m01 * inv_det,
            -self.m10 * inv_det,
            self.m00 * inv_det
        )

    def mul_vec(self, v: Vec2f) -> Vec2f:
        return Vec2f(
            self.m00 * v.x + self.m01 * v.y,
            self.m10 * v.x + self.m11 * v.y
        )

    def add(self, other: Mat2x2f) -> Mat2x2f:
        return Mat2x2f(
            self.m00 + other.m00,
            self.m01 + other.m01,
            self.m10 + other.m10,
            self.m11 + other.m11
        )


struct Mat3x3f(ImplicitlyCopyable):
    var m00: Float32
    var m01: Float32
    var m02: Float32
    var m10: Float32
    var m11: Float32
    var m12: Float32
    var m20: Float32
    var m21: Float32
    var m22: Float32

    def __init__(out self):
        self.m00 = 1.0
        self.m01 = 0.0
        self.m02 = 0.0
        self.m10 = 0.0
        self.m11 = 1.0
        self.m12 = 0.0
        self.m20 = 0.0
        self.m21 = 0.0
        self.m22 = 1.0

    def __init__(
        out self,
        m00: Float32, m01: Float32, m02: Float32,
        m10: Float32, m11: Float32, m12: Float32,
        m20: Float32, m21: Float32, m22: Float32
    ):
        self.m00 = m00
        self.m01 = m01
        self.m02 = m02
        self.m10 = m10
        self.m11 = m11
        self.m12 = m12
        self.m20 = m20
        self.m21 = m21
        self.m22 = m22

    @staticmethod
    def diagonal(d: Vec3f) -> Mat3x3f:
        return Mat3x3f(
            d.x, 0.0, 0.0,
            0.0, d.y, 0.0,
            0.0, 0.0, d.z
        )

    def determinant(self) -> Float32:
        return (
            self.m00 * (self.m11 * self.m22 - self.m12 * self.m21) -
            self.m01 * (self.m10 * self.m22 - self.m12 * self.m20) +
            self.m02 * (self.m10 * self.m21 - self.m11 * self.m20)
        )

    def transpose(self) -> Mat3x3f:
        return Mat3x3f(
            self.m00, self.m10, self.m20,
            self.m01, self.m11, self.m21,
            self.m02, self.m12, self.m22
        )

    def mul_mat(self, other: Mat3x3f) -> Mat3x3f:
        return Mat3x3f(
            self.m00 * other.m00 + self.m01 * other.m10 + self.m02 * other.m20,
            self.m00 * other.m01 + self.m01 * other.m11 + self.m02 * other.m21,
            self.m00 * other.m02 + self.m01 * other.m12 + self.m02 * other.m22,

            self.m10 * other.m00 + self.m11 * other.m10 + self.m12 * other.m20,
            self.m10 * other.m01 + self.m11 * other.m11 + self.m12 * other.m21,
            self.m10 * other.m02 + self.m11 * other.m12 + self.m12 * other.m22,

            self.m20 * other.m00 + self.m21 * other.m10 + self.m22 * other.m20,
            self.m20 * other.m01 + self.m21 * other.m11 + self.m22 * other.m21,
            self.m20 * other.m02 + self.m21 * other.m12 + self.m22 * other.m22
        )

    def mul_vec(self, v: Vec3f) -> Vec3f:
        return Vec3f(
            self.m00 * v.x + self.m01 * v.y + self.m02 * v.z,
            self.m10 * v.x + self.m11 * v.y + self.m12 * v.z,
            self.m20 * v.x + self.m21 * v.y + self.m22 * v.z
        )

    def inverse(self) -> Mat3x3f:
        var det = self.determinant()
        if det > -0.0000001 and det < 0.0000001:
            return Mat3x3f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        var inv_det = 1.0 / det

        var c00 = (self.m11 * self.m22 - self.m12 * self.m21) * inv_det
        var c01 = (self.m02 * self.m21 - self.m01 * self.m22) * inv_det
        var c02 = (self.m01 * self.m12 - self.m02 * self.m11) * inv_det

        var c10 = (self.m12 * self.m20 - self.m10 * self.m22) * inv_det
        var c11 = (self.m00 * self.m22 - self.m02 * self.m20) * inv_det
        var c12 = (self.m02 * self.m10 - self.m00 * self.m12) * inv_det

        var c20 = (self.m10 * self.m21 - self.m11 * self.m20) * inv_det
        var c21 = (self.m01 * self.m20 - self.m00 * self.m21) * inv_det
        var c22 = (self.m00 * self.m11 - self.m01 * self.m10) * inv_det

        return Mat3x3f(
            c00, c01, c02,
            c10, c11, c12,
            c20, c21, c22
        )


struct Mat4x4f(ImplicitlyCopyable):
    var m00: Float32
    var m01: Float32
    var m02: Float32
    var m03: Float32
    var m10: Float32
    var m11: Float32
    var m12: Float32
    var m13: Float32
    var m20: Float32
    var m21: Float32
    var m22: Float32
    var m23: Float32
    var m30: Float32
    var m31: Float32
    var m32: Float32
    var m33: Float32

    def __init__(out self):
        self.m00 = 1.0
        self.m01 = 0.0
        self.m02 = 0.0
        self.m03 = 0.0
        self.m10 = 0.0
        self.m11 = 1.0
        self.m12 = 0.0
        self.m13 = 0.0
        self.m20 = 0.0
        self.m21 = 0.0
        self.m22 = 1.0
        self.m23 = 0.0
        self.m30 = 0.0
        self.m31 = 0.0
        self.m32 = 0.0
        self.m33 = 1.0

    def mul_vec(self, v: Vec4f) -> Vec4f:
        return Vec4f(
            self.m00 * v.x + self.m01 * v.y + self.m02 * v.z + self.m03 * v.w,
            self.m10 * v.x + self.m11 * v.y + self.m12 * v.z + self.m13 * v.w,
            self.m20 * v.x + self.m21 * v.y + self.m22 * v.z + self.m23 * v.w,
            self.m30 * v.x + self.m31 * v.y + self.m32 * v.z + self.m33 * v.w
        )

    def mul_mat(self, other: Mat4x4f) -> Mat4x4f:
        var res = Mat4x4f()
        res.m00 = self.m00 * other.m00 + self.m01 * other.m10 + self.m02 * other.m20 + self.m03 * other.m30
        res.m01 = self.m00 * other.m01 + self.m01 * other.m11 + self.m02 * other.m21 + self.m03 * other.m31
        res.m02 = self.m00 * other.m02 + self.m01 * other.m12 + self.m02 * other.m22 + self.m03 * other.m32
        res.m03 = self.m00 * other.m03 + self.m01 * other.m13 + self.m02 * other.m23 + self.m03 * other.m33

        res.m10 = self.m10 * other.m00 + self.m11 * other.m10 + self.m12 * other.m20 + self.m13 * other.m30
        res.m11 = self.m10 * other.m01 + self.m11 * other.m11 + self.m12 * other.m21 + self.m13 * other.m31
        res.m12 = self.m10 * other.m02 + self.m11 * other.m12 + self.m12 * other.m22 + self.m13 * other.m32
        res.m13 = self.m10 * other.m03 + self.m11 * other.m13 + self.m12 * other.m23 + self.m13 * other.m33

        res.m20 = self.m20 * other.m00 + self.m21 * other.m10 + self.m22 * other.m20 + self.m23 * other.m30
        res.m21 = self.m20 * other.m01 + self.m21 * other.m11 + self.m22 * other.m21 + self.m23 * other.m31
        res.m22 = self.m20 * other.m02 + self.m21 * other.m12 + self.m22 * other.m22 + self.m23 * other.m32
        res.m23 = self.m20 * other.m03 + self.m21 * other.m13 + self.m22 * other.m23 + self.m23 * other.m33

        res.m30 = self.m30 * other.m00 + self.m31 * other.m10 + self.m32 * other.m20 + self.m33 * other.m30
        res.m31 = self.m30 * other.m01 + self.m31 * other.m11 + self.m32 * other.m21 + self.m33 * other.m31
        res.m32 = self.m30 * other.m02 + self.m31 * other.m12 + self.m32 * other.m22 + self.m33 * other.m32
        res.m33 = self.m30 * other.m03 + self.m31 * other.m13 + self.m32 * other.m23 + self.m33 * other.m33
        return res

    @staticmethod
    def look_at(eye: Vec3f, target: Vec3f, up: Vec3f) -> Mat4x4f:
        var f = target.sub(eye).normalized()
        var s = f.cross(up).normalized()
        var u = s.cross(f)

        var res = Mat4x4f()
        res.m00 = s.x
        res.m01 = s.y
        res.m02 = s.z
        res.m03 = -s.dot(eye)

        res.m10 = u.x
        res.m11 = u.y
        res.m12 = u.z
        res.m13 = -u.dot(eye)

        res.m20 = -f.x
        res.m21 = -f.y
        res.m22 = -f.z
        res.m23 = f.dot(eye)

        res.m30 = 0.0
        res.m31 = 0.0
        res.m32 = 0.0
        res.m33 = 1.0
        return res

    @staticmethod
    def perspective(fov_y_rad: Float32, aspect: Float32, near: Float32, far: Float32) -> Mat4x4f:
        var tan_half_fov = tan(fov_y_rad * 0.5)
        var res = Mat4x4f()
        res.m00 = 1.0 / (aspect * tan_half_fov)
        res.m01 = 0.0
        res.m02 = 0.0
        res.m03 = 0.0

        res.m10 = 0.0
        res.m11 = 1.0 / tan_half_fov
        res.m12 = 0.0
        res.m13 = 0.0

        res.m20 = 0.0
        res.m21 = 0.0
        res.m22 = -(far + near) / (far - near)
        res.m23 = -(2.0 * far * near) / (far - near)

        res.m30 = 0.0
        res.m31 = 0.0
        res.m32 = -1.0
        res.m33 = 0.0
        return res

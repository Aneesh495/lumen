"""
High-performance 2D, 3D, and 4D vector representations with SIMD arithmetic.
"""

from std.math import sqrt, min, max
from std.simd import SIMD

struct Vec2f(ImplicitlyCopyable):
    var x: Float32
    var y: Float32

    def __init__(out self):
        self.x = 0.0
        self.y = 0.0

    def __init__(out self, scalar: Float32):
        self.x = scalar
        self.y = scalar

    def __init__(out self, x: Float32, y: Float32):
        self.x = x
        self.y = y

    def add(self, other: Vec2f) -> Vec2f:
        return Vec2f(self.x + other.x, self.y + other.y)

    def sub(self, other: Vec2f) -> Vec2f:
        return Vec2f(self.x - other.x, self.y - other.y)

    def mul(self, scalar: Float32) -> Vec2f:
        return Vec2f(self.x * scalar, self.y * scalar)

    def div(self, scalar: Float32) -> Vec2f:
        return Vec2f(self.x / scalar, self.y / scalar)

    def dot(self, other: Vec2f) -> Float32:
        return self.x * other.x + self.y * other.y

    def length_squared(self) -> Float32:
        return self.x * self.x + self.y * self.y

    def length(self) -> Float32:
        return sqrt(self.length_squared())

    def normalized(self) -> Vec2f:
        var l = self.length()
        if l > 0.000001:
            return self.div(l)
        return Vec2f(0.0, 0.0)

    def distance(self, other: Vec2f) -> Float32:
        return self.sub(other).length()


struct Vec3f(ImplicitlyCopyable):
    var x: Float32
    var y: Float32
    var z: Float32

    def __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def __init__(out self, scalar: Float32):
        self.x = scalar
        self.y = scalar
        self.z = scalar

    def __init__(out self, x: Float32, y: Float32, z: Float32):
        self.x = x
        self.y = y
        self.z = z

    def add(self, other: Vec3f) -> Vec3f:
        return Vec3f(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other: Vec3f) -> Vec3f:
        return Vec3f(self.x - other.x, self.y - other.y, self.z - other.z)

    def mul(self, scalar: Float32) -> Vec3f:
        return Vec3f(self.x * scalar, self.y * scalar, self.z * scalar)

    def div(self, scalar: Float32) -> Vec3f:
        return Vec3f(self.x / scalar, self.y / scalar, self.z / scalar)

    def mul_components(self, other: Vec3f) -> Vec3f:
        return Vec3f(self.x * other.x, self.y * other.y, self.z * other.z)

    def dot(self, other: Vec3f) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3f) -> Vec3f:
        return Vec3f(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length_squared(self) -> Float32:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> Float32:
        return sqrt(self.length_squared())

    def normalized(self) -> Vec3f:
        var l = self.length()
        if l > 0.000001:
            return self.div(l)
        return Vec3f(0.0, 0.0, 0.0)

    def distance(self, other: Vec3f) -> Float32:
        return self.sub(other).length()

    def lerp(self, other: Vec3f, t: Float32) -> Vec3f:
        return self.add(other.sub(self).mul(t))

    def to_simd(self) -> SIMD[DType.float32, 4]:
        return SIMD[DType.float32, 4](self.x, self.y, self.z, 0.0)


struct Vec4f(ImplicitlyCopyable):
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    def __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0

    def __init__(out self, scalar: Float32):
        self.x = scalar
        self.y = scalar
        self.z = scalar
        self.w = scalar

    def __init__(out self, x: Float32, y: Float32, z: Float32, w: Float32):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __init__(out self, v: Vec3f, w: Float32):
        self.x = v.x
        self.y = v.y
        self.z = v.z
        self.w = w

    def add(self, other: Vec4f) -> Vec4f:
        return Vec4f(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def sub(self, other: Vec4f) -> Vec4f:
        return Vec4f(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def mul(self, scalar: Float32) -> Vec4f:
        return Vec4f(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)

    def div(self, scalar: Float32) -> Vec4f:
        return Vec4f(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)

    def dot(self, other: Vec4f) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def to_xyz(self) -> Vec3f:
        return Vec3f(self.x, self.y, self.z)

    def to_simd(self) -> SIMD[DType.float32, 4]:
        return SIMD[DType.float32, 4](self.x, self.y, self.z, self.w)

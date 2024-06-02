"""
Depth buffer accumulation, range normalization, and surface normal extraction.
"""

from src.core.buffer import FloatBuffer
from src.math.vector import Vec3f
from std.math import sqrt

struct DepthMap:
    """
    2D dense depth buffer storing floating-point optical distances per pixel.
    """
    var width: Int
    var height: Int
    var buffer: FloatBuffer

    def __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.buffer = FloatBuffer(width * height, 0.0)

    def get_depth(self, x: Int, y: Int) -> Float32:
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            return self.buffer.get(y * self.width + x)
        return 0.0

    def set_depth(mut self, x: Int, y: Int, val: Float32):
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            self.buffer.set(y * self.width + x, val)

    def compute_min_max(self, mut out_min: Float32, mut out_max: Float32):
        var min_val: Float32 = 1e9
        var max_val: Float32 = -1e9
        var n = self.width * self.height

        for i in range(n):
            var d = self.buffer.get(i)
            if d > 0.001:
                if d < min_val: min_val = d
                if d > max_val: max_val = d

        if min_val > max_val:
            min_val = 0.0
            max_val = 1.0

        out_min = min_val
        out_max = max_val

    def extract_normal(self, x: Int, y: Int) -> Vec3f:
        """
        Derive local surface normal from 2D finite difference depth gradients.
        """
        var x0 = x - 1
        if x0 < 0: x0 = 0
        var x1 = x + 1
        if x1 >= self.width: x1 = self.width - 1

        var y0 = y - 1
        if y0 < 0: y0 = 0
        var y1 = y + 1
        if y1 >= self.height: y1 = self.height - 1

        var dz_dx = (self.get_depth(x1, y) - self.get_depth(x0, y)) * 0.5
        var dz_dy = (self.get_depth(x, y1) - self.get_depth(x, y0)) * 0.5

        # Normal vector: (-dz/dx, -dz/dy, 1.0) normalized
        var n = Vec3f(-dz_dx, -dz_dy, 1.0).normalized()
        return n

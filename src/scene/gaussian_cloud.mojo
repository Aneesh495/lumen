"""
High-capacity Gaussian cloud container managing scene collections and bounding extents.
"""

from .gaussian import Gaussian3D, GaussianSoA
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from src.core.types import MAX_SH_COEFFS_PER_GAUSSIAN
from std.collections import List
from std.math import min, max, sqrt

struct GaussianCloud:
    """
    Primary container representing a complete 3D Gaussian radiance field scene.
    """
    var soa: GaussianSoA
    var bbox_min: Vec3f
    var bbox_max: Vec3f

    def __init__(out self, capacity: Int = 1000):
        self.soa = GaussianSoA(capacity)
        self.bbox_min = Vec3f(1e9, 1e9, 1e9)
        self.bbox_max = Vec3f(-1e9, -1e9, -1e9)

    def count(self) -> Int:
        return self.soa.count()

    def add_gaussian(mut self, g: Gaussian3D, sh: List[Float32]):
        self.soa.append(g, sh)

        # Update bounding extents
        var px = g.position.x
        var py = g.position.y
        var pz = g.position.z

        if px < self.bbox_min.x: self.bbox_min.x = px
        if py < self.bbox_min.y: self.bbox_min.y = py
        if pz < self.bbox_min.z: self.bbox_min.z = pz

        if px > self.bbox_max.x: self.bbox_max.x = px
        if py > self.bbox_max.y: self.bbox_max.y = py
        if pz > self.bbox_max.z: self.bbox_max.z = pz

    def get_gaussian(self, index: Int) -> Gaussian3D:
        return self.soa.get_gaussian(index)

    def get_center_of_mass(self) -> Vec3f:
        var n = self.count()
        if n == 0:
            return Vec3f(0.0, 0.0, 0.0)

        var sx: Float32 = 0.0
        var sy: Float32 = 0.0
        var sz: Float32 = 0.0

        for i in range(n):
            sx += self.soa.pos_x[i]
            sy += self.soa.pos_y[i]
            sz += self.soa.pos_z[i]

        var inv_n = 1.0 / Float32(n)
        return Vec3f(sx * inv_n, sy * inv_n, sz * inv_n)

    def get_extent_radius(self) -> Float32:
        var center = self.get_center_of_mass()
        var max_d_sq: Float32 = 0.0
        var n = self.count()

        for i in range(n):
            var pos = Vec3f(self.soa.pos_x[i], self.soa.pos_y[i], self.soa.pos_z[i])
            var d_sq = pos.distance(center) * pos.distance(center)
            if d_sq > max_d_sq:
                max_d_sq = d_sq

        return sqrt(max_d_sq)

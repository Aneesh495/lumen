"""
Axis-Aligned Bounding Box (AABB) representations, slab method ray intersections, and spatial queries.
"""

from src.math.vector import Vec3f
from src.scene.camera import Ray3f
from std.math import min, max

struct AABB3f(ImplicitlyCopyable):
    var min_pt: Vec3f
    var max_pt: Vec3f

    def __init__(out self):
        self.min_pt = Vec3f(1e9, 1e9, 1e9)
        self.max_pt = Vec3f(-1e9, -1e9, -1e9)

    def __init__(out self, min_pt: Vec3f, max_pt: Vec3f):
        self.min_pt = min_pt
        self.max_pt = max_pt

    def center(self) -> Vec3f:
        return self.min_pt.add(self.max_pt).mul(0.5)

    def extents(self) -> Vec3f:
        return self.max_pt.sub(self.min_pt)

    def surface_area(self) -> Float32:
        var d = self.extents()
        if d.x < 0.0 or d.y < 0.0 or d.z < 0.0:
            return 0.0
        return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)

    def contains(self, p: Vec3f) -> Bool:
        return (
            p.x >= self.min_pt.x and p.x <= self.max_pt.x and
            p.y >= self.min_pt.y and p.y <= self.max_pt.y and
            p.z >= self.min_pt.z and p.z <= self.max_pt.z
        )

    def intersects(self, other: AABB3f) -> Bool:
        return (
            self.min_pt.x <= other.max_pt.x and self.max_pt.x >= other.min_pt.x and
            self.min_pt.y <= other.max_pt.y and self.max_pt.y >= other.min_pt.y and
            self.min_pt.z <= other.max_pt.z and self.max_pt.z >= other.min_pt.z
        )

    def expand_point(mut self, p: Vec3f):
        if p.x < self.min_pt.x: self.min_pt.x = p.x
        if p.y < self.min_pt.y: self.min_pt.y = p.y
        if p.z < self.min_pt.z: self.min_pt.z = p.z

        if p.x > self.max_pt.x: self.max_pt.x = p.x
        if p.y > self.max_pt.y: self.max_pt.y = p.y
        if p.z > self.max_pt.z: self.max_pt.z = p.z

    def expand_box(mut self, other: AABB3f):
        self.expand_point(other.min_pt)
        self.expand_point(other.max_pt)

    def intersect_ray(
        self,
        ray: Ray3f,
        t_min: Float32,
        t_max: Float32,
        mut out_t_near: Float32,
        mut out_t_far: Float32
    ) -> Bool:
        """
        Slab method ray-AABB intersection test.
        """
        var inv_dx = 1.0 / (ray.direction.x if ray.direction.x != 0.0 else 1e-6)
        var inv_dy = 1.0 / (ray.direction.y if ray.direction.y != 0.0 else 1e-6)
        var inv_dz = 1.0 / (ray.direction.z if ray.direction.z != 0.0 else 1e-6)

        var t1_x = (self.min_pt.x - ray.origin.x) * inv_dx
        var t2_x = (self.max_pt.x - ray.origin.x) * inv_dx
        var tx_min = min(t1_x, t2_x)
        var tx_max = max(t1_x, t2_x)

        var t1_y = (self.min_pt.y - ray.origin.y) * inv_dy
        var t2_y = (self.max_pt.y - ray.origin.y) * inv_dy
        var ty_min = min(t1_y, t2_y)
        var ty_max = max(t1_y, t2_y)

        var t1_z = (self.min_pt.z - ray.origin.z) * inv_dz
        var t2_z = (self.max_pt.z - ray.origin.z) * inv_dz
        var tz_min = min(t1_z, t2_z)
        var tz_max = max(t1_z, t2_z)

        var t_enter = max(max(tx_min, ty_min), tz_min)
        var t_exit = min(min(tx_max, ty_max), tz_max)

        if t_enter > t_exit or t_exit < t_min or t_enter > t_max:
            return False

        out_t_near = t_enter
        out_t_far = t_exit
        return True

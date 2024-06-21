"""
Accelerated ray-splat intersection testing and interactive spatial point picking.
"""

from .bvh import LBVH
from src.scene.camera import Ray3f
from src.scene.gaussian_cloud import GaussianCloud
from src.math.vector import Vec3f
from std.collections import List

struct RayHit(ImplicitlyCopyable):
    var hit: Bool
    var splat_index: Int
    var distance: Float32
    var point: Vec3f

    def __init__(out self):
        self.hit = False
        self.splat_index = -1
        self.distance = 1e9
        self.point = Vec3f(0.0, 0.0, 0.0)

    def __init__(out self, hit: Bool, splat_idx: Int, dist: Float32, pt: Vec3f):
        self.hit = hit
        self.splat_index = splat_idx
        self.distance = dist
        self.point = pt


def raycast_closest_splat(
    bvh: LBVH,
    cloud: GaussianCloud,
    ray: Ray3f,
    max_dist: Float32 = 1000.0
) -> RayHit:
    """
    Traverse LBVH to find the closest Gaussian intersecting the cast ray.
    """
    if bvh.node_count() == 0 or bvh.root_index == -1:
        return RayHit()

    var closest_hit = RayHit()
    closest_hit.distance = max_dist

    # Traversal stack (fixed capacity for deep trees)
    var stack = List[Int](capacity=64)
    stack.append(bvh.root_index)

    var dummy_near: Float32 = 0.0
    var dummy_far: Float32 = 0.0

    while len(stack) > 0:
        var curr_idx = stack.pop()
        var node = bvh.nodes[curr_idx]

        var intersects = node.bbox.intersect_ray(
            ray,
            0.0,
            closest_hit.distance,
            dummy_near,
            dummy_far
        )

        if not intersects:
            continue

        if node.is_leaf():
            var s_idx = node.splat_index
            if s_idx >= 0 and s_idx < cloud.count():
                var center = Vec3f(cloud.soa.pos_x[s_idx], cloud.soa.pos_y[s_idx], cloud.soa.pos_z[s_idx])
                # Projection of center onto ray
                var oc = center.sub(ray.origin)
                var t_proj = oc.dot(ray.direction)

                if t_proj > 0.0 and t_proj < closest_hit.distance:
                    var pt_on_ray = ray.at(t_proj)
                    var dist_to_center = pt_on_ray.distance(center)

                    var scale = Vec3f(cloud.soa.scale_x[s_idx], cloud.soa.scale_y[s_idx], cloud.soa.scale_z[s_idx])
                    var r = scale.x
                    if scale.y > r: r = scale.y
                    if scale.z > r: r = scale.z

                    if dist_to_center <= r * 3.0: # Inside 3-sigma ellipsoid
                        closest_hit.hit = True
                        closest_hit.splat_index = s_idx
                        closest_hit.distance = t_proj
                        closest_hit.point = pt_on_ray
        else:
            if node.left_child >= 0:
                stack.append(node.left_child)
            if node.right_child >= 0:
                stack.append(node.right_child)

    return closest_hit

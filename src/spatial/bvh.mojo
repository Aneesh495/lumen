"""
Linear Bounding Volume Hierarchy (LBVH) construction from Morton-coded Gaussian clouds.
"""

from .aabb import AABB3f
from .morton import compute_morton_3d
from src.math.vector import Vec3f
from src.scene.gaussian_cloud import GaussianCloud
from src.scene.camera import Ray3f
from src.core.buffer import UInt64Buffer, IntBuffer
from src.raster.radix_sort import radix_sort_64bit
from std.collections import List

struct BVHNode(ImplicitlyCopyable):
    var bbox: AABB3f
    var left_child: Int
    var right_child: Int
    var splat_index: Int

    def __init__(out self):
        self.bbox = AABB3f()
        self.left_child = -1
        self.right_child = -1
        self.splat_index = -1

    def is_leaf(self) -> Bool:
        return self.left_child == -1 and self.right_child == -1


struct LBVH:
    """
    Hierarchical spatial acceleration structure built over 3D Gaussian collections.
    """
    var nodes: List[BVHNode]
    var root_index: Int

    def __init__(out self):
        self.nodes = List[BVHNode]()
        self.root_index = -1

    def node_count(self) -> Int:
        return len(self.nodes)

    @staticmethod
    def build_from_cloud(cloud: GaussianCloud) -> LBVH:
        var bvh = LBVH()
        var n = cloud.count()
        if n == 0:
            return bvh^

        var bbox_min = cloud.bbox_min
        var bbox_max = cloud.bbox_max
        var extent = bbox_max.sub(bbox_min)

        var inv_ex = 1.0 / (extent.x if extent.x > 0.001 else 1.0)
        var inv_ey = 1.0 / (extent.y if extent.y > 0.001 else 1.0)
        var inv_ez = 1.0 / (extent.z if extent.z > 0.001 else 1.0)

        # 1. Compute Morton codes
        var morton_keys = UInt64Buffer(n, 0)
        var splat_indices = IntBuffer(n, 0)

        for i in range(n):
            var pos = Vec3f(cloud.soa.pos_x[i], cloud.soa.pos_y[i], cloud.soa.pos_z[i])
            var norm_pos = Vec3f(
                (pos.x - bbox_min.x) * inv_ex,
                (pos.y - bbox_min.y) * inv_ey,
                (pos.z - bbox_min.z) * inv_ez
            )
            var m_code = compute_morton_3d(norm_pos)
            morton_keys.set(i, UInt64(m_code))
            splat_indices.set(i, i)

        # 2. Sort by Morton code
        radix_sort_64bit(morton_keys, splat_indices)

        # 3. Build Leaf Nodes
        for i in range(n):
            var orig_idx = splat_indices.get(i)
            var pos = Vec3f(cloud.soa.pos_x[orig_idx], cloud.soa.pos_y[orig_idx], cloud.soa.pos_z[orig_idx])
            var scale = Vec3f(cloud.soa.scale_x[orig_idx], cloud.soa.scale_y[orig_idx], cloud.soa.scale_z[orig_idx])
            var r = scale.x
            if scale.y > r: r = scale.y
            if scale.z > r: r = scale.z
            var r3 = r * 3.0 # 3-sigma radius

            var leaf = BVHNode()
            leaf.bbox = AABB3f(
                Vec3f(pos.x - r3, pos.y - r3, pos.z - r3),
                Vec3f(pos.x + r3, pos.y + r3, pos.z + r3)
            )
            leaf.splat_index = orig_idx
            bvh.nodes.append(leaf)

        # 4. Hierarchical Pairing to Root
        # Bottom-up pairing of adjacent nodes
        var current_level_start = 0
        var current_level_count = n

        while current_level_count > 1:
            var next_level_count = (current_level_count + 1) // 2
            for i in range(next_level_count):
                var left_idx = current_level_start + i * 2
                var right_idx = left_idx + 1

                var parent = BVHNode()
                parent.left_child = left_idx
                parent.bbox = bvh.nodes[left_idx].bbox

                if right_idx < current_level_start + current_level_count:
                    parent.right_child = right_idx
                    parent.bbox.expand_box(bvh.nodes[right_idx].bbox)

                bvh.nodes.append(parent)

            current_level_start += current_level_count
            current_level_count = next_level_count

        bvh.root_index = len(bvh.nodes) - 1
        return bvh^

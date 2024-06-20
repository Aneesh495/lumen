"""
Unit tests for Linear Bounding Volume Hierarchy (LBVH) construction and node hierarchy.
"""

from src.spatial.bvh import LBVH, BVHNode
from src.scene.datasets import create_synthetic_cube_scene
from src.math.vector import Vec3f
from std.testing import assert_true

def test_lbvh_cube_construction() raises:
    # 4x4x4 cube (64 splats)
    var cloud = create_synthetic_cube_scene(4, Float32(0.5))
    var bvh = LBVH.build_from_cloud(cloud)

    assert_true(bvh.node_count() > 64, "Total nodes must include leaves and internal nodes")
    assert_true(bvh.root_index >= 0, "Root node must be defined")

    var root = bvh.nodes[bvh.root_index]
    assert_true(not root.is_leaf(), "Root node of 64 splats must be an internal node")

    # Root bbox must contain center of mass (0, 0, 0)
    assert_true(root.bbox.contains(Vec3f(0.0, 0.0, 0.0)), "Root bbox must contain origin")

    # Root bbox must contain cloud bounds
    assert_true(root.bbox.min_pt.x <= cloud.bbox_min.x, "Root bbox must encompass min extent")
    assert_true(root.bbox.max_pt.x >= cloud.bbox_max.x, "Root bbox must encompass max extent")

    # Verify first 64 nodes are valid leaves
    for i in range(64):
        var leaf = bvh.nodes[i]
        assert_true(leaf.is_leaf(), "First N nodes must be leaf nodes")
        assert_true(leaf.splat_index >= 0 and leaf.splat_index < 64, "Leaf splat index must be in range")

def main() raises:
    print("Testing LBVH Cube Construction...")
    test_lbvh_cube_construction()
    print("All LBVH tests passed successfully!")

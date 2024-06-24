"""
Unit tests for LBVH-accelerated raycasting and closest Gaussian picking.
"""

from src.spatial.bvh import LBVH
from src.spatial.raycast import raycast_closest_splat, RayHit
from src.scene.camera import Ray3f
from src.scene.gaussian import Gaussian3D
from src.scene.gaussian_cloud import GaussianCloud
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from std.collections import List
from std.testing import assert_true, assert_almost_equal

def test_raycast_direct_hit_and_miss() raises:
    var cloud = GaussianCloud()
    var dummy_sh = List[Float32](capacity=48)
    for _ in range(48): dummy_sh.append(0.0)

    # Splat 0: at (0, 0, 5), radius 0.5
    # Splat 1: at (0, 0, 10), radius 0.5 (behind splat 0)
    # Splat 2: at (5, 5, 5), radius 0.5 (off to the side)
    cloud.add_gaussian(Gaussian3D(Vec3f(0.0, 0.0, 5.0), Vec3f(0.5, 0.5, 0.5), Quatf.identity(), 1.0), dummy_sh)
    cloud.add_gaussian(Gaussian3D(Vec3f(0.0, 0.0, 10.0), Vec3f(0.5, 0.5, 0.5), Quatf.identity(), 1.0), dummy_sh)
    cloud.add_gaussian(Gaussian3D(Vec3f(5.0, 5.0, 5.0), Vec3f(0.5, 0.5, 0.5), Quatf.identity(), 1.0), dummy_sh)

    var bvh = LBVH.build_from_cloud(cloud)

    # 1. Cast ray from origin along +Z: (0, 0, 1)
    # Must hit splat 0 (closest) at distance 5.0
    var ray_hit = Ray3f(Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0))
    var hit_info = raycast_closest_splat(bvh, cloud, ray_hit)

    assert_true(hit_info.hit, "Direct ray must hit splat")
    assert_true(hit_info.splat_index == 0, "Must hit closest splat 0, not occluded splat 1")
    assert_almost_equal(hit_info.distance, Float32(5.0), atol=1e-3)

    # 2. Cast ray missing everything: along -Z: (0, 0, -1)
    var ray_miss = Ray3f(Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, -1.0))
    var miss_info = raycast_closest_splat(bvh, cloud, ray_miss)
    assert_true(not miss_info.hit, "Opposite ray must miss all splats")

def main() raises:
    print("Testing Raycast Direct Hit and Miss...")
    test_raycast_direct_hit_and_miss()
    print("All raycast tests passed successfully!")

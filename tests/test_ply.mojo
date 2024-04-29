"""
Unit tests for Stanford PLY serialization and roundtrip deserialization.
"""

from src.scene.ply import write_ply_ascii, read_ply_ascii
from src.scene.gaussian import Gaussian3D
from src.scene.gaussian_cloud import GaussianCloud
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from std.collections import List
from std.testing import assert_true, assert_almost_equal

def test_ply_roundtrip() raises:
    var cloud_out = GaussianCloud()

    var dummy_sh = List[Float32](capacity=48)
    dummy_sh.append(0.5)
    dummy_sh.append(0.2)
    dummy_sh.append(0.8)
    for _ in range(45):
        dummy_sh.append(0.0)

    var g1 = Gaussian3D(Vec3f(1.0, 2.0, 3.0), Vec3f(0.1, 0.2, 0.3), Quatf.identity(), 0.95)
    var g2 = Gaussian3D(Vec3f(-4.0, 5.0, -6.0), Vec3f(0.4, 0.5, 0.6), Quatf(0.0, 1.0, 0.0, 0.0), 0.5)

    cloud_out.add_gaussian(g1, dummy_sh)
    cloud_out.add_gaussian(g2, dummy_sh)

    var test_path: String = "/private/tmp/lumen_roundtrip.ply"
    var status = write_ply_ascii(cloud_out, test_path)
    assert_true(status.is_ok(), "Writing PLY should succeed")

    var cloud_in = read_ply_ascii(test_path)
    assert_true(cloud_in.count() == 2, "Read cloud must contain exactly 2 splats")

    var g1_in = cloud_in.get_gaussian(0)
    assert_almost_equal(g1_in.position.x, Float32(1.0), atol=1e-4)
    assert_almost_equal(g1_in.position.y, Float32(2.0), atol=1e-4)
    assert_almost_equal(g1_in.position.z, Float32(3.0), atol=1e-4)
    assert_almost_equal(g1_in.opacity, Float32(0.95), atol=1e-4)

    var g2_in = cloud_in.get_gaussian(1)
    assert_almost_equal(g2_in.position.x, Float32(-4.0), atol=1e-4)
    assert_almost_equal(g2_in.position.y, Float32(5.0), atol=1e-4)
    assert_almost_equal(g2_in.position.z, Float32(-6.0), atol=1e-4)

def main() raises:
    print("Testing PLY ASCII Roundtrip...")
    test_ply_roundtrip()
    print("All PLY tests passed successfully!")

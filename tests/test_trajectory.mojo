"""
Unit tests for Catmull-Rom spline camera trajectories and orbit path generators.
"""

from src.scene.trajectory import CameraTrajectory, catmull_rom_spline
from src.core.config import CameraConfig
from src.math.vector import Vec3f
from std.testing import assert_true, assert_almost_equal

def test_catmull_rom_endpoints() raises:
    var p0 = Vec3f(0.0, 0.0, 0.0)
    var p1 = Vec3f(1.0, 2.0, 3.0)
    var p2 = Vec3f(4.0, 5.0, 6.0)
    var p3 = Vec3f(7.0, 8.0, 9.0)

    # At t = 0.0, spline must equal p1
    var at_zero = catmull_rom_spline(p0, p1, p2, p3, 0.0)
    assert_almost_equal(at_zero.x, Float32(1.0), atol=1e-5)
    assert_almost_equal(at_zero.y, Float32(2.0), atol=1e-5)
    assert_almost_equal(at_zero.z, Float32(3.0), atol=1e-5)

    # At t = 1.0, spline must equal p2
    var at_one = catmull_rom_spline(p0, p1, p2, p3, 1.0)
    assert_almost_equal(at_one.x, Float32(4.0), atol=1e-5)
    assert_almost_equal(at_one.y, Float32(5.0), atol=1e-5)
    assert_almost_equal(at_one.z, Float32(6.0), atol=1e-5)

def test_orbit_trajectory_generation() raises:
    var config = CameraConfig()
    var center = Vec3f(0.0, 0.0, 0.0)
    var radius: Float32 = 5.0
    var height: Float32 = 2.0
    var traj = CameraTrajectory.generate_orbit(config, center, radius, height, 36)

    assert_true(traj.keyframe_count() == 36, "Must contain 36 keyframes")

    # Sample at t = 0.0 -> angle = 0 -> eye at (5, 2, 0)
    var cam_start = traj.sample(0.0)
    assert_almost_equal(cam_start.position.x, Float32(5.0), atol=1e-4)
    assert_almost_equal(cam_start.position.y, Float32(2.0), atol=1e-4)
    assert_almost_equal(cam_start.position.z, Float32(0.0), atol=1e-4)

    # Sample at midpoint t = 0.5 -> angle = PI -> eye at (-5, 2, 0) within chord tolerance
    var cam_half = traj.sample(0.5)
    assert_almost_equal(cam_half.position.x, Float32(-5.0), atol=0.05)
    assert_almost_equal(cam_half.position.y, Float32(2.0), atol=0.05)

def main() raises:
    print("Testing Catmull-Rom Spline Endpoints...")
    test_catmull_rom_endpoints()
    print("Testing Orbit Trajectory Generation...")
    test_orbit_trajectory_generation()
    print("All trajectory tests passed successfully!")

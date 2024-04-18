"""
Unit tests for perspective camera projection, world-to-screen mapping, and ray generation.
"""

from src.scene.camera import PinholeCamera, Ray3f
from src.core.config import CameraConfig
from src.math.vector import Vec2f, Vec3f
from src.math.quaternion import Quatf
from std.testing import assert_true, assert_almost_equal

def test_camera_projection_center() raises:
    # Camera at (0, 0, 10), identity orientation (facing +Z)
    var config = CameraConfig(
        width=1280,
        height=720,
        focal_x=800.0,
        focal_y=800.0,
        principal_x=640.0,
        principal_y=360.0
    )
    var cam = PinholeCamera(config, Vec3f(0.0, 0.0, 0.0), Quatf.identity())

    # Point at (0, 0, 10) directly along optical axis
    var p_world = Vec3f(0.0, 0.0, 10.0)
    var p_screen = cam.world_to_screen(p_world)

    assert_almost_equal(p_screen.x, Float32(640.0), atol=1e-4)
    assert_almost_equal(p_screen.y, Float32(360.0), atol=1e-4)

    # Point at (2, 0, 10) -> u = 800 * (2 / 10) + 640 = 160 + 640 = 800
    var p_right = Vec3f(2.0, 0.0, 10.0)
    var p_right_screen = cam.world_to_screen(p_right)
    assert_almost_equal(p_right_screen.x, Float32(800.0), atol=1e-4)
    assert_almost_equal(p_right_screen.y, Float32(360.0), atol=1e-4)

def test_camera_ray_generation() raises:
    var config = CameraConfig(
        width=1280,
        height=720,
        focal_x=800.0,
        focal_y=800.0,
        principal_x=640.0,
        principal_y=360.0
    )
    var cam = PinholeCamera(config, Vec3f(0.0, 0.0, 0.0), Quatf.identity())

    # Ray through center pixel (640, 360) must point along +Z (0, 0, 1)
    var ray_center = cam.generate_ray(640.0, 360.0)
    assert_almost_equal(ray_center.origin.x, Float32(0.0), atol=1e-4)
    assert_almost_equal(ray_center.origin.y, Float32(0.0), atol=1e-4)
    assert_almost_equal(ray_center.origin.z, Float32(0.0), atol=1e-4)

    assert_almost_equal(ray_center.direction.x, Float32(0.0), atol=1e-4)
    assert_almost_equal(ray_center.direction.y, Float32(0.0), atol=1e-4)
    assert_almost_equal(ray_center.direction.z, Float32(1.0), atol=1e-4)

def main() raises:
    print("Testing Camera Screen Projection...")
    test_camera_projection_center()
    print("Testing Camera Ray Generation...")
    test_camera_ray_generation()
    print("All camera tests passed successfully!")

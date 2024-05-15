"""
Unit tests for frustum culling, near-far clipping, and screen boundary validation.
"""

from src.raster.culling import FrustumCuller
from src.scene.camera import PinholeCamera
from src.core.config import CameraConfig
from std.testing import assert_true

def test_frustum_depth_culling() raises:
    var config = CameraConfig(near_plane=0.1, far_plane=100.0)
    var cam = PinholeCamera(config)
    var culler = FrustumCuller(cam)

    # In front of camera within bounds
    assert_true(culler.is_depth_visible(5.0), "Z = 5.0 must be visible")
    assert_true(culler.is_depth_visible(0.1), "Z = near plane must be visible")
    assert_true(culler.is_depth_visible(100.0), "Z = far plane must be visible")

    # Behind camera or too close
    assert_true(not culler.is_depth_visible(-1.0), "Z = -1.0 behind camera must be culled")
    assert_true(not culler.is_depth_visible(0.0), "Z = 0.0 must be culled")
    assert_true(not culler.is_depth_visible(0.05), "Z = 0.05 inside near plane must be culled")
    assert_true(not culler.is_depth_visible(150.0), "Z = 150.0 beyond far plane must be culled")

def test_frustum_screen_culling() raises:
    var config = CameraConfig(width=800, height=600)
    var cam = PinholeCamera(config)
    var culler = FrustumCuller(cam)

    # Completely inside screen
    assert_true(culler.is_screen_box_visible(400.0, 300.0, 20), "Center splat must be visible")

    # Straddling screen border (partially visible)
    assert_true(culler.is_screen_box_visible(-10.0, 300.0, 20), "Border-straddling splat must be visible")
    assert_true(culler.is_screen_box_visible(810.0, 300.0, 20), "Right-straddling splat must be visible")

    # Completely off-screen
    assert_true(not culler.is_screen_box_visible(-50.0, 300.0, 20), "Left off-screen splat must be culled")
    assert_true(not culler.is_screen_box_visible(900.0, 300.0, 20), "Right off-screen splat must be culled")
    assert_true(not culler.is_screen_box_visible(400.0, -50.0, 20), "Top off-screen splat must be culled")
    assert_true(not culler.is_screen_box_visible(400.0, 700.0, 20), "Bottom off-screen splat must be culled")

def main() raises:
    print("Testing Frustum Depth Culling...")
    test_frustum_depth_culling()
    print("Testing Frustum Screen Boundary Culling...")
    test_frustum_screen_culling()
    print("All culling tests passed successfully!")

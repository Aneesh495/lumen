"""
Integration tests for the complete software rasterization pipeline and pixel synthesis.
"""

from src.raster.pipeline import rasterize_scene, RenderOutput
from src.core.config import RenderConfig, CameraConfig
from src.scene.gaussian import Gaussian3D
from src.scene.gaussian_cloud import GaussianCloud
from src.scene.camera import PinholeCamera
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from std.collections import List
from std.testing import assert_true, assert_almost_equal

def test_single_splat_render() raises:
    # 64x64 test image with focal length 50
    var cam_cfg = CameraConfig(
        width=64,
        height=64,
        focal_x=50.0,
        focal_y=50.0,
        principal_x=32.0,
        principal_y=32.0,
        near_plane=0.1,
        far_plane=100.0
    )
    var camera = PinholeCamera(cam_cfg, Vec3f(0.0, 0.0, 0.0), Quatf.identity())
    var render_cfg = RenderConfig(tile_width=16, tile_height=16)

    # Place a vibrant red Gaussian directly at (0, 0, 5.0)
    var cloud = GaussianCloud()
    var g = Gaussian3D(
        Vec3f(0.0, 0.0, 5.0),
        Vec3f(0.5, 0.5, 0.5), # Radius approx 0.5
        Quatf.identity(),
        1.0 # Fully opaque
    )

    # Red color in SH: c0 = (Color - 0.5) / 0.28209479
    # R = 1.0 -> c0 = 0.5 / 0.28209479 approx 1.77245
    # G = 0.0 -> c0 = -0.5 / 0.28209479 approx -1.77245
    # B = 0.0 -> c0 = -1.77245
    var sh = List[Float32](capacity=48)
    sh.append(1.77245)
    sh.append(-1.77245)
    sh.append(-1.77245)
    for _ in range(45):
        sh.append(0.0)

    cloud.add_gaussian(g, sh)

    var output = rasterize_scene(cloud, camera, render_cfg)

    assert_true(output.visible_splats == 1, "Must detect exactly 1 visible splat")
    assert_true(output.total_splat_instances > 0, "Must create splat instances")

    # Center pixel (32, 32) should be bright red
    var center_pixel = output.get_pixel_rgb(32, 32)
    assert_true(center_pixel.x > 0.8, "Center red channel must be > 0.8")
    assert_true(center_pixel.y < 0.2, "Center green channel must be < 0.2")
    assert_true(center_pixel.z < 0.2, "Center blue channel must be < 0.2")

    # Corner pixel (0, 0) should be uncolored background (0, 0, 0)
    var corner_pixel = output.get_pixel_rgb(0, 0)
    assert_almost_equal(corner_pixel.x, Float32(0.0), atol=1e-4)
    assert_almost_equal(corner_pixel.y, Float32(0.0), atol=1e-4)
    assert_almost_equal(corner_pixel.z, Float32(0.0), atol=1e-4)

    # Depth at center pixel should be approximately 5.0
    var center_depth = output.depth_buffer.get(32 * 64 + 32)
    assert_true(center_depth > 4.5 and center_depth < 5.5, "Center depth must be close to 5.0")

def main() raises:
    print("Testing End-to-End Single Splat Rasterization...")
    test_single_splat_render()
    print("All rasterizer tests passed successfully!")

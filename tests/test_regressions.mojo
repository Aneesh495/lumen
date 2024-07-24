"""
Regression test suite documenting and preventing bugs and edge cases:
1. Singular covariance stabilization via low-pass kernel.
2. Near-plane and behind-camera negative depth culling.
3. SLERP singularity when quaternions are collinear (dot product near 1.0).
4. Morton code boundary clamping for 30-bit spatial curves.
5. Tile binning safety for offscreen or non-intersecting tiles.
"""

from src.core.types import FloatType, LOWPASS_FILTER_VARIANCE
from src.core.config import RenderConfig, CameraConfig
from src.math.vector import Vec3f
from src.math.matrix import Mat3x3f
from src.math.quaternion import Quatf
from src.math.covariance import Covariance3D
from src.math.projection import compute_ewa_covariance, Covariance2D
from src.spatial.morton import compute_morton_3d
from src.scene.gaussian import Gaussian3D
from src.scene.gaussian_cloud import GaussianCloud
from src.scene.camera import PinholeCamera
from src.raster.pipeline import rasterize_scene
from src.raster.culling import FrustumCuller
from std.collections import List
from std.testing import assert_true, assert_almost_equal

def test_regression_singular_covariance() raises:
    """
    Bug: A planar or line-like Gaussian with zero scale along one or two axes
    produces a singular covariance matrix with det(Sigma) = 0.
    Fix: Projective 2D covariance adds a low-pass filter variance blur (0.3 px)
    preventing non-invertible matrices and NaN in inverse covariance.
    """
    # Extremely degenerate scale: zero in Y and Z axes
    var scale_degenerate = Vec3f(1.0, 0.0, 0.0)
    var rot = Quatf.identity()
    var cov3d = Covariance3D.from_scale_and_rotation(scale_degenerate, rot)

    # Project to 2D screen covariance at depth 2.0 with focal length 100.0
    var view_pos = Vec3f(0.0, 0.0, 2.0)
    var view_rot = Mat3x3f()
    var cov2d = compute_ewa_covariance(
        cov3d,
        view_pos,
        view_rot,
        Float32(100.0),
        Float32(100.0),
        LOWPASS_FILTER_VARIANCE
    )

    # Verify determinant is strictly positive due to low-pass filter
    var det = cov2d.determinant()
    assert_true(det > 0.0, "Stabilized 2D covariance determinant must be strictly positive")

    # Invert covariance: must produce non-NaN finite values
    var inv_cov = cov2d.inverse()
    assert_true(inv_cov.c00 > 0.0, "Inverse c00 must be positive")
    assert_true(inv_cov.c11 > 0.0, "Inverse c11 must be positive")

def test_regression_behind_camera_culling() raises:
    """
    Bug: Splats behind the camera eye (z <= 0) inverted screen coordinates and
    produced ghost images reflected through the camera center.
    Fix: Frustum culling strictly enforces vz >= near_plane before projection.
    """
    var cam_cfg = CameraConfig(
        width=100,
        height=100,
        focal_x=50.0,
        focal_y=50.0,
        principal_x=50.0,
        principal_y=50.0,
        near_plane=0.5,
        far_plane=100.0
    )
    var camera = PinholeCamera(cam_cfg)
    var culler = FrustumCuller(camera)

    # Splat clearly behind the camera at view z = -5.0
    assert_true(not culler.is_depth_visible(-5.0), "Splat with negative depth must be culled")

    # Splat between origin and near plane at z = 0.2 (< near_plane 0.5)
    assert_true(not culler.is_depth_visible(0.2), "Splat closer than near plane must be culled")

    # Splat within frustum at z = 5.0
    assert_true(culler.is_depth_visible(5.0), "Splat in valid frustum volume must be visible")

def test_regression_slerp_collinear() raises:
    """
    Bug: When two orientation quaternions are nearly identical (q1 dot q2 -> 1.0),
    sin(theta) -> 0 causing 1 / sin(theta) division by zero.
    Fix: Quaternion SLERP detects dot > 0.9995 and falls back to normalized linear interpolation.
    """
    var q1 = Quatf(1.0, 0.0, 0.0, 0.0)
    var q2 = Quatf(0.99999, 0.001, 0.0, 0.0).normalized()

    # SLERP at midpoint t = 0.5
    var q_mid = q1.slerp(q2, 0.5)

    assert_almost_equal(q_mid.w, Float32(1.0), atol=1e-3)
    assert_almost_equal(q_mid.norm(), Float32(1.0), atol=1e-4)
    assert_true(not (q_mid.w != q_mid.w), "SLERP result must not be NaN")

def test_regression_morton_boundaries() raises:
    """
    Bug: Normalized coordinates at exact domain boundaries (0.0 or 1.0)
    could exceed the 10-bit integer range [0, 1023] and corrupt adjacent bit fields.
    Fix: Morton encoder clamps inputs to [0.0, 1.0] and maps 1.0 to 1023.
    """
    var code_min = compute_morton_3d(Vec3f(0.0, 0.0, 0.0))
    assert_true(code_min == 0, "Origin Morton code must be 0")

    var code_max = compute_morton_3d(Vec3f(1.0, 1.0, 1.0))
    assert_true(code_max <= 0x3FFFFFFF, "Maximum Morton code must fit in 30 bits")

def test_regression_empty_and_extreme_rasterize() raises:
    """
    Bug: Rasterizing an empty scene or a scene where all splats are outside
    view bounds could cause empty buffer dereferences or tile indexing errors.
    Fix: Pipeline checks for zero valid splats and gracefully synthesizes clean background.
    """
    var cam_cfg = CameraConfig(
        width=32,
        height=32,
        focal_x=20.0,
        focal_y=20.0,
        principal_x=16.0,
        principal_y=16.0,
        near_plane=0.1,
        far_plane=50.0
    )
    var camera = PinholeCamera(cam_cfg, Vec3f(0.0, 0.0, 0.0), Quatf.identity())
    var render_cfg = RenderConfig(tile_width=16, tile_height=16)

    # Empty cloud
    var empty_cloud = GaussianCloud()
    var out_empty = rasterize_scene(empty_cloud, camera, render_cfg)
    assert_true(out_empty.visible_splats == 0, "Visible splat count for empty cloud must be zero")

    # Cloud with extreme offscreen coordinates: (100000, 100000, 5)
    var extreme_cloud = GaussianCloud()
    var dummy_sh = List[Float32](capacity=48)
    for _ in range(48): dummy_sh.append(0.0)
    extreme_cloud.add_gaussian(Gaussian3D(Vec3f(100000.0, 100000.0, 5.0), Vec3f(0.1, 0.1, 0.1), Quatf.identity(), 1.0), dummy_sh)

    var out_extreme = rasterize_scene(extreme_cloud, camera, render_cfg)
    assert_true(out_extreme.visible_splats == 0, "Extreme offscreen splat must be culled")

def main() raises:
    print("============================================================")
    print(" Lumen Bug Regression & Boundary Verification Suite")
    print("============================================================")

    test_regression_singular_covariance()
    print("[PASS] test_regression_singular_covariance")

    test_regression_behind_camera_culling()
    print("[PASS] test_regression_behind_camera_culling")

    test_regression_slerp_collinear()
    print("[PASS] test_regression_slerp_collinear")

    test_regression_morton_boundaries()
    print("[PASS] test_regression_morton_boundaries")

    test_regression_empty_and_extreme_rasterize()
    print("[PASS] test_regression_empty_and_extreme_rasterize")

    print("============================================================")
    print(" All regression tests passed with zero defects!")
    print("============================================================")

"""
Unit tests for synthetic scene generation, coordinate normalization, and dataset models.
"""

from src.scene.datasets import create_synthetic_cube_scene, DatasetCamera, DatasetScene
from src.scene.camera import PinholeCamera
from src.core.config import CameraConfig
from src.math.vector import Vec3f
from std.testing import assert_true, assert_almost_equal

def test_synthetic_cube_generation() raises:
    # 4x4x4 cube with spacing 0.5 -> 64 Gaussians
    var cloud = create_synthetic_cube_scene(4, Float32(0.5))
    assert_true(cloud.count() == 64, "Synthetic cube must generate exactly 64 Gaussians")

    # Center of mass must be at origin (0, 0, 0) due to symmetry
    var com = cloud.get_center_of_mass()
    assert_almost_equal(com.x, Float32(0.0), atol=1e-5)
    assert_almost_equal(com.y, Float32(0.0), atol=1e-5)
    assert_almost_equal(com.z, Float32(0.0), atol=1e-5)

    # Half width = (4 - 1) * 0.5 * 0.5 = 0.75
    assert_almost_equal(cloud.bbox_max.x, Float32(0.75), atol=1e-5)
    assert_almost_equal(cloud.bbox_min.x, Float32(-0.75), atol=1e-5)

def test_dataset_scene_cameras() raises:
    var scene = DatasetScene()
    var config = CameraConfig()
    var cam = PinholeCamera(config)

    scene.add_camera(DatasetCamera("frame_001.png", cam))
    scene.add_camera(DatasetCamera("frame_002.png", cam))

    assert_true(scene.camera_count() == 2, "Scene must register 2 cameras")

def main() raises:
    print("Testing Synthetic Cube Generation...")
    test_synthetic_cube_generation()
    print("Testing Dataset Scene Cameras...")
    test_dataset_scene_cameras()
    print("All dataset tests passed successfully!")

"""
Dataset parsers for COLMAP Structure-from-Motion and synthetic NeRF Blender formats.
"""

from .gaussian import Gaussian3D
from .gaussian_cloud import GaussianCloud
from .camera import PinholeCamera
from src.core.config import CameraConfig
from src.core.types import MAX_SH_COEFFS_PER_GAUSSIAN, parse_float
from src.math.vector import Vec3f
from src.math.matrix import Mat4x4f
from src.math.quaternion import Quatf
from std.collections import List

struct DatasetFormat:
    comptime COLMAP: Int = 0
    comptime NERF_SYNTHETIC: Int = 1


struct DatasetCamera(ImplicitlyCopyable):
    var image_name: String
    var camera: PinholeCamera

    def __init__(out self, image_name: String, camera: PinholeCamera):
        self.image_name = image_name
        self.camera = camera


struct DatasetScene:
    var cameras: List[DatasetCamera]
    var point_cloud: GaussianCloud
    var scene_radius: Float32

    def __init__(out self):
        self.cameras = List[DatasetCamera]()
        self.point_cloud = GaussianCloud()
        self.scene_radius = 1.0

    def add_camera(mut self, cam: DatasetCamera):
        self.cameras.append(cam)

    def camera_count(self) -> Int:
        return len(self.cameras)


def create_synthetic_cube_scene(splats_per_axis: Int = 5, spacing: Float32 = 0.5) -> GaussianCloud:
    """
    Generate synthetic 3D volumetric lattice for testing and calibration.
    """
    var cloud = GaussianCloud(splats_per_axis * splats_per_axis * splats_per_axis)
    var half = Float32(splats_per_axis - 1) * spacing * 0.5

    var dummy_sh = List[Float32](capacity=48)
    for _ in range(48):
        dummy_sh.append(0.0)

    for i in range(splats_per_axis):
        var x = Float32(i) * spacing - half
        for j in range(splats_per_axis):
            var y = Float32(j) * spacing - half
            for k in range(splats_per_axis):
                var z = Float32(k) * spacing - half

                var pos = Vec3f(x, y, z)
                var scale = Vec3f(spacing * 0.3, spacing * 0.3, spacing * 0.3)
                var rot = Quatf.identity()
                var opacity: Float32 = 0.8

                # Color variation based on spatial position
                var sh = List[Float32](capacity=48)
                sh.append((x + half) / (2.0 * half)) # R
                sh.append((y + half) / (2.0 * half)) # G
                sh.append((z + half) / (2.0 * half)) # B
                for _ in range(45):
                    sh.append(0.0)

                cloud.add_gaussian(Gaussian3D(pos, scale, rot, opacity), sh)

    return cloud^

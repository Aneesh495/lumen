"""
Scene representations, camera trajectories, PLY point clouds, and dataset ingestion.
"""

from .gaussian import (
    Gaussian3D,
    GaussianSoA,
)

from .gaussian_cloud import (
    GaussianCloud,
)

from .camera import (
    Ray3f,
    PinholeCamera,
)

from .trajectory import (
    CameraKeyframe,
    CameraTrajectory,
    catmull_rom_spline,
)

from .ply import (
    PLYHeader,
    write_ply_ascii,
    read_ply_ascii,
)

from .datasets import (
    DatasetFormat,
    DatasetCamera,
    DatasetScene,
    create_synthetic_cube_scene,
)

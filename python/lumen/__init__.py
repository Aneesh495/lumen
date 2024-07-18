"""
Lumen Python SDK: Real-time 3D Gaussian Splatting and Radiance Field Rasterization.

High-performance Python client and toolchain for Lumen.
"""

from .camera import Camera, CameraTrajectory
from .cloud import GaussianCloud, Gaussian3D
from .client import LumenEngine, RasterizerConfig, RenderResult
from .visualizer import export_ppm, export_depth_visualization, ascii_render

__version__ = "0.1.0"
__all__ = [
    "Camera",
    "CameraTrajectory",
    "GaussianCloud",
    "Gaussian3D",
    "LumenEngine",
    "RasterizerConfig",
    "RenderResult",
    "export_ppm",
    "export_depth_visualization",
    "ascii_render",
]

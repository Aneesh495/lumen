# bench_rasterizer.mojo - Benchmark end-to-end Gaussian rasterizer throughput

from std.time import perf_counter_ns
from std.collections import List
from src.core.config import RenderConfig, CameraConfig
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from src.scene.gaussian import Gaussian3D
from src.scene.gaussian_cloud import GaussianCloud
from src.scene.camera import PinholeCamera
from src.scene.datasets import create_synthetic_cube_scene
from src.raster.pipeline import rasterize_scene

def benchmark_density(name: String, points_per_axis: Int, iterations: Int) raises:
    var cam_cfg = CameraConfig(
        width=160,
        height=120,
        focal_x=120.0,
        focal_y=120.0,
        principal_x=80.0,
        principal_y=60.0,
        near_plane=0.1,
        far_plane=100.0
    )
    var camera = PinholeCamera(cam_cfg, Vec3f(0.0, 0.0, 4.0), Quatf.identity())
    var render_cfg = RenderConfig(tile_width=16, tile_height=16)
    var cloud = create_synthetic_cube_scene(points_per_axis, 0.5)
    var splats = cloud.count()

    # Warmup
    var _ = rasterize_scene(cloud, camera, render_cfg)

    var start_ns = perf_counter_ns()
    for _ in range(iterations):
        var _ = rasterize_scene(cloud, camera, render_cfg)
    var end_ns = perf_counter_ns()

    var total_elapsed_ms = Float64(end_ns - start_ns) / 1000000.0
    var avg_elapsed_ms = total_elapsed_ms / Float64(iterations)
    var fps = 1000.0 / avg_elapsed_ms
    var throughput_msplats = (Float64(splats) * fps) / 1000000.0

    print("  Benchmark [" + name + "] (" + String(splats) + " splats, " + String(iterations) + " iters):")
    print("    Mean latency : " + String(avg_elapsed_ms) + " ms")
    print("    Throughput   : " + String(fps) + " FPS (" + String(throughput_msplats) + " M-splats/sec)")

def main() raises:
    print("============================================================")
    print(" Lumen Rasterizer Engine Throughput Benchmark")
    print("============================================================")

    benchmark_density("Cube 4x4x4", 4, 30)
    benchmark_density("Cube 6x6x6", 6, 20)
    benchmark_density("Cube 8x8x8", 8, 10)

    print("============================================================")
    print(" Benchmark completed successfully.")
    print("============================================================")

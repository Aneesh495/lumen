"""
Lumen: High-Performance 3D Gaussian Splatting and Radiance Field Engine.
Command-Line Interface and Primary Entrypoint.
"""

import std.sys
from src.core.types import (
    TILE_WIDTH,
    TILE_HEIGHT,
    parse_float,
)
from src.core.config import CameraConfig, RenderConfig, TrainingConfig
from src.scene.gaussian import Gaussian3D
from src.scene.gaussian_cloud import GaussianCloud
from src.scene.camera import PinholeCamera
from src.scene.trajectory import CameraTrajectory
from src.scene.ply import write_ply_ascii, read_ply_ascii
from src.scene.datasets import create_synthetic_cube_scene
from src.raster.pipeline import rasterize_scene
from src.optim.trainer import TrainingCoordinator
from src.optim.optimizer import AdamOptimizer
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from std.collections import List
from std.time import perf_counter_ns

def print_banner():
    print("================================================================================")
    print("  LUMEN: High-Performance 3D Gaussian Splatting & Radiance Field Engine")
    print("================================================================================")

def print_help():
    print_banner()
    print("Usage: lumen <subcommand> [options]\n")
    print("Subcommands:")
    print("  render    Execute tile-based SIMD rasterization on a 3D Gaussian cloud")
    print("  train     Run continuous radiance field optimization with density control")
    print("  inspect   Display spatial bounding box, center of mass, and splat metrics")
    print("  bench     Execute comprehensive microbenchmark suite for SIMD rasterizer")
    print("  demo      Generate synthetic volumetric scene and render novel views")
    print("\nOptions:")
    print("  --scene <path>     Path to Stanford PLY scene file")
    print("  --out <path>       Output rendered image or checkpoint file")
    print("  --width <int>      Render viewport width (default: 800)")
    print("  --height <int>     Render viewport height (default: 600)")
    print("  --iters <int>      Training optimization iterations (default: 1000)")
    print("  --help, -h         Display this reference documentation")

def write_ppm_image(filepath: String, width: Int, height: Int, rgb_data: List[Float32]):
    try:
        with open(filepath, "w") as f:
            f.write("P3\n")
            f.write(String(width) + " " + String(height) + "\n")
            f.write("255\n")
            for y in range(height):
                var line = String("")
                for x in range(width):
                    var idx = (y * width + x) * 3
                    var r = Int(rgb_data[idx + 0] * 255.0)
                    var g = Int(rgb_data[idx + 1] * 255.0)
                    var b = Int(rgb_data[idx + 2] * 255.0)
                    if r < 0: r = 0
                    elif r > 255: r = 255
                    if g < 0: g = 0
                    elif g > 255: g = 255
                    if b < 0: b = 0
                    elif b > 255: b = 255
                    line += String(r) + " " + String(g) + " " + String(b) + " "
                line += "\n"
                f.write(line)
        print("[✓] Saved rendered image to:", filepath)
    except e:
        print("[!] Warning: Could not write PPM image:", e)

def cmd_inspect(scene_path: String):
    print_banner()
    print("[*] Inspecting Gaussian scene:", scene_path)
    var cloud = read_ply_ascii(scene_path)
    var count = cloud.count()
    print("  Total Gaussian splats: ", count)

    if count > 0:
        var com = cloud.get_center_of_mass()
        print("  Center of Mass:         (" + String(com.x) + ", " + String(com.y) + ", " + String(com.z) + ")")
        print("  Bounding Box Min:       (" + String(cloud.bbox_min.x) + ", " + String(cloud.bbox_min.y) + ", " + String(cloud.bbox_min.z) + ")")
        print("  Bounding Box Max:       (" + String(cloud.bbox_max.x) + ", " + String(cloud.bbox_max.y) + ", " + String(cloud.bbox_max.z) + ")")
        print("  Spatial Extent Radius:  ", cloud.get_extent_radius())
    else:
        print("[!] Warning: Scene contains zero Gaussians")

def cmd_demo(width: Int, height: Int, out_path: String):
    print_banner()
    print("[*] Generating synthetic volumetric scene (cube lattice)...")
    var cloud = create_synthetic_cube_scene(6, Float32(0.4))
    print("  Generated splats:       ", cloud.count())

    var cam_cfg = CameraConfig(width=width, height=height, focal_x=Float32(width), focal_y=Float32(width))
    var eye = Vec3f(2.5, 2.0, 3.0)
    var target = Vec3f(0.0, 0.0, 0.0)
    var up = Vec3f(0.0, 1.0, 0.0)
    var cam = PinholeCamera.look_at(cam_cfg, eye, target, up)
    var render_cfg = RenderConfig()

    print("[*] Rendering novel viewpoint via tile-based SIMD rasterizer...")
    var t0 = perf_counter_ns()
    var out_render = rasterize_scene(cloud, cam, render_cfg)
    var t1 = perf_counter_ns()
    var elapsed_ms = Float64(t1 - t0) / 1000000.0

    print("  Visible splats:         ", out_render.visible_splats)
    print("  Splat-tile instances:   ", out_render.total_splat_instances)
    print("  Rasterization latency:  ", elapsed_ms, "ms")

    write_ppm_image(out_path, width, height, out_render.rgb_buffer.data)

def cmd_bench():
    print_banner()
    print("[*] Running Lumen SIMD Rasterizer Performance Benchmark...")
    var cloud = create_synthetic_cube_scene(8, Float32(0.3)) # 512 splats
    var width = 800
    var height = 600
    var cam_cfg = CameraConfig(width=width, height=height, focal_x=800.0, focal_y=800.0)
    var cam = PinholeCamera.look_at(cam_cfg, Vec3f(3.0, 2.0, 4.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0))
    var render_cfg = RenderConfig()

    # Warmup
    _ = rasterize_scene(cloud, cam, render_cfg)

    var trials = 5
    var total_time_ms: Float64 = 0.0

    for _ in range(trials):
        var t0 = perf_counter_ns()
        _ = rasterize_scene(cloud, cam, render_cfg)
        var t1 = perf_counter_ns()
        total_time_ms += Float64(t1 - t0) / 1000000.0

    var avg_ms = total_time_ms / Float64(trials)
    var fps = 1000.0 / avg_ms
    print("  Resolution:             ", width, "x", height)
    print("  Scene Gaussian count:   ", cloud.count())
    print("  Average frame latency:  ", avg_ms, "ms")
    print("  Throughput:             ", fps, "FPS")
    print("[✓] Benchmark completed successfully!")

def main():
    var args = std.sys.argv()
    if len(args) <= 1:
        print_help()
        return

    var cmd = args[1]
    if cmd == "--help" or cmd == "-h" or cmd == "help":
        print_help()
        return
    elif cmd == "bench":
        cmd_bench()
    elif cmd == "demo":
        var out_path = String("/private/tmp/lumen_demo.ppm")
        cmd_demo(320, 240, out_path)
    elif cmd == "inspect":
        if len(args) >= 3:
            cmd_inspect(args[2])
        else:
            print("Error: Missing --scene <path> argument")
    else:
        print("Unknown subcommand:", cmd)
        print_help()

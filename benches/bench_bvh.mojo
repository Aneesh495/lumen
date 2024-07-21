# bench_bvh.mojo - Benchmark Linear BVH construction and hierarchical raycasting

from std.time import perf_counter_ns
from src.spatial.bvh import LBVH
from src.spatial.raycast import raycast_closest_splat
from src.scene.camera import Ray3f
from src.scene.datasets import create_synthetic_cube_scene
from src.math.vector import Vec3f

def benchmark_lbvh_build(splats_per_axis: Int, iterations: Int) raises:
    var cloud = create_synthetic_cube_scene(splats_per_axis, Float32(0.5))
    var splats = cloud.count()

    # Warmup
    var _ = LBVH.build_from_cloud(cloud)

    var start_ns = perf_counter_ns()
    for _ in range(iterations):
        var _ = LBVH.build_from_cloud(cloud)
    var end_ns = perf_counter_ns()

    var total_elapsed_ms = Float64(end_ns - start_ns) / 1000000.0
    var avg_elapsed_ms = total_elapsed_ms / Float64(iterations)
    var builds_per_sec = 1000.0 / avg_elapsed_ms

    print("  Benchmark [LBVH Build] (" + String(splats) + " splats, " + String(iterations) + " iters):")
    print("    Mean latency : " + String(avg_elapsed_ms) + " ms")
    print("    Throughput   : " + String(builds_per_sec) + " builds/sec")

def benchmark_raycast_queries(splats_per_axis: Int, num_rays: Int, iterations: Int) raises:
    var cloud = create_synthetic_cube_scene(splats_per_axis, Float32(0.5))
    var bvh = LBVH.build_from_cloud(cloud)

    var rays = List[Ray3f]()
    for i in range(num_rays):
        var rx = Float32(i % 10) * 0.1 - 0.5
        var ry = Float32((i * 3) % 10) * 0.1 - 0.5
        rays.append(Ray3f(Vec3f(rx, ry, -5.0), Vec3f(0.0, 0.0, 1.0)))

    # Warmup
    for i in range(10):
        var _ = raycast_closest_splat(bvh, cloud, rays[i])

    var start_ns = perf_counter_ns()
    for _ in range(iterations):
        for i in range(num_rays):
            var _ = raycast_closest_splat(bvh, cloud, rays[i])
    var end_ns = perf_counter_ns()

    var total_elapsed_ms = Float64(end_ns - start_ns) / 1000000.0
    var avg_elapsed_ms = total_elapsed_ms / Float64(iterations)
    var queries_per_sec = (Float64(num_rays) * 1000.0) / avg_elapsed_ms

    print("  Benchmark [Raycast Traversal] (" + String(num_rays) + " rays, " + String(iterations) + " iters):")
    print("    Mean latency : " + String(avg_elapsed_ms) + " ms")
    print("    Throughput   : " + String(queries_per_sec) + " rays/sec")

def main() raises:
    print("============================================================")
    print(" Lumen Linear BVH Construction & Raycast Benchmark")
    print("============================================================")

    benchmark_lbvh_build(6, 20)
    benchmark_lbvh_build(8, 10)
    benchmark_raycast_queries(6, 500, 20)

    print("============================================================")
    print(" Benchmark completed successfully.")
    print("============================================================")

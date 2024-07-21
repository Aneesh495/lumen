# bench_spherical_harmonics.mojo - Benchmark Spherical Harmonics basis polynomial evaluations

from std.time import perf_counter_ns
from src.math.vector import Vec3f
from src.math.spherical_harmonics import SHBasis

def benchmark_sh_evaluations(eval_count: Int, iterations: Int) raises:
    # Prepare normalized view vectors
    var dirs = List[Vec3f]()
    for i in range(eval_count):
        var fx = Float32(i % 100) - 50.0
        var fy = Float32((i * 3) % 100) - 50.0
        var fz = Float32((i * 7) % 100) + 1.0
        dirs.append(Vec3f(fx, fy, fz).normalized())

    # Warmup
    for i in range(100):
        var _ = SHBasis.evaluate(dirs[i])

    var total_ns: Int = 0

    for _ in range(iterations):
        var t0 = perf_counter_ns()
        for i in range(eval_count):
            var _ = SHBasis.evaluate(dirs[i])
        var t1 = perf_counter_ns()
        total_ns += (t1 - t0)

    var avg_elapsed_ms = (Float64(total_ns) / 1000000.0) / Float64(iterations)
    var evals_per_sec = (Float64(eval_count) * 1000.0) / avg_elapsed_ms
    var million_evals_sec = evals_per_sec / 1000000.0

    print("  Benchmark [SH Degree 3 Basis] (" + String(eval_count) + " dirs, " + String(iterations) + " iters):")
    print("    Mean latency : " + String(avg_elapsed_ms) + " ms")
    print("    Throughput   : " + String(million_evals_sec) + " M-evals/sec")

def main() raises:
    print("============================================================")
    print(" Lumen Spherical Harmonics Basis Polynomial Benchmark")
    print("============================================================")

    benchmark_sh_evaluations(10000, 30)
    benchmark_sh_evaluations(50000, 10)

    print("============================================================")
    print(" Benchmark completed successfully.")
    print("============================================================")

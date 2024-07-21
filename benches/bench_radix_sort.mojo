# bench_radix_sort.mojo - Benchmark 64-bit LSD Radix Sort performance

from std.time import perf_counter_ns
from src.core.buffer import UInt64Buffer, IntBuffer
from src.raster.radix_sort import encode_tile_depth_key, radix_sort_64bit

def benchmark_sort_size(name: String, count: Int, iterations: Int) raises:
    # Initialize pseudo-random keys
    var base_keys = UInt64Buffer(count, 0)
    var base_vals = IntBuffer(count, 0)

    for i in range(count):
        var pseudo_tile = UInt32((i * 7919 + 31) % 500)
        var pseudo_depth = Float32((i * 104729 + 17) % 10000) * 0.01
        base_keys.set(i, encode_tile_depth_key(pseudo_tile, pseudo_depth))
        base_vals.set(i, i)

    # Warmup
    var warmup_keys = UInt64Buffer(count, 0)
    var warmup_vals = IntBuffer(count, 0)
    for i in range(count):
        warmup_keys.set(i, base_keys.get(i))
        warmup_vals.set(i, base_vals.get(i))
    radix_sort_64bit(warmup_keys, warmup_vals)

    var total_ns: Int = 0

    for _ in range(iterations):
        var keys = UInt64Buffer(count, 0)
        var vals = IntBuffer(count, 0)
        for i in range(count):
            keys.set(i, base_keys.get(i))
            vals.set(i, base_vals.get(i))

        var t0 = perf_counter_ns()
        radix_sort_64bit(keys, vals)
        var t1 = perf_counter_ns()
        total_ns += (t1 - t0)

    var avg_elapsed_ms = (Float64(total_ns) / 1000000.0) / Float64(iterations)
    var keys_per_sec = (Float64(count) * 1000.0) / avg_elapsed_ms
    var million_keys_sec = keys_per_sec / 1000000.0

    print("  Benchmark [" + name + "] (" + String(count) + " keys, " + String(iterations) + " iters):")
    print("    Mean latency : " + String(avg_elapsed_ms) + " ms")
    print("    Throughput   : " + String(million_keys_sec) + " M-keys/sec")

def main() raises:
    print("============================================================")
    print(" Lumen 64-bit LSD Radix Sort Benchmark")
    print("============================================================")

    benchmark_sort_size("Batch 1,000", 1000, 50)
    benchmark_sort_size("Batch 10,000", 10000, 20)
    benchmark_sort_size("Batch 50,000", 50000, 10)

    print("============================================================")
    print(" Benchmark completed successfully.")
    print("============================================================")

"""
High-throughput 64-bit LSD radix sort for composite tile identifier and depth key ordering.
"""

from src.core.buffer import UInt64Buffer, IntBuffer
from std.collections import List

def float_to_ordered_uint32(f: Float32) -> UInt64:
    """
    Convert positive Float32 depth into monotonic 32-bit integer bits.
    Since camera depth z > 0, standard integer representation preserves order.
    """
    var scaled = Int64(f * 1000.0) # Millimeter precision
    if scaled < 0:
        scaled = 0
    elif scaled > 0x7FFFFFFF:
        scaled = 0x7FFFFFFF
    return UInt64(scaled)


def encode_tile_depth_key(tile_id: UInt32, depth: Float32) -> UInt64:
    """
    Compose 64-bit key: upper 32 bits = tile_id, lower 32 bits = depth.
    """
    var upper = UInt64(tile_id) << 32
    var lower = float_to_ordered_uint32(depth)
    return upper | lower


def decode_tile_id(key: UInt64) -> UInt32:
    return UInt32((key >> 32) & 0xFFFFFFFF)


def decode_depth_approx(key: UInt64) -> Float32:
    var lower = key & 0xFFFFFFFF
    return Float32(lower) * 0.001


def radix_sort_64bit(mut keys: UInt64Buffer, mut values: IntBuffer):
    """
    In-place 64-bit Least Significant Digit (LSD) radix sort.
    Executes 8 passes of 8 bits (256 buckets) with double buffering.
    """
    var n = keys.size()
    if n <= 1:
        return

    # Temporary ping-pong buffers
    var keys_temp = UInt64Buffer(n, 0)
    var vals_temp = IntBuffer(n, 0)

    # 8 passes of 8 bits = 64 bits total
    for pass_idx in range(8):
        var shift = pass_idx * 8

        # 256 histogram histogram counts
        var counts = List[Int](capacity=256)
        for _ in range(256):
            counts.append(0)

        # 1. Count occurrences of each byte
        for i in range(n):
            var byte_val = Int((keys.get(i) >> shift) & 0xFF)
            counts[byte_val] += 1

        # 2. Compute prefix sums (cumulative offsets)
        var offsets = List[Int](capacity=256)
        var total = 0
        for b in range(256):
            offsets.append(total)
            total += counts[b]

        # 3. Scatter elements into temp buffer based on offsets
        for i in range(n):
            var k = keys.get(i)
            var v = values.get(i)
            var byte_val = Int((k >> shift) & 0xFF)
            var dest_idx = offsets[byte_val]
            offsets[byte_val] += 1

            keys_temp.set(dest_idx, k)
            vals_temp.set(dest_idx, v)

        # 4. Copy back to primary buffers
        for i in range(n):
            keys.set(i, keys_temp.get(i))
            values.set(i, vals_temp.get(i))

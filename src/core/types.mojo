"""
Fundamental numeric types, dimensional constants, and coordinate conventions for Lumen.
"""

from std.simd import SIMD
from std.math import pow

# Global floating-point precision type aliases
comptime FloatType = Float32
comptime FloatVec4 = SIMD[DType.float32, 4]
comptime FloatVec3 = SIMD[DType.float32, 4] # Padded for 128-bit SIMD alignment
comptime FloatVec2 = SIMD[DType.float32, 2]

# Integer scalar and vector types
comptime IntType = Int32
comptime UIntType = UInt32
comptime IndexType = Int64
comptime KeyType = UInt64

# Geometric and rasterization constants
comptime TILE_WIDTH: Int = 16
comptime TILE_HEIGHT: Int = 16
comptime TILE_PIXELS: Int = 256 # 16 * 16 pixels per tile

# Spherical Harmonics constants
comptime SH_DEGREE_0_COEFFS: Int = 1
comptime SH_DEGREE_1_COEFFS: Int = 4
comptime SH_DEGREE_2_COEFFS: Int = 9
comptime SH_DEGREE_3_COEFFS: Int = 16
comptime SH_CHANNELS_RGB: Int = 3
comptime MAX_SH_COEFFS_PER_GAUSSIAN: Int = 48 # 16 * 3

# Alpha blending thresholds
comptime MIN_ALPHA_THRESHOLD: Float32 = 0.00392156862 # 1.0 / 255.0
comptime TRANSMITTANCE_EPSILON: Float32 = 0.0001
comptime LOWPASS_FILTER_VARIANCE: Float32 = 0.3 # EWA screen-space anti-aliasing variance

# Mathematical constants
comptime PI_F32: Float32 = 3.14159265358979323846
comptime TWO_PI_F32: Float32 = 6.28318530717958647692
comptime HALF_PI_F32: Float32 = 1.57079632679489661923
comptime INV_PI_F32: Float32 = 0.31830988618379067154
comptime DEG_TO_RAD_F32: Float32 = 0.017453292519943295
comptime RAD_TO_DEG_F32: Float32 = 57.29577951308232

# Coordinate conventions
struct CoordinateSystem:
    # Right-handed coordinate frame: +X right, +Y up, +Z backwards (OpenGL convention)
    comptime OPENGL: Int = 0
    # Right-handed camera frame: +X right, +Y down, +Z forward (COLMAP / OpenCV convention)
    comptime OPENCV: Int = 1

# Color representation models
struct ColorSpace:
    comptime LINEAR_RGB: Int = 0
    comptime SRGB: Int = 1

# Render output channels
struct RenderChannel:
    comptime RGB: Int = 1
    comptime DEPTH: Int = 2
    comptime ALPHA: Int = 4
    comptime NORMALS: Int = 8


def parse_float(s: String) -> Float32:
    """
    High-performance zero-dependency ASCII string to Float32 parser.
    """
    var sign: Float32 = 1.0
    var val: Float32 = 0.0
    var div: Float32 = 1.0
    var in_decimal = False
    var in_exp = False
    var exp_sign: Float32 = 1.0
    var exp_val: Float32 = 0.0
    var first = True
    var exp_first = False

    for b in s.bytes():
        var c = Int(b)
        if first:
            first = False
            if c == 45: # '-'
                sign = -1.0
                continue
            elif c == 43: # '+'
                continue
        if c == 101 or c == 69: # 'e' or 'E'
            in_exp = True
            exp_first = True
            continue
        if in_exp:
            if exp_first:
                exp_first = False
                if c == 45:
                    exp_sign = -1.0
                    continue
                elif c == 43:
                    continue
            if c >= 48 and c <= 57:
                exp_val = exp_val * 10.0 + Float32(c - 48)
        else:
            if c == 46: # '.'
                in_decimal = True
            elif c >= 48 and c <= 57:
                var digit = Float32(c - 48)
                if in_decimal:
                    div *= 10.0
                    val += digit / div
                else:
                    val = val * 10.0 + digit

    var total = sign * val
    if in_exp:
        total *= pow(Float32(10.0), exp_sign * exp_val)
    return total

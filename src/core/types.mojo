"""
Fundamental numeric types, dimensional constants, and coordinate conventions for Lumen.
"""

from std.simd import SIMD

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

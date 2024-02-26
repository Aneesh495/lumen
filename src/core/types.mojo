"""
Fundamental numeric types, dimensional constants, and coordinate conventions for Lumen.
"""

from std.simd import SIMD

# Global floating-point precision type aliases
alias FloatType = Float32
alias FloatVec4 = SIMD[DType.float32, 4]
alias FloatVec3 = SIMD[DType.float32, 4] # Padded for 128-bit SIMD alignment
alias FloatVec2 = SIMD[DType.float32, 2]

# Integer scalar and vector types
alias IntType = Int32
alias UIntType = UInt32
alias IndexType = Int64
alias KeyType = UInt64

# Geometric and rasterization constants
alias TILE_WIDTH: Int = 16
alias TILE_HEIGHT: Int = 16
alias TILE_PIXELS: Int = TILE_WIDTH * TILE_HEIGHT # 256 pixels per tile

# Spherical Harmonics constants
alias SH_DEGREE_0_COEFFS: Int = 1
alias SH_DEGREE_1_COEFFS: Int = 4
alias SH_DEGREE_2_COEFFS: Int = 9
alias SH_DEGREE_3_COEFFS: Int = 16
alias SH_CHANNELS_RGB: Int = 3
alias MAX_SH_COEFFS_PER_GAUSSIAN: Int = SH_DEGREE_3_COEFFS * SH_CHANNELS_RGB # 48 floats

# Alpha blending thresholds
alias MIN_ALPHA_THRESHOLD: Float32 = 1.0 / 255.0
alias TRANSMITTANCE_EPSILON: Float32 = 0.0001
alias LOWPASS_FILTER_VARIANCE: Float32 = 0.3 # EWA screen-space anti-aliasing variance

# Mathematical constants
alias PI_F32: Float32 = 3.14159265358979323846
alias TWO_PI_F32: Float32 = 6.28318530717958647692
alias HALF_PI_F32: Float32 = 1.57079632679489661923
alias INV_PI_F32: Float32 = 0.31830988618379067154
alias DEG_TO_RAD_F32: Float32 = PI_F32 / 180.0
alias RAD_TO_DEG_F32: Float32 = 180.0 / PI_F32

# Coordinate conventions
struct CoordinateSystem:
    # Right-handed coordinate frame: +X right, +Y up, +Z backwards (OpenGL convention)
    alias OPENGL: Int = 0
    # Right-handed camera frame: +X right, +Y down, +Z forward (COLMAP / OpenCV convention)
    alias OPENCV: Int = 1

# Color representation models
struct ColorSpace:
    alias LINEAR_RGB: Int = 0
    alias SRGB: Int = 1

# Render output channels
struct RenderChannel:
    alias RGB: Int = 1
    alias DEPTH: Int = 2
    alias ALPHA: Int = 4
    alias NORMALS: Int = 8

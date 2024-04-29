"""
Core system definitions, memory management, configuration, and diagnostics for Lumen.
"""

from .types import (
    FloatType,
    FloatVec4,
    FloatVec3,
    FloatVec2,
    IntType,
    UIntType,
    IndexType,
    KeyType,
    TILE_WIDTH,
    TILE_HEIGHT,
    TILE_PIXELS,
    SH_DEGREE_0_COEFFS,
    SH_DEGREE_1_COEFFS,
    SH_DEGREE_2_COEFFS,
    SH_DEGREE_3_COEFFS,
    SH_CHANNELS_RGB,
    MAX_SH_COEFFS_PER_GAUSSIAN,
    MIN_ALPHA_THRESHOLD,
    TRANSMITTANCE_EPSILON,
    LOWPASS_FILTER_VARIANCE,
    PI_F32,
    TWO_PI_F32,
    HALF_PI_F32,
    INV_PI_F32,
    DEG_TO_RAD_F32,
    RAD_TO_DEG_F32,
    CoordinateSystem,
    ColorSpace,
    RenderChannel,
    parse_float,
)

from .config import (
    CameraConfig,
    RenderConfig,
    TrainingConfig,
)

from .error import (
    StatusCode,
    LumenStatus,
    validate_dimension,
    validate_range,
)

from .buffer import (
    FloatBuffer,
    UInt64Buffer,
    IntBuffer,
)

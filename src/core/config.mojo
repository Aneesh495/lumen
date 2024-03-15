"""
Configuration structures for camera intrinsics, rendering passes, and optimization runs.
"""

from .types import (
    FloatType,
    IntType,
    TILE_WIDTH,
    TILE_HEIGHT,
    LOWPASS_FILTER_VARIANCE,
    MIN_ALPHA_THRESHOLD,
    TRANSMITTANCE_EPSILON,
    RenderChannel,
    CoordinateSystem,
)

struct CameraConfig(ImplicitlyCopyable):
    """
    Intrinsic parameters and resolution limits for perspective camera projections.
    """
    var width: Int
    var height: Int
    var focal_x: Float32
    var focal_y: Float32
    var principal_x: Float32
    var principal_y: Float32
    var near_plane: Float32
    var far_plane: Float32
    var convention: Int

    def __init__(
        out self,
        width: Int = 1280,
        height: Int = 720,
        focal_x: Float32 = 800.0,
        focal_y: Float32 = 800.0,
        principal_x: Float32 = 640.0,
        principal_y: Float32 = 360.0,
        near_plane: Float32 = 0.01,
        far_plane: Float32 = 1000.0,
        convention: Int = CoordinateSystem.OPENGL,
    ):
        self.width = width
        self.height = height
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.principal_x = principal_x
        self.principal_y = principal_y
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.convention = convention

    def aspect_ratio(self) -> Float32:
        return Float32(self.width) / Float32(self.height)

    def fov_x(self) -> Float32:
        from std.math import atan
        return 2.0 * atan(Float32(self.width) / (2.0 * self.focal_x))

    def fov_y(self) -> Float32:
        from std.math import atan
        return 2.0 * atan(Float32(self.height) / (2.0 * self.focal_y))


struct RenderConfig(ImplicitlyCopyable):
    """
    Settings governing the tile-based SIMD rasterization pipeline.
    """
    var tile_width: Int
    var tile_height: Int
    var channels: Int
    var lowpass_filter: Float32
    var min_alpha: Float32
    var early_stop_transmittance: Float32
    var max_sh_degree: Int
    var white_background: Bool
    var compute_depth: Bool
    var compute_normals: Bool

    def __init__(
        out self,
        tile_width: Int = TILE_WIDTH,
        tile_height: Int = TILE_HEIGHT,
        channels: Int = RenderChannel.RGB,
        lowpass_filter: Float32 = LOWPASS_FILTER_VARIANCE,
        min_alpha: Float32 = MIN_ALPHA_THRESHOLD,
        early_stop_transmittance: Float32 = TRANSMITTANCE_EPSILON,
        max_sh_degree: Int = 3,
        white_background: Bool = False,
        compute_depth: Bool = True,
        compute_normals: Bool = False,
    ):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.channels = channels
        self.lowpass_filter = lowpass_filter
        self.min_alpha = min_alpha
        self.early_stop_transmittance = early_stop_transmittance
        self.max_sh_degree = max_sh_degree
        self.white_background = white_background
        self.compute_depth = compute_depth
        self.compute_normals = compute_normals

    def tiles_x(self, width: Int) -> Int:
        return (width + self.tile_width - 1) // self.tile_width

    def tiles_y(self, height: Int) -> Int:
        return (height + self.tile_height - 1) // self.tile_height

    def total_tiles(self, width: Int, height: Int) -> Int:
        return self.tiles_x(width) * self.tiles_y(height)


struct TrainingConfig(ImplicitlyCopyable):
    """
    Hyperparameters and scheduling parameters for continuous radiance field optimization.
    """
    var iterations: Int
    var position_lr_init: Float32
    var position_lr_final: Float32
    var feature_lr: Float32
    var opacity_lr: Float32
    var scaling_lr: Float32
    var rotation_lr: Float32
    var lambda_dssim: Float32
    var densify_from_iter: Int
    var densify_until_iter: Int
    var densification_interval: Int
    var opacity_reset_interval: Int
    var densify_grad_threshold: Float32
    var prune_opacity_threshold: Float32
    var max_screen_size_ratio: Float32

    def __init__(
        out self,
        iterations: Int = 30000,
        position_lr_init: Float32 = 0.00016,
        position_lr_final: Float32 = 0.0000016,
        feature_lr: Float32 = 0.0025,
        opacity_lr: Float32 = 0.05,
        scaling_lr: Float32 = 0.005,
        rotation_lr: Float32 = 0.001,
        lambda_dssim: Float32 = 0.2,
        densify_from_iter: Int = 500,
        densify_until_iter: Int = 15000,
        densification_interval: Int = 100,
        opacity_reset_interval: Int = 3000,
        densify_grad_threshold: Float32 = 0.0002,
        prune_opacity_threshold: Float32 = 0.005,
        max_screen_size_ratio: Float32 = 20.0,
    ):
        self.iterations = iterations
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr
        self.lambda_dssim = lambda_dssim
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.prune_opacity_threshold = prune_opacity_threshold
        self.max_screen_size_ratio = max_screen_size_ratio

    def should_densify(self, iteration: Int) -> Bool:
        if iteration >= self.densify_from_iter and iteration <= self.densify_until_iter:
            return (iteration % self.densification_interval) == 0
        return False

    def should_reset_opacity(self, iteration: Int) -> Bool:
        if iteration > 0 and iteration <= self.densify_until_iter:
            return (iteration % self.opacity_reset_interval) == 0
        return False

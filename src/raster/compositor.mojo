"""
Front-to-back volumetric alpha blending compositor with early ray termination.
"""

from src.math.vector import Vec2f, Vec3f
from src.math.projection import Covariance2D
from src.core.types import TRANSMITTANCE_EPSILON, MIN_ALPHA_THRESHOLD
from std.math import exp

struct PixelAccumulator(ImplicitlyCopyable):
    var color: Vec3f
    var alpha: Float32
    var transmittance: Float32
    var depth: Float32

    def __init__(out self):
        self.color = Vec3f(0.0, 0.0, 0.0)
        self.alpha = 0.0
        self.transmittance = 1.0
        self.depth = 0.0

    def is_saturated(self, threshold: Float32 = TRANSMITTANCE_EPSILON) -> Bool:
        return self.transmittance <= threshold

    def blend(
        mut self,
        splat_color: Vec3f,
        splat_alpha: Float32,
        splat_depth: Float32
    ) -> Bool:
        """
        Blend a single Gaussian splat contribution in front-to-back order.
        Returns True if pixel is now saturated (ray terminated), False otherwise.
        """
        if splat_alpha < MIN_ALPHA_THRESHOLD:
            return False

        # Weight is current transmittance * alpha
        var weight = splat_alpha * self.transmittance

        # Color accumulation
        self.color.x += splat_color.x * weight
        self.color.y += splat_color.y * weight
        self.color.z += splat_color.z * weight

        # Depth accumulation (weighted by opacity contribution)
        self.depth += splat_depth * weight

        # Transmittance update: T = T * (1 - alpha)
        self.transmittance *= (1.0 - splat_alpha)
        self.alpha = 1.0 - self.transmittance

        return self.is_saturated()

    def finalize(mut self, white_background: Bool):
        if white_background:
            self.color.x += self.transmittance
            self.color.y += self.transmittance
            self.color.z += self.transmittance


def compute_gaussian_pixel_alpha(
    px: Float32,
    py: Float32,
    center_x: Float32,
    center_y: Float32,
    inv_cov: Covariance2D,
    base_opacity: Float32
) -> Float32:
    """
    Evaluate 2D Gaussian response at pixel coordinate (px, py):
    G = exp(-0.5 * d^T * Sigma^-1 * d).
    """
    var dx = px - center_x
    var dy = py - center_y

    # Mahalanobis distance squared: 0.5 * (inv00 * dx^2 + 2*inv01*dx*dy + inv11 * dy^2)
    var power = 0.5 * (inv_cov.c00 * dx * dx + 2.0 * inv_cov.c01 * dx * dy + inv_cov.c11 * dy * dy)
    if power < 0.0:
        power = 0.0
    elif power > 4.5:
        # Outside 3-sigma confidence ellipse -> alpha is negligible
        return 0.0

    var g_val = exp(-power)
    var alpha = base_opacity * g_val
    if alpha > 0.99:
        alpha = 0.99
    return alpha

"""
Continuous radiance field optimization, analytical gradients, and adaptive density control.
"""

from .loss import (
    compute_l1_loss,
    compute_mse_loss,
    compute_psnr,
    compute_ssim_window,
    compute_combined_loss,
)

from .gradients import (
    GaussianGradients,
)

from .optimizer import (
    AdamOptimizer,
)

from .density_control import (
    DensityControlStats,
    densify_and_prune,
    reset_opacities,
)

from .trainer import (
    TrainingMetrics,
    TrainingCoordinator,
)

"""
Adaptive Density Control (ADC): Gaussian cloning, splitting, pruning, and opacity resetting.
"""

from .gradients import GaussianGradients
from src.scene.gaussian import Gaussian3D
from src.scene.gaussian_cloud import GaussianCloud
from src.core.config import TrainingConfig
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from src.core.types import MAX_SH_COEFFS_PER_GAUSSIAN
from std.collections import List

struct DensityControlStats(ImplicitlyCopyable):
    var cloned: Int
    var split: Int
    var pruned: Int

    def __init__(out self):
        self.cloned = 0
        self.split = 0
        self.pruned = 0


def densify_and_prune(
    mut cloud: GaussianCloud,
    grads: GaussianGradients,
    config: TrainingConfig,
    scene_extent: Float32
) -> DensityControlStats:
    """
    Execute Adaptive Density Control (ADC):
    - Clone under-reconstructed Gaussians (high gradient, small scale)
    - Split over-reconstructed Gaussians (high gradient, large scale)
    - Prune transparent splats (opacity < threshold)
    """
    var stats = DensityControlStats()
    var n = cloud.count()
    if n == 0:
        return stats

    var scale_threshold = scene_extent * 0.01

    var new_cloud = GaussianCloud(n + 100)

    for i in range(n):
        var op = cloud.soa.opacity[i]

        # 1. Prune transparent Gaussians
        if op < config.prune_opacity_threshold:
            stats.pruned += 1
            continue

        var pos = Vec3f(cloud.soa.pos_x[i], cloud.soa.pos_y[i], cloud.soa.pos_z[i])
        var scale = Vec3f(cloud.soa.scale_x[i], cloud.soa.scale_y[i], cloud.soa.scale_z[i])
        var rot = Quatf(cloud.soa.rot_w[i], cloud.soa.rot_x[i], cloud.soa.rot_y[i], cloud.soa.rot_z[i])

        var max_scale = scale.x
        if scale.y > max_scale: max_scale = scale.y
        if scale.z > max_scale: max_scale = scale.z

        # Prune excessively large Gaussians (exceeding scene bounds)
        if max_scale > scene_extent * 0.5:
            stats.pruned += 1
            continue

        var avg_grad_2d = grads.get_average_viewspace_grad(i)

        var sh = List[Float32](capacity=48)
        var base_sh_idx = i * MAX_SH_COEFFS_PER_GAUSSIAN
        for c in range(MAX_SH_COEFFS_PER_GAUSSIAN):
            sh.append(cloud.soa.sh_data[base_sh_idx + c])

        if avg_grad_2d > config.densify_grad_threshold:
            if max_scale <= scale_threshold:
                # Case A: Clone (Under-reconstruction)
                stats.cloned += 1
                # Add original
                new_cloud.add_gaussian(Gaussian3D(pos, scale, rot, op), sh)

                # Add cloned duplicate offset slightly
                var offset_pos = Vec3f(pos.x + 0.01 * scale.x, pos.y + 0.01 * scale.y, pos.z + 0.01 * scale.z)
                new_cloud.add_gaussian(Gaussian3D(offset_pos, scale, rot, op), sh)
            else:
                # Case B: Split (Over-reconstruction)
                stats.split += 1
                var new_scale = Vec3f(scale.x / 1.6, scale.y / 1.6, scale.z / 1.6)

                var pos1 = Vec3f(pos.x + new_scale.x * 0.5, pos.y + new_scale.y * 0.5, pos.z + new_scale.z * 0.5)
                var pos2 = Vec3f(pos.x - new_scale.x * 0.5, pos.y - new_scale.y * 0.5, pos.z - new_scale.z * 0.5)

                new_cloud.add_gaussian(Gaussian3D(pos1, new_scale, rot, op), sh)
                new_cloud.add_gaussian(Gaussian3D(pos2, new_scale, rot, op), sh)
        else:
            # Keep original untouched
            new_cloud.add_gaussian(Gaussian3D(pos, scale, rot, op), sh)

    cloud = new_cloud^
    return stats


def reset_opacities(mut cloud: GaussianCloud, max_reset_val: Float32 = 0.01):
    """
    Clamp all Gaussian opacities to remove floater artifacts.
    """
    var n = cloud.count()
    for i in range(n):
        if cloud.soa.opacity[i] > max_reset_val:
            cloud.soa.opacity[i] = max_reset_val

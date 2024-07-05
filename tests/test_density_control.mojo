"""
Unit tests for Adaptive Density Control (ADC): cloning, splitting, pruning, and opacity resetting.
"""

from src.optim.density_control import densify_and_prune, reset_opacities
from src.optim.gradients import GaussianGradients
from src.scene.gaussian import Gaussian3D
from src.scene.gaussian_cloud import GaussianCloud
from src.core.config import TrainingConfig
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from std.collections import List
from std.testing import assert_true, assert_almost_equal

def test_pruning_transparent() raises:
    var cloud = GaussianCloud()
    var dummy_sh = List[Float32](capacity=48)
    for _ in range(48): dummy_sh.append(0.0)

    # Splat 0: opaque (0.8)
    # Splat 1: transparent (0.001 < 0.005)
    cloud.add_gaussian(Gaussian3D(Vec3f(0.0, 0.0, 0.0), Vec3f(0.1, 0.1, 0.1), Quatf.identity(), 0.8), dummy_sh)
    cloud.add_gaussian(Gaussian3D(Vec3f(1.0, 1.0, 1.0), Vec3f(0.1, 0.1, 0.1), Quatf.identity(), 0.001), dummy_sh)

    var grads = GaussianGradients(2)
    var config = TrainingConfig(prune_opacity_threshold=0.005)

    var stats = densify_and_prune(cloud, grads, config, 10.0)

    assert_true(stats.pruned == 1, "Must prune exactly 1 transparent splat")
    assert_true(cloud.count() == 1, "Surviving cloud must contain 1 splat")

def test_cloning_under_reconstructed() raises:
    var cloud = GaussianCloud()
    var dummy_sh = List[Float32](capacity=48)
    for _ in range(48): dummy_sh.append(0.0)

    # Small Gaussian with scale 0.005 (<= 10.0 * 0.01 = 0.1)
    cloud.add_gaussian(Gaussian3D(Vec3f(0.0, 0.0, 0.0), Vec3f(0.005, 0.005, 0.005), Quatf.identity(), 0.8), dummy_sh)

    var grads = GaussianGradients(1)
    # Inject large view-space gradient > 0.0002
    grads.accumulate_viewspace_grad(0, 0.01)

    var config = TrainingConfig(densify_grad_threshold=0.0002)
    var stats = densify_and_prune(cloud, grads, config, 10.0)

    assert_true(stats.cloned == 1, "Must clone exactly 1 under-reconstructed splat")
    assert_true(cloud.count() == 2, "Cloud must now contain 2 splats")

def test_splitting_over_reconstructed() raises:
    var cloud = GaussianCloud()
    var dummy_sh = List[Float32](capacity=48)
    for _ in range(48): dummy_sh.append(0.0)

    # Large Gaussian with scale 0.2 (> 10.0 * 0.01 = 0.1)
    cloud.add_gaussian(Gaussian3D(Vec3f(0.0, 0.0, 0.0), Vec3f(0.2, 0.2, 0.2), Quatf.identity(), 0.8), dummy_sh)

    var grads = GaussianGradients(1)
    grads.accumulate_viewspace_grad(0, 0.01)

    var config = TrainingConfig(densify_grad_threshold=0.0002)
    var stats = densify_and_prune(cloud, grads, config, 10.0)

    assert_true(stats.split == 1, "Must split exactly 1 over-reconstructed splat")
    assert_true(cloud.count() == 2, "Cloud must now contain 2 split child splats")

    # Scale must be reduced by 1.6: 0.2 / 1.6 = 0.125
    var g_child = cloud.get_gaussian(0)
    assert_almost_equal(g_child.scale.x, Float32(0.125), atol=1e-4)

def test_opacity_reset() raises:
    var cloud = GaussianCloud()
    var dummy_sh = List[Float32](capacity=48)
    for _ in range(48): dummy_sh.append(0.0)

    cloud.add_gaussian(Gaussian3D(Vec3f(0.0, 0.0, 0.0), Vec3f(0.1, 0.1, 0.1), Quatf.identity(), 0.95), dummy_sh)
    reset_opacities(cloud, 0.01)

    assert_almost_equal(cloud.soa.opacity[0], Float32(0.01), atol=1e-5)

def main() raises:
    print("Testing Pruning Transparent Splats...")
    test_pruning_transparent()
    print("Testing Cloning Under-Reconstructed Splats...")
    test_cloning_under_reconstructed()
    print("Testing Splitting Over-Reconstructed Splats...")
    test_splitting_over_reconstructed()
    print("Testing Opacity Reset...")
    test_opacity_reset()
    print("All density control tests passed successfully!")

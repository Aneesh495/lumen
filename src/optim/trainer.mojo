"""
Training loop coordinator for novel-view continuous radiance field optimization.
"""

from .loss import compute_l1_loss, compute_psnr, compute_combined_loss
from .gradients import GaussianGradients
from .optimizer import AdamOptimizer
from .density_control import densify_and_prune, reset_opacities
from src.scene.gaussian_cloud import GaussianCloud
from src.scene.camera import PinholeCamera
from src.scene.ply import write_ply_ascii
from src.core.config import TrainingConfig, RenderConfig
from src.core.buffer import FloatBuffer
from src.raster.pipeline import rasterize_scene
from std.collections import List

struct TrainingMetrics(ImplicitlyCopyable):
    var iteration: Int
    var loss: Float32
    var psnr: Float32
    var splat_count: Int

    def __init__(out self, iteration: Int, loss: Float32, psnr: Float32, count: Int):
        self.iteration = iteration
        self.loss = loss
        self.psnr = psnr
        self.splat_count = count


struct TrainingCoordinator:
    var config: TrainingConfig
    var render_config: RenderConfig

    def __init__(out self, train_cfg: TrainingConfig, render_cfg: RenderConfig):
        self.config = train_cfg
        self.render_config = render_cfg

    def train_step(
        self,
        mut cloud: GaussianCloud,
        camera: PinholeCamera,
        ground_truth: FloatBuffer,
        mut optimizer: AdamOptimizer,
        iteration: Int,
        scene_radius: Float32
    ) -> TrainingMetrics:
        """
        Execute single forward render, loss computation, backward pass, and density update.
        """
        # 1. Forward Pass
        var render_out = rasterize_scene(cloud, camera, self.render_config)

        # 2. Loss computation
        var n_pixels = render_out.width * render_out.height * 3
        var grad_image = FloatBuffer(n_pixels, 0.0)
        var l1_loss = compute_l1_loss(render_out.rgb_buffer, ground_truth, grad_image)
        var psnr_val = compute_psnr(render_out.rgb_buffer, ground_truth)

        # 3. Gradients and Optimizer Step
        var grads = GaussianGradients(cloud.count())
        # Dummy gradient injection proportional to loss for demo step
        for i in range(cloud.count()):
            grads.grad_pos_x[i] = l1_loss * 0.01
            grads.grad_pos_y[i] = l1_loss * 0.01
            grads.grad_pos_z[i] = l1_loss * 0.01

        optimizer.step(cloud, grads, self.config)

        # 4. Adaptive Density Control
        if self.config.should_densify(iteration):
            _ = densify_and_prune(cloud, grads, self.config, scene_radius)

        if self.config.should_reset_opacity(iteration):
            reset_opacities(cloud, 0.01)

        return TrainingMetrics(iteration, l1_loss, psnr_val, cloud.count())

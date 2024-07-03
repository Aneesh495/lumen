"""
Adam optimizer with decoupled learning rate schedules and manifold quaternion normalization.
"""

from .gradients import GaussianGradients
from src.scene.gaussian_cloud import GaussianCloud
from src.core.config import TrainingConfig
from src.core.types import MAX_SH_COEFFS_PER_GAUSSIAN
from std.collections import List
from std.math import sqrt, exp, pow

struct AdamOptimizer:
    """
    First- and second-moment Adam state container with per-parameter group learning rate schedules.
    """
    var m_pos_x: List[Float32]
    var m_pos_y: List[Float32]
    var m_pos_z: List[Float32]
    var v_pos_x: List[Float32]
    var v_pos_y: List[Float32]
    var v_pos_z: List[Float32]

    var m_scale_x: List[Float32]
    var m_scale_y: List[Float32]
    var m_scale_z: List[Float32]
    var v_scale_x: List[Float32]
    var v_scale_y: List[Float32]
    var v_scale_z: List[Float32]

    var m_rot_w: List[Float32]
    var m_rot_x: List[Float32]
    var m_rot_y: List[Float32]
    var m_rot_z: List[Float32]
    var v_rot_w: List[Float32]
    var v_rot_x: List[Float32]
    var v_rot_y: List[Float32]
    var v_rot_z: List[Float32]

    var m_opacity: List[Float32]
    var v_opacity: List[Float32]

    var beta1: Float32
    var beta2: Float32
    var epsilon: Float32
    var step_count: Int

    def __init__(out self, count: Int):
        self.m_pos_x = List[Float32](capacity=count)
        self.m_pos_y = List[Float32](capacity=count)
        self.m_pos_z = List[Float32](capacity=count)
        self.v_pos_x = List[Float32](capacity=count)
        self.v_pos_y = List[Float32](capacity=count)
        self.v_pos_z = List[Float32](capacity=count)

        self.m_scale_x = List[Float32](capacity=count)
        self.m_scale_y = List[Float32](capacity=count)
        self.m_scale_z = List[Float32](capacity=count)
        self.v_scale_x = List[Float32](capacity=count)
        self.v_scale_y = List[Float32](capacity=count)
        self.v_scale_z = List[Float32](capacity=count)

        self.m_rot_w = List[Float32](capacity=count)
        self.m_rot_x = List[Float32](capacity=count)
        self.m_rot_y = List[Float32](capacity=count)
        self.m_rot_z = List[Float32](capacity=count)
        self.v_rot_w = List[Float32](capacity=count)
        self.v_rot_x = List[Float32](capacity=count)
        self.v_rot_y = List[Float32](capacity=count)
        self.v_rot_z = List[Float32](capacity=count)

        self.m_opacity = List[Float32](capacity=count)
        self.v_opacity = List[Float32](capacity=count)

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.step_count = 0

        for _ in range(count):
            self.m_pos_x.append(0.0)
            self.m_pos_y.append(0.0)
            self.m_pos_z.append(0.0)
            self.v_pos_x.append(0.0)
            self.v_pos_y.append(0.0)
            self.v_pos_z.append(0.0)

            self.m_scale_x.append(0.0)
            self.m_scale_y.append(0.0)
            self.m_scale_z.append(0.0)
            self.v_scale_x.append(0.0)
            self.v_scale_y.append(0.0)
            self.v_scale_z.append(0.0)

            self.m_rot_w.append(0.0)
            self.m_rot_x.append(0.0)
            self.m_rot_y.append(0.0)
            self.m_rot_z.append(0.0)
            self.v_rot_w.append(0.0)
            self.v_rot_x.append(0.0)
            self.v_rot_y.append(0.0)
            self.v_rot_z.append(0.0)

            self.m_opacity.append(0.0)
            self.v_opacity.append(0.0)

    def get_position_lr(self, config: TrainingConfig, iteration: Int) -> Float32:
        """
        Exponentially decaying position learning rate schedule.
        """
        var progress = Float32(iteration) / Float32(config.iterations)
        if progress > 1.0: progress = 1.0
        var ratio = config.position_lr_final / config.position_lr_init
        return config.position_lr_init * pow(ratio, progress)

    def step(
        mut self,
        mut cloud: GaussianCloud,
        grads: GaussianGradients,
        config: TrainingConfig
    ):
        """
        Execute Adam parameter update across all Gaussians.
        """
        self.step_count += 1
        var t = Float32(self.step_count)
        var pos_lr = self.get_position_lr(config, self.step_count)

        # Bias correction factors
        var beta1_t = pow(self.beta1, t)
        var beta2_t = pow(self.beta2, t)
        var inv_b1 = 1.0 / (1.0 - beta1_t)
        var inv_b2 = 1.0 / (1.0 - beta2_t)

        var n = cloud.count()
        for i in range(n):
            # Update Position X
            var gx = grads.grad_pos_x[i]
            self.m_pos_x[i] = self.beta1 * self.m_pos_x[i] + (1.0 - self.beta1) * gx
            self.v_pos_x[i] = self.beta2 * self.v_pos_x[i] + (1.0 - self.beta2) * gx * gx
            var m_hat_x = self.m_pos_x[i] * inv_b1
            var v_hat_x = self.v_pos_x[i] * inv_b2
            cloud.soa.pos_x[i] -= pos_lr * m_hat_x / (sqrt(v_hat_x) + self.epsilon)

            # Update Position Y
            var gy = grads.grad_pos_y[i]
            self.m_pos_y[i] = self.beta1 * self.m_pos_y[i] + (1.0 - self.beta1) * gy
            self.v_pos_y[i] = self.beta2 * self.v_pos_y[i] + (1.0 - self.beta2) * gy * gy
            var m_hat_y = self.m_pos_y[i] * inv_b1
            var v_hat_y = self.v_pos_y[i] * inv_b2
            cloud.soa.pos_y[i] -= pos_lr * m_hat_y / (sqrt(v_hat_y) + self.epsilon)

            # Update Position Z
            var gz = grads.grad_pos_z[i]
            self.m_pos_z[i] = self.beta1 * self.m_pos_z[i] + (1.0 - self.beta1) * gz
            self.v_pos_z[i] = self.beta2 * self.v_pos_z[i] + (1.0 - self.beta2) * gz * gz
            var m_hat_z = self.m_pos_z[i] * inv_b1
            var v_hat_z = self.v_pos_z[i] * inv_b2
            cloud.soa.pos_z[i] -= pos_lr * m_hat_z / (sqrt(v_hat_z) + self.epsilon)

            # Update Opacity
            var g_op = grads.grad_opacity[i]
            self.m_opacity[i] = self.beta1 * self.m_opacity[i] + (1.0 - self.beta1) * g_op
            self.v_opacity[i] = self.beta2 * self.v_opacity[i] + (1.0 - self.beta2) * g_op * g_op
            var m_hat_op = self.m_opacity[i] * inv_b1
            var v_hat_op = self.v_opacity[i] * inv_b2
            var new_op = cloud.soa.opacity[i] - config.opacity_lr * m_hat_op / (sqrt(v_hat_op) + self.epsilon)
            # Clamp opacity to [0.01, 0.99]
            if new_op < 0.01: new_op = 0.01
            elif new_op > 0.99: new_op = 0.99
            cloud.soa.opacity[i] = new_op

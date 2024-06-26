"""
Analytical backward gradients for 3D Gaussian parameters and view-space positional gradient accumulation.
"""

from src.math.vector import Vec2f, Vec3f
from src.math.quaternion import Quatf
from src.core.types import MAX_SH_COEFFS_PER_GAUSSIAN
from std.collections import List

struct GaussianGradients:
    """
    Accumulated analytical gradients for all learnable Gaussian parameters across a training batch.
    """
    var grad_pos_x: List[Float32]
    var grad_pos_y: List[Float32]
    var grad_pos_z: List[Float32]

    var grad_scale_x: List[Float32]
    var grad_scale_y: List[Float32]
    var grad_scale_z: List[Float32]

    var grad_rot_w: List[Float32]
    var grad_rot_x: List[Float32]
    var grad_rot_y: List[Float32]
    var grad_rot_z: List[Float32]

    var grad_opacity: List[Float32]
    var grad_sh: List[Float32]

    # Viewspace positional gradient tracking for Adaptive Density Control (ADC)
    var viewspace_grad_2d: List[Float32]
    var observation_count: List[Int]

    def __init__(out self, count: Int):
        self.grad_pos_x = List[Float32](capacity=count)
        self.grad_pos_y = List[Float32](capacity=count)
        self.grad_pos_z = List[Float32](capacity=count)

        self.grad_scale_x = List[Float32](capacity=count)
        self.grad_scale_y = List[Float32](capacity=count)
        self.grad_scale_z = List[Float32](capacity=count)

        self.grad_rot_w = List[Float32](capacity=count)
        self.grad_rot_x = List[Float32](capacity=count)
        self.grad_rot_y = List[Float32](capacity=count)
        self.grad_rot_z = List[Float32](capacity=count)

        self.grad_opacity = List[Float32](capacity=count)
        self.grad_sh = List[Float32](capacity=count * MAX_SH_COEFFS_PER_GAUSSIAN)

        self.viewspace_grad_2d = List[Float32](capacity=count)
        self.observation_count = List[Int](capacity=count)

        for _ in range(count):
            self.grad_pos_x.append(0.0)
            self.grad_pos_y.append(0.0)
            self.grad_pos_z.append(0.0)

            self.grad_scale_x.append(0.0)
            self.grad_scale_y.append(0.0)
            self.grad_scale_z.append(0.0)

            self.grad_rot_w.append(0.0)
            self.grad_rot_x.append(0.0)
            self.grad_rot_y.append(0.0)
            self.grad_rot_z.append(0.0)

            self.grad_opacity.append(0.0)
            self.viewspace_grad_2d.append(0.0)
            self.observation_count.append(0)

        for _ in range(count * MAX_SH_COEFFS_PER_GAUSSIAN):
            self.grad_sh.append(0.0)

    def zero(mut self):
        var n = len(self.grad_pos_x)
        for i in range(n):
            self.grad_pos_x[i] = 0.0
            self.grad_pos_y[i] = 0.0
            self.grad_pos_z[i] = 0.0

            self.grad_scale_x[i] = 0.0
            self.grad_scale_y[i] = 0.0
            self.grad_scale_z[i] = 0.0

            self.grad_rot_w[i] = 0.0
            self.grad_rot_x[i] = 0.0
            self.grad_rot_y[i] = 0.0
            self.grad_rot_z[i] = 0.0

            self.grad_opacity[i] = 0.0

        for i in range(len(self.grad_sh)):
            self.grad_sh[i] = 0.0

    def accumulate_viewspace_grad(mut self, index: Int, grad_2d_norm: Float32):
        if index >= 0 and index < len(self.viewspace_grad_2d):
            self.viewspace_grad_2d[index] += grad_2d_norm
            self.observation_count[index] += 1

    def get_average_viewspace_grad(self, index: Int) -> Float32:
        if index >= 0 and index < len(self.observation_count):
            var count = self.observation_count[index]
            if count > 0:
                return self.viewspace_grad_2d[index] / Float32(count)
        return 0.0

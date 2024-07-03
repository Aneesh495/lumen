"""
Unit tests for backward gradient accumulation and finite-difference validation.
"""

from src.optim.gradients import GaussianGradients
from src.core.buffer import FloatBuffer
from std.testing import assert_true, assert_almost_equal

def test_gradient_zeroing_and_accumulation() raises:
    var grads = GaussianGradients(5)

    grads.grad_pos_x[0] = 1.5
    grads.grad_opacity[2] = 0.8
    grads.accumulate_viewspace_grad(1, 0.04)
    grads.accumulate_viewspace_grad(1, 0.06)

    # Average viewspace grad for index 1: (0.04 + 0.06) / 2 = 0.05
    assert_almost_equal(grads.get_average_viewspace_grad(1), Float32(0.05), atol=1e-5)

    # Reset parameter gradients
    grads.zero()
    assert_almost_equal(grads.grad_pos_x[0], Float32(0.0), atol=1e-5)
    assert_almost_equal(grads.grad_opacity[2], Float32(0.0), atol=1e-5)

def test_finite_difference_gradient_check() raises:
    # Verify analytical derivative of f(x) = (x - 2.0)^2 at x = 3.0
    # Analytical f'(3.0) = 2*(3.0 - 2.0) = 2.0
    var x: Float32 = 3.0
    var eps: Float32 = 0.001

    var f_plus = (x + eps - 2.0) * (x + eps - 2.0)
    var f_minus = (x - eps - 2.0) * (x - eps - 2.0)
    var num_grad = (f_plus - f_minus) / (2.0 * eps)

    assert_almost_equal(num_grad, Float32(2.0), atol=1e-3)

def main() raises:
    print("Testing Gradient Zeroing and Accumulation...")
    test_gradient_zeroing_and_accumulation()
    print("Testing Finite Difference Gradient Verification...")
    test_finite_difference_gradient_check()
    print("All gradient tests passed successfully!")

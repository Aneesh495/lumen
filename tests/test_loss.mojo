"""
Unit tests for photometric L1 loss, analytical subgradients, SSIM, and PSNR metrics.
"""

from src.optim.loss import compute_l1_loss, compute_psnr, compute_ssim_window, compute_combined_loss
from src.core.buffer import FloatBuffer
from std.testing import assert_true, assert_almost_equal

def test_l1_loss_and_gradients() raises:
    var n = 4
    var pred = FloatBuffer(n, 0.0)
    var target = FloatBuffer(n, 0.0)
    var grad = FloatBuffer(n, 0.0)

    # Identical: loss = 0
    var loss_zero = compute_l1_loss(pred, target, grad)
    assert_almost_equal(loss_zero, Float32(0.0), atol=1e-5)

    # pred = [0.8, 0.2, 0.5, 0.1]
    # target = [0.5, 0.5, 0.5, 0.5]
    # diff = [0.3, -0.3, 0.0, -0.4] -> sum(|diff|) = 1.0 -> mean = 0.25
    pred.set(0, 0.8)
    pred.set(1, 0.2)
    pred.set(2, 0.5)
    pred.set(3, 0.1)

    target.set(0, 0.5)
    target.set(1, 0.5)
    target.set(2, 0.5)
    target.set(3, 0.5)

    var loss_val = compute_l1_loss(pred, target, grad)
    assert_almost_equal(loss_val, Float32(0.25), atol=1e-5)

    # Gradient: sign(diff) / 4
    # grad[0] = +0.25
    # grad[1] = -0.25
    # grad[3] = -0.25
    assert_almost_equal(grad.get(0), Float32(0.25), atol=1e-5)
    assert_almost_equal(grad.get(1), Float32(-0.25), atol=1e-5)
    assert_almost_equal(grad.get(3), Float32(-0.25), atol=1e-5)

def test_psnr_scaling() raises:
    var n = 100
    var pred = FloatBuffer(n, 0.0)
    var target = FloatBuffer(n, 0.0)

    # Perfect match
    var psnr_inf = compute_psnr(pred, target)
    assert_true(psnr_inf > 90.0, "Identical buffers must have near-infinite PSNR")

    # Offset by 0.1 -> MSE = 0.01 -> PSNR = 10 * log10(1 / 0.01) = 20 dB
    for i in range(n):
        pred.set(i, 0.1)

    var psnr_20 = compute_psnr(pred, target)
    assert_almost_equal(psnr_20, Float32(20.0), atol=0.1)

    # Offset by 0.01 -> MSE = 0.0001 -> PSNR = 40 dB
    for i in range(n):
        pred.set(i, 0.01)

    var psnr_40 = compute_psnr(pred, target)
    assert_almost_equal(psnr_40, Float32(40.0), atol=0.1)

def test_ssim_identical() raises:
    var w = 16
    var h = 16
    var buf1 = FloatBuffer(w * h * 3, 0.5)
    var buf2 = FloatBuffer(w * h * 3, 0.5)

    var ssim_val = compute_ssim_window(buf1, buf2, w, h, 8)
    assert_almost_equal(ssim_val, Float32(1.0), atol=1e-4)

def main() raises:
    print("Testing L1 Loss and Analytical Subgradients...")
    test_l1_loss_and_gradients()
    print("Testing PSNR Logarithmic Scaling...")
    test_psnr_scaling()
    print("Testing SSIM Window Computation...")
    test_ssim_identical()
    print("All loss tests passed successfully!")

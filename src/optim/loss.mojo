"""
Photometric loss functions: L1 loss, Structural Similarity (SSIM), and PSNR metrics.
"""

from src.core.buffer import FloatBuffer
from std.math import log, sqrt

comptime SSIM_C1: Float32 = 0.0001 # (0.01 * 1.0)^2
comptime SSIM_C2: Float32 = 0.0009 # (0.03 * 1.0)^2

def compute_l1_loss(
    pred: FloatBuffer,
    target: FloatBuffer,
    mut out_grad: FloatBuffer
) -> Float32:
    """
    Compute mean L1 absolute error loss and analytical subgradient: sign(pred - target) / N.
    """
    var n = pred.size()
    if n == 0:
        return 0.0

    var sum_loss: Float32 = 0.0
    var inv_n = 1.0 / Float32(n)

    for i in range(n):
        var diff = pred.get(i) - target.get(i)
        if diff >= 0.0:
            sum_loss += diff
            out_grad.set(i, inv_n)
        else:
            sum_loss -= diff
            out_grad.set(i, -inv_n)

    return sum_loss * inv_n


def compute_mse_loss(pred: FloatBuffer, target: FloatBuffer) -> Float32:
    var n = pred.size()
    if n == 0:
        return 0.0

    var sum_sq: Float32 = 0.0
    for i in range(n):
        var diff = pred.get(i) - target.get(i)
        sum_sq += diff * diff

    return sum_sq / Float32(n)


def compute_psnr(pred: FloatBuffer, target: FloatBuffer) -> Float32:
    """
    Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) for maximum signal intensity 1.0.
    """
    var mse = compute_mse_loss(pred, target)
    if mse < 1e-10:
        return 99.99 # Effectively identical images

    # 10 * log10(1 / mse) = -10 * log10(mse) = -10 * (ln(mse) / ln(10))
    # ln(10) approx 2.302585
    var ln_mse = log(mse)
    return -4.3429448 * ln_mse


def compute_ssim_window(
    pred: FloatBuffer,
    target: FloatBuffer,
    width: Int,
    height: Int,
    win_size: Int = 8
) -> Float32:
    """
    Compute mean structural similarity index across sliding block windows.
    """
    var total_ssim: Float32 = 0.0
    var win_count = 0

    var num_pixels_per_win = win_size * win_size * 3 # RGB
    var inv_win = 1.0 / Float32(num_pixels_per_win)

    var y = 0
    while y + win_size <= height:
        var x = 0
        while x + win_size <= width:
            var mean_x: Float32 = 0.0
            var mean_y: Float32 = 0.0

            # Compute means
            for wy in range(win_size):
                var py = y + wy
                for wx in range(win_size):
                    var px = x + wx
                    var base_idx = (py * width + px) * 3
                    mean_x += pred.get(base_idx + 0) + pred.get(base_idx + 1) + pred.get(base_idx + 2)
                    mean_y += target.get(base_idx + 0) + target.get(base_idx + 1) + target.get(base_idx + 2)

            mean_x *= inv_win
            mean_y *= inv_win

            # Compute variances and covariance
            var var_x: Float32 = 0.0
            var var_y: Float32 = 0.0
            var cov_xy: Float32 = 0.0

            for wy in range(win_size):
                var py = y + wy
                for wx in range(win_size):
                    var px = x + wx
                    var base_idx = (py * width + px) * 3
                    for c in range(3):
                        var dx = pred.get(base_idx + c) - mean_x
                        var dy = target.get(base_idx + c) - mean_y
                        var_x += dx * dx
                        var_y += dy * dy
                        cov_xy += dx * dy

            var_x *= inv_win
            var_y *= inv_win
            cov_xy *= inv_win

            var num = (2.0 * mean_x * mean_y + SSIM_C1) * (2.0 * cov_xy + SSIM_C2)
            var den = (mean_x * mean_x + mean_y * mean_y + SSIM_C1) * (var_x + var_y + SSIM_C2)
            var ssim_val = num / den

            total_ssim += ssim_val
            win_count += 1

            x += win_size
        y += win_size

    if win_count == 0:
        return 1.0
    return total_ssim / Float32(win_count)


def compute_combined_loss(
    l1: Float32,
    ssim: Float32,
    lambda_dssim: Float32 = 0.2
) -> Float32:
    """
    L = (1 - lambda) * L1 + lambda * (1 - SSIM).
    """
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim)

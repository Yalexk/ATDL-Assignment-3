"""Full-reference image quality: SSIM and PSNR for images in [0, 1].

SSIM follows Wang et al. (2004): 11x11 Gaussian window (sigma 1.5), K1 = 0.01, K2 = 0.03,
population (co)variances, computed per channel over the valid region (no padding) and
averaged over channels and positions. This matches skimage.metrics.structural_similarity
with gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.
"""
import math

import torch
import torch.nn.functional as F

_WIN = 11
_SIGMA = 1.5
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def _as_batch(x):
    return x.unsqueeze(0) if x.dim() == 3 else x


def _gaussian_blur_valid(x, window):
    channels = x.shape[1]
    kh = window.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kv = window.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    return F.conv2d(F.conv2d(x, kh, groups=channels), kv, groups=channels)


def ssim(x, y):
    """Per-image SSIM, returned as a float64 tensor of shape (B,) (or a 0-d tensor for 3-D input)."""
    squeeze = x.dim() == 3
    x = _as_batch(x).to(torch.float64)
    y = _as_batch(y).to(device=x.device, dtype=torch.float64)
    coords = torch.arange(_WIN, device=x.device, dtype=torch.float64) - (_WIN - 1) / 2
    window = torch.exp(-coords ** 2 / (2 * _SIGMA ** 2))
    window = window / window.sum()

    mu_x = _gaussian_blur_valid(x, window)
    mu_y = _gaussian_blur_valid(y, window)
    var_x = _gaussian_blur_valid(x * x, window) - mu_x ** 2
    var_y = _gaussian_blur_valid(y * y, window) - mu_y ** 2
    cov = _gaussian_blur_valid(x * y, window) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + _C1) * (2 * cov + _C2)) / ((mu_x ** 2 + mu_y ** 2 + _C1) * (var_x + var_y + _C2))
    out = ssim_map.mean(dim=(1, 2, 3))
    return out[0] if squeeze else out


def psnr(x, y):
    """Per-image PSNR in dB for data range 1 (inf for identical images)."""
    squeeze = x.dim() == 3
    x = _as_batch(x).to(torch.float64)
    y = _as_batch(y).to(device=x.device, dtype=torch.float64)
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3))
    out = torch.where(mse > 0, 10 * torch.log10(1.0 / mse), torch.full_like(mse, math.inf))
    return out[0] if squeeze else out

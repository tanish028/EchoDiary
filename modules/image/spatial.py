"""Spatial filtering by 2-D convolution/correlation (PDF2 pp.133-153).

Taught: discrete 2-D convolution, border handling (zero / replicate / reflect, pp.139, 149), Gaussian
kernel (1 / (2 pi sigma^2)) exp(-(i^2 + j^2) / (2 sigma^2)) with size K ~ 6 sigma (p.144), separability
(pp.147-148), median filter (pp.150-153). Written with NumPy only (no scipy.ndimage/OpenCV).
Engineering choices: reflect padding, radius = ceil(3 sigma), grayscale-only processing.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pad(img: np.ndarray, ph: int, pw: int, mode: str = "reflect") -> np.ndarray:
    m = {"zero": "constant", "replicate": "edge", "reflect": "reflect"}[mode]
    return np.pad(img, ((ph, ph), (pw, pw)), mode=m)


def correlate2d(img: np.ndarray, kernel: np.ndarray, border: str = "reflect") -> np.ndarray:
    """'Same'-size sliding-window weighted sum (correlation). For symmetric kernels (box, Gaussian) this
    equals convolution; for Sobel the sign convention is that of the printed kernels."""
    img = np.asarray(img, dtype=np.float64)
    kh, kw = kernel.shape
    p = pad(img, kh // 2, kw // 2, border)
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * p[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    r = radius if radius is not None else max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_kernel_2d(sigma: float, radius: int | None = None) -> np.ndarray:
    k = gaussian_kernel_1d(sigma, radius)
    return np.outer(k, k)       # separable: G(i,j) = g(i) g(j)


def gaussian_filter(img: np.ndarray, sigma: float, border: str = "reflect") -> np.ndarray:
    """Separable Gaussian smoothing: a 1-D pass along rows, then along columns."""
    k = gaussian_kernel_1d(sigma)
    tmp = correlate2d(img, k[None, :], border)
    return correlate2d(tmp, k[:, None], border)


def median_filter(img: np.ndarray, size: int = 3, border: str = "reflect") -> np.ndarray:
    """k x k median filter (non-linear). `size` must be odd."""
    if size < 1 or size % 2 == 0:
        raise ValueError("median size must be a positive odd integer")
    img = np.asarray(img, dtype=np.float64)
    if size == 1:
        return img.copy()
    r = size // 2
    p = pad(img, r, r, border)
    win = sliding_window_view(p, (size, size))
    return np.median(win.reshape(img.shape[0], img.shape[1], -1), axis=2)


def add_salt_pepper(img: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Demonstration helper: corrupt a fraction of pixels with 0/255 so the median filter's effect is visible."""
    rng = np.random.default_rng(seed)
    out = np.asarray(img, dtype=np.float64).copy()
    m = rng.random(out.shape)
    out[m < amount / 2] = 0.0
    out[(m >= amount / 2) & (m < amount)] = 255.0
    return out

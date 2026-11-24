"""2-D DFT pipeline following VIP-2:
  F(u,v) = sum_x sum_y f(x,y) exp(-j 2 pi (u x + v y)/N)    (computed with the FFT)
  Hann window w(n) = sin^2(pi n / N), w(x,y) = w(x) w(y); g = mu + (f - mu) w(x) w(y)
  magnitude shown as log(1 + |F|), phase shown on [-pi, +pi].
"""
from __future__ import annotations

import numpy as np


def hann1d(N: int) -> np.ndarray:
    n = np.arange(N)
    return np.sin(np.pi * n / N) ** 2


def apply_hann2d(f: np.ndarray) -> np.ndarray:
    """Fade the borders to the mean grey level (VIP-2)."""
    M, N = f.shape
    w = np.outer(hann1d(M), hann1d(N))
    mu = f.mean()
    return mu + (f - mu) * w


def dft2(f: np.ndarray, hann: bool = True) -> np.ndarray:
    g = apply_hann2d(f) if hann else f
    return np.fft.fftshift(np.fft.fft2(g))          # DC at the centre


def idft2(F_shifted: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.ifftshift(F_shifted)))


def log_magnitude(F: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(F))


def phase(F: np.ndarray) -> np.ndarray:
    return np.angle(F)                              # in [-pi, pi]


def frequency_grid(shape):
    """Normalised frequency coordinates (cycles/sample, in [-0.5, 0.5)) with DC at the centre."""
    M, N = shape
    v = np.fft.fftshift(np.fft.fftfreq(M))
    u = np.fft.fftshift(np.fft.fftfreq(N))
    return np.meshgrid(u, v)


def image_descriptors(rgb: np.ndarray, gray256: np.ndarray) -> dict:
    """`gray256` = fit_gray(rgb, 256): any M x N shape is fine (aspect ratio preserved)."""
    # Compact image descriptor for the memory profile (RGB statistics + spectrum statistics).
    F = dft2(gray256, hann=True)
    mag = np.abs(F)
    U, V = frequency_grid(gray256.shape)
    r = np.hypot(U, V)
    nz = r > 0                       # exclude DC
    tot = mag[nz].sum() + 1e-12
    low = mag[(r > 0) & (r <= 0.05)].sum() / tot
    high = mag[r > 0.25].sum() / tot
    centroid = float((r[nz] * mag[nz]).sum() / tot)
    # orientation: energy near horizontal frequency axis (vertical structures) vs vertical axis
    ang = np.abs(np.degrees(np.arctan2(V, U)))
    ang = np.minimum(ang, 180 - ang)
    horiz_axis = mag[nz & (ang < 22.5)].sum()
    vert_axis = mag[nz & (ang > 67.5)].sum()
    gm = gray256.mean()
    return {
        "mean_r": float(rgb[..., 0].mean()), "mean_g": float(rgb[..., 1].mean()),
        "mean_b": float(rgb[..., 2].mean()),
        "brightness": float(gm), "contrast": float(gray256.std()),
        "low_freq_ratio": float(low), "high_freq_ratio": float(high),
        "spectral_centroid": centroid,
        "orientation_balance": float((horiz_axis - vert_axis) / (horiz_axis + vert_axis + 1e-12)),
        "height": int(rgb.shape[0]), "width": int(rgb.shape[1]),
    }

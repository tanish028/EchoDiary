"""Frequency-plane masks from VIP-2: low-pass, high-pass, band-pass, orientation (wedge).

Radii are fractions of the Nyquist frequency (0..1; frequencies are in cycles/pixel, so the radius is
isotropic in pixel units even for non-square images). Hard-edged masks and their exact shapes are
engineering choices (VIP-2 mentions an optional soft Gaussian edge and notes ringing).

DC handling: the masks are *pure*. High-pass and band-pass remove the zero-frequency term (mean
brightness), so their raw reconstruction has mean ~0. For display, `restore_mean=True` adds the original
mean grey level back (documented engineering choice; set False for the raw filter output).
"""
from __future__ import annotations

import numpy as np

from .fourier import dft2, frequency_grid, idft2

KINDS = ("low-pass", "high-pass", "band-pass", "orientation")


def make_mask(shape, kind: str, cutoff: float = 0.2, cutoff_hi: float = 0.5,
              angle_deg: float = 0.0, width_deg: float = 20.0) -> np.ndarray:
    U, V = frequency_grid(shape)
    r = np.hypot(U, V) / 0.5            # 1 = Nyquist along an axis
    if kind == "low-pass":
        m = r <= cutoff
    elif kind == "high-pass":
        m = r >= cutoff                 # DC (r = 0) is removed unless cutoff == 0
    elif kind == "band-pass":
        lo, hi = sorted((cutoff, cutoff_hi))
        m = (r >= lo) & (r <= hi)
    elif kind == "orientation":
        ang = np.degrees(np.arctan2(V, U))
        d = np.abs((ang - angle_deg + 90) % 180 - 90)   # axial distance (spectrum is symmetric)
        m = (d <= width_deg / 2) | (r == 0)             # keep DC so the image stays visible
    else:
        raise ValueError(f"Unknown mask '{kind}'")
    return m.astype(float)


def filter_image(gray: np.ndarray, kind: str, use_hann: bool = False, restore_mean: bool = True, **kw):
    """Return (filtered image, mask, masked spectrum F*mask) via the inverse 2-D DFT."""
    F = dft2(gray, hann=use_hann)
    mask = make_mask(gray.shape, kind, **kw)
    out = idft2(F * mask)
    if restore_mean and kind in ("high-pass", "band-pass"):
        out = out + gray.mean()
    return out, mask, F * mask


def swap_magnitude_phase(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Magnitude of a with phase of b (VIP-2 demo)."""
    Fa, Fb = dft2(a, hann=False), dft2(b, hann=False)
    return idft2(np.abs(Fa) * np.exp(1j * np.angle(Fb)))

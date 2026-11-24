"""Sobel edge detection (PDF2 pp.118-128): discrete gradient with the 3x3 Sobel kernels,
gradient magnitude S = sqrt(Gx^2 + Gy^2) and orientation theta = atan2(Gy, Gx) (p.123).

Taught: the kernels, magnitude, orientation. Engineering: thresholding the magnitude to display an edge
map, and reflect border handling.
"""
from __future__ import annotations

import numpy as np

from .spatial import correlate2d

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)   # horizontal gradient dI/dx
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)   # vertical gradient dI/dy
MAX_MAG = 4 * 255 * np.sqrt(2)      # largest possible magnitude for 8-bit input


def sobel(gray: np.ndarray, border: str = "reflect"):
    """Return (Gx, Gy, magnitude, orientation in radians)."""
    g = np.asarray(gray, dtype=np.float64)
    gx = correlate2d(g, SOBEL_X, border)
    gy = correlate2d(g, SOBEL_Y, border)
    return gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)


def edge_map(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    """Binary edge map: magnitude >= threshold."""
    return (magnitude >= threshold).astype(np.float64)

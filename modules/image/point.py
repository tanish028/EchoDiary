"""Point processing g(x,y) = T(f(x,y)) (PDF2 pp.100-114: negative, log, nth power / gamma correction,
piecewise-linear contrast stretching).

Taught: the general form g = T(f), negative, log, gamma. Engineering contribution: the specific
brightness/contrast parametrisation s = contrast * (r - 128) + 128 + brightness (a linear, i.e. simple
piecewise-linear, transform) and the interactive controls.
"""
from __future__ import annotations

import numpy as np


def _clip(x):
    return np.clip(x, 0, 255)


def brightness_contrast(gray: np.ndarray, contrast: float = 1.0, brightness: float = 0.0) -> np.ndarray:
    return _clip(contrast * (np.asarray(gray, dtype=np.float64) - 128.0) + 128.0 + brightness)


def invert(gray: np.ndarray) -> np.ndarray:
    """Image negative: s = 255 - r."""
    return 255.0 - np.asarray(gray, dtype=np.float64)


def gamma_correction(gray: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """s = 255 (r/255)^gamma. gamma < 1 brightens, gamma > 1 darkens."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return 255.0 * (np.asarray(gray, dtype=np.float64) / 255.0) ** gamma


def log_transform(gray: np.ndarray) -> np.ndarray:
    """s = c log(1 + r) with c = 255 / log(256) so that 255 maps to 255; expands dark values."""
    c = 255.0 / np.log(256.0)
    return c * np.log1p(np.asarray(gray, dtype=np.float64))


OPS = ("brightness / contrast", "negative", "gamma", "log")


def apply_op(gray: np.ndarray, op: str, **p) -> np.ndarray:
    if op == "brightness / contrast":
        return brightness_contrast(gray, p.get("contrast", 1.0), p.get("brightness", 0.0))
    if op == "negative":
        return invert(gray)
    if op == "gamma":
        return gamma_correction(gray, p.get("gamma", 1.0))
    if op == "log":
        return log_transform(gray)
    raise ValueError(f"Unknown operation '{op}'")


def transfer_curve(op: str, **p):
    """(r, s) arrays of the intensity transformation s = T(r), for plotting."""
    r = np.arange(256, dtype=np.float64)
    return r, apply_op(r, op, **p)

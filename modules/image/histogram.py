"""Grayscale intensity histograms (PDF2 pp.115-117: dark / light / low-contrast / high-contrast images).

The histogram itself is taught. The plain-language description and the L1 histogram distance used for
comparison are engineering contributions.
"""
from __future__ import annotations

import numpy as np


def gray_histogram(gray: np.ndarray, bins: int = 256, normalise: bool = False) -> np.ndarray:
    """Counts of pixels per intensity bin over [0, 256). `normalise=True` -> sums to 1."""
    h, _ = np.histogram(np.clip(gray, 0, 255.999), bins=bins, range=(0, 256))
    h = h.astype(np.float64)
    if normalise:
        s = h.sum()
        h = h / s if s > 0 else h
    return h


def histogram_stats(gray: np.ndarray) -> dict:
    g = np.asarray(gray, dtype=np.float64).ravel()
    p5, p50, p95 = np.percentile(g, [5, 50, 95])
    return {"mean": float(g.mean()), "std": float(g.std()), "min": float(g.min()), "max": float(g.max()),
            "median": float(p50), "p5": float(p5), "p95": float(p95), "range_90": float(p95 - p5)}


def describe_histogram(st: dict) -> str:
    """Plain-language reading of the intensity distribution (thresholds are engineering choices)."""
    m, sd, r = st["mean"], st["std"], st["range_90"]
    tone = "predominantly dark" if m < 85 else "predominantly bright" if m > 170 else "mid-toned"
    spread = ("low contrast: the pixel values are packed into a narrow band" if r < 60
              else "high contrast: the pixel values spread over most of the range" if r > 160
              else "moderate contrast")
    clip = ""
    if st["min"] <= 2 and st["max"] >= 253:
        clip = " Pixels reach both extremes (pure black and pure white)."
    elif st["max"] >= 253:
        clip = " Some pixels are saturated at white."
    elif st["min"] <= 2:
        clip = " Some pixels are crushed to black."
    return (f"The photo is {tone} (mean {m:.0f}/255, median {st['median']:.0f}) with {spread} "
            f"(5th-95th percentile spans {st['p5']:.0f}-{st['p95']:.0f}, std {sd:.0f}).{clip}")


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Half the L1 distance between two *normalised* histograms = 1 - histogram intersection; in [0, 1]."""
    return float(0.5 * np.abs(np.asarray(h1) - np.asarray(h2)).sum())

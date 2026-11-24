"""Low-level visual similarity between photographs (Visual Memory Search).

THIS IS NOT SEMANTIC OR OBJECT RECOGNITION. Two photos are "similar" only if their pixel-intensity
distribution and edge statistics are alike.

Descriptors (all from a grayscale copy whose longer side is VISUAL_SIDE px, aspect ratio preserved):
  * normalised 32-bin grayscale intensity histogram            (histograms: taught, PDF2 pp.115-117)
  * Sobel gradient-magnitude histogram (16 bins, sqrt scale)   (Sobel/magnitude: taught, PDF2 pp.123-128)
  * Sobel orientation histogram (8 bins, magnitude-weighted)   (orientation formula taught, p.123)
  * edge density and a 3 x 3 grid of edge densities (coarse layout)           [engineering]
Distance/ranking (histogram L1, weights, combination) are ENGINEERING contributions. Whole-image
normalised cross-correlation (template matching, PDF2 pp.154-161) is deliberately NOT used: it only works
for aligned images of the same scene, not for arbitrary photographs.
"""
from __future__ import annotations

import numpy as np

from .edges import MAX_MAG, sobel
from .histogram import gray_histogram, histogram_distance
from .preprocessing import fit_gray

VISUAL_SIDE = 128
HIST_BINS, MAG_BINS, ORI_BINS, GRID = 32, 16, 8, 3
EDGE_THRESHOLD = 100.0          # Sobel magnitude (0..1442) above which a pixel counts as an edge
VERSION = 1

# component weights inside the edge distance, and default histogram/edge balance (engineering)
EDGE_PART_WEIGHTS = {"edge_strength": 0.4, "edge_direction": 0.3, "edge_layout": 0.3}
DEFAULT_WEIGHTS = {"histogram": 0.5, "edges": 0.5}


def _norm(h: np.ndarray) -> np.ndarray:
    s = h.sum()
    return h / s if s > 0 else h


def visual_descriptors(rgb: np.ndarray) -> dict:
    """JSON-serialisable descriptor dict (persisted in the database)."""
    g = fit_gray(rgb, VISUAL_SIDE)
    _, _, mag, ang = sobel(g)
    hist = gray_histogram(g, HIST_BINS, normalise=True)
    mag_hist = _norm(np.histogram(np.sqrt(np.clip(mag / MAX_MAG, 0, 1)), bins=MAG_BINS, range=(0, 1))[0].astype(float))
    idx = np.minimum(((ang + np.pi) / (2 * np.pi) * ORI_BINS).astype(int), ORI_BINS - 1)   # 0..7
    ori = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=ORI_BINS).astype(float)
    ori_hist = _norm(ori)
    edges = mag >= EDGE_THRESHOLD
    H, W = g.shape
    grid = [float(edges[int(i * H / GRID):int((i + 1) * H / GRID), int(j * W / GRID):int((j + 1) * W / GRID)].mean())
            for i in range(GRID) for j in range(GRID)]
    return {"version": VERSION, "side": VISUAL_SIDE,
            "hist": hist.round(6).tolist(), "mag_hist": mag_hist.round(6).tolist(),
            "ori_hist": ori_hist.round(6).tolist(), "edge_density": float(edges.mean()),
            "mean_gradient": float(mag.mean()), "grid": [round(v, 6) for v in grid]}


def component_distances(a: dict, b: dict) -> dict:
    """Four distances, each bounded in [0, 1] by construction (this is the normalisation step)."""
    d_hist = histogram_distance(np.array(a["hist"]), np.array(b["hist"]))
    d_mag = histogram_distance(np.array(a["mag_hist"]), np.array(b["mag_hist"]))
    oa, ob = np.array(a["ori_hist"]), np.array(b["ori_hist"])
    d_ori = 0.0 if (oa.sum() == 0 and ob.sum() == 0) else histogram_distance(oa, ob) if (oa.sum() and ob.sum()) else 1.0
    d_grid = float(np.abs(np.array(a["grid"]) - np.array(b["grid"])).mean())
    return {"histogram": d_hist, "edge_strength": d_mag, "edge_direction": d_ori, "edge_layout": d_grid}


def combine(dists: dict, weights: dict | None = None) -> dict:
    """similarity = 1 - (w_h * d_hist + w_e * d_edges), weights renormalised to sum to 1."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    tot = w.get("histogram", 0) + w.get("edges", 0)
    if tot <= 0:
        raise ValueError("At least one weight must be positive")
    wh, we = w.get("histogram", 0) / tot, w.get("edges", 0) / tot
    d_edges = sum(EDGE_PART_WEIGHTS[k] * dists[k] for k in EDGE_PART_WEIGHTS)
    d = wh * dists["histogram"] + we * d_edges
    return {"similarity": float(1.0 - d), "distance": float(d), "edge_distance": float(d_edges)}


def rank(reference: dict, candidates: list[tuple[int, dict]], weights: dict | None = None) -> list[dict]:
    """candidates = [(entry_id, descriptor)]. Best match first; ties broken by entry id (stable)."""
    out = []
    for eid, desc in candidates:
        d = component_distances(reference, desc)
        out.append({"id": eid, "components": d, **combine(d, weights)})
    return sorted(out, key=lambda r: (-r["similarity"], r["id"]))

"""Multimodal memory representation and cross-entry retrieval.

ENGINEERING DESIGN DECISION (not a method taught in DA321): each modality is turned into its own
representation, z-scored over the diary corpus where numeric, compared with cosine similarity
(the vector-space notion from PDF1/PDF2), then combined by a user-weighted average of the
per-modality similarities (late fusion); the image part uses the visual-search similarity (intensity histogram + Sobel edge statistics)
when both entries have persisted descriptors. Scores are similarity measures, NOT validated measures
of emotion or memory quality.
"""
from __future__ import annotations

import math

import numpy as np

from modules.image import visual
from modules.text.retrieval import cosine
from modules.text.tfidf import corpus_tfidf

AUDIO_KEYS = ["duration_s", "rms", "energy_mean", "zcr_mean", "centroid_hz", "active_fraction",
              "f0_hz", "periodicity"]
IMAGE_KEYS = ["mean_r", "mean_g", "mean_b", "brightness", "contrast", "low_freq_ratio",
              "high_freq_ratio", "spectral_centroid", "orientation_balance", "edge_density", "mean_gradient"]
LOG_KEYS = {"duration_s", "rms", "energy_mean"}      # heavy-tailed -> log scale before z-score


def _vec(feats: dict | None, keys: list[str]):
    if not feats:
        return None
    v = []
    for k in keys:
        x = float(feats.get(k, 0.0) or 0.0)
        v.append(math.log10(x + 1e-9) if k in LOG_KEYS else x)
    return np.array(v)


def _zscore_matrix(rows: list[np.ndarray | None]):
    present = [r for r in rows if r is not None]
    if not present:
        return [None] * len(rows)
    M = np.vstack(present)
    mu = M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-9] = 1.0
    return [None if r is None else (r - mu) / sd for r in rows]


def _cos_dense(a, b) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(a @ b / (na * nb))


def build_profiles(entries: list[dict]) -> list[dict]:
    """entries: dicts with 'transcript', 'audio_features', 'image_features'."""
    A = _zscore_matrix([_vec(e.get("audio_features"), AUDIO_KEYS) for e in entries])
    I = _zscore_matrix([_vec(e.get("image_features"), IMAGE_KEYS) for e in entries])
    _, _, T = corpus_tfidf([e.get("transcript") or "" for e in entries])
    return [{"audio": a, "text": t, "image": i} for a, t, i in zip(A, T, I)]


def similar_entries(target_idx: int, entries: list[dict], weights=(1.0, 1.0, 1.0), top_k=5):
    """Rank other entries by weighted per-modality similarity to entries[target_idx].
    Modalities missing for either entry are excluded and weights renormalised."""
    if len(entries) < 2:
        return []
    P = build_profiles(entries)
    wa, wt, wi = weights
    out = []
    for j, p in enumerate(P):
        if j == target_idx:
            continue
        t = P[target_idx]
        parts = {}
        if t["audio"] is not None and p["audio"] is not None:
            parts["audio"] = (wa, _cos_dense(t["audio"], p["audio"]))
        if t["text"] and p["text"]:
            parts["text"] = (wt, cosine(t["text"], p["text"]))
        tv, pv = entries[target_idx].get("visual_features"), entries[j].get("visual_features")
        if tv and pv:
            # image part = low-level visual similarity (histogram + Sobel edge statistics, 0..1), the same
            # measure as Visual Search (engineering combination of taught descriptors)
            parts["image"] = (wi, visual.combine(visual.component_distances(tv, pv))["similarity"])
        elif t["image"] is not None and p["image"] is not None:
            parts["image"] = (wi, _cos_dense(t["image"], p["image"]))
        wsum = sum(w for w, _ in parts.values())
        if wsum <= 0:
            continue
        score = sum(w * s for w, s in parts.values()) / wsum
        out.append({"index": j, "score": score, "parts": {k: s for k, (_, s) in parts.items()}})
    return sorted(out, key=lambda d: -d["score"])[:top_k]


def mood_summary(entries: list[dict]) -> list[dict]:
    """Mean of measured characteristics per user-selected mood (descriptive statistics only)."""
    by: dict[str, list[dict]] = {}
    for e in entries:
        by.setdefault(e["mood"], []).append(e)
    rows = []
    for mood, es in by.items():
        af = [e["audio_features"] for e in es if e.get("audio_features")]
        imf = [e["image_features"] for e in es if e.get("image_features")]
        n_tok = [len((e.get("transcript") or "").split()) for e in es]
        rows.append({
            "Mood (user-selected)": mood, "Entries": len(es),
            "Mean energy": float(np.mean([a["energy_mean"] for a in af])) if af else None,
            "Mean ZCR": float(np.mean([a["zcr_mean"] for a in af])) if af else None,
            "Mean spectral centroid (Hz)": float(np.mean([a["centroid_hz"] for a in af])) if af else None,
            "Mean words": float(np.mean(n_tok)) if n_tok else None,
            "Mean image brightness": float(np.mean([i["brightness"] for i in imf])) if imf else None,
        })
    return sorted(rows, key=lambda r: -r["Entries"])

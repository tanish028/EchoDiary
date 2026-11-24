"""Framing and windowing (SAP-1: STFT with frame length L, hop H).

Windows (SAP-1): Hann w[n]=0.5-0.5cos(2*pi*n/L), Hamming w[n]=0.54-0.46cos(2*pi*n/L),
0 <= n < L; rectangular w[n]=1.
"""
from __future__ import annotations

import numpy as np


def make_window(name: str, L: int) -> np.ndarray:
    n = np.arange(L)
    name = name.lower()
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / L)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / L)
    if name in ("rect", "rectangular"):
        return np.ones(L)
    raise ValueError(f"Unknown window '{name}'")


def frame_params(fs: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> tuple[int, int]:
    """Defaults 25 ms / 10 ms follow the spectrogram figure in PDF1 (engineering default)."""
    L = max(8, int(round(fs * frame_ms / 1000)))
    H = max(1, int(round(fs * hop_ms / 1000)))
    return L, H


def frame_signal(x: np.ndarray, L: int, H: int) -> np.ndarray:
    """Split x into overlapping frames -> array (n_frames, L). Zero-pads the tail."""
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)))
    n_frames = 1 + int(np.ceil((len(x) - L) / H))
    total = (n_frames - 1) * H + L
    xp = np.pad(x, (0, total - len(x)))
    idx = np.arange(L)[None, :] + H * np.arange(n_frames)[:, None]
    return xp[idx]


def window_frames(frames: np.ndarray, window: str = "hann") -> np.ndarray:
    return frames * make_window(window, frames.shape[1])[None, :]

"""Frame-based audio measurements.

Taught:    autocorrelation r(k) = sum x[n]x[n+k] / sum x[n]^2   (SAP-4)
Syllabus-listed, no teaching material found (standard textbook definitions, documented):
  short-time energy  E[m] = sum_n (x[n] w[n])^2  /  L   (mean-square per frame, Hann window)
  zero-crossing rate Z[m] = (1/(L-1)) * sum_n 1[ sign(x[n]) != sign(x[n-1]) ]
Windowing policy (documented engineering choice): spectrogram and energy use the Hann window (SAP-1);
ZCR is computed on the raw samples (a window only rescales samples and cannot create or remove sign
changes); autocorrelation uses the raw 40 ms segment, exactly as in the SAP-4 formula.
The IEEE 1386250 abstract uses RMS and zero-crossings for speech/music discrimination;
that classifier is NOT reproduced here.
"""
from __future__ import annotations

import numpy as np

from .framing import frame_params, frame_signal, window_frames
from .stft import stft


def short_time_energy(x: np.ndarray, fs: int, frame_ms: float = 25, hop_ms: float = 10,
                      window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    L, H = frame_params(fs, frame_ms, hop_ms)
    fr = window_frames(frame_signal(x, L, H), window)
    e = np.sum(fr ** 2, axis=1) / L
    t = (np.arange(len(e)) * H + L / 2) / fs
    return t, e


def zero_crossing_rate(x: np.ndarray, fs: int, frame_ms: float = 25, hop_ms: float = 10):
    L, H = frame_params(fs, frame_ms, hop_ms)
    fr = frame_signal(x, L, H)
    s = np.signbit(fr)  # sign without window (windowing does not move zero crossings)
    z = np.sum(s[:, 1:] != s[:, :-1], axis=1) / (L - 1)
    t = (np.arange(len(z)) * H + L / 2) / fs
    return t, z


def autocorrelation(seg: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation r(k), r(0)=1 (SAP-4). Zero-energy segment -> zeros."""
    seg = np.asarray(seg, dtype=float)
    n = len(seg)
    max_lag = min(max_lag if max_lag is not None else n - 1, n - 1)
    r0 = float(np.sum(seg ** 2))
    if r0 <= 1e-12:
        return np.zeros(max_lag + 1)
    full = np.correlate(seg, seg, mode="full")[n - 1:n + max_lag]
    return full / r0


def loudest_segment(x: np.ndarray, fs: int, dur_ms: float = 40.0) -> tuple[int, np.ndarray]:
    """Pick the highest-energy window as the default analysis segment (engineering choice)."""
    L = int(fs * dur_ms / 1000)
    if len(x) <= L:
        return 0, x
    cs = np.cumsum(np.insert(x ** 2, 0, 0.0))
    energy = cs[L:] - cs[:-L]
    s = int(np.argmax(energy))
    return s, x[s:s + L]


def pitch_from_autocorr(r: np.ndarray, fs: int, fmin: float = 70, fmax: float = 400):
    """Peak-pick r(k) in the lag range of [fmin, fmax] (peak-picking is an engineering extension
    of SAP-4's autocorrelation; pitch ranges 85-255 Hz are quoted on SAP-3)."""
    lo, hi = int(fs / fmax), min(int(fs / fmin), len(r) - 1)
    if hi <= lo:
        return None, 0.0
    k = lo + int(np.argmax(r[lo:hi + 1]))
    return fs / k, float(r[k])


def audio_feature_vector(x: np.ndarray, fs: int, original_duration_s: float | None = None) -> dict:
    """Compact per-entry descriptor stored in the database. `x` is the (possibly truncated) analysis
    signal; `original_duration_s` is the true length of the stored file."""
    _, e = short_time_energy(x, fs)
    _, z = zero_crossing_rate(x, fs)
    sp = stft(x, fs)
    p = np.maximum(np.abs(sp.X), 1e-12)
    centroid_t = (p * sp.freqs[None, :]).sum(1) / p.sum(1)
    active = e > 0.1 * np.max(e) if np.max(e) > 0 else np.zeros_like(e, dtype=bool)
    _, seg = loudest_segment(x, fs, 40.0)
    r = autocorrelation(seg, max_lag=int(fs * 0.025))
    f0, strength = pitch_from_autocorr(r, fs)
    return {
        "duration_s": round(original_duration_s if original_duration_s is not None else len(x) / fs, 3),
        "analysed_duration_s": round(len(x) / fs, 3),
        "truncated": bool(original_duration_s is not None and original_duration_s > len(x) / fs + 1e-6),
        "fs": int(fs),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "energy_mean": float(np.mean(e)),
        "energy_std": float(np.std(e)),
        "zcr_mean": float(np.mean(z)),
        "zcr_std": float(np.std(z)),
        "centroid_hz": float(np.mean(centroid_t[active]) if active.any() else np.mean(centroid_t)),
        "active_fraction": float(np.mean(active)),
        "f0_hz": float(f0) if f0 else 0.0,
        "periodicity": float(strength),
    }

"""STFT: Audio -> framing -> windowing -> DFT -> magnitude -> spectrogram (SAP-1, PDF1 pp.435-439)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .framing import frame_params, frame_signal, window_frames


@dataclass
class Spectrogram:
    X: np.ndarray        # complex (n_frames, N//2+1)
    mag_db: np.ndarray   # 20 log10|X|  (SAP-1 dB scaling)
    times: np.ndarray    # frame centre times (s)
    freqs: np.ndarray    # Hz
    L: int
    H: int
    N: int


def stft(x: np.ndarray, fs: int, frame_ms: float = 25.0, hop_ms: float = 10.0,
         window: str = "hann", n_fft: int | None = None) -> Spectrogram:
    L, H = frame_params(fs, frame_ms, hop_ms)
    N = n_fft or int(2 ** np.ceil(np.log2(L)))
    N = max(N, L)
    frames = window_frames(frame_signal(x, L, H), window)
    X = np.fft.rfft(frames, n=N, axis=1)            # FFT = fast DFT (PDF1 pp.389-425)
    mag = np.abs(X)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-10))
    times = (np.arange(X.shape[0]) * H + L / 2) / fs
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    return Spectrogram(X, mag_db, times, freqs, L, H, N)


def single_frame_spectrum(x: np.ndarray, fs: int, t: float, frame_ms: float = 25.0,
                          window: str = "hann"):
    """Magnitude spectrum (dB) of the frame starting at time t (frequency-domain view)."""
    L, _ = frame_params(fs, frame_ms)
    s = int(np.clip(t * fs, 0, max(0, len(x) - L)))
    seg = x[s:s + L]
    seg = np.pad(seg, (0, L - len(seg)))
    N = int(2 ** np.ceil(np.log2(L)))
    w = window_frames(seg[None, :], window)[0]
    mag = np.abs(np.fft.rfft(w, n=N))
    return np.fft.rfftfreq(N, 1 / fs), 20 * np.log10(np.maximum(mag, 1e-10))

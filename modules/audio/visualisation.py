"""Plotly figures for the audio pipeline."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from modules.viz_style import ACCENT, ACCENT2, PINK, style

from .features import autocorrelation, short_time_energy, zero_crossing_rate
from .stft import Spectrogram, single_frame_spectrum


def _decimate(x: np.ndarray, n: int = 4000):
    """Min/max envelope decimation so long files plot quickly."""
    if len(x) <= n * 2:
        return np.arange(len(x)), x
    k = len(x) // n
    xs = x[: k * n].reshape(n, k)
    idx = (np.arange(n) * k)
    lo, hi = xs.min(1), xs.max(1)
    return np.repeat(idx, 2), np.column_stack([lo, hi]).ravel()


def waveform_fig(x, fs):
    idx, y = _decimate(x)
    fig = go.Figure(go.Scatter(x=idx / fs, y=y, mode="lines", line=dict(color=ACCENT, width=1)))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Amplitude")
    return style(fig, "Waveform")


def downsample_spectrogram(sp: Spectrogram, max_hz: float | None = None, max_cols: int = 800):
    """Display-only reduction of the data sent to the browser: keep rows <= max_hz and average the dB values
    of consecutive frames so at most `max_cols` time columns remain (float32). The analysis itself is untouched."""
    keep = sp.freqs <= (max_hz or sp.freqs[-1])
    z = sp.mag_db[:, keep]                          # (frames, bins)
    t = sp.times
    n = z.shape[0]
    if n > max_cols:
        k = int(np.ceil(n / max_cols))
        m = (n // k) * k
        z = z[:m].reshape(m // k, k, -1).mean(axis=1)
        t = t[:m].reshape(m // k, k).mean(axis=1)
    return z.T.astype(np.float32).round(1), t, sp.freqs[keep]


def spectrogram_fig(sp: Spectrogram, dyn_range_db: float = 70, max_hz: float | None = None,
                    max_cols: int = 800):
    z, t, f = downsample_spectrogram(sp, max_hz, max_cols)
    zmax = float(sp.mag_db.max())
    fig = go.Figure(go.Heatmap(z=z, x=t, y=f, colorscale="Magma", zmin=zmax - dyn_range_db, zmax=zmax,
                               colorbar=dict(title="dB")))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Frequency (Hz)", range=[0, max_hz or sp.freqs[-1]])
    return style(fig, f"Spectrogram (L={sp.L}, H={sp.H}, N={sp.N})", 340)


def frame_spectrum_fig(x, fs, t, window="hann"):
    f, db = single_frame_spectrum(x, fs, t, window=window)
    fig = go.Figure(go.Scatter(x=f, y=db, mode="lines", line=dict(color=ACCENT2)))
    fig.update_xaxes(title="Frequency (Hz)")
    fig.update_yaxes(title="Magnitude (dB)")
    return style(fig, f"Magnitude spectrum of one {window} frame at t={t:.2f}s")


def energy_zcr_fig(x, fs):
    t, e = short_time_energy(x, fs)
    t2, z = zero_crossing_rate(x, fs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e, name="Short-time energy", line=dict(color=ACCENT)))
    fig.add_trace(go.Scatter(x=t2, y=z, name="ZCR", yaxis="y2", line=dict(color=PINK)))
    fig.update_layout(yaxis=dict(title="Energy"),
                      yaxis2=dict(title="ZCR (crossings/sample)", overlaying="y", side="right",
                                  showgrid=False),
                      xaxis=dict(title="Time (s)"), legend=dict(orientation="h", y=1.12))
    return style(fig, "Short-time energy and zero-crossing rate", 300)


def autocorr_fig(seg, fs, max_lag_ms: float = 25.0, f0=None):
    r = autocorrelation(seg, max_lag=int(fs * max_lag_ms / 1000))
    lags_ms = np.arange(len(r)) / fs * 1000
    fig = go.Figure(go.Scatter(x=lags_ms, y=r, mode="lines", line=dict(color=ACCENT2)))
    if f0:
        fig.add_vline(x=1000 / f0, line=dict(color=PINK, dash="dash"),
                      annotation_text=f"peak ≈ {f0:.0f} Hz")
    fig.update_xaxes(title="Lag k (ms)")
    fig.update_yaxes(title="r(k)", range=[-1.05, 1.05])
    return style(fig, "Normalised autocorrelation r(k) of the loudest 40 ms")

"""Audio loading and validation (WAV only; engineering decision).

The ORIGINAL file is always stored untouched. Analysis works on at most MAX_AUDIO_SECONDS of it; the
returned AudioData keeps both durations so the UI never reports the truncated length as the original one.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from scipy.io import wavfile

from modules.config import MAX_AUDIO_SECONDS, get_logger

log = get_logger("audio.pre")


class AudioError(ValueError):
    """Raised for unreadable or unusable audio (message is shown to the user)."""


@dataclass
class AudioData:
    x: np.ndarray                 # mono float signal in [-1, 1] (possibly truncated for analysis)
    fs: int
    original_duration_s: float    # duration of the stored file
    analysed_duration_s: float    # duration actually analysed (<= MAX_AUDIO_SECONDS)

    @property
    def truncated(self) -> bool:
        return self.analysed_duration_s < self.original_duration_s - 1e-6


def _decode(src) -> Tuple[np.ndarray, int]:
    try:
        if isinstance(src, (bytes, bytearray)):
            src = io.BytesIO(src)
        fs, x = wavfile.read(src)
    except Exception as exc:  # scipy raises ValueError / struct.error for non-WAV or unsupported encodings
        raise AudioError("Could not read this file as WAV audio (it may be corrupted, an unsupported "
                         "encoding, or not a .wav file). Please use a PCM .wav file or the recorder.") from exc
    if x.size == 0:
        raise AudioError("The audio file is empty.")
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max + 1)
    elif x.dtype.kind == "u":  # 8-bit PCM
        x = (x.astype(np.float64) - 128.0) / 128.0
    elif x.dtype.kind == "f":
        x = x.astype(np.float64)
    else:
        raise AudioError(f"Unsupported WAV sample type '{x.dtype}'.")
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.nan_to_num(x)
    if fs < 4000:
        raise AudioError(f"Sampling rate {fs} Hz is too low for analysis (need >= 4 kHz).")
    if x.size < int(0.05 * fs):
        raise AudioError("Audio is shorter than 50 ms; please record a longer clip.")
    return x, int(fs)


def load_wav_full(src: Union[str, bytes, "io.BytesIO"]) -> AudioData:
    x, fs = _decode(src)
    original = len(x) / fs
    max_n = int(MAX_AUDIO_SECONDS * fs)
    if len(x) > max_n:
        log.info("Audio of %.0f s truncated to %d s for analysis (original file is kept)", original,
                 MAX_AUDIO_SECONDS)
        x = x[:max_n]
    return AudioData(x, fs, original, len(x) / fs)


def load_wav(src) -> Tuple[np.ndarray, int]:
    """(analysis signal, fs). Use load_wav_full when the original duration matters."""
    d = load_wav_full(src)
    return d.x, d.fs


def validate(x: np.ndarray, fs: int) -> dict:
    """Basic sanity report used by the UI."""
    peak = float(np.max(np.abs(x)))
    return {
        "duration_s": len(x) / fs,
        "fs": fs,
        "peak": peak,
        "is_silent": peak < 1e-4,
        "is_clipped": bool(np.mean(np.abs(x) > 0.999) > 0.01),
    }

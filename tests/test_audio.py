import io

import numpy as np
import pytest
from scipy.io import wavfile

from modules.audio import preprocessing as pre
from modules.audio.features import (audio_feature_vector, autocorrelation, pitch_from_autocorr,
                                    short_time_energy, zero_crossing_rate)
from modules.audio.framing import frame_params, frame_signal, make_window, window_frames
from modules.audio.preprocessing import AudioError, load_wav, load_wav_full
from modules.audio.stft import stft
from modules.audio.visualisation import downsample_spectrogram, spectrogram_fig

from conftest import FS, make_wav


def tone(f, dur=1.0, amp=0.5):
    t = np.arange(int(FS * dur)) / FS
    return amp * np.sin(2 * np.pi * f * t)


def test_window_formulas():
    L = 8
    assert np.allclose(make_window("hann", L)[0], 0.0)
    assert np.isclose(make_window("hann", L)[4], 1.0)
    assert np.isclose(make_window("hamming", L)[0], 0.08)
    assert np.all(make_window("rect", L) == 1)
    with pytest.raises(ValueError):
        make_window("blackman", L)


def test_framing_shape_and_content():
    L, H = frame_params(FS)
    assert (L, H) == (400, 160)
    x = np.arange(2000.0)
    fr = frame_signal(x, L, H)
    assert fr.shape[1] == L and np.array_equal(fr[1], x[H:H + L])
    assert frame_signal(np.ones(10), L, H).shape == (1, L)         # shorter than one frame
    assert window_frames(fr, "hann").shape == fr.shape


def test_stft_peak_at_tone_frequency_and_shapes():
    sp = stft(tone(1000), FS)
    k = np.argmax(sp.mag_db.mean(0))
    assert abs(sp.freqs[k] - 1000) < FS / sp.N * 1.5
    assert sp.X.shape == (len(sp.times), sp.N // 2 + 1) and sp.N >= sp.L


def test_zcr_of_sine():
    _, z = zero_crossing_rate(tone(1000), FS)
    assert np.isclose(np.mean(z), 2 * 1000 / FS, atol=0.01)


def test_energy_scales_with_amplitude_and_default_is_hann():
    _, e1 = short_time_energy(tone(300, amp=0.2), FS)
    _, e2 = short_time_energy(tone(300, amp=0.4), FS)
    assert np.isclose(e2.mean() / e1.mean(), 4.0, rtol=0.02)
    _, d = short_time_energy(tone(300), FS)
    _, h = short_time_energy(tone(300), FS, window="hann")
    _, hm = short_time_energy(tone(300), FS, window="hamming")
    assert np.allclose(d, h) and not np.allclose(d, hm)


def test_autocorr_normalised_and_pitch():
    r = autocorrelation(tone(200, 0.04)[:640], max_lag=300)
    assert np.isclose(r[0], 1.0)
    f0, s = pitch_from_autocorr(r, FS)
    assert abs(f0 - 200) < 10 and s > 0.5


def test_silence_does_not_crash():
    x = np.zeros(FS)
    assert np.all(autocorrelation(x[:400]) == 0)
    fv = audio_feature_vector(x, FS)
    assert fv["rms"] == 0.0 and np.isfinite([v for v in fv.values() if isinstance(v, float)]).all()


@pytest.mark.parametrize("dtype", [np.int16, np.uint8, np.float32, np.int32])
def test_supported_wav_encodings(dtype):
    buf = io.BytesIO()
    x = tone(300)
    arr = {np.int16: x * 32767, np.uint8: x * 127 + 128, np.float32: x, np.int32: x * 2 ** 31 * 0.99}[dtype]
    wavfile.write(buf, FS, arr.astype(dtype))
    y, fs = load_wav(buf.getvalue())
    assert fs == FS and np.isclose(np.abs(y).max(), 0.5, atol=0.02)


def test_stereo_is_mixed_down():
    buf = io.BytesIO()
    wavfile.write(buf, FS, (np.stack([tone(300), tone(300)], 1) * 32767).astype(np.int16))
    x, fs = load_wav(buf.getvalue())
    assert x.ndim == 1 and fs == FS


@pytest.mark.parametrize("bad", [b"", b"not a wav", b"RIFF" + b"\x00" * 40, b"ID3\x03" + bytes(range(200))])
def test_invalid_audio_gives_readable_error(bad):
    with pytest.raises(AudioError) as ei:
        load_wav(bad)
    assert "WAV" in str(ei.value) or "empty" in str(ei.value)


def test_too_short_and_low_rate_rejected():
    buf = io.BytesIO(); wavfile.write(buf, FS, np.zeros(10, dtype=np.int16))
    with pytest.raises(AudioError):
        load_wav(buf.getvalue())
    buf = io.BytesIO(); wavfile.write(buf, 1000, np.zeros(4000, dtype=np.int16))
    with pytest.raises(AudioError):
        load_wav(buf.getvalue())


def test_truncation_keeps_original_duration(monkeypatch):
    monkeypatch.setattr(pre, "MAX_AUDIO_SECONDS", 2)
    data = make_wav(dur=5.0)
    ad = load_wav_full(data)
    assert ad.truncated and np.isclose(ad.original_duration_s, 5.0) and np.isclose(ad.analysed_duration_s, 2.0)
    fv = audio_feature_vector(ad.x, ad.fs, ad.original_duration_s)
    assert fv["duration_s"] == 5.0 and fv["analysed_duration_s"] == 2.0 and fv["truncated"] is True
    short = load_wav_full(make_wav(dur=1.0))
    assert not short.truncated
    assert audio_feature_vector(short.x, short.fs, short.original_duration_s)["truncated"] is False


def test_spectrogram_downsampling_limits_payload():
    x = tone(440, dur=60)
    sp = stft(x, FS)
    assert sp.mag_db.shape[0] > 5000
    z, t, f = downsample_spectrogram(sp, max_hz=4000, max_cols=800)
    assert z.shape[1] <= 800 and z.shape[0] == len(f) and f.max() <= 4000 and z.dtype == np.float32
    assert len(t) == z.shape[1]
    # the strongest band is still the 440 Hz tone after averaging
    assert abs(f[np.argmax(z.mean(1))] - 440) < 60
    spectrogram_fig(sp, 70, 4000)                                  # builds without error
    z2, _, _ = downsample_spectrogram(stft(tone(440, 1.0), FS), max_cols=800)
    assert z2.shape[1] == stft(tone(440, 1.0), FS).mag_db.shape[0]  # short clips are not reduced

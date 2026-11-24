import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FS = 16000


def make_wav(freq=200.0, dur=1.0, amp=0.5, fs=FS, dtype=np.int16) -> bytes:
    t = np.arange(int(fs * dur)) / fs
    x = amp * np.sin(2 * np.pi * freq * t)
    if dtype == np.int16:
        x = x * 32767
    elif dtype == np.uint8:
        x = x * 127 + 128
    buf = io.BytesIO()
    wavfile.write(buf, fs, x.astype(dtype))
    return buf.getvalue()


def make_image(kind="noise", size=(64, 48), seed=0, fmt="PNG") -> bytes:
    """size = (width, height). Deterministic synthetic photos with clearly different low-level statistics."""
    w, h = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    if kind == "noise":
        a = rng.random((h, w, 3)) * 255
    elif kind == "vstripes":
        v = 128 + 120 * np.sign(np.sin(2 * np.pi * 8 * x / w))
        a = np.stack([v, v, v], -1)
    elif kind == "hstripes":
        v = 128 + 120 * np.sign(np.sin(2 * np.pi * 8 * y / h))
        a = np.stack([v, v, v], -1)
    elif kind == "dark":
        a = np.full((h, w, 3), 30.0) + rng.random((h, w, 3)) * 5
    elif kind == "bright":
        a = np.full((h, w, 3), 230.0) + rng.random((h, w, 3)) * 5
    elif kind == "gradient":
        v = 255 * x / max(w - 1, 1)
        a = np.stack([v, v * 0.5, 255 - v], -1)
    else:
        raise ValueError(kind)
    buf = io.BytesIO()
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if fmt == "JPEG":
        im = im.convert("RGB")
    im.save(buf, fmt)
    return buf.getvalue()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Fresh, empty data directory + initialised database for every test."""
    from modules import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(config, "IMAGE_DIR", tmp_path / "images")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "echodiary.db")
    from modules.database import db
    db.init_db()
    return db


@pytest.fixture()
def media():
    class M:
        wav = staticmethod(make_wav)
        img = staticmethod(make_image)
    return M

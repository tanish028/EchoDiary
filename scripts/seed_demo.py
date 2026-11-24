"""Create clearly-labelled DEMONSTRATION entries from synthetic audio/images (no real data).
Usage: python scripts/seed_demo.py     (also available from the Dashboard when the diary is empty)"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from scipy.io import wavfile

from modules.database import db
from modules.pipeline import create_memory

FS = 16000


def synth_voice(f0: float, dur: float, vibrato: float, noise: float, seed: int) -> bytes:
    """Synthetic 'voice-like' tone: harmonic series with amplitude envelope (demo only)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * dur)) / FS
    inst = f0 * (1 + 0.03 * np.sin(2 * np.pi * vibrato * t))
    phase = 2 * np.pi * np.cumsum(inst) / FS
    x = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = (np.sin(np.pi * t / dur) ** 0.5) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t) ** 2)
    x = x * env + noise * rng.standard_normal(len(t))
    x = 0.6 * x / np.max(np.abs(x))
    buf = io.BytesIO()
    wavfile.write(buf, FS, (x * 32767).astype(np.int16))
    return buf.getvalue()


def synth_image(kind: str, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    n = 256
    y, x = np.mgrid[0:n, 0:n] / n
    if kind == "sunset":
        r = 255 * (0.9 - 0.4 * y); g = 255 * (0.5 - 0.3 * y); b = 255 * (0.3 + 0.5 * y)
        img = np.stack([r, g, b], -1) + 8 * rng.standard_normal((n, n, 3))
    elif kind == "stripes":
        v = 0.5 + 0.5 * np.sin(2 * np.pi * 12 * x)
        img = np.stack([60 + 120 * v, 90 + 100 * v, 160 + 80 * v], -1)
    elif kind == "dots":
        v = 0.5 + 0.5 * np.sin(2 * np.pi * 6 * x) * np.sin(2 * np.pi * 6 * y)
        img = np.stack([200 * v, 220 * v, 120 + 60 * v], -1)
    else:  # dusk
        v = np.exp(-((x - .5) ** 2 + (y - .6) ** 2) / 0.05)
        img = np.stack([40 + 60 * v, 40 + 40 * v, 90 + 120 * v], -1) + 5 * rng.standard_normal((n, n, 3))
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


DEMO = [
    ("[DEMO] Morning walk", "Happy", "2026-11-02", 210, 4.0, 0.02, "sunset",
     "Went for a long walk this morning. The sunrise over the lake was beautiful and I felt happy and light. I want to remember this walk."),
    ("[DEMO] Exam stress", "Anxious", "2026-11-05", 130, 6.5, 0.08, "stripes",
     "Too many exams this week. I could not sleep and I keep thinking about the exam tomorrow. I need to study more and stay calm."),
    ("[DEMO] Dinner with friends", "Grateful", "2026-11-09", 180, 3.0, 0.03, "dots",
     "Dinner with my friends tonight. We laughed a lot and cooked together. I am grateful for these friends and this evening."),
    ("[DEMO] Quiet evening", "Calm", "2026-11-14", 120, 2.0, 0.01, "dusk",
     "A quiet evening at home. I read a book and listened to the rain. Nothing special, just a calm walk through the evening."),
    ("[DEMO] Project deadline", "Tired", "2026-11-20", 100, 5.0, 0.06, "stripes",
     "Finished the project report very late at night. I am tired but proud. The deadline is tomorrow and the exam is over."),
]


def seed() -> int:
    db.init_db()
    n = 0
    for i, (title, mood, d, f0, vib, noise, img, text) in enumerate(DEMO):
        create_memory(title, text, mood, d, synth_voice(f0, 4.0, vib, noise, i), synth_image(img, i),
                      notes="Synthetic demonstration data (generated tone + generated image).",
                      is_demo=True)
        n += 1
    return n


if __name__ == "__main__":
    print(f"Seeded {seed()} demo entries")

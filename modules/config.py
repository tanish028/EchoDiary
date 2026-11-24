"""Central configuration and logging for EchoDiary."""
from __future__ import annotations

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ECHODIARY_DATA", ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio"
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "echodiary.db"

MOODS = ["Happy", "Calm", "Excited", "Grateful", "Neutral", "Tired", "Anxious", "Sad", "Angry"]
MOOD_COLORS = {
    "Happy": "#fbbf24", "Calm": "#38bdf8", "Excited": "#f472b6", "Grateful": "#34d399",
    "Neutral": "#94a3b8", "Tired": "#a78bfa", "Anxious": "#fb923c", "Sad": "#6366f1",
    "Angry": "#f87171",
}

MAX_AUDIO_SECONDS = 180      # analysis cap (engineering decision)
MAX_IMAGE_SIDE = 512         # FFT size cap (VIP-2 demo uses 256 / 512)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def ensure_dirs() -> None:
    for d in (DATA_DIR, AUDIO_DIR, IMAGE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("echodiary")
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(h)
        root.setLevel(logging.INFO)
    return root.getChild(name)

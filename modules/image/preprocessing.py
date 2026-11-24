"""Image loading, RGB channels, grayscale (PDF2 pp.39-44: image = M x N x C numpy array,
grayscale = im.mean(axis=2))."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from modules.config import MAX_IMAGE_SIDE


class ImageError(ValueError):
    pass


def load_image(src, max_side: int = MAX_IMAGE_SIDE * 2) -> np.ndarray:
    """Load path / bytes / file-like -> uint8 RGB array (M, N, 3). Large images are downscaled."""
    try:
        if isinstance(src, (bytes, bytearray)):
            src = io.BytesIO(src)
        im = Image.open(src)
        im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError("Could not read this file as an image (use JPG, PNG, ...).") from exc
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    return np.asarray(im, dtype=np.uint8)


def channels(rgb: np.ndarray):
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Average of the channels, as in the slides (float64, range 0..255)."""
    return rgb.astype(np.float64).mean(axis=2)


def fit_gray(rgb: np.ndarray, max_side: int = 256) -> np.ndarray:
    """Grayscale resized so that the LONGER side equals `max_side`, keeping the aspect ratio
    (bilinear). The 2-D DFT is then computed on this M x N array directly (no square padding and
    no stretching). Resizing is an engineering step, not a taught method."""
    g = to_grayscale(rgb)
    h, w = g.shape
    scale = max_side / max(h, w)
    nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
    im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64)

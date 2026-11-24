"""High-level processing used by the UI and the demo seeder: validate + analyse + store, edit, backfill.

Safety rules (audit fixes):
  * every modality is validated/analysed BEFORE any file or row is written;
  * new media files are written first, the database transaction second; if the database step fails the new
    files are deleted again, so no orphans are left;
  * on edit, old media files are removed only AFTER the database commit succeeded.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from modules import config
from modules.audio.features import audio_feature_vector
from modules.audio.preprocessing import load_wav_full
from modules.database import db
from modules.image.fourier import image_descriptors
from modules.image.preprocessing import ImageError, fit_gray, load_image
from modules.image.visual import visual_descriptors
from modules.text.bow import bag_of_words
from modules.text.ngrams import text_stats

log = config.get_logger("pipeline")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def analyse_audio(data: bytes) -> dict:
    ad = load_wav_full(data)                                     # may raise AudioError
    return audio_feature_vector(ad.x, ad.fs, ad.original_duration_s)


def analyse_image(data: bytes) -> tuple[dict, dict]:
    """-> (image_features, visual_features). Raises ImageError for unreadable images."""
    rgb = load_image(data)
    vis = visual_descriptors(rgb)
    imf = image_descriptors(rgb, fit_gray(rgb, 256))
    imf["edge_density"] = vis["edge_density"]
    imf["mean_gradient"] = vis["mean_gradient"]
    return imf, vis


def analyse_text(transcript: str) -> dict:
    s = text_stats(transcript)
    s["top_terms"] = [t for t, _ in bag_of_words(transcript).most_common(10)]
    return s


def _suffix(s: str | None) -> str:
    s = (s or ".png").lower()
    return s if s in IMAGE_SUFFIXES else ".png"


def _check_common(title: str, mood: str) -> None:
    if mood not in config.MOODS:
        raise ValueError("Please select a mood.")
    if not (title or "").strip():
        raise ValueError("Please enter a title.")


def create_memory(title: str, transcript: str, mood: str, entry_date: date | str,
                  audio_bytes: bytes | None, image_bytes: bytes | None, notes: str = "",
                  is_demo: bool = False, image_suffix: str = ".png") -> int:
    """Validate + analyse every provided modality, store files, then insert the row.
    Raises AudioError / ImageError / ValueError with user-readable messages."""
    _check_common(title, mood)
    transcript = transcript or ""
    if not (audio_bytes or image_bytes or transcript.strip()):
        raise ValueError("Add at least one of: audio, transcript or image.")
    af = analyse_audio(audio_bytes) if audio_bytes else None
    imf, vis = analyse_image(image_bytes) if image_bytes else (None, None)
    tf = analyse_text(transcript) if transcript.strip() else None
    written: list[str] = []
    try:
        ap = db.save_media(audio_bytes, config.AUDIO_DIR, ".wav") if audio_bytes else None
        if ap:
            written.append(ap)
        ip = db.save_media(image_bytes, config.IMAGE_DIR, _suffix(image_suffix)) if image_bytes else None
        if ip:
            written.append(ip)
        return db.add_entry(title.strip(), transcript.strip(), mood, str(entry_date), ap, ip, af, tf, imf,
                            notes, is_demo, visual_features=vis)
    except Exception:
        for rel in written:
            db.delete_media(rel)
        raise


@dataclass
class EditResult:
    recomputed: list[str]
    removed_files: list[str]


def update_memory(entry_id: int, *, title: str, transcript: str, mood: str, entry_date: date | str,
                  notes: str = "", new_audio: bytes | None = None, remove_audio: bool = False,
                  new_image: bytes | None = None, new_image_suffix: str = ".png",
                  remove_image: bool = False) -> EditResult:
    """Edit an existing memory in place (same id). Only the modalities that changed are re-analysed.
    Nothing is modified if validation, analysis or the database write fails."""
    old = db.get_entry(entry_id)
    if not old:
        raise ValueError("This memory no longer exists.")
    _check_common(title, mood)
    transcript = (transcript or "").strip()
    fields: dict = {}
    recomputed: list[str] = []
    to_delete: list[str] = []
    written: list[str] = []

    if title.strip() != old["title"]:
        fields["title"] = title.strip()
    if str(entry_date) != old["entry_date"]:
        fields["entry_date"] = str(entry_date)
    if mood != old["mood"]:
        fields["mood"] = mood
    if (notes or "") != (old["notes"] or ""):
        fields["notes"] = notes or ""
    if transcript != (old["transcript"] or "").strip():
        fields["transcript"] = transcript
        fields["text_features"] = analyse_text(transcript) if transcript else None
        recomputed.append("text")

    # --- analyse replacements first (raises before anything is written)
    new_af = analyse_audio(new_audio) if new_audio else None
    new_imf, new_vis = analyse_image(new_image) if new_image else (None, None)

    has_audio = bool(new_audio) or (bool(old["audio_path"]) and not remove_audio)
    has_image = bool(new_image) or (bool(old["image_path"]) and not remove_image)
    if not (has_audio or has_image or transcript):
        raise ValueError("A memory needs at least one of: audio, transcript or photo.")

    try:
        if new_audio:
            rel = db.save_media(new_audio, config.AUDIO_DIR, ".wav"); written.append(rel)
            fields.update(audio_path=rel, audio_features=new_af)
            recomputed.append("audio")
            if old["audio_path"]:
                to_delete.append(old["audio_path"])
        elif remove_audio and old["audio_path"]:
            fields.update(audio_path=None, audio_features=None)
            to_delete.append(old["audio_path"])
        if new_image:
            rel = db.save_media(new_image, config.IMAGE_DIR, _suffix(new_image_suffix)); written.append(rel)
            fields.update(image_path=rel, image_features=new_imf, visual_features=new_vis)
            recomputed.append("image")
            if old["image_path"]:
                to_delete.append(old["image_path"])
        elif remove_image and old["image_path"]:
            fields.update(image_path=None, image_features=None, visual_features=None)
            to_delete.append(old["image_path"])
        db.update_entry(entry_id, **fields)                     # one transaction
    except Exception:
        for rel in written:                                     # roll back files; DB row is unchanged
            db.delete_media(rel)
        raise
    removed = [rel for rel in to_delete if db.delete_media(rel)]
    return EditResult(recomputed, removed)


def backfill_visual_features() -> dict:
    """Compute and persist visual descriptors (and refreshed image features) for photos saved before schema v2.
    Entries whose photo is missing/corrupt are skipped and reported; nothing is recomputed otherwise."""
    done, skipped = 0, []
    for e in db.list_entries():
        if not e["image_path"] or e["visual_features"]:
            continue
        p = db.abs_path(e["image_path"])
        try:
            if not p or not p.exists():
                raise ImageError("file missing")
            imf, vis = analyse_image(p.read_bytes())
            db.update_entry(e["id"], _touch=False, image_features=imf, visual_features=vis)
            done += 1
        except (ImageError, OSError) as exc:
            log.warning("Visual descriptors skipped for entry %s: %s", e["id"], exc)
            skipped.append(e["id"])
    return {"computed": done, "skipped": skipped}

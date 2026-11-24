"""SQLite persistence. Media files live under data/audio and data/images; features are stored as
JSON so the UI never has to recompute them (the transcript is stored too, so text features can be
recomputed)."""
from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from modules import config
from modules.config import get_logger

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,          -- ISO timestamp
    entry_date TEXT NOT NULL,          -- YYYY-MM-DD
    transcript TEXT NOT NULL DEFAULT '',
    mood TEXT NOT NULL,
    audio_path TEXT,
    image_path TEXT,
    audio_features TEXT,               -- JSON
    text_features TEXT,                -- JSON
    image_features TEXT,               -- JSON
    notes TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    visual_features TEXT,              -- JSON (schema v2): persisted Sobel/histogram descriptors
    updated_at TEXT                    -- ISO timestamp of the last edit (schema v2)
);
"""
SCHEMA_VERSION = 2
_V2_COLUMNS = {"visual_features": "TEXT", "updated_at": "TEXT"}


@contextmanager
def connect():
    config.ensure_dirs()
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create the table, or upgrade an existing v1 database in place (no data is touched)."""
    with connect() as con:
        con.executescript(SCHEMA)
        have = {r["name"] for r in con.execute("PRAGMA table_info(entries)")}
        for col, typ in _V2_COLUMNS.items():
            if col not in have:
                con.execute(f"ALTER TABLE entries ADD COLUMN {col} {typ}")
                log.info("Migrated database: added column %s", col)
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k in ("audio_features", "text_features", "image_features", "visual_features"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    return d


def save_media(data: bytes, folder: Path, suffix: str) -> str:
    """Write bytes to a unique file; returns the path relative to DATA_DIR (portable)."""
    config.ensure_dirs()
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{suffix}"
    p = folder / name
    p.write_bytes(data)
    return str(p.relative_to(config.DATA_DIR)).replace("\\", "/")


def abs_path(rel: str | None) -> Path | None:
    return (config.DATA_DIR / rel) if rel else None


def add_entry(title: str, transcript: str, mood: str, entry_date: str,
              audio_path: str | None, image_path: str | None, audio_features: dict | None,
              text_features: dict | None, image_features: dict | None, notes: str = "",
              is_demo: bool = False, visual_features: dict | None = None) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO entries (title, created_at, entry_date, transcript, mood, audio_path,"
            " image_path, audio_features, text_features, image_features, notes, is_demo,"
            " visual_features) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (title, datetime.now().isoformat(timespec="seconds"), entry_date, transcript, mood,
             audio_path, image_path,
             json.dumps(audio_features) if audio_features else None,
             json.dumps(text_features) if text_features else None,
             json.dumps(image_features) if image_features else None, notes, int(is_demo),
             json.dumps(visual_features) if visual_features else None))
        log.info("Saved entry %s (%s)", cur.lastrowid, title)
        return int(cur.lastrowid)


def get_entry(entry_id: int) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
    return _row(r) if r else None


def list_entries(mood: str | None = None, date_from: str | None = None,
                 date_to: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM entries WHERE 1=1", []
    if mood and mood != "All":
        q += " AND mood=?"; args.append(mood)
    if date_from:
        q += " AND entry_date>=?"; args.append(date_from)
    if date_to:
        q += " AND entry_date<=?"; args.append(date_to)
    q += " ORDER BY entry_date DESC, id DESC"
    with connect() as con:
        return [_row(r) for r in con.execute(q, args).fetchall()]


def count_entries() -> int:
    with connect() as con:
        return int(con.execute("SELECT COUNT(*) FROM entries").fetchone()[0])


def delete_entry(entry_id: int) -> bool:
    e = get_entry(entry_id)
    if not e:
        return False
    with connect() as con:
        con.execute("DELETE FROM entries WHERE id=?", (entry_id,))
    for rel in (e["audio_path"], e["image_path"]):
        delete_media(rel)
    log.info("Deleted entry %s", entry_id)
    return True


def delete_demo_entries() -> int:
    ids = [e["id"] for e in list_entries() if e["is_demo"]]
    for i in ids:
        delete_entry(i)
    return len(ids)


_UPDATABLE = {"title", "entry_date", "transcript", "mood", "audio_path", "image_path", "notes",
              "audio_features", "text_features", "image_features", "visual_features"}
_JSON_COLS = {"audio_features", "text_features", "image_features", "visual_features"}


def update_entry(entry_id: int, _touch: bool = True, **fields) -> None:
    """Update selected columns of an existing row in ONE transaction (all or nothing).
    Keeps the id and created_at. JSON columns accept dicts or None."""
    bad = set(fields) - _UPDATABLE
    if bad:
        raise ValueError(f"Cannot update column(s): {sorted(bad)}")
    if not fields:
        return
    vals = [json.dumps(v) if (k in _JSON_COLS and v) else (None if k in _JSON_COLS else v)
            for k, v in fields.items()]
    sets = ", ".join(f"{k}=?" for k in fields)
    args = list(vals)
    if _touch:                      # a backfill of derived data must not look like a user edit
        sets += ", updated_at=?"
        args.append(datetime.now().isoformat(timespec="seconds"))
    with connect() as con:
        cur = con.execute(f"UPDATE entries SET {sets} WHERE id=?", (*args, entry_id))
        if cur.rowcount != 1:
            raise ValueError(f"Memory {entry_id} no longer exists.")
    log.info("Updated entry %s: %s", entry_id, ", ".join(sorted(fields)))


def delete_media(rel: str | None) -> bool:
    """Best-effort removal of one media file (path relative to DATA_DIR). Never raises."""
    try:
        p = abs_path(rel)
        if p and p.exists():
            p.unlink()
            return True
    except OSError as exc:
        log.warning("Could not delete %s: %s", rel, exc)
    return False


def media_missing(entry: dict) -> dict:
    """{'audio': bool, 'image': bool}: True when a path is stored but the file is gone from disk."""
    out = {}
    for k in ("audio", "image"):
        rel = entry.get(f"{k}_path")
        p = abs_path(rel)
        out[k] = bool(rel) and not (p and p.exists())
    return out


def find_orphans() -> dict:
    """Consistency check between rows and files: files no row references, rows whose file is missing."""
    referenced = set()
    missing = []
    for e in list_entries():
        for k in ("audio_path", "image_path"):
            if e[k]:
                referenced.add(e[k])
                if not abs_path(e[k]).exists():
                    missing.append({"id": e["id"], "title": e["title"], "path": e[k]})
    on_disk = set()
    for folder in (config.AUDIO_DIR, config.IMAGE_DIR):
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file() and p.name != ".gitkeep":
                    on_disk.add(str(p.relative_to(config.DATA_DIR)).replace("\\", "/"))
    return {"orphan_files": sorted(on_disk - referenced), "missing_files": missing}

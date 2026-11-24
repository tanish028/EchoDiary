"""DB CRUD, editing, multimodal and integration workflows A-G (all on a throw-away data directory)."""
import sqlite3
from pathlib import Path

import pytest

from modules import config
from modules.audio.preprocessing import AudioError
from modules.image.preprocessing import ImageError
from modules.image import visual
from modules.multimodal.integration import mood_summary, similar_entries
from modules.pipeline import (EditResult, backfill_visual_features, create_memory, update_memory)
from modules.text.corpus import build_corpus

from conftest import make_image, make_wav


def files(folder):
    return sorted(p.name for p in folder.glob("*") if p.is_file())


def mk(title="T", transcript="lake sunrise walk", mood="Calm", audio=True, image="vstripes", d="2026-11-01", **kw):
    return create_memory(title, transcript, mood, d, make_wav() if audio else None,
                         make_image(image) if image else None, **kw)


# ================================================================ DB CRUD
def test_schema_v2_and_crud(env):
    db = env
    with sqlite3.connect(config.DB_PATH) as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == 2
        cols = {r[1] for r in c.execute("PRAGMA table_info(entries)")}
    assert {"visual_features", "updated_at"} <= cols
    eid = mk()
    e = db.get_entry(eid)
    assert e["title"] == "T" and e["audio_features"]["duration_s"] == 1.0
    assert e["visual_features"]["hist"] and e["updated_at"] in (None, "")
    assert db.count_entries() == 1 and [x["id"] for x in db.list_entries("Calm")] == [eid]
    assert db.list_entries("Sad") == [] and db.list_entries(date_from="2026-12-01") == []
    assert db.get_entry(999) is None
    assert db.delete_entry(eid) and db.get_entry(eid) is None
    assert files(config.AUDIO_DIR) == [] and files(config.IMAGE_DIR) == []
    assert db.delete_entry(eid) is False


def test_migration_from_v1(tmp_path, monkeypatch):
    for k, v in dict(DATA_DIR=tmp_path, AUDIO_DIR=tmp_path / "a", IMAGE_DIR=tmp_path / "i",
                     DB_PATH=tmp_path / "old.db").items():
        monkeypatch.setattr(config, k, v)
    from modules.database import db
    with sqlite3.connect(config.DB_PATH) as c:                    # the first-release schema
        c.execute("""CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
            entry_date TEXT NOT NULL, created_at TEXT NOT NULL, transcript TEXT DEFAULT '', mood TEXT NOT NULL,
            audio_path TEXT, image_path TEXT, audio_features TEXT, text_features TEXT, image_features TEXT,
            notes TEXT DEFAULT '', is_demo INTEGER DEFAULT 0)""")
        c.execute("INSERT INTO entries (title,entry_date,created_at,transcript,mood) VALUES ('old','2026-01-01','x','hello world','Calm')")
    db.init_db(); db.init_db()                                    # idempotent
    e = db.list_entries()[0]
    assert e["title"] == "old" and e["visual_features"] is None
    assert backfill_visual_features() == {"computed": 0, "skipped": []}


def test_backfill_computes_for_old_photos(env):
    db = env
    eid = mk()
    db.update_entry(eid, _touch=False, visual_features=None)
    assert db.get_entry(eid)["visual_features"] is None
    assert backfill_visual_features()["computed"] == 1
    e = db.get_entry(eid)
    assert e["visual_features"] and not e["updated_at"]


# ================================================================ Workflow A: create + revisit
def test_A_create_and_revisit(env):
    db = env
    eid = mk("Morning lake", "I walked by the lake", "Happy", notes="n")
    e = db.get_entry(eid)
    assert (e["mood"], e["notes"]) == ("Happy", "n")
    assert e["text_features"]["n_tokens"] == 5 and e["image_features"]["width"] == 64
    assert db.abs_path(e["audio_path"]).exists() and db.abs_path(e["image_path"]).exists()
    assert db.media_missing(e) == {"audio": False, "image": False}


def test_create_validation_and_no_orphans(env, monkeypatch):
    db = env
    with pytest.raises(ValueError):
        create_memory("", "x", "Calm", "2026-11-01", None, None)
    with pytest.raises(ValueError):
        create_memory("t", "x", "Nope", "2026-11-01", None, None)
    with pytest.raises(ValueError):
        create_memory("t", "  ", "Calm", "2026-11-01", None, None)
    with pytest.raises(AudioError):
        create_memory("t", "", "Calm", "2026-11-01", b"junk", make_image_bytes())
    with pytest.raises(ImageError):
        create_memory("t", "", "Calm", "2026-11-01", make_wav(), b"junk")
    assert files(config.AUDIO_DIR) == [] == files(config.IMAGE_DIR) and db.count_entries() == 0

    def boom(*a, **k):
        raise RuntimeError("disk full")
    with monkeypatch.context() as m:
        m.setattr(db, "add_entry", boom)                          # failed DB write after files were written
        with pytest.raises(RuntimeError):
            mk()
    assert files(config.AUDIO_DIR) == [] == files(config.IMAGE_DIR)


def make_image_bytes():
    return make_image("noise")


def test_duplicate_titles_allowed_and_independent(env):
    a, b = mk("Same"), mk("Same")
    assert a != b and env.count_entries() == 2
    assert len(files(config.IMAGE_DIR)) == 2


# ================================================================ Workflow B: editing
def test_B_edit_text_fields_only_recomputes_text(env):
    db = env
    eid = mk()
    old = db.get_entry(eid)
    r = update_memory(eid, title="New", transcript="completely different words", mood="Sad",
                      entry_date="2026-10-10", notes="x")
    assert r.recomputed == ["text"] and r.removed_files == []
    e = db.get_entry(eid)
    assert (e["id"], e["title"], e["mood"], e["entry_date"], e["notes"]) == (eid, "New", "Sad", "2026-10-10", "x")
    assert e["text_features"]["n_tokens"] == 3 and e["updated_at"]
    assert e["audio_path"] == old["audio_path"] and e["image_path"] == old["image_path"]
    assert e["audio_features"] == old["audio_features"] and e["visual_features"] == old["visual_features"]


def test_B_title_only_edit_recomputes_nothing(env):
    eid = mk()
    r = update_memory(eid, title="Renamed", transcript="lake sunrise walk", mood="Calm", entry_date="2026-11-01")
    assert r.recomputed == [] and env.get_entry(eid)["title"] == "Renamed"


def test_B_replace_image_and_audio_leaves_no_orphans(env):
    db = env
    eid = mk()
    old = db.get_entry(eid)
    r = update_memory(eid, title="T", transcript="lake sunrise walk", mood="Calm", entry_date="2026-11-01",
                      new_audio=make_wav(freq=500, dur=2.0), new_image=make_image("dark"), new_image_suffix=".png")
    assert set(r.recomputed) == {"audio", "image"} and len(r.removed_files) == 2
    e = db.get_entry(eid)
    assert e["audio_features"]["duration_s"] == 2.0 and e["image_features"]["brightness"] < 60
    assert e["visual_features"] != old["visual_features"]
    assert e["audio_path"] != old["audio_path"] and not db.abs_path(old["audio_path"]).exists()
    assert not db.abs_path(old["image_path"]).exists()
    assert len(files(config.AUDIO_DIR)) == 1 == len(files(config.IMAGE_DIR))
    assert db.find_orphans() == {"orphan_files": [], "missing_files": []}


def test_B_remove_media_and_require_one_modality(env):
    db = env
    eid = mk()
    update_memory(eid, title="T", transcript="lake", mood="Calm", entry_date="2026-11-01",
                  remove_audio=True, remove_image=True)
    e = db.get_entry(eid)
    assert e["audio_path"] is None and e["image_path"] is None and e["visual_features"] is None
    assert files(config.AUDIO_DIR) == [] == files(config.IMAGE_DIR)
    with pytest.raises(ValueError):
        update_memory(eid, title="T", transcript="", mood="Calm", entry_date="2026-11-01")
    assert db.get_entry(eid)["transcript"] == "lake"                     # unchanged after the rejected edit
    update_memory(eid, title="T", transcript="lake", mood="Calm", entry_date="2026-11-01",
                  new_image=make_image("bright"), new_image_suffix=".jpg")   # add a photo to a photo-less entry
    assert db.get_entry(eid)["image_path"].endswith(".jpg")


def test_B_failed_edit_does_not_corrupt(env, monkeypatch):
    db = env
    eid = mk()
    before = db.get_entry(eid)
    imgs, auds = files(config.IMAGE_DIR), files(config.AUDIO_DIR)
    with pytest.raises(ImageError):                                      # bad replacement: nothing written
        update_memory(eid, title="X", transcript="y", mood="Sad", entry_date="2026-11-02", new_image=b"junk")
    with pytest.raises(AudioError):
        update_memory(eid, title="X", transcript="y", mood="Sad", entry_date="2026-11-02", new_audio=b"junk")

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    with monkeypatch.context() as m:
        m.setattr(db, "update_entry", boom)                               # DB failure after new files were written
        with pytest.raises(sqlite3.OperationalError):
            update_memory(eid, title="X", transcript="y", mood="Sad", entry_date="2026-11-02",
                          new_image=make_image("dark"), new_audio=make_wav(freq=300))
    assert db.get_entry(eid) == before
    assert files(config.IMAGE_DIR) == imgs and files(config.AUDIO_DIR) == auds
    assert db.abs_path(before["image_path"]).exists()
    with pytest.raises(ValueError):
        update_memory(12345, title="x", transcript="y", mood="Sad", entry_date="2026-11-02")
    with pytest.raises(ValueError):
        db.update_entry(eid, id=5)                                        # id cannot be edited


# ================================================================ Workflow C: text search across languages
def test_C_text_search_english_hindi_mixed(env):
    db = env
    a = mk("en", "I walked by the lake at sunrise", audio=False, image=None)
    b = mk("hi", "आज मैं बहुत खुश हूँ और मौसम सुंदर है", audio=False, image=None)
    c = mk("mix", "Dinner with friends और बहुत मज़ा आया", audio=False, image=None)
    corpus = build_corpus(db.list_entries())
    def top(q):
        hits, _ = corpus.search(q)
        return hits[0][0] if hits else None
    assert top("lake sunrise") == a
    assert top("खुश") == b and top("मौसम") == b
    assert top("friends मज़ा") == c
    assert top("") is None and top("!!!") is None and top("zzzz") is None
    # edits are reflected: Search, Analysis and Insights all rebuild the same corpus
    update_memory(a, title="en", transcript="सुबह की सैर झील", mood="Calm", entry_date="2026-11-01")
    corpus = build_corpus(db.list_entries())
    assert corpus.search("झील")[0][0][0] == a and corpus.search("lake")[0] == []


# ================================================================ Workflow D: visual search
def test_D_visual_search_returns_correct_entry(env):
    db = env
    ids = {k: mk(k, "x", image=k, audio=False) for k in ("vstripes", "dark", "bright", "hstripes", "gradient")}
    ents = db.list_entries()
    ref = visual.visual_descriptors(__import__("modules.image.preprocessing", fromlist=["x"]).load_image(
        make_image("vstripes", (120, 90), seed=7)))              # a different size/seed of the same pattern
    ranked = visual.rank(ref, [(e["id"], e["visual_features"]) for e in ents])
    assert ranked[0]["id"] == ids["vstripes"]
    assert db.get_entry(ranked[0]["id"])["title"] == "vstripes"            # "open diary entry" target is correct
    assert len(ranked) == 5


def test_D_empty_missing_and_corrupt_images(env):
    db = env
    assert visual.rank({"hist": [0] * 32}, []) == []
    eid = mk("a", "x", audio=False)
    other = mk("b", "x", audio=False, image="dark")
    db.abs_path(db.get_entry(eid)["image_path"]).unlink()                  # missing file
    db.abs_path(db.get_entry(other)["image_path"]).write_bytes(b"corrupt")  # corrupt file
    db.update_entry(eid, _touch=False, visual_features=None)
    db.update_entry(other, _touch=False, visual_features=None)
    r = backfill_visual_features()
    assert r["computed"] == 0 and sorted(r["skipped"]) == sorted([eid, other])
    assert len(db.find_orphans()["missing_files"]) == 1


# ================================================================ Workflow E: multimodal
def test_E_multimodal_audio_image_formats(env):
    db = env
    import io, numpy as np
    from scipy.io import wavfile
    buf = io.BytesIO(); wavfile.write(buf, 8000, (np.sin(np.arange(16000) / 5) * 2 ** 30).astype(np.int32))
    a = create_memory("a", "calm lake water", "Calm", "2026-11-01", make_wav(freq=200), make_image("gradient", fmt="JPEG"), image_suffix=".jpg")
    b = create_memory("b", "calm lake morning", "Calm", "2026-11-02", buf.getvalue(), make_image("gradient", (50, 90), fmt="PNG"))
    c = create_memory("c", "exam stress deadline", "Anxious", "2026-11-03", make_wav(freq=3000, amp=0.1), make_image("noise"))
    ents = db.list_entries()
    idx = [e["id"] for e in ents].index(a)
    res = similar_entries(idx, ents)
    assert len(res) == 2 and {"audio", "text", "image"} <= set(res[0]["parts"])
    assert ents[res[0]["index"]]["id"] == b
    assert all(-1.0001 <= r["score"] <= 1.0001 for r in res)   # audio cosine on z-scores may be negative
    assert similar_entries(0, ents[:1]) == []
    assert mood_summary(ents)


def test_E_long_audio_flags_truncation(env, monkeypatch):
    from modules.audio import preprocessing as pre
    monkeypatch.setattr(pre, "MAX_AUDIO_SECONDS", 1)
    eid = create_memory("long", "", "Calm", "2026-11-01", make_wav(dur=3.0), None)
    af = env.get_entry(eid)["audio_features"]
    assert af["truncated"] and af["duration_s"] == 3.0 and af["analysed_duration_s"] == 1.0


# ================================================================ Workflow F: persistence
def test_F_persistence_restart_edit_delete(env):
    db = env
    eid = mk("persist")
    db.init_db()                                                           # "restart": re-open the same files
    assert db.get_entry(eid)["title"] == "persist"
    update_memory(eid, title="persist2", transcript="new text", mood="Happy", entry_date="2026-11-05")
    db.init_db()
    assert db.get_entry(eid)["title"] == "persist2"
    assert db.find_orphans() == {"orphan_files": [], "missing_files": []}
    db.delete_entry(eid)
    assert db.find_orphans() == {"orphan_files": [], "missing_files": []}
    assert files(config.AUDIO_DIR) == [] == files(config.IMAGE_DIR)


def test_F_orphan_detection_and_cleanup(env):
    db = env
    eid = mk()
    stray = config.IMAGE_DIR / "stray.png"; stray.write_bytes(b"x")
    h = db.find_orphans()
    assert h["orphan_files"] == ["images/stray.png"]
    assert db.delete_media("images/stray.png") and not stray.exists()
    assert db.delete_media("images/never-existed.png") is False and db.delete_media(None) is False
    db.abs_path(db.get_entry(eid)["audio_path"]).unlink()
    assert db.media_missing(db.get_entry(eid)) == {"audio": True, "image": False}
    assert db.delete_entry(eid)                                           # deleting with a missing file still works


def test_F_demo_entries(env):
    db = env
    mk("[DEMO] one", is_demo=True); mk("real")
    assert db.delete_demo_entries() == 1 and db.count_entries() == 1


# ================================================================ Workflow G: misc errors
def test_G_empty_collections_and_unsupported(env):
    db = env
    assert db.list_entries() == [] and build_corpus([]).search("anything") == ([], build_corpus([]).search("x")[1])
    assert mood_summary([]) == []
    with pytest.raises(ImageError):
        create_memory("t", "", "Calm", "2026-11-01", None, b"GIF89a-not-really")
    with pytest.raises(AudioError):
        create_memory("t", "", "Calm", "2026-11-01", b"ID3\x03\x00mp3data", None)

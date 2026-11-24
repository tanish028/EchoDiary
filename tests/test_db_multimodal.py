import importlib

import pytest

from scripts import seed_demo


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHODIARY_DATA", str(tmp_path))
    import modules.config as c
    importlib.reload(c)
    import modules.database.db as db
    importlib.reload(db)
    import modules.pipeline as p
    importlib.reload(p)
    importlib.reload(seed_demo)
    db.init_db()
    return db, p


def test_create_get_delete_and_persistence(env):
    db, p = env
    eid = p.create_memory("t", "hello lake", "Calm", "2026-01-01", seed_demo.synth_voice(150, 1, 3, .01, 0),
                          seed_demo.synth_image("dusk", 0))
    e = db.get_entry(eid)
    assert e["audio_features"]["duration_s"] > 0 and e["image_features"] and e["text_features"]
    assert db.abs_path(e["audio_path"]).exists()
    # "restart": reload modules, data must remain
    importlib.reload(db)
    assert db.count_entries() == 1
    assert db.delete_entry(eid) and db.count_entries() == 0
    assert not db.delete_entry(eid)


def test_validation_errors(env):
    _, p = env
    with pytest.raises(ValueError):
        p.create_memory("", "x", "Calm", "2026-01-01", None, None)
    with pytest.raises(ValueError):
        p.create_memory("t", "", "Calm", "2026-01-01", None, None)
    with pytest.raises(ValueError):
        p.create_memory("t", "x", "Bogus", "2026-01-01", None, None)
    from modules.audio.preprocessing import AudioError
    with pytest.raises(AudioError):
        p.create_memory("t", "x", "Calm", "2026-01-01", b"junk", None)


def test_text_only_entry_and_filters(env):
    db, p = env
    p.create_memory("a", "just text", "Sad", "2026-02-01", None, None)
    p.create_memory("b", "more text", "Happy", "2026-03-01", None, None)
    assert len(db.list_entries(mood="Sad")) == 1
    assert len(db.list_entries(date_from="2026-02-15")) == 1


def test_multimodal_similarity_and_mood_summary(env):
    db, p = env
    from modules.multimodal.integration import mood_summary, similar_entries
    assert seed_demo.seed() == 5
    es = db.list_entries()
    sims = similar_entries(0, es)
    assert len(sims) == 4 and all(-1.01 <= s["score"] <= 1.01 for s in sims)
    assert similar_entries(0, es[:1]) == []
    assert sum(r["Entries"] for r in mood_summary(es)) == 5
    assert db.delete_demo_entries() == 5

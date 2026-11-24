"""Headless UI smoke tests with streamlit.testing.v1.AppTest."""
import importlib

import pytest
from modules.image import point
from streamlit.testing.v1 import AppTest

from scripts import seed_demo

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
PAGES = ["dashboard", "create_memory", "analysis", "my_memories", "visual_search", "insights"]


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHODIARY_DATA", str(tmp_path))
    import modules.config as c; importlib.reload(c)
    import modules.database.db as db; importlib.reload(db)
    import modules.pipeline as p; importlib.reload(p)
    importlib.reload(seed_demo)
    db.init_db()
    return db


def run(page):
    """Run through app.py so st.navigation / st.page_link work as in the real app."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    if page != "dashboard":
        at = at.switch_page(f"views/{page}.py").run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_empty(data, page):
    assert not run(page).exception


@pytest.mark.parametrize("page", PAGES)
def test_pages_with_data(data, page):
    seed_demo.seed()
    assert not run(page).exception


def test_single_entry_pages(data):
    seed_demo.synth_voice  # noqa
    from modules.pipeline import create_memory
    create_memory("only", "just one entry here", "Calm", "2026-01-01", seed_demo.synth_voice(120, 2, 3, .01, 0),
                  seed_demo.synth_image("dusk", 0))
    for p in PAGES:
        assert not run(p).exception, p


def test_text_only_entry_pages(data):
    from modules.pipeline import create_memory
    create_memory("words", "no media at all", "Sad", "2026-01-01", None, None)
    for p in PAGES:
        assert not run(p).exception, p


def test_analysis_controls(data):
    seed_demo.seed()
    at = run("analysis")
    assert not at.exception
    for w in ("hamming", "rectangular"):
        next(s for s in at.selectbox if s.label == "Window").set_value(w).run()
        assert not at.exception, w
    get = lambda label: next(s for s in at.selectbox if s.label == label)   # re-fetch after each rerun
    for kind in ("high-pass", "band-pass", "orientation", "low-pass"):
        get("Mask").set_value(kind).run()
        assert not at.exception, kind
    get("Longer side (pixels)").set_value(512).run()
    assert not at.exception


def test_search_page(data):
    seed_demo.seed()
    at = run("my_memories")
    for q in ("lake sunrise", "exam", "zzzz", "the", "   "):
        at.text_input[0].set_value(q).run()
        assert not at.exception, q


def test_home_demo_button(data):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    btn = next(b for b in at.button if "demo" in b.label.lower())
    btn.click().run()
    assert not at.exception
    assert data.count_entries() == 5


def test_analysis_image_tabs_and_vocabulary(data):
    seed_demo.seed()
    at = run("analysis")
    assert not at.exception
    sel = lambda label: next(s for s in at.selectbox if s.label == label)
    for op in point.OPS:
        sel("Operation").set_value(op).run()
        assert not at.exception, op
    radio = next(r for r in at.radio if r.label == "Filter")
    radio.set_value("Median").run()
    assert not at.exception
    next(r for r in at.radio if r.label == "Filter").set_value("Gaussian").run()
    assert not at.exception


def test_analysis_all_entries(data):
    seed_demo.seed()
    n = data.count_entries()
    for i in range(n):
        at = run("analysis")
        at.selectbox[0].set_value(data.list_entries()[i]["id"]).run() if at.selectbox else None
        assert not at.exception, i


def test_visual_search_page_flow(data):
    from modules.pipeline import create_memory
    from conftest import make_image
    create_memory("a", "x", "Calm", "2026-01-01", None, make_image("vstripes"))
    at = run("visual_search")
    assert not at.exception and any("Low-level visual similarity" in m.value for m in at.markdown)
    create_memory("b", "y", "Calm", "2026-01-02", None, make_image("dark"))
    at = run("visual_search")
    at.radio[0].set_value("Use the photo of a saved memory").run()
    assert not at.exception
    assert any("Open diary entry" == b.label for b in at.button)


def test_visual_search_empty_and_no_photos(data):
    assert not run("visual_search").exception
    from modules.pipeline import create_memory
    create_memory("t", "just text", "Calm", "2026-01-01", None, None)
    at = run("visual_search")
    assert not at.exception and any("photo" in i.value for i in at.info)


def test_titles_are_escaped(data):
    from modules.pipeline import create_memory
    create_memory("<script>alert(1)</script><b>x</b>", "hello", "Calm", "2026-01-01", None, None)
    for p in ("dashboard", "my_memories", "analysis", "insights"):
        at = run(p)
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        assert "<script>" not in html, p


def test_detail_and_edit_flow(data):
    from modules.pipeline import create_memory
    from conftest import make_image
    eid = create_memory("orig", "lake walk", "Calm", "2026-01-01", None, make_image("dark"))
    at = run("my_memories")
    next(b for b in at.button if b.label == "Open").click().run()
    assert not at.exception and at.session_state["detail_id"] == eid
    next(b for b in at.button if "Edit" in b.label).click().run()
    assert not at.exception
    next(t for t in at.text_input if t.label == "Title").set_value("renamed").run()
    next(b for b in at.button if "Save changes" in b.label).click().run()
    assert not at.exception
    assert data.get_entry(eid)["title"] == "renamed"
    # destructive removal needs a confirmation before Save is enabled
    next(b for b in at.button if "Edit" in b.label).click().run()
    next(r for r in at.radio if r.label == "Photo").set_value("Remove").run()
    save = next(b for b in at.button if "Save changes" in b.label)
    assert save.disabled
    next(c for c in at.checkbox if "permanently" in c.label).check().run()
    next(b for b in at.button if "Save changes" in b.label).click().run()
    assert not at.exception and data.get_entry(eid)["image_path"] is None


def test_delete_confirmation(data):
    from modules.pipeline import create_memory
    eid = create_memory("gone", "text here", "Calm", "2026-01-01", None, None)
    at = run("my_memories")
    next(b for b in at.button if b.label == "Open").click().run()
    next(b for b in at.button if "Delete" in b.label).click().run()
    assert data.get_entry(eid) is not None                     # not deleted before confirmation
    next(b for b in at.button if b.label == "Yes, delete").click().run()
    assert data.get_entry(eid) is None and not at.exception

"""Browser end-to-end workflow (needs: pip install playwright, a Chrome/Chromium binary, a running app).
Usage: python tests/e2e_browser.py http://localhost:8601 /path/to/chrome
Creates 3 memories through the real UI, searches, visual-searches, edits, deletes; fails on any Streamlit exception."""
import io, re, sys, tempfile, os
import numpy as np
from PIL import Image
from scipy.io import wavfile
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8601"
CHROME = sys.argv[2] if len(sys.argv) > 2 else "/usr/bin/google-chrome"
tmp = tempfile.mkdtemp()


def img(kind, name, size=(120, 80)):
    w, h = size; y, x = np.mgrid[0:h, 0:w]
    v = {"v": 128 + 120 * np.sign(np.sin(2 * np.pi * 8 * x / w)), "d": np.full((h, w), 30.0),
         "g": 255 * x / w}[kind]
    p = os.path.join(tmp, name); Image.fromarray(np.stack([v] * 3, -1).astype(np.uint8)).save(p); return p


def wav(name, f=220):
    t = np.arange(16000) / 16000; p = os.path.join(tmp, name)
    wavfile.write(p, 16000, (0.5 * np.sin(2 * np.pi * f * t) * 32767).astype(np.int16)); return p


results = []
def check(name, cond):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)


with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--use-fake-device-for-media-stream",
                                                         "--use-fake-ui-for-media-stream"])
    pg = b.new_context(viewport={"width": 1400, "height": 1000}, permissions=["microphone"]).new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    def no_exc(label):
        pg.wait_for_timeout(1500)
        check(f"no exception: {label}", pg.locator('[data-testid="stException"]').count() == 0)
    def nav(label):
        pg.get_by_role("link", name=label).first.click(); pg.wait_for_timeout(1500)
    pg.goto(URL); pg.wait_for_selector("text=EchoDiary", timeout=30000); no_exc("home (empty)")
    for lbl in ("My Memories", "Memory Analysis", "Visual Search", "Multimodal Insights", "Create Memory"):
        nav(lbl); no_exc(lbl + " (empty)")

    def create(title, text, mood, image, audio=None):
        nav("Create Memory")
        pg.get_by_label("Title", exact=True).fill(title)
        pg.get_by_label("Transcript", exact=False).fill(text); pg.keyboard.press("Control+Enter")
        pg.get_by_label("How did you feel?", exact=False).click()
        pg.get_by_role("option", name=mood, exact=True).click()
        if audio:
            pg.get_by_role("tab", name="Upload WAV").click(); pg.wait_for_timeout(500)
            pg.locator('input[type=file]').nth(0).set_input_files(audio)
        pg.locator('input[type=file]').last.set_input_files(image)
        pg.wait_for_timeout(1500)
        pg.get_by_role("button", name="Save memory").click(); pg.wait_for_timeout(2500)
    # microphone: fake capture device supplied by Chrome flags
    nav("Create Memory"); pg.get_by_role("tab", name="Record").click(); pg.wait_for_timeout(800)
    try:
        pg.get_by_role("button", name="Record").first.click(timeout=8000); pg.wait_for_timeout(2500)
        pg.get_by_role("button", name="Stop recording").click(timeout=8000); pg.wait_for_timeout(2500)
        check("microphone recording produced a clip", pg.locator("audio").count() >= 1)
    except Exception as exc:
        check(f"microphone recording ({type(exc).__name__})", False)
    no_exc("after mic recording")
    create("Lake walk", "I walked by the lake at sunrise", "Calm", img("v", "a.png"), wav("a.wav"))
    check("success message after save", pg.get_by_text("saved", exact=False).count() > 0); no_exc("after create 1")
    create("आज की खुशी", "आज मैं बहुत खुश हूँ", "Happy", img("d", "b.png"))
    create("Late night", "<b>exam</b> stress and no sleep", "Anxious", img("g", "c.png", (90, 150)))
    nav("My Memories"); no_exc("memories with data")
    check("3 memories listed", pg.get_by_text("3 of 3 memories").count() == 1)
    pg.get_by_placeholder("e.g. lake sunrise walk").fill("खुश"); pg.keyboard.press("Enter"); pg.wait_for_timeout(1500)
    check("Hindi search finds Hindi memory", pg.get_by_text("1 of 3 memories").count() == 1)
    pg.get_by_placeholder("e.g. lake sunrise walk").fill(""); pg.keyboard.press("Enter"); pg.wait_for_timeout(1000)
    pg.wait_for_selector("text=3 of 3 memories"); pg.wait_for_timeout(1500)
    pg.locator("button", has_text=re.compile(r"^Open$")).first.click(); pg.wait_for_selector("text=← All memories", timeout=15000)
    pg.wait_for_timeout(1000); no_exc("detail")
    check("detail shows Edit button", pg.get_by_role("button", name="Edit").count() >= 1)
    pg.get_by_role("button", name="Edit").first.click(); pg.wait_for_timeout(1200)
    pg.get_by_label("Title").first.fill("Renamed entry")
    pg.get_by_role("button", name="Save changes").click(); pg.wait_for_timeout(2500); no_exc("after edit")
    check("edit saved", pg.get_by_text("Renamed entry").count() >= 1)
    nav("Memory Analysis"); no_exc("analysis")
    for tab in ("Text", "Image"):
        pg.get_by_role("tab", name=tab, exact=False).first.click(); pg.wait_for_timeout(1200); no_exc("analysis " + tab)
    for sub in ("Histogram", "Point processing", "Spatial filters", "Edges (Sobel)", "Fourier"):
        pg.get_by_role("tab", name=sub).click(); pg.wait_for_timeout(1500); no_exc("image tab " + sub)
    nav("Visual Search"); pg.get_by_text("Use the photo of a saved memory").click(); pg.wait_for_timeout(2500)
    no_exc("visual search"); check("results shown", pg.get_by_text("Open diary entry").count() >= 1)
    pg.get_by_role("button", name="Open diary entry").first.click(); pg.wait_for_timeout(2000); no_exc("open from visual search")
    check("opened full entry", pg.get_by_role("button", name="Edit").count() >= 1)
    nav("Multimodal Insights"); no_exc("insights")
    nav("My Memories"); pg.wait_for_timeout(2500)
    if pg.get_by_text("← All memories").count() == 0:        # the entry opened from Visual Search is still open
        pg.locator("button", has_text=re.compile(r"^Open$")).first.click(); pg.wait_for_selector("text=← All memories", timeout=15000)
    pg.get_by_role("button", name="Delete").first.click(); pg.wait_for_timeout(800)
    pg.get_by_role("button", name="Yes, delete").click(); pg.wait_for_timeout(1500)
    check("deleted -> 2 memories", pg.get_by_text("2 of 2 memories").count() == 1)
    check("no uncaught JS errors", not errors)
    b.close()
bad = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(bad)}/{len(results)} browser checks passed"); sys.exit(1 if bad else 0)

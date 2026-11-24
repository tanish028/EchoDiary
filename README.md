# EchoDiary — Your Voice. Your Emotions. Your Story.

# EchoDiary — Your Voice. Your Emotions. Your Story.

 **Live Demo:** [echodiary.streamlit.app](https://echodiary.streamlit.app/)

**The problem it solves.** Voice notes, photos and written thoughts about the same day end up in three different
apps, and none of them lets you *look back* — find "that sunrise walk", see which memories look or sound alike, or
understand your own recordings. EchoDiary keeps one diary entry per moment (audio + transcript + photo + a mood *you*
choose) and uses classical signal-processing and text-retrieval methods from the DA321 course (Multimodal Data
Processing and Learning I, IIT Guwahati) to explore it. **No deep learning, speech recognition, emotion detection,
object/face recognition or external AI services are used.**

## Features
| Page | What you can do |
|---|---|
| **Home** | Library overview, mood chart, timeline, library check (files vs database), optional demo data |
| **Create Memory** | Title, date, mood, WAV upload **or microphone recording**, transcript (typed — there is no speech recognition), photo, notes |
| **My Memories** | Search by transcript (TF-IDF cosine; English, Hindi, mixed), filter by mood/date, open a full entry, **edit** (title, mood, date, notes, transcript, replace/remove audio and photo — with confirmation), delete |
| **Memory Analysis** | *Audio:* waveform, STFT spectrogram, single-frame spectrum, short-time energy, zero-crossing rate, autocorrelation/pitch peak. *Text:* BoW, TF, IDF, TF-IDF keywords, **vocabulary table**, n-grams. *Image:* RGB channels, **grayscale histogram + comparison**, **point processing** (brightness/contrast, negative, gamma, log, transfer curve), **Gaussian and median filtering** (before/after), **Sobel** (Gx, Gy, magnitude, edge map), 2-D Hann/DFT/magnitude/phase/frequency-domain filtering/inverse |
| **Visual Search** | Upload a photo (or pick a saved one) and rank your saved photos by *low-level* visual similarity (intensity histogram + Sobel edge statistics, adjustable weights); open the matching diary entry |
| **Multimodal Insights** | Per-entry audio/text/image summary, rank among your other entries, and cross-entry similarity combining all three modalities |

Every analysis tab labels which parts are course concepts and which are engineering decisions; the full list is in
[`COURSE_MAPPING.md`](COURSE_MAPPING.md).

## Architecture
```
app.py                  Streamlit entry (navigation, startup, CSS)
views/                  one file per page (dashboard, create_memory, my_memories, analysis, visual_search, insights)
components/             shared UI: cards (HTML-escaped), charts, common loaders (missing/corrupt media handling), sidebar
modules/
  config.py             paths, limits, moods, logging
  audio/                preprocessing (WAV decode, truncation metadata), framing/windows, stft, features, visualisation
  text/                 preprocessing (Unicode tokeniser), bow, tfidf, ngrams, retrieval, corpus (single shared corpus)
  image/                preprocessing, histogram, point, spatial (Gaussian/median), edges (Sobel), fourier, filtering, visual (descriptors + ranking)
  multimodal/           late-fusion similarity, mood summary
  database/db.py        SQLite (schema v2, in-place migration), media files, orphan checks
  pipeline.py           create / edit / backfill workflows with safe file handling
scripts/seed_demo.py    synthetic demo entries (marked [DEMO])
tests/                  pytest suite + tests/e2e_browser.py (Playwright)
data/                   echodiary.db, audio/, images/  (your data; override with ECHODIARY_DATA)
```
Stack: Python 3.11+, Streamlit, NumPy, SciPy (WAV reading only), Pillow (image decoding/resizing), Plotly, pandas,
SQLite. The spatial filters, Sobel, histograms and descriptors are implemented in NumPy, not library calls.

## Install and run
```bash
cd EchoDiary
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
`streamlit run /full/path/to/EchoDiary/app.py` also works from any directory. Open http://localhost:8501.
Microphone recording needs a browser with microphone permission (localhost counts as secure).

## Run the tests
```bash
python -m pytest -q tests                       # unit + workflow + headless-UI tests (no server needed)
python tests/e2e_browser.py http://localhost:8501 /path/to/chrome   # optional real-browser check (pip install playwright; app must be running with an EMPTY data folder)
```

## Using the app
1. **Create → Save.** Fill title/mood, add any of audio / transcript / photo (at least one). Longer WAVs than 180 s are stored whole but analysed on the first 180 s, and the app tells you.
2. **Revisit.** My Memories → Open. Edit changes are applied in place (same ID); only changed parts are re-analysed; replacing or removing media asks for confirmation and old files are deleted only after the change is saved.
3. **Analyse.** Memory Analysis → choose an entry → Audio / Text / Image tabs.
4. **Search.** Text: My Memories search box. Visual: Visual Search page (it says clearly that it compares histograms and edges, not content).
5. **Compare.** Multimodal Insights.

Supported media: audio = **PCM/float WAV** (8/16/32-bit integer or 32-bit float, mono or stereo, ≥ 4 kHz); photos = PNG, JPEG, WebP, BMP.
MP3/M4A/OGG are not supported — convert to WAV first.

## Limitations
- No speech recognition: transcripts are typed. No emotion detection: mood is whatever you select.
- Visual similarity is low-level (intensity + edge statistics). It can rank unrelated scenes as similar and cannot recognise objects, people or places.
- Audio analysis is capped at 180 s; the spectrogram *display* is downsampled for responsiveness (the STFT is not).
- Hindi support is tokenisation and TF-IDF only (no stemming/lemmatisation); the English and Hindi stop-word lists are small hand-written lists.
- The browser test script assumes an empty data folder; browsers other than Chrome were not tested.
- The IDF log base (10) and some slide formulas were taken from the earlier audit and are not re-verified against the slides (see COURSE_MAPPING.md).

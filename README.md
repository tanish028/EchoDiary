EchoDiary

Your Voice. Your Emotions. Your Story.

Live Demo: https://echodiary.streamlit.app/

EchoDiary is a multimodal digital diary. Each entry can hold a voice recording, a text transcript, a photo, and a mood you pick yourself. The app stores entries in a local SQLite database and lets you look back through them: search by what you wrote, find photos that look alike, inspect the audio and images, and see which entries are similar across all three modalities.

It was built for DA321 (Multimodal Data Processing and Learning I) at IIT Guwahati, so everything is done with classical signal processing and text retrieval, not machine learning. There is no speech recognition (you type the transcript), no emotion detection (the mood is whatever you choose), and no object or face recognition.

What you can do
Create a memory with a title, date, mood, a WAV file or a microphone recording, a transcript, a photo and optional notes. At least one of audio, transcript or photo is required.
Browse and search your entries. Search compares your query against the transcripts with TF-IDF and cosine similarity, and works for English, Hindi (Devanagari) and mixed text. You can also filter by mood and date.
Open and edit an entry. Title, mood, date, notes and transcript can be changed, and the audio and photo can be replaced or removed. Only the parts that changed are re-analysed, and replacing or removing a file asks for confirmation.
Analyse an entry in the Memory Analysis page, which has separate audio, text and image sections (details below).
Search by photo on the Visual Search page: upload an image, or pick one you already saved, and get your most similar saved photos back.
Compare across modalities on the Insights page, which shows a per-entry summary and a list of similar entries.
Check the library from the home page, which lists files missing from disk and files that no entry uses, and can remove the unused ones after confirmation. It can also load a few synthetic demo entries.
How each modality is processed

Audio. The WAV is decoded to mono, then split into 25 ms frames with a 10 ms hop. The spectrogram is a Hann-windowed STFT shown in dB, and you can switch the window (Hann, Hamming, rectangular), frame length and hop. The page also shows a single-frame spectrum, short-time energy, zero-crossing rate, and autocorrelation on the loudest 40 ms with a simple pitch-peak estimate. Recordings longer than 180 seconds are stored in full but only the first 180 seconds are analysed, and the app shows both durations. The spectrogram is trimmed to 8 kHz and averaged over frames when it is very wide, so the plot stays light in the browser. This only affects what is drawn, not the STFT itself.

Text. Transcripts are tokenised with a Unicode-aware tokeniser that handles English, Devanagari and accented Latin text. From there the pipeline is bag-of-words, vocabulary, term frequency, IDF (log10 of N over document frequency) and TF-IDF, plus word n-grams and character bigrams. Search, Analysis and Insights all use the same corpus, which is the transcripts of all entries. Entries with no words are not counted in N. The vocabulary table (counts, document frequency, IDF) is visible in the Text tab. English and Hindi stop-word lists are small hand-written ones. There is no stemming, lemmatisation or BM25.

Images. Photos are converted to grayscale (mean of R, G and B) and resized with the aspect ratio kept. The image tabs cover:

RGB channel split.
Grayscale histogram with a plain-language description, and a comparison against another entry's photo.
Point processing: brightness/contrast, negative, gamma and log, with the transformation curve and before/after histograms.
Gaussian filtering (separable kernel, size about 6 sigma) and median filtering, with an optional salt-and-pepper noise demo. Both are written in NumPy.
Sobel edge detection: horizontal gradient, vertical gradient, magnitude and a thresholded edge map.
Fourier tools: 2-D Hann window, DFT, log magnitude, phase, low/high/band-pass and orientation masks, and the inverse transform. High-pass and band-pass remove the mean brightness, so there is a switch to add it back for display.

Visual search. Each photo gets a small descriptor made from a 32-bin intensity histogram, a 16-bin Sobel magnitude histogram, an 8-bin edge orientation histogram, overall edge density and a 3x3 grid of edge densities. Descriptors are saved in the database. Distances between two photos are normalised to the range 0 to 1 and combined with adjustable weights (histogram vs. edges, default 50/50) into a similarity score. This compares intensity and edge statistics only, so it does not know what is in a photo. Whole-image cross-correlation is not used.

Multimodal similarity. To find similar entries, the app scores each modality separately and averages the scores with weights you set on the Insights page. Audio uses cosine similarity on standardised descriptors (RMS, spectral centroid, energy and ZCR statistics and so on), text uses TF-IDF cosine, and images use the visual-search similarity. If an entry is missing a modality, that modality is left out and the weights are renormalised.

COURSE_MAPPING.md lists each feature and says whether it comes directly from the course material or is my own design choice (descriptor layout, weights, stop-word lists, thresholds and so on).

Project structure
app.py                  entry point: navigation, startup, styling
views/                  one file per page (dashboard, create_memory, my_memories,
                        analysis, visual_search, insights)
components/             shared UI pieces: cards, charts, media loading, sidebar
modules/
  config.py             paths, limits, moods, logging
  pipeline.py           create / edit / backfill workflows
  audio/                WAV loading, framing and windows, STFT, features, plots
  text/                 tokeniser, BoW, TF-IDF, n-grams, retrieval, shared corpus
  image/                histogram, point ops, Gaussian/median, Sobel, Fourier,
                        frequency filters, visual descriptors and ranking
  multimodal/           cross-modality similarity, mood summary
  database/db.py        SQLite storage and file handling
scripts/seed_demo.py    synthetic demo entries
tests/                  pytest suite and a Playwright browser script
data/                   echodiary.db, audio/, images/

Built with Python, Streamlit, NumPy, SciPy (only for reading WAV files), Pillow (image decoding and resizing), Plotly, pandas and SQLite. See requirements.txt for versions.

Running it locally
bash
git clone <repo-url>
cd EchoDiary
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

Then open http://localhost:8501. Entries are stored under data/, or in the folder named by the ECHODIARY_DATA environment variable if it is set. Microphone recording needs browser permission for the microphone.

Supported audio is WAV (8/16/32-bit integer or 32-bit float, mono or stereo, at least 4 kHz). Supported photos are PNG, JPEG, WebP and BMP. MP3, M4A and OGG are not supported, so convert those to WAV first.

Tests
bash
python -m pytest -q tests

This runs unit tests for the audio, text and image modules, the database and edit workflows, and headless tests of every page. There is also a browser test that drives the real app with Playwright:

bash
pip install playwright
python tests/e2e_browser.py http://localhost:8501 /path/to/chrome

Run that one against an app with an empty data folder, because it creates and deletes its own entries.

Limitations
Transcripts are typed by hand and moods are chosen by the user. Nothing is inferred from the audio or photo.
Visual search only compares histograms and edge statistics. It can rank unrelated scenes as similar, and it cannot recognise objects, people or places.
Audio analysis stops at 180 seconds, and only WAV files are accepted.
Hindi support covers tokenisation and TF-IDF search. There is no stemming, so different word forms are treated as different words.
The stop-word lists are short and hand-written.
The browser test was only run in Chrome.
Data is stored in a local SQLite file, so there are no accounts or syncing. On a hosted deployment such as Streamlit Cloud, the storage may not persist between restarts.

# EchoDiary
### Your Voice. Your Emotions. Your Story.

 **Live Demo:** [https://echodiary.streamlit.app/](https://echodiary.streamlit.app/)

EchoDiary is a multimodal digital diary. Each entry can hold a voice recording, a text transcript, a photo, and a mood you pick yourself.

The app stores entries in a local SQLite database and lets you look back through them: search by what you wrote, find photos that look alike, inspect the audio and images, and see which entries are similar across all three modalities.

It was built for **DA321 (Multimodal Data Processing and Learning I) at IIT Guwahati**, so everything is done with classical signal processing and text retrieval, not machine learning.

There is:
- No speech recognition (you type the transcript).
- No emotion detection (the mood is whatever you choose).
- No object or face recognition.

---

## What You Can Do

- **Create a memory:** Add a title, date, mood, WAV file or microphone recording, transcript, photo, and optional notes. At least one of audio, transcript, or photo is required.
- **Browse and search entries:** Search compares your query against transcripts using TF-IDF and cosine similarity. It supports English, Hindi (Devanagari), and mixed text. You can also filter by mood and date.
- **Open and edit entries:** Change the title, mood, date, notes, and transcript. Audio and photos can also be replaced or removed. Only changed parts are re-analysed, and replacing or removing a file requires confirmation.
- **Analyse memories:** The Memory Analysis page has separate audio, text, and image sections.
- **Search by photo:** Upload an image or select one already saved to find the most similar photos in your collection.
- **Compare across modalities:** The Insights page shows per-entry summaries and similar entries.
- **Manage the library:** The home page identifies files missing from disk and files not used by any entry. Unused files can be removed after confirmation. It can also load synthetic demo entries.

---

## How Each Modality Is Processed

### 1. Audio Processing

The WAV file is decoded to mono and divided into **25 ms frames with a 10 ms hop**.

The audio analysis includes:

- **Spectrogram:** Hann-windowed Short-Time Fourier Transform (STFT), displayed in dB.
- **Window selection:** Switch between Hann, Hamming, and rectangular windows.
- **Frame configuration:** Adjust frame length and hop size.
- **Single-frame spectrum.**
- **Short-time energy.**
- **Zero-crossing rate (ZCR).**
- **Autocorrelation:** Calculated on the loudest 40 ms segment, with a simple pitch-peak estimate.

Recordings longer than 180 seconds are stored in full, but only the first 180 seconds are analysed. The app displays both durations.

To keep plots lightweight in the browser, the spectrogram is trimmed to 8 kHz and averaged over frames when it becomes very wide. This affects only the visualisation, not the STFT itself.

### 2. Text Processing

Transcripts are processed using a Unicode-aware tokeniser that handles English, Devanagari, and accented Latin text.

The text-processing pipeline includes:

- Bag of Words (BoW).
- Vocabulary generation.
- Term Frequency (TF).
- Inverse Document Frequency (IDF).
- TF-IDF.
- Word n-grams.
- Character bigrams.

IDF is calculated using:

\[
IDF(t) = \log_{10}\left(\frac{N}{df(t)}\right)
\]

where \(N\) is the number of entries containing words and \(df(t)\) is the number of documents containing the term.

Search, Analysis, and Insights all use the same corpus: the transcripts of all entries. Entries with no words are not counted in \(N\).

The Text tab displays a vocabulary table containing term counts, document frequency, and IDF.

English and Hindi stop-word lists are small and hand-written. There is no stemming, lemmatisation, or BM25.

### 3. Image Processing

Photos are converted to grayscale using the mean of their R, G, and B channels and resized while preserving their aspect ratio.

The image-processing section includes:

**RGB Channel Split**
- Separate red, green, and blue channels.

**Grayscale Histogram**
- Display the grayscale intensity histogram with a plain-language description.
- Compare the histogram against another entry's photo.

**Point Processing**
- Brightness and contrast adjustment.
- Negative transformation.
- Gamma transformation.
- Log transformation.
- Transformation curves and before/after histograms.

**Image Filtering**
- Gaussian filtering using a separable kernel (size approximately 6 sigma).
- Median filtering.
- Optional salt-and-pepper noise demonstration.

Both Gaussian and median filtering are implemented in NumPy.

**Sobel Edge Detection**
- Horizontal gradient.
- Vertical gradient.
- Gradient magnitude.
- Thresholded edge map.

**Fourier Analysis**
- 2D Hann window.
- Discrete Fourier Transform (DFT).
- Log magnitude spectrum.
- Phase spectrum.
- Low-pass, high-pass, band-pass, and orientation masks.
- Inverse transform.

High-pass and band-pass filters remove the mean brightness, so an option is provided to add it back for display.

### 4. Visual Search

Each photo is represented by a compact descriptor containing:

- A 32-bin intensity histogram.
- A 16-bin Sobel magnitude histogram.
- An 8-bin edge orientation histogram.
- Overall edge density.
- A 3 × 3 grid of edge densities.

Descriptors are saved in the database.

Distances between two photos are normalised to the range 0–1 and combined using adjustable weights for histogram and edge information. The default weighting is 50/50.

This method compares intensity and edge statistics only. It does not understand the contents of a photo or recognise objects, people, or places.

Whole-image cross-correlation is not used.

### 5. Multimodal Similarity

The app finds similar entries by scoring each modality separately and combining the results.

- **Audio:** Cosine similarity on standardised descriptors, including RMS, spectral centroid, energy, and ZCR statistics.
- **Text:** TF-IDF cosine similarity.
- **Images:** Visual-search similarity.

The Insights page allows users to adjust the weights assigned to each modality.

If an entry is missing a modality, that modality is excluded and the remaining weights are renormalised.

---

## Course Mapping

`COURSE_MAPPING.md` lists each feature and identifies whether it comes directly from the course material or is an implementation/design choice.

Examples of design choices include:
- Descriptor layout.
- Similarity weights.
- Stop-word lists.
- Thresholds.

---

## Project Structure

```text
EchoDiary/
│
├── app.py                  # Entry point: navigation, startup, styling
│
├── views/                  # One file per page
│   ├── dashboard
│   ├── create_memory
│   ├── my_memories
│   ├── analysis
│   ├── visual_search
│   └── insights
│
├── components/             # Shared UI components
│                           # Cards, charts, media loading, sidebar
│
├── modules/
│   ├── config.py           # Paths, limits, moods, logging
│   ├── pipeline.py         # Create, edit, and backfill workflows
│   │
│   ├── audio/              # WAV loading, framing, windows,
│   │                       # STFT, features, and plots
│   │
│   ├── text/               # Tokeniser, BoW, TF-IDF, n-grams,
│   │                       # retrieval, and shared corpus
│   │
│   ├── image/              # Histogram, point operations,
│   │                       # Gaussian/median filters, Sobel,
│   │                       # Fourier, frequency filters,
│   │                       # visual descriptors and ranking
│   │
│   ├── multimodal/         # Cross-modality similarity, mood summary
│   │
│   └── database/
│       └── db.py           # SQLite storage and file handling
│
├── scripts/
│   └── seed_demo.py        # Synthetic demo entries
│
├── tests/                  # Pytest suite and Playwright browser script
│
├── data/
│   ├── echodiary.db
│   ├── audio/
│   └── images/
│
├── requirements.txt
├── README.md
└── COURSE_MAPPING.md
```

---

## Technologies Used

The project is built with:

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | User interface |
| NumPy | Numerical operations and signal/image processing |
| SciPy | WAV file reading |
| Pillow | Image decoding and resizing |
| Plotly | Interactive visualisations |
| pandas | Data handling |
| SQLite | Local database storage |

See `requirements.txt` for dependency versions.

---

## Running Locally

### 1. Clone the repository

```bash
git clone <repo-url>
cd EchoDiary
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the application at:

`http://localhost:8501`

Entries are stored under `data/`, or in the folder specified by the `ECHODIARY_DATA` environment variable if it is set.

Microphone recording requires browser permission for microphone access.

### Supported File Formats

**Audio**
- WAV: 8-bit, 16-bit, or 32-bit integer; 32-bit float.
- Mono or stereo.
- Sample rate of at least 4 kHz.

MP3, M4A, and OGG are not supported. Convert these files to WAV before uploading.

**Images**
- PNG.
- JPEG.
- WebP.
- BMP.

---

## Tests

Run the test suite using:

```bash
python -m pytest -q tests
```

This runs unit tests for:
- Audio, text, and image modules.
- Database operations.
- Edit workflows.
- Headless tests of every page.

There is also a browser test that drives the actual application using Playwright.

### Running the Browser Test

Install Playwright:

```bash
pip install playwright
```

Then run:

```bash
python tests/e2e_browser.py http://localhost:8501 /path/to/chrome
```

Run this test against an application with an empty data folder, because it creates and deletes its own entries.

---

## Limitations

- Transcripts are typed by hand, and moods are selected by the user. Nothing is inferred from the audio or photo.
- Visual search compares only histograms and edge statistics. It can rank unrelated scenes as similar and cannot recognise objects, people, or places.
- Audio analysis is limited to 180 seconds, and only WAV files are accepted.
- Hindi support covers tokenisation and TF-IDF search. There is no stemming, so different word forms are treated as different words.
- The stop-word lists are short and hand-written.
- The browser test has only been run in Chrome.
- Data is stored in a local SQLite file, so there are no accounts or syncing. On a hosted deployment such as Streamlit Cloud, storage may not persist between restarts.

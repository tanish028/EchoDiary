# COURSE_MAPPING — implemented feature → DA321 source

Legend: **TAUGHT** = method appears in the inspected lecture slides or course website.
**SYLLABUS-ONLY** = listed in the syllabus; no teaching material found in what was inspected.
**ENGINEERING** = my design decision, *not* something the professor taught.
Sources: PDF1 = `iitG_DA321_M26_lect_to_studs_till_midsem.pdf`; PDF2 = `…_images_eeg_text(1).pdf`;
SAP-n = spinlabiitg.github.io/sap module n; VIP-n = spinlabiitg.github.io/vip module n.
Page numbers come from the earlier audit (`EchoDiary_DA321_Technical_Audit.md`); slide equations were read from
low-resolution contact sheets, so re-check exact values against the slides before quoting them.

## Audio
| Feature (code) | DA321 concept | Reference | Method implemented | Status / engineering assumptions |
|---|---|---|---|---|
| Framing (`framing.py`) | short-term analysis, L and H | SAP-1; PDF1 pp.435-439 | overlapping frames of L samples, hop H, zero-padded tail | TAUGHT. Defaults 25 ms / 10 ms from the spectrogram figure (ENGINEERING default). |
| Windows | Hann, Hamming, rectangular | SAP-1 | Hann 0.5−0.5cos(2πn/L); Hamming 0.54−0.46cos(2πn/L); 0≤n<L | TAUGHT (formulas as quoted in the audit). |
| STFT (`stft.py`) | STFT | SAP-1; PDF1 pp.435-439 | window each frame → FFT (`numpy.fft.rfft`) with N = next power of two ≥ L | TAUGHT. The FFT is the fast DFT (PDF1 pp.389-425); the library is only a tool. |
| Spectrogram | dB magnitude spectrogram | SAP-1 | S_dB = 20·log10\|X\|, user-set dynamic range, Magma-like colour map | TAUGHT; colour map is ENGINEERING. The plot is trimmed to ≤ 8 kHz and neighbouring frames are averaged above 800 columns **for display only** (ENGINEERING; the STFT itself is unchanged). |
| Truncation | — | — | recordings longer than 180 s are analysed on the first 180 s; the original duration is stored separately and a warning is shown | ENGINEERING. |
| Frequency view | magnitude spectrum of a frame | SAP-1 | one windowed frame → \|FFT\| in dB | TAUGHT. |
| Short-time energy (`features.py`) | *energy* (syllabus) | — (IEEE 1386250 abstract uses RMS) | E[m] = Σ(x·w)²/L with Hann window by default (Hamming selectable) | SYLLABUS-ONLY. Textbook definition; not reproduced from a lecture. |
| Zero-crossing rate | *zero crossings* (syllabus) | — (IEEE 1386250 abstract) | fraction of adjacent samples with a sign change per frame | SYLLABUS-ONLY. The paper's speech/music classifier is **not** implemented. |
| Autocorrelation | autocorrelation | SAP-4 | r(k) = Σ x[n]x[n+k] / Σ x[n]² on the loudest 40 ms | TAUGHT. Choice of segment = ENGINEERING. |
| Pitch-like peak | autocorrelation / F0 concept | SAP-3, SAP-4 | highest r(k) with lag in 70–400 Hz | ENGINEERING (peak picking is not taught). |
| Descriptors (RMS, centroid, active fraction, mean/std of energy & ZCR) | — | — | stored per entry | ENGINEERING summary statistics. |

## Text
| Feature | DA321 concept | Reference | Method | Status / assumptions |
|---|---|---|---|---|
| Tokenisation (`preprocessing.py`) | words as units | PDF2 pp.198-206 | Unicode NFC, lower-case; a word is a run of letters/digits/combining marks (Devanagari matras kept) with internal apostrophes; punctuation/emoji dropped | Words-as-units TAUGHT; **the Unicode rules are ENGINEERING**. No stemming/lemmatisation/subwords. |
| Stop-words | BoW vocabulary without stop-words | PDF2 pp.203-206 | short hand-written English and Hindi lists | Concept TAUGHT; both lists are ENGINEERING. |
| Vocabulary view (`bow.vocabulary_table`) | vocabulary construction | PDF2 pp.203-206 | every distinct term with count and document frequency, shown in Analysis | TAUGHT concept; the table is a display. |
| One shared corpus (`corpus.py`) | IDF over a document collection | PDF2 pp.210-213 | corpus = transcripts of all entries (titles/notes excluded); N counts only entries that contain at least one token; Search, Analysis and Insights all use it | ENGINEERING decisions on what the corpus is. |
| Bag-of-Words (`bow.py`) | BoW | PDF2 pp.198-206 | multiset of tokens, order lost | TAUGHT. |
| TF | term frequency | PDF2 p.210 | count / total words | TAUGHT. |
| IDF | inverse document frequency | PDF2 p.211 | log₁₀(N/DF) | TAUGHT; **base 10 inferred** from the worked example (0.48 ≈ log₁₀3) — confirm on the slide. |
| TF-IDF | TF-IDF | PDF2 pp.212-213 | TF×IDF; terms in every document score 0 | TAUGHT. |
| N-grams (`ngrams.py`) | bag of n-grams | PDF2 pp.215-216 | word uni/bi/trigram counts | TAUGHT. |
| Character bigrams | character bigram model | PDF2 pp.190-196 | counts only | TAUGHT (counting); no smoothing taught, none applied. |
| Vector-space search (`retrieval.py`) | vector spaces, inner product, norm | PDF1 pp.152-225; PDF2 p.213 | cosine of TF-IDF vectors; unseen query words ignored; zero vectors → score 0 | Inner product/norm TAUGHT; **cosine ranking = ENGINEERING**. |
| TF fallback | — | — | if all TF-IDF scores are 0 (1 entry / words in every entry) rank by TF cosine | ENGINEERING; shown to the user. |
| Not implemented | BM-25, lemmatisation, morphology, subwords, LM smoothing, "Hipf's law" | — | — | SYLLABUS-ONLY / no teaching material found ("Hipf's law" appears in no source; spelling kept). |

## Image
| Feature | DA321 concept | Reference | Method | Status / assumptions |
|---|---|---|---|---|
| Loading / RGB | image as M×N×C array; channels | PDF2 pp.39-43 | Pillow → NumPy; channel display | TAUGHT (library is a tool). |
| Grayscale | mean of channels | PDF2 p.44 | `rgb.mean(axis=2)` | TAUGHT. |
| Aspect-preserving resize | — | — | longer side → 256/512 px (bilinear), no square padding; the DFT is computed on the M×N array | ENGINEERING step (replaces the earlier square stretch). |
| 2-D Hann window | 2-D Hann | VIP-2 | w(n)=sin²(πn/N); g = μ + (f−μ)w(x)w(y) | TAUGHT. |
| 2-D DFT | 2-D DFT | VIP-2 | `numpy.fft.fft2` + `fftshift` | TAUGHT (FFT = fast DFT). |
| Magnitude / phase | display conventions | VIP-2 | log(1+\|F\|); phase on [−π, π] | TAUGHT. Display clipping at 99.9th percentile = ENGINEERING. |
| Filtering (`filtering.py`) | frequency-plane masks | VIP-2 | low/high/band-pass (circular) and orientation (wedge) masks; inverse DFT. High-pass/band-pass remove DC; the mean is added back **for display only** (switchable) | Mask kinds TAUGHT; exact shapes, hard edges and the mean restoration = ENGINEERING. |
| Grayscale histogram (`histogram.py`) | intensity histogram | PDF2 pp.115-117 | counts per intensity bin; normalised for comparison | Histogram TAUGHT. The plain-language reading and the half-L1 distance are ENGINEERING. |
| Point processing (`point.py`) | negative, log, gamma/power-law | PDF2 pp.100-114 | s = T(r) with transfer-curve plot | Transforms TAUGHT as lecture topics (exact slide formulas not re-verified); the **brightness/contrast parametrisation s = c·(r−128)+128+b is ENGINEERING**. |
| Gaussian filter (`spatial.py`) | convolution, Gaussian kernel, separability | PDF2 pp.130-149 | kernel (1/2πσ²)exp(−(i²+j²)/2σ²), half-width ⌈3σ⌉ (K≈6σ), applied as two 1-D passes; reflect padding | Kernel/size rule/separability TAUGHT per audit; reflect border is one of the taught border options but the choice is ENGINEERING. Implemented in NumPy, not a library call. |
| Median filter (`spatial.py`) | median filter | PDF2 pp.150-153 | sliding-window median, odd sizes 3–9 | TAUGHT. Salt-and-pepper demo noise = ENGINEERING. |
| Sobel (`edges.py`) | Sobel gradient, magnitude, orientation | PDF2 pp.118-129 | Gx, Gy, √(Gx²+Gy²), atan2; threshold edge map | Sobel/magnitude TAUGHT. The threshold edge map is ENGINEERING. Roberts and Prewitt are **not** implemented. |
| Descriptors | — | — | mean RGB, brightness, contrast, low/high-frequency magnitude share, spectral centroid, orientation balance | ENGINEERING summaries of taught quantities. |
| Visual Memory Search (`visual.py`) | histogram + Sobel descriptors | PDF2 pp.115-128 | grayscale copy (longer side 128) → 32-bin intensity histogram, 16-bin Sobel-magnitude histogram, 8-bin orientation histogram, edge density, 3×3 edge-density grid; distances bounded in [0,1]; similarity = 1 − (w_h·d_hist + w_e·d_edge) with d_edge = 0.4·strength + 0.3·direction + 0.3·layout, default w = 0.5/0.5; descriptors stored in SQLite | The ingredients (histograms, Sobel) are TAUGHT. **Everything about the descriptor layout, distances, weights and ranking is ENGINEERING.** Whole-image normalised cross-correlation (template matching, PDF2 pp.154-161) is deliberately **not** used. This is low-level similarity, not recognition. |
| Not implemented | morphology, restoration, Roberts/Prewitt, NCC search, learned or semantic features | — | — | Out of scope. |

## Multimodal & other
| Feature | Source | Method | Status |
|---|---|---|---|
| Per-modality representations | audio descriptor vector, TF-IDF vector, image descriptor vector | stored/compared separately | ENGINEERING. No multimodal-learning method was found in the supplied material. |
| Late-fusion similarity (`integration.py`) | inner product / norm (PDF1) | audio: z-scored descriptors → cosine; text: TF-IDF cosine; image: the Visual Search similarity (falls back to descriptor cosine for entries without stored descriptors); weighted mean over modalities both entries have | ENGINEERING design decision. |
| Editing and media replacement (`pipeline.update_memory`) | — | in-place update, only changed modalities recomputed; new files written first, DB transaction, old files deleted after commit; failures roll back | ENGINEERING. |
| Percentile profile (position among the *other* entries, needs ≥ 3 others), mood table | — | descriptive statistics only | ENGINEERING; not validated measures of emotion or memory. |
| Mood | — | user-selected label | ENGINEERING; no emotion recognition. |
| SQLite, Streamlit, Plotly, CSS | — | persistence/UI | ENGINEERING. |

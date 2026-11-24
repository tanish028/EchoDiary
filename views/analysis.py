import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts
from components.cards import header_line, hero, note
from components.common import empty_state, load_entry_audio, load_entry_image, pick_entry
from modules.audio import visualisation as av
from modules.audio.features import autocorrelation, loudest_segment, pitch_from_autocorr
from modules.audio.stft import stft
from modules.database import db
from modules.image import edges, fourier, filtering, point, spatial
from modules.image.histogram import describe_histogram, gray_histogram, histogram_distance, histogram_stats
from modules.image.preprocessing import channels, fit_gray
from modules.text.bow import vocabulary_table
from modules.text.corpus import build_corpus
from modules.text.ngrams import char_bigrams, text_stats, word_ngrams
from modules.text.tfidf import term_frequency, top_keywords
from modules.viz_style import style

TAUGHT = "<span class='taught'>taught</span>"
ENG = "<span class='eng'>engineering</span>"

hero("Memory analysis", "Audio, text and image pipelines built from DA321 concepts.")
entries = db.list_entries()
if not entries:
    empty_state(); st.stop()
e = pick_entry(entries, "analysis_pick")
st.markdown(header_line(e), unsafe_allow_html=True)
cols = st.columns(3)
cols[0].page_link("views/my_memories.py", label="Open full entry", icon="📚")
cols[1].page_link("views/visual_search.py", label="Find visually similar photos", icon="🔎")
cols[2].page_link("views/insights.py", label="Multimodal insights", icon="🔗")

t_audio, t_text, t_image = st.tabs(["🎙️ Audio", "📝 Text", "🖼️ Image"])

# ---------------------------------------------------------------- AUDIO
with t_audio:
    ad = load_entry_audio(e)
    if ad is None:
        if not e["audio_path"]:
            st.info("This memory has no audio.")
    else:
        x, fs = ad.x, ad.fs
        note("Pipeline (SAP-1, PDF1 pp.435-439): <b>audio → framing → windowing → Fourier transform → "
             "magnitude → spectrogram</b>. Windows: Hann for the spectrogram and energy; ZCR uses raw samples; "
             "autocorrelation uses the raw 40 ms segment. Energy / ZCR use textbook definitions "
             "<span class='eng'>syllabus-listed, no teaching material found</span>; autocorrelation is "
             f"{TAUGHT} (SAP-4).")
        if ad.truncated:
            st.warning(f"This recording is {ad.original_duration_s:.0f} s long. The stored file is complete (the "
                       f"player below plays all of it), but the analysis covers only the first "
                       f"{ad.analysed_duration_s:.0f} s.")
        st.caption(f"Stored file: {ad.original_duration_s:.1f} s · analysed: {ad.analysed_duration_s:.1f} s · "
                   f"{fs} Hz")
        st.audio(str(db.abs_path(e["audio_path"])))
        st.plotly_chart(av.waveform_fig(x, fs), width="stretch", theme=None)

        c1, c2, c3, c4 = st.columns(4)
        window = c1.selectbox("Window", ["hann", "hamming", "rectangular"], key="win")
        frame_ms = c2.select_slider("Frame length L (ms)", [10, 15, 25, 32, 50, 100], value=25,
                                    help="Short = wideband (time detail); long = narrowband (frequency detail).")
        hop_ms = c3.select_slider("Hop H (ms)", [2.5, 5, 10, 12.5, 25], value=10)
        dyn = c4.slider("Dynamic range (dB)", 30, 90, 70)
        sp = stft(x, fs, frame_ms, hop_ms, window)
        st.plotly_chart(av.spectrogram_fig(sp, dyn, max_hz=min(fs / 2, 8000)), width="stretch", theme=None)
        st.caption("For responsiveness the plot shows frequencies up to 8 kHz and averages neighbouring frames when "
                   "there are more than 800; the underlying STFT is unchanged.")

        dur = len(x) / fs
        t0 = st.slider("Inspect one frame at time t (s)", 0.0, float(max(dur - 0.03, 0.01)),
                       float(min(dur / 2, max(dur - 0.03, 0.0))), 0.01)
        st.plotly_chart(av.frame_spectrum_fig(x, fs, t0, window), width="stretch", theme=None)
        st.plotly_chart(av.energy_zcr_fig(x, fs), width="stretch", theme=None)

        start, seg = loudest_segment(x, fs, 40.0)
        r = autocorrelation(seg, max_lag=int(fs * 0.025))
        f0, strength = pitch_from_autocorr(r, fs)
        st.plotly_chart(av.autocorr_fig(seg, fs, f0=f0), width="stretch", theme=None)
        if f0 and strength > 0.3:
            st.caption(f"Strongest autocorrelation peak in 70–400 Hz: r = {strength:.2f} at ≈ {f0:.0f} Hz "
                       "(peak-picking is an engineering extension of the taught r(k)).")
        else:
            st.caption("No clear periodic peak in the loudest 40 ms (typical for noise or unvoiced sound).")
        if e["audio_features"]:
            with st.expander("Stored audio descriptors"):
                st.dataframe(pd.DataFrame(e["audio_features"], index=["value"]).T, width="stretch")

# ---------------------------------------------------------------- TEXT
with t_text:
    text = e["transcript"]
    if not text.strip():
        st.info("This memory has no transcript.")
    else:
        note("BoW (PDF2 pp.198-206), vocabulary (pp.203-206), TF = count / total words, IDF = log₁₀(N / DF), "
             f"TF-IDF = TF × IDF (pp.210-213), n-grams (pp.215-216) — {TAUGHT}. Unicode tokenisation and the "
             f"stop-word lists are {ENG} decisions. One corpus (all transcripts) is used on every page.")
        corpus = build_corpus(entries)
        i = corpus.index_of(e["id"])
        vec, bow = corpus.vecs[i], corpus.bows[i]
        stats = text_stats(text)
        m = st.columns(4)
        m[0].metric("Tokens", stats["n_tokens"]); m[1].metric("Unique", stats["n_unique"])
        m[2].metric("Type/token ratio", f"{stats['type_token_ratio']:.2f}")
        m[3].metric("Avg word length", f"{stats['avg_token_len']:.1f}")
        k_tab, v_tab, n_tab = st.tabs(["Keywords (TF-IDF)", "Vocabulary", "N-grams"])
        with k_tab:
            kws = top_keywords(vec, 10)
            left, right = st.columns(2)
            with left:
                n_docs = sum(1 for b in corpus.bows if b)
                if kws:
                    st.plotly_chart(charts.bar_h([k for k, _ in kws], [w for _, w in kws],
                                                 f"Top TF-IDF keywords (N={n_docs} entries)"),
                                    width="stretch", theme=None)
                else:
                    st.info("All TF-IDF weights are 0 — with a single entry (or when every word occurs in every "
                            "entry) IDF = log(N/DF) = 0, exactly as the slides state. Showing term frequency instead.")
                    tf = term_frequency(bow)
                    top = sorted(tf.items(), key=lambda p: -p[1])[:10]
                    if top:
                        st.plotly_chart(charts.bar_h([k for k, _ in top], [w for _, w in top], "Top term frequencies"),
                                        width="stretch", theme=None)
            with right:
                tf = term_frequency(bow)
                rows = [{"term": t, "count": c, "TF": round(tf[t], 4), "IDF": round(corpus.idf.get(t, 0.0), 4),
                         "TF-IDF": round(vec.get(t, 0.0), 4)} for t, c in bow.most_common(12)]
                st.markdown("**Bag-of-Words → TF / IDF / TF-IDF**")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with v_tab:
            vt = vocabulary_table(corpus.texts)
            mine = set(bow)
            st.markdown(f"**Vocabulary construction** — every distinct word (stop-words removed) across all "
                        f"{len(entries)} transcripts: **{len(vt)} terms**; this entry uses **{len(mine)}** of them.")
            only_mine = st.toggle("Show only words used in this memory", value=True)
            rows = [{"term": r["term"], "in this memory": bow.get(r["term"], 0), "total count": r["count"],
                     "document frequency (DF)": r["df"], "IDF": round(corpus.idf.get(r["term"], 0.0), 3)}
                    for r in vt if (r["term"] in mine or not only_mine)]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=300)
            top = vt[:12]
            st.plotly_chart(charts.bar_h([r["term"] for r in top], [r["df"] for r in top],
                                         "Most widespread words (document frequency)", "#38bdf8", 320),
                            width="stretch", theme=None)
        with n_tab:
            n_cols = st.columns(3)
            for col, n, name in zip(n_cols, (1, 2, 3), ("Unigrams", "Bigrams", "Trigrams")):
                with col:
                    st.caption(name)
                    st.dataframe(pd.DataFrame(word_ngrams(text, n).most_common(8), columns=["n-gram", "count"]),
                                 width="stretch", hide_index=True)
            with st.expander("Character bigram counts (PDF2 pp.190-196; counts only, no smoothing taught)"):
                st.dataframe(pd.DataFrame(char_bigrams(text).most_common(15), columns=["bigram", "count"]),
                             width="stretch", hide_index=True)

# ---------------------------------------------------------------- IMAGE
with t_image:
    rgb = load_entry_image(e)
    if rgb is None:
        if not e["image_path"]:
            st.info("This memory has no photo.")
    else:
        gray = fit_gray(rgb, 256)           # working copy: longer side 256 px, aspect ratio preserved
        h_, w_ = gray.shape
        note(f"Working image: grayscale (mean of R, G, B — {TAUGHT}, PDF2 p.44), longer side 256 px, aspect ratio "
             f"kept ({w_}×{h_}); resizing is an {ENG} step. Each tab below says which parts are taught.")
        s_over, s_hist, s_point, s_spat, s_edge, s_four = st.tabs(
            ["Overview", "Histogram", "Point processing", "Spatial filters", "Edges (Sobel)", "Fourier"])

        with s_over:
            c = st.columns([1.2, 1, 1, 1])
            c[0].image(rgb, caption=f"Photo ({rgb.shape[1]}×{rgb.shape[0]})", width="stretch")
            for col, ch, name in zip(c[1:], channels(rgb), ("Red", "Green", "Blue")):
                tint = np.zeros(rgb.shape, np.uint8)
                tint[..., ("Red", "Green", "Blue").index(name)] = ch
                col.image(tint, caption=f"{name} channel (mean {ch.mean():.0f})", width="stretch")
            st.caption(f"RGB array and channel split: {TAUGHT} (PDF2 pp.39-43).")
            if e["image_features"]:
                with st.expander("Stored image descriptors"):
                    st.dataframe(pd.DataFrame(e["image_features"], index=["value"]).T, width="stretch")

        with s_hist:
            st.caption(f"Intensity histogram of the grayscale image — {TAUGHT} (PDF2 pp.115-117). The plain-language "
                       f"reading and the comparison distance are {ENG}.")
            hist = gray_histogram(gray, 256)
            fig = go.Figure(go.Bar(x=np.arange(256), y=hist, marker_color="#818cf8", name="this photo"))
            fig.update_xaxes(title="Intensity (0 = black, 255 = white)"); fig.update_yaxes(title="Pixels")
            others = [x for x in entries if x["id"] != e["id"] and x["image_path"]]
            cmp_entry = None
            if others:
                choice = st.selectbox("Compare with the photo of another memory (optional)",
                                      [None] + [o["id"] for o in others],
                                      format_func=lambda i: "— none —" if i is None else
                                      next(f'{o["title"]} · {o["entry_date"]}' for o in others if o["id"] == i))
                cmp_entry = next((o for o in others if o["id"] == choice), None)
            st_ = histogram_stats(gray)
            if cmp_entry is not None:
                crgb = load_entry_image(cmp_entry)
                if crgb is not None:
                    cg = fit_gray(crgb, 256)
                    h1, h2 = gray_histogram(gray, 64, True), gray_histogram(cg, 64, True)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=np.arange(64) * 4, y=h1, name="this photo", fill="tozeroy",
                                              line=dict(color="#818cf8")))
                    fig2.add_trace(go.Scatter(x=np.arange(64) * 4, y=h2, name=cmp_entry["title"], fill="tozeroy",
                                              line=dict(color="#f472b6")))
                    fig2.update_xaxes(title="Intensity"); fig2.update_yaxes(title="Fraction of pixels")
                    st.plotly_chart(style(fig2, "Normalised histograms (64 bins)", 320), width="stretch", theme=None)
                    d = histogram_distance(h1, h2)
                    st.metric("Histogram distance (½·L1, 0 = identical, 1 = no overlap)", f"{d:.3f}")
                    st.caption(f"Compared photo: “{cmp_entry['title']}” — {describe_histogram(histogram_stats(cg))}")
            else:
                st.plotly_chart(style(fig, "Grayscale intensity histogram", 320), width="stretch", theme=None)
            st.write(describe_histogram(st_))

        with s_point:
            st.caption(f"Point processing g = T(f): negative, log, gamma — {TAUGHT} (PDF2 pp.100-114). The "
                       f"brightness/contrast parametrisation and the controls are {ENG}.")
            op = st.selectbox("Operation", point.OPS)
            params = {}
            if op == "brightness / contrast":
                p1, p2 = st.columns(2)
                params["contrast"] = p1.slider("Contrast (×)", 0.2, 3.0, 1.0, 0.05)
                params["brightness"] = p2.slider("Brightness (+/−)", -100, 100, 0, 5)
            elif op == "gamma":
                params["gamma"] = st.slider("γ (below 1 brightens, above 1 darkens)", 0.2, 4.0, 1.0, 0.05)
            out = point.apply_op(gray, op, **params)
            a, b, c = st.columns(3)
            a.plotly_chart(charts.image_fig(gray, "Original", "gray", 0, 255), width="stretch", theme=None)
            b.plotly_chart(charts.image_fig(out, f"After: {op}", "gray", 0, 255), width="stretch", theme=None)
            rr, ss = point.transfer_curve(op, **params)
            figc = go.Figure(go.Scatter(x=rr, y=ss, line=dict(color="#38bdf8")))
            figc.add_trace(go.Scatter(x=[0, 255], y=[0, 255], line=dict(color="rgba(148,163,184,.4)", dash="dot"),
                                      name="identity"))
            figc.update_xaxes(title="Input r"); figc.update_yaxes(title="Output s = T(r)")
            c.plotly_chart(style(figc, "Transformation curve", 330), width="stretch", theme=None)
            h0, h1 = gray_histogram(gray, 64, True), gray_histogram(out, 64, True)
            figh = go.Figure()
            figh.add_trace(go.Scatter(x=np.arange(64) * 4, y=h0, name="before", fill="tozeroy", line=dict(color="#818cf8")))
            figh.add_trace(go.Scatter(x=np.arange(64) * 4, y=h1, name="after", fill="tozeroy", line=dict(color="#f472b6")))
            st.plotly_chart(style(figh, "Histogram before / after", 260), width="stretch", theme=None)

        with s_spat:
            st.caption(f"Gaussian smoothing (kernel (1/2πσ²)·exp(−(i²+j²)/2σ²), size ≈ 6σ, separable) and the "
                       f"median filter — {TAUGHT} (PDF2 pp.133-153). Reflect padding and the demo noise are {ENG}.")
            kind = st.radio("Filter", ["Gaussian", "Median"], horizontal=True)
            noise = st.slider("Add salt-and-pepper noise first (demonstration, % of pixels)", 0, 30, 0)
            src = spatial.add_salt_pepper(gray, noise / 100) if noise else gray
            if kind == "Gaussian":
                sigma = st.slider("σ (pixels)", 0.5, 6.0, 1.5, 0.1)
                out = spatial.gaussian_filter(src, sigma)
                st.caption(f"Kernel size K = {2 * int(np.ceil(3 * sigma)) + 1} × {2 * int(np.ceil(3 * sigma)) + 1}")
            else:
                size = st.select_slider("Window size (odd)", [3, 5, 7, 9], value=3)
                with st.spinner("Filtering…"):
                    out = spatial.median_filter(src, size)
            a, b = st.columns(2)
            a.plotly_chart(charts.image_fig(src, "Input" + (" (with noise)" if noise else ""), "gray", 0, 255),
                           width="stretch", theme=None)
            b.plotly_chart(charts.image_fig(out, f"{kind} filtered", "gray", 0, 255), width="stretch", theme=None)
            if noise:
                st.caption(f"Mean absolute error vs the clean image: noisy {np.abs(src - gray).mean():.1f} → "
                           f"filtered {np.abs(out - gray).mean():.1f} (median is designed for this kind of noise).")

        with s_edge:
            st.caption(f"Sobel 3×3 kernels, gradient magnitude √(Gx²+Gy²) — {TAUGHT} (PDF2 pp.118-128). "
                       f"Thresholding the magnitude into an edge map is an {ENG} display choice.")
            gx, gy, mag, ang = edges.sobel(gray)
            a, b, c = st.columns(3)
            lim = float(np.percentile(np.abs(gx), 99) or 1)
            a.plotly_chart(charts.image_fig(gx, "Horizontal gradient Gx", "RdBu", -lim, lim), width="stretch", theme=None)
            lim_y = float(np.percentile(np.abs(gy), 99) or 1)
            b.plotly_chart(charts.image_fig(gy, "Vertical gradient Gy", "RdBu", -lim_y, lim_y), width="stretch", theme=None)
            c.plotly_chart(charts.image_fig(mag, "Gradient magnitude", "gray", 0, float(np.percentile(mag, 99.5) or 1)),
                           width="stretch", theme=None)
            thr = st.slider("Edge-map threshold on the magnitude", 10, 600, 100, 10)
            em = edges.edge_map(mag, thr)
            d1, d2 = st.columns([1, 2])
            d1.plotly_chart(charts.image_fig(em, f"Edge map (≥ {thr})", "gray", 0, 1), width="stretch", theme=None)
            d2.metric("Edge density", f"{em.mean() * 100:.1f}% of pixels")
            d2.caption("The visual-search feature compares edge statistics like this one (strength, direction and "
                       "layout) between photos; it does not recognise objects.")

        with s_four:
            st.caption(f"2-D DFT, log(1+|F|) magnitude, phase in [−π, π], 2-D Hann window sin²(πn/N), frequency-plane "
                       f"masks, inverse DFT — {TAUGHT} (VIP-2). The FFT is the fast DFT. Display clipping and mask "
                       f"shapes are {ENG}.")
            o = st.columns(3)
            N = o[0].selectbox("Longer side (pixels)", [256, 512], index=0,
                               help="The photo is resized so its longer side has this length; the aspect ratio is kept "
                                    "and the DFT is computed on the resulting M×N array (no square padding).")
            use_hann = o[1].toggle("Apply 2-D Hann window", value=True,
                                   help="Fades borders to the mean grey level to avoid edge seams (VIP-2).")
            fgray = fit_gray(rgb, N)
            F = fourier.dft2(fgray, hann=use_hann)
            g1, g2, g3 = st.columns(3)
            g1.plotly_chart(charts.image_fig(fgray, f"Grayscale ({fgray.shape[1]}×{fgray.shape[0]})", "gray", 0, 255),
                            width="stretch", theme=None)
            LM = fourier.log_magnitude(F)
            zmax = float(np.percentile(LM, 99.9))
            g2.plotly_chart(charts.image_fig(LM, "log(1 + |F|)", "Magma", 0, zmax, "log(1+|F|)"),
                            width="stretch", theme=None)
            g3.plotly_chart(charts.image_fig(fourier.phase(F), "Phase ∠F ∈ [−π, π]", "Twilight", -np.pi, np.pi, "rad"),
                            width="stretch", theme=None)
            st.caption("Colour range is clipped at the 99.9th percentile so the DC spike does not hide detail. DC "
                       "(zero frequency) is at the centre; energy far from the centre = edges and texture.")
            st.markdown("**Frequency-domain filtering**")
            f1, f2, f3 = st.columns(3)
            fk = f1.selectbox("Mask", list(filtering.KINDS))
            kw = {}
            if fk in ("low-pass", "high-pass"):
                kw["cutoff"] = f2.slider("Cut-off (fraction of Nyquist)", 0.02, 1.0, 0.2, 0.01)
            elif fk == "band-pass":
                lo, hi = f2.slider("Pass band (fraction of Nyquist)", 0.02, 1.0, (0.1, 0.4), 0.01)
                kw["cutoff"], kw["cutoff_hi"] = lo, hi
            else:
                kw["angle_deg"] = f2.slider("Spectrum direction kept (°)", 0, 179, 0, 5)
                kw["width_deg"] = f3.slider("Wedge width (°)", 5, 90, 30, 5)
            restore = True
            if fk in ("high-pass", "band-pass"):
                restore = f3.toggle("Add mean brightness back (display)", value=True,
                                    help="High-pass/band-pass remove the zero-frequency term, i.e. the mean "
                                         "brightness. Switch off to see the raw filter output.")
            outf, mask, Fm = filtering.filter_image(fgray, fk, use_hann=False, restore_mean=restore, **kw)
            h1, h2, h3 = st.columns(3)
            h1.plotly_chart(charts.image_fig(mask, "Mask (white = kept)", "gray", 0, 1), width="stretch", theme=None)
            h2.plotly_chart(charts.image_fig(fourier.log_magnitude(Fm), "Masked spectrum", "Magma", 0, zmax),
                            width="stretch", theme=None)
            if restore or fk in ("low-pass", "orientation"):
                h3.plotly_chart(charts.image_fig(np.clip(outf, 0, 255), "Reconstruction (inverse 2-D DFT)", "gray", 0, 255),
                                width="stretch", theme=None)
            else:
                lim = float(np.percentile(np.abs(outf), 99) or 1)
                h3.plotly_chart(charts.image_fig(outf, "Raw output (mean ≈ 0)", "RdBu", -lim, lim),
                                width="stretch", theme=None)

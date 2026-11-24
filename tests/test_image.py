import numpy as np
import pytest

from modules.image import edges, point, spatial, visual
from modules.image.filtering import filter_image, make_mask, swap_magnitude_phase
from modules.image.fourier import (apply_hann2d, dft2, hann1d, idft2, image_descriptors,
                                   log_magnitude, phase)
from modules.image.histogram import describe_histogram, gray_histogram, histogram_distance, histogram_stats
from modules.image.preprocessing import ImageError, fit_gray, load_image, to_grayscale
from modules.pipeline import analyse_image

from conftest import make_image


# ------------------------------------------------------------ preprocessing
def test_load_errors_and_formats():
    for bad in (b"", b"junk", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20):
        with pytest.raises(ImageError):
            load_image(bad)
    for fmt in ("PNG", "JPEG", "BMP"):
        assert load_image(make_image("gradient", (40, 30), fmt=fmt)).shape == (30, 40, 3)


def test_grayscale_is_channel_mean():
    rgb = np.zeros((40, 60, 3), np.uint8); rgb[..., 0] = 200
    assert np.isclose(to_grayscale(rgb).mean(), 200 / 3)


def test_fit_gray_preserves_aspect_ratio():
    rgb = np.zeros((100, 300, 3), np.uint8)
    g = fit_gray(rgb, 150)
    assert g.shape == (50, 150)
    assert fit_gray(np.zeros((300, 100, 3), np.uint8), 150).shape == (150, 50)


# ------------------------------------------------------------ histogram
def test_histogram_counts_and_normalisation():
    g = np.array([[0, 0], [255, 128]], float)
    h = gray_histogram(g, 256)
    assert h.sum() == 4 and h[0] == 2 and h[255] == 1 and h[128] == 1
    assert np.isclose(gray_histogram(g, 32, True).sum(), 1.0)


def test_histogram_distance_and_description():
    dark, bright = np.full((8, 8), 20.0), np.full((8, 8), 230.0)
    a, b = gray_histogram(dark, 32, True), gray_histogram(bright, 32, True)
    assert histogram_distance(a, a) == 0 and np.isclose(histogram_distance(a, b), 1.0)
    assert "dark" in describe_histogram(histogram_stats(dark)).lower()
    assert "bright" in describe_histogram(histogram_stats(bright)).lower()


# ------------------------------------------------------------ point processing
def test_point_ops():
    g = np.array([[0.0, 100.0, 255.0]])
    assert np.array_equal(point.invert(g), [[255, 155, 0]])
    assert np.array_equal(point.brightness_contrast(g, 1.0, 0.0), g)
    assert point.brightness_contrast(g, 1.0, 50)[0, 0] == 50 and point.brightness_contrast(g, 1.0, 50)[0, 2] == 255
    assert point.brightness_contrast(np.array([[128.0]]), 2.0, 0)[0, 0] == 128       # contrast pivots on 128
    assert point.gamma_correction(g, 0.5)[0, 1] > 100 > point.gamma_correction(g, 2.0)[0, 1]
    assert np.isclose(point.log_transform(g)[0, 2], 255) and point.log_transform(g)[0, 0] == 0
    for op in point.OPS:
        out = point.apply_op(g, op)
        assert out.shape == g.shape and out.min() >= 0 and out.max() <= 255
        r, s = point.transfer_curve(op)
        assert len(r) == len(s)


# ------------------------------------------------------------ spatial filters
def test_gaussian_kernel_and_filter():
    k = spatial.gaussian_kernel_2d(1.5)
    assert np.isclose(k.sum(), 1.0) and k.shape[0] == k.shape[1] == 2 * int(np.ceil(4.5)) + 1
    assert np.allclose(k, k.T)
    c = np.full((20, 20), 77.0)
    assert np.allclose(spatial.gaussian_filter(c, 2.0), 77.0)               # constant preserved at borders
    g = np.random.default_rng(0).uniform(0, 255, (40, 40))
    assert spatial.gaussian_filter(g, 2.0).std() < g.std() / 2


def test_gaussian_separable_equals_2d():
    g = np.random.default_rng(1).uniform(0, 255, (30, 30))
    full = spatial.correlate2d(g, spatial.gaussian_kernel_2d(1.2))
    assert np.allclose(full, spatial.gaussian_filter(g, 1.2), atol=1e-6)


def test_median_removes_impulse_and_validates_size():
    g = np.full((15, 15), 100.0); g[7, 7] = 255
    assert np.all(spatial.median_filter(g, 3) == 100)
    with pytest.raises(ValueError):
        spatial.median_filter(g, 4)
    noisy = spatial.add_salt_pepper(np.full((40, 40), 128.0), 0.1)
    assert (noisy != 128).any()
    assert np.abs(spatial.median_filter(noisy, 3) - 128).mean() < np.abs(noisy - 128).mean()


# ------------------------------------------------------------ Sobel
def test_sobel_step_edge():
    g = np.zeros((10, 10)); g[:, 5:] = 100
    gx, gy, mag, ang = edges.sobel(g)
    assert gx[5, 4] == 400 and gx[5, 5] == 400 and np.all(gy[2:-2, 2:-2] == 0)
    assert np.isclose(mag[5, 4], 400)
    em = edges.edge_map(mag, 100)
    assert em[5, 4] == 1 and em[5, 0] == 0 and set(np.unique(em)) <= {0, 1}
    _, gy2, _, _ = edges.sobel(g.T)
    assert gy2[4, 5] == 400                                               # horizontal edge -> vertical gradient


def test_sobel_flat_image_has_no_edges():
    _, _, mag, _ = edges.sobel(np.full((12, 12), 50.0))
    assert mag.max() == 0


# ------------------------------------------------------------ Fourier
def test_hann_properties_and_border():
    w = hann1d(8)
    assert w[0] == 0 and np.isclose(w[4], 1.0)
    f = np.random.default_rng(0).uniform(0, 255, (64, 48))
    g = apply_hann2d(f)
    assert np.allclose(g[0, :], f.mean()) and np.allclose(g[:, 0], f.mean())


def test_dft_roundtrip_non_square():
    f = np.random.default_rng(1).uniform(0, 255, (24, 40))
    F = dft2(f, hann=False)
    assert F.shape == (24, 40) and np.allclose(idft2(F), f)
    assert phase(F).min() >= -np.pi and phase(F).max() <= np.pi and (log_magnitude(F) >= 0).all()


def test_filters_lowpass_highpass_and_dc():
    n = 64
    y, x = np.mgrid[0:n, 0:n]
    f = 128 + 100 * np.sin(2 * np.pi * 16 * x / n)
    low, _, _ = filter_image(f, "low-pass", cutoff=0.2)
    high, _, _ = filter_image(f, "high-pass", cutoff=0.2)
    assert low.std() < 5 and high.std() > 50
    assert abs(high.mean() - f.mean()) < 1e-6                               # mean restored for display
    raw, _, _ = filter_image(f, "high-pass", cutoff=0.2, restore_mean=False)
    assert abs(raw.mean()) < 1e-6                                           # raw high-pass has no DC


def test_masks():
    for k in ("low-pass", "high-pass", "band-pass", "orientation"):
        m = make_mask((32, 48), k, cutoff=0.3, cutoff_hi=0.6, angle_deg=0, width_deg=30)
        assert m.shape == (32, 48)
    assert make_mask((32, 32), "low-pass", cutoff=0.3)[16, 16] == 1
    assert make_mask((32, 32), "high-pass", cutoff=0.3)[16, 16] == 0       # documented: DC removed
    with pytest.raises(ValueError):
        make_mask((8, 8), "nope")


def test_swap_and_descriptors_non_square():
    a = np.random.default_rng(2).uniform(0, 255, (32, 32)); b = a.T.copy()
    assert swap_magnitude_phase(a, b).shape == a.shape
    rgb = np.zeros((40, 60, 3), np.uint8); rgb[..., 0] = 200
    d = image_descriptors(rgb, fit_gray(rgb, 64))
    assert d["mean_r"] == 200 and np.isfinite(list(d.values())).all()
    assert d["width"] == 60 and d["height"] == 40


# ------------------------------------------------------------ visual descriptors / ranking
def _desc(kind, size=(64, 48), seed=0):
    return visual.visual_descriptors(load_image(make_image(kind, size, seed)))


def test_descriptor_structure_and_json():
    import json
    d = _desc("noise")
    assert len(d["hist"]) == visual.HIST_BINS and len(d["mag_hist"]) == visual.MAG_BINS
    assert len(d["ori_hist"]) == visual.ORI_BINS and len(d["grid"]) == visual.GRID ** 2
    assert np.isclose(sum(d["hist"]), 1.0)
    assert json.loads(json.dumps(d)) == d
    _, v = analyse_image(make_image("noise"))
    assert v == d


def test_identity_and_bounds():
    a, b = _desc("vstripes"), _desc("dark")
    assert np.isclose(visual.combine(visual.component_distances(a, a))["similarity"], 1.0)
    comp = visual.component_distances(a, b)
    assert all(0 <= v <= 1 for v in comp.values())
    s = visual.combine(comp)["similarity"]
    assert 0 <= s < 1


def test_ranking_prefers_similar_content():
    ref = _desc("vstripes", (64, 48), 0)
    cands = [(1, _desc("dark")), (2, _desc("vstripes", (80, 40), 1)), (3, _desc("bright")), (4, _desc("hstripes"))]
    r = visual.rank(ref, cands)
    assert r[0]["id"] == 2
    assert [x["similarity"] for x in r] == sorted([x["similarity"] for x in r], reverse=True)
    # vertical vs horizontal stripes have similar histograms, but different edge direction
    hist_only = visual.rank(ref, cands, {"histogram": 1, "edges": 0})
    edge_only = visual.rank(ref, cands, {"histogram": 0, "edges": 1})
    sim_h = {x["id"]: x["similarity"] for x in hist_only}
    sim_e = {x["id"]: x["similarity"] for x in edge_only}
    assert sim_h[4] > sim_e[4]
    with pytest.raises(ValueError):
        visual.combine(visual.component_distances(ref, ref), {"histogram": 0, "edges": 0})


def test_ranking_empty_and_size_independent():
    assert visual.rank(_desc("noise"), []) == []
    small, large = _desc("gradient", (32, 24)), _desc("gradient", (256, 192))
    assert visual.combine(visual.component_distances(small, large))["similarity"] > 0.9
    flat = visual.visual_descriptors(np.full((10, 10, 3), 90, np.uint8))     # no edges at all
    assert np.isfinite(visual.combine(visual.component_distances(flat, flat))["similarity"])

import numpy as np
import pytest

import config as C
from config import DEFAULT_FIELD
import aurora_colour as AC

H = np.linspace(C.AURORA_H_MIN_KM, C.AURORA_H_MAX_KM, 3201)
NAMES = [line.name for line in AC.AURORA_LINES]


@pytest.mark.parametrize("species", ["N2", "O"])
def test_density_is_exponential(species):
    n_ref, H_s = AC.SPECIES[species]
    assert AC.density(species, C.ATMOSPHERE_H_REF_KM) == pytest.approx(n_ref)
    h = np.array([120.0, 200.0])
    assert np.allclose(AC.density(species, h + H_s) / AC.density(species, h), np.exp(-1))


def test_heavier_gas_falls_off_faster():
    assert C.N2_SCALE_HEIGHT_KM < C.O_SCALE_HEIGHT_KM
    assert AC.density("N2", 100.0) > AC.density("O", 100.0)
    assert AC.density("O", 300.0) > AC.density("N2", 300.0)


def test_total_density_is_sum():
    assert np.allclose(AC.total_density(H), AC.density("N2", H) + AC.density("O", H))


def test_column_density_is_integral_of_density():
    h = np.linspace(150.0, 2000.0, 200001)
    numeric = np.trapezoid(AC.total_density(h), h * 1e3)
    assert AC.column_density(150.0) == pytest.approx(numeric, rel=1e-4)


def test_deposition_factor_limits():
    assert AC.deposition_factor(1000.0) == pytest.approx(1.0, abs=1e-6)
    assert AC.deposition_factor(60.0) < 1e-6
    assert np.all(np.diff(AC.deposition_factor(H)) >= 0)     # more beam higher up


def test_chapman_peak_where_column_equals_N0():
    n_ref, H_s = AC.SPECIES["N2"]
    h = np.linspace(60, 300, 240001)
    n = n_ref * np.exp(-(h - C.ATMOSPHERE_H_REF_KM) / H_s)
    N = n * H_s * 1e3
    N0 = 3e22
    h_peak = h[np.argmax(n * np.exp(-N / N0))]
    h_expected = C.ATMOSPHERE_H_REF_KM + H_s * np.log(n_ref * H_s * 1e3 / N0)
    assert h_peak == pytest.approx(h_expected, abs=0.01)


@pytest.mark.parametrize("quencher", ["total", "N2", "O"])
def test_quench_factor_half_at_quench_height(quencher):
    assert AC.quench_factor(180.0, 180.0, quencher) == pytest.approx(0.5)
    assert AC.quench_factor(400.0, 180.0, quencher) > 0.9
    assert AC.quench_factor(90.0, 180.0, quencher) < 0.2
    assert np.all(np.diff(AC.quench_factor(H, 180.0, quencher)) > 0)


def test_no_quench_height_means_no_quenching():
    assert np.all(AC.quench_factor(H, None) == 1.0)


def test_n2_transfer_factor_is_n2_share():
    f = AC.n2_transfer_factor(H)
    assert np.all((f > 0) & (f < 1))
    assert np.all(np.diff(f) < 0)                              # N₂ thins out upwards


def test_line_profiles_shape_and_positive():
    P = AC.line_profiles(H)
    assert P.shape == (len(H), len(AC.AURORA_LINES))
    assert np.all(P >= 0)


def test_line_profiles_scale_with_strength():
    line = AC.AURORA_LINES[0]
    double = AC.AuroraLine(line.name, line.wavelength_nm, line.species,
                           2 * line.strength, line.quench_height_km, line.note,
                           line.via_N2_transfer, line.quencher)
    assert np.allclose(AC.line_profiles(H, (double,))[:, 0],
                       2 * AC.line_profiles(H, (line,))[:, 0])


def test_peak_heights_in_the_right_ranges():
    peaks = AC.peak_heights()
    assert 100 <= peaks["O green"] <= 200
    assert peaks["O red"] > 200
    assert peaks["N₂⁺ blue-violet"] < peaks["O green"]
    assert peaks["N₂ deep red"] < peaks["O green"]
    for h in peaks.values():
        assert 100 <= h <= 300                                # real aurora heights


def test_red_peak_matches_analytic_offset():
    red = AC.AURORA_LINES[NAMES.index("O red")]
    H_N2, H_O = C.N2_SCALE_HEIGHT_KM, C.O_SCALE_HEIGHT_KM
    expected = red.quench_height_km + H_N2 * np.log((H_O - H_N2) / H_N2)
    assert AC.peak_heights()["O red"] == pytest.approx(expected, abs=1.0)


def test_red_dominates_high_green_dominates_low():
    P = AC.line_profiles(np.array([130.0, 250.0]))
    g, r = NAMES.index("O green"), NAMES.index("O red")
    assert P[0, g] > P[0, r]
    assert P[1, r] > P[1, g]


def test_integrated_ratios_in_literature_ranges():
    I = AC.integrated_intensity()
    assert 2 <= I["O green"] / I["N₂⁺ blue-violet"] <= 4
    assert 0.1 <= I["O red"] / I["O green"] <= 1


def test_cie_table_known_values():
    assert AC.cie_cmf(555.0)[1] == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(AC.cie_cmf([300.0, 900.0]), 0.0)


def test_wyman_fit_close_to_table():
    lam = np.arange(400.0, 701.0, 1.0)
    diff = np.abs(AC.cie_cmf(lam, "table") - AC.cie_cmf(lam, "wyman"))
    assert diff.max() < 0.05


def test_lines_to_xyz_is_weighted_sum():
    wl = np.array([557.7, 630.0])
    I = np.array([2.0, 3.0])
    expected = 2 * AC.cie_cmf(557.7) + 3 * AC.cie_cmf(630.0)
    assert np.allclose(AC.lines_to_xyz(I, wl), expected)


def test_d65_white_is_white():
    rgb = AC.xyz_to_linear_srgb([0.95047, 1.0, 1.08883])
    assert np.allclose(rgb, 1.0, atol=2e-3)


def test_spectral_lines_are_out_of_gamut():
    for wl in (557.7, 630.0, 427.8):
        assert AC.xyz_to_linear_srgb(AC.cie_cmf(wl)).min() < 0


def test_oklab_round_trip():
    rgb = np.random.default_rng(1).uniform(-0.2, 1.0, size=(50, 3))
    assert np.allclose(AC.oklab_to_linear_srgb(AC.linear_srgb_to_oklab(rgb)), rgb,
                       atol=1e-9)


@pytest.mark.parametrize("method", ["oklch", "grey", "clip"])
def test_gamut_methods_give_non_negative_rgb(method):
    for wl in (427.8, 557.7, 630.0, 670.0):
        xyz = AC.cie_cmf(wl)
        rgb, amount = AC.bring_into_gamut(AC.xyz_to_linear_srgb(xyz), xyz[1], method)
        assert np.all(rgb >= 0)
        assert amount > 0


@pytest.mark.parametrize("method", ["oklch", "grey"])
def test_in_gamut_colour_unchanged(method):
    rgb = np.array([0.3, 0.5, 0.2])
    out, amount = AC.bring_into_gamut(rgb, 0.4, method)
    assert np.allclose(out, rgb, atol=1e-9)
    assert amount == pytest.approx(0.0, abs=1e-9)


def test_oklch_keeps_lightness_and_hue():
    xyz = AC.cie_cmf(630.0)
    rgb_in = AC.xyz_to_linear_srgb(xyz)
    rgb_out, _ = AC.reduce_chroma_into_gamut(rgb_in)
    lab_in, lab_out = AC.linear_srgb_to_oklab(rgb_in), AC.linear_srgb_to_oklab(rgb_out)
    assert lab_out[0] == pytest.approx(lab_in[0], abs=1e-6)
    hue = lambda lab: np.arctan2(lab[2], lab[1])
    assert hue(lab_out) == pytest.approx(hue(lab_in), abs=1e-6)


def test_bad_gamut_method_raises():
    with pytest.raises(ValueError):
        AC.bring_into_gamut(np.array([0.1, 0.2, 0.3]), 0.2, "nope")


def test_srgb_gamma():
    assert AC.srgb_gamma(0.0) == 0.0
    assert AC.srgb_gamma(1.0) == pytest.approx(1.0)
    c = 0.0031308                                              # continuous at the joint
    assert 12.92 * c == pytest.approx(1.055 * c ** (1 / 2.4) - 0.055, abs=1e-6)
    assert np.all(np.diff(AC.srgb_gamma(np.linspace(0, 1, 101))) > 0)


def test_log_brightness():
    assert AC.log_brightness(5.0, 5.0, 3) == pytest.approx(1.0)
    assert AC.log_brightness(5e-3, 5.0, 3) == pytest.approx(0.0)
    assert AC.log_brightness(5e-2, 5.0, 3) == pytest.approx(1 / 3)
    assert AC.log_brightness(0.0, 5.0, 3) == 0.0
    assert AC.log_brightness(1e-9, 5.0, 3) == 0.0             # clipped below


def test_557_renders_green():
    r, g, b = AC.line_swatch(557.7)
    assert g == pytest.approx(1.0) and g > r > b


def test_630_renders_red():
    r, g, b = AC.line_swatch(630.0)
    assert r == pytest.approx(1.0) and g < 0.05 and b < 0.3


def test_428_renders_blue_violet():
    r, g, b = AC.line_swatch(427.8)
    assert b == pytest.approx(1.0) and r > g and g < 0.05 and r < 0.7


def test_colourise_zero_light_is_black():
    rgb, b = AC.colourise(np.zeros((3, 2)), [557.7, 630.0], Y_ref=1.0)
    assert np.all(rgb == 0) and np.all(b == 0)


def test_colourise_hue_independent_of_overall_scale():
    I = np.array([[1.0, 0.5, 0.1, 0.1]])
    wl = [line.wavelength_nm for line in AC.AURORA_LINES]
    rgb1, _ = AC.colourise(I, wl)
    rgb2, _ = AC.colourise(1e6 * I, wl)
    assert np.allclose(rgb1, rgb2)


def test_colourise_swappable_line_table():
    rgb, _ = AC.colourise(np.array([1.0]), np.array([589.0]))
    r, g, b = rgb
    assert r == pytest.approx(1.0) and 0.4 < g < 1.0 and b < 0.05


def test_colourise_table_and_wyman_agree():
    for wl in (427.8, 557.7, 630.0):
        assert np.allclose(AC.line_swatch(wl, "table"), AC.line_swatch(wl, "wyman"),
                           atol=0.08)


def test_out_of_gamut_amount_between_0_and_1():
    for wl in (427.8, 557.7, 630.0):
        assert 0 < AC.out_of_gamut_amount(wl) < 1


CROSSINGS = [{"chi_deg": 60.0}, {"chi_deg": 120.0}]


def test_band_weight_peaks_at_boundary_and_is_symmetric():
    w = AC.band_weight(np.array([60.0, 58.0, 62.0]), CROSSINGS[:1], band_width_deg=2.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(w[2]) == pytest.approx(np.exp(-0.5))


def test_band_weight_zero_far_from_band():
    chi = np.array([0.0, 30.0, 90.0, 180.0])
    assert np.all(AC.band_weight(chi, CROSSINGS, band_width_deg=2.0, cutoff=4.0) == 0)
    assert AC.band_weight(np.array([60 + 8.01]), CROSSINGS, 2.0, 4.0)[0] == 0
    assert AC.band_weight(np.array([60 + 7.99]), CROSSINGS, 2.0, 4.0)[0] > 0


def test_band_weight_with_footpoint_B():
    def footB(chi):                       # stronger field further from the noon equator
        return 1.0 + np.asarray(chi) / 100.0
    chi = np.array([59.0, 60.0, 61.0])
    plain = AC.band_weight(chi, CROSSINGS[:1], 2.0)
    weighted = AC.band_weight(chi, CROSSINGS[:1], 2.0, footpoint_B=footB)
    assert weighted[1] == pytest.approx(1.0)
    assert np.allclose(weighted, plain * footB(chi) / footB(60.0))


@pytest.fixture(scope="module")
def glow():
    boundaries = {"north": [{"chi_deg": 60.0}], "south": [{"chi_deg": 120.0}]}
    cfg = DEFAULT_FIELD.with_(tilt_deg=0.0)
    return AC.glow_image_xz(cfg, boundaries, n_pixels=241), boundaries


def test_glow_only_in_the_bands(glow):
    rgba, _ = glow
    n = rgba.shape[0]
    g = np.linspace(-C.AURORA_PLOT_HALF_WIDTH_RE, C.AURORA_PLOT_HALF_WIDTH_RE, n)
    X, Z = np.meshgrid(g, g)
    chi = np.degrees(np.arctan2(np.abs(Z), -X))
    lit = rgba[..., 3] > 0
    assert lit.any()
    cut = C.AURORA_BAND_WIDTH_DEG * C.AURORA_BAND_CUTOFF + 1.0
    assert np.all(np.abs(chi[lit & (Z >= 0)] - 60) < cut)
    assert np.all(np.abs(chi[lit & (Z < 0)] - 120) < cut)
    assert (lit & (Z >= 0)).any() and (lit & (Z < 0)).any()


def test_glow_only_in_the_height_shell(glow):
    rgba, _ = glow
    n = rgba.shape[0]
    g = np.linspace(-C.AURORA_PLOT_HALF_WIDTH_RE, C.AURORA_PLOT_HALF_WIDTH_RE, n)
    X, Z = np.meshgrid(g, g)
    h_km = (np.hypot(X, Z) - 1.0) * 6371.0 / C.ALTITUDE_EXAGGERATION
    lit = rgba[..., 3] > 0
    assert h_km[lit].min() >= C.AURORA_H_MIN_KM - 1e-6
    assert h_km[lit].max() <= C.AURORA_H_MAX_KM + 1e-6
    assert np.all((rgba >= 0) & (rgba <= 1))


def test_xyz_to_xy_of_d65_white():
    xy = AC.xyz_to_xy(np.array([0.95047, 1.0, 1.08883]))
    assert np.allclose(xy, [0.3127, 0.3290], atol=1e-4)


def test_xyz_to_xy_no_light_is_nan():
    assert np.all(np.isnan(AC.xyz_to_xy(np.zeros(3))))


def test_linear_srgb_to_xyz_round_trip():
    rgb = np.random.default_rng(3).uniform(-0.2, 1.0, size=(20, 3))
    assert np.allclose(AC.xyz_to_linear_srgb(AC.linear_srgb_to_xyz(rgb)), rgb)


def test_srgb_primaries_chromaticities():
    xy = AC.xyz_to_xy(AC.linear_srgb_to_xyz(np.eye(3)))
    assert np.allclose(xy, [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], atol=2e-3)


def test_glow_composites_to_colourise_over_black(glow):
    rgba, _ = glow
    lit = rgba[..., 3] > 0
    shown = rgba[lit, :3] * rgba[lit, 3:4]
    assert shown.max() <= 1.0
    assert np.allclose(rgba[lit, :3].max(axis=-1), 1.0, atol=1e-6)

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config as C


SPECIES = {
    # name: (density at h_ref [m⁻³], scale height [km])
    "N2": (C.N2_DENSITY_REF_M3, C.N2_SCALE_HEIGHT_KM),
    "O": (C.O_DENSITY_REF_M3, C.O_SCALE_HEIGHT_KM),
}


def density(species, h_km):
    """n(h) = n_ref · exp(−(h − h_ref)/H)   [m⁻³]."""
    n_ref, H = SPECIES[species]
    return n_ref * np.exp(-(np.asarray(h_km, dtype=float) - C.ATMOSPHERE_H_REF_KM) / H)


def total_density(h_km):
    return sum(density(s, h_km) for s in SPECIES)


def column_density(h_km):
    """N(h): particles above height h per m², sum of n(h)·H over species."""
    return sum(density(s, h_km) * SPECIES[s][1] * 1e3 for s in SPECIES)


def deposition_factor(h_km, N0=C.CHAPMAN_N0_M2):
    """exp(−N(h)/N₀), fraction of the electron beam left at height h."""
    return np.exp(-column_density(h_km) / N0)


def quencher_density(h_km, quencher="total"):
    """Density of the gas that does the quenching: "total", "N2" or "O"."""
    if quencher == "total":
        return total_density(h_km)
    return density(quencher, h_km)


def quench_factor(h_km, quench_height_km, quencher="total"):
    """Fraction of excited atoms that emit, q = 1/(1 + n/n_crit).

    n_crit is the quencher density at the quench height so q = ½ there.
    630 nm is quenched mainly by N₂ (Streit et al. 1976).
    """
    if quench_height_km is None:
        return np.ones_like(np.asarray(h_km, dtype=float))
    return 1.0 / (1.0 + quencher_density(h_km, quencher)
                  / quencher_density(quench_height_km, quencher))


@dataclass(frozen=True)
class AuroraLine:
    name: str
    wavelength_nm: float
    species: str              # which gas density it follows ("N2" or "O")
    strength: float           # relative emission per unit (density × beam)
    quench_height_km: float   # None = emits immediately, never quenched
    note: str
    # True = fed by energy transfer from N₂, see n2_transfer_factor()
    via_N2_transfer: bool = False
    # gas that quenches the upper state
    quencher: str = "total"


def n2_transfer_factor(h_km):
    """n_N2 / (n_N2 + n_O).  The green line is mostly fed by energy transfer from N₂."""
    n2, o = density("N2", h_km), density("O", h_km)
    return n2 / (n2 + o)


# strengths are tuned so the integrated ratios match the literature
# 557.7/427.8 ~ 2-4 (Rees 1989, Vallance Jones 1974)
# 630.0/557.7 ~ 0.1-1 (Chamberlain 1961)
# quench heights are rounded from the same books
AURORA_LINES = (
    AuroraLine("O green", 557.7, "O", 1.0, 95.0,
               "O(¹S→¹D), the main auroral line; lifetime 0.74 s; upper "
               "state largely fed by energy transfer from excited N₂",
               via_N2_transfer=True),
    AuroraLine("O red", 630.0, "O", 1.0, 200.0,
               "O(¹D→³P), lifetime ≈ 110 s, so quenched (mainly by N₂) "
               "below ≈ 200 km", quencher="N2"),
    AuroraLine("N₂⁺ blue-violet", 427.8, "N2", 0.04, None,
               "N₂⁺ first negative band; needs ionisation, prompt"),
    AuroraLine("N₂ deep red", 670.0, "N2", 0.04, None,
               "N₂ first positive bands (~650–690 nm, one representative "
               "wavelength); prompt; pink lower border"),
)


def line_profiles(h_km, lines=AURORA_LINES, N0=C.CHAPMAN_N0_M2):
    """I(h) = S · n(h) · exp(−N(h)/N₀) · q(h), times the N₂ share for the green line.

    Shape h_km.shape + (n_lines,).
    """
    h = np.asarray(h_km, dtype=float)
    beam = deposition_factor(h, N0)
    out = []
    for line in lines:
        I = (line.strength * density(line.species, h) * beam
             * quench_factor(h, line.quench_height_km, line.quencher))
        if line.via_N2_transfer:
            I = I * n2_transfer_factor(h)
        out.append(I)
    return np.stack(out, axis=-1)


def peak_heights(lines=AURORA_LINES, h_km=None):
    """Height (km) of maximum emission for each line."""
    if h_km is None:
        h_km = np.linspace(C.AURORA_H_MIN_KM, C.AURORA_H_MAX_KM, 3201)
    I = line_profiles(h_km, lines)
    return {line.name: h_km[np.argmax(I[:, i])] for i, line in enumerate(lines)}


def integrated_intensity(lines=AURORA_LINES, h_km=None):
    """Emission of each line summed over all heights (a vertical column)."""
    if h_km is None:
        h_km = np.linspace(C.AURORA_H_MIN_KM, C.AURORA_H_MAX_KM, 3201)
    I = line_profiles(h_km, lines)
    return {line.name: np.trapezoid(I[:, i], h_km) for i, line in enumerate(lines)}


CIE_TABLE_PATH = Path(__file__).parent / "data" / "ciexyz31_1.csv"
_cie_table = None


def _load_cie_table():
    """CIE 1931 2° x̄, ȳ, z̄ at 1 nm steps.

    http://cvrl.ucl.ac.uk/database/data/cmfs/ciexyz31_1.csv
    """
    global _cie_table
    if _cie_table is None:
        _cie_table = np.loadtxt(CIE_TABLE_PATH, delimiter=",")
    return _cie_table


def _piecewise_gaussian(lam, mu, sigma_left, sigma_right):
    sigma = np.where(lam < mu, sigma_left, sigma_right)
    return np.exp(-0.5 * ((lam - mu) / sigma) ** 2)


def cie_cmf_wyman(wavelength_nm):
    """Fit to the CIE functions (Wyman, Sloan & Shirley 2013, JCGT 2(2)).

    Used if the table file is missing.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    g = _piecewise_gaussian
    x = (1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7)
         - 0.065 * g(lam, 501.1, 20.4, 26.2))
    y = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def cie_cmf(wavelength_nm, source="table"):
    """x̄, ȳ, z̄ at the given wavelengths, shape (..., 3)."""
    lam = np.asarray(wavelength_nm, dtype=float)
    if source == "table" and CIE_TABLE_PATH.exists():
        t = _load_cie_table()
        return np.stack([np.interp(lam, t[:, 0], t[:, k], left=0, right=0)
                         for k in (1, 2, 3)], axis=-1)
    return cie_cmf_wyman(lam)


def lines_to_xyz(intensities, wavelengths_nm, cmf_source="table"):
    """X = Σ I_line · x̄(λ_line), same for Y and Z.  Shape (..., 3)."""
    return np.asarray(intensities, dtype=float) @ cie_cmf(wavelengths_nm, cmf_source)


# XYZ to linear sRGB (IEC 61966-2-1, D65)
XYZ_TO_LINEAR_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])


def xyz_to_linear_srgb(xyz):
    return np.asarray(xyz, dtype=float) @ XYZ_TO_LINEAR_SRGB.T


def linear_srgb_to_xyz(rgb):
    """Inverse of xyz_to_linear_srgb."""
    return np.asarray(rgb, dtype=float) @ np.linalg.inv(XYZ_TO_LINEAR_SRGB).T


def xyz_to_xy(xyz):
    """x = X/(X+Y+Z), y = Y/(X+Y+Z).  NaN if there is no light."""
    xyz = np.asarray(xyz, dtype=float)
    s = xyz.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, xyz[..., :2] / s, np.nan)


# single lines are too saturated for sRGB so some values go negative
# oklch keeps lightness and hue (CSS Color 4), clip sets negatives to 0,
# grey mixes with grey of the same luminance

_LINEAR_SRGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]])
_LMS_TO_OKLAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]])
_OKLAB_TO_LMS = np.array([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480]])
_LMS_TO_LINEAR_SRGB = np.array([
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010]])


def linear_srgb_to_oklab(rgb):
    """OKLab (Ottosson 2020)."""
    lms = np.asarray(rgb, dtype=float) @ _LINEAR_SRGB_TO_LMS.T
    return np.cbrt(lms) @ _LMS_TO_OKLAB.T


def oklab_to_linear_srgb(lab):
    lms = (np.asarray(lab, dtype=float) @ _OKLAB_TO_LMS.T) ** 3
    return lms @ _LMS_TO_LINEAR_SRGB.T


def reduce_chroma_into_gamut(rgb_linear, n_iter=40):
    """Keep L and hue, shrink chroma until no channel is negative.  Returns (rgb, 1 − s)."""
    rgb = np.asarray(rgb_linear, dtype=float)
    lab = linear_srgb_to_oklab(rgb)
    ok = (rgb >= 0).all(axis=-1)
    lo = np.where(ok, 1.0, 0.0)
    hi = np.ones(rgb.shape[:-1])
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        trial = lab.copy()
        trial[..., 1:] *= mid[..., None]
        good = (oklab_to_linear_srgb(trial) >= -1e-12).all(axis=-1)
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)
    lab[..., 1:] *= lo[..., None]
    return np.maximum(oklab_to_linear_srgb(lab), 0.0), 1.0 - lo


def desaturate_into_gamut(rgb_linear, Y):
    """Mix with grey of the same luminance until no channel is negative.  Returns (rgb, t)."""
    rgb = np.asarray(rgb_linear, dtype=float)
    Y = np.asarray(Y, dtype=float)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_each = np.where(rgb < 0, -rgb / (Y - rgb), 0.0)
    t = np.clip(np.nan_to_num(t_each.max(axis=-1, keepdims=True)), 0.0, 1.0)
    out = rgb + t * (Y - rgb)
    return np.maximum(out, 0.0), t[..., 0]


def srgb_gamma(c):
    """Linear → sRGB-encoded values (the standard sRGB transfer curve)."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def log_brightness(Y, Y_ref, decades=C.BRIGHTNESS_DECADES):
    """b = 1 + log10(Y/Y_ref)/decades, clipped to [0, 1]."""
    with np.errstate(divide="ignore"):
        b = 1.0 + np.log10(np.asarray(Y, dtype=float) / Y_ref) / decades
    return np.clip(np.nan_to_num(b, nan=0.0, neginf=0.0), 0.0, 1.0)


def bring_into_gamut(rgb_linear, Y, method="oklch"):
    """Returns (rgb, amount)."""
    if method == "oklch":
        return reduce_chroma_into_gamut(rgb_linear)
    if method == "grey":
        return desaturate_into_gamut(rgb_linear, Y)
    if method == "clip":
        rgb = np.asarray(rgb_linear, dtype=float)
        return np.maximum(rgb, 0.0), (rgb < 0).any(axis=-1).astype(float)
    raise ValueError("gamut method must be 'oklch', 'grey' or 'clip'")


def colourise(intensities, wavelengths_nm, Y_ref=None,
              decades=C.BRIGHTNESS_DECADES, cmf_source="table", gamut="oklch"):
    """Line intensities to sRGB.  Returns (rgb, brightness)."""
    xyz = lines_to_xyz(intensities, wavelengths_nm, cmf_source)
    Y = xyz[..., 1]
    rgb_lin, _ = bring_into_gamut(xyz_to_linear_srgb(xyz), Y, gamut)
    peak = rgb_lin.max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        hue = np.where(peak > 0, rgb_lin / peak, 0.0)
    if Y_ref is None:
        Y_ref = np.max(Y)
    b = log_brightness(Y, Y_ref, decades)
    return srgb_gamma(hue) * b[..., None], b


def line_swatch(wavelength_nm, cmf_source="table", gamut="oklch"):
    """Full-brightness sRGB colour of a single line on its own."""
    rgb, _ = colourise(np.array([1.0]), np.array([wavelength_nm]),
                       cmf_source=cmf_source, gamut=gamut)
    return rgb


def out_of_gamut_amount(wavelength_nm, gamut="oklch"):
    """How much saturation was given up to show a single line (0 = none)."""
    xyz = cie_cmf(wavelength_nm)
    _, amount = bring_into_gamut(xyz_to_linear_srgb(xyz), xyz[..., 1], gamut)
    return float(amount)


def band_weight(chi_deg, crossings, band_width_deg=C.AURORA_BAND_WIDTH_DEG,
                cutoff=C.AURORA_BAND_CUTOFF, footpoint_B=None):
    """Gaussian in angle around each boundary footpoint, 0 beyond cutoff widths."""
    chi = np.asarray(chi_deg, dtype=float)
    w = np.zeros_like(chi)
    for b in crossings:
        d = (chi - b["chi_deg"]) / band_width_deg
        term = np.where(np.abs(d) <= cutoff, np.exp(-0.5 * d**2), 0.0)
        if footpoint_B is not None:
            term = term * footpoint_B(chi) / footpoint_B(np.array(b["chi_deg"]))
        w = w + term
    return w


def hemisphere_band_weight(chi_deg, cfg, crossings, hemisphere,
                           weight_by_B=C.AURORA_WEIGHT_BY_FOOTPOINT_B):
    """band_weight for one hemisphere, optionally weighted by footpoint |B|."""
    from sources import B_total
    from topology import meridian_footpoint

    def foot_B(c):
        c = np.atleast_1d(np.asarray(c, dtype=float))
        pts = np.array([meridian_footpoint(ci, hemisphere) for ci in c.ravel()])
        return np.linalg.norm(B_total(pts, cfg), axis=-1).reshape(c.shape)

    return band_weight(chi_deg, crossings,
                       footpoint_B=foot_B if weight_by_B else None)


def glow_image_xz(cfg, boundaries, half_width_RE=C.AURORA_PLOT_HALF_WIDTH_RE,
                  n_pixels=C.AURORA_IMAGE_PIXELS,
                  exaggeration=C.ALTITUDE_EXAGGERATION, lines=AURORA_LINES):
    """RGBA image of the glow in the x–z plane, h = (r − R_E)/exaggeration."""
    from constants import R_E

    g = np.linspace(-half_width_RE, half_width_RE, n_pixels)
    X, Z = np.meshgrid(g, g)
    r = np.hypot(X, Z)
    h_km = (r - 1.0) * R_E / 1e3 / exaggeration
    inside = (h_km >= C.AURORA_H_MIN_KM) & (h_km <= C.AURORA_H_MAX_KM)
    chi = np.degrees(np.arctan2(np.abs(Z), -X))

    chi_grid = np.linspace(0.0, 180.0, 3601)
    weight = np.zeros_like(r)
    for hemisphere, key in ((+1, "north"), (-1, "south")):
        w_grid = hemisphere_band_weight(chi_grid, cfg, boundaries[key], hemisphere)
        sel = inside & ((Z >= 0) if hemisphere > 0 else (Z < 0))
        weight[sel] = np.interp(chi[sel], chi_grid, w_grid)

    lit = inside & (weight > 0)
    intens = line_profiles(h_km[lit], lines) * weight[lit][:, None]
    wavelengths = np.array([line.wavelength_nm for line in lines])
    rgb, b = colourise(intens, wavelengths)

    # hue at full strength with b as alpha, otherwise brightness is applied twice
    rgba = np.zeros(r.shape + (4,))
    with np.errstate(invalid="ignore", divide="ignore"):
        rgba[lit, :3] = np.where(b[:, None] > 0, rgb / b[:, None], 0.0)
    rgba[lit, 3] = b
    return np.clip(rgba, 0.0, 1.0)

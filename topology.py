import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import least_squares

from constants import R_E
import config as C
from config import DEFAULT_FIELD
from field import trace_field_line
from sources import (B_total, field_function, magnetic_north,
                     sun_field_at_earth_magnitude)


def trace(start, cfg=DEFAULT_FIELD, direction=None,
          r_outer_RE=C.TRACE_R_OUTER_RE, max_step_RE=C.TRACE_MAX_STEP_RE):
    """trace_field_line in the combined field.  Returns (points, end_reason)."""
    return trace_field_line(start, r_outer=r_outer_RE * R_E,
                            max_step=max_step_RE * R_E,
                            max_length=C.TRACE_MAX_LENGTH_RE * R_E,
                            field_func=field_function(cfg),
                            direction=direction)


def _status(n_earth_ends):
    return {2: "closed", 1: "open", 0: "solar"}[n_earth_ends]


def classify_footpoint(foot, cfg=DEFAULT_FIELD, **trace_kwargs):
    """Trace from the surface and classify as closed or open."""
    points, end = trace(foot, cfg, **trace_kwargs)
    if end == "surface":
        status = "closed"
    else:
        status = "closed" if cfg.w_sun == 0 else "open"
    return {"points": points, "ends": ("surface", end), "status": status,
            "beyond_limit": end != "surface" and cfg.w_sun == 0}


def classify_seed(seed, cfg=DEFAULT_FIELD, **trace_kwargs):
    """Trace from any point in space in BOTH directions and classify the line."""
    back, end_back = trace(seed, cfg, direction=-1, **trace_kwargs)
    fwd, end_fwd = trace(seed, cfg, direction=+1, **trace_kwargs)
    points = np.vstack([back[::-1], fwd[1:]])
    n_earth = (end_back == "surface") + (end_fwd == "surface")
    return {"points": points, "ends": (end_back, end_fwd),
            "status": _status(n_earth)}


def meridian_footpoint(chi_deg, hemisphere=+1):
    """Point on the surface.  chi = 0 at noon equator, 90 at the pole, 180 at midnight."""
    c = np.radians(chi_deg)
    return R_E * np.array([-np.cos(c), 0.0, hemisphere * np.sin(c)])


def chi_to_latitude(chi_deg, hemisphere=+1):
    """Geographic latitude (signed) and side ("noon"/"midnight") for chi."""
    lat = 90.0 - abs(chi_deg - 90.0)
    return hemisphere * lat, ("noon" if chi_deg < 90 else "midnight")


def magnetic_latitude(position, cfg=DEFAULT_FIELD):
    """Latitude relative to the magnetic equator: arcsin(r̂ · n_mag)."""
    r_hat = np.asarray(position) / np.linalg.norm(position)
    return np.degrees(np.arcsin(np.clip(r_hat @ magnetic_north(cfg), -1, 1)))


def _is_open(chi, hemisphere, cfg, trace_kwargs):
    foot = meridian_footpoint(chi, hemisphere)
    return classify_footpoint(foot, cfg, **trace_kwargs)["status"] == "open"


def open_closed_boundary(cfg=DEFAULT_FIELD, hemisphere=+1,
                         scan_step_deg=C.BOUNDARY_SCAN_STEP_DEG,
                         tol_deg=C.BOUNDARY_TOLERANCE_DEG, **trace_kwargs):
    """Find closed/open changes along the meridian.  Returns a list of dicts, [] if none."""
    chis = np.arange(scan_step_deg, 180.0, scan_step_deg)
    is_open = np.array([_is_open(c, hemisphere, cfg, trace_kwargs) for c in chis])

    crossings = []
    for i in np.nonzero(is_open[1:] != is_open[:-1])[0]:
        lo, hi = chis[i], chis[i + 1]
        open_lo = is_open[i]
        while hi - lo > tol_deg:
            mid = 0.5 * (lo + hi)
            if _is_open(mid, hemisphere, cfg, trace_kwargs) == open_lo:
                lo = mid
            else:
                hi = mid
        chi = 0.5 * (lo + hi)
        lat, side = chi_to_latitude(chi, hemisphere)
        foot = meridian_footpoint(chi, hemisphere)
        # chi grows away from the Sun so open at larger chi = dayside edge
        edge = "nightside" if open_lo else "dayside"
        crossings.append({"chi_deg": chi, "latitude_deg": lat, "side": side,
                          "edge": edge,
                          "magnetic_latitude_deg": magnetic_latitude(foot, cfg),
                          "footpoint": foot})
    return crossings


def boundary_summary(cfg=DEFAULT_FIELD, **kwargs):
    """Boundary crossings in both hemispheres: {"north": [...], "south": [...]}."""
    return {"north": open_closed_boundary(cfg, +1, **kwargs),
            "south": open_closed_boundary(cfg, -1, **kwargs)}


def _polish_nulls(seeds_RE, residual, B_ref, cfg, half_width_RE):
    """Least squares polish of starting guesses, only keeps nulls inside the box."""
    found = []
    for seed in seeds_RE:
        sol = least_squares(residual, seed, method="lm", xtol=1e-14, ftol=1e-14)
        p = sol.x if len(sol.x) == 3 else np.array([sol.x[0], 0.0, sol.x[1]])
        if np.linalg.norm(p) < C.NULL_MIN_RADIUS_RE:
            continue
        if np.max(np.abs(p)) > half_width_RE:
            continue
        if np.linalg.norm(B_total(p * R_E, cfg)) > C.NULL_TOLERANCE * B_ref:
            continue
        if any(np.linalg.norm(p - q) < C.NULL_MERGE_DISTANCE_RE for q in found):
            continue
        found.append(p)
    return np.array(found).reshape(-1, 3)


def find_nulls(cfg=DEFAULT_FIELD, half_width_RE=C.NULL_SEARCH_HALF_WIDTH_RE,
               n_grid=C.NULL_SEARCH_GRID_POINTS):
    """Find points where |B_total| = 0.  Returns positions in R_E."""
    if cfg.w_sun == 0:
        return np.zeros((0, 3))          # a lone dipole has no nulls outside Earth
    B_ref = abs(cfg.w_sun) * sun_field_at_earth_magnitude(cfg)

    g = np.linspace(-half_width_RE, half_width_RE, n_grid)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):   # grid hits r = 0
        absB = np.linalg.norm(B_total(P * R_E, cfg), axis=-1)
    absB[np.linalg.norm(P, axis=-1) < C.NULL_MIN_RADIUS_RE] = np.inf
    is_min = (absB == minimum_filter(absB, size=3, mode="nearest")) & np.isfinite(absB)

    def residual(p):
        return B_total(p * R_E, cfg) / B_ref

    return _polish_nulls(P[is_min], residual, B_ref, cfg, half_width_RE)


def find_nulls_xz(cfg=DEFAULT_FIELD, half_width_RE=C.NULL_SEARCH_HALF_WIDTH_RE,
                  n_grid=4 * C.NULL_SEARCH_GRID_POINTS):
    """Nulls lying in the x–z plane (y = 0): the X-points seen on the plots."""
    if cfg.w_sun == 0:
        return np.zeros((0, 3))
    B_ref = abs(cfg.w_sun) * sun_field_at_earth_magnitude(cfg)

    g = np.linspace(-half_width_RE, half_width_RE, n_grid)
    X, Z = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X, np.zeros_like(X), Z], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        absB = np.linalg.norm(B_total(P * R_E, cfg), axis=-1)
    absB[np.hypot(X, Z) < C.NULL_MIN_RADIUS_RE] = np.inf
    is_min = (absB == minimum_filter(absB, size=3, mode="nearest")) & np.isfinite(absB)

    def residual(q):                      # q = (x, z); all three components
        return B_total(np.array([q[0], 0.0, q[1]]) * R_E, cfg) / B_ref

    return _polish_nulls(P[is_min][:, [0, 2]], residual, B_ref, cfg,
                         half_width_RE)


def describe_nulls(nulls_RE, ring_min_points=6, flatness=0.02, roundness=0.05):
    """Work out whether a set of nulls is isolated points or a ring."""
    n = len(nulls_RE)
    out = {"n": n, "kind": "none" if n == 0 else "points", "points": nulls_RE}
    if n < ring_min_points:
        return out
    mean = nulls_RE.mean(axis=0)
    _, s, vt = np.linalg.svd(nulls_RE - mean, full_matrices=False)
    if s[-1] > flatness * s[0]:
        return out
    # Kasa circle fit
    u, v = (nulls_RE - mean) @ vt[0], (nulls_RE - mean) @ vt[1]
    A = np.column_stack([2 * u, 2 * v, np.ones_like(u)])
    a, b, _ = np.linalg.lstsq(A, u**2 + v**2, rcond=None)[0]
    centre = mean + a * vt[0] + b * vt[1]
    radii = np.linalg.norm(nulls_RE - centre, axis=1)
    if radii.std() < roundness * radii.mean():
        out.update(kind="ring", centre=centre, radius=radii.mean(),
                   radius_spread=radii.max() - radii.min(), normal=vt[-1])
    return out


def field_lines_xz(cfg=DEFAULT_FIELD, footpoint_spacing_deg=C.FOOTPOINT_SPACING_DEG,
                   solar_seed_spacing_RE=C.SOLAR_SEED_SPACING_RE,
                   solar_seed_half_width_RE=None, boundaries=None, **trace_kwargs):
    """Trace lines in the x–z plane from footpoints and from seeds on the x axis."""
    lines = []
    chis = np.arange(footpoint_spacing_deg, 180.0, footpoint_spacing_deg)
    for hemisphere in (+1, -1):
        for chi in chis:
            lines.append(classify_footpoint(meridian_footpoint(chi, hemisphere),
                                            cfg, **trace_kwargs))

    if solar_seed_half_width_RE and cfg.w_sun != 0:
        xs = np.arange(2.0, solar_seed_half_width_RE, solar_seed_spacing_RE)
        for x in np.concatenate([-xs, xs]):
            line = classify_seed(np.array([x, 0.0, 0.0]) * R_E, cfg, **trace_kwargs)
            if line["status"] == "solar":
                lines.append(line)

    if boundaries:
        for hemisphere, key in ((+1, "north"), (-1, "south")):
            for b in boundaries[key]:
                for dchi in (-0.05, +0.05):
                    line = classify_footpoint(
                        meridian_footpoint(b["chi_deg"] + dchi, hemisphere),
                        cfg, **trace_kwargs)
                    line["separatrix"] = True
                    lines.append(line)
    return lines

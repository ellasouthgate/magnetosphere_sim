import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from constants import R_E
import config as C
from config import DEFAULT_FIELD, OLD_MODEL
from field import B_dipole
from sources import (B_total, B_earth, B_sun, earth_moment, sun_moment,
                     sun_field_at_earth_magnitude, earth_equatorial_surface_field,
                     effective_r_balance, geographic_north, magnetic_north,
                     field_on_xz_grid)
from topology import (find_nulls, find_nulls_xz, describe_nulls,
                      boundary_summary, field_lines_xz)
import aurora_colour as AC
import plotting

FIGURE_DIR = Path(__file__).parent / "figures"


def fmt_vec(v, scale=1.0, digits=3):
    return "(" + ", ".join(f"{x / scale:+.{digits}f}" for x in v) + ")"


def print_configuration(cfg):
    print("config")
    print(f"  tilt = {cfg.tilt_deg}, offset = {cfg.magnetic_offset_deg} "
          f"at azimuth {cfg.magnetic_offset_azimuth_deg}")
    print(f"  n_geo = {fmt_vec(geographic_north(cfg))}")
    print(f"  n_mag = {fmt_vec(magnetic_north(cfg))}")
    print(f"  m_earth = {fmt_vec(earth_moment(cfg), 1e22)} x 10^22 A m^2")
    print(f"  B0 = {earth_equatorial_surface_field(cfg) * 1e9:.0f} nT")
    print(f"  sun at x = -{cfg.sun_distance_RE:g} R_E, "
          f"m_sun = {fmt_vec(sun_moment(cfg), 1e30)} x 10^30 A m^2")
    print(f"  r_balance = {cfg.r_balance_RE:g} R_E, B_sun at Earth = "
          f"{sun_field_at_earth_magnitude(cfg) * 1e9:.2f} nT "
          f"{fmt_vec(B_sun([0.0, 0.0, 0.0], cfg), 1e-9, 2)} ({cfg.sun_field_direction})")
    print(f"  w_sun = {cfg.w_sun:g}, effective r_balance = "
          f"{effective_r_balance(cfg):.2f} R_E\n")


def check_fields(cfg):
    print("field checks")
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 20, size=(2000, 3)) * R_E
    pts = pts[np.linalg.norm(pts, axis=1) > R_E]

    old, new = B_dipole(pts), B_total(pts, OLD_MODEL)
    print(f"  w_sun = 0, tilt = 0 vs B_dipole: identical = {np.array_equal(old, new)}, "
          f"max diff = {np.max(np.abs(old - new)):.1e} T")

    parts = B_earth(pts, cfg) + cfg.w_sun * B_sun(pts, cfg)
    total = B_total(pts, cfg)
    rel = np.max(np.linalg.norm(total - parts, axis=1) / np.linalg.norm(total, axis=1))
    print(f"  superposition: max rel diff = {rel:.1e}")

    print("  r [R_E], |B_tot - B_earth|/|B_earth|, |B_tot - w B_sun|/|w B_sun|")
    for r in (1, 2, 5, 10, 50, 100, 200):
        p = np.array([-r * R_E, 0.0, 0.0])
        Bt, Be, Bs = B_total(p, cfg), B_earth(p, cfg), cfg.w_sun * B_sun(p, cfg)
        print(f"  {r:>5g}  {np.linalg.norm(Bt - Be) / np.linalg.norm(Be):.2e}  "
              f"{np.linalg.norm(Bt - Bs) / np.linalg.norm(Bs):.2e}")
    print()


def report_nulls(cfg):
    print("null points")
    nulls = find_nulls(cfg)
    d = describe_nulls(nulls)
    if d["kind"] == "none":
        print("  none")
    elif d["kind"] == "ring":
        print(f"  ring, {d['n']} points, radius {d['radius']:.3f} R_E, "
              f"centre {fmt_vec(d['centre'])}, normal {fmt_vec(d['normal'])}")
    else:
        print(f"  {d['n']} points")
    for p in nulls:
        print(f"    r = {fmt_vec(p)} R_E, |B| = "
              f"{np.linalg.norm(B_total(p * R_E, cfg)):.1e} T")
    print("  in x-z plane: " + ", ".join(fmt_vec(p) for p in find_nulls_xz(cfg)) + "\n")


def report_boundaries(cfg, label):
    boundaries = boundaries_for(cfg)
    print(f"  {label}:")
    for key in ("north", "south"):
        for b in sorted(boundaries[key], key=lambda b: b["edge"]):
            print(f"    {key}, {b['edge']}: mag lat {b['magnetic_latitude_deg']:+.2f}, "
                  f"geo lat {b['latitude_deg']:+.2f} ({b['side']})")
        if not boundaries[key]:
            print(f"    {key}: no open field lines")


_boundary_cache = {}


def boundaries_for(cfg):
    if cfg not in _boundary_cache:
        _boundary_cache[cfg] = boundary_summary(cfg)
    return _boundary_cache[cfg]


def run_stage1():
    print("stage 1, field")
    cfg = DEFAULT_FIELD
    print_configuration(cfg)
    check_fields(cfg)
    report_nulls(cfg)

    print("open/closed boundary")
    r_b = cfg.r_balance_RE
    analytic = np.degrees(np.arccos(np.sqrt((3 / (2 * r_b)) / (1 + 1 / (2 * r_b**3)))))
    print(f"  analytic, no tilt, uniform field: {analytic:.2f}")
    report_boundaries(cfg.with_(tilt_deg=0.0), "no tilt")
    report_boundaries(cfg, f"tilt {cfg.tilt_deg}")
    print()
    return {}


def run_stage5():
    print("stage 5, combined figure")
    cfg = DEFAULT_FIELD
    hw = C.COMBINED_HALF_WIDTH_RE
    exag = C.COMBINED_EXAGGERATION_FULL
    boundaries = boundaries_for(cfg)
    nulls_xz = find_nulls_xz(cfg)
    lines = field_lines_xz(cfg, footpoint_spacing_deg=C.COMBINED_FOOTPOINT_SPACING_DEG,
                           solar_seed_half_width_RE=hw,
                           solar_seed_spacing_RE=C.SOLAR_SEED_SPACING_RE / 2,
                           boundaries=boundaries, max_step_RE=C.TRACE_MAX_STEP_ZOOM_RE)
    x, z, B = field_on_xz_grid(cfg, hw, C.FIELD_MAP_GRID_POINTS)
    glow_hw = min(hw, C.AURORA_PLOT_HALF_WIDTH_RE)
    rgba = AC.glow_image_xz(cfg, boundaries, half_width_RE=glow_hw,
                            n_pixels=C.COMBINED_GLOW_PIXELS, exaggeration=exag)
    fig = plotting.plot_combined(cfg, x, z, B, lines, nulls_xz, boundaries, rgba, glow_hw,
                                 exag, hw, "Earth + solar field")
    return {"ext_5a_combined.png": fig}


def print_line_table():
    print("emission lines")
    for line in AC.AURORA_LINES:
        q = f"{line.quench_height_km:g} km" if line.quench_height_km else "none"
        print(f"  {line.name}: {line.wavelength_nm:g} nm, {line.species}, "
              f"strength {line.strength:g}, quench {q}, "
              f"N2 transfer {line.via_N2_transfer}")
    print()


def check_aurora():
    print("aurora checks")
    for line in AC.AURORA_LINES:
        r, g, b = AC.line_swatch(line.wavelength_nm)
        print(f"  {line.wavelength_nm:g} nm: ({r:.2f}, {g:.2f}, {b:.2f})")

    peaks = AC.peak_heights()
    print("  peak heights:")
    for name, h in peaks.items():
        print(f"    {name}: {h:.1f} km")
    h = np.linspace(C.AURORA_H_MIN_KM, C.AURORA_H_MAX_KM, 3201)
    P = AC.line_profiles(h)
    names = [line.name for line in AC.AURORA_LINES]
    green, red = P[:, names.index("O green")], P[:, names.index("O red")]
    above = h[red > green]
    above = above[above > 140]
    print(f"  red > green above {above.min():.0f} km")

    I = AC.integrated_intensity()
    print(f"  557.7/427.8 = {I['O green'] / I['N₂⁺ blue-violet']:.2f} (2-4)")
    print(f"  630.0/557.7 = {I['O red'] / I['O green']:.2f} (0.1-1)")
    print()
    return peaks


def run_stage3():
    print("stage 3, aurora colours")
    print_line_table()
    peaks = check_aurora()
    h = np.linspace(C.AURORA_H_MIN_KM, C.AURORA_H_MAX_KM, 641)
    fig = plotting.plot_colour_pipeline(AC.AURORA_LINES, h, AC.line_profiles(h), peaks)
    return {"ext_6d_colour_pipeline.png": fig}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "3", "5", "all"], default="all")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(exist_ok=True)
    figures = {}
    if args.stage in ("1", "all"):
        figures.update(run_stage1())
    if args.stage in ("3", "all"):
        figures.update(run_stage3())
    if args.stage in ("5", "all"):
        figures.update(run_stage5())

    for filename, fig in figures.items():
        fig.savefig(FIGURE_DIR / filename, dpi=150)
        print(f"Saved figures/{filename}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

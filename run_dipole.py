"""
Run the Stage 1 example: the dipole magnetosphere.

Prints validation checks and the L-shell table, and saves four figures to
the figures/ folder.

Usage:
    py run_dipole.py           (save figures only)
    py run_dipole.py --show    (also open the figures in windows,
                                useful for rotating the 3D plot)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from constants import R_E, M_PROTON
from field import (B_dipole, equatorial_surface_field, footpoint_latitude,
                   trace_field_line, magnetopause_standoff,
                   open_closed_boundary_latitude)
import plotting

FIGURE_DIR = Path(__file__).parent / "figures"

# Solar wind conditions: (density [cm⁻³], speed [km/s])
QUIET_SOLAR_WIND = (5.0, 400.0)
STORM_SOLAR_WIND = (20.0, 700.0)

FIELD_STRENGTH_SCALE = 1.0


def check_surface_field():
    print("=== Check 1: field strength at Earth's surface ===")
    B_eq = B_dipole([R_E, 0, 0], FIELD_STRENGTH_SCALE)
    B_pole = B_dipole([0, 0, R_E], FIELD_STRENGTH_SCALE)
    B0 = equatorial_surface_field(FIELD_STRENGTH_SCALE)
    print(f"  B0 formula (μ₀/4π)|m|/R_E³  = {B0:.4e} T   (expect ≈ 3.1e-5 T)")
    print(f"  |B| at equator  = {np.linalg.norm(B_eq):.4e} T,  vector = {B_eq}")
    print(f"  |B| at N pole   = {np.linalg.norm(B_pole):.4e} T,  vector = {B_pole}")
    print(f"  ratio pole/equator = {np.linalg.norm(B_pole) / np.linalg.norm(B_eq):.6f}"
          "   (expect 2)")
    print("  Direction: at the equator B points north (+z), and at the north")
    print("  pole it points down into Earth (−z), as expected for m = −|m| ẑ.\n")


def check_traced_lines():
    print("=== Check 2: traced field lines vs r = L·R_E·sin²θ ===")
    print("  Each line is traced numerically from its northern footpoint until")
    print("  it comes back down to the surface.  For every traced point we compute")
    print("  L_est = r / (R_E·sin²θ), which should equal L everywhere.")
    print(f"  {'L':>4} | {'max |L_est − L| / L':>20} | {'end lat [°]':>11} | expected")
    for L in range(2, 11):
        lam = np.radians(footpoint_latitude(L))
        start = R_E * np.array([np.cos(lam), 0.0, np.sin(lam)])
        points, end_reason = trace_field_line(start)
        r = np.linalg.norm(points, axis=1)
        sin_theta_sq = (points[:, 0] ** 2 + points[:, 1] ** 2) / r**2
        L_est = r / (R_E * sin_theta_sq)
        max_rel_err = np.max(np.abs(L_est - L)) / L
        end_lat = np.degrees(np.arcsin(points[-1, 2] / r[-1]))
        print(f"  {L:>4} | {max_rel_err:>20.2e} | {end_lat:>11.3f} | "
              f"{-footpoint_latitude(L):.3f}  ({end_reason})")
    print()


def print_L_shell_table(r_mp):
    print("=== L-shell footpoint latitudes  (cos²λ = 1/L) ===")
    print(f"  {'L':>4} | {'λ [°]':>7} | status (r_mp = {r_mp / R_E:.2f} R_E)")
    for L in range(2, 11):
        lam = footpoint_latitude(L)
        status = "closed" if L * R_E < r_mp else "open"
        note = "  ← auroral zone" if 65 <= lam <= 70 else ""
        print(f"  {L:>4} | {lam:>7.2f} | {status}{note}")
    print(f"  Open/closed boundary footpoint latitude: "
          f"{open_closed_boundary_latitude(r_mp):.2f}°\n")


def check_magnetopause():
    print("=== Check 3: magnetopause standoff distance ===")
    print("  r_mp = R_E · [ (2B₀)² / (2μ₀·ρ·v²) ]^(1/6)")
    results = {}
    for name, (n, v) in [("quiet", QUIET_SOLAR_WIND), ("storm", STORM_SOLAR_WIND)]:
        r_mp = magnetopause_standoff(n, v, FIELD_STRENGTH_SCALE)
        p_dyn_nPa = n * 1e6 * M_PROTON * (v * 1e3) ** 2 * 1e9
        print(f"  {name}: n = {n:g} cm⁻³, v = {v:g} km/s, ρv² = {p_dyn_nPa:.2f} nPa"
              f"  →  r_mp = {r_mp / R_E:.2f} R_E")
        results[name] = r_mp
    shrinks = results["storm"] < results["quiet"]
    print(f"  Storm boundary is closer to Earth than quiet: {shrinks}\n")
    return results["quiet"]


def main():
    # The Windows console defaults to an old encoding that cannot print
    # symbols like μ₀ or ², so switch the output to UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    show = "--show" in sys.argv

    check_surface_field()
    check_traced_lines()
    r_mp = check_magnetopause()
    print_L_shell_table(r_mp)

    FIGURE_DIR.mkdir(exist_ok=True)
    L_values = [2, 3, 4, 6, 8, 10, 12, 15]

    figures = {
        "1_meridional_field_lines.png":
            plotting.plot_meridional_field_lines(L_values, r_mp, FIELD_STRENGTH_SCALE),
        "2_field_strength_map.png":
            plotting.plot_field_strength_map(r_mp, FIELD_STRENGTH_SCALE),
        "3_field_lines_3d.png":
            plotting.plot_3d_field_lines([3, 6, 9, 12], 8, r_mp, FIELD_STRENGTH_SCALE),
        "4_footpoint_latitude.png":
            plotting.plot_footpoint_latitude(r_mp),
    }
    for filename, fig in figures.items():
        fig.savefig(FIGURE_DIR / filename, dpi=150)
        print(f"Saved figures/{filename}")

    if show:
        plt.show()


if __name__ == "__main__":
    main()

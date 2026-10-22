import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from constants import R_E, M_PROTON
from field import (B_dipole, equatorial_surface_field, footpoint_latitude,
                   trace_field_line, magnetopause_standoff,
                   open_closed_boundary_latitude)
from particles import (PROTON, ELECTRON, initial_conditions, speed_from_energy,
                       check_speed, bounce_period_estimate, drift_period_estimate)
from pusher import run_particle, measured_bounce_period, measured_drift_period
import plotting

FIGURE_DIR = Path(__file__).parent / "figures"

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


# ===========================================================================
# Stage 2: single particles
# ===========================================================================

DT_FRACTION = 1 / 30

ELECTRON_BOUNCE_FRACTION = 0.55


def print_launch_info(info):
    trapped = info["pitch_angle_deg"] > info["loss_cone_deg"]
    print(f"  {info['species']}, K = {info['energy_eV']:.4g} eV, launched at "
          f"L = {info['L']:g} on the equator, pitch angle {info['pitch_angle_deg']:g}°")
    print(f"    speed v             = {info['speed']:.4e} m/s  ({info['v_over_c']:.4f} c)")
    print(f"    v∥, v⊥              = {info['v_par']:.4e}, {info['v_perp']:.4e} m/s")
    print(f"    local |B|           = {info['B_mag']:.4e} T")
    print(f"    Larmor radius r_L   = {info['larmor_radius']:.4e} m  "
          f"({info['larmor_radius'] / R_E:.4g} R_E)")
    print(f"    gyroperiod T_c      = {info['gyroperiod']:.4e} s")
    print(f"    magnetic moment μ   = {info['magnetic_moment']:.4e} J/T")
    print(f"    loss cone at this L = {info['loss_cone_deg']:.2f}°  "
          f"({'trapped' if trapped else 'INSIDE loss cone: should hit Earth'})")
    print(f"    |B| at mirror point = {info['B_mirror']:.4e} T  (sets the time step)")
    check_speed(info["speed"])


def print_run_checks(result, info, species):
    print(f"    dt = {result['dt']:.3e} s, {result['n_steps']:,} steps, "
          f"{result['runtime']:.1f} s of computer time "
          f"({1e6 * result['runtime'] / result['n_steps']:.0f} μs per step)")
    print(f"    ended because: {result['end_reason']}")

    K = 0.5 * species.mass * np.sum(result["v"] ** 2, axis=1)
    print(f"    max |K − K₀|/K₀ = {np.max(np.abs(K - K[0]) / K[0]):.2e}  "
          "(Boris with no E field: rounding error only)")

    if result["hit"] is not None:
        h = result["hit"]
        print(f"    HIT EARTH at t = {h['time']:.4f} s:  latitude {h['latitude_deg']:.2f}°, "
              f"longitude {h['longitude_deg']:.2f}° from noon (MLT {h['mlt_hours']:.2f} h)")
        print(f"    expected footpoint latitude for L = {info['L']:g}: "
              f"±{footpoint_latitude(info['L']):.2f}°")
        return

    tb = measured_bounce_period(result)
    tb_est = bounce_period_estimate(info["L"], info["speed"], info["pitch_angle_deg"])
    if tb is None:
        print(f"    bounce period: run too short to measure (estimate {tb_est:.4g} s)")
        return
    print(f"    bounce period: simulated {tb:.4g} s, estimate {tb_est:.4g} s "
          f"({100 * (tb / tb_est - 1):+.1f}%)")

    td, direction = measured_drift_period(result)
    if td is None:
        print("    drift: run shorter than one bounce, not measured")
        return
    td_est = drift_period_estimate(info["L"], info["energy_eV"],
                                   info["pitch_angle_deg"], species)
    print(f"    drift: {direction}, period {td:.4g} s, estimate {td_est:.4g} s "
          f"({100 * (td / td_est - 1):+.1f}%)")


def run_case(label, species, energy_eV, L, pitch_angle_deg, t_max, r_mp,
             save_every=1):
    print(f"--- {label} ---")
    r0, v0, info = initial_conditions(species, energy_eV, L, pitch_angle_deg,
                                      field_strength_scale=FIELD_STRENGTH_SCALE)
    print_launch_info(info)
    print(f"    running for up to {t_max:.4g} s ...")
    result = run_particle(r0, v0, species, t_max, info["B_mirror"],
                          dt_fraction=DT_FRACTION,
                          field_strength_scale=FIELD_STRENGTH_SCALE,
                          r_outer=r_mp, save_every=save_every)
    print_run_checks(result, info, species)
    print()
    return result, info


def run_stage2(r_mp):
    print("=" * 70)
    print("STAGE 2: single particles in the dipole field (Boris pusher)")
    print("=" * 70)
    figures = {}
    L, alpha = 4.0, 45.0

    E_p = 10e6
    t_drift = drift_period_estimate(L, E_p, alpha, PROTON)
    res, _ = run_case("10 MeV proton, trapped", PROTON, E_p, L, alpha,
                      1.05 * t_drift, r_mp)
    title = "10 MeV proton, L = 4, pitch angle 45°"
    figures["5_proton_trajectory_3d.png"] = plotting.plot_trajectory_3d(
        res, PROTON, L, r_mp, title + " (one drift orbit)")

    E_lc = 1e6
    t_bounce_lc = bounce_period_estimate(L, speed_from_energy(E_lc, PROTON), 3.0)
    run_case("1 MeV proton, inside the loss cone", PROTON,
             E_lc, L, 3.0, 0.5 * t_bounce_lc, r_mp)

    E_e = 1e3
    t_bounce_e = bounce_period_estimate(L, speed_from_energy(E_e, ELECTRON), alpha)
    run_case("1 keV electron, trapped", ELECTRON, E_e, L, alpha,
             ELECTRON_BOUNCE_FRACTION * t_bounce_e, r_mp, save_every=5)
    return figures


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all")
    parser.add_argument("--show", action="store_true",
                        help="open the figures in windows as well as saving them")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(exist_ok=True)
    figures = {}
    animations = {}

    if args.stage in ("1", "all"):
        check_surface_field()
        check_traced_lines()
        r_mp = check_magnetopause()
        print_L_shell_table(r_mp)
        field_fig = plotting.plot_field_direction_3d(
            r_mp, field_strength_scale=FIELD_STRENGTH_SCALE)
        figures["6_field_direction_3d.png"] = field_fig
        animations["6_field_direction_3d.gif"] = plotting.animate_rotation(field_fig)
    else:
        r_mp = magnetopause_standoff(*QUIET_SOLAR_WIND, FIELD_STRENGTH_SCALE)

    if args.stage in ("2", "all"):
        figures.update(run_stage2(r_mp))

    for filename, fig in figures.items():
        fig.savefig(FIGURE_DIR / filename, dpi=150)
        print(f"Saved figures/{filename}")

    for filename, anim in animations.items():
        anim.save(FIGURE_DIR / filename, writer="pillow", dpi=80)
        print(f"Saved figures/{filename}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

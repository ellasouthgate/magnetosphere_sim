"""
Stage 2 validation checks: launch values, the Boris pusher, and the
particle motion in the dipole field.

Run from the project folder with:   py -m pytest
The whole file takes roughly 10–20 seconds.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from constants import R_E, E_CHARGE, M_PROTON
from field import equatorial_surface_field, footpoint_latitude
from particles import (PROTON, ELECTRON, initial_conditions, speed_from_energy,
                       gyroperiod, larmor_radius, loss_cone_angle, check_speed,
                       bounce_period_estimate, drift_period_estimate)
from pusher import (boris_rotate, run_particle, measured_bounce_period,
                    measured_drift_period)


# ---------------------------------------------------------------------------
# Launch values (the numbers teammates check by hand)
# ---------------------------------------------------------------------------

def test_launch_values_for_10_MeV_proton_at_L4():
    r0, v0, info = initial_conditions(PROTON, 10e6, 4, 45)

    # Done "by hand" from the formulas, independently of the code:
    v = np.sqrt(2 * 10e6 * E_CHARGE / M_PROTON)
    B = equatorial_surface_field() / 4**3          # equator: B = B₀ / L³
    v_perp = v * np.sin(np.radians(45))
    assert info["speed"] == pytest.approx(v, rel=1e-12)
    assert info["B_mag"] == pytest.approx(B, rel=1e-12)
    assert info["larmor_radius"] == pytest.approx(M_PROTON * v_perp / (E_CHARGE * B), rel=1e-12)
    assert info["gyroperiod"] == pytest.approx(2 * np.pi * M_PROTON / (E_CHARGE * B), rel=1e-12)

    # Rounded values printed by run_dipole.py
    assert info["speed"] == pytest.approx(4.377e7, rel=1e-3)
    assert info["larmor_radius"] == pytest.approx(6.684e5, rel=1e-3)
    assert info["gyroperiod"] == pytest.approx(0.1357, rel=1e-3)

    # Position and velocity actually have the requested size and pitch angle.
    assert np.linalg.norm(r0) == pytest.approx(4 * R_E)
    assert np.linalg.norm(v0) == pytest.approx(v)
    b_hat = np.array([0, 0, 1.0])                  # B points north at the equator
    pitch = np.degrees(np.arccos(np.dot(v0, b_hat) / np.linalg.norm(v0)))
    assert pitch == pytest.approx(45)


def test_loss_cone_matches_dipole_formula():
    # sin²α_lc = 1 / (L³·√(4 − 3/L))
    for L in [2, 4, 6.5]:
        expected = np.degrees(np.arcsin(np.sqrt(1 / (L**3 * np.sqrt(4 - 3 / L)))))
        assert loss_cone_angle(L) == pytest.approx(expected, rel=1e-10)


def test_relativistic_warning():
    assert check_speed(speed_from_energy(1e3, ELECTRON)) is True      # 0.06 c
    assert check_speed(speed_from_energy(10e6, PROTON)) is False      # 0.15 c


# ---------------------------------------------------------------------------
# The Boris step on its own, in a uniform field
# ---------------------------------------------------------------------------

def test_boris_rotation_keeps_speed_exactly():
    rng = np.random.default_rng(1)
    v = rng.normal(size=3) * 1e7
    B = rng.normal(size=3) * 1e-6
    for _ in range(1000):
        v_new = boris_rotate(v, B, PROTON.charge / PROTON.mass, 0.01)
        assert np.linalg.norm(v_new) == pytest.approx(np.linalg.norm(v), rel=1e-13)
        v = v_new


def test_uniform_field_circle_has_larmor_radius_and_gyroperiod():
    """
    In a uniform field the orbit is a circle of radius r_L, completed in one
    gyroperiod T_c.  Push a proton for one T_c with 200 steps and check.
    """
    B = np.array([0, 0, 1e-6])
    v_perp = 1e6
    q_over_m = PROTON.charge / PROTON.mass
    T_c = gyroperiod(1e-6, PROTON)
    r_L = larmor_radius(v_perp, 1e-6, PROTON)
    n = 200
    dt = T_c / n

    r = np.zeros(3)
    v0 = np.array([v_perp, 0, 0])
    v = boris_rotate(v0, B, q_over_m, -dt / 2)
    path = [r]
    for _ in range(n):
        v = boris_rotate(v, B, q_over_m, dt)
        r = r + v * dt
        path.append(r)
    path = np.array(path)

    centre = path.mean(axis=0)
    radii = np.linalg.norm(path - centre, axis=1)
    assert radii.mean() == pytest.approx(r_L, rel=1e-3)
    # Back at the start after one gyroperiod (small phase error only).
    assert np.linalg.norm(path[-1] - path[0]) < 0.01 * r_L


# ---------------------------------------------------------------------------
# Motion in the dipole field
# ---------------------------------------------------------------------------

def _run(species, energy_eV, L, pitch_deg, t_max, **kwargs):
    r0, v0, info = initial_conditions(species, energy_eV, L, pitch_deg)
    result = run_particle(r0, v0, species, t_max, info["B_mirror"],
                          progress=False, **kwargs)
    return result, info


@pytest.fixture(scope="module")
def trapped_proton():
    """1 MeV proton, L = 4, 45° pitch angle, followed for 3 bounces."""
    t_b = bounce_period_estimate(4, speed_from_energy(1e6, PROTON), 45)
    return _run(PROTON, 1e6, 4, 45, 3 * t_b)


def test_energy_is_conserved(trapped_proton):
    result, _ = trapped_proton
    K = 0.5 * PROTON.mass * np.sum(result["v"] ** 2, axis=1)
    assert np.max(np.abs(K - K[0])) / K[0] < 1e-12


def test_bounce_period_matches_estimate(trapped_proton):
    result, info = trapped_proton
    tb = measured_bounce_period(result)
    assert tb == pytest.approx(bounce_period_estimate(4, info["speed"], 45), rel=0.05)


def test_particle_mirrors_where_mu_conservation_predicts(trapped_proton):
    """
    If μ is conserved the particle mirrors where B = B_eq / sin²α.
    Along a dipole field line  B(λ) / B_eq = √(1 + 3 sin²λ) / cos⁶λ,
    so solve that for the mirror latitude and compare with the highest
    latitude the particle actually reached.
    """
    result, _ = trapped_proton
    ratio = 1 / np.sin(np.radians(45)) ** 2
    lam_m = np.degrees(brentq(
        lambda lam: np.sqrt(1 + 3 * np.sin(lam) ** 2) / np.cos(lam) ** 6 - ratio,
        0, np.pi / 2 - 0.01))

    r = result["r"]
    lat = np.degrees(np.arcsin(r[:, 2] / np.linalg.norm(r, axis=1)))
    assert lat.max() == pytest.approx(lam_m, abs=0.5)
    assert -lat.min() == pytest.approx(lam_m, abs=0.5)


def test_proton_drifts_west_at_the_expected_rate():
    t_d = drift_period_estimate(4, 10e6, 45, PROTON)
    result, _ = _run(PROTON, 10e6, 4, 45, 0.3 * t_d)
    period, direction = measured_drift_period(result)
    assert direction == "westward"
    assert period == pytest.approx(t_d, rel=0.05)


def test_particle_in_loss_cone_hits_earth_at_footpoint():
    t_b = bounce_period_estimate(4, speed_from_energy(1e6, PROTON), 3)
    result, info = _run(PROTON, 1e6, 4, 3, 0.5 * t_b)
    assert info["pitch_angle_deg"] < info["loss_cone_deg"]
    assert result["end_reason"] == "hit_earth"
    hit = result["hit"]
    assert np.linalg.norm(hit["position"]) == pytest.approx(R_E, rel=1e-9)
    assert abs(hit["latitude_deg"]) == pytest.approx(footpoint_latitude(4), abs=0.2)
    assert 0 <= hit["mlt_hours"] < 24
    # The stored trajectory ends exactly at the impact point.
    assert np.allclose(result["r"][-1], hit["position"])


def test_run_stops_at_outer_boundary():
    # The 10 MeV proton's gyration takes it out to ~4.1 R_E.
    result, _ = _run(PROTON, 10e6, 4, 45, 1.0, r_outer=4.05 * R_E)
    assert result["end_reason"] == "left_boundary"


def test_electron_gyrates_opposite_way_to_proton():
    # Gyration sense: protons go round B clockwise (left-handed), electrons
    # anticlockwise.  Check the sign of (v × v_next) · B after one step.
    B = np.array([0, 0, 1e-6])
    v = np.array([1e6, 0, 0])
    for species, sign in [(PROTON, -1), (ELECTRON, +1)]:
        dt = gyroperiod(1e-6, species) / 30
        v_next = boris_rotate(v, B, species.charge / species.mass, dt)
        assert np.sign(np.dot(np.cross(v, v_next), B)) == sign

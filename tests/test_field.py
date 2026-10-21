"""
Stage 1 validation checks: the dipole field, L-shells and the magnetopause.

Run from the project folder with:   py -m pytest
"""

import numpy as np
import pytest

from constants import R_E, MU_0, M_DIPOLE
from field import (B_dipole, equatorial_surface_field, footpoint_latitude,
                   trace_field_line, magnetopause_standoff, field_line_for_L)


# ---------------------------------------------------------------------------
# The dipole field
# ---------------------------------------------------------------------------

def test_equatorial_surface_field_is_about_3_1e_minus_5_tesla():
    B_eq = np.linalg.norm(B_dipole([R_E, 0, 0]))
    assert B_eq == pytest.approx(3.1e-5, rel=0.02)
    assert B_eq == pytest.approx(equatorial_surface_field(), rel=1e-12)


def test_polar_field_is_twice_equatorial():
    B_eq = np.linalg.norm(B_dipole([R_E, 0, 0]))
    B_north = np.linalg.norm(B_dipole([0, 0, R_E]))
    B_south = np.linalg.norm(B_dipole([0, 0, -R_E]))
    assert B_north == pytest.approx(2 * B_eq, rel=1e-12)
    assert B_south == pytest.approx(2 * B_eq, rel=1e-12)


def test_field_direction():
    # m = −|m| ẑ: the field points north (+z) at the equator, and down
    # into Earth (−z) at the north pole.
    B_eq = B_dipole([R_E, 0, 0])
    B_np = B_dipole([0, 0, R_E])
    assert B_eq[2] > 0 and B_np[2] < 0
    assert np.allclose(B_eq[:2], 0) and np.allclose(B_np[:2], 0)


def test_field_falls_off_as_one_over_r_cubed():
    B1 = np.linalg.norm(B_dipole([R_E, 0, 0]))
    B4 = np.linalg.norm(B_dipole([4 * R_E, 0, 0]))
    assert B4 == pytest.approx(B1 / 4**3, rel=1e-12)


def test_matches_spherical_formula_at_any_latitude():
    # Standard result: |B| = (μ₀/4π)·|m|/r³ · √(1 + 3 sin²λ)
    for lam_deg in [-75, -30, 0, 10, 45, 80]:
        lam = np.radians(lam_deg)
        r = 2.5 * R_E
        pos = r * np.array([np.cos(lam), 0, np.sin(lam)])
        expected = MU_0 / (4 * np.pi) * M_DIPOLE / r**3 * np.sqrt(1 + 3 * np.sin(lam) ** 2)
        assert np.linalg.norm(B_dipole(pos)) == pytest.approx(expected, rel=1e-12)


def test_works_for_arrays_of_points():
    points = np.array([[R_E, 0, 0], [0, 2 * R_E, 0], [0, 0, 3 * R_E], [R_E, R_E, R_E]])
    B_all = B_dipole(points)
    assert B_all.shape == (4, 3)
    for p, B in zip(points, B_all):
        assert np.allclose(B, B_dipole(p), rtol=1e-14, atol=0)

    grid = np.ones((5, 7, 3)) * R_E
    assert B_dipole(grid).shape == (5, 7, 3)


def test_field_strength_scale_multiplies_the_field():
    p = [3 * R_E, R_E, -R_E]
    assert np.allclose(B_dipole(p, 2.5), 2.5 * B_dipole(p), rtol=1e-14, atol=0)
    assert np.all(B_dipole(p, 0.0) == 0)


# ---------------------------------------------------------------------------
# L-shells
# ---------------------------------------------------------------------------

def test_footpoint_latitude_known_values():
    # cos²λ = 1/L
    assert footpoint_latitude(1) == pytest.approx(0.0, abs=1e-12)
    assert footpoint_latitude(2) == pytest.approx(45.0, rel=1e-12)
    assert footpoint_latitude(4) == pytest.approx(60.0, rel=1e-12)


def test_auroral_zone_maps_to_L_6_to_8():
    for L in [6, 7, 8]:
        assert 65 <= footpoint_latitude(L) <= 70


def test_footpoint_latitude_rejects_L_below_1():
    with pytest.raises(ValueError):
        footpoint_latitude(0.5)


@pytest.mark.parametrize("L", [2, 4, 6, 8, 10])
def test_traced_field_line_follows_r_equals_L_sin_squared_theta(L):
    lam = np.radians(footpoint_latitude(L))
    start = R_E * np.array([np.cos(lam), 0.0, np.sin(lam)])
    points, end_reason = trace_field_line(start)

    r = np.linalg.norm(points, axis=1)
    sin_theta_sq = (points[:, 0] ** 2 + points[:, 1] ** 2) / r**2
    L_est = r / (R_E * sin_theta_sq)
    assert np.max(np.abs(L_est - L)) / L < 1e-6

    # The line should come down at the mirror-image southern footpoint,
    # and cross the equator at L·R_E.
    assert end_reason == "surface"
    end_lat = np.degrees(np.arcsin(points[-1, 2] / r[-1]))
    assert end_lat == pytest.approx(-footpoint_latitude(L), abs=1e-4)
    assert r.max() == pytest.approx(L * R_E, rel=1e-4)


# ---------------------------------------------------------------------------
# Magnetopause
# ---------------------------------------------------------------------------

def test_quiet_standoff_is_about_10_earth_radii():
    r_mp = magnetopause_standoff(n_cm3=5, v_kms=400)
    assert 9 * R_E < r_mp < 11 * R_E


def test_standoff_shrinks_in_a_storm():
    quiet = magnetopause_standoff(5, 400)
    storm = magnetopause_standoff(20, 700)
    assert storm < quiet
    # r_mp ∝ (ρv²)^(−1/6)
    pressure_ratio = (20 * 700**2) / (5 * 400**2)
    assert storm / quiet == pytest.approx(pressure_ratio ** (-1 / 6), rel=1e-12)


def test_standoff_scales_as_cube_root_of_field_strength():
    base = magnetopause_standoff(5, 400)
    assert magnetopause_standoff(5, 400, 8.0) == pytest.approx(2 * base, rel=1e-12)
    assert magnetopause_standoff(5, 400, 0.0) == 0.0


def test_open_and_closed_field_lines():
    r_mp = magnetopause_standoff(5, 400)      # ≈ 10.2 R_E

    segments, status = field_line_for_L(4, 0.0, r_mp)
    assert status == "closed" and len(segments) == 1

    segments, status = field_line_for_L(12, 0.0, r_mp)
    assert status == "open" and len(segments) == 2
    for seg in segments:
        # Drawing stops at the boundary (within one tracing step).
        assert np.linalg.norm(seg, axis=1).max() <= r_mp + 0.03 * R_E

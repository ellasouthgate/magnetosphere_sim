import numpy as np
import pytest

from constants import R_E, MU_0, M_DIPOLE, M_DIPOLE_VEC
from config import DEFAULT_FIELD, OLD_MODEL
from field import B_dipole, trace_field_line, local_time_hours
from sources import (dipole_field, geographic_north, magnetic_north,
                     earth_moment, B_earth, B_sun, B_total, field_function,
                     sun_moment, sun_position, sun_field_at_earth_magnitude,
                     earth_equatorial_surface_field, effective_r_balance,
                     rotation_about_y)
from topology import (classify_footpoint, classify_seed, meridian_footpoint,
                      chi_to_latitude, magnetic_latitude, open_closed_boundary,
                      find_nulls, find_nulls_xz, describe_nulls)

RNG = np.random.default_rng(42)
POINTS = RNG.uniform(-20, 20, size=(500, 3)) * R_E
POINTS = POINTS[np.linalg.norm(POINTS, axis=1) > 1.2 * R_E]

UNIFORM = DEFAULT_FIELD.with_(tilt_deg=0.0, sun_distance_RE=1e5)


def old_B_dipole(r):
    """B_dipole formula written out."""
    r = np.asarray(r, dtype=float)
    r_mag = np.linalg.norm(r, axis=-1, keepdims=True)
    r_hat = r / r_mag
    m_dot_rhat = np.sum(r_hat * M_DIPOLE_VEC, axis=-1, keepdims=True)
    return (MU_0 / (4 * np.pi)) * (3 * m_dot_rhat * r_hat - M_DIPOLE_VEC) / r_mag**3


def test_B_dipole_is_bit_for_bit_unchanged():
    assert np.array_equal(B_dipole(POINTS), old_B_dipole(POINTS))


def test_dipole_field_shifts_with_position():
    m = np.array([1e22, -2e22, 3e22])
    shift = np.array([5, -3, 2]) * R_E
    assert np.allclose(dipole_field(POINTS + shift, m, shift),
                       dipole_field(POINTS, m), rtol=1e-12, atol=0)


def test_dipole_field_is_linear_in_moment():
    m1, m2 = np.array([1e22, 0, 0]), np.array([0, 2e22, -1e22])
    assert np.allclose(dipole_field(POINTS, m1 + m2),
                       dipole_field(POINTS, m1) + dipole_field(POINTS, m2),
                       rtol=1e-12, atol=1e-30)


def test_dipole_field_rotates_with_moment():
    R = rotation_about_y(37.0)
    m = np.array([0.0, 0.0, -M_DIPOLE])
    p = np.array([3.0, 1.0, 2.0]) * R_E
    assert np.allclose(dipole_field(R @ p, R @ m), R @ dipole_field(p, m),
                       rtol=1e-12, atol=0)


def test_dipole_field_is_divergence_free_and_curl_free():
    m = np.array([2e22, -1e22, 5e22])
    p, h = np.array([2.0, -1.5, 1.0]) * R_E, 1e-4 * R_E
    J = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        J[:, j] = (dipole_field(p + e, m) - dipole_field(p - e, m)) / (2 * h)
    scale = np.abs(J).max()
    assert abs(np.trace(J)) < 1e-8 * scale
    assert np.allclose(J, J.T, atol=1e-8 * scale)


def test_rotation_axis_leans_towards_the_sun():
    n = geographic_north(DEFAULT_FIELD.with_(tilt_deg=23.4))
    assert n[0] < 0 and n[1] == 0
    assert np.degrees(np.arccos(n[2])) == pytest.approx(23.4)


def test_zero_tilt_moment_is_old_moment():
    assert np.allclose(earth_moment(OLD_MODEL), M_DIPOLE_VEC, rtol=0, atol=0)


def test_earth_moment_points_to_geographic_south():
    for tilt in (0, 10, 23.4, 45):
        cfg = DEFAULT_FIELD.with_(tilt_deg=tilt)
        m = earth_moment(cfg)
        assert np.linalg.norm(m) == pytest.approx(M_DIPOLE)
        assert np.dot(m, geographic_north(cfg)) < 0


def test_offset_towards_sun_adds_to_tilt():
    cfg = DEFAULT_FIELD.with_(tilt_deg=23.4, magnetic_offset_deg=11.0)
    angle = np.degrees(np.arccos(magnetic_north(cfg)[2]))
    assert angle == pytest.approx(34.4)
    assert np.degrees(np.arccos(np.dot(magnetic_north(cfg),
                                       geographic_north(cfg)))) == pytest.approx(11.0)


def test_offset_azimuth_90_leans_towards_dusk():
    cfg = DEFAULT_FIELD.with_(tilt_deg=0.0, magnetic_offset_deg=11.0,
                              magnetic_offset_azimuth_deg=90.0)
    n = magnetic_north(cfg)
    assert n[1] < 0 and abs(n[0]) < 1e-15


def test_tilted_field_is_rotated_untilted_field():
    cfg = DEFAULT_FIELD.with_(w_sun=0.0, tilt_deg=23.4)
    R = rotation_about_y(-23.4)
    p = np.array([2.0, 0.5, -1.0]) * R_E
    assert np.allclose(B_earth(R @ p, cfg), R @ B_dipole(p), rtol=1e-12, atol=0)


def test_sun_sits_on_minus_x():
    p = sun_position(DEFAULT_FIELD)
    assert p[0] == pytest.approx(-1000 * R_E) and p[1] == 0 and p[2] == 0


def test_r_balance_sets_field_at_earth():
    cfg = DEFAULT_FIELD.with_(r_balance_RE=10.0)
    B_at_earth = B_sun(np.zeros(3), cfg)
    expected = earth_equatorial_surface_field(cfg) / 10.0**3
    assert np.linalg.norm(B_at_earth) == pytest.approx(expected, rel=1e-12)
    assert np.linalg.norm(B_at_earth) == pytest.approx(31e-9, rel=0.02)
    B_e = np.linalg.norm(B_dipole([10 * R_E, 0, 0]))
    assert B_e == pytest.approx(sun_field_at_earth_magnitude(cfg), rel=1e-12)


@pytest.mark.parametrize("direction, sign", [("south", -1), ("north", +1)])
def test_sun_field_direction_at_earth(direction, sign):
    cfg = DEFAULT_FIELD.with_(sun_field_direction=direction)
    B = B_sun(np.zeros(3), cfg)
    assert np.sign(B[2]) == sign
    assert np.allclose(B[:2], 0, atol=1e-25)
    assert sun_moment(cfg)[2] == pytest.approx(-sign * abs(sun_moment(cfg)[2]))


def test_bad_sun_direction_raises():
    with pytest.raises(ValueError):
        sun_moment(DEFAULT_FIELD.with_(sun_field_direction="east"))


def test_sun_field_is_nearly_uniform_near_earth():
    B0 = B_sun(np.zeros(3), DEFAULT_FIELD)
    for p in ([10 * R_E, 0, 0], [0, 0, 10 * R_E], [-10 * R_E, 5 * R_E, 0]):
        rel = np.linalg.norm(B_sun(p, DEFAULT_FIELD) - B0) / np.linalg.norm(B0)
        assert rel < 0.04          # ≈ 3·r/D = 3 %


def test_effective_r_balance():
    assert effective_r_balance(DEFAULT_FIELD.with_(w_sun=8.0)) == pytest.approx(5.0)
    assert effective_r_balance(DEFAULT_FIELD.with_(w_sun=0.0)) == np.inf


def test_old_model_reproduced_exactly():
    assert np.array_equal(B_total(POINTS, OLD_MODEL), B_dipole(POINTS))
    assert np.array_equal(field_function(OLD_MODEL)(POINTS[0]), B_dipole(POINTS[0]))


def test_superposition():
    cfg = DEFAULT_FIELD.with_(w_sun=0.7, magnetic_offset_deg=11.0)
    assert np.allclose(B_total(POINTS, cfg),
                       B_earth(POINTS, cfg) + 0.7 * B_sun(POINTS, cfg),
                       rtol=1e-14, atol=0)


def test_field_function_matches_B_total():
    f = field_function(DEFAULT_FIELD)
    for p in POINTS[:20]:
        assert np.allclose(f(p), B_total(p, DEFAULT_FIELD), rtol=1e-14, atol=0)


def test_near_earth_is_earth_far_away_is_sun():
    cfg = DEFAULT_FIELD
    for r in (1.0, 2.0):
        p = np.array([0.0, 0.0, r]) * R_E
        rel = np.linalg.norm(B_total(p, cfg) - B_earth(p, cfg)) / np.linalg.norm(B_earth(p, cfg))
        assert rel < 0.01
    for r in (100.0, 200.0):
        p = np.array([0.0, 0.0, r]) * R_E
        Bs = cfg.w_sun * B_sun(p, cfg)
        assert np.linalg.norm(B_total(p, cfg) - Bs) / np.linalg.norm(Bs) < 0.01


def test_w_sun_weights_the_solar_part():
    p = np.array([5.0, 1.0, 3.0]) * R_E
    B1 = B_total(p, DEFAULT_FIELD.with_(w_sun=1.0))
    B3 = B_total(p, DEFAULT_FIELD.with_(w_sun=3.0))
    assert np.allclose(B3 - B1, 2 * B_sun(p, DEFAULT_FIELD), rtol=1e-12, atol=1e-25)


def test_trace_field_line_default_unchanged():
    start = R_E * np.array([np.cos(1.0), 0.0, np.sin(1.0)])
    a, ea = trace_field_line(start)
    b, eb = trace_field_line(start, field_func=lambda r: B_dipole(r))
    assert ea == eb == "surface" and np.array_equal(a, b)


def test_trace_direction_argument():
    seed = np.array([3.0, 0.0, 0.0]) * R_E      # equator, B points +z there
    up, _ = trace_field_line(seed, direction=+1, max_step=0.05 * R_E)
    down, _ = trace_field_line(seed, direction=-1, max_step=0.05 * R_E)
    assert up[-1, 2] > 0 and down[-1, 2] < 0


def test_classification_closed_open_solar():
    cfg = DEFAULT_FIELD.with_(tilt_deg=0.0)
    assert classify_footpoint(meridian_footpoint(30), cfg)["status"] == "closed"
    assert classify_footpoint(meridian_footpoint(90), cfg)["status"] == "open"
    assert classify_seed(np.array([-25.0, 0, 0]) * R_E, cfg)["status"] == "solar"
    assert classify_seed(np.array([3.0, 0, 0]) * R_E, cfg)["status"] == "closed"


def test_old_model_has_no_open_lines():
    line = classify_footpoint(meridian_footpoint(88), OLD_MODEL)
    assert line["status"] == "closed" and line["beyond_limit"]


def test_meridian_footpoints():
    assert np.allclose(meridian_footpoint(0) / R_E, [-1, 0, 0])         # noon
    assert np.allclose(meridian_footpoint(90, -1) / R_E, [0, 0, -1])    # S pole
    assert chi_to_latitude(30) == (30, "noon")
    assert chi_to_latitude(150, -1) == (-30, "midnight")
    assert magnetic_latitude(meridian_footpoint(60), OLD_MODEL) == pytest.approx(60)


def analytic_boundary_latitude(r_b):
    """Dipole + uniform southward field, no tilt (flux function argument)."""
    return np.degrees(np.arccos(np.sqrt((3 / (2 * r_b)) / (1 + 1 / (2 * r_b**3)))))


@pytest.fixture(scope="module")
def uniform_boundary():
    return open_closed_boundary(UNIFORM, +1, scan_step_deg=5.0, tol_deg=0.01)


def test_boundary_matches_analytic_result(uniform_boundary):
    assert len(uniform_boundary) == 2
    for b in uniform_boundary:
        assert b["latitude_deg"] == pytest.approx(analytic_boundary_latitude(10.0),
                                                  abs=0.05)
    assert {b["edge"] for b in uniform_boundary} == {"dayside", "nightside"}


def test_tilt_moves_the_polar_cap_towards_the_sun():
    cfg = UNIFORM.with_(tilt_deg=23.4)
    north = open_closed_boundary(cfg, +1, scan_step_deg=5.0, tol_deg=0.05)
    south = open_closed_boundary(cfg, -1, scan_step_deg=5.0, tol_deg=0.05)
    assert all(b["chi_deg"] < 95 for b in north)
    assert all(b["chi_deg"] > 85 for b in south)
    n_day = [b for b in north if b["edge"] == "dayside"][0]
    s_night = [b for b in south if b["edge"] == "nightside"][0]
    assert n_day["latitude_deg"] == pytest.approx(-s_night["latitude_deg"], abs=0.1)


def test_no_boundary_without_sun():
    assert open_closed_boundary(OLD_MODEL, +1, scan_step_deg=10.0) == []


def test_southward_null_ring_untilted():
    nulls = find_nulls(UNIFORM)
    d = describe_nulls(nulls)
    assert d["kind"] == "ring"
    assert d["radius"] == pytest.approx(10.0, rel=1e-3)
    assert abs(d["normal"][2]) == pytest.approx(1.0, abs=1e-6)
    for p in nulls:
        assert np.linalg.norm(B_total(p * R_E, UNIFORM)) < 1e-6 * 31e-9


def test_northward_nulls_above_the_poles():
    cfg = UNIFORM.with_(sun_field_direction="north")
    nulls = find_nulls(cfg)
    assert len(nulls) == 2
    for p in nulls:
        assert abs(p[2]) == pytest.approx(2 ** (1 / 3) * 10.0, rel=1e-3)
        assert np.hypot(p[0], p[1]) < 0.01


def test_tilted_nulls_are_two_points_with_zero_field():
    nulls = find_nulls_xz(DEFAULT_FIELD)
    assert len(nulls) == 2
    assert nulls[:, 0].min() < 0 < nulls[:, 0].max()     # one dayside, one nightside
    for p in nulls:
        assert np.linalg.norm(B_total(p * R_E, DEFAULT_FIELD)) < 1e-6 * 31e-9


def test_no_nulls_without_sun():
    assert len(find_nulls(OLD_MODEL)) == 0


def test_local_time_sun_on_minus_x():
    assert local_time_hours(180.0) == pytest.approx(12.0)    # −x: noon
    assert local_time_hours(0.0) == pytest.approx(0.0)       # +x: midnight
    assert local_time_hours(-90.0) == pytest.approx(18.0)    # −y: dusk
    assert local_time_hours(90.0) == pytest.approx(6.0)      # +y: dawn


def test_field_on_xz_grid():
    from sources import field_on_xz_grid
    x, z, B = field_on_xz_grid(DEFAULT_FIELD, half_width_RE=5.0, n=21)
    assert B.shape == (21, 21, 3)
    i, j = 3, 17
    expected = B_total(np.array([x[j], 0.0, z[i]]) * R_E, DEFAULT_FIELD) * 1e9
    assert np.allclose(B[i, j], expected)
    assert np.all(np.isnan(B[10, 10]))                          # centre of Earth
    assert np.allclose(B[..., 1][~np.isnan(B[..., 1])], 0.0, atol=1e-9)   # B_y = 0 in this plane

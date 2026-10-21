"""
Dipole model of Earth's magnetosphere.

Coordinate system (used everywhere in the project):
    - Origin at Earth's centre.
    - z axis along the magnetic dipole axis (= geographic north, no tilt).
    - x axis points towards the Sun, so the x–z plane is the noon–midnight
      meridian and x > 0 is the dayside.
    - y completes the right-handed set (points towards dusk).

All quantities are in SI units (metres, tesla, ...) unless the variable
name says otherwise (e.g. n_cm3, v_kms).

Contents:
    B_dipole                   magnetic field vector at a position
    equatorial_surface_field   B₀, the field strength at the equator on Earth's surface
    footpoint_latitude         where an L-shell meets Earth's surface
    L_shell_line               the analytic field line r = L·R_E·sin²θ
    trace_field_line           numerically follow the field from a start point
    magnetopause_standoff      dayside boundary distance from pressure balance
    field_line_for_L           trace one L-shell, cut at the magnetopause, label open/closed
"""

import numpy as np
from scipy.integrate import solve_ivp

from constants import MU_0, M_PROTON, R_E, M_DIPOLE, M_DIPOLE_VEC


# ---------------------------------------------------------------------------
# The dipole field
# ---------------------------------------------------------------------------

def B_dipole(r, field_strength_scale=1.0):
    """
    Magnetic field of Earth's dipole at position(s) r.

        B(r) = (μ₀/4π) · [3(m·r̂)r̂ − m] / |r|³

    with m = −|m| ẑ (see constants.py).

    Parameters
    ----------
    r : array, shape (3,) or (..., 3)
        Position(s) in metres.  The last axis holds (x, y, z), so a single
        point or an array of N points of shape (N, 3) both work.
    field_strength_scale : float
        Multiplies the whole field.  1.0 = real Earth.  Will be used later for
        weaker / stronger / no-field magnetospheres.

    Returns
    -------
    B : array, same shape as r
        Field vector(s) in tesla.  Undefined (division by zero) at r = 0.
    """
    r = np.asarray(r, dtype=float)

    # keepdims=True leaves a length-1 last axis, so these broadcast
    # correctly against the (..., 3) vectors below.
    r_mag = np.linalg.norm(r, axis=-1, keepdims=True)
    r_hat = r / r_mag
    m_dot_rhat = np.sum(r_hat * M_DIPOLE_VEC, axis=-1, keepdims=True)

    B = (MU_0 / (4 * np.pi)) * (3 * m_dot_rhat * r_hat - M_DIPOLE_VEC) / r_mag**3
    return field_strength_scale * B


def equatorial_surface_field(field_strength_scale=1.0):
    """
    B₀ = field strength on Earth's surface at the magnetic equator [T].

        B₀ = (μ₀/4π) · |m| / R_E³   (≈ 3.1e-5 T for Earth)

    At the poles the field is 2·B₀.
    """
    return field_strength_scale * (MU_0 / (4 * np.pi)) * M_DIPOLE / R_E**3


# ---------------------------------------------------------------------------
# L-shells (analytic)
# ---------------------------------------------------------------------------

def footpoint_latitude(L):
    """
    Magnetic latitude λ [degrees] where the L-shell meets Earth's surface.

    A dipole field line obeys r = L·R_E·sin²θ (θ = colatitude, measured from
    the z axis).  Setting r = R_E gives sin²θ = 1/L.  Latitude λ = 90° − θ,
    so sin θ = cos λ and therefore

        cos²λ = 1/L   →   λ = arccos(√(1/L))

    The same latitude applies in both hemispheres (±λ).
    """
    L = np.asarray(L, dtype=float)
    if np.any(L < 1):
        raise ValueError("L must be >= 1 (an L < 1 field line is inside Earth).")
    return np.degrees(np.arccos(np.sqrt(1.0 / L)))


def L_shell_line(L, phi=0.0, n_points=400):
    """
    Points on the analytic dipole field line r = L·R_E·sin²θ, from the
    northern footpoint to the southern footpoint, at longitude phi [rad].

    Returns an (n_points, 3) array of positions in metres.
    Used for checking the numerically traced lines.
    """
    lam_foot = np.radians(footpoint_latitude(L))
    # Colatitude runs from the northern footpoint to the southern one.
    theta = np.linspace(np.pi / 2 - lam_foot, np.pi / 2 + lam_foot, n_points)
    r = L * R_E * np.sin(theta) ** 2
    return np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])


# ---------------------------------------------------------------------------
# Numerical field-line tracing
# ---------------------------------------------------------------------------

def trace_field_line(start, r_outer=None, field_strength_scale=1.0,
                     max_step=0.02 * R_E, max_length=100 * R_E):
    """
    Follow the magnetic field from `start` by integrating

        dr/ds = ± B(r) / |B(r)|

    where s is the arc length along the line.  The sign is chosen so the line
    initially moves AWAY from Earth (useful when starting at a footpoint).

    Tracing stops when the line:
        - comes back down to Earth's surface (|r| = R_E), or
        - reaches the outer radius r_outer (e.g. the magnetopause), if given, or
        - has travelled max_length.

    This is a genuine numerical calculation (it only uses B_dipole), so
    comparing its result with r = L·R_E·sin²θ is a real test of B_dipole.

    Parameters
    ----------
    start : array, shape (3,)
        Starting point [m].  Should be outside Earth (|start| >= R_E).
    r_outer : float or None
        Stop if the line reaches this distance from Earth's centre [m].
    field_strength_scale : float
        Passed to B_dipole.  (It does not change the SHAPE of field lines,
        only the strength, but we pass it for consistency.)
    max_step : float
        Largest step in s [m].  Keeps the output points close together for
        smooth plots.

    Returns
    -------
    points : array, shape (N, 3)
        Positions along the line [m].
    end_reason : str
        "surface", "outer_boundary" or "max_length".
    """
    start = np.asarray(start, dtype=float)
    r_hat0 = start / np.linalg.norm(start)
    sign = 1.0 if np.dot(B_dipole(start, field_strength_scale), r_hat0) > 0 else -1.0

    def direction(s, r):
        B = B_dipole(r, field_strength_scale)
        return sign * B / np.linalg.norm(B)

    # Event: line comes back down through Earth's surface (only counts when
    # moving inwards, so it does not trigger at the start on the surface).
    def hit_surface(s, r):
        return np.linalg.norm(r) - R_E
    hit_surface.terminal = True
    hit_surface.direction = -1

    events = [hit_surface]

    if r_outer is not None:
        # Event: line reaches the outer boundary while moving outwards.
        def hit_outer(s, r):
            return np.linalg.norm(r) - r_outer
        hit_outer.terminal = True
        hit_outer.direction = +1
        events.append(hit_outer)

    sol = solve_ivp(direction, (0.0, max_length), start, events=events,
                    max_step=max_step, rtol=1e-9, atol=1e-3)

    points = sol.y.T
    if sol.status == 1:   # stopped by an event
        if len(sol.t_events[0]) > 0:
            end_reason = "surface"
        else:
            end_reason = "outer_boundary"
    else:
        end_reason = "max_length"
    return points, end_reason


# ---------------------------------------------------------------------------
# Magnetopause (simple boundary)
# ---------------------------------------------------------------------------

def magnetopause_standoff(n_cm3, v_kms, field_strength_scale=1.0):
    """
    Dayside magnetopause standoff distance r_mp [m] from pressure balance.

    At the subsolar point (the point on the boundary facing the Sun) we
    balance the magnetic pressure of Earth's field against the dynamic
    pressure of the solar wind:

        B_mp² / (2μ₀) = ρ·v²

    The solar wind compresses the field so that just inside the boundary it
    is roughly TWICE the pure dipole value, B_mp ≈ 2·B₀·(R_E/r)³.
    Solving for r:

        r_mp = R_E · [ (2B₀)² / (2μ₀·ρ·v²) ]^(1/6)

    with ρ = n·m_p (solar wind treated as pure hydrogen).

    Because B₀ scales with field_strength_scale, r_mp ∝ (scale)^(1/3).

    SIMPLIFICATION: this gives only ONE number, the distance to the nose of
    the magnetopause.  We do not model the compressed field itself, the
    magnetosheath, or the real shape of the boundary (blunt on the dayside,
    stretched into a long tail on the nightside).

    Parameters
    ----------
    n_cm3 : float
        Solar wind number density [particles per cm³].  Quiet ≈ 5.
    v_kms : float
        Solar wind speed [km/s].  Quiet ≈ 400.
    field_strength_scale : float
        Same meaning as in B_dipole.

    Returns
    -------
    r_mp : float
        Standoff distance [m].  0 if there is no field.
    """
    n = n_cm3 * 1e6          # cm⁻³ → m⁻³
    v = v_kms * 1e3          # km/s → m/s
    rho = n * M_PROTON       # mass density [kg/m³]
    dynamic_pressure = rho * v**2          # [Pa]

    B0 = equatorial_surface_field(field_strength_scale)
    return R_E * ((2 * B0) ** 2 / (2 * MU_0 * dynamic_pressure)) ** (1 / 6)


def open_closed_boundary_latitude(r_mp):
    """
    Footpoint latitude [degrees] separating closed (equatorward) from
    "open" (poleward) field lines, for a spherical boundary of radius r_mp.
    This is just footpoint_latitude(L) with L = r_mp / R_E.
    """
    return footpoint_latitude(r_mp / R_E)


def field_line_for_L(L, phi, r_mp, field_strength_scale=1.0):
    """
    Trace the field line labelled L at longitude phi [rad], cut off at a
    SPHERICAL magnetopause of radius r_mp, and classify it.

    How "open" and "closed" are decided (SIMPLIFIED MODEL):
        A dipole line never goes further from Earth than its equatorial
        crossing L·R_E.  So with a spherical boundary of radius r_mp:
            L·R_E <  r_mp  → the whole line stays inside       → "closed"
            L·R_E >= r_mp  → the line would cross the boundary → "open"
        In a pure dipole every line is really closed.  Real open field lines
        connect to the solar wind's own magnetic field (the IMF), which this
        model does not contain.  "Open" here only means "this line reaches
        the boundary, so our model stops being valid along it."

    The line is traced numerically starting at its northern footpoint.
    For an open line we trace from BOTH footpoints out to the boundary, so it
    is drawn as two separate pieces.

    Returns
    -------
    segments : list of (N, 3) arrays
        One segment (closed line) or two (open line, north and south pieces).
    status : str
        "closed" or "open".
    """
    lam = np.radians(footpoint_latitude(L))
    # Start exactly on the surface.  The "hit surface" event only counts
    # inward crossings, so it does not trigger at the start.
    r0 = R_E

    def footpoint(hemisphere_sign):
        return r0 * np.array([np.cos(lam) * np.cos(phi),
                              np.cos(lam) * np.sin(phi),
                              hemisphere_sign * np.sin(lam)])

    if L * R_E < r_mp:
        points, _ = trace_field_line(footpoint(+1), r_outer=r_mp,
                                     field_strength_scale=field_strength_scale)
        return [points], "closed"

    north, _ = trace_field_line(footpoint(+1), r_outer=r_mp,
                                field_strength_scale=field_strength_scale)
    south, _ = trace_field_line(footpoint(-1), r_outer=r_mp,
                                field_strength_scale=field_strength_scale)
    return [north, south], "open"

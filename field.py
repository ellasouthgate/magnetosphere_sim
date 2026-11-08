import numpy as np
from scipy.integrate import solve_ivp

from constants import MU_0, M_PROTON, R_E, M_DIPOLE, M_DIPOLE_VEC
from config import SUN_DIRECTION
from sources import dipole_field



def B_dipole(r, field_strength_scale=1.0):
    """Earth's untilted dipole (m = −|m| ẑ at the origin): the old model.

    Now a special case of sources.dipole_field; the numbers are bit-for-bit
    the same as before.
    """
    return field_strength_scale * dipole_field(r, M_DIPOLE_VEC)


def equatorial_surface_field(field_strength_scale=1.0):
    return field_strength_scale * (MU_0 / (4 * np.pi)) * M_DIPOLE / R_E**3



def footpoint_latitude(L):
    L = np.asarray(L, dtype=float)
    if np.any(L < 1):
        raise ValueError("L must be >= 1 (an L < 1 field line is inside Earth).")
    return np.degrees(np.arccos(np.sqrt(1.0 / L)))


def L_shell_line(L, phi=0.0, n_points=400):
    lam_foot = np.radians(footpoint_latitude(L))
    theta = np.linspace(np.pi / 2 - lam_foot, np.pi / 2 + lam_foot, n_points)
    r = L * R_E * np.sin(theta) ** 2
    return np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])




def trace_field_line(start, r_outer=None, field_strength_scale=1.0,
                     max_step=0.02 * R_E, max_length=100 * R_E,
                     field_func=None, direction=None):
    """Follow a field line from `start` until it hits Earth or leaves r_outer.

    field_func : function r -> B.  None (default) = the old dipole B_dipole.
    direction  : +1 to go along B, −1 against it.  None (default) = whichever
                 way points away from Earth at the start (the old behaviour).
    """
    if field_func is None:
        def field_func(r):
            return B_dipole(r, field_strength_scale)

    start = np.asarray(start, dtype=float)
    if direction is None:
        r_hat0 = start / np.linalg.norm(start)
        sign = 1.0 if np.dot(field_func(start), r_hat0) > 0 else -1.0
    else:
        sign = float(np.sign(direction))

    def direction_of_B(s, r):
        B = field_func(r)
        return sign * B / np.linalg.norm(B)

    def hit_surface(s, r):
        return np.linalg.norm(r) - R_E
    hit_surface.terminal = True
    hit_surface.direction = -1

    events = [hit_surface]

    if r_outer is not None:
        def hit_outer(s, r):
            return np.linalg.norm(r) - r_outer
        hit_outer.terminal = True
        hit_outer.direction = +1
        events.append(hit_outer)

    sol = solve_ivp(direction_of_B, (0.0, max_length), start, events=events,
                    max_step=max_step, rtol=1e-9, atol=1e-3)

    points = sol.y.T
    if sol.status == 1:
        if len(sol.t_events[0]) > 0:
            end_reason = "surface"
        else:
            end_reason = "outer_boundary"
    else:
        end_reason = "max_length"
    return points, end_reason



def local_time_hours(longitude_deg):
    """Local time from longitude φ (measured from +x towards +y, eastward).

    Noon (12 h) is the longitude facing the Sun (SUN_DIRECTION; the Sun is on
    −x, so noon is φ = 180°).  Local time increases eastward at 15° per hour:
        LT = 12 + (φ − φ_noon)/15   (mod 24)
    """
    lon_noon = np.degrees(np.arctan2(SUN_DIRECTION[1], SUN_DIRECTION[0]))
    return (12 + (np.asarray(longitude_deg) - lon_noon) / 15) % 24


def magnetopause_standoff(n_cm3, v_kms, field_strength_scale=1.0):
    n = n_cm3 * 1e6
    v = v_kms * 1e3
    rho = n * M_PROTON
    dynamic_pressure = rho * v**2

    B0 = equatorial_surface_field(field_strength_scale)
    return R_E * ((2 * B0) ** 2 / (2 * MU_0 * dynamic_pressure)) ** (1 / 6)


def open_closed_boundary_latitude(r_mp):
    return footpoint_latitude(r_mp / R_E)


def field_line_for_L(L, phi, r_mp, field_strength_scale=1.0):
    lam = np.radians(footpoint_latitude(L))
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

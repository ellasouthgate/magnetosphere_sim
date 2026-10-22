import numpy as np
from scipy.integrate import solve_ivp

from constants import MU_0, M_PROTON, R_E, M_DIPOLE, M_DIPOLE_VEC


# ---------------------------------------------------------------------------
# The dipole field
# ---------------------------------------------------------------------------

def B_dipole(r, field_strength_scale=1.0):
    r = np.asarray(r, dtype=float)

    r_mag = np.linalg.norm(r, axis=-1, keepdims=True)
    r_hat = r / r_mag
    m_dot_rhat = np.sum(r_hat * M_DIPOLE_VEC, axis=-1, keepdims=True)

    B = (MU_0 / (4 * np.pi)) * (3 * m_dot_rhat * r_hat - M_DIPOLE_VEC) / r_mag**3
    return field_strength_scale * B


def equatorial_surface_field(field_strength_scale=1.0):
    return field_strength_scale * (MU_0 / (4 * np.pi)) * M_DIPOLE / R_E**3


# ---------------------------------------------------------------------------
# L-shells (analytic)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Numerical field-line tracing
# ---------------------------------------------------------------------------

def trace_field_line(start, r_outer=None, field_strength_scale=1.0,
                     max_step=0.02 * R_E, max_length=100 * R_E):
    start = np.asarray(start, dtype=float)
    r_hat0 = start / np.linalg.norm(start)
    sign = 1.0 if np.dot(B_dipole(start, field_strength_scale), r_hat0) > 0 else -1.0

    def direction(s, r):
        B = B_dipole(r, field_strength_scale)
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

    sol = solve_ivp(direction, (0.0, max_length), start, events=events,
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


# ---------------------------------------------------------------------------
# Magnetopause (simple boundary)
# ---------------------------------------------------------------------------

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

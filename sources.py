import numpy as np

from constants import MU_0, R_E, M_DIPOLE
from config import DEFAULT_FIELD


def dipole_field(r, moment_vec, position=(0.0, 0.0, 0.0)):
    """Field of a point dipole with moment m at `position`.

    B = (μ₀/4π) · [3(m·d̂)d̂ − m] / |d|³
    """
    r = np.asarray(r, dtype=float)
    m = np.asarray(moment_vec, dtype=float)
    d = r - np.asarray(position, dtype=float)

    d_mag = np.linalg.norm(d, axis=-1, keepdims=True)
    d_hat = d / d_mag
    m_dot_dhat = np.sum(d_hat * m, axis=-1, keepdims=True)

    return (MU_0 / (4 * np.pi)) * (3 * m_dot_dhat * d_hat - m) / d_mag**3


def rotation_about_y(angle_deg):
    """Matrix rotating vectors by angle_deg about +y (z turns towards +x)."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def geographic_north(cfg=DEFAULT_FIELD):
    """n_geo = (−sin ψ, 0, cos ψ), the north pole leans towards the Sun."""
    return rotation_about_y(-cfg.tilt_deg) @ np.array([0.0, 0.0, 1.0])


def magnetic_north(cfg=DEFAULT_FIELD):
    """Unit vector along the magnetic axis, rotation axis plus the offset."""
    d = np.radians(cfg.magnetic_offset_deg)
    a = np.radians(cfg.magnetic_offset_azimuth_deg)
    axis = np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])
    return rotation_about_y(-cfg.tilt_deg) @ axis


def earth_moment(cfg=DEFAULT_FIELD):
    """m = −|m| · n_mag, the moment points to geographic south."""
    return -cfg.field_strength_scale * M_DIPOLE * magnetic_north(cfg)


def B_earth(r, cfg=DEFAULT_FIELD):
    """Earth's (possibly tilted) dipole field."""
    return dipole_field(r, earth_moment(cfg))


def earth_equatorial_surface_field(cfg=DEFAULT_FIELD):
    """B₀ = (μ₀/4π)·|m|/R_E³, the dipole field on the magnetic equator."""
    return cfg.field_strength_scale * (MU_0 / (4 * np.pi)) * M_DIPOLE / R_E**3


def sun_position(cfg=DEFAULT_FIELD):
    """The solar dipole sits on the −x axis at distance D: (−D, 0, 0)."""
    return np.array([-cfg.sun_distance_RE * R_E, 0.0, 0.0])


def sun_field_at_earth_magnitude(cfg=DEFAULT_FIELD):
    """|B_sun| at Earth, B₀ / r_balance³."""
    return earth_equatorial_surface_field(cfg) / cfg.r_balance_RE**3


def sun_moment(cfg=DEFAULT_FIELD):
    """Solar moment.  Field at Earth is opposite the moment, so south at Earth is +z."""
    if cfg.sun_field_direction not in ("south", "north"):
        raise ValueError("sun_field_direction must be 'south' or 'north'")
    D = cfg.sun_distance_RE * R_E
    m_mag = sun_field_at_earth_magnitude(cfg) * 4 * np.pi * D**3 / MU_0
    sign = +1.0 if cfg.sun_field_direction == "south" else -1.0
    return np.array([0.0, 0.0, sign * m_mag])


def B_sun(r, cfg=DEFAULT_FIELD):
    """Field of the toy solar dipole (weight not included)."""
    return dipole_field(r, sun_moment(cfg), sun_position(cfg))


def B_total(r, cfg=DEFAULT_FIELD):
    """B_total(r) = B_earth(r) + w_sun · B_sun(r)."""
    B = B_earth(r, cfg)
    if cfg.w_sun != 0:
        B = B + cfg.w_sun * B_sun(r, cfg)
    return B


def field_function(cfg=DEFAULT_FIELD):
    """B_total with cfg fixed, for trace_field_line (faster than calling B_total)."""
    m_e = earth_moment(cfg)
    if cfg.w_sun == 0:
        def B(r):
            return dipole_field(r, m_e)
        return B
    m_s, p_s, w = sun_moment(cfg), sun_position(cfg), cfg.w_sun

    def B(r):
        return dipole_field(r, m_e) + w * dipole_field(r, m_s, p_s)
    return B


def effective_r_balance(cfg=DEFAULT_FIELD):
    """Balance distance (R_E) including the weight: r_balance / w_sun^(1/3)."""
    if cfg.w_sun == 0:
        return np.inf
    return cfg.r_balance_RE / abs(cfg.w_sun) ** (1 / 3)


def field_on_xz_grid(cfg=DEFAULT_FIELD, half_width_RE=20.0, n=401):
    """B_total on a grid in the x–z plane.  Returns x, z (R_E) and B (nT), NaN inside Earth."""
    g = np.linspace(-half_width_RE, half_width_RE, n)
    X, Z = np.meshgrid(g, g)
    pts = np.stack([X, np.zeros_like(X), Z], axis=-1) * R_E
    with np.errstate(divide="ignore", invalid="ignore"):   # r = 0 if on the grid
        B = B_total(pts.reshape(-1, 3), cfg).reshape(n, n, 3) * 1e9
    B[np.hypot(X, Z) < 1.0] = np.nan
    return g, g, B

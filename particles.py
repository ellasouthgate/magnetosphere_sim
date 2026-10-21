"""
Particle species and initial conditions for the test-particle runs.

Everything is NON-RELATIVISTIC for now: kinetic energy K = ½·m·v².
A warning is printed when the speed exceeds 0.1c, because then the
relativistic correction (γ − 1 ≈ ½(v/c)² ≈ 0.5% at 0.1c) starts to matter.

Contents:
    Species / PROTON / ELECTRON   mass and charge of each particle type
    speed_from_energy             v from kinetic energy in eV
    gyroperiod                    T_c = 2π·m / (|q|·B)
    larmor_radius                 r_L = m·v⊥ / (|q|·B)
    magnetic_moment               μ = m·v⊥² / (2B)
    loss_cone_angle               smallest equatorial pitch angle that stays trapped
    mirror_field                  field strength where the particle mirrors
    bounce_period_estimate        textbook bounce period, for checking runs
    drift_period_estimate         textbook drift period, for checking runs
    initial_conditions            position and velocity for a launch on the equator
"""

from dataclasses import dataclass

import numpy as np

from constants import E_CHARGE, M_PROTON, M_ELECTRON, C_LIGHT, R_E
from field import B_dipole, equatorial_surface_field, footpoint_latitude


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Species:
    name: str
    mass: float      # [kg]
    charge: float    # [C], with sign


PROTON = Species("proton", M_PROTON, +E_CHARGE)
ELECTRON = Species("electron", M_ELECTRON, -E_CHARGE)


# ---------------------------------------------------------------------------
# Basic gyro-motion quantities
# ---------------------------------------------------------------------------

def speed_from_energy(energy_eV, species):
    """
    Non-relativistic speed [m/s] for kinetic energy K given in eV:
        v = √(2K / m)
    """
    return np.sqrt(2 * energy_eV * E_CHARGE / species.mass)


def gyroperiod(B_mag, species):
    """Gyroperiod T_c = 2π·m / (|q|·B)  [s]."""
    return 2 * np.pi * species.mass / (abs(species.charge) * B_mag)


def larmor_radius(v_perp, B_mag, species):
    """Larmor (gyro) radius r_L = m·v⊥ / (|q|·B)  [m]."""
    return species.mass * v_perp / (abs(species.charge) * B_mag)


def magnetic_moment(v_perp, B_mag, species):
    """
    Magnetic moment μ = m·v⊥² / (2B)  [J/T = A·m²].

    This is the first adiabatic invariant: it stays (nearly) constant as long
    as B changes slowly over one gyration.  Because μ is constant and energy
    is constant, v⊥² must grow as the particle moves into stronger field
    near the poles, until v∥ = 0 and the particle is reflected (mirrors).
    """
    return species.mass * v_perp**2 / (2 * B_mag)


# ---------------------------------------------------------------------------
# Mirror points and the loss cone
# ---------------------------------------------------------------------------

def footpoint_field(L, field_strength_scale=1.0):
    """|B| [T] on Earth's surface where the L-shell meets it."""
    lam = np.radians(footpoint_latitude(L))
    foot = R_E * np.array([np.cos(lam), 0.0, np.sin(lam)])
    return np.linalg.norm(B_dipole(foot, field_strength_scale))


def mirror_field(B_eq, pitch_angle_deg):
    """
    Field strength [T] where a particle with equatorial pitch angle α mirrors.

    Energy and μ are both conserved, so v⊥²/B is constant and v⊥ = v at the
    mirror point:
        sin²α / B_eq = 1 / B_mirror   →   B_mirror = B_eq / sin²α
    """
    return B_eq / np.sin(np.radians(pitch_angle_deg)) ** 2


def loss_cone_angle(L, field_strength_scale=1.0):
    """
    Equatorial loss-cone angle α_lc [degrees] for the L-shell.

    A particle whose mirror field would be stronger than the field at
    Earth's surface reaches the atmosphere before it can mirror, and is lost.
    That is where the aurora comes from.  Setting B_mirror = B_foot:
        sin²α_lc = B_eq / B_foot
    For a dipole this works out as sin²α_lc = 1 / (L³·√(4 − 3/L)).
    """
    B_eq = np.linalg.norm(B_dipole([L * R_E, 0.0, 0.0], field_strength_scale))
    B_foot = footpoint_field(L, field_strength_scale)
    return np.degrees(np.arcsin(np.sqrt(B_eq / B_foot)))


# ---------------------------------------------------------------------------
# Textbook estimates of the bounce and drift periods (for checking runs)
# ---------------------------------------------------------------------------

def bounce_period_estimate(L, speed, pitch_angle_deg):
    """
    Approximate dipole bounce period [s] (Baumjohann & Treumann, Basic Space
    Plasma Physics, section 2.6):
        τ_b ≈ (4·L·R_E / v) · (1.30 − 0.56·sin α_eq)
    """
    return 4 * L * R_E / speed * (1.30 - 0.56 * np.sin(np.radians(pitch_angle_deg)))


def drift_period_estimate(L, energy_eV, pitch_angle_deg, species,
                          field_strength_scale=1.0):
    """
    Approximate gradient + curvature drift period around Earth [s]
    (Baumjohann & Treumann, section 3.3):
        τ_d ≈ π·|q|·B₀·R_E² / (3·L·K) · 1 / (0.35 + 0.15·sin α_eq)
    with K the kinetic energy in joules and B₀ the equatorial surface field.
    """
    B0 = equatorial_surface_field(field_strength_scale)
    K = energy_eV * E_CHARGE
    sin_a = np.sin(np.radians(pitch_angle_deg))
    return (np.pi * abs(species.charge) * B0 * R_E**2 / (3 * L * K)
            / (0.35 + 0.15 * sin_a))


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def initial_conditions(species, energy_eV, L, pitch_angle_deg,
                       longitude_deg=0.0, field_strength_scale=1.0):
    """
    Position and velocity for a particle launched on the magnetic equator.

    Parameters
    ----------
    species : Species
    energy_eV : float
        Kinetic energy in eV (e.g. 10e6 for 10 MeV).
    L : float
        Launch distance from Earth's centre, in R_E, on the equator.
    pitch_angle_deg : float
        Angle α between the velocity and the magnetic field.
        v∥ = v·cos α,  v⊥ = v·sin α.
    longitude_deg : float
        Angle round from the x axis (0 = noon, towards the Sun).

    How the velocity is split:
        v∥ points along B̂ (northwards at the equator).
        v⊥ points radially OUTWARDS.  With this choice the centre of the
        gyration circle (the "guiding centre") sits at the same distance from
        Earth as the particle, just shifted sideways by r_L in longitude.
        So the guiding centre really is on the L-shell we asked for.
        (If v⊥ pointed sideways instead, the guiding centre would be r_L
        closer to or further from Earth: 0.1 R_E for the 10 MeV proton.)

    Returns
    -------
    r0, v0 : arrays of shape (3,)
        Position [m] and velocity [m/s].
    info : dict
        Speed, |B|, Larmor radius, gyroperiod, etc. at the launch point, for
        printing and comparing with hand calculations.
    """
    phi = np.radians(longitude_deg)
    r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
    r0 = L * R_E * r_hat

    B0 = B_dipole(r0, field_strength_scale)
    B_mag = np.linalg.norm(B0)
    b_hat = B0 / B_mag

    # Unit vector perpendicular to B, as close to radially outwards as
    # possible.  (On the equator r̂ is already perpendicular to B.)
    e_perp = r_hat - np.dot(r_hat, b_hat) * b_hat
    e_perp /= np.linalg.norm(e_perp)

    v = speed_from_energy(energy_eV, species)
    alpha = np.radians(pitch_angle_deg)
    v_par = v * np.cos(alpha)
    v_perp = v * np.sin(alpha)
    v0 = v_par * b_hat + v_perp * e_perp

    info = {
        "species": species.name,
        "energy_eV": energy_eV,
        "L": L,
        "pitch_angle_deg": pitch_angle_deg,
        "speed": v,
        "v_over_c": v / C_LIGHT,
        "v_par": v_par,
        "v_perp": v_perp,
        "B_mag": B_mag,
        "larmor_radius": larmor_radius(v_perp, B_mag, species),
        "gyroperiod": gyroperiod(B_mag, species),
        "magnetic_moment": magnetic_moment(v_perp, B_mag, species),
        "loss_cone_deg": loss_cone_angle(L, field_strength_scale),
        "B_mirror": min(mirror_field(B_mag, pitch_angle_deg),
                        footpoint_field(L, field_strength_scale)),
    }
    return r0, v0, info


def check_speed(v, threshold=0.1):
    """Print a warning if v > threshold·c (non-relativistic model becomes inaccurate)."""
    if v > threshold * C_LIGHT:
        gamma = 1 / np.sqrt(1 - (v / C_LIGHT) ** 2)
        print(f"  WARNING: v = {v / C_LIGHT:.3f} c exceeds {threshold:g} c.  "
              f"The motion is mildly relativistic (γ ≈ {gamma:.4f}); the "
              "non-relativistic pusher will be slightly off.")
        return False
    return True

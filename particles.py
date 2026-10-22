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
    mass: float
    charge: float


PROTON = Species("proton", M_PROTON, +E_CHARGE)
ELECTRON = Species("electron", M_ELECTRON, -E_CHARGE)


# ---------------------------------------------------------------------------
# Basic gyro-motion quantities
# ---------------------------------------------------------------------------

def speed_from_energy(energy_eV, species):
    return np.sqrt(2 * energy_eV * E_CHARGE / species.mass)


def gyroperiod(B_mag, species):
    return 2 * np.pi * species.mass / (abs(species.charge) * B_mag)


def larmor_radius(v_perp, B_mag, species):
    return species.mass * v_perp / (abs(species.charge) * B_mag)


def magnetic_moment(v_perp, B_mag, species):
    return species.mass * v_perp**2 / (2 * B_mag)


# ---------------------------------------------------------------------------
# Mirror points and the loss cone
# ---------------------------------------------------------------------------

def footpoint_field(L, field_strength_scale=1.0):
    lam = np.radians(footpoint_latitude(L))
    foot = R_E * np.array([np.cos(lam), 0.0, np.sin(lam)])
    return np.linalg.norm(B_dipole(foot, field_strength_scale))


def mirror_field(B_eq, pitch_angle_deg):
    return B_eq / np.sin(np.radians(pitch_angle_deg)) ** 2


def loss_cone_angle(L, field_strength_scale=1.0):
    B_eq = np.linalg.norm(B_dipole([L * R_E, 0.0, 0.0], field_strength_scale))
    B_foot = footpoint_field(L, field_strength_scale)
    return np.degrees(np.arcsin(np.sqrt(B_eq / B_foot)))


# ---------------------------------------------------------------------------
# Textbook estimates of the bounce and drift periods (for checking runs)
# ---------------------------------------------------------------------------

def bounce_period_estimate(L, speed, pitch_angle_deg):
    return 4 * L * R_E / speed * (1.30 - 0.56 * np.sin(np.radians(pitch_angle_deg)))


def drift_period_estimate(L, energy_eV, pitch_angle_deg, species,
                          field_strength_scale=1.0):
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
    phi = np.radians(longitude_deg)
    r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
    r0 = L * R_E * r_hat

    B0 = B_dipole(r0, field_strength_scale)
    B_mag = np.linalg.norm(B0)
    b_hat = B0 / B_mag

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
    if v > threshold * C_LIGHT:
        gamma = 1 / np.sqrt(1 - (v / C_LIGHT) ** 2)
        print(f"  WARNING: v = {v / C_LIGHT:.3f} c exceeds {threshold:g} c.  "
              f"The motion is mildly relativistic (γ ≈ {gamma:.4f}); the "
              "non-relativistic pusher will be slightly off.")
        return False
    return True

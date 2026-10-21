"""
Boris integrator for a charged particle in the dipole magnetosphere.

Equation of motion (Lorentz force, no electric field, non-relativistic):

    m · dv/dt = q · (v × B(r))
        dr/dt = v

WHY THE BORIS ALGORITHM (and not Euler):
    A magnetic force is always perpendicular to v, so it can only turn the
    velocity, never change its size: kinetic energy is exactly constant.
    Plain (forward) Euler does  v_new = v + (q/m)(v × B)·dt.  The added
    piece is perpendicular to v, so |v_new|² = |v|² + (something)² > |v|²:
    the speed grows on EVERY step and the particle spirals outwards.
    The Boris step instead ROTATES v about B by the right angle, so |v| is
    unchanged to rounding error, however long we run.  It is also
    time-reversible, which keeps errors in the orbit (the phase, and the
    magnetic moment μ) bounded instead of slowly growing.
    It is the standard method in plasma simulation for this reason.

One Boris step (E = 0), with time step Δt:
    t  = (q·B / m) · Δt/2            (B evaluated at the current position)
    s  = 2t / (1 + |t|²)
    v' = v + v × t
    v⁺ = v + v' × s                  (v⁺ is v rotated about B)
    r_new = r + v⁺ · Δt

Leapfrog timing: positions live at whole steps (t = n·Δt) and velocities at
half steps (t = (n + ½)·Δt).  To set this up we rotate the starting
velocity BACKWARDS by half a step before the loop.

Choice of time step:
    The particle must be followed through every gyration, so Δt must be a
    small fraction of the gyroperiod T_c = 2π·m / (|q|·B).  T_c is shortest
    where B is strongest, i.e. at the mirror points, so we use
        Δt = dt_fraction · T_c(B_mirror)
    and keep Δt FIXED for the whole run (Boris keeps its good long-run
    behaviour best with a fixed step).
"""

import time as _time

import numpy as np

from constants import R_E
from field import B_dipole
from particles import gyroperiod


def boris_rotate(v, B, q_over_m, dt):
    """
    Rotate velocity v about the magnetic field B for a time dt.
    This is the core of the Boris step (see the module docstring).
    The size |v| is unchanged apart from rounding error.
    """
    t = q_over_m * B * (dt / 2)
    s = 2 * t / (1 + np.dot(t, t))
    v_prime = v + np.cross(v, t)
    return v + np.cross(v_prime, s)


def run_particle(r0, v0, species, t_max, B_ref, dt_fraction=1 / 30,
                 field_strength_scale=1.0, r_outer=None, save_every=1,
                 progress=True):
    """
    Follow one particle with the Boris algorithm.

    Parameters
    ----------
    r0, v0 : arrays, shape (3,)
        Starting position [m] and velocity [m/s].
    species : particles.Species
    t_max : float
        How long to run [s].
    B_ref : float
        The field strength [T] used to choose the time step.  Pass the
        STRONGEST field the particle will see (its mirror-point field), so
        the gyration is resolved everywhere along the orbit.
    dt_fraction : float
        Δt = dt_fraction · T_c(B_ref).  Default 1/30, i.e. 30 steps per
        gyration at the mirror points (more near the equator).
    field_strength_scale : float
        Passed to B_dipole.
    r_outer : float or None
        If given, stop when the particle gets further than this [m] from
        Earth's centre (e.g. the magnetopause).
    save_every : int
        Only store every N-th step.  Use > 1 for long electron runs, which
        can take millions of steps.
    progress : bool
        Print a progress line roughly every 10% of the run.

    Returns
    -------
    result : dict with keys
        "t"       times of the saved points [s], shape (N,)
        "r"       positions [m], shape (N, 3)
        "v"       velocities [m/s], shape (N, 3), at the half step after "t"
        "B"       field at each saved position [T], shape (N, 3)
        "dt"      the time step used [s]
        "n_steps" number of steps actually taken
        "end_reason"  "t_max", "hit_earth" or "left_boundary"
        "hit"     None, or a dict with the impact point, its latitude,
                  longitude, magnetic local time and time of impact
        "runtime" wall-clock time of the loop [s]
    """
    q_over_m = species.charge / species.mass
    dt = dt_fraction * gyroperiod(B_ref, species)
    n_steps_max = int(np.ceil(t_max / dt))

    r = np.array(r0, dtype=float)
    # Set up the leapfrog: move v back by half a step, so it represents the
    # velocity at t = −Δt/2.
    v = boris_rotate(np.array(v0, dtype=float), B_dipole(r, field_strength_scale),
                     q_over_m, -dt / 2)

    n_saved_max = n_steps_max // save_every + 2
    t_out = np.empty(n_saved_max)
    r_out = np.empty((n_saved_max, 3))
    v_out = np.empty((n_saved_max, 3))
    B_out = np.empty((n_saved_max, 3))
    n_saved = 0

    end_reason = "t_max"
    hit = None
    report_every = max(n_steps_max // 10, 1)
    wall_start = _time.perf_counter()

    step = 0
    for step in range(n_steps_max):
        B = B_dipole(r, field_strength_scale)
        v = boris_rotate(v, B, q_over_m, dt)

        if step % save_every == 0:
            t_out[n_saved] = step * dt
            r_out[n_saved] = r
            v_out[n_saved] = v
            B_out[n_saved] = B
            n_saved += 1

        r_old = r
        r = r + v * dt

        r_mag = np.linalg.norm(r)
        if r_mag < R_E:
            end_reason = "hit_earth"
            hit = _impact_point(r_old, r, step * dt, dt)
            break
        if r_outer is not None and r_mag > r_outer:
            end_reason = "left_boundary"
            break

        if progress and step > 0 and step % report_every == 0:
            print(f"    {100 * step / n_steps_max:3.0f}% "
                  f"({step:,} steps, {_time.perf_counter() - wall_start:.1f} s)")

    # Always store the final position, so plots run right up to the end.
    t_out[n_saved] = (step + 1) * dt if hit is None else hit["time"]
    r_out[n_saved] = r if hit is None else hit["position"]
    v_out[n_saved] = v
    B_out[n_saved] = B_dipole(r_out[n_saved], field_strength_scale)
    n_saved += 1

    return {
        "t": t_out[:n_saved],
        "r": r_out[:n_saved],
        "v": v_out[:n_saved],
        "B": B_out[:n_saved],
        "dt": dt,
        "n_steps": step + 1,
        "end_reason": end_reason,
        "hit": hit,
        "runtime": _time.perf_counter() - wall_start,
    }


def measured_bounce_period(result):
    """
    Bounce period [s] measured from a run that started on the equator.

    The particle crosses the equator (z = 0) every HALF bounce, so
        bounce period = 2 × (average time between equator crossings).
    The start (t = 0, on the equator) counts as the first crossing.
    Returns None if the run never came back to the equator.
    """
    t, z = result["t"], result["r"][:, 2]
    crossings = np.where(np.sign(z[:-1]) != np.sign(z[1:]))[0]
    crossing_times = np.concatenate([[0.0], t[crossings[t[crossings] > 0]]])
    if len(crossing_times) < 2:
        return None
    return 2 * np.mean(np.diff(crossing_times))


def measured_drift_period(result):
    """
    Drift period [s] around Earth, from the average rate of change of
    longitude (a straight-line fit of longitude against time).

    The drift speed changes with latitude, so this is only meaningful when
    the run covers at least one full bounce; otherwise returns (None, None).

    Returns (period, direction), direction being "eastward" or "westward".
    """
    tb = measured_bounce_period(result)
    t, r = result["t"], result["r"]
    if tb is None or t[-1] < tb:
        return None, None
    lon = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    rate = np.polyfit(t, lon, 1)[0]                  # rad/s
    direction = "eastward" if rate > 0 else "westward"
    return 2 * np.pi / abs(rate), direction


def _impact_point(r_outside, r_inside, t_outside, dt):
    """
    The last step went from outside Earth (r_outside, at time t_outside) to
    inside (r_inside, a time dt later).
    Find where that straight step crosses |r| = R_E, and convert to
    latitude, longitude and magnetic local time (MLT).

    Solve |r_outside + f·(r_inside − r_outside)| = R_E for f in [0, 1]
    (a quadratic in f; we want the first crossing, the smaller root).

    NOTE on longitude: our x axis points at the Sun and does not rotate with
    Earth, so this is the longitude measured from the NOON meridian, not a
    geographic longitude.  Space physicists express it as magnetic local
    time:  MLT = 12 h + longitude / 15°  (noon = 12, midnight = 0 / 24).
    With no dipole tilt, magnetic and geographic latitude are the same.
    """
    d = r_inside - r_outside
    a = np.dot(d, d)
    b = 2 * np.dot(r_outside, d)
    c = np.dot(r_outside, r_outside) - R_E**2
    f = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    position = r_outside + f * d

    lat = np.degrees(np.arcsin(position[2] / R_E))
    lon = np.degrees(np.arctan2(position[1], position[0]))
    mlt = (12 + lon / 15) % 24
    return {"position": position, "latitude_deg": lat, "longitude_deg": lon,
            "mlt_hours": mlt, "time": t_outside + f * dt}

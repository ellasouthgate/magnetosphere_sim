import time as _time

import numpy as np

from constants import R_E
from field import B_dipole
from particles import gyroperiod


def boris_rotate(v, B, q_over_m, dt):
    t = q_over_m * B * (dt / 2)
    s = 2 * t / (1 + np.dot(t, t))
    v_prime = v + np.cross(v, t)
    return v + np.cross(v_prime, s)


def run_particle(r0, v0, species, t_max, B_ref, dt_fraction=1 / 30,
                 field_strength_scale=1.0, r_outer=None, save_every=1,
                 progress=True):
    q_over_m = species.charge / species.mass
    dt = dt_fraction * gyroperiod(B_ref, species)
    n_steps_max = int(np.ceil(t_max / dt))

    r = np.array(r0, dtype=float)
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
    t, z = result["t"], result["r"][:, 2]
    crossings = np.where(np.sign(z[:-1]) != np.sign(z[1:]))[0]
    crossing_times = np.concatenate([[0.0], t[crossings[t[crossings] > 0]]])
    if len(crossing_times) < 2:
        return None
    return 2 * np.mean(np.diff(crossing_times))


def measured_drift_period(result):
    tb = measured_bounce_period(result)
    t, r = result["t"], result["r"]
    if tb is None or t[-1] < tb:
        return None, None
    lon = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    rate = np.polyfit(t, lon, 1)[0]
    direction = "eastward" if rate > 0 else "westward"
    return 2 * np.pi / abs(rate), direction


def _impact_point(r_outside, r_inside, t_outside, dt):
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

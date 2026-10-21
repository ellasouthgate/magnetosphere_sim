# magnetosphere_sim

A simple model of Earth's magnetosphere for our aurora group project.

- **Stage 1:** Earth's magnetic field as a dipole, field lines labelled by L-shell, and a simple magnetopause boundary set by the solar wind.
- **Stage 2:** single charged particles (protons and electrons) moving through that field, pushed with the Boris algorithm.

Later we will use the particle code to trace auroral electrons down to the atmosphere.

---

## Quick start

You need Python 3 with numpy, matplotlib, scipy and pytest:

```
py -m pip install -r requirements.txt
```

(On Mac/Linux use `python3` instead of `py`.)

Run the examples:

```
py run_dipole.py --stage 1     # the magnetosphere: checks, L-shell table, figures 1–4 (a few seconds)
py run_dipole.py --stage 2     # the particles: figures 5–12 (about 4–8 minutes, mostly the electron)
py run_dipole.py               # both
py run_dipole.py --show        # also open the figures in windows (you can rotate the 3D ones)
```

Figures are saved in `figures/`. Run the validation tests (about 20 seconds):

```
py -m pytest
```

---

## What each file does

| File | Contents |
|---|---|
| `constants.py` | All physical constants (μ₀, e, m_p, m_e, c, R_E, Earth's dipole moment), with sources. **Change constants here only.** |
| `field.py` | The dipole field `B_dipole`, L-shell footpoint latitudes, numerical field-line tracing, the magnetopause standoff distance, and open/closed field-line classification. |
| `particles.py` | Particle species (proton, electron), launch conditions, and the standard formulas: speed, gyroperiod, Larmor radius, magnetic moment, mirror field, loss cone, textbook bounce and drift periods. |
| `pusher.py` | The Boris integrator (`run_particle`), detection of where a particle hits Earth, and functions that measure the bounce and drift periods from a run. |
| `plotting.py` | All the figures. Every function takes SI inputs and plots positions in Earth radii. |
| `run_dipole.py` | Runs the Stage 1 and Stage 2 examples, prints the checks, and saves the figures. The settings (solar wind, time step fraction, etc.) are constants at the top of each section. |
| `tests/` | The validation checks as pytest tests: `test_field.py` (Stage 1) and `test_particles.py` (Stage 2). |

---

## Coordinates and units

- Origin at Earth's centre. **z** points along the dipole axis (geographic north, since there is no dipole tilt yet). **x** points towards the Sun, so the x–z plane is the noon–midnight meridian and x > 0 is the dayside. **y** completes the right-handed set and points towards dusk.
- Everything inside the code is in SI units (m, s, T, kg, C, J). Positions are only converted to Earth radii (R_E = 6.371 × 10⁶ m) for plotting and printing. Inputs that are not SI say so in their name: `energy_eV`, `n_cm3`, `v_kms`.
- "Longitude" is measured from the noon meridian, positive eastward. It is **not** a geographic longitude, because our axes stay pointed at the Sun while Earth rotates underneath. Space physicists express it as magnetic local time:

  MLT = 12 h + longitude / 15°  (noon = 12 h, dusk = 18 h, midnight = 0 h)

---

## The physics

### 1. The dipole field

Earth's field is treated as the field of a magnetic dipole at its centre:

```
B(r) = (μ₀/4π) · [3(m·r̂)r̂ − m] / |r|³
```

- m = −|m| ẑ with |m| = 8.0 × 10²² A·m². The minus sign is because Earth's magnetic moment points towards geographic **south**. Field lines leave the southern hemisphere and enter the northern hemisphere.
- In terms of magnetic latitude λ, the strength is |B| = B₀ · (R_E/r)³ · √(1 + 3 sin²λ), where B₀ = (μ₀/4π) · |m| / R_E³ ≈ 3.1 × 10⁻⁵ T is the field at the equator on Earth's surface. At the poles it is 2B₀.
- Every function takes a `field_strength_scale` (default 1.0) that multiplies the whole field. It is there so we can later model weaker, stronger, or no-field planets.

### 2. L-shells

A dipole field line is the curve

```
r = L · R_E · sin²θ       (θ = colatitude, measured from the z axis)
```

L is the distance in Earth radii where the line crosses the equator. Setting r = R_E gives the latitude λ where the line meets the ground:

```
cos²λ = 1/L
```

| L | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| λ (°) | 45.0 | 54.7 | 60.0 | 63.4 | 65.9 | 67.8 | 69.3 | 70.5 | 71.6 |

The auroral zone (about 65–70° magnetic latitude) maps to L ≈ 6–8.

`field.py` also traces field lines numerically by integrating dr/ds = B/|B| (s = distance along the line). The tests check that the traced lines agree with r = L · R_E · sin²θ to better than one part in a million.

### 3. The magnetopause (simplified)

The solar wind pushes on the magnetosphere. At the nose of the boundary (the point facing the Sun), the solar wind's dynamic pressure ρv² balances the magnetic pressure B²/(2μ₀) just inside the boundary. There the field is compressed to roughly **twice** the dipole value. This gives the standoff distance:

```
r_mp = R_E · [ (2B₀)² / (2μ₀ · ρ · v²) ]^(1/6)       with ρ = n · m_p
```

| Solar wind | n (cm⁻³) | v (km/s) | ρv² (nPa) | r_mp (R_E) |
|---|---|---|---|---|
| quiet | 5 | 400 | 1.3 | 10.2 |
| storm | 20 | 700 | 16.4 | 6.7 |

Because of the 1/6 power, the boundary moves only slowly with pressure: 12× more pressure moves it in by about a third.

**Simplifications:**
- The boundary is drawn as a sphere of radius r_mp. It is solid on the dayside and dotted on the nightside, where the real magnetosphere is stretched into a long tail.
- The field inside is still a pure dipole. We do not model the compressed field itself.
- "Open" field lines are the ones that would cross the sphere, i.e. L ≥ r_mp / R_E. Their footpoints are poleward of about 72° for quiet conditions. In reality open field lines connect to the Sun's magnetic field, which is not in this model. Here "open" only means "our model stops being valid along this line".

### 4. Charged particle motion

A particle of charge q and mass m feels the Lorentz force. There is no electric field yet:

```
m · dv/dt = q · (v × B)
```

The force is always perpendicular to v, so the speed and kinetic energy never change. The particle is non-relativistic for now: K = ½mv². The code prints a warning if v > 0.1c.

**Three motions, on very different time scales:**

1. **Gyration** around the field line.
   - Gyroperiod: T_c = 2πm / (|q|B)
   - Larmor radius: r_L = m·v⊥ / (|q|B)
   - Here v⊥ and v∥ are the parts of the velocity perpendicular and parallel to B, and the pitch angle α is the angle between v and B.
2. **Bouncing** between mirror points in the north and south.
   - The magnetic moment μ = m·v⊥² / (2B) is (nearly) conserved when B changes slowly over one gyration.
   - Energy is also conserved, so as the particle moves into the stronger field near the poles v⊥ must grow and v∥ must shrink. When v∥ reaches 0 the particle turns round. This happens where B = B_eq / sin²α_eq.
   - Bounce period estimate: τ_b ≈ (4·L·R_E / v) · (1.30 − 0.56·sin α_eq)
3. **Drift** around Earth, caused by the gradient and curvature of the field.
   - Protons drift west and electrons drift east.
   - Drift period estimate: τ_d ≈ π·|q|·B₀·R_E² / (3·L·K) · 1/(0.35 + 0.15·sin α_eq)

For a 10 MeV proton at L = 4 these are about 0.14 s, 2 s and 72 s. For a 1 keV electron they are about 7 × 10⁻⁵ s, 5 s and about 10 days.

**Loss cone and the aurora.** A particle whose mirror point would be below Earth's surface hits the atmosphere instead. That is how auroral particles arrive. This happens when the equatorial pitch angle is smaller than the loss-cone angle:

```
sin²α_lc = B_eq / B_surface = 1 / (L³ · √(4 − 3/L))
```

At L = 4 the loss cone is only 5.3°, so most particles stay trapped. The code stops a run when |r| < R_E and records the impact latitude, longitude and MLT.

### 5. The Boris algorithm

`pusher.py` advances the particle with the Boris algorithm, which is the standard method in plasma simulation. We use it instead of the simple Euler method because:

- **Euler gains energy on every step.** It does v_new = v + (q/m)(v × B)·Δt. The added piece is perpendicular to v, so |v_new| > |v| on every single step, and the particle spirals outward.
- **Boris never changes the speed.** It rotates v about B, so |v| is unchanged apart from rounding error, however long the run. It is also time-reversible, which keeps other errors (gyration phase, μ) bounded instead of growing.

One step, with time step Δt:

```
t  = (q B / m) · Δt/2
s  = 2t / (1 + |t|²)
v' = v + v × t
v⁺ = v + v' × s          (v rotated about B)
r_new = r + v⁺ · Δt
```

Positions are stored at whole steps and velocities at half steps ("leapfrog"). Before the first step, the starting velocity is rotated back by half a step.

**Time step.** Δt = `dt_fraction` × T_c, with T_c calculated at the particle's mirror point, where B is strongest and T_c shortest. The default `dt_fraction` is 1/30. Δt is kept fixed for the whole run.

---

## Validation

`run_dipole.py` prints these checks, and `py -m pytest` runs them as 31 tests.

**Stage 1**

| Check | Result |
|---|---|
| \|B\| at the equatorial surface | 3.094 × 10⁻⁵ T |
| \|B\| at the poles / at the equator | exactly 2 |
| Traced field lines vs r = L · R_E · sin²θ | agree to 6 × 10⁻¹² |
| Traced footpoints vs cos²λ = 1/L | agree to 0.001° |
| Standoff distance, quiet | 10.2 R_E |
| Standoff distance, storm | 6.7 R_E (shrinks, as it should) |

**Stage 2**

| Check | Result |
|---|---|
| Kinetic energy | constant to about 10⁻¹⁴ (rounding error) |
| Uniform field: orbit radius and period | equal r_L and T_c |
| Bounce period vs textbook estimate | within 2% |
| Proton drift period vs textbook estimate | within 0.1%, westward |
| Mirror latitude vs μ conservation | within 0.5° |
| Proton inside the loss cone | hits Earth at 60.00°, the L = 4 footpoint |

---

## Known limitations and things to watch out for

- **Non-relativistic.** A 10 MeV proton already moves at 0.15c, about 1% off in speed. Radiation-belt electrons (100 keV and above) are strongly relativistic and would need the relativistic version of Boris, which is a small change.
- **High-energy protons are only marginally "adiabatic".** For a 10 MeV proton at L = 4, r_L is about 10% of the field line's radius of curvature at the equator (L·R_E/3). So μ is not well conserved and the pitch angle gets scattered at every equator crossing ("field-line curvature scattering"). This is real physics, not a bug. It is why the loss-cone example uses a 1 MeV proton: a 10 MeV proton launched in the loss cone gets kicked out of it and never reaches Earth.
- **Electrons are slow to simulate.** Each Boris step takes about 100 µs in plain Python. A 1 keV electron at L = 4 needs about 4 million steps per bounce, which is several minutes. So the example electron only runs for just over half a bounce (`ELECTRON_BOUNCE_FRACTION` in `run_dipole.py`). Options for speeding up, simplest first:
  1. Write the 3-vector maths in the loop with plain Python floats instead of small numpy arrays. This should be roughly 5–10× faster and still readable.
  2. Compile the loop with numba. About 100× faster, but it adds a dependency.
  3. Push many particles at once with array operations. This is the natural choice for the aurora runs, which need many electrons.
- No dipole tilt, no Earth rotation, no electric fields, no compressed or tail field, no ring current.
- `field_strength_scale = 0` (no field) works in `B_dipole` and `magnetopause_standoff`. The particle code cannot use it yet, because the time step is set from the gyroperiod, which is infinite when B = 0.

---

## References

- Baumjohann, W. & Treumann, R. A., *Basic Space Plasma Physics* (Imperial College Press). Dipole field, adiabatic invariants, bounce and drift periods, magnetopause standoff.
- Kivelson, M. G. & Russell, C. T., *Introduction to Space Physics* (Cambridge). General background.
- Birdsall, C. K. & Langdon, A. B., *Plasma Physics via Computer Simulation*. The Boris algorithm.
- Constants: CODATA 2018 recommended values. Earth's dipole moment: IGRF-13.

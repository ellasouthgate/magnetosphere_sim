"""
Physical constants used throughout the project, all in SI units.

Keep every constant here so the whole team uses the same values.
Sources are given next to each value.
"""

import numpy as np

# --- Fundamental constants -------------------------------------------------

# Vacuum permeability μ₀ [T·m/A].  CODATA 2018 recommended value.
# (Since the 2019 SI redefinition μ₀ is measured, not exactly 4π × 10⁻⁷,
#  but the two agree to ~1 part in 10¹⁰.)
MU_0 = 1.25663706212e-6

# Elementary charge e [C].  Exact by definition in the 2019 SI.
E_CHARGE = 1.602176634e-19

# Proton mass m_p [kg].  CODATA 2018.
M_PROTON = 1.67262192369e-27

# Electron mass m_e [kg].  CODATA 2018.
M_ELECTRON = 9.1093837015e-31

# Speed of light c [m/s].  Exact by definition.
C_LIGHT = 299792458.0

# --- Earth -----------------------------------------------------------------

# Mean Earth radius R_E [m].  IUGG mean radius (6371.0 km).
R_E = 6.371e6

# Magnitude of Earth's magnetic dipole moment |m| [A·m²].
# The IGRF-13 model gives about 7.7e22 A·m² for 2020; 8.0e22 is the
# round value used in most textbooks and in our project spec.
M_DIPOLE = 8.0e22

# Dipole moment vector [A·m²].  It points towards geographic SOUTH
# (m = −|m| ẑ), which is why field lines leave the southern hemisphere
# and enter the northern hemisphere.  No dipole tilt for now.
M_DIPOLE_VEC = np.array([0.0, 0.0, -M_DIPOLE])

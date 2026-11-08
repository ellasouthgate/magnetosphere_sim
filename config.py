"""All the model parameters.

Sun on −x (left of plots), z = north at zero tilt, y = z × x so +y is dawn.
"""

from dataclasses import dataclass, replace

import numpy as np

# unit vector towards the Sun
SUN_DIRECTION = np.array([-1.0, 0.0, 0.0])


@dataclass(frozen=True)
class FieldConfig:
    """Parameters of the combined field  B_total = B_earth + w_sun · B_sun."""

    # tilt of rotation axis towards the Sun (21 June), real value 23.4
    tilt_deg: float = 23.4

    # extra angle between magnetic and rotation axis, real value ~11
    magnetic_offset_deg: float = 0.0

    # direction of that offset, 0 = towards Sun, 90 = dusk
    magnetic_offset_azimuth_deg: float = 0.0

    # multiplies Earth's dipole moment
    field_strength_scale: float = 1.0

    # weight of solar field, 0 = Earth only
    w_sun: float = 1.0

    # distance of the solar dipole in R_E, real Sun is ~23500 R_E
    sun_distance_RE: float = 1000.0

    # r where Earth's equatorial field equals the solar field, 10 R_E gives ~31 nT
    # real IMF is ~5 nT
    r_balance_RE: float = 10.0

    # solar field direction at Earth, south or north
    sun_field_direction: str = "south"

    def with_(self, **changes):
        """Copy with some values changed, e.g. cfg.with_(w_sun=0)."""
        return replace(self, **changes)


# default model and Earth only model
DEFAULT_FIELD = FieldConfig()
OLD_MODEL = FieldConfig(tilt_deg=0.0, w_sun=0.0)


# tracing stops outside this sphere (R_E)
TRACE_R_OUTER_RE = 40.0
# max step along a field line (R_E)
TRACE_MAX_STEP_RE = 0.05
TRACE_MAX_STEP_ZOOM_RE = 0.01      # near Earth
# give up after this length (R_E)
TRACE_MAX_LENGTH_RE = 400.0

# null point search grid
NULL_SEARCH_HALF_WIDTH_RE = 30.0
NULL_SEARCH_GRID_POINTS = 31        # per axis
NULL_MIN_RADIUS_RE = 1.5            # ignore anything closer than this to Earth
NULL_TOLERANCE = 1e-6               # accept if |B| < this × |B_sun at Earth|
NULL_MERGE_DISTANCE_RE = 0.2        # nulls closer than this are the same one

# boundary scan along the noon-midnight meridian
BOUNDARY_SCAN_STEP_DEG = 3.0
BOUNDARY_TOLERANCE_DEG = 0.02


FOOTPOINT_SPACING_DEG = 5.0         # spacing of footpoints along the meridian
SOLAR_SEED_SPACING_RE = 2.5         # seeds on the x axis for solar lines


SIGNED_LOG_X0_NT = 1.0

# floor for |B| before log10
LOG_B_FLOOR_NT = 1e-3

FIELD_MAP_GRID_POINTS = 501


COMBINED_HALF_WIDTH_RE = 14.0       # x, z in ±14 R_E
COMBINED_FOOTPOINT_SPACING_DEG = 2.5
COMBINED_GLOW_PIXELS = 1400         # resolution of the glow image (per axis)

# height exaggeration in the full view
COMBINED_EXAGGERATION_FULL = 25.0
COMBINED_INSET_PAD_RE = 0.25

# log10|B| colour range
COMBINED_LOGB_RANGE = (0.0, 5.0)


# atmosphere, n(h) = n_ref · exp(−(h − h_ref)/H), rounded from MSIS at 100 km
ATMOSPHERE_H_REF_KM = 100.0
N2_DENSITY_REF_M3 = 1.0e19          # N₂ at 100 km  [m⁻³]
O_DENSITY_REF_M3 = 4.0e17           # O at 100 km   [m⁻³]
N2_SCALE_HEIGHT_KM = 10.0           # H_N2: heavier, so it falls off faster
O_SCALE_HEIGHT_KM = 50.0            # H_O

# column an electron gets through, 5e22 puts the peak near 110 km
CHAPMAN_N0_M2 = 5.0e22

# heights the profiles are computed at (km)
AURORA_H_MIN_KM = 80.0
AURORA_H_MAX_KM = 400.0

# aurora band around the open/closed boundary
AURORA_BAND_WIDTH_DEG = 2.0         # 1-sigma width of the auroral band
AURORA_BAND_CUTOFF = 4.0            # in band widths
# weight brightness by |B| at the footpoint
AURORA_WEIGHT_BY_FOOTPOINT_B = True

# heights are drawn this many times too big
ALTITUDE_EXAGGERATION = 10.0
AURORA_PLOT_HALF_WIDTH_RE = 5.0     # side view shows x, z in ±5 R_E
AURORA_IMAGE_PIXELS = 900           # resolution of the glow image

# log brightness range, in factors of 10
BRIGHTNESS_DECADES = 3.0

"""
Plotting functions for the dipole magnetosphere.

Every function takes inputs in SI units (metres, tesla) and draws positions
in Earth radii (R_E).  Each function returns the matplotlib Figure so the
caller can save or show it.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Wedge

from constants import R_E
from field import (B_dipole, field_line_for_L, footpoint_latitude,
                   open_closed_boundary_latitude)

# One fixed colour per meaning, used in every plot.
# Closed vs open lines also differ in line style, so they can be told apart
# without colour (colour-blind readers, black-and-white printing).
COLOUR_CLOSED = "#2a6fdb"      # blue
COLOUR_OPEN = "#d9822b"        # orange
COLOUR_BOUNDARY = "#222222"    # near-black
COLOUR_EARTH_DAY = "#cfe3f7"
COLOUR_EARTH_NIGHT = "#3b4a5a"
COLOUR_AURORA = "#3fae6a"      # green, for the auroral-zone band


def _draw_earth_2d(ax):
    """Earth as a circle of radius 1 R_E, dayside (x > 0) light, nightside dark."""
    ax.add_patch(Wedge((0, 0), 1, -90, 90, facecolor=COLOUR_EARTH_DAY,
                       edgecolor="black", lw=0.8, zorder=5))
    ax.add_patch(Wedge((0, 0), 1, 90, 270, facecolor=COLOUR_EARTH_NIGHT,
                       edgecolor="black", lw=0.8, zorder=5))


def _draw_magnetopause_2d(ax, r_mp):
    """
    Circle of radius r_mp.  Solid on the dayside, where the pressure-balance
    estimate applies; dashed on the nightside, where the real magnetosphere is
    stretched into a long tail that this simple model does not include.
    """
    a = r_mp / R_E
    angle = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.plot(a * np.cos(angle), a * np.sin(angle), color=COLOUR_BOUNDARY, lw=2,
            label=f"Magnetopause (r_mp = {a:.1f} R_E)")
    ax.plot(-a * np.cos(angle), a * np.sin(angle), color=COLOUR_BOUNDARY, lw=1,
            ls=":", label="Nightside boundary (not physical, see field.py)")


def _draw_sun_arrow(ax, x_tip, y=0.0):
    ax.annotate("Sun", xy=(x_tip, y), xytext=(x_tip - 3, y),
                ha="right", va="center", fontsize=11, color="#b8860b",
                arrowprops=dict(arrowstyle="->", color="#b8860b", lw=2))


# ---------------------------------------------------------------------------
# Plot 1: meridional plane field lines
# ---------------------------------------------------------------------------

def plot_meridional_field_lines(L_values, r_mp, field_strength_scale=1.0,
                                extent=16):
    """
    Field lines in the x–z plane (noon–midnight meridian) for each L in
    L_values, on both the dayside (longitude 0) and nightside (longitude 180°),
    plus Earth, the magnetopause, and the direction to the Sun.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    labelled = set()
    for L in L_values:
        for phi in (0.0, np.pi):
            segments, status = field_line_for_L(L, phi, r_mp, field_strength_scale)
            colour = COLOUR_CLOSED if status == "closed" else COLOUR_OPEN
            style = "-" if status == "closed" else "--"
            for seg in segments:
                label = None
                if status not in labelled:
                    label = f"{status} field line"
                    labelled.add(status)
                ax.plot(seg[:, 0] / R_E, seg[:, 2] / R_E, color=colour,
                        ls=style, lw=1.2, label=label)

        # Label each closed line with its L value, just outside its
        # dayside equator crossing.
        if L * R_E < r_mp:
            ax.text(L + 0.15, 0.2, f"{L:g}", fontsize=8, ha="left",
                    color="#444444")
    ax.text(0.5, 0.97, "Numbers on the dayside equator are L values",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color="#444444")

    _draw_magnetopause_2d(ax, r_mp)
    _draw_earth_2d(ax)
    _draw_sun_arrow(ax, extent - 0.3, extent * 0.85)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x [R_E]  (towards Sun →)")
    ax.set_ylabel("z [R_E]  (magnetic north ↑)")
    ax.set_title("Dipole field lines in the noon–midnight meridian")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2: |B| colour map
# ---------------------------------------------------------------------------

def plot_field_strength_map(r_mp, field_strength_scale=1.0, extent=16,
                            n_grid=400):
    """
    Colour map of |B| (log scale) in the x–z plane.  Inside Earth is masked.
    Outside the magnetopause the dipole field is not physical, so that region
    is greyed out.
    """
    x = np.linspace(-extent, extent, n_grid) * R_E
    z = np.linspace(-extent, extent, n_grid) * R_E
    X, Z = np.meshgrid(x, z)
    positions = np.stack([X, np.zeros_like(X), Z], axis=-1)   # shape (n, n, 3)
    r = np.linalg.norm(positions, axis=-1)

    # Avoid evaluating B at the origin; points inside Earth get masked anyway.
    positions[r < 0.5 * R_E] = [R_E, 0, 0]
    B_mag = np.linalg.norm(B_dipole(positions, field_strength_scale), axis=-1)
    B_mag = np.ma.masked_where(r < R_E, B_mag)

    fig, ax = plt.subplots(figsize=(9, 8))
    mesh = ax.pcolormesh(X / R_E, Z / R_E, B_mag, shading="auto",
                         norm=LogNorm(vmin=B_mag.min(), vmax=B_mag.max()),
                         cmap="viridis")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("|B| [T]  (log scale)")

    # Grey veil outside the magnetopause.
    outside = np.ma.masked_where(r <= r_mp, np.ones_like(r))
    ax.pcolormesh(X / R_E, Z / R_E, outside, shading="auto",
                  cmap="Greys", vmin=0, vmax=3, alpha=0.6)

    _draw_magnetopause_2d(ax, r_mp)
    _draw_earth_2d(ax)
    _draw_sun_arrow(ax, extent - 0.3, extent * 0.85)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x [R_E]  (towards Sun →)")
    ax.set_ylabel("z [R_E]")
    ax.set_title("Dipole field strength |B| in the noon–midnight meridian\n"
                 "(greyed region: outside magnetopause, dipole not valid)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3: 3D view
# ---------------------------------------------------------------------------

def plot_3d_field_lines(L_values, n_longitudes, r_mp, field_strength_scale=1.0):
    """
    3D view of field lines at n_longitudes evenly spaced longitudes, with
    Earth and the dayside half of the magnetopause sphere (wireframe).
    """
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")

    for phi in np.linspace(0, 2 * np.pi, n_longitudes, endpoint=False):
        for L in L_values:
            segments, status = field_line_for_L(L, phi, r_mp, field_strength_scale)
            colour = COLOUR_CLOSED if status == "closed" else COLOUR_OPEN
            style = "-" if status == "closed" else "--"
            for seg in segments:
                ax.plot(seg[:, 0] / R_E, seg[:, 1] / R_E, seg[:, 2] / R_E,
                        color=colour, ls=style, lw=0.9)

    # Earth
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color=COLOUR_EARTH_DAY, edgecolor="none", alpha=1.0)

    # Dayside magnetopause (x > 0 half of the sphere)
    a = r_mp / R_E
    u, v = np.mgrid[-np.pi / 2:np.pi / 2:25j, 0:np.pi:25j]
    ax.plot_wireframe(a * np.cos(u) * np.sin(v), a * np.sin(u) * np.sin(v),
                      a * np.cos(v), color=COLOUR_BOUNDARY, lw=0.3, alpha=0.4)

    # Sun direction
    ax.quiver(a + 1, 0, 0, 3, 0, 0, color="#b8860b", lw=2)
    ax.text(a + 4.5, 0, 0, "Sun", color="#b8860b")

    # Proxy lines just for the legend
    ax.plot([], [], color=COLOUR_CLOSED, label="closed field line")
    ax.plot([], [], color=COLOUR_OPEN, ls="--", label="open field line")
    ax.plot([], [], color=COLOUR_BOUNDARY, lw=0.5,
            label=f"dayside magnetopause ({a:.1f} R_E)")
    ax.legend(loc="upper left", fontsize=9)

    lim = 1.1 * max(max(L_values), a)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [R_E]")
    ax.set_ylabel("y [R_E]")
    ax.set_zlabel("z [R_E]")
    ax.set_title("Dipole field lines at several longitudes")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4: footpoint latitude vs L
# ---------------------------------------------------------------------------

def plot_footpoint_latitude(r_mp, L_max=15):
    """
    Footpoint magnetic latitude λ against L, from cos²λ = 1/L.
    Shows the auroral zone (65–70°) and the open/closed boundary set by r_mp.
    """
    L_curve = np.linspace(1, L_max, 500)
    L_marks = np.arange(2, 11)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhspan(65, 70, color=COLOUR_AURORA, alpha=0.15,
               label="Auroral zone (≈ 65–70°)")

    L_boundary = r_mp / R_E
    ax.axvline(L_boundary, color=COLOUR_BOUNDARY, lw=1.2, ls="--")
    ax.text(L_boundary + 0.15, 20,
            f"magnetopause\nL = {L_boundary:.1f}\nλ = "
            f"{open_closed_boundary_latitude(r_mp):.1f}°",
            fontsize=9, va="bottom")
    ax.text(L_boundary - 0.3, 5, "closed", ha="right", color=COLOUR_CLOSED)
    ax.text(L_boundary + 0.3, 5, "open", ha="left", color=COLOUR_OPEN)

    ax.plot(L_curve, footpoint_latitude(L_curve), color=COLOUR_CLOSED, lw=2,
            label="λ = arccos(√(1/L))")
    ax.plot(L_marks, footpoint_latitude(L_marks), "o", color=COLOUR_CLOSED,
            markersize=6, label="L = 2 … 10")

    ax.set_xlim(1, L_max)
    ax.set_ylim(0, 90)
    ax.set_xlabel("L  (equatorial crossing distance in R_E)")
    ax.set_ylabel("Footpoint magnetic latitude λ [°]")
    ax.set_title("Where each L-shell meets Earth's surface")
    ax.grid(alpha=0.2)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    return fig

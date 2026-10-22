import numpy as np
import matplotlib.pyplot as plt

from constants import R_E
from field import B_dipole, field_line_for_L

COLOUR_EARTH_DAY = "#cfe3f7"


def _draw_earth(ax):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color=COLOUR_EARTH_DAY, edgecolor="none", alpha=0.9)


def _set_cube_axes(ax, lim, title):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (Sun →)")
    ax.set_ylabel("y")
    ax.set_zlabel("z ")
    ax.set_title(title)


# ===========================================================================
# Stage 1: the field
# ===========================================================================

def plot_field_direction_3d(r_mp, L_values=(2, 4, 6, 8), n_longitudes=6,
                            arrow_fractions=(0.2, 0.5, 0.8),
                            field_strength_scale=1.0):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    lim = 1.1 * max(L_values)
    colours = plt.cm.viridis(np.linspace(0, 0.85, len(L_values)))

    for L, colour in zip(L_values, colours):
        for phi in np.linspace(0, 2 * np.pi, n_longitudes, endpoint=False):
            segments, _ = field_line_for_L(L, phi, r_mp, field_strength_scale)
            for seg in segments:
                ax.plot(*(seg.T / R_E), color=colour, lw=0.8)
                idx = (np.array(arrow_fractions) * (len(seg) - 1)).astype(int)
                B = B_dipole(seg[idx], field_strength_scale)
                ax.quiver(*(seg[idx].T / R_E), *B.T, length=0.08 * lim,
                          normalize=True, arrow_length_ratio=0.6, color=colour)
        ax.plot([], [], color=colour, label=f"L = {L:g}")

    _draw_earth(ax)
    _set_cube_axes(ax, lim, "Dipole field lines, arrows show the direction of B")
    ax.view_init(elev=20, azim=-60 + 180 / n_longitudes)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def animate_rotation(fig, n_frames=120, elev=20, interval_ms=50):
    from matplotlib.animation import FuncAnimation

    ax = fig.axes[0]
    azim0 = ax.azim

    def update(i):
        ax.view_init(elev=elev, azim=azim0 + 360 * i / n_frames)
        return []

    return FuncAnimation(fig, update, frames=n_frames, interval=interval_ms)


# ===========================================================================
# Stage 2: particle trajectories
# ===========================================================================

def plot_trajectory_3d(result, species, L_context, r_mp, title,
                       field_strength_scale=1.0, n_context_lines=6):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")

    for phi in np.linspace(0, 2 * np.pi, n_context_lines, endpoint=False):
        segments, _ = field_line_for_L(L_context, phi, r_mp, field_strength_scale)
        for seg in segments:
            ax.plot(*(seg.T / R_E), color="#999999", lw=0.6, alpha=0.6)

    _draw_earth(ax)

    pts = result["r"] / R_E
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap="plasma", lw=0.6)
    lc.set_array(result["t"][:-1])
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("time [s]")

    ax.scatter(*pts[0], color="black", s=25, label="start")
    if result["hit"] is not None:
        ax.scatter(*(result["hit"]["position"] / R_E), color="red", marker="*",
                   s=150, label=f"hits Earth at λ = {result['hit']['latitude_deg']:.1f}°")
    ax.plot([], [], color="#999999", label=f"field lines, L = {L_context:g}")

    _set_cube_axes(ax, 1.1 * L_context, title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig

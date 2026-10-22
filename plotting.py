import numpy as np
import matplotlib.pyplot as plt

from constants import R_E
from field import field_line_for_L

COLOUR_EARTH_DAY = "#cfe3f7"


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

    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color=COLOUR_EARTH_DAY, edgecolor="none", alpha=0.9)

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

    lim = 1.1 * L_context
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [R_E] (Sun →)")
    ax.set_ylabel("y [R_E]")
    ax.set_zlabel("z [R_E]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig

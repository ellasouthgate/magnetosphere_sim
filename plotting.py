import numpy as np
import matplotlib.pyplot as plt

from constants import R_E
from field import B_dipole, field_line_for_L
import config as C
from scales import log10_magnitude
from sources import geographic_north

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
    ax.set_xlabel("x  (← Sun)")
    ax.set_ylabel("y")
    ax.set_zlabel("z ")
    ax.set_title(title)


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


def northern_glow_box(boundaries, exaggeration, pad_RE=C.COMBINED_INSET_PAD_RE):
    """Square box (x0, x1, z0, z1) in R_E around the northern aurora."""
    top = 1.0 + C.AURORA_H_MAX_KM * exaggeration / 6371.0
    pts = []
    for b in boundaries["north"]:
        f = b["footpoint"][[0, 2]] / R_E
        pts += [f, top * f]
    pts = np.array(pts)
    lo, hi = pts.min(axis=0) - pad_RE, pts.max(axis=0) + pad_RE
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo).max()
    return (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)


def _draw_combined(ax, x, z, L, lines, nulls_RE, cfg, glow_rgba, glow_half_width,
                   logB_range):
    im = ax.imshow(L, extent=(x[0], x[-1], z[0], z[-1]), origin="lower", cmap="gray",
                   vmin=logB_range[0], vmax=logB_range[1])

    colours = {"closed": "C0", "open": "C1", "solar": "C2"}
    for line in lines:
        p = line["points"] / R_E
        if line.get("separatrix"):
            ax.plot(p[:, 0], p[:, 2], "r", lw=1)
        else:
            ax.plot(p[:, 0], p[:, 2], colours[line["status"]], lw=0.6)

    ax.add_patch(plt.Circle((0, 0), 1, color="lightblue", zorder=5))
    n = geographic_north(cfg)
    ax.plot([-1.5 * n[0], 1.5 * n[0]], [-1.5 * n[2], 1.5 * n[2]], "k--", lw=1, zorder=6)

    ext = (-glow_half_width, glow_half_width, -glow_half_width, glow_half_width)
    ax.imshow(glow_rgba, extent=ext, origin="lower", zorder=7)

    if len(nulls_RE):
        ax.plot(nulls_RE[:, 0], nulls_RE[:, 2], "yx", ms=8, mew=2, zorder=8)
    return im


def plot_combined(cfg, x, z, B_nT, lines, nulls_RE, boundaries, glow_rgba,
                  glow_half_width, exaggeration, half_width, title,
                  logB_range=C.COMBINED_LOGB_RANGE):
    """log10|B|, field lines, nulls and aurora, with a zoom on the north pole."""
    L = log10_magnitude(B_nT)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    im = _draw_combined(ax, x, z, L, lines, nulls_RE, cfg, glow_rgba, glow_half_width,
                        logB_range)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_xlabel("x [R_E] (Sun on the left)")
    ax.set_ylabel("z [R_E]")
    ax.set_title(title)

    ax.plot([], [], "C0", label="closed")
    ax.plot([], [], "C1", label="open")
    ax.plot([], [], "C2", label="solar")
    ax.plot([], [], "r", label="open/closed boundary")
    ax.plot([], [], "yx", label="null point")
    ax.plot([], [], "k--", label="rotation axis")
    ax.legend(loc="lower left", fontsize=8)

    _draw_combined(ax2, x, z, L, lines, nulls_RE, cfg, glow_rgba, glow_half_width,
                   logB_range)
    x0, x1, z0, z1 = northern_glow_box(boundaries, exaggeration)
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(z0, z1)
    ax2.set_xlabel("x [R_E]")
    ax2.set_ylabel("z [R_E]")
    ax2.set_title(f"north pole zoom (aurora heights x{exaggeration:g})")

    fig.colorbar(im, ax=[ax, ax2], label="log10 |B| [nT]", shrink=0.8)
    return fig


def plot_colour_pipeline(lines, h_km, profiles, peaks):
    """Spectrum, CIE curves, chromaticity diagram and colour vs height."""
    import aurora_colour as AC

    wl = np.array([l.wavelength_nm for l in lines])
    colours = [AC.line_swatch(w) for w in wl]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    I = profiles.sum(axis=0)
    ax1.bar(wl, I / I.max(), width=4, color=colours)
    for line, i in zip(lines, I / I.max()):
        ax1.text(line.wavelength_nm, i, line.name, ha="center", va="bottom", fontsize=8)
    ax1.set_xlim(380, 720)
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel("wavelength [nm]")
    ax1.set_ylabel("relative intensity")
    ax1.set_title("emission lines")

    lam = np.linspace(380, 720, 341)
    cmf = AC.cie_cmf(lam)
    ax2.plot(lam, cmf[:, 0], "r", label="x bar")
    ax2.plot(lam, cmf[:, 1], "g", label="y bar")
    ax2.plot(lam, cmf[:, 2], "b", label="z bar")
    for w in wl:
        ax2.axvline(w, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("wavelength [nm]")
    ax2.set_ylabel("response")
    ax2.set_title("CIE 1931 colour matching functions")
    ax2.legend()

    locus = AC.xyz_to_xy(cmf)
    ax3.plot(np.r_[locus[:, 0], locus[0, 0]], np.r_[locus[:, 1], locus[0, 1]], "k",
             label="spectral locus")
    prim = AC.xyz_to_xy(AC.linear_srgb_to_xyz(np.eye(3)))
    ax3.plot(np.r_[prim[:, 0], prim[0, 0]], np.r_[prim[:, 1], prim[0, 1]], "b",
             label="sRGB")
    for i, line in enumerate(lines):
        xyz = AC.cie_cmf(line.wavelength_nm)
        rgb, _ = AC.reduce_chroma_into_gamut(AC.xyz_to_linear_srgb(xyz))
        ax3.plot(*AC.xyz_to_xy(xyz), "o", color="C1", label="line" if i == 0 else None)
        ax3.plot(*AC.xyz_to_xy(AC.linear_srgb_to_xyz(rgb)), "s", color="C2",
                 label="line on screen" if i == 0 else None)
    xy_h = AC.xyz_to_xy(AC.lines_to_xyz(profiles, wl))
    ax3.plot(xy_h[:, 0], xy_h[:, 1], "r", label="aurora at each height")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("chromaticity")
    ax3.legend(fontsize=8)

    rgb_b, _ = AC.colourise(profiles, wl)
    rgb_h, _ = AC.colourise(profiles, wl, decades=1e9)
    img = np.concatenate([rgb_b[:, None, :], rgb_h[:, None, :]], axis=1)
    ax4.imshow(img, extent=(0, 2, h_km[0], h_km[-1]), origin="lower", aspect="auto",
               interpolation="nearest")
    ax4.set_xticks([0.5, 1.5], ["log brightness", "hue only"])
    ax4.set_ylabel("height [km]")
    ax4.set_title("colour vs height")

    fig.tight_layout()
    return fig

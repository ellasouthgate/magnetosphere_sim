import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT_FIELD
from constants import R_E
import plotting
from plotting import plot_combined


def test_plot_combined_runs():
    from sources import field_on_xz_grid
    cfg = DEFAULT_FIELD
    x, z, B = field_on_xz_grid(cfg, 3.0, 41)
    foot = np.array([-0.7, 0.0, 0.7]) * R_E
    lines = [{"points": np.array([[1.0, 0, 0.1], [2.0, 0, 0.5]]) * R_E, "status": "closed"},
             {"points": np.array([[0.1, 0, 1.0], [0.3, 0, 2.5]]) * R_E, "status": "open"},
             {"points": np.array([[0.2, 0, 1.0], [0.4, 0, 2.5]]) * R_E, "status": "open",
              "separatrix": True}]
    boundaries = {"north": [{"footpoint": foot}], "south": []}
    rgba = np.zeros((50, 50, 4))
    fig = plot_combined(cfg, x, z, B, lines, np.array([[2.0, 0, 1.0]]), boundaries,
                        rgba, 3.0, 10.0, 3.0, "test")
    assert fig.axes[0].get_xlim() == (-3.0, 3.0)
    assert len(fig.axes) == 3
    assert "x10" in fig.axes[1].get_title()
    plt.close(fig)


def test_northern_glow_box_contains_the_glow():
    from plotting import northern_glow_box
    b = {"north": [{"footpoint": np.array([-0.7, 0.0, 0.7]) * R_E},
                   {"footpoint": np.array([0.0, 0.0, 1.0]) * R_E}], "south": []}
    x0, x1, z0, z1 = northern_glow_box(b, exaggeration=25.0, pad_RE=0.2)
    top = 1 + 400 * 25 / 6371
    for f in ([-0.7, 0.7], [0.0, 1.0], [-0.7 * top, 0.7 * top], [0.0, top]):
        assert x0 < f[0] < x1 and z0 < f[1] < z1
    assert x1 - x0 == pytest.approx(z1 - z0)


def test_plot_colour_pipeline_runs():
    import aurora_colour as AC
    h = np.linspace(80, 400, 81)
    P = AC.line_profiles(h)
    fig = plotting.plot_colour_pipeline(AC.AURORA_LINES, h, P, AC.peak_heights())
    assert len(fig.axes) == 4
    plt.close(fig)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
from gammapy.utils.testing import mpl_plot_check, requires_dependency
from gammapy.visualization import plot_contour_line
from gammapy.visualization import plot_theta2_distribution
import astropy.units as u
from astropy.table import Table


@requires_dependency("matplotlib")
def test_map_panel_plotter():
    import matplotlib.pyplot as plt

    t = np.linspace(0.0, 6.1, 10)
    x = np.cos(t)
    y = np.sin(t)

    ax = plt.subplot(111)
    with mpl_plot_check():
        plot_contour_line(ax, x, y)


@requires_dependency("matplotlib")
def test_plot_theta2_distribution():
    theta2_lo = [0, 0.05, 0.1, 0.15] * u.deg ** 2
    theta2_hi = [0.05, 0.1, 0.15, 0.2] * u.deg ** 2
    theta2_edges = edges_from_lo_hi(theta2_lo, theta2_hi)
    theta2_axis = MapAxis.from_edges(
        theta2_edges, interp="lin", name="offset", unit=theta2_edges.unit
    )
    on_cnts = [2, 0, 0, 0]
    off_cnts = [1, 0, 0, 0]
    acceptance = [1, 1, 1, 1]
    acceptance_off = [1, 1, 1, 1]
    alpha = [1, 1, 1, 1]
    theta2_distribution_table = Table(
        {
            "theta2_min": theta2_lo,
            "theta2_max": theta2_hi,
            "counts": on_cnts,
            "counts_off": off_cnts,
            "acceptance": acceptance,
            "acceptance_off": acceptance_off,
            "alpha": alpha,
        }
    )
    plot_theta2_distribution(theta2_distribution_table=theta2_distribution_table)

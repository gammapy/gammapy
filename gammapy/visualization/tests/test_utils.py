# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Table
from gammapy.utils.testing import mpl_plot_check, requires_dependency
from gammapy.visualization import plot_contour_line, plot_theta_squared_table


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
    table = Table()
    table["theta2_min"] = [0, 0.1]
    table["theta2_max"] = [0.1, 0.2]

    for column in [
        "counts",
        "counts_off",
        "excess",
        "excess_errp",
        "excess_errn",
        "sqrt_ts",
    ]:
        table[column] = [1, 1]

    plot_theta_squared_table(table=table)

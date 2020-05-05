# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.utils.testing import mpl_plot_check, requires_dependency
from gammapy.visualization import plot_contour_line


@requires_dependency("matplotlib")
def test_map_panel_plotter():
    import matplotlib.pyplot as plt

    t = np.linspace(0., 6.1, 10)
    x = np.cos(t)
    y = np.sin(t)

    ax = plt.subplot(111)
    with mpl_plot_check():
        plot_contour_line(ax, x, y)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
from gammapy.maps import Map
from gammapy.utils.testing import mpl_plot_check
from gammapy.visualization import MapPanelPlotter


def test_map_panel_plotter():
    fig = plt.figure()
    plotter = MapPanelPlotter(
        figure=fig, xlim=Angle([-5, 5], "deg"), ylim=Angle([-2, 2], "deg"), npanels=2
    )
    map_image = Map.create(width=(180, 10), binsz=1)

    with mpl_plot_check():
        plotter.plot(map_image)

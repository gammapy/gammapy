# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.coordinates import Angle
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency, mpl_plot_check, requires_data
from ...image import (
    colormap_hess,
    colormap_milagro,
    MapPanelPlotter,
    illustrate_colormap,
)
from ...maps import Map


def _check_cmap_rgb_vals(vals, cmap, vmin=0, vmax=1):
    """Helper function to check RGB values of color images"""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    norm = Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    for val, rgb_expected in vals:
        rgb_actual = sm.to_rgba(val)[:-1]
        assert_allclose(rgb_actual, rgb_expected, atol=1e-5)


@requires_dependency("matplotlib")
def test_colormap_hess():
    transition = 0.5
    cmap = colormap_hess(transition=transition)
    vals = [
        (0, (0.0, 0.0, 0.0)),
        (0.25, (0.0, 0.0, 0.50196078)),
        (0.5, (1.0, 0.0058823529411764722, 0.0)),
        (0.75, (1.0, 0.75882352941176501, 0.0)),
        (1, (1.0, 1.0, 1.0)),
    ]
    _check_cmap_rgb_vals(vals, cmap)


@requires_dependency("matplotlib")
def test_colormap_milagro():
    transition = 0.5
    cmap = colormap_milagro(transition=transition)
    vals = [
        (0, (1.0, 1.0, 1.0)),
        (0.25, (0.4979388, 0.4979388, 0.4979388)),
        (0.5, (0.00379829, 0.3195442, 0.79772102)),
        (0.75, (0.51610773, 0.25806707, 0.49033536)),
        (1.0, (1.0, 1.0, 1.0)),
    ]
    _check_cmap_rgb_vals(vals, cmap)


@requires_data()
@requires_dependency("matplotlib")
def test_map_panel_plotter():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plotter = MapPanelPlotter(
        figure=fig, xlim=Angle([-5, 5], "deg"), ylim=Angle([-2, 2], "deg"), npanels=2
    )
    map_image = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz")

    with mpl_plot_check():
        plotter.plot(map_image)


@requires_dependency("matplotlib")
def test_illustrate_colormap():
    with mpl_plot_check():
        illustrate_colormap("afmhot")

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.visualization import colormap_hess, colormap_milagro


def _check_cmap_rgb_vals(vals, cmap, vmin=0, vmax=1):
    """Helper function to check RGB values of color images"""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    norm = Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    for val, rgb_expected in vals:
        rgb_actual = sm.to_rgba(val)[:-1]
        assert_allclose(rgb_actual, rgb_expected, atol=1e-5)


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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...image import (colormap_hess,
                      colormap_milagro,
                      GalacticPlaneSurveyPanelPlot,
                      )

try:
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_cmap_rgb_vals(vals, cmap, vmin, vmax):
    """Helper function to check RGB values of color maps"""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    for val, rgb_expected in vals:
        rgb_actual = sm.to_rgba(val)[:-1]
        assert_allclose(rgb_actual, rgb_expected, atol=1e-5)


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_colormap_hess():
    vmin, vmax, vtransition = -5, 15, 5
    cmap = colormap_hess(vmin=vmin, vmax=vmax, vtransition=vtransition)
    vals = [(-5, (0.0, 0.0, 0.0)),
            (0, (0.0, 0.0, 0.50196078)),
            (5, (1.0, 0.0058823529411764722, 0.0)),
            (10, (1.0, 0.75882352941176501, 0.0)),
            (15, (1.0, 1.0, 1.0)),
           ]
    _check_cmap_rgb_vals(vals, cmap, vmin, vmax)


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_colormap_milagro():
    vmin, vmax, vtransition = -5, 15, 5
    cmap = colormap_milagro(vmin=vmin, vmax=vmax, vtransition=vtransition)
    vals = [(-5, (1.0, 1.0, 1.0)),
            (0, (0.4979388, 0.4979388, 0.4979388)),
            (5, (0.00379829, 0.3195442, 0.79772102)),
            (10, (0.51610773, 0.25806707, 0.49033536)),
            (15, (1.0, 1.0, 1.0)),
            ]
    _check_cmap_rgb_vals(vals, cmap, vmin, vmax)


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_GalacticPlaneSurveyPanelPlot():

    plot = GalacticPlaneSurveyPanelPlot(npanels=3)
    assert_allclose(plot.panel_parameters['npanels'], 3)

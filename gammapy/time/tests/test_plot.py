# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
from astropy.tests.helper import remote_data
from ..plot import plot_fermi_3fgl_light_curve

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_plot_time_difference_distribution():
    pass


# TODO: put 3FGL in gammapy-extra for testing ...
@remote_data
@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_plot_fermi_3fgl_light_curve():
    plot_fermi_3fgl_light_curve('3FGL J0349.9-2102',
                                time_start='2010-01-01',
                                time_end='2015-02-02')

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from tempfile import NamedTemporaryFile
from astropy.units import Quantity
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.time import Time
from astropy.table import Table
from ...utils.testing import assert_quantity
from ..utils import np_to_pha, LogEnergyAxis, linear_extrapolator

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestPHA(object):

    def setup_class(self):
        """Create test PHA file."""

        counts = np.array([0., 0., 5., 10., 20., 15., 10., 5., 2.5, 1., 1., 0.])
        stat_err = np.sqrt(counts)
        channel = (np.arange(len(counts)))

        exposure = 3600.

        dstart = Time('2011-01-01 00:00:00', scale='utc')
        dstop = Time('2011-01-31 00:00:00', scale='utc')
        dbase = Time('2011-01-01 00:00:00', scale='utc')

        pha = np_to_pha(channel=channel, counts=counts, exposure=exposure,
                        dstart=dstart, dstop=dstop, dbase=dbase,
                        stat_err=stat_err)

        #self.PHA_FILENAME = tmpdir.join('dummy.pha').strpath
        self.PHA_FILENAME = NamedTemporaryFile(suffix='.pha', delete=False).name
        self.PHA_SUM = np.sum(counts)

        pha.writeto(self.PHA_FILENAME)

    def test_pha(self):
        pha = Table.read(self.PHA_FILENAME)
        assert_allclose(pha['COUNTS'].sum(), 69.5)


@pytest.mark.skipif('not HAS_SCIPY')
def test_LogEnergyAxis():
    from scipy.stats import gmean
    energy = Quantity([1, 10, 100], 'TeV')
    energy_axis = LogEnergyAxis(energy)

    assert_allclose(energy_axis.x, [0, 1, 2])
    assert_quantity(energy_axis.energy, energy)

    energy = Quantity(gmean([1, 10]), 'TeV')
    pix = energy_axis.world2pix(energy.to('MeV'))
    assert_allclose(pix, 0.5)

    world = energy_axis.pix2world(pix)
    assert_quantity(world, energy)


@pytest.mark.xfail
def test_linear_extrapolator():
    # Exact case
    y_vals = np.arange(5, 100, 0.1)
    x_vals = 2*y_vals
    f = linear_extrapolator(x_vals, y_vals)
    test_value = f(200)
    assert_allclose(test_value, 400, 1e-1)

    # Realistic case
    # TODO: Find a better way to implement this
    x_vals = np.arange(1, 10000)
    y_vals = 2*x_vals
    # Introduce Poisson fluctuations on the 0.1 scale
    new_x = x_vals - 0.1 * np.random.poisson(np.ones(x_vals.shape))
    new_y = y_vals - 0.1 * np.random.poisson(np.ones(y_vals.shape))
    g = linear_extrapolator(new_x, new_y)
    test_value = g(200)
    # Error within 10%
    assert_allclose(test_value, 400, 40)

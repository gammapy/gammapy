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
from ...spectrum import np_to_pha, LogEnergyAxis, energy_bin_centers_log_spacing

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


def test_energy_bin_centers_log_spacing():
    energy_bounds = Quantity([1, 2, 10], 'GeV')
    actual = energy_bin_centers_log_spacing(energy_bounds)
    desired = Quantity([1.41421356, 4.47213595], 'GeV')
    assert_quantity(actual, desired)

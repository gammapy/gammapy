# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from astropy.table import Table
from ...utils.testing import requires_dependency
from ...spectrum import np_to_pha, LogEnergyAxis


class TestPHA(object):
    def test_pha(self, tmpdir):
        """Create test PHA file."""

        counts = np.array([0., 0., 5., 10., 20., 15., 10., 5., 2.5, 1., 1., 0.])
        stat_err = np.sqrt(counts)
        channel = (np.arange(len(counts)))

        exposure = 3600.

        dstart = Time('2011-01-01T00:00:00')
        dstop = Time('2011-01-31T00:00:00')
        dbase = Time('2011-01-01T00:00:00')

        pha = np_to_pha(channel=channel, counts=counts, exposure=exposure,
                        dstart=dstart, dstop=dstop, dbase=dbase,
                        stat_err=stat_err)

        filename = str(tmpdir / 'pha_test.pha')
        pha.writeto(filename)

        pha = Table.read(filename)
        assert_allclose(pha['COUNTS'].sum(), 69.5)


@requires_dependency('scipy')
def test_LogEnergyAxis():
    from scipy.stats import gmean
    energy = Quantity([1, 10, 100], 'TeV')
    energy_axis = LogEnergyAxis(energy)

    assert_allclose(energy_axis.x, [0, 1, 2])
    assert_quantity_allclose(energy_axis.energy, energy)

    energy = Quantity(gmean([1, 10]), 'TeV')
    pix = energy_axis.world2pix(energy.to('MeV'))
    assert_allclose(pix, 0.5)

    world = energy_axis.pix2world(pix)
    assert_quantity_allclose(world, energy)

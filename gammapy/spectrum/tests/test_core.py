# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import numpy as np
import astropy.units as u
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import EnergyBounds
from .. import CountsSpectrum, PHACountsSpectrum


@requires_dependency('scipy')
class TestCountsSpectrum:

    def setup(self):
        self.counts = [0, 0, 2, 5, 17, 3] * u.ct
        self.bins = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
        self.spec = CountsSpectrum(data=self.counts, energy=self.bins)

    def test_init_wo_unit(self):
        counts = [2, 5]
        energy = [1, 2, 3] * u.TeV
        spec = CountsSpectrum(data=counts, energy=energy)
        assert spec.data.unit.is_equivalent(u.ct)

        counts = u.Quantity([2, 5])
        spec = CountsSpectrum(data=counts, energy=energy)
        assert spec.data.unit.is_equivalent(u.ct)

    def test_wrong_init(self):
        bins = EnergyBounds.equal_log_spacing(1, 10, 7, 'TeV')
        with pytest.raises(ValueError):
            CountsSpectrum(data=self.counts, energy=bins)

    def test_evaluate(self):
        test_e = self.bins[2] + 0.1 * u.TeV
        test_eval = self.spec.evaluate(energy=test_e)
        assert_allclose(test_eval, self.counts[2])

    @requires_dependency('matplotlib')
    def test_plot(self):
        self.spec.plot()
        self.spec.plot_hist()

    def test_io(self, tmpdir):
        filename = tmpdir / 'test.fits'
        self.spec.write(filename)
        spec2 = CountsSpectrum.read(filename)
        assert_quantity_allclose(spec2.energy.data, self.bins)


@requires_dependency('scipy')
class TestPHACountsSpectrum:

    def setup(self):
        counts = [1, 2, 5, 6, 1, 7, 23]
        self.binning = EnergyBounds.equal_log_spacing(1, 10, 7, 'TeV')
        quality = [1, 1, 1, 0, 0, 1, 1]
        self.spec = PHACountsSpectrum(data=counts,
                                      energy=self.binning,
                                      quality=quality)
        self.spec.backscal = 0.3
        self.spec.obs_id = 42
        self.spec.livetime = 3 * u.h

    def test_init_wo_unit(self):
        counts = [2, 5]
        energy = [1, 2, 3] * u.TeV
        spec = PHACountsSpectrum(data=counts, energy=energy)
        assert spec.data.unit.is_equivalent(u.ct)

        counts = u.Quantity([2, 5])
        spec = PHACountsSpectrum(data=counts, energy=energy)
        assert spec.data.unit.is_equivalent(u.ct)

    def test_basic(self):
        assert 'PHACountsSpectrum' in str(self.spec)
        assert_quantity_allclose(self.spec.lo_threshold,
                                 self.binning[3])
        assert_quantity_allclose(self.spec.hi_threshold,
                                 self.binning[5])

    def test_thresholds(self):
        self.spec.quality = np.zeros(self.spec.energy.nbins, dtype=int)
        self.spec.lo_threshold = 1.5 * u.TeV
        self.spec.hi_threshold = 4.5 * u.TeV
        assert (self.spec.quality == [1, 1, 0, 0, 1, 1, 1]).all()
        assert_quantity_allclose(self.spec.lo_threshold, 1.93069773 * u.TeV)
        assert_quantity_allclose(self.spec.hi_threshold, 3.72759372 * u.TeV)

    def test_io(self, tmpdir):
        filename = tmpdir / 'test2.fits'
        self.spec.write(filename)
        spec2 = PHACountsSpectrum.read(filename)
        assert_quantity_allclose(spec2.energy.data, self.spec.energy.data)

    def test_backscal_array(self, tmpdir):
        self.spec.backscal = np.arange(self.spec.energy.nbins)
        table = self.spec.to_table()
        assert table['BACKSCAL'][2] == 2

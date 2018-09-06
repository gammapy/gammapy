# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
import numpy as np
import astropy.units as u
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, mpl_plot_check
from ...utils.energy import EnergyBounds
from .. import CountsSpectrum, PHACountsSpectrum


@requires_dependency("scipy")
class TestCountsSpectrum:
    def setup(self):
        self.counts = [0, 0, 2, 5, 17, 3]
        self.bins = EnergyBounds.equal_log_spacing(1, 10, 6, "TeV")
        self.spec = CountsSpectrum(
            data=self.counts, energy_lo=self.bins[:-1], energy_hi=self.bins[1:]
        )

    def test_init_wo_unit(self):
        counts = [2, 5]
        energy = [1, 2, 3] * u.TeV
        spec = CountsSpectrum(data=counts, energy_lo=energy[:-1], energy_hi=energy[1:])
        assert spec.data.data.unit.is_equivalent("")

        counts = u.Quantity([2, 5])
        spec = CountsSpectrum(data=counts, energy_lo=energy[:-1], energy_hi=energy[:-1])

        assert spec.data.data.unit.is_equivalent("")

    def test_wrong_init(self):
        bins = EnergyBounds.equal_log_spacing(1, 10, 7, "TeV")
        with pytest.raises(ValueError):
            CountsSpectrum(
                data=self.counts,
                energy_lo=bins.lower_bounds,
                energy_hi=bins.upper_bounds,
            )

    def test_evaluate(self):
        test_e = self.bins[2] + 0.1 * u.TeV
        test_eval = self.spec.data.evaluate(energy=test_e)
        assert_allclose(test_eval, self.counts[2])

    @requires_dependency("matplotlib")
    def test_plot(self):
        with mpl_plot_check():
            self.spec.plot()

        with mpl_plot_check():
            self.spec.plot_hist()

    def test_io(self, tmpdir):
        filename = tmpdir / "test.fits"
        self.spec.write(filename)
        spec2 = CountsSpectrum.read(filename)
        assert_quantity_allclose(spec2.energy.bins, self.bins)

    def test_rebin(self):
        rebinned_spec = self.spec.rebin(2)
        assert rebinned_spec.energy.nbins == self.spec.energy.nbins / 2
        assert rebinned_spec.data.data.shape[0] == self.spec.data.data.shape[0] / 2
        assert rebinned_spec.total_counts == self.spec.total_counts

        with pytest.raises(ValueError):
            rebinned_spec = self.spec.rebin(4)

        actual = rebinned_spec.data.evaluate(energy=[2, 3, 5] * u.TeV)
        desired = [0, 7, 20]
        assert (actual == desired).all()


@requires_dependency("scipy")
class TestPHACountsSpectrum:
    def setup(self):
        counts = [1, 2, 5, 6, 1, 7, 23, 2]
        self.binning = EnergyBounds.equal_log_spacing(1, 10, 8, "TeV")
        quality = [1, 1, 1, 0, 0, 1, 1, 1]
        self.spec = PHACountsSpectrum(
            data=counts,
            energy_lo=self.binning.lower_bounds,
            energy_hi=self.binning.upper_bounds,
            quality=quality,
        )
        self.spec.backscal = 0.3
        self.spec.obs_id = 42
        self.spec.livetime = 3 * u.h

    def test_init_wo_unit(self):
        counts = [2, 5]
        energy = [1, 2, 3] * u.TeV
        spec = PHACountsSpectrum(
            data=counts, energy_lo=energy[:-1], energy_hi=energy[1:]
        )
        assert spec.data.data.unit.is_equivalent("")

        counts = u.Quantity([2, 5])
        spec = PHACountsSpectrum(
            data=counts, energy_lo=energy[:-1], energy_hi=energy[1:]
        )
        assert spec.data.data.unit.is_equivalent("")

    def test_basic(self):
        assert "PHACountsSpectrum" in str(self.spec)
        assert_quantity_allclose(self.spec.lo_threshold, self.binning[3])
        assert_quantity_allclose(self.spec.hi_threshold, self.binning[5])

    def test_thresholds(self):
        self.spec.quality = np.zeros(self.spec.energy.nbins, dtype=int)
        self.spec.lo_threshold = 1.5 * u.TeV
        self.spec.hi_threshold = 4.5 * u.TeV
        assert (self.spec.quality == [1, 1, 0, 0, 0, 1, 1, 1]).all()
        assert_quantity_allclose(self.spec.lo_threshold, 1.778279410038922 * u.TeV)
        assert_quantity_allclose(self.spec.hi_threshold, 4.216965034285822 * u.TeV)

    def test_io(self, tmpdir):
        filename = tmpdir / "test2.fits"
        self.spec.write(filename)
        spec2 = PHACountsSpectrum.read(filename)
        assert_quantity_allclose(spec2.energy.bins, self.spec.energy.bins)

    def test_backscal_array(self, tmpdir):
        self.spec.backscal = np.arange(self.spec.energy.nbins)
        table = self.spec.to_table()
        assert table["BACKSCAL"][2] == 2

    def test_rebin(self):
        spec_rebinned = self.spec.rebin(2)
        assert (spec_rebinned.quality == [1, 0, 0, 1]).all()
        assert_quantity_allclose(spec_rebinned.hi_threshold, 5.623413251903491 * u.TeV)
        assert_quantity_allclose(spec_rebinned.lo_threshold, 1.778279410038922 * u.TeV)

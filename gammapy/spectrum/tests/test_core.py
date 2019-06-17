# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import numpy as np
import astropy.units as u
from ...utils.testing import (
    requires_dependency,
    mpl_plot_check,
    assert_quantity_allclose,
)
from ...utils.energy import energy_logspace
from .. import CountsSpectrum, PHACountsSpectrum


class TestCountsSpectrum:
    def setup(self):
        self.counts = [0, 0, 2, 5, 17, 3]
        self.bins = energy_logspace(1, 10, 7, "TeV")
        self.spec = CountsSpectrum(
            data=self.counts, energy_lo=self.bins[:-1], energy_hi=self.bins[1:]
        )

    def test_wrong_init(self):
        bins = energy_logspace(1, 10, 8, "TeV")
        with pytest.raises(ValueError):
            CountsSpectrum(
                data=self.counts,
                energy_lo=bins[:-1],
                energy_hi=bins[1:],
            )

    @requires_dependency("matplotlib")
    def test_plot(self):
        with mpl_plot_check():
            self.spec.plot(show_energy=1 * u.TeV)

        with mpl_plot_check():
            self.spec.plot_hist()

        with mpl_plot_check():
            self.spec.peek()

    def test_io(self, tmpdir):
        filename = tmpdir / "test.fits"
        self.spec.write(filename)
        spec2 = CountsSpectrum.read(filename)
        assert_quantity_allclose(spec2.energy.edges, self.bins)

    def test_downsample(self):
        rebinned_spec = self.spec.downsample(2)
        assert rebinned_spec.energy.nbin == self.spec.energy.nbin / 2
        assert rebinned_spec.data.shape[0] == self.spec.data.shape[0] / 2
        assert rebinned_spec.total_counts == self.spec.total_counts

        idx = rebinned_spec.energy.coord_to_idx([2, 3, 5] * u.TeV)
        actual = rebinned_spec.data[idx]
        desired = [0, 7, 20]
        assert (actual == desired).all()


class TestPHACountsSpectrum:
    def setup(self):
        counts = [1, 2, 5, 6, 1, 7, 23, 2]
        self.binning = energy_logspace(1, 10, 9, "TeV")
        quality = [1, 1, 1, 0, 0, 1, 1, 1]
        self.spec = PHACountsSpectrum(
            data=counts,
            energy_lo=self.binning[:-1],
            energy_hi=self.binning[1:],
            quality=quality,
        )
        self.spec.backscal = 0.3
        self.spec.obs_id = 42
        self.spec.livetime = 3 * u.h


    def test_basic(self):
        assert "PHACountsSpectrum" in str(self.spec)
        assert_quantity_allclose(self.spec.lo_threshold, self.binning[3])
        assert_quantity_allclose(self.spec.hi_threshold, self.binning[5])

    def test_thresholds(self):
        self.spec.quality = np.zeros(self.spec.energy.nbin, dtype=int)
        self.spec.lo_threshold = 1.5 * u.TeV
        self.spec.hi_threshold = 4.5 * u.TeV
        assert (self.spec.quality == [1, 1, 0, 0, 0, 1, 1, 1]).all()
        assert_quantity_allclose(self.spec.lo_threshold, 1.778279410038922 * u.TeV)
        assert_quantity_allclose(self.spec.hi_threshold, 4.216965034285822 * u.TeV)

    def test_io(self, tmpdir):
        filename = tmpdir / "test2.fits"
        self.spec.write(filename)
        spec2 = PHACountsSpectrum.read(filename)
        assert_quantity_allclose(
            spec2.energy.edges * spec2.energy.unit,
            self.spec.energy.edges * self.spec.energy.unit,
        )

    def test_backscal_array(self):
        self.spec.backscal = np.arange(self.spec.energy.nbin)
        table = self.spec.to_table()
        assert table["BACKSCAL"][2] == 2

    def test_rebin(self):
        spec_rebinned = self.spec.rebin(2)
        assert (spec_rebinned.quality == [1, 0, 0, 1]).all()
        assert_quantity_allclose(spec_rebinned.hi_threshold, 5.623413251903491 * u.TeV)
        assert_quantity_allclose(spec_rebinned.lo_threshold, 1.778279410038922 * u.TeV)

    @requires_dependency("sherpa")
    def test_to_sherpa(self):
        sherpa_dataset = self.spec.to_sherpa("test")
        assert_allclose(sherpa_dataset.counts, self.spec.data)

    def test_reset_thresholds(self):
        self.spec.reset_thresholds()
        assert_allclose(self.spec.quality, 0.0)

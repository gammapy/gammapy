# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from ...utils.testing import (
    requires_dependency,
    mpl_plot_check,
    assert_quantity_allclose,
)
from ...utils.energy import energy_logspace
from .. import CountsSpectrum


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
            CountsSpectrum(data=self.counts, energy_lo=bins[:-1], energy_hi=bins[1:])

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

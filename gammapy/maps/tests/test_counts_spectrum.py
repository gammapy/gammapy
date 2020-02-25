# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from gammapy.maps import CountsSpectrum, MapAxis, WcsGeom
from gammapy.utils.regions import compound_region_to_list, make_region
from gammapy.utils.testing import (
    assert_quantity_allclose,
    mpl_plot_check,
    requires_dependency,
)


class TestCountsSpectrum:
    def setup(self):
        self.counts = [0, 0, 2, 5, 17, 3]
        self.bins = MapAxis.from_energy_bounds(1, 10, 6, "TeV").edges

        # Create region and associated wcs
        region = make_region("galactic;circle(0,1,0.5)")
        self.region = region.union(make_region("galactic;box(1,-0.25,1.2,3.5,30)"))
        self.wcs = WcsGeom.create(
            npix=500, binsz=0.01, skydir=(0, 0), frame="galactic"
        ).wcs

        self.spec = CountsSpectrum(
            data=self.counts,
            energy_lo=self.bins[:-1],
            energy_hi=self.bins[1:],
            region=self.region,
            wcs=self.wcs,
        )

    def test_wrong_init(self):
        bins = MapAxis.from_energy_bounds(1, 10, 8, "TeV").edges
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

    def test_io(self, tmp_path):
        self.spec.write(tmp_path / "tmp.fits")
        spec2 = CountsSpectrum.read(tmp_path / "tmp.fits")
        assert_quantity_allclose(spec2.energy.edges, self.bins)
        regions = compound_region_to_list(spec2.region)
        assert len(regions) == 2
        assert_allclose(regions[0].center.l.to_value("deg"), 0.0)
        assert_allclose(regions[0].radius.to_value("deg"), 0.5)
        assert_allclose(regions[1].center.b.to_value("deg"), -0.25)
        assert_allclose(regions[1].angle.to_value("deg"), 30)

    def test_downsample(self):
        rebinned_spec = self.spec.downsample(2)
        assert rebinned_spec.energy.nbin == self.spec.energy.nbin / 2
        assert rebinned_spec.data.shape[0] == self.spec.data.shape[0] / 2
        assert rebinned_spec.total_counts == self.spec.total_counts

        idx = rebinned_spec.energy.coord_to_idx([2, 3, 5] * u.TeV)
        actual = rebinned_spec.data[idx]
        desired = [0, 7, 20]
        assert (actual == desired).all()

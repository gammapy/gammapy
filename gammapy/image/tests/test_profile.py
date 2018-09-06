# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, mpl_plot_check
from ...maps import WcsNDMap, WcsGeom
from ..profile import compute_binning, ImageProfile, ImageProfileEstimator


@requires_dependency("pandas")
def test_compute_binning():
    data = [1, 3, 2, 2, 4]
    bin_edges = compute_binning(data, n_bins=3, method="equal width")
    assert_allclose(bin_edges, [1, 2, 3, 4])

    bin_edges = compute_binning(data, n_bins=3, method="equal entries")
    # TODO: create test-cases that have been verified by hand here!
    assert_allclose(bin_edges, [1, 2, 2.66666667, 4])


@pytest.fixture(scope="session")
def checkerboard_image():
    nxpix, nypix = 10, 6

    # set up data as a checkerboard of 0.5 and 1.5, so that the mean and sum
    # are not compeletely trivial to compute
    data = 1.5 * np.ones((nypix, nxpix))
    data[slice(0, nypix + 1, 2), slice(0, nxpix + 1, 2)] = 0.5
    data[slice(1, nypix + 1, 2), slice(1, nxpix + 1, 2)] = 0.5

    geom = WcsGeom.create(npix=(nxpix, nypix), coordsys="GAL", binsz=0.02)
    return WcsNDMap(geom=geom, data=data, unit="cm-2 s-1")


@pytest.fixture(scope="session")
def cosine_profile():
    table = Table()
    table["x_ref"] = np.linspace(-90, 90, 11) * u.deg
    table["profile"] = np.cos(table["x_ref"].to("rad")) * u.Unit("cm-2 s-1")
    table["profile_err"] = 0.1 * table["profile"]
    return ImageProfile(table)


@requires_dependency("scipy")
class TestImageProfileEstimator(object):
    def test_lat_profile_sum(self, checkerboard_image):
        p = ImageProfileEstimator(axis="lat", method="sum")
        profile = p.run(checkerboard_image)

        desired = 10 * np.ones(6) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    def test_lon_profile_sum(self, checkerboard_image):
        p = ImageProfileEstimator(axis="lon", method="sum")
        profile = p.run(checkerboard_image)

        desired = 6 * np.ones(10) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    def test_radial_profile_sum(self, checkerboard_image):
        center = SkyCoord(0, 0, unit="deg", frame="galactic")
        p = ImageProfileEstimator(axis="radial", method="sum", center=center)
        profile = p.run(checkerboard_image)

        desired = [4., 8., 20., 12., 12.] * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    def test_lat_profile_mean(self, checkerboard_image):
        p = ImageProfileEstimator(axis="lat", method="mean")
        profile = p.run(checkerboard_image)

        desired = np.ones(6) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    def test_lon_profile_mean(self, checkerboard_image):
        p = ImageProfileEstimator(axis="lon", method="mean")
        profile = p.run(checkerboard_image)

        desired = np.ones(10) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    def test_x_edges_lat(self, checkerboard_image):
        x_edges = Angle(np.linspace(-0.06, 0.06, 4), "deg")

        p = ImageProfileEstimator(x_edges=x_edges, axis="lat", method="sum")
        profile = p.run(checkerboard_image)

        desired = 20 * np.ones(3) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    def test_x_edges_lon(self, checkerboard_image):
        x_edges = Angle(np.linspace(-0.1, 0.1, 6), "deg")

        p = ImageProfileEstimator(x_edges=x_edges, axis="lon", method="sum")
        profile = p.run(checkerboard_image)

        desired = 12 * np.ones(5) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)


class TestImageProfile(object):
    def test_normalize(self, cosine_profile):
        normalized = cosine_profile.normalize(mode="integral")
        profile = normalized.profile
        assert_quantity_allclose(profile.sum(), 1 * u.Unit("cm-2 s-1"))

        normalized = cosine_profile.normalize(mode="peak")
        profile = normalized.profile
        assert_quantity_allclose(profile.max(), 1 * u.Unit("cm-2 s-1"))

    def test_profile_x_edges(self, cosine_profile):
        assert_quantity_allclose(cosine_profile.x_ref.sum(), 0 * u.deg)

    @requires_dependency("scipy")
    @pytest.mark.parametrize("kernel", ["gauss", "box"])
    def test_smooth(self, cosine_profile, kernel):
        # smoothing should preserve the mean
        desired_mean = cosine_profile.profile.mean()
        smoothed = cosine_profile.smooth(kernel, radius=3)

        assert_quantity_allclose(smoothed.profile.mean(), desired_mean)

        # smoothing should decrease errors
        assert smoothed.profile_err.mean() < cosine_profile.profile_err.mean()

    @requires_dependency("matplotlib")
    def test_peek(self, cosine_profile):
        with mpl_plot_check():
            cosine_profile.peek()

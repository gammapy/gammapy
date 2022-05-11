# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from gammapy.estimators import ImageProfile, ImageProfileEstimator
from gammapy.maps import WcsGeom, WcsNDMap
from gammapy.utils.testing import assert_quantity_allclose, mpl_plot_check


@pytest.fixture(scope="session")
def checkerboard_image():
    nxpix, nypix = 10, 6

    # set up data as a checkerboard of 0.5 and 1.5, so that the mean and sum
    # are not completely trivial to compute
    data = 1.5 * np.ones((nypix, nxpix))
    data[slice(0, nypix + 1, 2), slice(0, nxpix + 1, 2)] = 0.5
    data[slice(1, nypix + 1, 2), slice(1, nxpix + 1, 2)] = 0.5

    geom = WcsGeom.create(npix=(nxpix, nypix), frame="galactic", binsz=0.02)
    return WcsNDMap(geom=geom, data=data, unit="cm-2 s-1")


@pytest.fixture(scope="session")
def cosine_profile():
    table = Table()
    table["x_ref"] = np.linspace(-90, 90, 11) * u.deg
    table["profile"] = np.cos(table["x_ref"].to("rad")) * u.Unit("cm-2 s-1")
    table["profile_err"] = 0.1 * table["profile"]
    return ImageProfile(table)


class TestImageProfileEstimator:
    @staticmethod
    def test_lat_profile_sum(checkerboard_image):
        p = ImageProfileEstimator(axis="lat", method="sum")
        profile = p.run(checkerboard_image)

        desired = 10 * np.ones(6) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    @staticmethod
    def test_lon_profile_sum(checkerboard_image):
        p = ImageProfileEstimator(axis="lon", method="sum")
        profile = p.run(checkerboard_image)

        desired = 6 * np.ones(10) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    @staticmethod
    def test_radial_profile_sum(checkerboard_image):
        center = SkyCoord(0, 0, unit="deg", frame="galactic")
        p = ImageProfileEstimator(axis="radial", method="sum", center=center)
        profile = p.run(checkerboard_image)

        desired = [4.0, 8.0, 20.0, 12.0, 12.0] * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

        with pytest.raises(ValueError):
            ImageProfileEstimator(axis="radial")

    @staticmethod
    def test_lat_profile_mean(checkerboard_image):
        p = ImageProfileEstimator(axis="lat", method="mean")
        profile = p.run(checkerboard_image)

        desired = np.ones(6) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    @staticmethod
    def test_lon_profile_mean(checkerboard_image):
        p = ImageProfileEstimator(axis="lon", method="mean")
        profile = p.run(checkerboard_image)

        desired = np.ones(10) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    @staticmethod
    def test_x_edges_lat(checkerboard_image):
        x_edges = Angle(np.linspace(-0.06, 0.06, 4), "deg")

        p = ImageProfileEstimator(x_edges=x_edges, axis="lat", method="sum")
        profile = p.run(checkerboard_image)

        desired = 20 * np.ones(3) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)

    @staticmethod
    def test_x_edges_lon(checkerboard_image):
        x_edges = Angle(np.linspace(-0.1, 0.1, 6), "deg")

        p = ImageProfileEstimator(x_edges=x_edges, axis="lon", method="sum")
        profile = p.run(checkerboard_image)

        desired = 12 * np.ones(5) * u.Unit("cm-2 s-1")
        assert_quantity_allclose(profile.profile, desired)


class TestImageProfile:
    @staticmethod
    def test_normalize(cosine_profile):
        normalized = cosine_profile.normalize(mode="integral")
        profile = normalized.profile
        assert_quantity_allclose(profile.sum(), 1 * u.Unit("cm-2 s-1"))

        normalized = cosine_profile.normalize(mode="peak")
        profile = normalized.profile
        assert_quantity_allclose(profile.max(), 1 * u.Unit("cm-2 s-1"))

    @staticmethod
    def test_profile_x_edges(cosine_profile):
        assert_quantity_allclose(cosine_profile.x_ref.sum(), 0 * u.deg)

    @staticmethod
    @pytest.mark.parametrize("kernel", ["gauss", "box"])
    def test_smooth(cosine_profile, kernel):
        # smoothing should preserve the mean
        desired_mean = cosine_profile.profile.mean()
        smoothed = cosine_profile.smooth(kernel, radius=3)

        assert_quantity_allclose(smoothed.profile.mean(), desired_mean)

        # smoothing should decrease errors
        assert smoothed.profile_err.mean() < cosine_profile.profile_err.mean()

    @staticmethod
    def test_peek(cosine_profile):
        with mpl_plot_check():
            cosine_profile.peek()

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Table
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...datasets import FermiGalacticCenter
from ...image import SkyImage
from ..profile import compute_binning, image_profile, ImageProfile


@requires_dependency('pandas')
def test_compute_binning():
    data = [1, 3, 2, 2, 4]
    bin_edges = compute_binning(data, n_bins=3, method='equal width')
    assert_allclose(bin_edges, [1, 2, 3, 4])

    bin_edges = compute_binning(data, n_bins=3, method='equal entries')
    # TODO: create test-cases that have been verified by hand here!
    assert_allclose(bin_edges, [1, 2, 2.66666667, 4])


@requires_data('gammapy-extra')
def test_image_lat_profile():
    """Tests GLAT profile with image of 1s of known size and shape."""
    image = SkyImage.empty_like(FermiGalacticCenter.counts(), fill=1.)
    coordinates = image.coordinates()
    l = coordinates.data.lon
    b = coordinates.data.lat
    lons, lats = l.degree, b.degree

    counts = SkyImage.empty_like(FermiGalacticCenter.counts(), fill=1.)

    mask = np.zeros_like(image.data)
    # Select Full Image
    lat = [lats.min(), lats.max()]
    lon = [lons.min(), lons.max()]
    # Pick minimum valid binning
    binsz = 0.5
    mask_array = np.zeros_like(image.data, dtype='bool')
    # Test output
    lat_profile1 = image_profile('lat', image.to_image_hdu(), lat, lon, binsz, errors=True)
    # atol 0.1 is sufficient to check if correct number of pixels are included
    assert_allclose(lat_profile1.table['profile'].data.astype(float),
                    2000 * np.ones(39), rtol=1, atol=0.1)
    assert_allclose(lat_profile1.table['profile_err'].data,
                    0.1 * lat_profile1.table['profile'].data)

    lat_profile2 = image_profile('lat', image.to_image_hdu(), lat, lon, binsz,
                                 counts.to_image_hdu(), errors=True)
    # atol 0.1 is sufficient to check if correct number of pixels are included
    assert_allclose(lat_profile2.table['profile_err'].data,
                    44.721359549995796 * np.ones(39), rtol=1, atol=0.1)

    lat_profile3 = image_profile('lat', image.to_image_hdu(), lat, lon, binsz,
                                 counts.to_image_hdu(), mask_array, errors=True)

    assert_allclose(lat_profile3.table['profile'].data, np.zeros(39))


@requires_data('gammapy-extra')
def test_image_lon_profile():
    """Tests GLON profile with image of 1s of known size and shape."""
    image = FermiGalacticCenter.counts()

    coordinates = SkyImage.from_image_hdu(image).coordinates()
    lons = coordinates.galactic.l.wrap_at('180d')
    lats = coordinates.galactic.b
    lons = lons.degree
    lats = lats.degree
    image.data = np.ones_like(image.data)

    counts = FermiGalacticCenter.counts()
    counts.data = np.ones_like(counts.data)

    mask = np.zeros_like(image.data)
    # Select Full Image
    lat = [lats.min(), lats.max()]
    lon = [lons.min(), lons.max()]
    # Pick minimum valid binning
    binsz = 0.5
    mask_array = np.zeros_like(image.data)
    # Test output
    lon_profile1 = image_profile('lon', image, lat, lon, binsz,
                                 errors=True)
    # atol 0.1 is sufficient to check if correct number of pixels are included
    assert_allclose(lon_profile1.table['profile'].data.astype(float),
                    1000 * np.ones(79), rtol=1, atol=0.1)
    assert_allclose(lon_profile1.table['profile_err'].data,
                    0.1 * lon_profile1.table['profile'].data)

    lon_profile2 = image_profile('lon', image, lat, lon, binsz,
                                 counts, errors=True)
    # atol 0.1 is sufficient to check if correct number of pixels are included
    assert_allclose(lon_profile2.table['profile_err'].data,
                    31.622776601683793 * np.ones(79), rtol=1, atol=0.1)

    lon_profile3 = image_profile('lon', image, lat, lon, binsz, counts,
                                 mask_array, errors=True)

    assert_allclose(lon_profile3.table['profile'].data, np.zeros(79))


class TestImageProfile(object):
    def setup(self):
        table = Table()
        table['x_ref'] = np.arange(-90, 90, 10) * u.deg
        table['profile'] = np.cos(table['x_ref'].to('rad')) * u.Unit('cm-2 s-1')
        table['profile_err'] = 0.1 * table['profile']
        self.profile = ImageProfile(table)

    def test_normalize(self):
        normalized = self.profile.normalize(mode='integral')
        profile = normalized.profile
        assert_quantity_allclose(profile.sum(), 1 * u.Unit('cm-2 s-1'))

        normalized = self.profile.normalize(mode='peak')
        profile = normalized.profile
        assert_quantity_allclose(profile.max(), 1 * u.Unit('cm-2 s-1'))


    @requires_dependency('scipy')
    def test_smooth(self):
        desired = self.profile.profile.sum()
        smoothed = self.profile.smooth()
        assert_quantity_allclose(smoothed.profile.sum(), desired)

    @requires_dependency('matplotlib')
    def test_peek(self):
        self.profile.peek()
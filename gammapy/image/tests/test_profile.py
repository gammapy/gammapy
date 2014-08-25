# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...datasets import FermiGalacticCenter
from ...image import (coordinates,
                      compute_binning,
                      image_profile,
                      )

try:
    import pandas
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@pytest.mark.skipif('not HAS_PANDAS')
def test_compute_binning():
    data = [1, 3, 2, 2, 4]
    bin_edges = compute_binning(data, n_bins=3, method='equal width')
    assert_allclose(bin_edges, [1, 2, 3, 4])

    bin_edges = compute_binning(data, n_bins=3, method='equal entries')
    # TODO: create test-cases that have been verified by hand here!
    assert_allclose(bin_edges, [1,  2,  2.66666667,  4])


def test_image_lat_profile():
    """Tests GLAT profile with image of 1s of known size and shape."""
    image = FermiGalacticCenter.counts()

    lons, lats = coordinates(image)
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
    lat_profile1 = image_profile('lat', image, lat, lon, binsz, errors=True)
    # atol 0.1 is sufficient to check if correct number of pixels are included
    assert_allclose(lat_profile1['BIN_VALUE'].data.astype(float),
                    2000 * np.ones(39), rtol=1, atol=0.1)
    assert_allclose(lat_profile1['BIN_ERR'].data,
                    0.1 * lat_profile1['BIN_VALUE'].data)

    lat_profile2 = image_profile('lat', image, lat, lon, binsz,
                                 counts, errors=True)
    # atol 0.1 is sufficient to check if correct number of pixels are included
    assert_allclose(lat_profile2['BIN_ERR'].data,
                    44.721359549995796 * np.ones(39), rtol=1, atol=0.1)

    lat_profile3 = image_profile('lat', image, lat, lon, binsz, counts,
                                 mask_array, errors=True)

    assert_allclose(lat_profile3['BIN_VALUE'].data, np.zeros(39))


def test_image_lon_profile():
    """Tests GLON profile with image of 1s of known size and shape."""
    image = FermiGalacticCenter.counts()

    lons, lats = coordinates(image)
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
    assert_allclose(lon_profile1['BIN_VALUE'].data.astype(float),
                    1000 * np.ones(79), rtol=1, atol=0.1)
    assert_allclose(lon_profile1['BIN_ERR'].data,
                    0.1 * lon_profile1['BIN_VALUE'].data)

    lon_profile2 = image_profile('lon', image, lat, lon, binsz,
                                 counts, errors=True)
    # atol 0.1 is sufficient to check if correct number of pixels are included
    assert_allclose(lon_profile2['BIN_ERR'].data,
                    31.622776601683793 * np.ones(79), rtol=1, atol=0.1)

    lon_profile3 = image_profile('lon', image, lat, lon, binsz, counts,
                                 mask_array, errors=True)

    assert_allclose(lon_profile3['BIN_VALUE'].data, np.zeros(79))

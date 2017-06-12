# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy.wcs import WCS
from ...utils.testing import requires_dependency, requires_data
from ...datasets import FermiGalacticCenter
from ...image import (
    block_reduce_hdu,
    lon_lat_rectangle_mask,
    SkyImage,
)


@pytest.mark.xfail
def test_process_image_pixels():
    """Check the example how to implement convolution given in the docstring"""
    from astropy.convolution import convolve as astropy_convolve

    def convolve(image, kernel):
        """Convolve image with kernel"""
        from ..utils import process_image_pixels
        images = dict(image=np.asanyarray(image))
        kernel = np.asanyarray(kernel)
        out = dict(image=np.empty_like(image))

        def convolve_function(images, kernel):
            value = np.sum(images['image'] * kernel)
            return dict(image=value)

        process_image_pixels(images, kernel, out, convolve_function)
        return out['image']

    random_state = np.testing.RandomState(seed=0)

    image = random_state.uniform(size=(7, 10))
    kernel = random_state.uniform(size=(3, 5))
    actual = convolve(image, kernel)
    desired = astropy_convolve(image, kernel, boundary='fill')
    assert_allclose(actual, desired)


@requires_dependency('skimage')
class TestBlockReduceHDU:
    def setup_class(self):
        # Arbitrarily choose CAR projection as independent from tests
        projection = 'CAR'

        # Create test image
        self.image = SkyImage.empty(nxpix=12, nypix=8, proj=projection)
        self.image.data = np.ones(self.image.data.shape)
        self.image_hdu = self.image.to_image_hdu()

    @pytest.mark.parametrize('operation', list([np.sum, np.mean]))
    def test_image(self, operation):
        image_1 = block_reduce_hdu(self.image_hdu, (2, 4), func=operation)
        if operation == np.sum:
            ref1 = [[8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8]]
        if operation == np.mean:
            ref1 = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
        assert_allclose(image_1.data, ref1)


@requires_dependency('skimage')
def test_ref_pixel():
    image = SkyImage.empty(nxpix=101, nypix=101, proj='CAR')
    footprint = image.wcs.calc_footprint(center=False)
    image_1 = block_reduce_hdu(image.to_image_hdu(), (10, 10), func=np.sum)
    footprint_1 = WCS(image_1.header).calc_footprint(center=False)
    # Lower left corner shouldn't change
    assert_allclose(footprint[0], footprint_1[0])


@requires_data('gammapy-extra')
def test_lon_lat_rectangle_mask():
    counts = SkyImage.from_image_hdu(FermiGalacticCenter.counts())
    coordinates = counts.coordinates()
    lons = coordinates.data.lon.wrap_at('180d')
    lats = coordinates.data.lat
    mask = lon_lat_rectangle_mask(lons.degree, lats.degree, lon_min=-1,
                                  lon_max=1, lat_min=-1, lat_max=1)
    assert_allclose(mask.sum(), 400)

    mask = lon_lat_rectangle_mask(lons.degree, lats.degree, lon_min=None,
                                  lon_max=None, lat_min=None,
                                  lat_max=None)
    assert_allclose(mask.sum(), 80601)

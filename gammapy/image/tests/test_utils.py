# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import pytest
from .. import utils


def test_binary_disk():
    actual = utils.binary_disk(1)
    desired = np.array([[False, True, False],
                        [True, True, True],
                        [False, True, False]])
    assert_equal(actual, desired)


def test_binary_ring():
    actual = utils.binary_ring(1, 2)
    desired = np.array([[False, False, True, False, False],
                        [False, True, True, True, False],
                        [True, True, False, True, True],
                        [False, True, True, True, False],
                        [False, False, True, False, False]])
    assert_equal(actual, desired)


class TestImageCoordinates(object):

    def setup_class(self):
        self.image = utils.make_empty_image(nxpix=3, nypix=2,
                                            binsz=10, proj='CAR')
        self.image.data = np.arange(3 * 2).reshape(self.image.data.shape)

    def test_coordinates(self):
        lon, lat = utils.coordinates(self.image)
        x, y = utils.coordinates(self.image)
        lon_sym = utils.coordinates(self.image, lon_sym=True)[0]
        # TODO: assert

    def test_separation(self):
        separation = utils.separation(self.image, (1, 0))
        separation = utils.separation(self.image, (0, 90))
        # TODO: assert

    @pytest.mark.xfail
    def test_contains(self):
        # world coordinates
        assert utils.contains(self.image, 0, 0) == True
        assert utils.contains(self.image, 14.9, -9.9) == True
        assert utils.contains(self.image, 20, 0) ==  False
        assert utils.contains(self.image, 0, -15) == False

        # pixel coordinates
        assert utils.contains(self.image, 0.6, 0.6, world=False) == True
        assert utils.contains(self.image, 3.4, 2.4, world=False) == True
        assert utils.contains(self.image, 0.4, 0, world=False) == False
        assert utils.contains(self.image, 0, 2.6, world=False) == False

        # one-dimensional arrays
        x, y = np.arange(4), np.arange(4)
        inside = utils.contains(self.image, x, y, world=False)
        assert_equal(inside, np.array([False, True, True, False]))

        # two-dimensional arrays
        x = y = np.zeros((3, 2))
        inside = utils.contains(self.image, x, y)
        assert_equal(inside, np.ones((3, 2), dtype=bool))


    def test_image_area(self):
        actual = utils.solid_angle(self.image)
        expected = 99.61946869
        assert_allclose(actual, expected)


@pytest.mark.xfail
def test_process_image_pixels():
    """Check the example how to implement convolution given in the docstring"""
    from astropy.convolution import convolve as astropy_convolve
    
    def convolve(image, kernel):
        '''Convolve image with kernel'''
        from ..utils import process_image_pixels
        images = dict(image=np.asanyarray(image))
        kernel = np.asanyarray(kernel)
        out = dict(image=np.empty_like(image))
        def convolve_function(images, kernel):
            value = np.sum(images['image'] * kernel)
            return dict(image=value)
        process_image_pixels(images, kernel, out, convolve_function)
        return out['image']

    np.random.seed(0)
    image = np.random.random((7, 10))
    kernel = np.random.random((3, 5))
    actual = convolve(image, kernel)
    desired = astropy_convolve(image, kernel, boundary='fill')
    assert_allclose(actual, desired)


def test_solid_angle():
    from astropy.units import Quantity
    nxpix, nypix, binsz = 50, 10, 0.1
    image = utils.make_empty_image(nxpix=nxpix, nypix=nypix, binsz=binsz)
    actual = utils.solid_angle(image, method='2').sum()
    expected = Quantity(nxpix * nypix * binsz ** 2, 'deg^2').to('sr').value
    assert_allclose(actual, expected, rtol=1e-4)

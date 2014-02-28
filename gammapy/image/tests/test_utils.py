# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import pytest
from astropy.io import fits
from astropy.wcs import WCS
from .. import utils

try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


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
    
@pytest.mark.skipif('not HAS_SKIMAGE')
class test_block_reduce_hdu():    

    @pytest.mark.parametrize(('projection'), list(['AIT', 'CAR']))
    def setup_class(self, projection):
        # Create test image
        self.image = utils.make_empty_image(5, 4, proj=projection)
        self.image.data = np.ones(self.image.data.shape)
        self.footprint = utils.calc_footprint(self.image.header)
        # Create test cube
        self.indices = np.arange(4)
        self.cube_images = []
        for index in self.indices:
            layer = np.ones(self.image.data.shape)
            self.cube_images.append(layer)
        self.image.data = self.cube_images
    
    @pytest.mark.parametrize(('operation'), list([np.sum, np.mean]))
    def test_image(self, operation):
        image_1 = utils.block_reduce_hdu(self.image, (2, 2), func=operation)
        footprint_1 = utils.calc_footprint(image_1.header)
        if operation == np.sum:
            ref1 = [[4, 4, 2], [4, 4, 2]]
        if operation == np.mean:
            ref1 = [[1, 1, 0.5], [1, 1, 0.5]]
        assert_allclose(image_1.data, ref1)
        assert_allclose(self.footprint, footprint_1)
    
    @pytest.mark.parametrize(('operation'), list([np.sum, np.mean]))
    def test_cube(self, operation):
        for index in self.indices:
            layer = self.image.data[index]
            layer_hdu = fits.ImageHDU(data=layer, header=self.image.header)
            image_1 = utils.block_reduce_hdu(layer_hdu, (2, 2), func=operation)
            footprint_1 = utils.calc_footprint(image_1.header)
            if operation == np.sum:
                ref1 = [[4, 4, 2], [4, 4, 2]]
            if operation == np.mean:
                ref1 = [[1, 1, 0.5], [1, 1, 0.5]]
            assert_allclose(image_1.data, ref1)
            assert_allclose(self.footprint, footprint_1)
            
@pytest.mark.parametrize(('projection'), list(['AIT', 'CAR']))            
def test_calc_footprint(projection):    
    image = utils.make_empty_image(50, 80, proj=projection)
    image.data = np.ones(image.data.shape)
    footprint = utils.calc_footprint(image.header)
    if projection == 'CAR':
        # Check values determined from separately generated fits file (using image.WCS.wcs.coordinates)
        assert_allclose(footprint[0], [ 2.45, -3.95])
        assert_allclose(footprint[1], [ 2.45, 3.95])
        assert_allclose(footprint[2], [ 357.55, 3.95])
        assert_allclose(footprint[3], [ 357.55, -3.95])  
    if projection == 'AIT':
        # Check values determined from separately generated fits file (using image.WCS.wcs.coordinates)
        assert_allclose(footprint[0], [ 2.45442318, -3.95055627])
        assert_allclose(footprint[1], [ 2.45442318, 3.95055627])
        assert_allclose(footprint[2], [ 357.54557682, 3.95055627])
        assert_allclose(footprint[3], [ 357.54557682, -3.95055627])  

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
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
    assert_almost_equal(actual, desired)

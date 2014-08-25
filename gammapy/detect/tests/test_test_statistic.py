# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from time import time
import numpy as np
from astropy.tests.helper import pytest
from ...detect import ts_image, TSMapCalculator


def make_test_images(shape=(100, 200)):
    np.random.seed(0)
    background = 10. * np.ones(shape)
    excess = 1
    images = dict()
    images['background'] = background + excess
    images['counts'] = np.random.poisson(images['background'])
    return images


def make_test_kernel(shape=(11, 11)):
    kernel = np.ones(shape=shape)
    return kernel


@pytest.mark.xfail
def test_simple():
    """A simple test case"""
    images = make_test_images(shape=(50, 70))
    kernel = make_test_kernel(shape=(11, 11))
    ts_results = ts_image(images, kernel)
    images['ts'] = ts_results['ts']
    images['ncalls'] = ts_results['ncalls']
    print('mean ncalls: {0}'.format(images['ncalls'].mean()))
    images['kernel'] = kernel
    from astropy.io import fits
    hdu_list = fits.HDUList()
    for name, data in images.items():
        hdu = fits.ImageHDU(data=data, name=name)
        hdu_list.append(hdu)
    hdu_list.writeto('images.fits', clobber=True)


@pytest.mark.xfail
def test_optimizers():
    """Compare speed for a few different optimizers"""
    optimizers = ['migrad', 'fmin']
    start_values = ['last', 'estimate']

    images = make_test_images(shape=(30, 50))
    kernel = make_test_kernel(shape=(5, 5))

    for optimizer in optimizers:
        for start_value in start_values:
            tsmc = TSMapCalculator(images, kernel,
                                   optmizier=optimizer,
                                   start_value=start_value)
            tsmc.run()
            tsmc.report()


@pytest.mark.xfail
def test_speed():
    image_sizes = [100, 200]
    kernel_sizes = [1, 11, 21]
    for image_size in image_sizes:
        for kernel_size in kernel_sizes:
            images = make_test_images(shape=(image_size, image_size))
            kernel = make_test_kernel(shape=(kernel_size, kernel_size))
            t = time()
            ts = ts_image(images, kernel)
            t = time() - t
            # Compute speed = 1e3 image pixels per second (kpps)
            kpix = 1e-3 * images.values()[0].size
            speed = kpix / t
            print('image: {image_size:5d}, kernel: {kernel_size:5d}, speed (kpps): {speed:10.1f}'
                  ''.format(**locals()))

"""
Not what I expected: speed doesn't go down with number of pixels in the kernel
image:   100, kernel:     1, speed (kpps):       13.6
image:   100, kernel:    11, speed (kpps):       10.2
image:   100, kernel:    21, speed (kpps):        8.5
image:   100, kernel:    31, speed (kpps):        6.8
image:   200, kernel:     1, speed (kpps):       13.7
image:   200, kernel:    11, speed (kpps):       10.2
image:   200, kernel:    21, speed (kpps):        8.3
image:   200, kernel:    31, speed (kpps):        6.5
image:   300, kernel:     1, speed (kpps):       13.7
image:   300, kernel:    11, speed (kpps):       10.2
image:   300, kernel:    21, speed (kpps):        8.2
image:   300, kernel:    31, speed (kpps):        6.5
"""

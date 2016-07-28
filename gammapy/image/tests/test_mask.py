# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.convolution import Box2DKernel
from regions import CirclePixelRegion, PixCoord
from .. import SkyMask
from ...utils.testing import requires_dependency
from .. import SkyMask


@requires_dependency('scipy')
def test_random_creation():
    exclusion = SkyMask.empty(nxpix=300, nypix=100)
    exclusion.fill_random_circles(n=6, max_rad=10)
    assert exclusion.data.shape[0] == 100

    excluded = np.where(exclusion.data == 0)
    assert excluded[0].size != 0


@requires_dependency('scipy')
def test_distance_image():
    mask = SkyMask.empty(nxpix=3, nypix=2)
    distance = mask.distance_image.data
    assert_allclose(distance, -1e10)

    mask = SkyMask.empty(nxpix=3, nypix=2, fill=1.)
    distance = mask.distance_image.data
    assert_allclose(distance, 1e10)

    data = np.array([[0., 0., 1.], [1., 1., 1.]])
    mask = SkyMask(data=data)
    distance = mask.distance_image.data
    expected = [[-1, -1, 1], [1, 1, 1.41421356]]
    assert_allclose(distance, expected)


@requires_dependency('scipy')
def test_open():
    mask = SkyMask.empty(nxpix=5, nypix=5)
    mask.data[2:3, 2:3] = 1
    structure = Box2DKernel(3).array
    mask = mask.open(structure)
    assert_equal(mask.data, 0)


@requires_dependency('scipy')
def test_close():
    mask = SkyMask.empty(nxpix=5, nypix=5)
    mask.data[1:-1, 1:-1] = 1
    mask.data[2, 2] = 0
    structure = Box2DKernel(3).array
    mask = mask.close(structure)
    assert mask.data.sum() == 9


@requires_dependency('scipy')
def test_erode():
    mask = SkyMask.empty(nxpix=5, nypix=5)
    mask.data[1:-1, 1:-1] = 1
    structure = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
    mask = mask.erode(structure)
    assert mask.data[2, 2] == 1
    assert mask.data.sum() == 1


@requires_dependency('scipy')
def test_dilate():
    mask = SkyMask.empty(nxpix=5, nypix=5)
    mask.data[2, 2] = 1
    structure = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
    mask = mask.dilate(structure)
    assert mask.data.sum() == 9


def test_fill_region():
    region = CirclePixelRegion(center=PixCoord(x=2, y=1), radius=2)
    mask = SkyMask.empty(nxpix=5, nypix=4, fill=0)
    mask.fill_region(region)

    expected_result = np.zeros((4, 5))
    expected_result[:-1, 1:-1] = 1
    assert_equal(mask.data, expected_result)

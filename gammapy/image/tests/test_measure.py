# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.tests.helper import pytest
from .. import measure

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def generate_example_image():
    """
    Generate some greyscale image to run the detection on.
    """
    x = y = np.linspace(0, 3 * np.pi, 100)
    X, Y = np.meshgrid(x, y)
    image = X * Y * np.sin(X) ** 2 * np.sin(Y) ** 2

    return image


@pytest.mark.skipif('not HAS_SCIPY')
def test_measure():
    image = generate_example_image()
    labels = np.zeros_like(image, dtype=int)
    labels[10:20, 20:30] = 1
    results = measure.measure_labeled_regions(image, labels)

    # TODO: check output!


class _TestImageCoordinates(object):

    def setUp(self):
        self.image = utils.make_empty_image(nxpix=3, nypix=2,
                                            binsz=10, proj='CAR')
        self.image.dat = np.arange(3 * 2).reshape(self.image.dat.shape)

    def test_lookup(self):
        self.assertEqual(utils.lookup(self.image, 1, 1, world=False), 0)
        assert_equal(utils.lookup(self.image, 5, 1, world=False), np.nan)
        self.assertEqual(utils.lookup(self.image, 3, 2, world=False), 5)
        assert_equal(utils.lookup(self.image, [1, 5], [1, 1], world=False), [0, np.nan])

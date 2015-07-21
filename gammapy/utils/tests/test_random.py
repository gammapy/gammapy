# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
from ..random import sample_sphere, sample_powerlaw, sample_sphere_distance


def test_sample_sphere():
    np.random.seed(0)
    lon, lat = sample_sphere(size=2)
    assert_allclose(lon, Quantity([3.44829694, 4.49366732], 'radian'))
    assert_allclose(lat, Quantity([0.20700192, 0.08988736], 'radian'))


def test_sample_powerlaw():
    np.random.seed(0)
    x = sample_powerlaw(x_min=0.1, x_max=10, gamma=2, size=2)
    assert_allclose(x, [0.21897428, 0.34250971])


def test_sample_sphere_distance():
    np.random.seed(0)
    x = sample_sphere_distance(distance_min=0.1, distance_max=42, size=2)
    assert_allclose(x, [34.386731, 37.559774])

    x = sample_sphere_distance(distance_min=0.1, distance_max=42, size=1e3)
    assert x.min() >= 0.1
    assert x.max() <= 42

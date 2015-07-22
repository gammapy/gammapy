# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from astropy.tests.helper import assert_quantity_allclose
from ..random import sample_sphere, sample_powerlaw, sample_sphere_distance


def test_sample_sphere():

    # test general case
    np.random.seed(0)
    lon, lat = sample_sphere(size=2)
    assert_quantity_allclose(lon, Angle([3.44829694, 4.49366732], 'radian'))
    assert_quantity_allclose(lat, Angle([0.20700192, 0.08988736], 'radian'))

    # test specify a limited range
    lon_min = Angle(40., 'degree')
    lon_max = Angle(45., 'degree')
    lat_min = Angle(10., 'degree')
    lat_max = Angle(15., 'degree')
    lon, lat = sample_sphere(size=10,
                             lon_range=Angle([lon_min, lon_max]),
                             lat_range=Angle([lat_min, lat_max]))
    assert (lon_min <= lon).all() & (lon < lon_max).all()
    assert (lat_min <= lat).all() & (lat < lat_max).all()

    # test long in (-180, 180) deg range
    lon_min = Angle(-40., 'degree')
    lon_max = Angle(0., 'degree')
    lon, lat = sample_sphere(size=10, lon_range=Angle([lon_min, lon_max]))
    assert (lon_min <= lon).all() & (lon < lon_max).all()
    lat_min = Angle(-90., 'degree')
    lat_max = Angle(90., 'degree')
    assert (lat_min <= lat).all() & (lat < lat_max).all()


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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle, Longitude, Latitude
from astropy.tests.helper import assert_quantity_allclose
from ..random import (sample_sphere, sample_powerlaw,
                      sample_sphere_distance, check_random_state)


def test_sample_sphere():

    # initialise random number generator
    rng = check_random_state(0)

    # test general case
    lon, lat = sample_sphere(size=2, random_state=rng)
    assert_quantity_allclose(lon, Longitude([3.44829694, 4.49366732], 'radian'))
    assert_quantity_allclose(lat, Latitude([0.20700192, 0.08988736], 'radian'))

    # test specify a limited range
    lon_range = Longitude([40., 45.], 'degree')
    lat_range = Latitude([10., 15.], 'degree')
    lon, lat = sample_sphere(size=10,
                             lon_range=lon_range,
                             lat_range=lat_range,
                             random_state=rng)
    assert ((lon_range[0] <= lon) & (lon < lon_range[1])).all()
    assert ((lat_range[0] <= lat) & (lat < lat_range[1])).all()

    # test lon within (-180, 180) deg range
    lon_range = Longitude([-40., 0.], 'degree', wrap_angle=Angle(180., 'degree'))
    lon, lat = sample_sphere(size=10, lon_range=lon_range, random_state=rng)
    assert ((lon_range[0] <= lon) & (lon < lon_range[1])).all()
    lat_range = Latitude([-90., 90.], 'degree')
    assert ((lat_range[0] <= lat) & (lat < lat_range[1])).all()

    # test lon range explicitly (0, 360) deg
    epsilon = 1.e-8
    lon_range = Longitude([0., 360.-epsilon], 'degree')
    lon, lat = sample_sphere(size=100, lon_range=lon_range, random_state=rng)
    angle_0 = Angle(0., 'degree')
    angle_360 = Angle(360., 'degree')
    angle_m90 = Angle(-90., 'degree')
    angle_90 = Angle(90., 'degree')
    # test values in the desired range
    assert ((angle_0 <= lon) & (lon < angle_360)).all()
    assert ((angle_m90 <= lat) & (lat < angle_90)).all()
    # test if values are distributed along the whole range
    nbins = 4
    angle_delta_0_360 = (angle_360 - angle_0)/nbins
    angle_delta_m90_90 = (angle_90 - angle_m90)/nbins
    for i in np.arange(nbins):
        assert ((angle_0 + i*angle_delta_0_360 <= lon) &
                (lon < angle_0 + (i + 1)*angle_delta_0_360)).any()
        assert ((angle_m90 + i*angle_delta_m90_90 <= lat) &
                (lat < angle_m90 + (i + 1)*angle_delta_m90_90)).any()


def test_sample_powerlaw():
    # initialise random number generator
    rng = check_random_state(0)

    x = sample_powerlaw(x_min=0.1, x_max=10, gamma=2, size=2, random_state=rng)
    assert_allclose(x, [0.21897428, 0.34250971])


def test_sample_sphere_distance():
    # initialise random number generator
    rng = check_random_state(0)

    x = sample_sphere_distance(distance_min=0.1, distance_max=42, size=2, random_state=rng)
    assert_allclose(x, [34.386731, 37.559774])

    x = sample_sphere_distance(distance_min=0.1, distance_max=42, size=1e3, random_state=rng)
    assert x.min() >= 0.1
    assert x.max() <= 42

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from ...utils.testing import assert_quantity_allclose
from ..random import sample_sphere, sample_powerlaw, sample_sphere_distance


def test_sample_sphere():
    random_state = np.random.RandomState(seed=0)

    # test general case
    lon, lat = sample_sphere(size=2, random_state=random_state)
    assert_quantity_allclose(lon, Angle([3.44829694, 4.49366732], "radian"))
    assert_quantity_allclose(lat, Angle([0.20700192, 0.08988736], "radian"))

    # test specify a limited range
    lon_range = Angle([40., 45.], "deg")
    lat_range = Angle([10., 15.], "deg")
    lon, lat = sample_sphere(
        size=10, lon_range=lon_range, lat_range=lat_range, random_state=random_state
    )
    assert ((lon_range[0] <= lon) & (lon < lon_range[1])).all()
    assert ((lat_range[0] <= lat) & (lat < lat_range[1])).all()

    # test lon within (-180, 180) deg range
    lon_range = Angle([-40., 0.], "deg")
    lon, lat = sample_sphere(size=10, lon_range=lon_range, random_state=random_state)
    assert ((lon_range[0] <= lon) & (lon < lon_range[1])).all()
    lat_range = Angle([-90., 90.], "deg")
    assert ((lat_range[0] <= lat) & (lat < lat_range[1])).all()

    # test lon range explicitly (0, 360) deg
    lon_range = Angle([0., 360.], "deg")
    lon, lat = sample_sphere(size=100, lon_range=lon_range, random_state=random_state)
    # test values in the desired range
    lat_range = Angle([-90., 90.], "deg")
    assert ((lon_range[0] <= lon) & (lon < lon_range[1])).all()
    assert ((lat_range[0] <= lat) & (lat < lat_range[1])).all()
    # test if values are distributed along the whole range
    nbins = 4
    lon_delta = (lon_range[1] - lon_range[0]) / nbins
    lat_delta = (lat_range[1] - lat_range[0]) / nbins
    for i in np.arange(nbins):
        assert (
            (lon_range[0] + i * lon_delta <= lon)
            & (lon < lon_range[0] + (i + 1) * lon_delta)
        ).any()
        assert (
            (lat_range[0] + i * lat_delta <= lat)
            & (lat < lat_range[0] + (i + 1) * lat_delta)
        ).any()

    # test lon range explicitly (-180, 180) deg
    lon_range = Angle([-180., 180.], "deg")
    lon, lat = sample_sphere(size=100, lon_range=lon_range, random_state=random_state)
    # test values in the desired range
    lat_range = Angle([-90., 90.], "deg")
    assert ((lon_range[0] <= lon) & (lon < lon_range[1])).all()
    assert ((lat_range[0] <= lat) & (lat < lat_range[1])).all()
    # test if values are distributed along the whole range
    nbins = 4
    lon_delta = (lon_range[1] - lon_range[0]) / nbins
    lat_delta = (lat_range[1] - lat_range[0]) / nbins
    for i in np.arange(nbins):
        assert (
            (lon_range[0] + i * lon_delta <= lon)
            & (lon < lon_range[0] + (i + 1) * lon_delta)
        ).any()
        assert (
            (lat_range[0] + i * lat_delta <= lat)
            & (lat < lat_range[0] + (i + 1) * lat_delta)
        ).any()

    # test box around Galactic center
    lon_range = Angle([-5., 5.], "deg")
    lon, lat = sample_sphere(size=10, lon_range=lon_range, random_state=random_state)
    # test if values are distributed along the whole range
    nbins = 2
    lon_delta = (lon_range[1] - lon_range[0]) / nbins
    for i in np.arange(nbins):
        assert (
            (lon_range[0] + i * lon_delta <= lon)
            & (lon < lon_range[0] + (i + 1) * lon_delta)
        ).any()

    # test box around Galactic anticenter
    lon_range = Angle([175., 185.], "deg")
    lon, lat = sample_sphere(size=10, lon_range=lon_range, random_state=random_state)
    # test if values are distributed along the whole range
    nbins = 2
    lon_delta = (lon_range[1] - lon_range[0]) / nbins
    for i in np.arange(nbins):
        assert (
            (lon_range[0] + i * lon_delta <= lon)
            & (lon < lon_range[0] + (i + 1) * lon_delta)
        ).any()


def test_sample_powerlaw():
    random_state = np.random.RandomState(seed=0)

    x = sample_powerlaw(x_min=0.1, x_max=10, gamma=2, size=2, random_state=random_state)
    assert_allclose(x, [0.14886601, 0.1873559])


def test_sample_sphere_distance():
    random_state = np.random.RandomState(seed=0)

    x = sample_sphere_distance(
        distance_min=0.1, distance_max=42, size=2, random_state=random_state
    )
    assert_allclose(x, [34.386731, 37.559774])

    x = sample_sphere_distance(
        distance_min=0.1, distance_max=42, size=int(1e3), random_state=random_state
    )
    assert x.min() >= 0.1
    assert x.max() <= 42

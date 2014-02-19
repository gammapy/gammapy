# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from .. import spherical

def test_separation():
    assert_allclose(spherical.separation(0, 0, 180, 0), 180)
    assert_allclose(spherical.separation(270, 0, 180, 0), 90)
    assert_allclose(spherical.separation(0, 0, 0, 90), 90)
    assert_allclose(spherical.separation(0, 89, 180, 89), 2)


def test_minimum_separation():
    lon1 = [0, 1, 1]
    lat1 = [0, 0, 1]
    lon2 = [1, 1]
    lat2 = [0, 0.5]
    separation = spherical.minimum_separation(lon1, lat1, lon2, lat2)
    assert_allclose(separation, [1, 0, 0.5])


def test_pair_correlation():
    pass


def test_pixel_solid_angle():
    corners = []
    corners.append(dict(lon=0, lat=0))
    corners.append(dict(lon=1, lat=0))
    corners.append(dict(lon=1, lat=1))
    corners.append(dict(lon=0, lat=1))
    solid_angle = spherical.pixel_solid_angle(corners, method='2')
    assert_allclose(solid_angle, 0.8617784049059853)

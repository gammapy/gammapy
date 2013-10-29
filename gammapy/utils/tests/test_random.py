# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from .. import random 
from numpy.testing import assert_allclose

def test_sample_sphere():
    np.random.seed(0)
    lon, lat = random.sample_sphere(size=2)
    assert_allclose(lon, [ 3.44829694,  4.49366732])
    assert_allclose(lat, [ 0.20700192,  0.08988736])

def test_sample_powerlaw():
    np.random.seed(0)
    x = random.sample_powerlaw(x_min=0.1, x_max=10, gamma=2, size=2)
    assert_allclose(x, [ 0.21897428,  0.34250971])

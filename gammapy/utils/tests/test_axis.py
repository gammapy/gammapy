# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u
from ..axis import sqrt_space


def test_sqrt_space():
    theta_min=0
    theta_max=2.5
    theta_bin=100
    offset = sqrt_space(theta_min, theta_max, theta_bin) * u.deg
    assert_allclose(len(offset), theta_bin)
    assert_allclose(offset[0].value, theta_min)
    assert_allclose(offset[-1].value, theta_max)
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from ..axis import sqrt_space



def test_sqrt_space():
    theta_min=0
    theta_max=2.5
    theta_bin=100
    offset = sqrt_space(theta_min, theta_max, theta_bin) * u.deg
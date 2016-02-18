# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Axis method
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle




def sqrt_space(start, stop, num):
    """
    Define a square root binning
    From a linspace distribution in the square of the value, you define a square root distribution in theta

    Parameters
    ----------
    start : float
    stop : float
    num : int

    Returns
    -------
    tab = `~numpy.ndarray`
        1D array with a square root scale
    """
    tab2=np.linspace(start, stop, num)
    tab=np.sqrt(tab2)
    return tab
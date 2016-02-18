# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Axis method
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def sqrt_space(start, stop, num):
    """Return numbers spaced evenly on a square root scale.

    This function is similar to `numpy.linspace` and `numpy.logspace`.

    Parameters
    ----------
    start : float
        start is the starting value of the sequence
    stop : float
        stop is the final value of the sequence
    num : int
        Number of samples to generate.

    Returns
    -------
    samples : `~numpy.ndarray`
        1D array with a square root scale

    Examples
    --------
    >>> from gammapy.utils.axis import sqrt_space
    >>> samples = sqrt_space(0, 2, 5)
    array([ 0.        ,  1.        ,  1.41421356,  1.73205081,  2.        ])

    """
    samples2 = np.linspace(start ** 2, stop ** 2, num)
    samples = np.sqrt(samples2)
    return samples

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions to deal with arrays and quantities.
"""
from __future__ import print_function, division
import numpy as np

__all__ = ['array_stats_str']


def array_stats_str(x, label=''):
    """Make a string summarising some stats for an array.

    Parameters
    ----------
    x : array-like
        Array
    label : str, optional
        Label

    Returns
    -------
    stats_str : str
        String with array stats
    """
    x = np.asanyarray(x)

    ss = ''
    if label:
        ss += '{0:15s}: '.format(label)

    min = x.min()
    max = x.max()
    size = x.size

    fmt = 'size = {size:5d}, min = {min:6.3f}, max = {max:6.3f}\n'
    ss += fmt.format(**locals())

    return ss

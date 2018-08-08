# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions to deal with arrays and quantities."""
from __future__ import absolute_import, division, print_function, unicode_literals
from ..extern import six
import numpy as np

__all__ = ['array_stats_str',
           'shape_2N',
           'shape_divisible_by',
           'symmetric_crop_pad_width']

all_integer_types = six.integer_types + (np.integer,)


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


def shape_2N(shape, N=3):
    """
    Round a given shape to values that are divisible by 2^N.

    Parameters
    ----------
    shape : tuple
        Input shape.
    N : int (default = 3), optional
        Exponent of two.

    Returns
    -------
    new_shape : Tuple
        New shape extended to integers divisible by 2^N
    """
    shape = np.array(shape)
    new_shape = shape + (2 ** N - np.mod(shape, 2 ** N))
    return tuple(new_shape)


def shape_divisible_by(shape, factor):
    """
    Round a given shape to values that are divisible by factor.

    Parameters
    ----------
    shape : tuple
        Input shape.
    factor : int
        Divisor.

    Returns
    -------
    new_shape : Tuple
        New shape extended to integers divisible by factor
    """
    shape = np.array(shape)
    new_shape = shape + (shape % factor)
    return tuple(new_shape)


def symmetric_crop_pad_width(shape, new_shape):
    """
    Compute symmetric crop or pad width to obtain a new shape from a given old
    shape of an array.

    Parameters
    ----------
    shape : tuple
        Old shape
    new_shape : tuple or str
        New shape
    """
    xdiff = abs(shape[1] - new_shape[1])
    ydiff = abs(shape[0] - new_shape[0])

    if (np.array([xdiff, ydiff]) % 2).any():
        raise ValueError('For symmetric crop / pad width, difference to new shape '
                         'must be even in all axes.')

    ywidth = (ydiff // 2, ydiff // 2)
    xwidth = (xdiff // 2, xdiff // 2)
    return ywidth, xwidth


def _is_int(val):
    return isinstance(val, all_integer_types)

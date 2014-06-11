# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.units import Quantity

__all__ = ['assert_quantity']


def assert_quantity(actual, desired, *args, **kwargs):
    """Assert value and unit for astropy Quantity.

    Calls `numpy.testing.assert_allclose`.
    """
    if not isinstance(actual, Quantity):
        raise ValueError("actual must be a Quantity object.")
    if not isinstance(desired, Quantity):
        raise ValueError("actual must be a Quantity object.")

    assert actual.unit == desired.unit
    assert_allclose(actual, desired, *args, **kwargs)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from ..modeling import Parameter


def test_parameter_init():
    par = Parameter('spam', 42, 'deg')
    assert par.name == 'spam'
    assert par.value == 42
    assert par.unit == 'deg'
    assert par.min is np.nan
    assert par.max is np.nan
    assert par.frozen is False


def test_parameter_repr():
    par = Parameter('spam', 42, 'deg')
    assert repr(par).startswith('Parameter(name=')

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from ..modeling import Parameter, ParameterList


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


def test_parameter_list():
    pars = ParameterList([
        Parameter('spam', 42, 'deg'),
        Parameter('ham', 99, 'TeV'),
    ])
    # This applies a unit transformation
    pars.set_parameter_errors({
        'ham': '10000 GeV',
    })
    pars.set_error(0, 0.1)
    assert_allclose(pars.covariance, [[1e-2, 0], [0, 100]])
    assert_allclose(pars.error('spam'), 0.1)
    assert_allclose(pars.error(1), 10)

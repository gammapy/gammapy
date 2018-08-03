# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..modeling import Parameter, ParameterList


def test_parameter_init():
    par = Parameter('model', 'spam', 42, 'deg')
    assert par.name == 'spam'
    assert par.modelname == 'model'
    assert par.fullname == 'model.spam'
    assert par.value == 42
    assert par.unit == 'deg'
    assert par.min is np.nan
    assert par.max is np.nan
    assert par.frozen is False


def test_parameter_repr():
    par = Parameter('model', 'spam', 42, 'deg')
    assert repr(par).startswith('Parameter(modelname=')


def test_parameter_list():
    pars = ParameterList([
        Parameter('model', 'spam', 42, 'deg'),
        Parameter('model', 'ham', 99, 'TeV'),
    ])
    # This applies a unit transformation
    pars.set_parameter_errors({
        'model.ham': '10000 GeV',
    })
    assert_allclose(pars.covariance, [[0, 0], [0, 100]])
    assert_allclose(pars.error('model.spam'), 0)
    assert_allclose(pars.error('model.ham'), 10)

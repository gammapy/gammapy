# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ...extern import six
from ..modeling import Parameter, ParameterList


def test_parameter_init():
    par = Parameter('spam', 42, 'deg')
    assert par.name == 'spam'
    assert par.factor == 42
    assert par.scale == 1
    assert par.value == 42
    assert par.unit == 'deg'
    assert par.min is np.nan
    assert par.max is np.nan
    assert par.frozen is False

    par = Parameter('spam', '42 deg')
    assert par.factor == 42
    assert par.scale == 1
    assert par.unit == 'deg'

    with pytest.raises(TypeError):
        Parameter(1, 2)

    p = Parameter('spam', 42)
    with pytest.raises(TypeError):
        p.factor = '99'
    with pytest.raises(TypeError):
        p.scale = '99'


def test_parameter_value():
    par = Parameter('spam', 42, 'deg', 10)

    value = par.value
    assert value == 420

    par.value = 70
    assert par.scale == 10
    assert_allclose(par.factor, 7)


def test_parameter_quantity():
    par = Parameter('spam', 42, 'deg', 10)

    quantity = par.quantity
    assert quantity.unit == 'deg'
    assert quantity.value == 420

    par.quantity = '70 deg'
    assert_allclose(par.factor, 7)
    assert par.scale == 10
    assert par.unit == 'deg'


def test_parameter_repr():
    par = Parameter('spam', 42, 'deg')
    assert repr(par).startswith('Parameter(name=')


def test_parameter_to_dict():
    par = Parameter('spam', 42, 'deg')
    d = par.to_dict()
    assert isinstance(d['unit'], six.string_types)


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

    pars.optimiser_rescale_parameters()
    assert_allclose(pars['spam'].factor, 1)
    assert_allclose(pars['spam'].scale, 42)
    assert_allclose(pars['ham'].factor, 1)
    assert_allclose(pars['ham'].scale, 99)
    assert_allclose(pars.covariance, [[1e-2, 0], [0, 100]])

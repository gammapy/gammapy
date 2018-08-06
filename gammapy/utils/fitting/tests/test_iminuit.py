# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...modeling import ParameterList, Parameter
from ...testing import requires_dependency
from .. import fit_iminuit


def fcn(parameters):
    x = parameters['model.x'].value
    y = parameters['model.y'].value
    z = parameters['model.z'].value
    return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2


@requires_dependency('iminuit')
def test_iminuit():
    pars = ParameterList(
        [Parameter('model', 'x', 2.1),
         Parameter('model', 'y', 3.1),
         Parameter('model', 'z', 4.1)]
    )

    minuit = fit_iminuit(function=fcn, parameters=pars)

    assert_allclose(pars['model.x'].value, 2, rtol=1e-2)
    assert_allclose(pars['model.y'].value, 3, rtol=1e-2)
    assert_allclose(pars['model.z'].value, 4, rtol=1e-2)

    assert_allclose(minuit.values['model.x'], 2, rtol=1e-2)
    assert_allclose(minuit.values['model.y'], 3, rtol=1e-2)
    assert_allclose(minuit.values['model.z'], 4, rtol=1e-2)

    # Test freeze
    pars['model.x'].frozen = True
    minuit = fit_iminuit(function=fcn, parameters=pars)
    assert minuit.list_of_fixed_param() == ['model.x']

    # Test limits
    pars['model.y'].min = 4
    minuit = fit_iminuit(function=fcn, parameters=pars)
    states = minuit.get_param_states()
    assert not states[0]['has_limits']
    assert not states[2]['has_limits']

    assert states[1]['has_limits']
    assert states[1]['lower_limit'] == 4
    assert states[1]['upper_limit'] == 0

    # Test stepsize via covariance matrix
    pars.set_parameter_errors({'model.x': '0.2', 'model.y': '0.1'})
    minuit = fit_iminuit(function=fcn, parameters=pars)

    assert minuit.migrad_ok()

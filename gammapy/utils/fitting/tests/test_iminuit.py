# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...modeling import ParameterList, Parameter
from ...testing import requires_dependency
from .. import fit_iminuit


@requires_dependency('iminuit')
def test_iminuit():
    def f(parameters):
        x = parameters['x'].value
        y = parameters['y'].value
        z = parameters['z'].value
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    pars_in = ParameterList(
        [Parameter('x', 2.1), Parameter('y', 3.1), Parameter('z', 4.1)]
    )

    pars_out, minuit = fit_iminuit(function=f, parameters=pars_in)

    assert_allclose(pars_in['x'].value, 2.1, rtol=1e-2)
    assert_allclose(pars_in['y'].value, 3.1, rtol=1e-2)
    assert_allclose(pars_in['z'].value, 4.1, rtol=1e-2)

    assert_allclose(pars_out['x'].value, 2, rtol=1e-2)
    assert_allclose(pars_out['y'].value, 3, rtol=1e-2)
    assert_allclose(pars_out['z'].value, 4, rtol=1e-2)

    assert_allclose(minuit.values['x'], 2, rtol=1e-2)
    assert_allclose(minuit.values['y'], 3, rtol=1e-2)
    assert_allclose(minuit.values['z'], 4, rtol=1e-2)

    # Test freeze
    pars_in['x'].frozen = True
    pars_out, minuit = fit_iminuit(function=f, parameters=pars_in)
    assert minuit.list_of_fixed_param() == ['x']

    # Test limits
    pars_in['y'].min = 4
    pars_out, minuit = fit_iminuit(function=f, parameters=pars_in)
    states = minuit.get_param_states()
    assert not states[0]['has_limits']
    assert not states[2]['has_limits']

    assert states[1]['has_limits']
    assert states[1]['lower_limit'] == 4
    assert states[1]['upper_limit'] == 0

    # Test stepsize via covariance matrix
    pars_in.set_parameter_errors({'x': '0.2', 'y': '0.1'})
    pars_out, minuit = fit_iminuit(function=f, parameters=pars_in)

    assert minuit.migrad_ok()

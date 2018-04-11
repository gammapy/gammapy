# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...modeling import ParameterList, Parameter
from ...testing import requires_dependency
from .. import fit_minuit


@requires_dependency('iminuit')  
def test_iminuit():
    def f(parameters):
        x = parameters['x'].value
        y = parameters['y'].value
        z = parameters['z'].value
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    p_in = ParameterList(
        [Parameter('x', 2.1), Parameter('y', 3.1), Parameter('z', 4.1)]
    )

    p_out = fit_minuit(function=f, parameters=p_in)

    assert_allclose(p_out['x'].value, 2, rtol=1e-2)
    assert_allclose(p_out['y'].value, 3, rtol=1e-2)
    assert_allclose(p_out['z'].value, 4, rtol=1e-2)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...modeling import ParameterList, Parameter
from .. import fit_minuit


def test_iminuit():
    def f(parameters):
        x = parameters['x'].value
        y = parameters['y'].value
        z = parameters['z'].value
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    p = ParameterList(
        [Parameter('x', 2.1), Parameter('y', 3.1), Parameter('z', 4.1)]
    )

    bf_pars = fit_minuit(function=f, parameters=p)

    assert_allclose(bf_pars['x'].value, 2, rtol=1e-2)
    assert_allclose(bf_pars['y'].value, 3, rtol=1e-2)
    assert_allclose(bf_pars['z'].value, 4, rtol=1e-2)

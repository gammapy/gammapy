# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from ...testing import requires_dependency
from ...modeling import Parameter, ParameterList
from ..sherpa import fit_sherpa


def fcn(parameters):
    x = parameters['model.x'].value
    y = parameters['model.y'].value
    z = parameters['model.z'].value
    return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2, 0


# TODO: levmar doesn't work yet; needs array of statval as return in likelihood
# optimiser='gridsearch' would require very low tolerance asserts, not added for now

@requires_dependency('sherpa')
@pytest.mark.parametrize('optimizer', ['moncar', 'simplex'])
def test_sherpa(optimizer):
    pars = ParameterList(
        [Parameter('model', 'x', 2.2), Parameter('model', 'y', 3.4),
         Parameter('model', 'z', 4.5)]
    )

    result = fit_sherpa(function=fcn, parameters=pars, optimizer=optimizer)

    assert result['success']
    assert result['nfev'] > 10
    assert_allclose(result['statval'], 0, atol=1e-2)
    assert_allclose(pars['model.x'].value, 2, rtol=1e-2)
    assert_allclose(pars['model.y'].value, 3, rtol=1e-2)
    assert_allclose(pars['model.z'].value, 4, rtol=1e-2)

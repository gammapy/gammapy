# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ..flux_points import _x_lafferty
import numpy as np


@pytest.mark.xfail
def test_lafferty_FluxPoints():
    """ Tests FluxPoints class using input function g(x) = 10^4 exp(-6x) against 
    check values from paper Lafferty & Wyatt. Nucl. Instr. and Meth. in Phys. 
    Res. A 355 (1995) 541-547, p. 542 Table 1
    """
    # These are the results from the paper
    checks = np.array([0.048, 0.190, 0.428, 0.762])

    f = lambda x: (10 ** 4) * np.exp(-6 * x)
    emins = np.array([0.0, 0.1, 0.3, 0.6])
    emaxs = np.array([0.1, 0.3, 0.6, 1.0])
    indices = np.arange(len(emins))
    for index in indices:
        value = _x_lafferty(xmin=emins[index], xmax=emaxs[index], function=f)
        assert_allclose(np.round(value, 3), checks[index], 1e-2)


@pytest.mark.xfail
def test_compute_integral_flux_points():
    pass


@pytest.mark.xfail
def test_compute_differential_flux_points():
    pass

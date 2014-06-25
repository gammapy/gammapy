# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from astropy.io import ascii, fits
from ..flux_point import _x_lafferty, _integrate, _ydiff_excess_equals_expected, _x_log_center, _integrate, _calc_x, compute_differential_flux_points
from ..crab import crab_flux, crab_integral_flux
import numpy as np


@pytest.mark.xfail
def test_x_lafferty():
    """ Tests flux_point class.

    Using input function g(x) = 10^4 exp(-6x) against 
    check values from paper Lafferty & Wyatt. Nucl. Instr. and Meth. in Phys. 
    Res. A 355 (1995) 541-547, p. 542 Table 1
    """
    # These are the results from the paper
    checks = np.array([0.048, 0.190, 0.428, 0.762])

    f = lambda x: (10 ** 4) * np.exp(-6 * x)
    emins = np.array([0.0, 0.1, 0.3, 0.6])
    emaxs = np.array([0.1, 0.3, 0.6, 1.0])
    indices = np.arange(len(emins))
    value = _x_lafferty(xmin=emins, xmax=emaxs, function=f)
    assert_allclose(np.round(value, 3), checks, 1e-2)


@pytest.mark.xfail
def test_x_log_center():
    """Tests method to determine log center x point values.
    """
    xmin = np.array([10, 30, 100, 300])
    xmax = np.array([30, 100, 300, 1000])
    indices = np.arange(len(xmin))
    correct_log_centers = []
    for index in indices:
        a = np.log(xmin[index])
        b = np.log(xmax[index])
        c = 0.5 * (a + b)
        center = np.exp(c)
        correct_log_centers.append(center)
    
    calc_log_centers = _x_log_center(xmin, xmax)
    assert_allclose(correct_log_centers, calc_log_centers, 1e-6)


@pytest.mark.xfail
def test_integration():
    """Tests integration scheme.

    Compares to numerical result with simple analytical test case y=x^2
    """
    function = lambda x: x ** 2
    xmin = -2
    xmax = 2
    
    indef_int = lambda x: (x ** 3) / 3
    # Calculate analytical result
    analyitical_int = indef_int(xmax) - indef_int(xmin)
    # Get numerical result
    numerical_int = _integrate(xmin, xmax, function, segments=1e3)
    # Compare, bounds suitable for number of segments
    assert_allclose(numerical_int, analyitical_int, 1e-2)


@pytest.mark.xfail
def test_ydiff_excess_equals_expected():
    """Tests y-value normalization adjustment method.
    """
    model = lambda x: x ** 2
    xmin = np.array([10, 20, 30, 40])
    xmax = np.array([20, 30, 40, 50])
    yint = np.array([42, 52, 62, 72])  # 'True' integral flux in this test bin
    # Get values
    y_values = _ydiff_excess_equals_expected(yint, xmin, xmax, 'Lafferty', model)
    # Set up test case comparison
    x_points = np.array(_calc_x(yint, xmin, xmax, 'Lafferty', model))
    y_model = model(x_points)
    # Test comparison result
    yint_model_check = []
    indices = np.arange(len(xmin))
    for index in indices:
        y_val = _integrate(xmin[index], xmax[index], model)
        yint_model_check.append(y_val)
    # Test output result
    yint_model = y_model * (yint / y_values)
    # Compare
    assert_allclose(yint_model, yint_model_check, 1e-6)


@pytest.mark.xfail
def test_compute_differential_flux_points():
    """Tests compute_differential_flux_points method.
    """
    emins = [0.01, 0.03, 0.1, 0.3]
    emaxs = [0.03, 0.1, 0.3, 1]
    table = Table()
    table['ENERGY_MIN'] = emins
    table['ENERGY_MAX'] = emaxs
    table['INT_FLUX'] = crab_integral_flux(energy_min=emins, energy_max=emaxs)
    # TODO: Extend this test to also consider errors
    power_law = compute_differential_flux_points(table=table, spec_index=2.63, x_method='Lafferty',
                                                 y_method='PowerLaw', function=None)
    diff_fluxes = power_law['DIFF_FLUX']
    energies = power_law['ENERGY']
    check_diff_fluxes = crab_flux(energies, reference='hess_pl')
    assert_allclose(diff_fluxes, check_diff_fluxes, 1e-6)

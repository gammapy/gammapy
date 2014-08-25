# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import itertools
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from ..flux_point import (_x_lafferty, _integrate, _ydiff_excess_equals_expected,
                          compute_differential_flux_points,
                          _energy_lafferty_power_law)
from ...spectrum.powerlaw import power_law_eval, power_law_integral_flux

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

x_methods = ['table', 'lafferty', 'log_center']
y_methods = ['power_law', 'model']
indices = [0, 1, 2, 3]


@pytest.mark.skipif('not HAS_SCIPY')
def test_x_lafferty():
    """Tests Lafferty & Wyatt x-point method.

    Using input function g(x) = 10^4 exp(-6x) against 
    check values from paper Lafferty & Wyatt. Nucl. Instr. and Meth. in Phys.
    Res. A 355 (1995) 541-547, p. 542 Table 1
    """
    # These are the results from the paper
    desired = np.array([0.048, 0.190, 0.428, 0.762])

    def f(x):
        return (10 ** 4) * np.exp(-6 * x)

    emins = np.array([0.0, 0.1, 0.3, 0.6])
    emaxs = np.array([0.1, 0.3, 0.6, 1.0])
    actual = _x_lafferty(xmin=emins, xmax=emaxs, function=f)
    assert_allclose(actual, desired, atol=1e-3)


def test_integration():
    def function(x):
        return x ** 2

    xmin = np.array([-2])
    xmax = np.array([2])

    def indef_int(x):
        return (x ** 3) / 3

    # Calculate analytical result
    desired = indef_int(xmax) - indef_int(xmin)
    # Get numerical result
    actual = _integrate(xmin, xmax, function, segments=1e3)
    # Compare, bounds suitable for number of segments
    assert_allclose(actual, desired, rtol=1e-2)


@pytest.mark.skipif('not HAS_SCIPY')
def test_ydiff_excess_equals_expected():
    """Tests y-value normalization adjustment method.
    """
    def model(x):
        return x ** 2

    xmin = np.array([10, 20, 30, 40])
    xmax = np.array([20, 30, 40, 50])
    yint = np.array([42, 52, 62, 72])  # 'True' integral flux in this test bin
    # Get values
    x_values = np.array(_x_lafferty(xmin, xmax, model))
    y_values = _ydiff_excess_equals_expected(yint, xmin, xmax, x_values, model)
    # Set up test case comparison
    y_model = model(np.array(x_values))
    # Test comparison result
    desired = _integrate(xmin, xmax, model)
    # Test output result
    actual = y_model * (yint / y_values)
    # Compare
    assert_allclose(actual, desired, rtol=1e-6)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize('index, x_method, y_method',
                         itertools.product(indices, ['lafferty', 'log_center'],
                                           y_methods))
def test_array_broadcasting(index, x_method, y_method):
    """Tests for array broadcasting in for likely input scenarios.
    """
    # API for power_law case can differ from model case if table not used
    # so both tested here
    in_array = 0.9 * np.arange(6).reshape(3, 2)
    values = dict(SPECTRAL_INDEX=[3 * in_array, 3., 3., 3.],
                  ENERGY_MIN=[1., 0.1 * in_array, 1., 1.],
                  ENERGY_MAX=[10., 10., 4 * in_array, 10.],
                  INT_FLUX=[30., 30., 30., 10. * in_array])
    # Define parameters
    spectral_index = values['SPECTRAL_INDEX'][index]
    energy_min = values['ENERGY_MIN'][index]
    energy_max = values['ENERGY_MAX'][index]
    int_flux = values['INT_FLUX'][index]
    int_flux_err = 0.1 * int_flux
    if y_method == 'power_law':
        model = None
    else:
        def model(x):
            return x ** 2

    table = compute_differential_flux_points(x_method, y_method, model=model,
                                             spectral_index=spectral_index,
                                             energy_min=energy_min,
                                             energy_max=energy_max,
                                             int_flux=int_flux,
                                             int_flux_err=int_flux_err)
    # Check output sized
    energy = table['ENERGY']
    actual = len(energy)
    desired = 6
    assert_allclose(actual, desired)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize('x_method,y_method', itertools.product(x_methods,
                                                                y_methods))
def test_compute_differential_flux_points(x_method, y_method):
    """Iterates through the 6 different combinations of input options.

    Tests against analytical result or result from gammapy.spectrum.powerlaw.
    """
    # Define the test cases for all possible options
    energy_min = np.array([1.0, 10.0])
    energy_max = np.array([10.0, 100.0])
    spectral_index = 2.0
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['ENERGY'] = np.array([2.0, 20.0])
    if x_method == 'log_center':
        energy = np.sqrt(energy_min * energy_max)
    elif x_method == 'table':
        energy = table['ENERGY'].data

    # Arbitrary model (simple exponential case)
    def diff_flux_model(x):
        return np.exp(x)

    # Integral of model
    def int_flux_model(E_min, E_max):
        return np.exp(E_max) - np.exp(E_min)

    if y_method == 'power_law':
        if x_method == 'lafferty':
            energy = _energy_lafferty_power_law(energy_min, energy_max,
                                                spectral_index)
            # Test that this is equal to analytically expected
            # log center result
            desired_energy = np.sqrt(energy_min * energy_max)
            assert_allclose(energy, desired_energy, rtol=1e-6)
        desired = power_law_eval(energy, 1, spectral_index, energy)
        int_flux = power_law_integral_flux(desired, spectral_index, energy,
                                           energy_min, energy_max)
    elif y_method == 'model':
        if x_method == 'lafferty':
            energy = _x_lafferty(energy_min, energy_max, diff_flux_model)
        desired = diff_flux_model(energy)
        int_flux = int_flux_model(energy_min, energy_max)
    int_flux_err = 0.1 * int_flux
    table['INT_FLUX'] = int_flux
    table['INT_FLUX_ERR'] = int_flux_err

    result_table = compute_differential_flux_points(x_method,
                                                    y_method,
                                                    table,
                                                    diff_flux_model,
                                                    spectral_index)
    # Test energy
    actual_energy = result_table['ENERGY'].data
    desired_energy = energy
    assert_allclose(actual_energy, desired_energy, rtol=1e-3)
    # Test flux
    actual = result_table['DIFF_FLUX'].data
    assert_allclose(actual, desired, rtol=1e-3)
    # Test error
    actual = result_table['DIFF_FLUX_ERR'].data
    desired = 0.1 * result_table['DIFF_FLUX'].data
    assert_allclose(actual, desired, rtol=1e-3)

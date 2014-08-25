# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Differential and integral flux point computations."""
from __future__ import print_function, division
import numpy as np
from astropy.table import Table
from ..spectrum.powerlaw import power_law_flux

__all__ = ['compute_differential_flux_points']


def compute_differential_flux_points(x_method='lafferty', y_method='power_law',
                                     table=None, model=None,
                                     spectral_index=None, energy_min=None,
                                     energy_max=None, int_flux=None,
                                     int_flux_err=None):
    """Creates differential flux points table from integral flux points table.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Integral flux data table in energy bins, including columns
        'ENERGY_MIN', 'ENERGY_MAX', 'INT_FLUX', 'INT_FLUX_ERR'
    energy_min : float, array_like
        If table not defined, minimum energy of bin(s) may be input
        directly as either a float or array.
    energy_max : float, array_like
        If table not defined, maximum energy of bin(s) input directly.
    int_flux : float, array_like
        If table not defined, integral flux in bin(s) input directly. If array,
        energy_min, energy_max must be either arrays of the same shape
        (for differing energy bins) or floats (for the same energy bin).
    int_flux_err : float, array_like
        Type must be the same as for int_flux
    x_method : {'lafferty', 'log_center', 'table'}
        Flux point energy computation method; either Lafferty & Wyatt
        model-based positioning, log bin center positioning
        or user-defined `~astropy.table.Table` positioning
        using column heading ['ENERGY']
    y_method : {'power_law', 'model'}
        Flux computation method assuming PowerLaw or user defined model function
    model : callable
        User-defined model function
    spectral_index : float, array_like
        Spectral index if default power law model is used. Either a float
        or array_like (in which case, energy_min, energy_max and int_flux
        must be floats to avoid ambiguity)

    Returns
    -------
    differential_flux_table : `~astropy.table.Table`
        Input table with appended columns 'ENERGY', 'DIFF_FLUX', 'DIFF_FLUX_ERR'

    Notes
    -----
    For usage, see also tutorial:
    https://gammapy.readthedocs.org/en/latest/tutorials/flux_point.html
    """
    # Use input values if not initially provided with a table
    # and broadcast quantities to arrays if required
    if table is None:
        spectral_index = np.array(spectral_index).reshape(np.array(spectral_index).size,)
        energy_min = np.array(energy_min).reshape(np.array(energy_min).size,)
        energy_max = np.array(energy_max).reshape(np.array(energy_max).size,)
        int_flux = np.array(int_flux).reshape(np.array(int_flux).size,)
        try:
            int_flux_err = np.array(int_flux_err).reshape(np.array(int_flux_err).size,)
        except:
            pass
        # TODO: Can a better implementation be found here?
        lengths = dict(SPECTRAL_INDEX=len(spectral_index),
                       ENERGY_MIN=len(energy_min),
                       ENERGY_MAX=len(energy_max),
                       FLUX=len(int_flux))
        max_length = np.array(list(lengths.values())).max()
        int_flux = np.array(int_flux) * np.ones(max_length)
        spectral_index = np.array(spectral_index) * np.ones(max_length)
        energy_min = np.array(energy_min) * np.ones(max_length)
        energy_max = np.array(energy_max) * np.ones(max_length)
        try:
            int_flux_err = np.array(int_flux_err) * np.ones(max_length)
        except:
            pass
    # Otherwise use the table provided
    else:
        energy_min = np.asanyarray(table['ENERGY_MIN'])
        energy_max = np.asanyarray(table['ENERGY_MAX'])
        int_flux = np.asanyarray(table['INT_FLUX'])
        try:
            int_flux_err = np.asanyarray(table['INT_FLUX_ERR'])
        except:
            pass
    # Compute x point
    if x_method == 'table':
        # This is only called if the provided table includes energies
        energy = np.array(table['ENERGY'])
    elif x_method == 'log_center':
        from scipy.stats import gmean
        energy = np.array(gmean((energy_min, energy_max)))
    elif x_method == 'lafferty':
        if y_method == 'power_law':
            # Uses analytical implementation available for the power law case
            energy = _energy_lafferty_power_law(energy_min, energy_max,
                                                spectral_index)
        else:
            energy = np.array(_x_lafferty(energy_min,
                                          energy_max, model))
    else:
        raise ValueError('Invalid x_method: {0}'.format(x_method))
    # Compute y point
    if y_method == 'power_law':
        g = -1 * np.abs(spectral_index)
        diff_flux = power_law_flux(int_flux, g, energy, energy_min, energy_max)
    elif y_method == 'model':
        diff_flux = _ydiff_excess_equals_expected(int_flux, energy_min,
                                                  energy_max, energy, model)
    else:
        raise ValueError('Invalid y_method: {0}'.format(y_method))
    # Add to table
    table = Table()
    table['ENERGY'] = energy
    table['DIFF_FLUX'] = diff_flux

    # Error processing if required
    try:
        # TODO: more rigorous implementation of error propagation should be implemented
        # I.e. based on MC simulation rather than gaussian error assumption
        err = int_flux_err / int_flux
        diff_flux_err = err * diff_flux
        table['DIFF_FLUX_ERR'] = diff_flux_err
    except:
        pass

    table.meta['spectral_index'] = spectral_index
    table.meta['spectral_index_description'] = "Spectral index assumed in the DIFF_FLUX computation"
    return table


def _x_lafferty(xmin, xmax, function):
    """The Lafferty & Wyatt method to compute X.

    Pass in a function and bin bounds x_min and x_max i.e. for energy
    See: Lafferty & Wyatt, Nucl. Instr. and Meth. in Phys. Res. A 355(1995) 541-547
    See: http://nbviewer.ipython.org/gist/cdeil/bdab5f236640ef52f736
    """
    from scipy.optimize import brentq
    from scipy import integrate

    indices = np.arange(len(xmin))

    x_points = []
    for index in indices:
        deltax = xmax[index] - xmin[index]
        I = integrate.quad(function, xmin[index], xmax[index], args=())
        F = (I[0] / deltax)

        def g(x):
            return function(x) - F

        x_point = brentq(g, xmin[index], xmax[index])
        x_points.append(x_point)
    return x_points


def _ydiff_excess_equals_expected(yint, xmin, xmax, x, model):
    """The ExcessEqualsExpected method to compute Y (differential).

    y / yint = y_model / yint_model"""
    yint_model = _integrate(xmin, xmax, model)
    y_model = model(x)
    return y_model * (yint / yint_model)


def _integrate(xmin, xmax, function, segments=1e6):
    """Integrates method function using the trapezium rule between xmin and xmax.
    """
    indices = np.arange(len(xmin))
    y_values = []
    for index in indices:
        x_vals = np.arange(xmin[index], xmax[index], 1.0 / segments)
        y_vals = function(x_vals)
        # Division by number of segments required for correct normalization
        y_values.append(np.trapz(y_vals) / segments)
    return y_values


def _energy_lafferty_power_law(energy_min, energy_max, spectral_index):
    """Analytical case for determining lafferty x-position for power law case.
    """
    # Cannot call into gammapy.powerlaw as implementation is different
    # due to different reference energies
    term0 = 1. - spectral_index
    term1 = energy_max - energy_min
    term2 = 1. / term0
    flux_lw = term2 / term1 * (energy_max ** term0 - energy_min ** term0)
    return np.exp(-np.log(flux_lw) / np.abs(spectral_index))

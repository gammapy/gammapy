# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Differential and integral flux point computations."""
from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
from astropy.utils.compat.odict import OrderedDict
from .powerlaw import power_law_flux

__all__ = ['compute_differential_flux_points']


def compute_differential_flux_points(table, spec_index=None, x_method='Lafferty',
                                 y_method='PowerLaw', function=None):
    """Creates differential flux points table from integral flux points table.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Integral flux data table in energy bins, including columns
        'ENERGY_MIN', 'ENERGY_MAX', 'INT_FLUX', 'INT_FLUX_ERR'
    spec_index : float
        Spectral index if default power law model is used
    x_method : {'Lafferty', 'LogCenter', 'Table'}
        Flux point energy computation method; either Lafferty & Wyatt
        model-based positioning, log bin center positioning 
        or user-defined `~astropy.table.Table` positioning
        using column heading ['ENERGY']
    y_method : {'PowerLaw', 'Model'}
        Flux computation method assuming PowerLaw or user defined model function
    function : function
        User-defined model function

    Returns
    -------
    differential_flux_table : `~astropy.table.Table`
        Input table with appended columns 'ENERGY', 'DIFF_FLUX', 'DIFF_FLUX_ERR'
    """

    emins = np.asanyarray(table['ENERGY_MIN'])
    emaxs = np.asanyarray(table['ENERGY_MAX'])
    i_flux = np.asanyarray(table['INT_FLUX'])
    try:
        i_err = np.asanyarray(table['INT_FLUX_ERR'])
    except:
        pass
    if y_method == 'PowerLaw':
        g = np.abs(spec_index)
        # Assumes function is continuous over energy bin boundaries
        e1 = np.min(emins)
        e2 = np.max(emaxs)
        plaw = lambda x: (x / (1 - g)) * (((e2 / x) ** (1 - g)) - ((e1 / x) ** (1 - g)))
        energy = _calc_x(i_flux, emins, emaxs,
                    x_method, plaw)
        f_val = power_law_flux(I=i_flux, g=spec_index, e=energy,
                                e1=emins, e2=emaxs)
    elif y_method == 'Model':
        yint = i_flux
        f_val = _ydiff_excess_equals_expected(yint, emins, emaxs,
                                            x_method, function)
        energy = _calc_x(i_flux, emins, emaxs,
                        x_method, function)
    else:
        raise ValueError('Unknown method {0}'.format(y_method))

    energy_col = Column(name='ENERGY', data=energy)
    table.add_column(energy_col)
    dflux_col = Column(name='DIFF_FLUX', data=f_val)
    table.add_column(dflux_col)

    try:
        # TODO: more rigorous implementation of error propagation should be implemented
        # I.e. based on MC simulation rather than gaussian error assumption
        err = i_err / i_flux
        f_err = err * f_val
        dflux_err_col = Column(name='DIFF_FLUX_ERR', data=f_err)
        table.add_column(dflux_err_col)
    except:
        pass

    table.meta['spectral_index'] = spec_index

    return table


def _calc_xy(yint, xmin, xmax, x_method, y_method, function=None):
    """Compute differential flux points (x,y).

    """
    x = _calc_x(yint, xmin, xmax, x_method, function)
    y = _calc_y(x, yint, xmin, xmax, y_method)
    return x, y


def _calc_x(yint, xmin, xmax, x_method, function):
    """Compute x position of differential flux point.

    """
    xmin = np.asanyarray(xmin)
    xmax = np.asanyarray(xmax)
    return XYMETHODS[x_method](xmin, xmax, function)


def _calc_y(x, yint, xmin, xmax, x_method, y_method, model):
    """Compute y position of differential flux point.

    """
    x = np.asanyarray(x)
    yint = np.asanyarray(yint)
    xmin = np.asanyarray(xmin)
    xmax = np.asanyarray(xmax)
    return XYMETHODS[y_method](yint, xmin, xmax, x_method, model)


def _x_log_center(xmin, xmax, function=None, table=None):
    """The LogCenter method to compute X.

    """
    x = np.sqrt(xmin * xmax)
    return x


def _x_lafferty(xmin, xmax, function, table=None):
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
        g = lambda x: function(x) - F
        x_point = brentq(g, xmin[index], xmax[index])
        x_points.append(x_point)
    return x_points


def _x_read_table(xmin=None, xmax=None, function=None, table=None):
    """Energy positions within bins to be read from a user-defined astropy table.

    """
    x_points = table['ENERGY']
    return x_points


def _ydiff_excess_equals_expected(yint, xmin, xmax, x_method, model):
    """The ExcessEqualsExpected method to compute Y (differential).

    y / yint = y_model / yint_model"""

    indices = np.arange(len(xmin))

    x = _calc_x(yint, xmin, xmax, x_method, model)

    y_values = []
    for index in indices:
        yint_model = _integrate(xmin[index], xmax[index], model)
        y_model = model(x[index])
        y = y_model * (yint[index] / yint_model)
        y_values.append(y)
    return y_values


def _integrate(xmin, xmax, function, segments=1e3):
    """Integrates method function using the trapezium rule between xmin and xmax.

    """
    x_vals = np.arange(xmin, xmax, 1.0 / segments)
    y_vals = function(x_vals)
    return np.trapz(y_vals)


XYMETHODS = OrderedDict()
"""Dictionary of available X and Y point evaluation methods.

Useful for automatic processing.
"""
XYMETHODS['LogCenter'] = _x_log_center
XYMETHODS['Lafferty'] = _x_lafferty
XYMETHODS['Table'] = _x_read_table
XYMETHODS['ExcessEqualsExpected'] = _ydiff_excess_equals_expected

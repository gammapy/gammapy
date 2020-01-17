# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.units import Quantity
from .interpolation import LogScale

__all__ = ["integrate_spectrum"]


def integrate_spectrum(func, xmin, xmax, ndecade=100, intervals=False):
    """Integrate 1d function using the log-log trapezoidal rule.

    If scalar values for xmin and xmax are passed an oversampled grid is generated using the
    ``ndecade`` keyword argument. If xmin and xmax arrays are passed, no
    oversampling is performed and the integral is computed in the provided
    grid.

    Parameters
    ----------
    func : callable
        Function to integrate.
    xmin : `~astropy.units.Quantity` or array-like
        Integration range minimum
    xmax : `~astropy.units.Quantity` or array-like
        Integration range minimum
    ndecade : int, optional
        Number of grid points per decade used for the integration.
        Default : 100.
    intervals : bool, optional
        Return integrals in the grid not the sum, default: False
    """
    is_quantity = False
    if isinstance(xmin, Quantity):
        unit = xmin.unit
        xmin = xmin.value
        xmax = xmax.to_value(unit)
        is_quantity = True

    if np.isscalar(xmin):
        logmin = np.log10(xmin)
        logmax = np.log10(xmax)
        n = int((logmax - logmin) * ndecade)
        x = np.logspace(logmin, logmax, n)
    else:
        x = np.append(xmin, xmax[-1])

    if is_quantity:
        x = x * unit

    y = func(x)

    val = _trapz_loglog(y, x, intervals=intervals)

    return val


# This function is copied over from https://github.com/zblz/naima/blob/master/naima/utils.py#L261
# and slightly modified to allow use with the uncertainties package


def _trapz_loglog(y, x, axis=-1, intervals=False):
    """Integrate using the composite trapezoidal rule in log-log space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.
    intervals : bool, optional
        Return array of shape x not the total integral, default: False

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    from gammapy.modeling.models import PowerLawSpectralModel

    # see https://stackoverflow.com/a/56840428
    x, y = np.moveaxis(x, axis, 0), np.moveaxis(y, axis, 0)

    emin, emax = x[:-1], x[1:]
    vals_emin, vals_emax = y[:-1], y[1:]

    # log scale has the build-in zero clipping
    log = LogScale()
    index = -log(vals_emin / vals_emax) / log(emin / emax)
    index[np.isnan(index)] = np.inf

    integral = PowerLawSpectralModel.evaluate_integral(
        emin=emin,
        emax=emax,
        index=index,
        reference=emin,
        amplitude=vals_emin
    )

    if intervals:
        return integral

    return integral.sum(axis=axis)
